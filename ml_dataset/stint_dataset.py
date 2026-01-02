"""
PyTorch Dataset for stint-level lap sequences.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class StintDataset(Dataset):
    """PyTorch Dataset for stint sequences."""

    def __init__(
        self,
        parquet_path: Path,
        filter_min_valid_laps: int = 1,
    ):
        """
        Initialize dataset.

        Args:
            parquet_path: Path to stint_sequences.parquet file
            filter_min_valid_laps: Minimum number of valid laps required per stint
        """
        self.parquet_path = parquet_path
        self.filter_min_valid_laps = filter_min_valid_laps

        # Load sequences
        logger.info(f"Loading sequences from {parquet_path}")
        self.df = pd.read_parquet(parquet_path)

        # Filter by minimum valid laps
        initial_count = len(self.df)
        self.df = self.df[self.df["num_valid_laps"] >= filter_min_valid_laps]
        filtered_count = initial_count - len(self.df)

        if filtered_count > 0:
            logger.info(
                f"Filtered out {filtered_count} stints with < {filter_min_valid_laps} valid laps"
            )

        logger.info(f"Loaded {len(self.df)} stint sequences")

        # Build categorical vocabularies
        self._build_vocabularies()

    def _build_vocabularies(self):
        """Build vocabularies for categorical features."""
        self.vocab = {}

        # Driver vocabulary
        all_drivers = set(self.df["driver_id"].unique())
        self.vocab["driver_id"] = {driver: idx for idx, driver in enumerate(sorted(all_drivers))}
        self.vocab["driver_id"]["<UNK>"] = len(self.vocab["driver_id"])

        # Car vocabulary (team_season)
        all_cars = set(self.df["car_id"].unique())
        self.vocab["car_id"] = {car: idx for idx, car in enumerate(sorted(all_cars))}
        self.vocab["car_id"]["<UNK>"] = len(self.vocab["car_id"])

        # Track vocabulary
        all_tracks = set(self.df["track_id"].unique())
        self.vocab["track_id"] = {track: idx for idx, track in enumerate(sorted(all_tracks))}
        self.vocab["track_id"]["<UNK>"] = len(self.vocab["track_id"])

        # Season vocabulary
        all_seasons = set(self.df["season_id"].unique())
        self.vocab["season_id"] = {int(season): idx for idx, season in enumerate(sorted(all_seasons))}

        # Compound vocabulary (from sequences - can vary per lap)
        all_compounds = set()
        for compounds_seq in self.df["compound_seq"]:
            all_compounds.update(compounds_seq)
        self.vocab["compound"] = {comp: idx for idx, comp in enumerate(sorted(all_compounds))}
        self.vocab["compound"]["<UNK>"] = len(self.vocab["compound"])

        logger.info(f"Vocabulary sizes:")
        logger.info(f"  Drivers: {len(self.vocab['driver_id'])}")
        logger.info(f"  Cars: {len(self.vocab['car_id'])}")
        logger.info(f"  Tracks: {len(self.vocab['track_id'])}")
        logger.info(f"  Seasons: {len(self.vocab['season_id'])}")
        logger.info(f"  Compounds: {len(self.vocab['compound'])}")

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single stint sequence.

        Returns:
            Dictionary containing:
                - categorical features (stint-level and sequence-level)
                - continuous features (sequences)
                - targets (sequence)
                - mask (sequence)
                - metadata
        """
        row = self.df.iloc[idx]

        # Sequence length
        seq_len = row["sequence_length"]

        # Categorical features (stint-level, broadcast to sequence)
        driver_idx = self.vocab["driver_id"].get(row["driver_id"], self.vocab["driver_id"]["<UNK>"])
        car_idx = self.vocab["car_id"].get(row["car_id"], self.vocab["car_id"]["<UNK>"])
        track_idx = self.vocab["track_id"].get(row["track_id"], self.vocab["track_id"]["<UNK>"])
        season_idx = self.vocab["season_id"].get(row["season_id"], 0)

        # Compound sequence (per-lap categorical)
        compound_seq = [
            self.vocab["compound"].get(comp, self.vocab["compound"]["<UNK>"])
            for comp in row["compound_seq"]
        ]

        # Continuous feature sequences
        lap_in_stint_seq = np.array(row["lap_in_stint_seq"], dtype=np.float32)
        tyre_age_seq = np.array(row["tyre_age_seq"], dtype=np.float32)
        fuel_proxy_seq = np.array(row["fuel_proxy_seq"], dtype=np.float32)
        stint_length_seq = np.array(row["stint_length_seq"], dtype=np.float32)

        # Target sequence
        lap_time_seq = np.array(row["lap_time_seconds_seq"], dtype=np.float32)

        # Mask sequence
        mask_seq = np.array(row["valid_mask_seq"], dtype=np.bool_)

        return {
            # Metadata
            "race_id": row["race_id"],
            "stint_index": row["stint_index"],
            "sequence_length": seq_len,

            # Stint-level categorical (scalars to be broadcast/embedded)
            "driver_id": torch.tensor(driver_idx, dtype=torch.long),
            "car_id": torch.tensor(car_idx, dtype=torch.long),
            "track_id": torch.tensor(track_idx, dtype=torch.long),
            "season_id": torch.tensor(season_idx, dtype=torch.long),

            # Sequence-level categorical [T]
            "compound": torch.tensor(compound_seq, dtype=torch.long),

            # Continuous features [T]
            "lap_in_stint": torch.tensor(lap_in_stint_seq, dtype=torch.float32),
            "tyre_age": torch.tensor(tyre_age_seq, dtype=torch.float32),
            "fuel_proxy": torch.tensor(fuel_proxy_seq, dtype=torch.float32),
            "stint_length": torch.tensor(stint_length_seq, dtype=torch.float32),

            # Target [T]
            "lap_time": torch.tensor(lap_time_seq, dtype=torch.float32),

            # Mask [T]
            "valid_mask": torch.tensor(mask_seq, dtype=torch.bool),
        }

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for embedding layers."""
        return {
            "driver": len(self.vocab["driver_id"]),
            "car": len(self.vocab["car_id"]),
            "track": len(self.vocab["track_id"]),
            "season": len(self.vocab["season_id"]),
            "compound": len(self.vocab["compound"]),
        }


def collate_stints(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader with padding.

    Pads sequences to max length in batch.

    Args:
        batch: List of samples from StintDataset

    Returns:
        Batched and padded tensors
    """
    # Find max sequence length in batch
    max_len = max(item["sequence_length"] for item in batch)
    batch_size = len(batch)

    # Initialize padded tensors
    # Stint-level categorical (batch_size,)
    driver_ids = torch.stack([item["driver_id"] for item in batch])
    car_ids = torch.stack([item["car_id"] for item in batch])
    track_ids = torch.stack([item["track_id"] for item in batch])
    season_ids = torch.stack([item["season_id"] for item in batch])

    # Sequence tensors (batch_size, max_len)
    compound = torch.zeros(batch_size, max_len, dtype=torch.long)
    lap_in_stint = torch.zeros(batch_size, max_len, dtype=torch.float32)
    tyre_age = torch.zeros(batch_size, max_len, dtype=torch.float32)
    fuel_proxy = torch.zeros(batch_size, max_len, dtype=torch.float32)
    stint_length = torch.zeros(batch_size, max_len, dtype=torch.float32)

    lap_time = torch.zeros(batch_size, max_len, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Padding mask (True for valid positions, False for padding)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Fill in actual values
    for i, item in enumerate(batch):
        seq_len = item["sequence_length"]

        compound[i, :seq_len] = item["compound"]
        lap_in_stint[i, :seq_len] = item["lap_in_stint"]
        tyre_age[i, :seq_len] = item["tyre_age"]
        fuel_proxy[i, :seq_len] = item["fuel_proxy"]
        stint_length[i, :seq_len] = item["stint_length"]

        lap_time[i, :seq_len] = item["lap_time"]
        valid_mask[i, :seq_len] = item["valid_mask"]
        padding_mask[i, :seq_len] = True

    return {
        # Metadata
        "race_id": [item["race_id"] for item in batch],
        "stint_index": [item["stint_index"] for item in batch],
        "sequence_lengths": torch.tensor([item["sequence_length"] for item in batch], dtype=torch.long),

        # Stint-level categorical [B]
        "driver_id": driver_ids,
        "car_id": car_ids,
        "track_id": track_ids,
        "season_id": season_ids,

        # Sequence-level features [B, T]
        "compound": compound,
        "lap_in_stint": lap_in_stint,
        "tyre_age": tyre_age,
        "fuel_proxy": fuel_proxy,
        "stint_length": stint_length,

        # Target [B, T]
        "lap_time": lap_time,

        # Masks [B, T]
        "valid_mask": valid_mask,
        "padding_mask": padding_mask,
    }
