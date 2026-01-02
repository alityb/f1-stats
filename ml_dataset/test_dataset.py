"""
Test script for PyTorch dataset and dataloader.

Usage:
    python -m ml_dataset.test_dataset
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_dataset.stint_dataset import StintDataset, collate_stints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Test dataset loading and batching."""

    logger.info("=" * 80)
    logger.info("DATASET AND DATALOADER TEST")
    logger.info("=" * 80)

    # Load dataset
    parquet_path = Path(cfg.paths.stints_dir) / "stint_sequences.parquet"

    if not parquet_path.exists():
        logger.error(f"Dataset not found at {parquet_path}")
        logger.error("Run: python -m ml_dataset.create_sequences")
        return

    dataset = StintDataset(parquet_path, filter_min_valid_laps=3)

    logger.info(f"\n✅ Loaded dataset with {len(dataset)} stints")

    # Get vocabulary sizes
    vocab_sizes = dataset.get_vocab_sizes()
    logger.info("\nVocabulary sizes (for embedding layers):")
    for key, size in vocab_sizes.items():
        logger.info(f"  {key}: {size}")

    # Test single sample
    logger.info("\n" + "-" * 80)
    logger.info("SINGLE SAMPLE TEST")
    logger.info("-" * 80)

    sample = dataset[0]
    logger.info(f"\nSample keys: {list(sample.keys())}")
    logger.info(f"\nSample shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape} (dtype: {value.dtype})")
        else:
            logger.info(f"  {key}: {value} (type: {type(value).__name__})")

    logger.info(f"\nSample data:")
    logger.info(f"  Sequence length: {sample['sequence_length']}")
    logger.info(f"  Valid laps: {sample['valid_mask'].sum().item()}")
    logger.info(f"  Driver ID: {sample['driver_id'].item()}")
    logger.info(f"  Compound sequence: {sample['compound'][:10].tolist()}... (first 10)")
    logger.info(f"  Lap times: {sample['lap_time'][:10].tolist()}... (first 10)")
    logger.info(f"  Valid mask: {sample['valid_mask'][:10].tolist()}... (first 10)")

    # Test DataLoader with batching
    logger.info("\n" + "-" * 80)
    logger.info("DATALOADER BATCHING TEST")
    logger.info("-" * 80)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_stints,
        num_workers=0,  # 0 for testing, increase for training
    )

    batch = next(iter(dataloader))

    logger.info(f"\nBatch keys: {list(batch.keys())}")
    logger.info(f"\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape} (dtype: {value.dtype})")
        elif isinstance(value, list):
            logger.info(f"  {key}: list of length {len(value)}")

    logger.info(f"\nBatch details:")
    logger.info(f"  Batch size: {batch['driver_id'].shape[0]}")
    logger.info(f"  Max sequence length: {batch['lap_time'].shape[1]}")
    logger.info(f"  Sequence lengths: {batch['sequence_lengths'].tolist()}")
    logger.info(f"  Valid laps per sequence: {batch['valid_mask'].sum(dim=1).tolist()}")
    logger.info(f"  Padding mask shape: {batch['padding_mask'].shape}")

    # Verify masking
    logger.info("\n" + "-" * 80)
    logger.info("MASK VERIFICATION")
    logger.info("-" * 80)

    for i in range(batch['driver_id'].shape[0]):
        seq_len = batch['sequence_lengths'][i].item()
        valid_count = batch['valid_mask'][i].sum().item()
        padding_count = batch['padding_mask'][i].sum().item()

        logger.info(f"\nSequence {i}:")
        logger.info(f"  Length: {seq_len}")
        logger.info(f"  Valid laps: {valid_count}")
        logger.info(f"  Padding positions filled: {padding_count} (should equal {seq_len})")
        logger.info(f"  Padding correctly applied: {padding_count == seq_len}")

    # Test multiple batches
    logger.info("\n" + "-" * 80)
    logger.info("ITERATION TEST (5 batches)")
    logger.info("-" * 80)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:
            break

        logger.info(
            f"Batch {batch_idx}: "
            f"size={batch['driver_id'].shape[0]}, "
            f"max_len={batch['lap_time'].shape[1]}, "
            f"total_valid_laps={batch['valid_mask'].sum().item()}"
        )

    logger.info("\n" + "=" * 80)
    logger.info("✅ All dataset tests passed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
