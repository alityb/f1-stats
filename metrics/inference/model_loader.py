"""
Model loading service for F1 Stats API.

Implements singleton pattern to load the trained pace decomposition model once at startup.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig

from ml_dataset.stint_dataset import StintDataset
from models.pace_model import PaceDecompositionModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton service for loading and caching the trained model."""

    _instance: Optional["ModelLoader"] = None
    _model: Optional[PaceDecompositionModel] = None
    _vocab_sizes: Optional[dict] = None
    _vocab_mappings: Optional[dict] = None
    _checkpoint_info: Optional[dict] = None

    def __init__(self, config: DictConfig):
        """Initialize model loader (use get_instance() instead)."""
        if ModelLoader._instance is not None:
            raise RuntimeError("ModelLoader is a singleton. Use get_instance().")

        self.config = config
        self.checkpoint_path = Path(config.api.checkpoint_path)

    @classmethod
    def get_instance(cls, config: Optional[DictConfig] = None) -> "ModelLoader":
        """
        Get singleton instance of ModelLoader.

        Args:
            config: Hydra config (required on first call)

        Returns:
            ModelLoader instance
        """
        if cls._instance is None:
            if config is None:
                raise ValueError("Config required for first ModelLoader initialization")
            cls._instance = cls(config)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """
        Load model from checkpoint.

        Based on pattern from training/test_trained_model.py:50-81
        """
        logger.info(f"Loading model from {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )

        # Load dataset to get vocab sizes
        parquet_path = Path(self.config.paths.stints_dir) / "stint_sequences.parquet"
        dataset = StintDataset(parquet_path, filter_min_valid_laps=1)
        self._vocab_sizes = dataset.get_vocab_sizes()
        self._vocab_mappings = dataset.vocab  # Access vocab dict directly

        # Initialize model
        self._model = PaceDecompositionModel(self.config, self._vocab_sizes)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        # Store checkpoint info
        self._checkpoint_info = {
            "epoch": checkpoint.get("epoch"),
            "best_val_metric": checkpoint.get("best_val_metric"),
            "path": str(self.checkpoint_path),
        }

        logger.info(
            f"Model loaded successfully from epoch {self._checkpoint_info['epoch']}"
        )
        logger.info(
            f"Best validation RMSE: {self._checkpoint_info['best_val_metric']:.4f}s"
        )
        logger.info(f"Vocabulary sizes: {self._vocab_sizes}")

    @property
    def model(self) -> PaceDecompositionModel:
        """Get the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call get_instance() first.")
        return self._model

    @property
    def vocab_sizes(self) -> dict:
        """Get vocabulary sizes."""
        if self._vocab_sizes is None:
            raise RuntimeError("Model not loaded. Call get_instance() first.")
        return self._vocab_sizes

    @property
    def vocab_mappings(self) -> dict:
        """Get vocabulary mappings (string -> ID)."""
        if self._vocab_mappings is None:
            raise RuntimeError("Model not loaded. Call get_instance() first.")
        return self._vocab_mappings

    @property
    def checkpoint_info(self) -> dict:
        """Get checkpoint information."""
        if self._checkpoint_info is None:
            raise RuntimeError("Model not loaded. Call get_instance() first.")
        return self._checkpoint_info

    @classmethod
    def reset(cls):
        """Reset singleton instance (for testing)."""
        cls._instance = None
        cls._model = None
        cls._vocab_sizes = None
        cls._vocab_mappings = None
        cls._checkpoint_info = None
