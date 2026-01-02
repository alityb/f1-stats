"""
Main training script for pace decomposition model.

Usage:
    python -m training.train
    python -m training.train training.epochs=50 training.optimizer.lr=0.0005
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_dataset.stint_dataset import StintDataset, collate_stints
from models.pace_model import PaceDecompositionModel
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_device(config: DictConfig) -> torch.device:
    """Get device for training."""
    device_str = config.training.device

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Using device: {device}")
    return device


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    logger.info("=" * 80)
    logger.info("F1 PACE DECOMPOSITION MODEL - TRAINING")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)

    # Get device
    device = get_device(cfg)

    # 1. Load dataset
    logger.info("\n" + "-" * 80)
    logger.info("Loading Dataset")
    logger.info("-" * 80)

    parquet_path = Path(cfg.paths.stints_dir) / "stint_sequences.parquet"

    if not parquet_path.exists():
        logger.error(f"Dataset not found at {parquet_path}")
        logger.error("Run: python -m ml_dataset.create_sequences")
        return

    dataset = StintDataset(parquet_path, filter_min_valid_laps=3)
    logger.info(f"Loaded {len(dataset)} stint sequences")

    # Get vocabulary sizes
    vocab_sizes = dataset.get_vocab_sizes()
    logger.info(f"Vocabulary sizes: {vocab_sizes}")

    # 2. Train/validation split
    val_ratio = cfg.training.validation.split_ratio
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed)
    )

    logger.info(f"\nTrain set: {len(train_dataset)} sequences")
    logger.info(f"Validation set: {len(val_dataset)} sequences")

    # 3. Create dataloaders
    logger.info("\n" + "-" * 80)
    logger.info("Creating DataLoaders")
    logger.info("-" * 80)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_stints,
        num_workers=cfg.training.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_stints,
        num_workers=cfg.training.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # 4. Initialize model
    logger.info("\n" + "-" * 80)
    logger.info("Initializing Model")
    logger.info("-" * 80)

    model = PaceDecompositionModel(cfg, vocab_sizes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 5. Initialize trainer
    logger.info("\n" + "-" * 80)
    logger.info("Initializing Trainer")
    logger.info("-" * 80)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
    )

    logger.info(f"Optimizer: {cfg.training.optimizer.type}")
    logger.info(f"Learning rate: {cfg.training.optimizer.lr}")
    logger.info(f"Epochs: {cfg.training.epochs}")
    logger.info(f"Batch size: {cfg.training.batch_size}")

    # 5.5. Resume from checkpoint if specified
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            logger.info(f"\n⚠️  Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return

    # 6. Train
    logger.info("\n" + "-" * 80)
    logger.info("Training")
    logger.info("-" * 80)

    trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("✅ Training Complete!")
    logger.info("=" * 80)

    # 7. Save final model
    final_model_path = Path("checkpoints") / "final_model.pt"
    trainer.save_checkpoint("final_model.pt")
    logger.info(f"\nFinal model saved to: {final_model_path}")

    logger.info(f"Best validation RMSE: {trainer.best_val_metric:.4f}s")


if __name__ == "__main__":
    main()
