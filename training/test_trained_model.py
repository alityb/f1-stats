"""
Test script for trained model - verify predictions and components.

Usage:
    python -m training.test_trained_model
    python -m training.test_trained_model checkpoint_path=checkpoints/best_model.pt
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
from training.loss import PaceDecompositionLoss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Test trained model."""

    logger.info("=" * 80)
    logger.info("TRAINED MODEL VERIFICATION")
    logger.info("=" * 80)

    # Get checkpoint path
    checkpoint_path = Path(cfg.get("checkpoint_path", "checkpoints/best_model.pt"))

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"\nLoading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,   # restore old behavior
    )
    logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
    logger.info(f"Best validation RMSE: {checkpoint['best_val_metric']:.4f}s")

    # Load dataset
    parquet_path = Path(cfg.paths.stints_dir) / "stint_sequences.parquet"
    dataset = StintDataset(parquet_path, filter_min_valid_laps=3)
    vocab_sizes = dataset.get_vocab_sizes()

    # Create validation split (same seed as training)
    torch.manual_seed(cfg.seed)
    val_size = int(len(dataset) * cfg.training.validation.split_ratio)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed)
    )

    logger.info(f"\nValidation set: {len(val_dataset)} sequences")

    # Create dataloader
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=collate_stints, num_workers=0
    )

    # Initialize model
    model = PaceDecompositionModel(cfg, vocab_sizes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("\n" + "-" * 80)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("-" * 80)

    # Get a batch
    batch = next(iter(val_loader))

    with torch.no_grad():
        output = model(batch)

    # Show predictions for first sequence
    seq_idx = 0
    seq_len = batch["sequence_lengths"][seq_idx].item()

    logger.info(f"\nSequence {seq_idx} (length: {seq_len})")
    logger.info(f"Race ID: {batch['race_id'][seq_idx]}")
    logger.info(f"Stint: {batch['stint_index'][seq_idx]}")

    pred = output["lap_time_pred"][seq_idx, :seq_len].numpy()
    target = batch["lap_time"][seq_idx, :seq_len].numpy()
    valid_mask = batch["valid_mask"][seq_idx, :seq_len].numpy()

    # Component breakdown
    components = {
        name: values[seq_idx, :seq_len].numpy()
        for name, values in output["components"].items()
    }

    logger.info(
        f"\n{'Lap':<5} {'Target':<8} {'Pred':<8} {'Error':<7} {'Valid':<7} "
        f"{'Base':<7} {'Car':<7} {'Driver':<7} {'Tyre':<7} {'Traffic':<7}"
    )
    logger.info("-" * 90)

    for i in range(min(15, seq_len)):
        error = pred[i] - target[i] if valid_mask[i] else 0
        logger.info(
            f"{i+1:<5} {target[i]:<8.2f} {pred[i]:<8.2f} {error:<+7.2f} "
            f"{'✓' if valid_mask[i] else '✗':<7} "
            f"{components['base_pace'][i]:<+7.2f} "
            f"{components['car_contrib'][i]:<+7.2f} "
            f"{components['driver_contrib'][i]:<+7.2f} "
            f"{components['tyre_deg'][i]:<+7.2f} "
            f"{components['traffic_penalty'][i]:<+7.2f}"
        )

    # Compute metrics on validation set
    logger.info("\n" + "-" * 80)
    logger.info("VALIDATION METRICS")
    logger.info("-" * 80)

    loss_fn = PaceDecompositionLoss(cfg)

    total_mse = 0.0
    total_mae = 0.0
    total_valid_laps = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            output = model(batch)

            # Compute metrics
            pred = output["lap_time_pred"]
            target = batch["lap_time"]
            valid_mask = batch["valid_mask"]

            # MSE
            squared_error = (pred - target) ** 2 * valid_mask
            total_mse += squared_error.sum().item()

            # MAE
            abs_error = torch.abs(pred - target) * valid_mask
            total_mae += abs_error.sum().item()

            total_valid_laps += valid_mask.sum().item()
            num_batches += 1

    mse = total_mse / total_valid_laps
    rmse = mse ** 0.5
    mae = total_mae / total_valid_laps

    logger.info(f"\nValidation Metrics (on {total_valid_laps} valid laps):")
    logger.info(f"  MSE:  {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}s")
    logger.info(f"  MAE:  {mae:.4f}s")

    # Component statistics
    logger.info("\n" + "-" * 80)
    logger.info("COMPONENT STATISTICS (Validation Set)")
    logger.info("-" * 80)

    all_components = {
        "base_pace": [],
        "car_contrib": [],
        "driver_contrib": [],
        "tyre_deg": [],
        "traffic_penalty": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            output = model(batch)
            valid_mask = batch["valid_mask"]

            for name, values in output["components"].items():
                # Only collect valid lap contributions
                valid_values = values[valid_mask].numpy()
                all_components[name].extend(valid_values)

    import numpy as np

    logger.info(f"\n{'Component':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    logger.info("-" * 60)

    for name, values in all_components.items():
        values = np.array(values)
        logger.info(
            f"{name:<20} {values.mean():<+10.2f} {values.std():<10.2f} "
            f"{values.min():<+10.2f} {values.max():<+10.2f}"
        )

    logger.info("\n" + "=" * 80)
    logger.info("✅ Model verification complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
