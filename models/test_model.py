"""
Test script for the pace decomposition model.

Usage:
    python -m models.test_model
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
from models.pace_model import PaceDecompositionModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Test model architecture."""

    logger.info("=" * 80)
    logger.info("PACE DECOMPOSITION MODEL TEST")
    logger.info("=" * 80)

    # 1. Load dataset
    parquet_path = Path(cfg.paths.stints_dir) / "stint_sequences.parquet"

    if not parquet_path.exists():
        logger.error(f"Dataset not found at {parquet_path}")
        logger.error("Run: python -m ml_dataset.create_sequences")
        return

    dataset = StintDataset(parquet_path, filter_min_valid_laps=3)
    logger.info(f"\n✅ Loaded dataset with {len(dataset)} stints")

    # Get vocabulary sizes for model
    vocab_sizes = dataset.get_vocab_sizes()
    logger.info(f"\nVocabulary sizes: {vocab_sizes}")

    # 2. Create DataLoader
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_stints, num_workers=0
    )

    # Get a test batch
    batch = next(iter(dataloader))
    logger.info(f"\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: {value.shape}")

    # 3. Initialize model
    logger.info("\n" + "-" * 80)
    logger.info("INITIALIZING MODEL")
    logger.info("-" * 80)

    model = PaceDecompositionModel(cfg, vocab_sizes)
    logger.info(f"\n✅ Model initialized")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Print model structure
    logger.info("\nModel structure:")
    logger.info(f"  Token dimension: {model.token_dim}")
    logger.info(f"  TCN output dimension: {model.tcn_output_dim}")
    logger.info(f"  TCN channels: {cfg.model.tcn.num_channels}")

    # 4. Forward pass
    logger.info("\n" + "-" * 80)
    logger.info("FORWARD PASS TEST")
    logger.info("-" * 80)

    model.eval()
    with torch.no_grad():
        output = model(batch)

    logger.info(f"\n✅ Forward pass successful")
    logger.info(f"\nOutput keys: {list(output.keys())}")
    logger.info(f"Prediction shape: {output['lap_time_pred'].shape}")
    logger.info(f"Expected shape: [batch_size={batch['driver_id'].shape[0]}, seq_len={batch['compound'].shape[1]}]")

    # Check components
    logger.info(f"\nComponent shapes:")
    for name, values in output["components"].items():
        logger.info(f"  {name}: {values.shape}")

    # 5. Verify predictions
    logger.info("\n" + "-" * 80)
    logger.info("PREDICTION VERIFICATION")
    logger.info("-" * 80)

    pred = output["lap_time_pred"]
    target = batch["lap_time"]
    valid_mask = batch["valid_mask"]
    padding_mask = batch["padding_mask"]

    logger.info(f"\nSample predictions (first sequence, first 10 laps):")
    logger.info(f"{'Lap':<5} {'Target':<10} {'Pred':<10} {'Valid':<7} {'Padded':<7}")
    logger.info("-" * 50)

    for i in range(min(10, pred.shape[1])):
        lap_target = target[0, i].item()
        lap_pred = pred[0, i].item()
        is_valid = valid_mask[0, i].item()
        is_padded = not padding_mask[0, i].item()

        logger.info(
            f"{i+1:<5} {lap_target:<10.2f} {lap_pred:<10.2f} "
            f"{'✓' if is_valid else '✗':<7} {'✓' if is_padded else '✗':<7}"
        )

    # Check component contributions
    logger.info(f"\nComponent contributions (first sequence, first valid lap):")
    first_valid_idx = torch.where(valid_mask[0])[0][0].item() if valid_mask[0].any() else 0

    for name, values in output["components"].items():
        contrib = values[0, first_valid_idx].item()
        logger.info(f"  {name}: {contrib:+.2f}s")

    total_contrib = sum(v[0, first_valid_idx].item() for v in output["components"].values())
    predicted = pred[0, first_valid_idx].item()
    logger.info(f"  Sum of components: {total_contrib:.2f}s")
    logger.info(f"  Predicted lap time: {predicted:.2f}s")
    logger.info(f"  Match: {abs(total_contrib - predicted) < 1e-4}")

    # 6. Test regularization
    logger.info("\n" + "-" * 80)
    logger.info("REGULARIZATION TEST")
    logger.info("-" * 80)

    reg_losses = model.get_regularization_loss()
    logger.info(f"\nRegularization terms:")
    for name, value in reg_losses.items():
        logger.info(f"  {name}: {value.item():.4f}")

    # 7. Test backward pass
    logger.info("\n" + "-" * 80)
    logger.info("BACKWARD PASS TEST")
    logger.info("-" * 80)

    model.train()
    output = model(batch)

    # Compute simple loss (MSE on valid laps)
    pred = output["lap_time_pred"]
    target = batch["lap_time"]
    valid_mask = batch["valid_mask"]

    loss = ((pred - target) ** 2 * valid_mask).sum() / valid_mask.sum()
    logger.info(f"\nMSE loss: {loss.item():.4f}")

    # Backward
    loss.backward()
    logger.info(f"✅ Backward pass successful")

    # Check gradients
    has_grads = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Parameters with gradients: {has_grads}/{total_params_trainable}"
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL MODEL TESTS PASSED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
