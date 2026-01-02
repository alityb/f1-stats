"""
Loss functions for pace decomposition model.
"""

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    MSE loss computed only on valid (non-masked) positions.

    Ignores padding and invalid laps (SC, outlaps, etc.)
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, valid_mask):
        """
        Compute masked MSE loss.

        Args:
            pred: Predicted lap times [B, T]
            target: Target lap times [B, T]
            valid_mask: Boolean mask [B, T] (True = valid lap for training)

        Returns:
            Scalar loss (mean over valid positions)
        """
        # Compute squared error
        squared_error = (pred - target) ** 2

        # Mask out invalid positions
        masked_error = squared_error * valid_mask

        # Mean over valid positions only
        num_valid = valid_mask.sum()

        if num_valid == 0:
            # No valid positions in batch (shouldn't happen with proper filtering)
            return torch.tensor(0.0, device=pred.device)

        loss = masked_error.sum() / num_valid

        return loss


class MaskedRMSELoss(nn.Module):
    """
    RMSE loss computed only on valid positions.

    More interpretable than MSE (in seconds).
    """

    def __init__(self):
        super(MaskedRMSELoss, self).__init__()
        self.mse_loss = MaskedMSELoss()

    def forward(self, pred, target, valid_mask):
        """
        Compute masked RMSE loss.

        Args:
            pred: Predicted lap times [B, T]
            target: Target lap times [B, T]
            valid_mask: Boolean mask [B, T]

        Returns:
            Scalar loss (RMSE in seconds)
        """
        mse = self.mse_loss(pred, target, valid_mask)
        return torch.sqrt(mse)


class PaceDecompositionLoss(nn.Module):
    """
    Complete loss function for pace decomposition model.

    Combines:
    - Masked MSE/RMSE on predictions
    - Regularization terms (embedding norm, driver shrinkage, etc.)
    """

    def __init__(self, config):
        super(PaceDecompositionLoss, self).__init__()

        self.config = config

        # Main prediction loss
        loss_type = config.training.loss.type
        if loss_type == "mse":
            self.pred_loss_fn = MaskedMSELoss()
        elif loss_type == "rmse":
            self.pred_loss_fn = MaskedRMSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Regularization weights
        self.emb_norm_weight = config.training.loss.regularization.embedding_norm_penalty
        self.driver_shrink_weight = config.training.loss.regularization.driver_shrinkage
        self.tyre_smooth_weight = config.training.loss.regularization.tyre_smoothness_penalty

    def forward(self, output, batch, model):
        """
        Compute total loss.

        Args:
            output: Model output dictionary
            batch: Batch dictionary
            model: Model instance (for regularization)

        Returns:
            Dictionary containing total loss and components
        """
        # Prediction loss (masked)
        pred_loss = self.pred_loss_fn(
            output["lap_time_pred"],
            batch["lap_time"],
            batch["valid_mask"]
        )

        # Regularization losses
        reg_losses = model.get_regularization_loss()

        emb_norm_loss = reg_losses["embedding_norm"] * self.emb_norm_weight
        driver_shrink_loss = reg_losses["driver_shrinkage"] * self.driver_shrink_weight

        # Total loss
        total_loss = pred_loss + emb_norm_loss + driver_shrink_loss

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "emb_norm_loss": emb_norm_loss,
            "driver_shrink_loss": driver_shrink_loss,
        }

    def compute_metrics(self, output, batch):
        """
        Compute evaluation metrics (without gradients).

        Args:
            output: Model output dictionary
            batch: Batch dictionary

        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            pred = output["lap_time_pred"]
            target = batch["lap_time"]
            valid_mask = batch["valid_mask"]

            # MSE
            mse = MaskedMSELoss()(pred, target, valid_mask)

            # RMSE
            rmse = torch.sqrt(mse)

            # MAE
            abs_error = torch.abs(pred - target) * valid_mask
            mae = abs_error.sum() / valid_mask.sum()

            return {
                "mse": mse.item(),
                "rmse": rmse.item(),
                "mae": mae.item(),
            }
