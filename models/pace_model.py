"""
F1 Race Pace Decomposition Model.

TCN-based model with factorized prediction heads for interpretable
lap time decomposition.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.tcn import TemporalConvNet
from models.prediction_heads import (
    BasePaceHead,
    CarContributionHead,
    DriverContributionHead,
    TyreDegradationHead,
    TrafficPenaltyHead,
    MLPHead,
)


class PaceDecompositionModel(nn.Module):
    """
    F1 pace decomposition model.

    Architecture:
        1. Embedding layers (driver, car, track, season, compound)
        2. Continuous feature encoder
        3. TCN sequence encoder
        4. Factorized prediction heads:
           - Base pace (from TCN)
           - Car contribution
           - Driver contribution
           - Tyre degradation
           - Traffic penalty

    Final prediction: lap_time = sum of all components
    """

    def __init__(self, config: DictConfig, vocab_sizes: dict):
        super(PaceDecompositionModel, self).__init__()

        self.config = config
        self.vocab_sizes = vocab_sizes

        # Extract dimensions from config
        emb_cfg = config.model.embeddings
        tcn_cfg = config.model.tcn
        heads_cfg = config.model.prediction_heads
        cont_cfg = config.model.continuous_encoder

        # 1. Embedding layers
        self.driver_embedding = nn.Embedding(vocab_sizes["driver"], emb_cfg.driver_dim)
        self.car_embedding = nn.Embedding(vocab_sizes["car"], emb_cfg.car_dim)
        self.track_embedding = nn.Embedding(vocab_sizes["track"], emb_cfg.track_dim)
        self.season_embedding = nn.Embedding(vocab_sizes["season"], emb_cfg.season_dim)
        self.compound_embedding = nn.Embedding(vocab_sizes["compound"], emb_cfg.compound_dim)

        # 2. Continuous feature encoder
        # Features: lap_in_stint, tyre_age, fuel_proxy, stint_length
        num_continuous_features = 4
        self.continuous_encoder = MLPHead(
            input_dim=num_continuous_features,
            hidden_dims=cont_cfg.hidden_dims,
            output_dim=cont_cfg.hidden_dims[-1],  # Output dimension
            activation=cont_cfg.activation,
            dropout=cont_cfg.dropout,
        )

        # 3. Compute token dimension (for TCN input)
        # Token per timestep = sum of all embeddings + continuous encoder output
        self.token_dim = (
            emb_cfg.driver_dim
            + emb_cfg.car_dim
            + emb_cfg.track_dim
            + emb_cfg.season_dim
            + emb_cfg.compound_dim
            + cont_cfg.hidden_dims[-1]  # Continuous encoder output
        )

        # 4. TCN sequence encoder
        self.tcn = TemporalConvNet(
            num_inputs=self.token_dim,
            num_channels=tcn_cfg.num_channels,
            kernel_size=tcn_cfg.kernel_size,
            dropout=tcn_cfg.dropout,
        )

        self.tcn_output_dim = tcn_cfg.num_channels[-1]

        # 5. Factorized prediction heads
        # Base pace head (from TCN output)
        self.base_head = BasePaceHead(
            tcn_output_dim=self.tcn_output_dim,
            hidden_dims=heads_cfg.base.hidden_dims,
            dropout=heads_cfg.base.get("dropout", 0.1),
        )

        # Car contribution head
        self.car_head = CarContributionHead(
            car_emb_dim=emb_cfg.car_dim,
            track_emb_dim=emb_cfg.track_dim,
            season_emb_dim=emb_cfg.season_dim,
            hidden_dims=heads_cfg.car.hidden_dims,
            dropout=heads_cfg.car.get("dropout", 0.1),
        )

        # Driver contribution head
        self.driver_head = DriverContributionHead(
            driver_emb_dim=emb_cfg.driver_dim,
            track_emb_dim=emb_cfg.track_dim,
            hidden_dims=heads_cfg.driver.hidden_dims,
            dropout=heads_cfg.driver.get("dropout", 0.1),
        )

        # Tyre degradation head
        # Continuous tyre features: tyre_age, stint_length (2 features)
        self.tyre_head = TyreDegradationHead(
            compound_emb_dim=emb_cfg.compound_dim,
            continuous_dim=2,  # tyre_age, stint_length
            hidden_dims=heads_cfg.tyre.hidden_dims,
            dropout=heads_cfg.tyre.get("dropout", 0.1),
        )

        # Traffic penalty head
        # For now, simple feature: just continuous encoded features
        # Could be extended with gap data, etc.
        self.traffic_head = TrafficPenaltyHead(
            input_dim=cont_cfg.hidden_dims[-1],
            hidden_dims=heads_cfg.traffic.hidden_dims,
            dropout=heads_cfg.traffic.get("dropout", 0.1),
        )

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dictionary containing:
                - driver_id: [B]
                - car_id: [B]
                - track_id: [B]
                - season_id: [B]
                - compound: [B, T]
                - lap_in_stint: [B, T]
                - tyre_age: [B, T]
                - fuel_proxy: [B, T]
                - stint_length: [B, T]
                - padding_mask: [B, T]

        Returns:
            Dictionary containing:
                - lap_time_pred: [B, T] - total predicted lap time
                - components: Dictionary of individual contributions
        """
        batch_size = batch["driver_id"].shape[0]
        seq_len = batch["compound"].shape[1]

        # 1. Get embeddings
        driver_emb = self.driver_embedding(batch["driver_id"])  # [B, driver_dim]
        car_emb = self.car_embedding(batch["car_id"])  # [B, car_dim]
        track_emb = self.track_embedding(batch["track_id"])  # [B, track_dim]
        season_emb = self.season_embedding(batch["season_id"])  # [B, season_dim]
        compound_emb = self.compound_embedding(batch["compound"])  # [B, T, compound_dim]

        # 2. Encode continuous features
        # Stack: [lap_in_stint, tyre_age, fuel_proxy, stint_length]
        continuous_features = torch.stack(
            [
                batch["lap_in_stint"],
                batch["tyre_age"],
                batch["fuel_proxy"],
                batch["stint_length"],
            ],
            dim=-1,
        )  # [B, T, 4]

        continuous_emb = self.continuous_encoder(continuous_features)  # [B, T, cont_dim]

        # 3. Build token representation per timestep
        # Broadcast stint-level embeddings to sequence
        driver_emb_seq = driver_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, driver_dim]
        car_emb_seq = car_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, car_dim]
        track_emb_seq = track_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, track_dim]
        season_emb_seq = season_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, season_dim]

        # Concatenate all features
        tokens = torch.cat(
            [
                driver_emb_seq,
                car_emb_seq,
                track_emb_seq,
                season_emb_seq,
                compound_emb,
                continuous_emb,
            ],
            dim=-1,
        )  # [B, T, token_dim]

        # 4. TCN requires [B, C, T] format
        tcn_input = tokens.transpose(1, 2)  # [B, token_dim, T]
        tcn_output = self.tcn(tcn_input)  # [B, tcn_out_dim, T]
        tcn_output = tcn_output.transpose(1, 2)  # [B, T, tcn_out_dim]

        # 5. Compute factorized components
        # Base pace (from TCN)
        base_pace = self.base_head(tcn_output).squeeze(-1)  # [B, T]

        # Car contribution (stint-level, broadcast to sequence)
        car_contrib = self.car_head(car_emb, track_emb, season_emb)  # [B, 1]
        car_contrib = car_contrib.expand(-1, seq_len)  # [B, T]

        # Driver contribution (stint-level, broadcast to sequence)
        driver_contrib = self.driver_head(driver_emb, track_emb)  # [B, 1]
        driver_contrib = driver_contrib.expand(-1, seq_len)  # [B, T]

        # Tyre degradation (per timestep)
        tyre_features = torch.stack(
            [batch["tyre_age"], batch["stint_length"]], dim=-1
        )  # [B, T, 2]
        tyre_deg = self.tyre_head(compound_emb, tyre_features).squeeze(-1)  # [B, T]

        # Traffic penalty (per timestep)
        traffic_penalty = self.traffic_head(continuous_emb).squeeze(-1)  # [B, T]

        # 6. Sum all components
        lap_time_pred = (
            base_pace + car_contrib + driver_contrib + tyre_deg + traffic_penalty
        )  # [B, T]

        return {
            "lap_time_pred": lap_time_pred,
            "components": {
                "base_pace": base_pace,
                "car_contrib": car_contrib,
                "driver_contrib": driver_contrib,
                "tyre_deg": tyre_deg,
                "traffic_penalty": traffic_penalty,
            },
        }

    def get_regularization_loss(self):
        """
        Compute regularization losses.

        Returns:
            Dictionary of regularization terms
        """
        reg_losses = {}

        # Embedding norm penalty
        emb_norm = (
            self.driver_embedding.weight.norm()
            + self.car_embedding.weight.norm()
            + self.track_embedding.weight.norm()
            + self.season_embedding.weight.norm()
            + self.compound_embedding.weight.norm()
        )
        reg_losses["embedding_norm"] = emb_norm

        # Driver shrinkage (L2 penalty on driver head parameters)
        driver_param_norm = sum(p.norm() for p in self.driver_head.parameters())
        reg_losses["driver_shrinkage"] = driver_param_norm

        return reg_losses
