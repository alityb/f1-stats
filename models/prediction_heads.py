"""
Factorized prediction heads for pace decomposition.

Decomposes lap time into interpretable components:
- Base pace (from TCN sequence encoding)
- Car contribution (team performance)
- Driver contribution (driver skill)
- Tyre degradation
- Traffic penalty
"""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Multi-layer perceptron head."""

    def __init__(self, input_dim, hidden_dims, output_dim=1, activation="relu", dropout=0.1):
        super(MLPHead, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Output tensor [..., output_dim]
        """
        return self.mlp(x)


class BasePaceHead(nn.Module):
    """
    Base pace prediction from TCN sequence encoding.

    Predicts lap time baseline from temporal context.
    """

    def __init__(self, tcn_output_dim, hidden_dims, dropout=0.1):
        super(BasePaceHead, self).__init__()
        self.head = MLPHead(
            input_dim=tcn_output_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
            dropout=dropout,
        )

    def forward(self, tcn_output):
        """
        Args:
            tcn_output: TCN output [batch_size, seq_len, tcn_dim]

        Returns:
            base_pace: [batch_size, seq_len, 1]
        """
        return self.head(tcn_output)


class CarContributionHead(nn.Module):
    """
    Car performance contribution.

    Combines car, track, and season embeddings to predict car-specific pace offset.
    """

    def __init__(self, car_emb_dim, track_emb_dim, season_emb_dim, hidden_dims, dropout=0.1):
        super(CarContributionHead, self).__init__()

        input_dim = car_emb_dim + track_emb_dim + season_emb_dim
        self.head = MLPHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
            dropout=dropout,
        )

    def forward(self, car_emb, track_emb, season_emb):
        """
        Args:
            car_emb: [batch_size, car_emb_dim]
            track_emb: [batch_size, track_emb_dim]
            season_emb: [batch_size, season_emb_dim]

        Returns:
            car_contribution: [batch_size, 1] (broadcast to sequence)
        """
        x = torch.cat([car_emb, track_emb, season_emb], dim=-1)
        return self.head(x)


class DriverContributionHead(nn.Module):
    """
    Driver skill contribution.

    Combines driver and track embeddings to predict driver-specific pace offset.
    Regularized with L2 penalty to shrink toward zero.
    """

    def __init__(self, driver_emb_dim, track_emb_dim, hidden_dims, dropout=0.1):
        super(DriverContributionHead, self).__init__()

        input_dim = driver_emb_dim + track_emb_dim
        self.head = MLPHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
            dropout=dropout,
        )

    def forward(self, driver_emb, track_emb):
        """
        Args:
            driver_emb: [batch_size, driver_emb_dim]
            track_emb: [batch_size, track_emb_dim]

        Returns:
            driver_contribution: [batch_size, 1] (broadcast to sequence)
        """
        x = torch.cat([driver_emb, track_emb], dim=-1)
        return self.head(x)


class TyreDegradationHead(nn.Module):
    """
    Tyre degradation term.

    Predicts degradation penalty based on compound, tyre age, and stint length.
    """

    def __init__(self, compound_emb_dim, continuous_dim, hidden_dims, dropout=0.1):
        super(TyreDegradationHead, self).__init__()

        # continuous_dim includes: tyre_age, stint_length
        input_dim = compound_emb_dim + continuous_dim
        self.head = MLPHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
            dropout=dropout,
        )

    def forward(self, compound_emb, tyre_features):
        """
        Args:
            compound_emb: [batch_size, seq_len, compound_emb_dim]
            tyre_features: [batch_size, seq_len, continuous_dim] (tyre_age, stint_length)

        Returns:
            tyre_degradation: [batch_size, seq_len, 1]
        """
        x = torch.cat([compound_emb, tyre_features], dim=-1)
        return self.head(x)


class TrafficPenaltyHead(nn.Module):
    """
    Traffic penalty term.

    Predicts time loss due to traffic/dirty air.
    """

    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super(TrafficPenaltyHead, self).__init__()

        self.head = MLPHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
            dropout=dropout,
        )

    def forward(self, traffic_features):
        """
        Args:
            traffic_features: [batch_size, seq_len, input_dim]

        Returns:
            traffic_penalty: [batch_size, seq_len, 1]
        """
        return self.head(traffic_features)
