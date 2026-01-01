# F1 Race Pace Decomposition

Deep learning approach to decompose F1 race lap times into interpretable components: car performance, driver skill, tyre degradation, and traffic effects.

## Project Structure

```
f1-race-pace/
├── data_ingestion/       # FastF1 data loading and caching
├── feature_engineering/  # Feature extraction and validation
├── ml_dataset/          # Dataset construction and PyTorch loaders
├── models/              # Neural network architectures (TCN, embeddings, heads)
├── training/            # Training loop, loss functions, optimization
├── metrics/             # Derived metrics (TPI, DAB, TME)
├── analysis_notebooks/  # Jupyter notebooks for exploration and validation
├── config/              # Hydra configuration files
├── data/                # Local data storage
│   ├── cache/          # FastF1 cache
│   └── duckdb/         # DuckDB database files
├── dataset/            # Processed datasets
│   └── stints/         # Stint-level sequences
└── exports/            # Final metric tables
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Ingest data from FastF1
python -m data_ingestion.ingest_races

# 2. Build features
python -m feature_engineering.build_features

# 3. Create stint sequences
python -m ml_dataset.create_sequences

# 4. Train model
python -m training.train

# 5. Export metrics
python -m metrics.export_metrics
```

## Configuration

All configuration is managed via Hydra. See `config/` directory.

## Target Seasons

Initial development focuses on 2023-2025 F1 seasons.

## Model Architecture

- **Sequence Encoder**: Temporal Convolutional Network (TCN)
- **Embeddings**: Driver, Car, Track, Season, Compound
- **Prediction Head**: Factorized components (base + car + driver + tyre + traffic)

## Metrics

- **TPI** (True Pace Index): Pure car+driver pace in clean air
- **DAB** (Driver Above Baseline): Driver contribution vs teammate
- **TME** (Tyre Management Efficiency): Degradation curve residuals
