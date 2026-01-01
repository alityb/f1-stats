"""
Utility script to reset (delete) the DuckDB database.

Usage:
    python -m data_ingestion.reset_db
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Delete the DuckDB database file."""
    db_path = Path(cfg.data.duckdb.db_path)

    if db_path.exists():
        print(f"Deleting database at: {db_path}")
        db_path.unlink()
        print("Database deleted successfully!")

        # Also delete WAL file if it exists
        wal_path = db_path.with_suffix(".db.wal")
        if wal_path.exists():
            wal_path.unlink()
            print("WAL file deleted.")
    else:
        print(f"No database found at: {db_path}")

    print("\nYou can now run ingestion again with the updated schema.")


if __name__ == "__main__":
    main()
