"""
FastF1 data loader with local caching.
"""

import logging
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FastF1Loader:
    """Loads F1 session data using FastF1 API with caching."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.cache_dir = Path(config.paths.cache_dir)
        self._setup_cache()

    def _setup_cache(self):
        """Initialize FastF1 cache directory."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.config.data.fastf1.cache_enabled:
            fastf1.Cache.enable_cache(str(self.cache_dir))
            logger.info(f"FastF1 cache enabled at {self.cache_dir}")
        else:
            logger.warning("FastF1 cache is disabled")

    def load_session(
        self, year: int, round_number: int, session_type: str = "Race"
    ) -> Optional[fastf1.core.Session]:
        """
        Load a specific F1 session.

        Args:
            year: Season year
            round_number: Race round number (1-based)
            session_type: Session type ('Race', 'Qualifying', 'Sprint', etc.)

        Returns:
            Loaded session object, or None if loading fails
        """
        try:
            logger.info(f"Loading {year} Round {round_number} - {session_type}")
            session = fastf1.get_session(year, round_number, session_type)
            session.load()
            logger.info(
                f"Successfully loaded {session.event['EventName']} - {session_type}"
            )
            return session
        except Exception as e:
            logger.error(
                f"Failed to load {year} R{round_number} {session_type}: {e}"
            )
            return None

    def extract_laps_dataframe(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract laps dataframe from a loaded session.

        Args:
            session: Loaded FastF1 session

        Returns:
            DataFrame containing all lap data
        """
        laps = session.laps
        logger.info(f"Extracted {len(laps)} laps from session")
        return laps

    def extract_race_metadata(self, session: fastf1.core.Session) -> dict:
        """
        Extract race metadata from session.

        Args:
            session: Loaded FastF1 session

        Returns:
            Dictionary containing race metadata
        """
        event = session.event

        # Convert numpy/pandas types to native Python types for DuckDB compatibility
        metadata = {
            "season": int(event["EventDate"].year),
            "round_number": int(event["RoundNumber"]),
            "event_name": str(event["EventName"]),
            "country": str(event["Country"]) if pd.notna(event["Country"]) else None,
            "location": str(event["Location"]) if pd.notna(event["Location"]) else None,
            "circuit_key": str(event.get("CircuitKey", event["Location"])),
            "event_date": pd.Timestamp(event["EventDate"]).date(),
            "session_type": str(session.name),
        }
        logger.debug(f"Extracted metadata: {metadata}")
        return metadata

    def load_season_races(
        self, year: int, session_type: str = "Race"
    ) -> list[tuple[fastf1.core.Session, dict]]:
        """
        Load all races from a season.

        Args:
            year: Season year
            session_type: Session type to load

        Returns:
            List of (session, metadata) tuples
        """
        logger.info(f"Loading all {session_type} sessions for {year} season")

        # Get race schedule for the year
        try:
            schedule = fastf1.get_event_schedule(year)
            race_rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"].tolist()
        except Exception as e:
            logger.error(f"Failed to get schedule for {year}: {e}")
            return []

        sessions_data = []
        for round_num in race_rounds:
            session = self.load_session(year, round_num, session_type)
            if session is not None:
                metadata = self.extract_race_metadata(session)
                sessions_data.append((session, metadata))
            else:
                logger.warning(f"Skipping {year} Round {round_num}")

        logger.info(
            f"Successfully loaded {len(sessions_data)}/{len(race_rounds)} "
            f"{session_type} sessions for {year}"
        )
        return sessions_data
