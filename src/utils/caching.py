# src/utils/caching.py

import abc
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheManager(abc.ABC):
    """
    Abstract Base Class (ABC) for a cache manager.

    This class defines a clear interface (`load`, `save`) that all caching
    strategies must implement. This use of polymorphism allows the main pipeline
    to be completely decoupled from the specific caching implementation,
    enabling different strategies to be injected for different environments
    (e.g., local development vs. production).
    """

    @abc.abstractmethod
    def load(self, file_path: Path) -> Any | None:
        """Loads data from the cache if it exists."""
        pass

    @abc.abstractmethod
    def save(self, file_path: Path, data: Any) -> None:
        """Saves data to the cache."""
        pass


class FileSystemCacheManager(CacheManager):
    """

    A concrete cache manager that reads and writes JSON files to the local file system.
    This implementation is intended for use during local development and testing to
    speed up iteration by caching the results of computationally expensive steps.
    """

    def load(self, file_path: Path) -> Any | None:
        """Loads data from a JSON file if it exists."""
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    logger.info(f"Cache HIT for: {file_path}")
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load cache file {file_path}: {e}")
                return None
        logger.info(f"Cache MISS for: {file_path}")
        return None

    def save(self, file_path: Path, data: Any) -> None:
        """Saves data to a JSON file, creating parent directories if needed."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Cache SAVED to: {file_path}")
        except IOError as e:
            logger.error(f"Failed to save cache file {file_path}: {e}")


class NoOpCacheManager(CacheManager):
    """
    A cache manager that performs no operations (a "Null Object" pattern).

    This implementation always results in a cache miss, forcing services to run
    their logic every time. It is intended for use in production environments
    (like the API server) where caching intermediate results is not desired.
    """

    def load(self, file_path: Path) -> Any | None:
        """Always returns None to simulate a cache miss."""
        return None

    def save(self, file_path: Path, data: Any) -> None:
        """Performs no operation."""
        pass
