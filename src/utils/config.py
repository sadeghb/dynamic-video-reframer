# src/utils/config.py

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Private, module-level variable to cache the loaded configuration in memory.
_config: Dict[str, Any] | None = None


def load_config() -> Dict[str, Any]:
    """
    Loads the main project configuration from the 'config/config.yaml' file.

    This function implements an in-memory cache. The configuration is read from
    the file system only on the first call, and subsequent calls return the
    cached result to prevent unnecessary file I/O.

    Raises:
        FileNotFoundError: If the config.yaml file cannot be found.
        TypeError: If the loaded YAML content is not a dictionary.
        YAMLError: If the file is not valid YAML.

    Returns:
        A dictionary containing the application configuration.
    """
    global _config
    
    # Return the cached config if it has already been loaded.
    if _config is not None:
        return _config

    config_path = Path("config/config.yaml")
    logger.info(f"Loading configuration for the first time from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            # Add a type check for robustness
            if not isinstance(loaded_config, dict):
                raise TypeError("config.yaml did not load as a valid dictionary.")
            
            _config = loaded_config
            return _config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at '{config_path}'.")
        raise
    except (yaml.YAMLError, TypeError) as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise
