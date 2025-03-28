# modules/config_manager.py

import os
from typing import Any, Dict, Optional
import yaml
from pathlib import Path

# Import logger setup, assuming log_manager.py is adjacent
# Use a try-except block for initial loading robustness before logging is fully configured
try:
    from .log_manager import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Could not import custom log_manager. Using default logger.")

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

# --- Private Cache ---
_config_cache: Optional[Dict[str, Any]] = None
_config_path: Optional[Path] = None

# --- Default Config Path ---
DEFAULT_CONFIG_FILENAME = "settings.yaml"

def find_config_file(filename: str = DEFAULT_CONFIG_FILENAME) -> Optional[Path]:
    """
    Searches for the config file in common locations:
    1. Current working directory.
    2. User's home directory.
    3. Script's directory.
    """
    cwd = Path.cwd()
    home = Path.home()
    script_dir = Path(__file__).parent.parent # Project root (one level up from modules)

    search_paths = [
        cwd / filename,
        home / filename,
        script_dir / filename,
    ]

    for path in search_paths:
        if path.is_file():
            logger.debug("Found config file at: %s", path)
            return path

    logger.debug("Config file '%s' not found in standard locations.", filename)
    return None


def load_config(config_path_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the application configuration from a YAML file.

    Uses a cached version after the first load unless an override path is given.
    Searches for the default file if no path is provided.

    Args:
        config_path_override: Explicit path to the configuration file.
                                If provided, it bypasses search and cache.

    Returns:
        A dictionary containing the configuration settings.

    Raises:
        ConfigError: If the configuration file cannot be found or loaded.
    """
    global _config_cache
    global _config_path

    if config_path_override:
        # If override path is given, force reload from that path
        logger.info("Loading configuration from override path: %s", config_path_override)
        path_to_load = Path(config_path_override)
        if not path_to_load.is_file():
            logger.error("Specified configuration file not found: %s", path_to_load)
            raise ConfigError(f"Specified configuration file not found: {path_to_load}")
        _config_path = path_to_load
        _config_cache = None # Force reload
    elif _config_cache is not None and _config_path is not None:
        # Return cached version if no override and already loaded
        logger.debug("Returning cached configuration from: %s", _config_path)
        return _config_cache
    else:
        # Find the default config file if not cached and no override
        logger.debug("Searching for default configuration file '%s'", DEFAULT_CONFIG_FILENAME)
        found_path = find_config_file(DEFAULT_CONFIG_FILENAME)
        if not found_path:
            logger.error("Default configuration file '%s' not found in standard search locations.", DEFAULT_CONFIG_FILENAME)
            raise ConfigError(f"Configuration file '{DEFAULT_CONFIG_FILENAME}' not found.")
        path_to_load = found_path
        _config_path = path_to_load
        logger.info("Loading configuration from: %s", _config_path)


    # Load the YAML file
    try:
        with open(path_to_load, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                raise ConfigError(f"Configuration file '{path_to_load}' is not a valid YAML dictionary.")
            _config_cache = config_data
            logger.debug("Configuration loaded successfully.")
            # Add basic validation or schema check here if needed in the future
            return _config_cache
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file '%s': %s", path_to_load, e, exc_info=True)
        raise ConfigError(f"Error parsing configuration file '{path_to_load}': {e}") from e
    except IOError as e:
        logger.error("Error reading configuration file '%s': %s", path_to_load, e, exc_info=True)
        raise ConfigError(f"Could not read configuration file '{path_to_load}': {e}") from e
    except Exception as e:
        logger.error("An unexpected error occurred while loading config: %s", e, exc_info=True)
        raise ConfigError(f"An unexpected error occurred loading config: {e}") from e


def get_config() -> Dict[str, Any]:
    """
    Returns the loaded configuration dictionary.

    Ensures that the configuration has been loaded, loading it if necessary.

    Returns:
        The configuration dictionary.

    Raises:
        ConfigError: If the configuration hasn't been loaded and cannot be loaded.
    """
    if _config_cache is None:
        logger.warning("Configuration accessed before explicit load. Attempting default load.")
        return load_config() # Attempt to load with defaults
    return _config_cache

def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Retrieves a setting using a dot-separated key path (e.g., "audio.vad.enabled").

    Args:
        key_path: The dot-separated path to the setting.
        default: The value to return if the key is not found. Defaults to None.

    Returns:
        The setting value or the default value.
    """
    config = get_config()
    keys = key_path.split('.')
    value = config
    try:
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                # If we encounter a non-dict while traversing, the path is invalid
                logger.warning("Invalid key path '%s' at segment '%s'.", key_path, key)
                return default
        return value
    except (KeyError, TypeError):
        logger.debug("Setting '%s' not found, returning default value: %s", key_path, default)
        return default