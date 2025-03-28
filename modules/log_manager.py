# modules/log_manager.py

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Any

# Make sure rich is installed: pip install rich
# Import only the module name, check for it later
_rich_logging_available = False
try:
    from rich.logging import RichHandler
    _rich_logging_available = True
except ImportError:
    RichHandler = None # Keep None available for type hinting if needed later

# --- Constants ---
LOG_DIR = "log"; MAX_LOG_SIZE_BYTES = 5*1024*1024; LOG_BACKUP_COUNT = 3
FILE_LOG_FORMAT = "%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Global Logger ---
logger = logging.getLogger("morpheus")

# --- Configuration Function ---
def setup_logging(
    log_level: str = "INFO",
    log_dir: str = LOG_DIR,
    max_log_size: int = MAX_LOG_SIZE_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
    app_name: str = "morpheus"
) -> logging.Logger:
    """Configures logging with Rich console (if available) and file logging."""
    global logger
    logger = logging.getLogger(app_name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Prevent adding multiple handlers
    if logger.hasHandlers(): logger.handlers.clear()

    # --- Console Handler ---
    console_handler: Optional[logging.Handler] = None
    if _rich_logging_available and RichHandler is not None: # Check BOTH flag and type alias
        try:
            # --- Try creating Rich Handler ONLY if import succeeded ---
            console_handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=True,
                show_level=True,
            )
            console_handler.setLevel(level)
            logger.addHandler(console_handler)
            # Log first message using potential Rich handler
            logger.info("Rich console handler enabled. Console Level: %s.", log_level.upper())
        except Exception as rich_err:
            print(f"Warning: Failed to initialize RichHandler: {rich_err}. Falling back.", file=sys.stderr)
            # Ensure handler is None if init failed
            console_handler = None
            # Remove potentially partially added handler if logger has it
            if logger.hasHandlers():
                logger.handlers[:] = [h for h in logger.handlers if not isinstance(h, RichHandler)]

    # Fallback or if RichHandler failed/unavailable
    if console_handler is None:
        # Check if a basic handler is already added (e.g., by basicConfig guard)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
             console_handler = logging.StreamHandler(sys.stdout)
             console_handler.setLevel(level)
             basic_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
             console_handler.setFormatter(basic_formatter)
             logger.addHandler(console_handler)
             logger.info("Using basic console logging. Console Level: %s.", log_level.upper())
        else:
             # A basic handler (likely from basicConfig) already exists, use it
             logger.info("Basic console handler already present. Console Level: %s.", log_level.upper())


    # --- File Handlers ---
    try:
        os.makedirs(log_dir, exist_ok=True)
        # General File Handler
        log_filename = os.path.join(log_dir, f"{app_name}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = RotatingFileHandler(log_filename, maxBytes=max_log_size, backupCount=backup_count, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        # Error File Handler
        error_log_filename = os.path.join(log_dir, f"{app_name}_errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler = RotatingFileHandler(error_log_filename, maxBytes=max_log_size, backupCount=backup_count, encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        logger.debug("File logging initialized to directory: '%s'", log_dir)
    except Exception as e:
        logger.error("Failed to set up file logging: %s", e, exc_info=True)
        logger.warning("File logging is disabled.")

    # --- Capture warnings ---
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.propagate = False
    # Clear any previous handlers from warnings logger
    if warnings_logger.hasHandlers(): warnings_logger.handlers.clear()
    # Add *current* handlers from main logger to warnings logger
    for handler in logger.handlers:
         # Avoid adding file handlers multiple times if setup is called again
         if not any(isinstance(h, type(handler)) and getattr(h, 'baseFilename', None) == getattr(handler, 'baseFilename', ' ') for h in warnings_logger.handlers):
              warnings_logger.addHandler(handler)

    logger.info("Logging setup complete.")
    return logger

# --- Initial Setup Guard REMOVED ---
# Removing this guard. If logger is accessed early, it might raise NoHandlerError,
# which is acceptable as setup_logging *must* be called by main() after config load.
# This prevents the fallback basicConfig from interfering with RichHandler setup.
# if not logger.hasHandlers():
#      logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
#      logger.warning("Logger accessed before explicit setup. Using basicConfig as fallback.")