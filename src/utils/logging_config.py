"""
Logging configuration for Particle Flow Filters.

Provides a centralized logging setup with consistent formatting across all modules.

Usage:
    from src.utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Starting experiment...")
    logger.debug("Debug details: %s", details)
    logger.warning("Low ESS detected")
    logger.error("Filter diverged")

Configuration:
    - LOG_LEVEL environment variable controls verbosity (DEBUG, INFO, WARNING, ERROR)
    - Default level is INFO
    - Logs to both console and file (if LOG_FILE is set)
"""

import logging
import os
import sys
from typing import Optional


# Default format includes timestamp, level, module, and message
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the entire application.
    
    Should be called once at the start of the program (e.g., in main()).
    
    Parameters
    ----------
    level : str, optional
        Logging level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        Defaults to LOG_LEVEL environment variable or INFO.
    log_file : str, optional
        Path to log file. If None, uses LOG_FILE environment variable.
        If neither is set, logs only to console.
    format_string : str, optional
        Custom format string. Defaults to DEFAULT_FORMAT.
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    # Determine level
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Determine format
    if format_string is None:
        format_string = DEFAULT_FORMAT
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=DEFAULT_DATE_FORMAT)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is None:
        log_file = os.environ.get("LOG_FILE")
    
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress verbose TensorFlow logging
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    
    Automatically configures logging if not already done.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__).
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
        
    Example
    -------
    >>> logger = get_logger(__name__)
    >>> logger.info("Experiment started with %d particles", num_particles)
    """
    # Auto-configure if needed
    if not _logging_configured:
        setup_logging()
    
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """
    Change the logging level at runtime.
    
    Parameters
    ----------
    level : str
        New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(numeric_level)


class LoggerAdapter:
    """
    Adapter to make print-style logging easier during migration.
    
    Example
    -------
    >>> log = LoggerAdapter(__name__)
    >>> log("This works like print")  # logs at INFO level
    >>> log.debug("Debug message")
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def __call__(self, message: str, *args, **kwargs):
        """Log at INFO level (default)."""
        self.logger.info(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)
