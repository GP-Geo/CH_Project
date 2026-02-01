"""Logging configuration for channel-heads package.

This module provides a centralized logging setup with sensible defaults.
Logs can be configured via environment variables or programmatically.

Environment Variables
---------------------
CHANNEL_HEADS_LOG_LEVEL : str
    Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
CHANNEL_HEADS_LOG_FILE : str
    Path to log file. If not set, logs go to stderr only.

Usage
-----
    from channel_heads.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Processing outlet %d", outlet_id)
    logger.debug("Found %d pairs", len(pairs))
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Package-level logger name
PACKAGE_NAME = "channel_heads"

# Default format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


def get_log_level() -> int:
    """Get log level from environment or default to INFO."""
    level_name = os.getenv("CHANNEL_HEADS_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_log_file() -> Optional[Path]:
    """Get log file path from environment if set."""
    log_file = os.getenv("CHANNEL_HEADS_LOG_FILE")
    return Path(log_file) if log_file else None


def setup_logging(
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure the package-level logger.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g., logging.DEBUG). Defaults to env or INFO.
    log_file : Path, optional
        Path to log file. Defaults to env or None (no file).
    format_string : str, optional
        Log message format. Defaults to timestamp + name + level + message.
    console : bool
        Whether to log to console/stderr. Default True.

    Returns
    -------
    logging.Logger
        Configured package-level logger.

    Example
    -------
    >>> from channel_heads.logging_config import setup_logging
    >>> import logging
    >>> logger = setup_logging(level=logging.DEBUG)
    >>> logger.debug("Debug message")
    """
    if level is None:
        level = get_log_level()
    if log_file is None:
        log_file = get_log_file()
    if format_string is None:
        format_string = DEFAULT_FORMAT

    # Get or create package logger
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(format_string)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Creates a child logger of the package logger. If the package logger
    hasn't been configured, sets up default configuration.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
        Logger instance for the module.

    Example
    -------
    >>> from channel_heads.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
    """
    # Ensure package logger exists with at least a null handler
    package_logger = logging.getLogger(PACKAGE_NAME)
    if not package_logger.handlers:
        # Add NullHandler to avoid "No handler found" warnings
        package_logger.addHandler(logging.NullHandler())

    # If name starts with package name, use as-is; otherwise, prefix it
    if name.startswith(PACKAGE_NAME):
        logger_name = name
    else:
        logger_name = f"{PACKAGE_NAME}.{name}"

    return logging.getLogger(logger_name)


def set_verbose(verbose: bool = True) -> None:
    """Set logging level based on verbosity flag.

    Convenience function for CLI tools.

    Parameters
    ----------
    verbose : bool
        If True, set to DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# Initialize with default configuration on import
_default_logger = setup_logging(console=False)  # Start quiet, CLI enables output
