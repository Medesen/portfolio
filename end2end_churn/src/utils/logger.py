"""
Structured logging configuration for the churn prediction service.

This module provides centralized logging configuration with:
- Console and file handlers
- Rotating file handler to prevent disk issues
- Different log levels for different environments
- Structured formatting with timestamps and context
- Request ID context for distributed tracing
"""

import logging
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Context variable for request ID (thread-safe for async)
# This allows us to track a request ID across async operations
request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """
    Logging filter that injects request_id into log records.

    This allows every log message to include the request ID for tracing,
    making it easy to grep logs for a specific request.

    Example log output:
        2025-10-29 08:00:15 - INFO - [req:abc-123] Prediction request received
        2025-10-29 08:00:15 - INFO - [req:abc-123] Model loaded successfully
        2025-10-29 08:00:15 - INFO - [req:abc-123] Prediction complete
    """

    def filter(self, record):
        """Add request_id to the log record."""
        request_id = request_id_context.get()
        record.request_id = request_id if request_id else "no-request-id"
        return True


def setup_logger(
    name: str = "churn_service", log_level: str = "INFO", log_file: str = "logs/churn_service.log"
) -> logging.Logger:
    """
    Configure and return a logger with console and file handlers.

    The logger uses a RotatingFileHandler to prevent log files from growing
    indefinitely. When a log file reaches 10MB, it's rotated and renamed with
    a .1 suffix, and a new log file is created. Up to 5 backup files are kept.

    Args:
        name: Logger name (used to distinguish different loggers)
        log_level: Logging level - DEBUG, INFO, WARNING, ERROR, or CRITICAL
            - DEBUG: Detailed information for debugging
            - INFO: General informational messages
            - WARNING: Warning messages (something unexpected but handled)
            - ERROR: Error messages (something failed)
            - CRITICAL: Critical errors (service is broken)
        log_file: Path to log file (relative to project root)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("my_service", "INFO", "logs/my_service.log")
        >>> logger.info("Service started")
        >>> logger.warning("Schema mismatch detected")
        >>> logger.error("Failed to process request", exc_info=True)
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers if logger already configured
    # This is important when logger is imported multiple times
    if logger.handlers:
        return logger

    # Prevent propagation to root logger
    logger.propagate = False

    # Create formatters with request ID for distributed tracing
    # Detailed formatter for file logs (includes filename, line number, and request ID)
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Simple formatter for console (includes request ID for tracing)
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (stdout)
    # This ensures logs appear in Docker logs and terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    console_handler.addFilter(RequestIdFilter())  # Add request ID filter for tracing
    logger.addHandler(console_handler)

    # File handler (rotating)
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # RotatingFileHandler prevents disk from filling up
        # maxBytes: When file reaches this size, rotate it
        # backupCount: Number of backup files to keep
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,  # Keep 5 old files (total 60MB max)
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(detailed_formatter)
        file_handler.addFilter(RequestIdFilter())  # Add request ID filter for tracing
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    This is useful when you want to get a logger that was already configured
    by setup_logger() in another module.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> # In module A
        >>> logger = setup_logger("churn_service")
        >>>
        >>> # In module B
        >>> logger = get_logger("churn_service")  # Gets same logger
    """
    return logging.getLogger(name)
