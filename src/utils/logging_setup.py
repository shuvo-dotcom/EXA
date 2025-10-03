import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Dict


_LOGGING_INITIALIZED = False


def _default_logs_dir() -> str:
    # Project root is two levels up from this file: src/utils -> project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    logs_dir = os.path.join(project_root, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        # If we cannot create logs dir, fall back to temp dir
        logs_dir = os.path.join(os.path.expanduser("~"), ".tjai_logs")
        os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def setup_logging(level: int = logging.INFO, logs_dir: Optional[str] = None, filename: str = "app.log") -> None:
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    logs_dir = logs_dir or _default_logs_dir()
    log_path = os.path.join(logs_dir, filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(isinstance(h, TimedRotatingFileHandler) for h in root_logger.handlers):
        file_handler = TimedRotatingFileHandler(log_path, when="midnight", backupCount=7, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    _LOGGING_INITIALIZED = True


def get_logger(name: str = "TJAI", level: int = logging.INFO, logs_dir: Optional[str] = None, filename: str = "app.log") -> logging.Logger:
    """Return a configured logger; initialize root handlers on first call."""
    setup_logging(level=level, logs_dir=logs_dir, filename=filename)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid double propagation to root if handlers are attached directly to child
    logger.propagate = True
    return logger


def make_adapter(logger: logging.Logger, extra: Optional[Dict] = None) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, extra or {})
