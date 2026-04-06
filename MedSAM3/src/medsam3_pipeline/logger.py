from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure console and file logging for a pipeline run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("medsam3_pipeline")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def close_logging(logger: logging.Logger) -> None:
    """Close and detach all handlers from a logger."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
