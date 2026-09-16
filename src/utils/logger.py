"""Logging setup for research pipeline execution.

Educational simulation system.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "emergency_health_ml", level: int = logging.INFO) -> logging.Logger:
    """Set up and return structured logger instance.

    Args:
        name: Logger module name.
        level: Logging level.

    Returns:
        logging.Logger configured instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
