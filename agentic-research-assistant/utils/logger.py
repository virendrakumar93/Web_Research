"""
Logging Utility
===============

Purpose:
    Provides a centralized, configured logger for all modules in the
    research assistant system.

Inputs:
    - module_name: Name of the module requesting a logger instance.

Outputs:
    - A configured logging.Logger instance with console output and
      consistent formatting.

Role in Architecture:
    Cross-cutting concern used by every module for structured logging
    and debugging output.
"""

import logging
import sys


def get_logger(module_name: str) -> logging.Logger:
    """Create and return a configured logger for the given module.

    Args:
        module_name: The name of the calling module (typically __name__).

    Returns:
        A logging.Logger instance with console handler and formatter.
    """
    logger = logging.getLogger(module_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
