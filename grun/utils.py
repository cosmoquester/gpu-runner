import logging
import os
import sys

import nvidia_smi

from .constant import GRUN_DIR
from .queue import TaskQueue


def initilize():
    nvidia_smi.nvmlInit()

    os.makedirs(GRUN_DIR, exist_ok=True)

    with TaskQueue():
        pass


def get_logger(name: str) -> logging.Logger:
    """Return logger for logging

    Args:
        name: logger name
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[GRUN] [%(asctime)s] %(message)s"))
        logger.addHandler(handler)
    return logger
