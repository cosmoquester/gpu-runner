import os

import nvidia_smi

from .constant import GRUN_DIR
from .queue import TaskQueue


def initilize():
    nvidia_smi.nvmlInit()

    os.makedirs(GRUN_DIR, exist_ok=True)

    with TaskQueue():
        pass
