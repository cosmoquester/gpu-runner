import json
import os
from typing import List

from filelock import BaseFileLock, FileLock

from .constant import GRUN_QUEUE, GRUN_QUEUE_LOCK
from .types import Task


class TaskQueue:
    _queue = None
    _init = False

    def __new__(cls) -> "TaskQueue":
        if cls._queue is None:
            cls._queue = super().__new__(cls)
        return cls._queue

    def __init__(self) -> None:
        if not TaskQueue._init:
            self.queue_lock: BaseFileLock = FileLock(GRUN_QUEUE_LOCK)
            self.queue_info: List[Task] = []
            TaskQueue._init = True

    def __enter__(self) -> "TaskQueue":
        self.queue_lock.acquire()
        if not os.path.exists(GRUN_QUEUE):
            with open(GRUN_QUEUE, "w", encoding="utf-8") as f:
                json.dump([], f)
        with open(GRUN_QUEUE, "r", encoding="utf-8") as f:
            self.queue_info = json.load(f)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        with open(GRUN_QUEUE, "w", encoding="utf-8") as f:
            json.dump(self.queue_info, f)
        self.queue_lock.release()

    def __len__(self) -> int:
        return len(self.queue_info)

    def __getitem__(self, idx: int) -> Task:
        return self.queue_info[idx]

    def enqueue(self, commando: str) -> None:
        self.queue_info.append({"pid": os.getpid(), "command": commando})

    def dequeue(self) -> Task:
        return self.queue_info.pop(0)
