import json
import os
from typing import List

from filelock import BaseFileLock, FileLock

from .constant import GRUN_QUEUE, GRUN_QUEUE_LOCK
from .types import Task


class TaskQueue:
    _queue_lock: BaseFileLock = FileLock(GRUN_QUEUE_LOCK)
    _queue_info: List[Task] = []

    def __enter__(self) -> "TaskQueue":
        TaskQueue._queue_lock.acquire()
        if not os.path.exists(GRUN_QUEUE):
            with open(GRUN_QUEUE, "w", encoding="utf-8") as f:
                json.dump([], f)
        with open(GRUN_QUEUE, "r", encoding="utf-8") as f:
            TaskQueue._queue_info = json.load(f)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        with open(GRUN_QUEUE, "w", encoding="utf-8") as f:
            json.dump(TaskQueue._queue_info, f)
        TaskQueue._queue_lock.release()

    @staticmethod
    def __len__() -> int:
        return len(TaskQueue._queue_info)

    @staticmethod
    def __getitem__(idx: int) -> Task:
        return TaskQueue._queue_info[idx]

    @staticmethod
    def enqueue(commando: str) -> None:
        TaskQueue._queue_info.append({"pid": os.getpid(), "command": commando})

    @staticmethod
    def dequeue() -> Task:
        return TaskQueue._queue_info.pop(0)


queue = TaskQueue()
