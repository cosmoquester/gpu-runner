import argparse
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from shutil import copytree
from typing import List, Optional, Tuple

import nvidia_smi
import psutil
from filelock import FileLock, Timeout
from pynvml.nvml import NVMLError_NoPermission, NVMLError_NotSupported

from .constant import GRUN_DIR
from .queue import queue
from .utils import initilize

initilize()

vgpu_warning = False


# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1, help="Number of GPUs to acquire.")
parser.add_argument("--wait", "-w", action="store_true", help="Wait until the required number of GPUs are available.")
parser.add_argument("--freeze", "-f", action="store_true", help="Freeze current working directory to ensure execute current code after wating.")
parser.add_argument("commands", nargs=argparse.REMAINDER, help="Commands to run.")
# fmt: on


def get_nonutilized_gpus(num_gpus: int) -> List[int]:
    """Find available GPUs based on utilization and memory usage.

    Args:
        num_gpus (int): Number of total GPUs.
    Returns:
        List[int]: List of non-utilized GPUs.
    """
    available_gpus = []
    for i in range(num_gpus):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)

        try:
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        except (NVMLError_NoPermission, NVMLError_NotSupported):
            global vgpu_warning
            if not vgpu_warning:
                vgpu_warning = True
                print("[GRUN]", "Error: VGPU is not allow to access the GPU information.", file=sys.stderr)
                print("[GRUN]", "Pass checking the GPU utilization.", file=sys.stderr)
            available_gpus.append(i)
            continue

        if util.gpu == 0 and mem.used / mem.total < 0.01:
            available_gpus.append(i)
    return available_gpus


def acquire_n_gpus(gpus: List[int], n: int) -> List[Tuple[int, FileLock]]:
    """Acquire and Lock n GPUs based on the list of GPUs.

    Args:
        gpus (List[int]): List of GPUs.
        n (int): Number of GPUs to acquire.
    Returns:
        List[Tuple[int, FileLock]]: List of acquired GPUs and their locks.
    """
    lock_files = [(gpu, os.path.join(GRUN_DIR, f"gpu_{gpu}.lock")) for gpu in gpus]
    locked_gpus = []

    for gpu, lock_file in lock_files:
        lock = FileLock(lock_file)

        try:
            lock.acquire(timeout=0.1)
        except Timeout:
            continue

        locked_gpus.append((gpu, lock))
        if len(locked_gpus) == n:
            break

    return locked_gpus


def ensure_n_gpus(n_gpus: int, num_required_gpus: int, interval: int = 3) -> List[Tuple[int, FileLock]]:
    """Ensure n GPUs are available by waiting and acquiring.

    Args:
        n_gpus (int): Number of total GPUs.
        num_required_gpus (int): Number of GPUs to acquire.
        interval (int, optional): Interval to check the availability of GPUs. Defaults to 3.
    Returns:
        List[Tuple[int, FileLock]]: List of acquired GPUs and their locks.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[GRUN]", f"[{timestamp}] Waiting for {num_required_gpus} GPUs...")

    while True:
        with queue:
            if len(queue) < 1:
                print("[GRUN]", "Queue is manipulated. Please try again.")
                exit(1)

            first_pid = queue[0]["pid"]
            prioritized = first_pid == os.getpid()
            if not prioritized and not psutil.pid_exists(first_pid):
                queue.dequeue()
                print("[GRUN]", f"Invalid process id {first_pid} is removed from the queue.")

        if not prioritized:
            time.sleep(interval)
            continue

        available_gpus = get_nonutilized_gpus(n_gpus)
        if len(available_gpus) < num_required_gpus:
            time.sleep(interval)
            continue

        locked_gpus = acquire_n_gpus(available_gpus, num_required_gpus)
        if len(locked_gpus) < num_required_gpus:
            for _, lock in locked_gpus:
                lock.release()
            time.sleep(interval)
            continue

        selected_gpus = [gpu for gpu, _ in locked_gpus]
        print("[GRUN]", f"Acquired {num_required_gpus} GPUs: {selected_gpus}")

        with queue:
            queue.dequeue()

        return locked_gpus


def release_gpus(locked_gpus: Optional[List[Tuple[int, FileLock]]] = None) -> None:
    if locked_gpus:
        for _, lock in locked_gpus:
            lock.release()


def main():
    args = parser.parse_args()

    if args.freeze:
        tmp_dir = tempfile.mkdtemp(prefix="grun_")
        dst_dir = os.path.join(tmp_dir, os.path.basename(os.getcwd()))
        copytree(os.getcwd(), dst_dir)
        os.chdir(dst_dir)
        print("[GRUN]", f"Freeze current working directory to {tmp_dir}")
        atexit.register(shutil.rmtree, tmp_dir)
    else:
        tmp_dir = None

    num_gpus = nvidia_smi.nvmlDeviceGetCount()
    print("[GRUN]", f"Number of GPUs: {num_gpus}")

    available_gpus = get_nonutilized_gpus(num_gpus)
    print("[GRUN]", f"Non-Utilized GPUs: {available_gpus}")

    locked_gpus = acquire_n_gpus(available_gpus, args.n)
    selected_gpus = [gpu for gpu, _ in locked_gpus]

    if len(locked_gpus) < args.n:
        print("[GRUN]", f"{args.n} GPUs requested, but only {len(locked_gpus)} gpus {selected_gpus} available.")

        if not args.wait:
            release_gpus(locked_gpus)
            exit(1)

        with queue:
            queue.enqueue(" ".join(args.commands))

        print("[GRUN]", "Start Waiting for more GPUs...")
        locked_gpus = ensure_n_gpus(num_gpus, args.n)
        selected_gpus = [gpu for gpu, _ in locked_gpus]

    print("[GRUN]", f"Selected GPUs: {selected_gpus}")

    command = " ".join(args.commands)
    print("[GRUN]", "Run:", command)

    try:
        subprocess.run(
            command,
            shell=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, selected_gpus))),
        )
    except Exception as e:
        print("[GRUN]", "Error:", e, file=sys.stderr)
        traceback.print_exc()
        exit(1)
    finally:
        release_gpus(locked_gpus)

    print("[GRUN]", "Done.")


if __name__ == "__main__":
    main()
