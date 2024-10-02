import os

GRUN_DIR = os.path.join(os.path.expanduser("~"), ".grun")
GRUN_QUEUE = os.path.join(GRUN_DIR, "queue.json")
GRUN_QUEUE_LOCK = os.path.join(GRUN_DIR, "queue.json.lock")
