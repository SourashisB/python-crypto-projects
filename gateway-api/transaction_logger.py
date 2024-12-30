import json
import threading

class TransactionLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock = threading.Lock()

    def log_transaction(self, transaction):
        with self.lock:
            with open(self.file_path, "a") as f:
                f.write(json.dumps(transaction) + "\n")