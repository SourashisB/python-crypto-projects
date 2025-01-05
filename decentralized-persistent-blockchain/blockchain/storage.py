import pickle
import os

class Storage:
    def __init__(self, filename="blockchain_data.pkl"):
        self.filename = filename

    def save(self, chain, nodes):
        with open(self.filename, "wb") as f:
            pickle.dump((chain, nodes), f)

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                return pickle.load(f)
        return None