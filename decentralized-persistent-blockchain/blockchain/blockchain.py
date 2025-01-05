import hashlib
import time
from .storage import Storage
from .logger import Logger

class Block:
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = []
        self.nodes = set()
        self.logger = Logger()
        self.storage = Storage()
        self.load_chain()

    def create_genesis_block(self):
        genesis_block = Block(0, time.time(), "Genesis Block", "0")
        self.chain.append(genesis_block)
        self.logger.log("Created genesis block.")

    def add_block(self, data, proof_of_work):
        previous_block = self.chain[-1]

        # Verify PoW
        if not self.is_valid_proof(data, proof_of_work, previous_block.hash):
            self.logger.log("Failed to add block: Invalid proof of work.")
            return False

        new_block = Block(len(self.chain), time.time(), data, previous_block.hash)
        new_block.nonce = proof_of_work
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)
        self.save_chain()
        self.logger.log(f"Block added with data: {data}")
        return True

    def add_node(self, node_address):
        self.nodes.add(node_address)
        self.save_chain()
        self.logger.log(f"Node added: {node_address}")

    def remove_node(self, node_address):
        if node_address in self.nodes:
            self.nodes.remove(node_address)
            self.save_chain()
            self.logger.log(f"Node removed: {node_address}")

    def save_chain(self):
        self.storage.save(self.chain, self.nodes)

    def load_chain(self):
        saved_data = self.storage.load()
        if saved_data:
            self.chain, self.nodes = saved_data
        else:
            self.create_genesis_block()

    def is_valid_proof(self, data, nonce, previous_hash, difficulty=4):
        """Verify if the proof of work is valid."""
        prefix = "0" * difficulty
        block_string = f"{data}{previous_hash}{nonce}"
        hash_attempt = hashlib.sha256(block_string.encode()).hexdigest()
        return hash_attempt.startswith(prefix)

    def is_valid_chain(self, chain):
        """Validate an entire blockchain."""
        for i in range(1, len(chain)):
            current_block = chain[i]
            previous_block = chain[i - 1]

            # Verify hash integrity
            if current_block.previous_hash != previous_block.hash:
                return False

            # Verify proof of work
            if not self.is_valid_proof(current_block.data, current_block.nonce, current_block.previous_hash):
                return False

        return True

    def replace_chain(self, new_chain):
        """Replace the chain if the new one is valid and longer."""
        if len(new_chain) > len(self.chain) and self.is_valid_chain(new_chain):
            self.chain = new_chain
            self.save_chain()
            self.logger.log("Chain replaced with a longer and valid chain.")
            return True
        return False