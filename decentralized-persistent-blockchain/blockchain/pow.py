import hashlib

class ProofOfWork:
    def __init__(self, difficulty=4):
        self.difficulty = difficulty

    def mine(self, data, previous_hash):
        """Mine a valid nonce for the block."""
        prefix = "0" * self.difficulty
        nonce = 0

        while True:
            block_string = f"{data}{previous_hash}{nonce}"
            hash_attempt = hashlib.sha256(block_string.encode()).hexdigest()

            if hash_attempt.startswith(prefix):
                return nonce

            nonce += 1