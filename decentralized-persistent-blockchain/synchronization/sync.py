import requests
from blockchain.blockchain import Blockchain

class Synchronizer:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def fetch_chain_from_node(self, node):
        """Fetch the blockchain from a node."""
        try:
            response = requests.get(f"http://{node}/chain")
            if response.status_code == 200:
                return response.json().get("chain", [])
        except Exception as e:
            self.blockchain.logger.log(f"Error fetching chain from {node}: {e}")
        return None

    def synchronize(self):
        """Synchronize the blockchain with all nodes."""
        for node in self.blockchain.nodes:
            chain_data = self.fetch_chain_from_node(node)
            if chain_data:
                new_chain = self.deserialize_chain(chain_data)
                self.blockchain.replace_chain(new_chain)

    def deserialize_chain(self, chain_data):
        """Convert serialized chain data into Block objects."""
        new_chain = []
        for block_data in chain_data:
            block = Block(
                index=block_data["index"],
                timestamp=block_data["timestamp"],
                data=block_data["data"],
                previous_hash=block_data["hash"],
                nonce=block_data.get("nonce", 0)
            )
            new_chain.append(block)
        return new_chain