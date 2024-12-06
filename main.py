import hashlib
import time
import json
import os


class Transaction:
    def __init__(self, sender, recipient, amount):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount

    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
        }

    @staticmethod
    def from_dict(data):
        return Transaction(data["sender"], data["recipient"], data["amount"])


class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.nonce = nonce
        self.hash_value = None

    def hash(self):
        block_string = json.dumps(
            {
                "index": self.index,
                "previous_hash": self.previous_hash,
                "transactions": [tx.to_dict() for tx in self.transactions],
                "timestamp": self.timestamp,
                "nonce": self.nonce,
            },
            sort_keys=True,
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "hash_value": self.hash_value,
        }

    @staticmethod
    def from_dict(data):
        transactions = [Transaction.from_dict(tx) for tx in data["transactions"]]
        block = Block(
            data["index"],
            data["previous_hash"],
            transactions,
            data["timestamp"],
            data["nonce"],
        )
        block.hash_value = data["hash_value"]
        return block


class Blockchain:
    def __init__(self, difficulty=4, reward=50, file_name="blockchain.json"):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = difficulty
        self.reward = reward
        self.file_name = file_name

        # Load blockchain from file if it exists
        if os.path.exists(self.file_name):
            self.load_blockchain()
        else:
            self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, "0", [])
        genesis_block.hash_value = genesis_block.hash()
        self.chain.append(genesis_block)
        self.save_blockchain()

    def get_last_block(self):
        return self.chain[-1]

    def add_transaction(self, sender, recipient, amount):
        transaction = Transaction(sender, recipient, amount)
        self.pending_transactions.append(transaction)

    def mine_block(self, miner_address):
        if not self.pending_transactions:
            print("No transactions to mine.")
            return None

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            previous_hash=last_block.hash(),
            transactions=self.pending_transactions,
        )

        print("Mining block...")
        while not new_block.hash().startswith("0" * self.difficulty):
            new_block.nonce += 1

        new_block.hash_value = new_block.hash()

        # Reward the miner
        self.pending_transactions = [
            Transaction("System", miner_address, self.reward)
        ]

        self.chain.append(new_block)
        print(f"Block mined! Hash: {new_block.hash_value}")

        self.save_blockchain()
        return new_block

    def is_valid_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check hash integrity
            if current_block.hash() != current_block.hash_value:
                print(f"Invalid hash at block {current_block.index}")
                return False

            # Check hash linkage
            if current_block.previous_hash != previous_block.hash_value:
                print(f"Invalid chain linkage at block {current_block.index}")
                return False

        return True

    def display_chain(self):
        for block in self.chain:
            print(f"Block {block.index}")
            print(f"Previous Hash: {block.previous_hash}")
            print(f"Transactions: {json.dumps([tx.to_dict() for tx in block.transactions], indent=4)}")
            print(f"Timestamp: {block.timestamp}")
            print(f"Nonce: {block.nonce}")
            print(f"Hash: {block.hash_value}")
            print("-" * 50)

    def save_blockchain(self):
        """
        Save the blockchain to a file in JSON format.
        """
        with open(self.file_name, "w") as file:
            data = {
                "chain": [block.to_dict() for block in self.chain],
                "pending_transactions": [tx.to_dict() for tx in self.pending_transactions],
            }
            json.dump(data, file, indent=4)

    def load_blockchain(self):
        """
        Load the blockchain from a file in JSON format.
        """
        with open(self.file_name, "r") as file:
            data = json.load(file)
            self.chain = [Block.from_dict(block) for block in data["chain"]]
            self.pending_transactions = [
                Transaction.from_dict(tx) for tx in data["pending_transactions"]
            ]


# CLI Interface
def cli():
    print("Welcome to CryptoCoin CLI!")
    blockchain = Blockchain()

    while True:
        print("\nOptions:")
        print("1. Add a transaction")
        print("2. Mine a block")
        print("3. Display blockchain")
        print("4. Validate blockchain")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            sender = input("Enter sender address: ")
            recipient = input("Enter recipient address: ")
            amount = float(input("Enter amount: "))
            blockchain.add_transaction(sender, recipient, amount)
            print("Transaction added!")

        elif choice == "2":
            miner_address = input("Enter miner address: ")
            blockchain.mine_block(miner_address)

        elif choice == "3":
            blockchain.display_chain()

        elif choice == "4":
            is_valid = blockchain.is_valid_chain()
            print("Blockchain is valid!" if is_valid else "Blockchain is invalid!")

        elif choice == "5":
            print("Exiting CLI...")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    cli()