import hashlib
import time
import json
import os
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, Toplevel


# Transaction Class
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


# Block Class
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


# Blockchain Class
class Blockchain:
    def __init__(self, difficulty=4, reward=50, file_name="blockchain.json"):
        self.chain = []
        self.pending_transactions = []
        self.difficulty = difficulty
        self.reward = reward
        self.file_name = file_name

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
            return None

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            previous_hash=last_block.hash(),
            transactions=self.pending_transactions,
        )

        while not new_block.hash().startswith("0" * self.difficulty):
            new_block.nonce += 1

        new_block.hash_value = new_block.hash()

        self.pending_transactions = [Transaction("System", miner_address, self.reward)]
        self.chain.append(new_block)
        self.save_blockchain()
        return new_block

    def is_valid_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash() != current_block.hash_value:
                return False

            if current_block.previous_hash != previous_block.hash_value:
                return False

        return True

    def save_blockchain(self):
        with open(self.file_name, "w") as file:
            data = {
                "chain": [block.to_dict() for block in self.chain],
                "pending_transactions": [tx.to_dict() for tx in self.pending_transactions],
            }
            json.dump(data, file, indent=4)

    def load_blockchain(self):
        with open(self.file_name, "r") as file:
            data = json.load(file)
            self.chain = [Block.from_dict(block) for block in data["chain"]]
            self.pending_transactions = [
                Transaction.from_dict(tx) for tx in data["pending_transactions"]
            ]


# GUI Wallet
class BlockchainWalletGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Blockchain Wallet")
        self.blockchain = Blockchain()

        # Sender Address
        Label(master, text="Sender Address:").grid(row=0, column=0, padx=10, pady=5)
        self.sender = StringVar()
        Entry(master, textvariable=self.sender).grid(row=0, column=1, padx=10, pady=5)

        # Recipient Address
        Label(master, text="Recipient Address:").grid(row=1, column=0, padx=10, pady=5)
        self.recipient = StringVar()
        Entry(master, textvariable=self.recipient).grid(row=1, column=1, padx=10, pady=5)

        # Amount
        Label(master, text="Amount:").grid(row=2, column=0, padx=10, pady=5)
        self.amount = StringVar()
        Entry(master, textvariable=self.amount).grid(row=2, column=1, padx=10, pady=5)

        # Buttons
        Button(master, text="Add Transaction", command=self.add_transaction).grid(
            row=3, column=0, padx=10, pady=5
        )
        Button(master, text="Mine Block", command=self.mine_block).grid(
            row=3, column=1, padx=10, pady=5
        )
        Button(master, text="Display Blockchain", command=self.display_chain).grid(
            row=4, column=0, padx=10, pady=5
        )
        Button(master, text="Validate Blockchain", command=self.validate_chain).grid(
            row=4, column=1, padx=10, pady=5
        )

    def add_transaction(self):
        sender = self.sender.get()
        recipient = self.recipient.get()
        try:
            amount = float(self.amount.get())
            self.blockchain.add_transaction(sender, recipient, amount)
            messagebox.showinfo("Success", "Transaction added!")
        except ValueError:
            messagebox.showerror("Error", "Invalid amount.")

    def mine_block(self):
        miner_address = self.sender.get()
        if not miner_address.strip():
            messagebox.showerror("Error", "Miner address is required.")
            return

        block = self.blockchain.mine_block(miner_address)
        if block:
            messagebox.showinfo("Success", f"Block mined successfully!\nHash: {block.hash_value}")
        else:
            messagebox.showwarning("Warning", "No transactions to mine.")

    def display_chain(self):
        new_window = Toplevel(self.master)
        new_window.title("Blockchain")
        text = ""
        for block in self.blockchain.chain:
            text += f"Block {block.index}:\n"
            text += f"Previous Hash: {block.previous_hash}\n"
            text += f"Transactions: {json.dumps([tx.to_dict() for tx in block.transactions], indent=4)}\n"
            text += f"Timestamp: {block.timestamp}\n"
            text += f"Nonce: {block.nonce}\n"
            text += f"Hash: {block.hash_value}\n"
            text += "-" * 50 + "\n"

        Label(new_window, text=text, justify="left", anchor="w").pack(padx=10, pady=10)

    def validate_chain(self):
        is_valid = self.blockchain.is_valid_chain()
        message = "Blockchain is valid!" if is_valid else "Blockchain is invalid!"
        messagebox.showinfo("Validation Result", message)


# Main Application
if __name__ == "__main__":
    root = Tk()
    app = BlockchainWalletGUI(root)
    root.mainloop()