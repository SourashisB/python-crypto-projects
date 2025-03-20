import hashlib
import json
import time
import random
import argparse
import os
from datetime import datetime
from prettytable import PrettyTable
from typing import Dict, List, Any, Optional, Tuple

class Block:
    def __init__(self, index: int, timestamp: float, data: Dict[str, Any], 
                 previous_hash: str, nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }


class Blockchain:
    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.inventory: Dict[str, int] = {}
        self.pending_deliveries: Dict[str, Dict[str, Any]] = {}
        self.delivery_id_counter = 1
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        genesis_block = Block(0, time.time(), {
            "action": "genesis",
            "message": "Genesis Block of Warehouse Blockchain"
        }, "0")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        return self.chain[-1]
    
    def add_block(self, data: Dict[str, Any]) -> Block:
        previous_block = self.get_latest_block()
        new_index = previous_block.index + 1
        new_block = Block(new_index, time.time(), data, previous_block.hash)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        return new_block
    
    def is_chain_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if hash is correctly calculated
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check if the current block points to the previous block's hash
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def add_to_inventory(self, item_id: str, quantity: int, source: str = "manual") -> Block:
        if item_id in self.inventory:
            self.inventory[item_id] += quantity
        else:
            self.inventory[item_id] = quantity
        
        data = {
            "action": "add_inventory",
            "item_id": item_id,
            "quantity": quantity,
            "source": source,
            "current_stock": self.inventory[item_id]
        }
        
        return self.add_block(data)
    
    def remove_from_inventory(self, item_id: str, quantity: int, reason: str = "manual") -> Optional[Block]:
        if item_id not in self.inventory or self.inventory[item_id] < quantity:
            print(f"Error: Not enough stock for item {item_id}")
            return None
        
        self.inventory[item_id] -= quantity
        
        data = {
            "action": "remove_inventory",
            "item_id": item_id,
            "quantity": quantity,
            "reason": reason,
            "current_stock": self.inventory[item_id]
        }
        
        return self.add_block(data)
    
    def create_incoming_delivery(self, supplier: str, items: Dict[str, int]) -> Tuple[str, Block]:
        delivery_id = f"IN-{self.delivery_id_counter}"
        self.delivery_id_counter += 1
        
        self.pending_deliveries[delivery_id] = {
            "type": "incoming",
            "supplier": supplier,
            "items": items,
            "status": "pending"
        }
        
        data = {
            "action": "create_incoming_delivery",
            "delivery_id": delivery_id,
            "supplier": supplier,
            "items": items,
            "status": "pending"
        }
        
        return delivery_id, self.add_block(data)
    
    def create_outgoing_delivery(self, customer: str, items: Dict[str, int]) -> Tuple[Optional[str], Optional[Block]]:
        # Check if we have enough inventory for all items
        for item_id, quantity in items.items():
            if item_id not in self.inventory or self.inventory[item_id] < quantity:
                print(f"Error: Not enough stock for item {item_id} to create outgoing delivery")
                return None, None
        
        # Reserve the items by removing them from inventory
        for item_id, quantity in items.items():
            self.inventory[item_id] -= quantity
        
        delivery_id = f"OUT-{self.delivery_id_counter}"
        self.delivery_id_counter += 1
        
        self.pending_deliveries[delivery_id] = {
            "type": "outgoing",
            "customer": customer,
            "items": items,
            "status": "pending"
        }
        
        data = {
            "action": "create_outgoing_delivery",
            "delivery_id": delivery_id,
            "customer": customer,
            "items": items,
            "status": "pending"
        }
        
        return delivery_id, self.add_block(data)
    
    def complete_delivery(self, delivery_id: str, success: bool, notes: str = "") -> Optional[Block]:
        if delivery_id not in self.pending_deliveries:
            print(f"Error: Delivery {delivery_id} not found")
            return None
        
        delivery = self.pending_deliveries[delivery_id]
        
        if delivery["type"] == "incoming":
            if success:
                # Add items to inventory
                for item_id, quantity in delivery["items"].items():
                    if item_id in self.inventory:
                        self.inventory[item_id] += quantity
                    else:
                        self.inventory[item_id] = quantity
                
                status = "completed"
            else:
                status = "failed"
        else:  # outgoing
            if not success:
                # Return items to inventory
                for item_id, quantity in delivery["items"].items():
                    self.inventory[item_id] += quantity
                
                status = "failed"
            else:
                status = "completed"
        
        # Update delivery status
        self.pending_deliveries[delivery_id]["status"] = status
        
        data = {
            "action": "complete_delivery",
            "delivery_id": delivery_id,
            "type": delivery["type"],
            "success": success,
            "items": delivery["items"],
            "notes": notes,
            "status": status
        }
        
        if delivery["type"] == "incoming":
            data["supplier"] = delivery["supplier"]
        else:
            data["customer"] = delivery["customer"]
        
        return self.add_block(data)
    
    def get_inventory_snapshot(self) -> Dict[str, int]:
        return self.inventory.copy()
    
    def save_to_file(self, filename: str) -> None:
        data = {
            "chain": [block.to_dict() for block in self.chain],
            "inventory": self.inventory,
            "pending_deliveries": self.pending_deliveries,
            "delivery_id_counter": self.delivery_id_counter
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Blockchain saved to {filename}")
    
    def load_from_file(self, filename: str) -> bool:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Recreate the blockchain
            self.chain = []
            for block_data in data["chain"]:
                block = Block(
                    block_data["index"],
                    block_data["timestamp"],
                    block_data["data"],
                    block_data["previous_hash"],
                    block_data["nonce"]
                )
                block.hash = block_data["hash"]
                self.chain.append(block)
            
            self.inventory = data["inventory"]
            self.pending_deliveries = data["pending_deliveries"]
            self.delivery_id_counter = data["delivery_id_counter"]
            
            print(f"Blockchain loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading blockchain: {str(e)}")
            return False


def print_block_info(block: Block) -> None:
    print(f"Block #{block.index}")
    print(f"Timestamp: {datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {json.dumps(block.data, indent=2)}")
    print(f"Previous Hash: {block.previous_hash}")
    print(f"Hash: {block.hash}")
    print(f"Nonce: {block.nonce}")
    print("-" * 50)


def print_inventory(blockchain: Blockchain) -> None:
    inventory = blockchain.get_inventory_snapshot()
    
    if not inventory:
        print("Inventory is empty.")
        return
    
    table = PrettyTable()
    table.field_names = ["Item ID", "Quantity"]
    
    for item_id, quantity in inventory.items():
        table.add_row([item_id, quantity])
    
    print(table)


def print_pending_deliveries(blockchain: Blockchain) -> None:
    if not blockchain.pending_deliveries:
        print("No pending deliveries.")
        return
    
    table = PrettyTable()
    table.field_names = ["Delivery ID", "Type", "Items", "Status", "Customer/Supplier"]
    
    for delivery_id, delivery in blockchain.pending_deliveries.items():
        if delivery["status"] == "pending":
            items_str = ", ".join([f"{item_id}({qty})" for item_id, qty in delivery["items"].items()])
            
            if delivery["type"] == "incoming":
                entity = delivery["supplier"]
            else:
                entity = delivery["customer"]
            
            table.add_row([delivery_id, delivery["type"], items_str, delivery["status"], entity])
    
    print(table)


def print_blockchain_summary(blockchain: Blockchain) -> None:
    print(f"Blockchain length: {len(blockchain.chain)} blocks")
    print(f"Is blockchain valid: {blockchain.is_chain_valid()}")
    
    # Count different types of actions
    action_counts = {}
    for block in blockchain.chain:
        if "action" in block.data:
            action = block.data["action"]
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 1
    
    print("Action summary:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")


def generate_random_activity(blockchain: Blockchain, num_actions: int) -> None:
    item_ids = ["ITEM-001", "ITEM-002", "ITEM-003", "ITEM-004", "ITEM-005"]
    suppliers = ["Supplier A", "Supplier B", "Supplier C"]
    customers = ["Customer X", "Customer Y", "Customer Z"]
    
    for _ in range(num_actions):
        action = random.choice(["add_inventory", "create_incoming", "create_outgoing", "complete_delivery"])
        
        if action == "add_inventory":
            item_id = random.choice(item_ids)
            quantity = random.randint(1, 10)
            blockchain.add_to_inventory(item_id, quantity, "random_generation")
            print(f"Added {quantity} of {item_id} to inventory")
            
        elif action == "create_incoming":
            supplier = random.choice(suppliers)
            items = {}
            for _ in range(random.randint(1, 3)):
                item_id = random.choice(item_ids)
                if item_id not in items:
                    items[item_id] = random.randint(1, 5)
            
            delivery_id, _ = blockchain.create_incoming_delivery(supplier, items)
            print(f"Created incoming delivery {delivery_id} from {supplier} with items: {items}")
            
        elif action == "create_outgoing":
            customer = random.choice(customers)
            items = {}
            
            # Make sure we select items that exist in inventory
            available_items = [item for item, qty in blockchain.inventory.items() if qty > 0]
            
            if not available_items:
                print("Cannot create outgoing delivery - no items in inventory")
                continue
                
            for _ in range(random.randint(1, 2)):
                if not available_items:
                    break
                    
                item_id = random.choice(available_items)
                max_qty = min(blockchain.inventory[item_id], 3)  # Don't take more than 3 at once
                
                if max_qty > 0:
                    items[item_id] = random.randint(1, max_qty)
            
            if not items:
                print("Cannot create outgoing delivery - no suitable items in inventory")
                continue
                
            delivery_id, block = blockchain.create_outgoing_delivery(customer, items)
            if delivery_id:
                print(f"Created outgoing delivery {delivery_id} to {customer} with items: {items}")
            
        elif action == "complete_delivery":
            pending_deliveries = list(blockchain.pending_deliveries.keys())
            
            if not pending_deliveries:
                print("No pending deliveries to complete")
                continue
                
            delivery_id = random.choice(pending_deliveries)
            success = random.choice([True, False])
            notes = "Randomly generated completion"
            
            blockchain.complete_delivery(delivery_id, success, notes)
            status = "successfully" if success else "unsuccessfully"
            print(f"Completed delivery {delivery_id} {status}")
        
        # Sleep briefly to avoid identical timestamps
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description='Warehouse Blockchain Management System')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Initialize parser
    init_parser = subparsers.add_parser('init', help='Initialize a new blockchain')
    init_parser.add_argument('--difficulty', type=int, default=2, help='Mining difficulty')
    
    # Load parser
    load_parser = subparsers.add_parser('load', help='Load blockchain from file')
    load_parser.add_argument('filename', help='Filename to load from')
    
    # Save parser
    save_parser = subparsers.add_parser('save', help='Save blockchain to file')
    save_parser.add_argument('filename', help='Filename to save to')
    
    # Add inventory parser
    add_inv_parser = subparsers.add_parser('add_inventory', help='Add items to inventory')
    add_inv_parser.add_argument('item_id', help='Item ID')
    add_inv_parser.add_argument('quantity', type=int, help='Quantity to add')
    
    # Remove inventory parser
    remove_inv_parser = subparsers.add_parser('remove_inventory', help='Remove items from inventory')
    remove_inv_parser.add_argument('item_id', help='Item ID')
    remove_inv_parser.add_argument('quantity', type=int, help='Quantity to remove')
    remove_inv_parser.add_argument('--reason', default='manual', help='Reason for removal')
    
    # Create incoming delivery parser
    incoming_parser = subparsers.add_parser('create_incoming', help='Create incoming delivery')
    incoming_parser.add_argument('supplier', help='Supplier name')
    incoming_parser.add_argument('--items', required=True, help='Items in format item1:qty1,item2:qty2')
    
    # Create outgoing delivery parser
    outgoing_parser = subparsers.add_parser('create_outgoing', help='Create outgoing delivery')
    outgoing_parser.add_argument('customer', help='Customer name')
    outgoing_parser.add_argument('--items', required=True, help='Items in format item1:qty1,item2:qty2')
    
    # Complete delivery parser
    complete_parser = subparsers.add_parser('complete_delivery', help='Complete a delivery')
    complete_parser.add_argument('delivery_id', help='Delivery ID')
    complete_parser.add_argument('--success', type=bool, default=True, help='Whether delivery was successful')
    complete_parser.add_argument('--notes', default='', help='Notes about the delivery')
    
    # Show inventory parser
    subparsers.add_parser('show_inventory', help='Show current inventory')
    
    # Show pending deliveries parser
    subparsers.add_parser('show_pending', help='Show pending deliveries')
    
    # Show blockchain parser
    show_parser = subparsers.add_parser('show_chain', help='Show blockchain')
    show_parser.add_argument('--count', type=int, default=5, help='Number of recent blocks to show')
    
    # Summary parser
    subparsers.add_parser('summary', help='Show blockchain summary')
    
    # Random activity parser
    random_parser = subparsers.add_parser('random_activity', help='Generate random activity')
    random_parser.add_argument('--count', type=int, default=10, help='Number of random actions')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create or load blockchain
    blockchain_file = "blockchain.dat"
    blockchain = None
    
    if args.command == 'init':
        blockchain = Blockchain(difficulty=args.difficulty)
        print(f"Initialized new blockchain with difficulty {args.difficulty}")
    elif args.command == 'load':
        blockchain = Blockchain()
        if not blockchain.load_from_file(args.filename):
            return
    else:
        # Try to load from default file if it exists
        if os.path.exists(blockchain_file):
            blockchain = Blockchain()
            blockchain.load_from_file(blockchain_file)
        else:
            blockchain = Blockchain()
            print("Initialized new blockchain with default settings")
    
    # Process commands
    if args.command == 'save':
        blockchain.save_to_file(args.filename)
    elif args.command == 'add_inventory':
        block = blockchain.add_to_inventory(args.item_id, args.quantity)
        print(f"Added {args.quantity} of {args.item_id} to inventory")
        print_block_info(block)
    elif args.command == 'remove_inventory':
        block = blockchain.remove_from_inventory(args.item_id, args.quantity, args.reason)
        if block:
            print(f"Removed {args.quantity} of {args.item_id} from inventory")
            print_block_info(block)
    elif args.command == 'create_incoming':
        items = {}
        for item_spec in args.items.split(','):
            item_id, qty = item_spec.split(':')
            items[item_id] = int(qty)
        
        delivery_id, block = blockchain.create_incoming_delivery(args.supplier, items)
        print(f"Created incoming delivery with ID: {delivery_id}")
        print_block_info(block)
    elif args.command == 'create_outgoing':
        items = {}
        for item_spec in args.items.split(','):
            item_id, qty = item_spec.split(':')
            items[item_id] = int(qty)
        
        result = blockchain.create_outgoing_delivery(args.customer, items)
        if result[0]:
            delivery_id, block = result
            print(f"Created outgoing delivery with ID: {delivery_id}")
            print_block_info(block)
    elif args.command == 'complete_delivery':
        block = blockchain.complete_delivery(args.delivery_id, args.success, args.notes)
        if block:
            status = "Successfully" if args.success else "Unsuccessfully"
            print(f"{status} completed delivery {args.delivery_id}")
            print_block_info(block)
    elif args.command == 'show_inventory':
        print_inventory(blockchain)
    elif args.command == 'show_pending':
        print_pending_deliveries(blockchain)
    elif args.command == 'show_chain':
        count = min(args.count, len(blockchain.chain))
        for i in range(len(blockchain.chain) - count, len(blockchain.chain)):
            print_block_info(blockchain.chain[i])
    elif args.command == 'summary':
        print_blockchain_summary(blockchain)
    elif args.command == 'random_activity':
        generate_random_activity(blockchain, args.count)
    
    # Save blockchain state automatically unless we're explicitly saving
    if args.command != 'save':
        blockchain.save_to_file(blockchain_file)


if __name__ == "__main__":
    main()