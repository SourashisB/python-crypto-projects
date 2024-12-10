import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Hedge Fund System
class HedgeFundSystem:
    def __init__(self, data_file="hedge_fund_data.json"):
        self.data_file = data_file
        self.data = {
            "prices": {},  # {instrument: [price1, price2, ...]}
            "transactions": {},  # {instrument: [{type, quantity, price, date}, ...]}
            "pnl": {}  # {instrument: pnl_value}
        }
        self.load_data()

    ##### Core Methods #####

    def add_price(self, instrument, price):
        """Add a price for a specific instrument."""
        if instrument not in self.data["prices"]:
            self.data["prices"][instrument] = []
        self.data["prices"][instrument].append(price)
        print(f"Added price {price} for {instrument}.")

    def add_transaction(self, instrument, transaction_type, quantity, price):
        """Add a transaction for a specific instrument."""
        if instrument not in self.data["transactions"]:
            self.data["transactions"][instrument] = []
        transaction = {
            "type": transaction_type,
            "quantity": quantity,
            "price": price,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.data["transactions"][instrument].append(transaction)
        print(f"Added transaction: {transaction}")

    def calculate_pnl(self, instrument):
        """Calculate PnL for a specific instrument."""
        if instrument not in self.data["transactions"]:
            print(f"No transactions found for {instrument}.")
            return 0

        transactions = self.data["transactions"][instrument]
        pnl = 0

        for transaction in transactions:
            if transaction["type"] == "buy":
                pnl -= transaction["quantity"] * transaction["price"]
            elif transaction["type"] == "sell":
                pnl += transaction["quantity"] * transaction["price"]

        self.data["pnl"][instrument] = pnl
        print(f"Calculated PnL for {instrument}: {pnl}")
        return pnl

    ##### Data Persistence #####

    def save_data(self):
        """Save data to a JSON file."""
        with open(self.data_file, "w") as f:
            json.dump(self.data, f, indent=4)
        print("Data saved to file.")

    def load_data(self):
        """Load data from a JSON file."""
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                self.data = json.load(f)
            print("Data loaded from file.")
        else:
            print("No existing data file found. Starting fresh.")

    ##### Visualization #####

    def display_data(self):
        """Display the complete data structure visually."""
        print("\n=== Hedge Fund Data ===")
        for key, value in self.data.items():
            print(f"\n{key.capitalize()}:\n", pd.DataFrame(value))

    def visualize_prices(self):
        """Visualize prices for all instruments."""
        print("\nVisualizing prices...")
        for instrument, prices in self.data["prices"].items():
            plt.plot(prices, label=instrument)
        plt.title("Instrument Prices")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def visualize_transactions(self):
        """Visualize transactions as a table."""
        transactions = []
        for instrument, txns in self.data["transactions"].items():
            for txn in txns:
                transactions.append({"Instrument": instrument, **txn})

        if transactions:
            df = pd.DataFrame(transactions)
            print("\n=== Transactions ===\n")
            print(df)
        else:
            print("No transactions to display.")

    ##### Utility #####

    def run_demo(self):
        """Run a demonstration of the system."""
        print("Running Hedge Fund System Demo...")
        self.add_price("AAPL", 150)
        self.add_price("AAPL", 155)
        self.add_price("GOOGL", 2800)
        self.add_transaction("AAPL", "buy", 10, 150)
        self.add_transaction("AAPL", "sell", 5, 155)
        self.add_transaction("GOOGL", "buy", 2, 2800)
        self.calculate_pnl("AAPL")
        self.calculate_pnl("GOOGL")
        self.save_data()
        self.display_data()
        self.visualize_prices()
        self.visualize_transactions()

# Main Execution
if __name__ == "__main__":
    system = HedgeFundSystem()
    system.run_demo()