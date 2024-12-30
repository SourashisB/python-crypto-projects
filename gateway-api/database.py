import sqlite3
import threading

class Database:
    def __init__(self, db_file="crypto_exchange.db"):
        self.connection = sqlite3.connect(db_file, check_same_thread=False)
        self.lock = threading.Lock()
        self.create_tables()

    def create_tables(self):
        with self.lock:
            cursor = self.connection.cursor()

            # Create orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    account TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL
                )
            """)

            # Create transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    buyer TEXT NOT NULL,
                    seller TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)

            self.connection.commit()

    def add_order(self, order):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO orders (type, account, price, quantity)
                VALUES (?, ?, ?, ?)
            """, (order["type"], order["account"], order["price"], order["quantity"]))
            self.connection.commit()

    def fetch_orders(self):
        with self.lock:
            cursor = self.connection.cursor()

            # Fetch buy and sell orders
            cursor.execute("SELECT * FROM orders WHERE type = 'buy' ORDER BY price DESC")
            buy_orders = cursor.fetchall()

            cursor.execute("SELECT * FROM orders WHERE type = 'sell' ORDER BY price ASC")
            sell_orders = cursor.fetchall()

            return buy_orders, sell_orders

    def remove_order(self, order_id):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM orders WHERE id = ?", (order_id,))
            self.connection.commit()

    def log_transaction(self, transaction):
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO transactions (buyer, seller, price, quantity, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (transaction["buyer"], transaction["seller"], transaction["price"],
                  transaction["quantity"], transaction["timestamp"]))
            self.connection.commit()