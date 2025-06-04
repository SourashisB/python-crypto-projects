class SegmentTree:
    """
    A Segment Tree implementation with lazy propagation for efficient range queries and updates.
    Supports range sum queries and range updates in O(log n) time.
    """
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        self._build(arr, 2 * node + 1, start, mid)
        self._build(arr, 2 * node + 2, mid + 1, end)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def _propagate(self, node, start, end):
        if self.lazy[node] != 0:
            self.tree[node] += (end - start + 1) * self.lazy[node]
            
            if start != end:  # Not a leaf node
                self.lazy[2 * node + 1] += self.lazy[node]
                self.lazy[2 * node + 2] += self.lazy[node]
            
            self.lazy[node] = 0
    
    def range_update(self, l, r, val):
        """Update values in range [l, r] by adding val to each element"""
        self._update(0, 0, self.n - 1, l, r, val)
    
    def _update(self, node, start, end, l, r, val):
        self._propagate(node, start, end)
        
        # No overlap
        if start > r or end < l:
            return
        
        # Complete overlap
        if start >= l and end <= r:
            self.tree[node] += (end - start + 1) * val
            if start != end:  # Not a leaf node
                self.lazy[2 * node + 1] += val
                self.lazy[2 * node + 2] += val
            return
        
        # Partial overlap - recurse to children
        mid = (start + end) // 2
        self._update(2 * node + 1, start, mid, l, r, val)
        self._update(2 * node + 2, mid + 1, end, l, r, val)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def range_sum(self, l, r):
        """Get sum of all elements in range [l, r]"""
        return self._query(0, 0, self.n - 1, l, r)
    
    def _query(self, node, start, end, l, r):
        # No overlap
        if start > r or end < l:
            return 0
        
        self._propagate(node, start, end)
        
        # Complete overlap
        if start >= l and end <= r:
            return self.tree[node]
        
        # Partial overlap
        mid = (start + end) // 2
        left_sum = self._query(2 * node + 1, start, mid, l, r)
        right_sum = self._query(2 * node + 2, mid + 1, end, l, r)
        return left_sum + right_sum


class SkipList:
    """
    A SkipList implementation providing O(log n) search, insert and delete operations.
    Uses probabilistic balancing to maintain multiple levels of linked lists.
    """
    class Node:
        def __init__(self, key, value, level):
            self.key = key
            self.value = value
            # Array of forward pointers
            self.forward = [None] * (level + 1)
        
        def __str__(self):
            return f"({self.key}: {self.value})"
    
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level  # Maximum level of the skip list
        self.p = p  # Probability factor for level generation
        self.level = 0  # Current level of skip list
        
        # Create header node with max level
        self.header = self.Node(-float("inf"), None, self.max_level)
    
    def _random_level(self):
        """Randomly determine the level of a new node"""
        level = 0
        while level < self.max_level and random.random() < self.p:
            level += 1
        return level
    
    def search(self, key):
        """Search for a key in the skip list"""
        current = self.header
        
        # Start from the highest level and work down
        for i in range(self.level, -1, -1):
            # Move forward on current level as far as possible
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        
        # At level 0, we're at our target position
        current = current.forward[0]
        
        # Check if we found the key
        if current and current.key == key:
            return current.value
        return None
    
    def insert(self, key, value):
        """Insert a key-value pair into the skip list"""
        # Array to store update positions at each level
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find positions to insert at each level
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # Update existing node if key already exists
        if current and current.key == key:
            current.value = value
            return
        
        # Generate random level for new node
        new_level = self._random_level()
        
        # Update the list's level if new node has higher level
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level
        
        # Create new node
        new_node = self.Node(key, value, new_level)
        
        # Insert node by updating pointers
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
    
    def delete(self, key):
        """Delete a key from the skip list"""
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find positions to update at each level
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        
        current = current.forward[0]
        
        # If key exists, remove it
        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]
            
            # Update the list's level if necessary
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1
            
            return True
        return False
    
    def __iter__(self):
        """Iterator to traverse the skip list"""
        current = self.header.forward[0]
        while current:
            yield (current.key, current.value)
            current = current.forward[0]


class PersistentRedBlackTree:
    """
    A Persistent Red-Black Tree that maintains history of all versions.
    Provides O(log n) operations with full history tracking.
    """
    RED = True
    BLACK = False
    
    class Node:
        def __init__(self, key, value, color=True):
            self.key = key
            self.value = value
            self.left = None
            self.right = None
            self.color = color  # True for RED, False for BLACK
        
        def copy(self):
            """Create a copy of the node"""
            new_node = PersistentRedBlackTree.Node(self.key, self.value, self.color)
            new_node.left = self.left
            new_node.right = self.right
            return new_node
    
    def __init__(self):
        self.nil = None  # Sentinel node
        self.root = None
        self.version_history = []
        self._save_version()
    
    def _save_version(self):
        """Save current state as a version"""
        self.version_history.append(self.root)
    
    def get_version(self, version):
        """Get tree at specific version"""
        if 0 <= version < len(self.version_history):
            return self.version_history[version]
        return None
    
    def _is_red(self, node):
        """Check if node is red"""
        if node is None:
            return False
        return node.color == self.RED
    
    def search(self, key, version=-1):
        """Search for a key in a specific version (default: latest)"""
        if version == -1:
            version = len(self.version_history) - 1
        
        node = self.get_version(version)
        while node is not None:
            if key == node.key:
                return node.value
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None
    
    def _left_rotate(self, x):
        """Perform left rotation (creates new nodes to maintain persistence)"""
        y = x.right
        new_x = x.copy()
        new_y = y.copy()
        
        # Update pointers
        new_x.right = y.left
        new_y.left = new_x
        
        # Return new root after rotation
        return new_y
    
    def _right_rotate(self, y):
        """Perform right rotation (creates new nodes to maintain persistence)"""
        x = y.left
        new_y = y.copy()
        new_x = x.copy()
        
        # Update pointers
        new_y.left = x.right
        new_x.right = new_y
        
        # Return new root after rotation
        return new_x
    
    def insert(self, key, value):
        """Insert a key-value pair into the tree"""
        # Create new root
        new_root = self._insert_recursive(self.root, key, value)
        
        # Root is always black
        new_root.color = self.BLACK
        
        # Update root and save version
        self.root = new_root
        self._save_version()
    
    def _insert_recursive(self, node, key, value):
        """Recursively insert and rebalance (returns new nodes)"""
        # Base case: Insert new node
        if node is None:
            return self.Node(key, value, self.RED)
        
        new_node = node.copy()
        
        # Normal BST insert
        if key < node.key:
            new_node.left = self._insert_recursive(node.left, key, value)
        elif key > node.key:
            new_node.right = self._insert_recursive(node.right, key, value)
        else:  # key == node.key
            new_node.value = value  # Replace value if key exists
            return new_node
        
        # Fix Red-Black tree properties
        if self._is_red(new_node.right) and not self._is_red(new_node.left):
            new_node = self._left_rotate(new_node)
        
        if self._is_red(new_node.left) and self._is_red(new_node.left.left):
            new_node = self._right_rotate(new_node)
        
        if self._is_red(new_node.left) and self._is_red(new_node.right):
            # Flip colors
            new_node.color = not new_node.color
            if new_node.left:
                new_left = new_node.left.copy()
                new_left.color = not new_node.left.color
                new_node.left = new_left
            
            if new_node.right:
                new_right = new_node.right.copy()
                new_right.color = not new_node.right.color
                new_node.right = new_right
        
        return new_node


# Application scenario: Stock Market Analysis System
import random
import time

def stock_market_analysis_scenario():
    """
    Scenario: A real-time stock market analysis system that tracks price changes,
    provides historical lookups, and identifies patterns for multiple stocks.
    """
    print("Stock Market Analysis System")
    print("----------------------------")
    
    # 1. Using SegmentTree for efficient price range queries
    print("\n1. Using Segment Tree for Stock Price Analysis")
    
    # Initial stock prices for a single stock over last 20 time periods
    stock_prices = [random.randint(950, 1050) for _ in range(20)]
    print(f"Initial stock prices: {stock_prices}")
    
    # Create segment tree for efficient range queries
    price_tree = SegmentTree(stock_prices)
    
    # Query average price in different time ranges
    first_week = price_tree.range_sum(0, 4) / 5
    second_week = price_tree.range_sum(5, 9) / 5
    third_week = price_tree.range_sum(10, 14) / 5
    fourth_week = price_tree.range_sum(15, 19) / 5
    
    print(f"Average price - Week 1: ${first_week:.2f}")
    print(f"Average price - Week 2: ${second_week:.2f}")
    print(f"Average price - Week 3: ${third_week:.2f}")
    print(f"Average price - Week 4: ${fourth_week:.2f}")
    
    # Update prices to simulate market volatility
    print("\nMarket correction affects prices in week 3-4...")
    price_tree.range_update(10, 19, -15)  # Price drop of $15
    
    # Query new averages after price update
    new_third_week = price_tree.range_sum(10, 14) / 5
    new_fourth_week = price_tree.range_sum(15, 19) / 5
    print(f"New average price - Week 3: ${new_third_week:.2f}")
    print(f"New average price - Week 4: ${new_fourth_week:.2f}")
    
    # 2. Using SkipList for efficiently tracking multiple stocks by symbol
    print("\n2. Using SkipList for Stock Portfolio Management")
    
    # Create a skip list to track stock portfolio
    portfolio = SkipList()
    
    # Add stocks to portfolio with quantity
    stocks = [
        ("AAPL", 50), ("MSFT", 75), ("GOOG", 25), ("AMZN", 30),
        ("META", 60), ("TSLA", 40), ("NVDA", 55), ("INTC", 80)
    ]
    
    for symbol, quantity in stocks:
        portfolio.insert(symbol, quantity)
    
    print("Current portfolio:")
    for symbol, quantity in portfolio:
        print(f"{symbol}: {quantity} shares")
    
    # Quickly lookup specific stocks
    print("\nQuickly looking up specific stocks:")
    symbols_to_check = ["AAPL", "TSLA", "NFLX"]
    for symbol in symbols_to_check:
        quantity = portfolio.search(symbol)
        if quantity:
            print(f"{symbol}: {quantity} shares")
        else:
            print(f"{symbol} not in portfolio")
    
    # Update portfolio
    print("\nUpdating portfolio...")
    portfolio.insert("AAPL", 70)  # Buying more AAPL
    portfolio.insert("NFLX", 45)  # Adding new stock
    portfolio.delete("INTC")      # Selling all INTC
    
    print("Updated portfolio:")
    for symbol, quantity in portfolio:
        print(f"{symbol}: {quantity} shares")
    
    # 3. Using PersistentRedBlackTree for historical price tracking
    print("\n3. Using Persistent Red-Black Tree for Historical Price Tracking")
    
    # Create persistent tree to track AAPL prices over time
    price_history = PersistentRedBlackTree()
    
    # Simulate recording prices at different times
    print("Recording AAPL prices throughout the day:")
    timestamps = [
        "09:30", "10:00", "10:30", "11:00", "11:30", 
        "12:00", "12:30", "13:00", "13:30", "14:00",
        "14:30", "15:00", "15:30", "16:00"
    ]
    
    # Store version mapping for easier reference
    version_map = {}
    
    base_price = 185.0
    for i, time_stamp in enumerate(timestamps):
        # Simulate price fluctuations
        change = random.uniform(-1.5, 1.5)
        # More volatility in the afternoon
        if i > 7:
            change *= 1.5
        
        price = base_price + change
        base_price = price  # Price carries over
        
        # Record price at this time
        price_history.insert(time_stamp, price)
        # Store the version number for this timestamp
        version_map[time_stamp] = i
        
        print(f"{time_stamp} - AAPL: ${price:.2f}")
    
    # Access historical prices from different versions using the latest version
    # This ensures we're looking at the correct version where the data exists
    print("\nLooking up historical prices:")
    latest_version = len(timestamps) - 1
    
    morning_time = "11:00"
    midday_time = "12:30"
    afternoon_time = "14:30"
    
    morning_price = price_history.search(morning_time, latest_version)
    midday_price = price_history.search(midday_time, latest_version)
    afternoon_price = price_history.search(afternoon_time, latest_version)
    
    print(f"Morning ({morning_time}): ${morning_price:.2f}")
    print(f"Midday ({midday_time}): ${midday_price:.2f}")
    print(f"Afternoon ({afternoon_time}): ${afternoon_price:.2f}")
    
    # Check price at a specific time across versions where it exists
    target_time = "12:00"
    print(f"\nTracking '{target_time}' price across versions:")
    
    # Calculate the version where the target time was first inserted
    target_version = version_map.get(target_time)
    
    if target_version is not None:
        # Look at this time from all versions after it was inserted
        for i in range(target_version, len(timestamps)):
            version_time = timestamps[i]
            price = price_history.search(target_time, i)
            if price:  # This check is redundant now but kept for safety
                print(f"As of {version_time}, {target_time} price was: ${price:.2f}")
    
    print("\nStock Market Analysis System simulation complete.")

# Run the scenario
if __name__ == "__main__":
    import random
    stock_market_analysis_scenario()