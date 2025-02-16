import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import threading
from sklearn.model_selection import ParameterGrid
from collections import deque

# ---------------------- Simulated Market Data ----------------------
class MarketSimulator:
    """Simulates a stock market with bid-ask spread."""
    def __init__(self, initial_price=100, volatility=0.5):
        self.price = initial_price
        self.volatility = volatility
        self.order_book = {"bids": deque(maxlen=100), "asks": deque(maxlen=100)}

    def update_price(self):
        """Random walk price movement."""
        self.price += np.random.normal(0, self.volatility)
        self.price = max(1, self.price)  # Ensure price stays positive

    def get_bid_ask(self):
        """Returns simulated bid-ask prices."""
        spread = np.random.uniform(0.01, 0.1)  # Simulated spread
        bid = self.price - spread / 2
        ask = self.price + spread / 2
        return bid, ask

    def generate_market_data(self, duration=10, interval=0.1):
        """Generates real-time market data."""
        for _ in range(int(duration / interval)):
            self.update_price()
            bid, ask = self.get_bid_ask()
            self.order_book["bids"].append(bid)
            self.order_book["asks"].append(ask)
            time.sleep(interval)

# ---------------------- Market-Making Strategy ----------------------
class MarketMaker:
    """A market-making trading algorithm."""
    def __init__(self, simulator, spread=0.05, position_limit=10):
        self.simulator = simulator
        self.spread = spread
        self.position = 0
        self.cash = 10000
        self.trades = []

    def place_orders(self):
        """Places bid and ask orders."""
        bid, ask = self.simulator.get_bid_ask()
        bid_price = bid - self.spread
        ask_price = ask + self.spread

        # Simulate execution
        if random.random() < 0.5:  # 50% chance of execution
            self.execute_trade(bid_price, "buy")
        if random.random() < 0.5:
            self.execute_trade(ask_price, "sell")

    def execute_trade(self, price, side):
        """Executes a trade."""
        if side == "buy" and self.position < 10:
            self.position += 1
            self.cash -= price
            self.trades.append(("buy", price))
        elif side == "sell" and self.position > -10:
            self.position -= 1
            self.cash += price
            self.trades.append(("sell", price))

    def run_strategy(self, duration=10, interval=0.1):
        """Runs the market-making strategy."""
        for _ in range(int(duration / interval)):
            self.place_orders()
            time.sleep(interval)

    def performance(self):
        """Evaluates strategy performance."""
        pnl = self.cash + self.position * self.simulator.price - 10000
        return {"PnL": pnl, "Final Position": self.position}

# ---------------------- Backtesting ----------------------
class Backtester:
    """Backtests the market-making strategy."""
    def __init__(self, simulator, strategy):
        self.simulator = simulator
        self.strategy = strategy

    def run(self, duration=10):
        """Runs the backtest."""
        market_thread = threading.Thread(target=self.simulator.generate_market_data, args=(duration,))
        strategy_thread = threading.Thread(target=self.strategy.run_strategy, args=(duration,))
        
        market_thread.start()
        strategy_thread.start()
        
        market_thread.join()
        strategy_thread.join()

        return self.strategy.performance()

# ---------------------- Hyperparameter Optimization ----------------------
class StrategyOptimizer:
    """Optimizes hyperparameters using grid search."""
    def __init__(self, simulator_class, strategy_class, param_grid):
        self.simulator_class = simulator_class
        self.strategy_class = strategy_class
        self.param_grid = list(ParameterGrid(param_grid))

    def optimize(self, duration=10):
        """Finds the best hyperparameters."""
        best_pnl = -float("inf")
        best_params = None
        results = []

        for params in self.param_grid:
            simulator = self.simulator_class()
            strategy = self.strategy_class(simulator, **params)
            backtester = Backtester(simulator, strategy)
            result = backtester.run(duration)

            if result["PnL"] > best_pnl:
                best_pnl = result["PnL"]
                best_params = params

            results.append((params, result["PnL"]))

        return best_params, best_pnl, results

# ---------------------- Running the Algorithm ----------------------
if __name__ == "__main__":
    print("Running Backtest...")

    # Simulation Settings
    market_sim = MarketSimulator()
    strategy = MarketMaker(market_sim, spread=0.05)
    backtester = Backtester(market_sim, strategy)
    
    # Run backtest
    results = backtester.run(duration=10)
    print("Backtest Performance:", results)

    # Hyperparameter Optimization
    param_grid = {
        "spread": [0.01, 0.05, 0.1],
        "position_limit": [5, 10, 15]
    }

    optimizer = StrategyOptimizer(MarketSimulator, MarketMaker, param_grid)
    best_params, best_pnl, all_results = optimizer.optimize(duration=10)

    print("Best Hyperparameters:", best_params)
    print("Best PnL:", best_pnl)

    # PnL Distribution
    plt.hist([r[1] for r in all_results], bins=10, alpha=0.7)
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.title("PnL Distribution of Parameter Grid")
    plt.show()