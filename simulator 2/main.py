import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import logging
from time import sleep
from threading import Thread
from sklearn.model_selection import ParameterGrid
from collections import deque
from multiprocessing import Pool
from typing import Dict, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# ---------------------- Simulated Market Data ----------------------
class MarketSimulator:
    """Simulates a stock market with bid-ask spread."""
    def __init__(self, initial_price: float = 100, volatility: float = 0.5, max_order_book_size: int = 100):
        self.price = initial_price
        self.volatility = volatility
        self.order_book = {"bids": deque(maxlen=max_order_book_size), "asks": deque(maxlen=max_order_book_size)}

    def update_price(self) -> None:
        """Random walk price movement."""
        self.price += np.random.normal(0, self.volatility)
        self.price = max(1, self.price)  # Ensure price stays positive

    def get_bid_ask(self) -> Tuple[float, float]:
        """Returns simulated bid-ask prices."""
        spread = np.random.uniform(0.01, 0.1)  # Simulated spread
        bid = self.price - spread / 2
        ask = self.price + spread / 2
        return bid, ask

    def generate_market_data(self, duration: float = 10, interval: float = 0.1, real_time: bool = True) -> None:
        """Generates real-time or simulated market data."""
        steps = int(duration / interval)
        for _ in range(steps):
            self.update_price()
            bid, ask = self.get_bid_ask()
            self.order_book["bids"].append(bid)
            self.order_book["asks"].append(ask)
            if real_time:
                sleep(interval)


# ---------------------- Market-Making Strategy ----------------------
class MarketMaker:
    """A market-making trading algorithm."""
    def __init__(self, simulator: MarketSimulator, spread: float = 0.05, position_limit: int = 10):
        self.simulator = simulator
        self.spread = spread
        self.position = 0
        self.cash = 10000
        self.trades: List[Tuple[str, float]] = []

    def place_orders(self) -> None:
        """Places bid and ask orders."""
        bid, ask = self.simulator.get_bid_ask()
        bid_price = bid - self.spread
        ask_price = ask + self.spread

        # Simulate execution with 50% chance
        if random.random() < 0.5:
            self.execute_trade(bid_price, "buy")
        if random.random() < 0.5:
            self.execute_trade(ask_price, "sell")

    def execute_trade(self, price: float, side: str) -> None:
        """Executes a trade."""
        if side == "buy" and self.position < 10:
            self.position += 1
            self.cash -= price
            self.trades.append(("buy", price))
            logging.info(f"Executed BUY at {price:.2f}")
        elif side == "sell" and self.position > -10:
            self.position -= 1
            self.cash += price
            self.trades.append(("sell", price))
            logging.info(f"Executed SELL at {price:.2f}")

    def run_strategy(self, duration: float = 10, interval: float = 0.1) -> None:
        """Runs the market-making strategy."""
        steps = int(duration / interval)
        for _ in range(steps):
            self.place_orders()
            sleep(interval)

    def performance(self) -> Dict[str, float]:
        """Evaluates strategy performance."""
        pnl = self.cash + self.position * self.simulator.price - 10000
        return {"PnL": pnl, "Final Position": self.position}


# ---------------------- Backtesting ----------------------
class Backtester:
    """Backtests the market-making strategy."""
    def __init__(self, simulator: MarketSimulator, strategy: MarketMaker):
        self.simulator = simulator
        self.strategy = strategy

    def run(self, duration: float = 10) -> Dict[str, float]:
        """Runs the backtest."""
        market_thread = Thread(target=self.simulator.generate_market_data, args=(duration,))
        strategy_thread = Thread(target=self.strategy.run_strategy, args=(duration,))

        market_thread.start()
        strategy_thread.start()

        market_thread.join()
        strategy_thread.join()

        return self.strategy.performance()


# ---------------------- Hyperparameter Optimization ----------------------
class StrategyOptimizer:
    """Optimizes hyperparameters using grid search."""
    def __init__(self, simulator_class, strategy_class, param_grid: Dict[str, List]):
        self.simulator_class = simulator_class
        self.strategy_class = strategy_class
        self.param_grid = list(ParameterGrid(param_grid))

    def optimize(self, duration: float = 10, parallel: bool = True) -> Tuple[Dict, float, List[Tuple[Dict, float]]]:
        """Finds the best hyperparameters."""
        if parallel:
            with Pool() as pool:
                results = pool.map(self._evaluate_params, [(params, duration) for params in self.param_grid])
        else:
            results = [self._evaluate_params((params, duration)) for params in self.param_grid]

        best_params, best_pnl = max(results, key=lambda x: x[1])
        return best_params, best_pnl, results

    def _evaluate_params(self, args: Tuple[Dict, float]) -> Tuple[Dict, float]:
        """Evaluates a single set of parameters."""
        params, duration = args
        simulator = self.simulator_class()
        strategy = self.strategy_class(simulator, **params)
        backtester = Backtester(simulator, strategy)
        result = backtester.run(duration)
        return params, result["PnL"]


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