from threading import Lock, Thread
from typing import Dict, List
import time
from core.asset import Asset
from simulation.price_generator import PriceGenerator

class Market:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.assets: Dict[str, Asset] = {}
            self.price_generator = PriceGenerator()
            self.is_running = False
            self.update_thread = None
            self.initialized = True

    def add_asset(self, asset: Asset):
        self.assets[asset.symbol] = asset

    def get_asset(self, symbol: str) -> Asset:
        return self.assets.get(symbol)

    def start_trading(self):
        if not self.is_running:
            self.is_running = True
            self.update_thread = Thread(target=self._update_prices)
            self.update_thread.daemon = True
            self.update_thread.start()

    def stop_trading(self):
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()

    def _update_prices(self):
        while self.is_running:
            for asset in self.assets.values():
                new_price = self.price_generator.generate_price(
                    asset.current_price,
                    volatility=0.01
                )
                asset.current_price = new_price
            time.sleep(1)