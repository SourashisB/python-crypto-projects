from typing import Dict, List
import numpy as np
from core.asset import Asset
from simulation.risk_calculator import RiskCalculator
from utils.analytics import Analytics

class Portfolio:
    def __init__(self):
        self.holdings: Dict[str, float] = {}
        self.risk_calculator = RiskCalculator()

    def __getitem__(self, symbol: str) -> float:
        return self.holdings.get(symbol, 0.0)

    def __setitem__(self, symbol: str, quantity: float):
        self.holdings[symbol] = quantity

    def add_position(self, symbol: str, quantity: float):
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + quantity

    def remove_position(self, symbol: str, quantity: float):
        current_quantity = self.holdings.get(symbol, 0.0)
        if current_quantity < quantity:
            raise ValueError("Insufficient position")
        self.holdings[symbol] = current_quantity - quantity

    def calculate_value(self, market_prices: Dict[str, float]) -> float:
        return sum(
            quantity * market_prices[symbol]
            for symbol, quantity in self.holdings.items()
        )

    def get_weights(self) -> np.ndarray:
        total_value = sum(self.holdings.values())
        return np.array([
            quantity / total_value
            for quantity in self.holdings.values()
        ])

    def calculate_risk_metrics(self, assets: List[Asset]) -> Dict[str, float]:
        weights = self.get_weights()
        return self.risk_calculator.calculate_portfolio_risk(weights, assets)