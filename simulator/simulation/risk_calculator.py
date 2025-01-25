import numpy as np
from typing import List, Dict
from utils.decorators import timer_decorator
from core.asset import Asset

class RiskCalculator:
    def __init__(self, window_size: int = 252):
        self.window_size = window_size

    @timer_decorator
    def calculate_volatility(self, returns: np.ndarray) -> float:
        return np.std(returns) * np.sqrt(252)

    def calculate_correlation_matrix(self, assets: List[Asset]) -> np.ndarray:
        n_assets = len(assets)
        correlation_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                returns_i = np.diff(assets[i]._price_history) / assets[i]._price_history[:-1]
                returns_j = np.diff(assets[j]._price_history) / assets[j]._price_history[:-1]
                correlation_matrix[i, j] = np.corrcoef(returns_i, returns_j)[0, 1]
        
        return correlation_matrix

    @timer_decorator
    def calculate_portfolio_risk(
        self,
        weights: np.ndarray,
        assets: List[Asset]
    ) -> Dict[str, float]:
        correlation_matrix = self.calculate_correlation_matrix(assets)
        volatilities = np.array([
            self.calculate_volatility(
                np.diff(asset._price_history) / asset._price_history[:-1]
            )
            for asset in assets
        ])
        
        portfolio_volatility = np.sqrt(
            weights.T @ (correlation_matrix * np.outer(volatilities, volatilities)) @ weights
        )
        
        return {
            "portfolio_volatility": portfolio_volatility,
            "diversification_score": 1 - (portfolio_volatility / np.sum(weights * volatilities))
        }