import numpy as np
from typing import List, Union
from utils.decorators import validate_numeric

class Analytics:
    @staticmethod
    @validate_numeric
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    @staticmethod
    @validate_numeric
    def calculate_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance

    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        return np.percentile(returns, (1 - confidence_level) * 100)

    @classmethod
    def calculate_metrics(cls, returns: np.ndarray, market_returns: np.ndarray) -> dict:
        return {
            "sharpe_ratio": cls.calculate_sharpe_ratio(returns),
            "beta": cls.calculate_beta(returns, market_returns),
            "var": cls.calculate_var(returns)
        }