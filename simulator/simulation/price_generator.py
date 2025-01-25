import numpy as np
from typing import Optional

class PriceGenerator:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

    def generate_price(self, current_price: float, volatility: float = 0.01) -> float:
        return current_price * np.exp(
            np.random.normal(0, volatility)
        )

    def generate_series(
        self,
        initial_price: float,
        n_points: int,
        volatility: float = 0.01
    ) -> np.ndarray:
        prices = [initial_price]
        for _ in range(n_points - 1):
            prices.append(
                self.generate_price(prices[-1], volatility)
            )
        return np.array(prices)