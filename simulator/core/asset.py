from typing import Union, Optional
from abc import ABC, abstractmethod
import numpy as np

class Asset(ABC):
    def __init__(self, symbol: str, initial_price: float):
        self._symbol = symbol
        self._price = initial_price
        self._price_history = [initial_price]

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def current_price(self) -> float:
        return self._price

    @current_price.setter
    def current_price(self, value: float):
        self._price = value
        self._price_history.append(value)

    @abstractmethod
    def calculate_return(self) -> float:
        pass

class Stock(Asset):
    def __init__(self, symbol: str, initial_price: float, dividend_yield: float = 0.0):
        super().__init__(symbol, initial_price)
        self.dividend_yield = dividend_yield

    def calculate_return(self) -> float:
        if len(self._price_history) < 2:
            return 0.0
        return (self._price_history[-1] / self._price_history[0] - 1) + self.dividend_yield

class Bond(Asset):
    def __init__(self, symbol: str, initial_price: float, coupon_rate: float, maturity: int):
        super().__init__(symbol, initial_price)
        self.coupon_rate = coupon_rate
        self.maturity = maturity

    def calculate_return(self) -> float:
        return self.coupon_rate + (self._price_history[-1] / self._price_history[0] - 1)

    def __add__(self, other: Union['Bond', float]) -> float:
        if isinstance(other, Bond):
            return self.current_price + other.current_price
        return self.current_price + other