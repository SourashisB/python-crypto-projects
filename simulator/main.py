from core.market import Market
from core.asset import Stock, Bond
from core.portfolio import Portfolio
import numpy as np
import time

def main():
    # Initialize market
    market = Market()
    
    # Create assets
    assets = [
        Stock("AAPL", 150.0, 0.005),
        Stock("GOOGL", 2800.0, 0.0),
        Bond("US10Y", 100.0, 0.03, 10),
        Bond("US2Y", 100.0, 0.02, 2)
    ]
    
    # Add assets to market
    for asset in assets:
        market.add_asset(asset)
    
    # Create portfolio
    portfolio = Portfolio()
    portfolio.add_position("AAPL", 10)
    portfolio.add_position("GOOGL", 5)
    portfolio.add_position("US10Y", 20)
    portfolio.add_position("US2Y", 30)
    
    # Start market simulation
    market.start_trading()
    
    try:
        while True:
            # Calculate and display portfolio metrics
            market_prices = {
                asset.symbol: asset.current_price
                for asset in assets
            }
            
            portfolio_value = portfolio.calculate_value(market_prices)
            risk_metrics = portfolio.calculate_risk_metrics(assets)
            
            print("\nPortfolio Status:")
            print(f"Total Value: ${portfolio_value:,.2f}")
            print(f"Portfolio Volatility: {risk_metrics['portfolio_volatility']:.4f}")
            print(f"Diversification Score: {risk_metrics['diversification_score']:.4f}")
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        market.stop_trading()

if __name__ == "__main__":
    main()