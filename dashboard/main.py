# stock_dashboard.py
import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.express as px

# Configuration
STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
HISTORICAL_PERIOD = '1y'
MOVING_AVERAGES = [50, 200]

def get_stock_data(symbol):
    """Fetch stock data including historical prices and moving averages."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=HISTORICAL_PERIOD)
        
        if hist.empty:
            return None

        # Calculate metrics
        current_price = hist['Close'][-1]
        previous_close = hist['Close'][-2] if len(hist) > 1 else current_price
        change_percent = ((current_price - previous_close) / previous_close) * 100
        volume = hist['Volume'][-1]

        # Calculate moving averages
        for ma in MOVING_AVERAGES:
            hist[f'MA_{ma}'] = hist['Close'].rolling(window=ma).mean()

        return {
            'symbol': symbol,
            'current_price': current_price,
            'previous_close': previous_close,
            'change_percent': change_percent,
            'volume': volume,
            'historical': hist,
            'info': stock.info
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def create_dashboard(stocks_data):
    """Create Streamlit dashboard with stock information and charts."""
    st.set_page_config(layout="wide")
    st.title("📈 Top 5 Stocks Dashboard")
    
    # Display metrics in columns
    cols = st.columns(5)
    for i, stock in enumerate(stocks_data):
        cols[i].subheader(stock['symbol'])
        cols[i].metric(
            label="Current Price",
            value=f"${stock['current_price']:.2f}",
            delta=f"{stock['change_percent']:.2f}%"
        )
        cols[i].write(f"**Previous Close:** ${stock['previous_close']:.2f}")
        cols[i].write(f"**Volume:** {stock['volume']:,.0f}")

    # Create tabs for detailed analysis
    tab1, tab2, tab3 = st.tabs(["Price Trends", "Moving Averages", "Stock Comparisons"])

    with tab1:
        st.header("Historical Price Trends")
        for stock in stocks_data:
            fig = px.line(
                stock['historical'],
                y='Close',
                title=f"{stock['symbol']} Price Trend ({HISTORICAL_PERIOD})"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Moving Average Analysis")
        for stock in stocks_data:
            fig = px.line(
                stock['historical'],
                y=['Close'] + [f'MA_{ma}' for ma in MOVING_AVERAGES],
                title=f"{stock['symbol']} Moving Averages"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Stock Comparison")
        comparison_df = pd.DataFrame({
            'Symbol': [s['symbol'] for s in stocks_data],
            'Current Price': [s['current_price'] for s in stocks_data],
            'YTD Change (%)': [s['info'].get('ytdReturn', 'N/A') for s in stocks_data],
            'Market Cap': [s['info'].get('marketCap', 'N/A') for s in stocks_data],
            'PE Ratio': [s['info'].get('trailingPE', 'N/A') for s in stocks_data]
        })
        st.dataframe(comparison_df, use_container_width=True)

        # Correlation heatmap
        st.subheader("Price Correlation Matrix")
        close_prices = pd.concat([s['historical']['Close'] for s in stocks_data], axis=1)
        close_prices.columns = [s['symbol'] for s in stocks_data]
        corr_matrix = close_prices.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # Fetch data for all stocks
    stocks_data = [data for symbol in STOCK_SYMBOLS 
                   if (data := get_stock_data(symbol)) is not None]
    
    if not stocks_data:
        st.error("Failed to fetch data for all stocks. Please try again later.")
    else:
        create_dashboard(stocks_data)