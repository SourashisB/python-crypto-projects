import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load CSV data (Ensure you have a CSV file named 'stock_data.csv')
csv_file = "NVIDIA_STOCK.csv"
df = pd.read_csv(csv_file, parse_dates=["Date"])

# Ensure the data is sorted
df = df.sort_values("Date")

# Calculate a 20-day moving average for trend visualization
df["20-day MA"] = df["Close"].rolling(window=20).mean()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Stock Market Dashboard"

# Layout of the dashboard
app.layout = html.Div(style={'backgroundColor': '#1e1e2f', 'color': 'white', 'fontFamily': 'Arial'}, children=[
    html.H1("📈 Stock Market Dashboard", style={'textAlign': 'center', 'padding': '10px'}),

    # Date Picker
    html.Div([
        html.Label("Select Date Range:", style={'fontSize': '18px'}),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=df["Date"].min(),
            end_date=df["Date"].max(),
            display_format='YYYY-MM-DD',
            style={'backgroundColor': '#2e2e3e', 'color': 'white', 'borderRadius': '5px'}
        ),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Stock Price Line Chart with Moving Average
    dcc.Graph(id="price-chart"),

    # Candlestick Chart for Stock Trends
    dcc.Graph(id="candlestick-chart"),

    # Volume Bar Chart
    dcc.Graph(id="volume-chart"),
])

# Callback to update graphs based on date range
@app.callback(
    [Output("price-chart", "figure"),
     Output("candlestick-chart", "figure"),
     Output("volume-chart", "figure")],
    [Input("date-picker", "start_date"),
     Input("date-picker", "end_date")]
)
def update_charts(start_date, end_date):
    # Filter data based on selected date range
    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Line Chart for Stock Prices with Moving Average
    price_fig = px.line(
        filtered_df, x="Date", y=["Close", "20-day MA"],
        title="Stock Prices & 20-Day Moving Average",
        labels={"value": "Price", "Date": "Date"},
        template="plotly_dark"
    )
    price_fig.update_traces(mode="lines", line=dict(width=2))

    # Candlestick Chart for Stock Trends
    candlestick_fig = go.Figure(data=[go.Candlestick(
        x=filtered_df["Date"],
        open=filtered_df["Open"],
        high=filtered_df["High"],
        low=filtered_df["Low"],
        close=filtered_df["Close"],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    candlestick_fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )

    # Volume Bar Chart
    volume_fig = px.bar(
        filtered_df, x="Date", y="Volume",
        title="Trading Volume",
        labels={"Volume": "Volume", "Date": "Date"},
        template="plotly_white",
        color_discrete_sequence=["#FF0000"]
    )

    return price_fig, candlestick_fig, volume_fig

# Run the application
if __name__ == "__main__":
    app.run_server(debug=True)