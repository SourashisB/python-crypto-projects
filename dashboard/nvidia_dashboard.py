import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load CSV data (Ensure you have a CSV file named 'NVIDIA_STOCK.csv')
csv_file = "NVIDIA_STOCK.csv"
df = pd.read_csv(csv_file, parse_dates=["Date"])

# Ensure the data is sorted
df = df.sort_values("Date")

# Calculate a 20-day moving average for trend visualization
df["20-day MA"] = df["Close"].rolling(window=20).mean()

# Calculate RSI (Relative Strength Index) for additional analysis
def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = calculate_rsi(df)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "NVIDIA Stock Dashboard"

# Define the layout
app.layout = html.Div(
    style={
        'backgroundColor': '#1e1e2f',
        'color': 'white',
        'fontFamily': 'Arial',
        'padding': '20px',
    },
    children=[
        html.H1("📊 NVIDIA Stock Dashboard", style={'textAlign': 'center', 'padding': '10px', 'color': '#FFA500'}),

        # Date Picker Section
        html.Div(
            [
                html.Label("Select Date Range:", style={'fontSize': '18px', 'color': 'white'}),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=df["Date"].min(),
                    end_date=df["Date"].max(),
                    display_format='YYYY-MM-DD',
                    style={'backgroundColor': '#2e2e3e', 'color': 'white', 'borderRadius': '5px'}
                ),
            ],
            style={'textAlign': 'center', 'marginBottom': '20px'},
        ),

        # Dropdown for Metric Selection
        html.Div(
            [
                html.Label("Select Metric for Line Chart:", style={'fontSize': '18px', 'color': 'white'}),
                dcc.Dropdown(
                    id='metric-dropdown',
                    options=[
                        {'label': 'Close', 'value': 'Close'},
                        {'label': 'Open', 'value': 'Open'},
                        {'label': 'High', 'value': 'High'},
                        {'label': 'Low', 'value': 'Low'}
                    ],
                    value='Close',
                    style={'width': '50%', 'margin': 'auto', 'backgroundColor': '#2e2e3e', 'color': 'black'}
                ),
            ],
            style={'textAlign': 'center', 'marginBottom': '20px'},
        ),

        # Toggle for MA
        html.Div(
            [
                dcc.Checklist(
                    id='ma-toggle',
                    options=[{'label': 'Show 20-Day Moving Average', 'value': 'show_ma'}],
                    value=['show_ma'],
                    style={'textAlign': 'center', 'marginBottom': '20px', 'color': 'white'}
                ),
            ],
            style={'textAlign': 'center'},
        ),

        # Summary Stats
        html.Div(
            id='summary-stats',
            style={
                'marginBottom': '20px',
                'padding': '10px',
                'border': '1px solid #FFA500',
                'borderRadius': '10px',
                'textAlign': 'center'
            },
        ),

        # Graphs
        dcc.Graph(id="price-chart"),
        dcc.Graph(id="candlestick-chart"),
        dcc.Graph(id="volume-chart"),
        dcc.Graph(id="rsi-chart"),  # RSI Chart
    ]
)

# Callback to update graphs and stats based on date range and selected metric
@app.callback(
    [
        Output("price-chart", "figure"),
        Output("candlestick-chart", "figure"),
        Output("volume-chart", "figure"),
        Output("rsi-chart", "figure"),
        Output("summary-stats", "children"),
    ],
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("metric-dropdown", "value"),
        Input("ma-toggle", "value"),
    ]
)
def update_charts(start_date, end_date, selected_metric, ma_toggle):
    # Filter data based on selected date range
    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Line Chart for Selected Metric with Optional Moving Average
    fig = px.line(
        filtered_df,
        x="Date",
        y=[selected_metric] + (["20-day MA"] if "show_ma" in ma_toggle else []),
        title=f"{selected_metric} Prices & 20-Day Moving Average",
        labels={"value": "Price", "Date": "Date"},
        template="plotly_dark"
    )
    fig.update_traces(mode="lines", line=dict(width=2))

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
        filtered_df,
        x="Date",
        y="Volume",
        title="Trading Volume",
        labels={"Volume": "Volume", "Date": "Date"},
        template="plotly_white",
        color_discrete_sequence=["#FF0000"]
    )

    # RSI Chart
    rsi_fig = px.line(
        filtered_df,
        x="Date",
        y="RSI",
        title="Relative Strength Index (RSI)",
        labels={"RSI": "RSI", "Date": "Date"},
        template="plotly_dark",
        color_discrete_sequence=["#FFA500"]
    )
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

    # Summary Stats
    highest_price = filtered_df["High"].max()
    lowest_price = filtered_df["Low"].min()
    avg_volume = filtered_df["Volume"].mean()
    summary = html.Div([
        html.P(f"📈 Highest Price: ${highest_price:.2f}"),
        html.P(f"📉 Lowest Price: ${lowest_price:.2f}"),
        html.P(f"📊 Average Volume: {avg_volume:,.0f} shares"),
    ])

    return fig, candlestick_fig, volume_fig, rsi_fig, summary


# Run the application
if __name__ == "__main__":
    app.run_server(debug=True)