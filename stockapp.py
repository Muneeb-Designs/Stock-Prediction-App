import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import joblib
import os
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import streamlit as st
import base64
import requests
import time
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ✅ Set page config FIRST
st.set_page_config(page_title="Stock Price Forecast", layout="wide")

# ✅ Convert GIF to base64
def get_base64_gif(gif_path):
    with open(gif_path, "rb") as gif_file:
        encoded = base64.b64encode(gif_file.read()).decode()
    return encoded

# ✅ Load the GIF
gif_base64 = get_base64_gif("Flow.gif")  # Replace with your gif file name


# ✅ Display as a short banner
st.markdown(
    f"""

    <style>
    .gif-banner {{
        width: 100%;
        height: 120px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 25px;
    }}
    </style>
    <img src="data:image/gif;base64,{gif_base64}" class="gif-banner">
    """,
    unsafe_allow_html=True
)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .reportview-container .markdown-text-container {
            padding-top: 2rem;
        }
        footer {
            visibility: hidden;
        }
        .custom-footer {
            text-align: center;
            font-size: 0.9rem;
            padding-top: 1rem;
            color: #666;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <style>
        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #4B8BBE;
        }
        .sidebar-subtitle {
            font-size: 18px;
            color: #FFFFFF;
        }
        .sidebar-text {
            font-size: 16px;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )



# ✅ Convert logo to base64
def get_base64_logo(logo_path):
    with open(logo_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# ✅ Load and display logo
logo_base64 = get_base64_logo("logo.png")

with st.sidebar:
    st.markdown(
        f"""
        <style>
        .sidebar-logo {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80%;
            max-height: 120px;
            object-fit: contain;
        }}
        .brand-text {{
            text-align: center;
            font-size: 16px;
            font-weight: 600;
            color: #FFFFFF;
            margin-top: -10px;
            margin-bottom: 10px;
        }}
        </style>

        <img src="data:image/png;base64,{logo_base64}" class="sidebar-logo">
        <div class="brand-text">Powered by LSTM & GRU Models</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")


    model_choice = st.selectbox("🔍 Select Model", ["LSTM", "GRU"])
    stocks = ['GOOGL', 'AAPL', 'NVDA', 'META', 'AMZN', 'MSFT', 'TSLA']
    selected_stock = st.selectbox("💼 Select Stock", stocks)

    st.markdown("---")
    st.markdown('<div class="sidebar-subtitle">📢 Disclaimer</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-text">
        • For educational and demo purposes only.<br>
        • Not financial advice.<br>
        • Consult a financial expert before investing.
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown('<div class="sidebar-text">📬 Contact: <a href="mailto:muneeburrehman302302@proton.me">muneeburrehman302302@proton.me</a></div>', unsafe_allow_html=True)


start = '2020-01-01'
end = '2025-04-13'

data = yf.download(selected_stock, start=start, end=end)

import streamlit as st
import requests
import time

# --- Layout in Two Columns ---
left_col, right_col = st.columns([2, 1])  # Wider left column for stock table

with left_col:
    st.subheader(f"📌 Stock Data for {selected_stock}")
    st.dataframe(data.tail(10), use_container_width=True)


# --- Fetch Trending Cryptocurrency Data from CoinGecko API ---
def get_trending_cryptos():
    url = "https://api.coingecko.com/api/v3/search/trending"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("coins", [])
        else:
            st.error(f"Error fetching trending crypto: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching crypto data: {e}")
        return []


# --- Crypto Section in the Right Column ---
with right_col:
    st.subheader("📰 Latest Crypto Updates (via CoinGecko)")

    # Fetch Trending Cryptos
    cryptos = get_trending_cryptos()

    # Limit to show only 1 or 2 cryptocurrencies
    if cryptos:
        for i, crypto in enumerate(cryptos[:2]):  # Show only 1 or 2 coins
            with st.container():
                # Custom styling for smaller size and animation
                st.markdown(
                    f"<h3 style='font-size: 18px; color: #4CAF50;'>{crypto['item']['name']}</h3>", unsafe_allow_html=True)
                st.write(f"**Symbol**: {crypto['item']['symbol']}")
                st.write(f"**Price**: {crypto['item']['price_btc']} BTC")
                st.markdown(f"[🔗 View on CoinGecko](https://www.coingecko.com/en/coins/{crypto['item']['id']})")
                st.markdown("---")
                time.sleep(0.1)  # Optional: animation effect
    else:
        st.info("No trending crypto found.")

# Prepare data
train_size = int(len(data) * 0.80)
data_train = pd.DataFrame(data['Close'][:train_size])
data_test = pd.DataFrame(data['Close'][train_size:])

model_path = f'models/{selected_stock}_{model_choice}_Model.keras' if model_choice == "GRU" else f'models/{selected_stock}_{model_choice}_Model.h5'

scaler_path = f'models/{selected_stock}_{model_choice}_scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f) if model_choice == "LSTM" else joblib.load(scaler_path)
else:
    st.error("⚠️ Model or scaler not found!")
    st.stop()

# Scale & prepare test data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test)

x = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
x = np.array(x)

# Predictions
predictions = model.predict(x)
predictions = scaler.inverse_transform(predictions)

# Moving Average Charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 Price vs MA50")
    ma_50_days = data['Close'].rolling(50).mean()
    fig1 = plt.figure(figsize=(8, 4))
    plt.plot(ma_50_days, 'r', label='MA50')
    plt.plot(data['Close'], 'g', label='Stock Price')
    plt.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("📉 Price vs MA50 vs MA100")
    ma_100_days = data['Close'].rolling(100).mean()
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(ma_50_days, 'r', label='MA50')
    plt.plot(ma_100_days, 'b', label='MA100')
    plt.plot(data['Close'], 'g', label='Stock Price')
    plt.legend()
    st.pyplot(fig2)

st.subheader("📉 Price vs MA100 vs MA200")
ma_200_days = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(10, 4))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data['Close'], 'g', label='Stock Price')
plt.legend()
st.pyplot(fig3)

# 📉 Price vs MA20 vs MA300
st.subheader("📉 Price vs MA20 vs MA300")
ma_20_days = data['Close'].rolling(20).mean()
ma_300_days = data['Close'].rolling(300).mean()

fig = plt.figure(figsize=(10, 4))
plt.plot(ma_20_days, 'red', label='MA20')
plt.plot(ma_300_days, 'blue', label='MA300')
plt.plot(data['Close'], 'green', label='Stock Price')
plt.legend()
st.pyplot(fig)


# Function to calculate Rate of Change (ROC)
def calculate_roc(data, period):
    return (data.pct_change(periods=period) * 100)

# Function to calculate Weighted Moving Average (WMA)
def calculate_wma(data, period):
    weights = np.arange(1, period + 1)
    return data.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Function to calculate Coppock Curve (10, 14, 11)
def calculate_coppock_curve(data):
    # Calculate ROC(10) and ROC(14)
    roc_10 = calculate_roc(data, 10)
    roc_14 = calculate_roc(data, 14)
    
    # Combine ROC(10) and ROC(14)
    combined_roc = roc_10 + roc_14
    
    # Calculate WMA(11) of combined ROC
    coppock_curve = calculate_wma(combined_roc, 11)
    
    return coppock_curve

# Calculate Coppock Curve for the 'Close' price column
data['Coppock Curve'] = calculate_coppock_curve(data['Close'])

# Plot the Coppock Curve
st.subheader(f"📉 Coppock Curve for {selected_stock} (Line Graph)")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the Coppock Curve line
ax.plot(data.index, data['Coppock Curve'], label='Coppock Curve (10, 14, 11)', color='blue', lw=2)

# Add a horizontal line at 0 to indicate the baseline
ax.axhline(0, color='red', linestyle='--', label='Zero Line')

# Highlight Buy signals (when Coppock Curve crosses above 0)
buy_signals = data[data['Coppock Curve'] > 0]
ax.plot(buy_signals.index, buy_signals['Coppock Curve'], 'go', label='Buy Signal', markersize=5)

# Add labels, title, and legend
ax.set_title(f'Coppock Curve for {selected_stock}')
ax.set_xlabel('Date')
ax.set_ylabel('Coppock Curve Value')
ax.legend(loc='best')
ax.grid(True)

# Display plot in Streamlit
st.pyplot(fig)




# Assume you already have this somewhere:
# selected_stock = st.selectbox("Choose Stock", ["AAPL", "TSLA", "GOOGL", "MSFT", ...])

# Download data using selected stock
df = yf.download(selected_stock, period="6mo", interval="1d")

# Compute ADL
def compute_adl(df):
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * df['Volume']
    adl = mfv.cumsum()
    return adl

df['ADL'] = compute_adl(df)

# Plot
st.subheader("📊 Advance Decline Line (ADL)")
fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(df.index, df['Close'], color='black', label='Close Price')
ax2 = ax1.twinx()
ax2.plot(df.index, df['ADL'], color='blue', label='ADL')

ax1.set_ylabel("Close Price")
ax2.set_ylabel("ADL Value")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

st.pyplot(fig)


# RSI Calculation
def compute_rsi(data, window=14):
    data = data.tail(100)  # Compute RSI only for the last 100 days
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Compute and plot RSI
import matplotlib.dates as mdates

st.subheader('Relative Strength Index (RSI)')

# Compute RSI
data['RSI'] = compute_rsi(data)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=data.index, y=data['RSI'], label='RSI', color='purple', ax=ax)

# Add overbought and oversold lines
ax.axhline(70, linestyle='--', color='red', label='Overbought')
ax.axhline(30, linestyle='--', color='green', label='Oversold')

# Format x-axis for better readability
ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Adjusts date spacing
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates
plt.xticks(rotation=45)  # Rotate x-axis labels for clarity

ax.set_title('Relative Strength Index (RSI)')
ax.legend()

# Display plot in Streamlit
st.pyplot(fig)

# Actual vs Predicted
st.subheader("🔁 Original vs Predicted Price")
fig4 = plt.figure(figsize=(10, 4))
plt.plot(predictions, 'r', label='Predicted Price')
plt.plot(data_test[100:].values, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original vs Predicted Prices')
plt.legend()
st.pyplot(fig4) 

# Future Forecasting
def predict_future(model, last_lookback, scaler, days=3):
    future_preds = []
    input_seq = last_lookback
    for _ in range(days):
        pred = model.predict(input_seq.reshape(1, input_seq.shape[0], 1))
        future_preds.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred, axis=0)
    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

st.subheader("🔮 Future Forecast (Next 3 Days)")
last_lookback = data_test_scaled[-100:]
future_prices = predict_future(model, last_lookback, scaler, days=3)
st.success(f"📅 Predicted Prices: {future_prices}")


# Footer
st.markdown("""
    <div class="custom-footer">
        Developed by Muneeb Ur Rehman | Powered by TensorFlow & Streamlit
    </div>
""", unsafe_allow_html=True)
