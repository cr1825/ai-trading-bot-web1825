import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from fyers_apiv2 import fyersModel
from streamlit import secrets

# Set page layout
st.set_page_config(layout="wide")
st.title("AI Trading Bot - Stock Price Prediction (Fyers API)")

# Fyers credentials from Streamlit secrets
client_id = secrets["FYERS_CLIENT_ID"]
access_token = secrets["FYERS_ACCESS_TOKEN"]

# Initialize Fyers
fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, log_path=None)

# User input for ticker
ticker = st.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS, RPOWER)", "RELIANCE")

# Fetch stock data
@st.cache_data
def fetch_stock_data(symbol):
    try:
        payload = {
            "symbol": f"NSE:{symbol}-EQ",
            "resolution": "D",
            "date_format": "1",
            "range_from": "2023-01-01",
            "range_to": "2024-06-01",
            "cont_flag": "1"
        }
        response = fyers.history(payload)
        if "candles" not in response or not response["candles"]:
            st.error(f"Error fetching data from Fyers: {response}")
            return pd.DataFrame()

        df = pd.DataFrame(response["candles"], columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        return df

    except Exception as e:
        st.error(f"Exception while fetching data: {e}")
        return pd.DataFrame()

# Run the prediction
if st.button("Run Prediction"):
    data = fetch_stock_data(ticker)

    if data.empty or len(data) < 60:
        st.error("Insufficient data. Try another symbol or check Fyers credentials.")
    else:
        # Feature engineering
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['Target'] = data['Close'].shift(-1)
        data.dropna(inplace=True)

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50']
        X = data[features]
        y = data['Target']

        if len(X) <= 50:
            st.error("Not enough data after preprocessing.")
        else:
            X_train = X[:-50]
            y_train = y[:-50]
            X_test = X[-50:]
            y_test = y[-50:]

            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            data_test = data[-50:].copy()
            data_test['Predicted_Close'] = predictions

            # Buy/Sell Signal
            data_test['Signal'] = 0
            data_test.loc[data_test['Predicted_Close'] > data_test['Close'], 'Signal'] = 1
            data_test.loc[data_test['Predicted_Close'] < data_test['Close'], 'Signal'] = -1

            st.subheader("Buy/Sell Signals")
            st.dataframe(data_test[['Close', 'Predicted_Close', 'Signal']].tail(10))

            # Plotting
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(data_test['Close'].values, label='Actual Close')
            ax.plot(data_test['Predicted_Close'].values, label='Predicted Close')
            ax.set_title(f"{ticker.upper()} - Predicted vs Actual")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            ax.set_title(f"{ticker} - Predicted vs Actual")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
