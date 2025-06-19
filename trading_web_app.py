import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("AI Trading Bot - Stock Price Prediction")

# API key placeholder
API_KEY = st.secrets["API_KEY"]


# User inputs
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, RELIANCE.BSE)", "AAPL")

# Fetch stock data
@st.cache_data
def fetch_stock_data(symbol, api_key, outputsize="compact"):
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        data.sort_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

if st.button("Run Prediction"):
    data = fetch_stock_data(ticker, API_KEY)

    if data.empty or len(data) < 60:
        st.error("Insufficient data. Try another symbol or check API key.")
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
            ax.set_title(f"{ticker} - Predicted vs Actual")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
