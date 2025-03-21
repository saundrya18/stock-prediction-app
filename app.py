import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import os

# Title of the app
st.title("Stock Price Prediction App using AI")
st.write("Enter a stock ticker below to predict future stock prices.")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "ADANIPOWER.NS")

# Function to fetch historical stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")  # Fetch last 5 years of data
    df = df.drop(columns=["Dividends", "Stock Splits"])  # Remove unnecessary columns
    return df

if ticker:
    data = get_stock_data(ticker)
    
    if not data.empty:
        st.subheader(f"Stock Price Data for {ticker}")
        st.write(data.tail())  # Show the last few rows

        # Plot stock price
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Closing Price"))
        fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Prepare Data for LSTM Model
        st.subheader("Training AI Model for Stock Price Prediction...")
        df = data[["Close"]]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)

        # Creating training dataset
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        def create_dataset(data, time_step=60):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            return np.array(X), np.array(Y)

        time_step = 100  # Increase history length for training
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Reshape for LSTM input
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model_filename = f"{ticker}_model.h5"

        # Load or Train Model
        if os.path.exists(model_filename):
            model = load_model(model_filename)
            st.write("Loaded pre-trained model for this stock.")
        else:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1)
            model.save(model_filename)
            st.write("Trained and saved new model for this stock.")

        # Make Predictions on Test Data
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # Inverse transform y_test for evaluation
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Evaluate Predictions
        mse = mean_squared_error(y_test_original, predicted_stock_price)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, predicted_stock_price)
        r2 = r2_score(y_test_original, predicted_stock_price)

        # Display metrics in Streamlit
        st.subheader("Model Performance Metrics 📊")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**R² Score:** {r2:.4f} (1 is best, closer to 0 is worse)")

        # Plot test predictions
        st.subheader("Stock Price Prediction vs Actual")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data.index[train_size + time_step + 1:], y=data["Close"][train_size + time_step + 1:], mode="lines", name="Actual Price"))
        fig_pred.add_trace(go.Scatter(x=data.index[train_size + time_step + 1:], y=predicted_stock_price.flatten(), mode="lines", name="Predicted Price"))
        fig_pred.update_layout(title="Stock Price Prediction vs Actual", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_pred)

        # Predict Future Prices (Next 7 Days)
        st.subheader("Predicting Future Stock Prices")
        future_days = 7
        last_100_days = scaled_data[-time_step:].reshape(1, time_step, 1)

        future_predictions = []
        for _ in range(future_days):
            pred = model.predict(last_100_days)[0][0]
            future_predictions.append(pred)
            last_100_days = np.append(last_100_days[:, 1:, :], [[[pred]]], axis=1)  # Slide the window

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create Future Dates for Display
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

        # Plot future predictions
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode="lines+markers", name="Future Predictions"))
        fig_future.update_layout(title="Future Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_future)

    else:
        st.warning("Invalid stock ticker! Please enter a valid one.")