import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.callbacks import EarlyStopping
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel news fetching

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# News API Key
NEWS_API_KEY = "e397561aaa3245309bdcc66a38168ecc"  

def fetch_news(ticker):
    """Fetches news articles for a given ticker."""
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        return articles
    except Exception:
        return {"articles": []}  # Return empty structure on error

def analyze_sentiment(text):
    """Analyzes the sentiment of a given text."""
    return analyzer.polarity_scores(text)['compound']

def get_news_sentiment(ticker):
    """
    Fetches news, analyzes sentiment of individual articles, and calculates an overall sentiment.
    Handles potential errors and empty news results.
    """
    try:
        articles = fetch_news(ticker)  # Fetch news articles

        if 'articles' not in articles or not articles['articles']:
            st.warning("No relevant news found. Proceeding with default predictions.")
            return 0, []  # Return neutral sentiment and empty list

        # Use ThreadPoolExecutor to parallelize sentiment analysis
        with ThreadPoolExecutor() as executor:
            sentiment_futures = [executor.submit(analyze_sentiment, article['title']) for article in articles['articles']]
            sentiments = [future.result() for future in as_completed(sentiment_futures)]

        news_data = [(article['title'], sentiment) for article, sentiment in zip(articles['articles'], sentiments)]
        overall_sentiment = np.mean(sentiments) if sentiments else 0

        return overall_sentiment, news_data

    except Exception as e:
        st.error(f"Error fetching/analyzing news: {e}")
        return 0, []  # Return neutral sentiment and empty list

# Title of the app
st.title("Stock Price Prediction App using AI with Sentiment Analysis")
st.write("Enter a stock ticker below to predict future stock prices.")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "ADANIPOWER.NS")

# Function to fetch historical stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")
        df = df.drop(columns=["Dividends", "Stock Splits"])  # Remove unnecessary columns
        return df
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame

if ticker:
    data = get_stock_data(ticker)

    if not data.empty:
        st.subheader(f"Stock Price Data for {ticker}")
        st.write(data.tail())

        # Plot stock price
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Closing Price"))
        fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Prepare Data for LSTM Model
        st.subheader("Training AI Model for Stock Price Prediction...")
        df = data[["Close", "Volume"]]  # Include Volume as a feature
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        # Creating training dataset
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        def create_dataset(data, time_step=100):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step)])
                Y.append(data[i + time_step, 0])  # Predict the "Close" price
            return np.array(X), np.array(Y)

        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Reshape for LSTM input
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)  # 2 features: Close, Volume
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)

        model_filename = f"{ticker}_model.h5"

        # Load or Train Model
        if os.path.exists(model_filename):
            model = load_model(model_filename)
            st.write("Loaded pre-trained model for this stock.")
            # Check the input shape of the loaded model
            if model.input_shape[-1] != 2:
                st.error(
                    "Error: Loaded model was trained with a different number of features.  Please delete the model file and retrain."
                )
                st.stop()
        else:
            # Define the LSTM model
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 2)),  # 2 features: Close, Volume
                Dropout(0.3),  # Increased dropout
                LSTM(64, return_sequences=False),
                Dropout(0.3),
                Dense(32),  # Increased Dense layer size
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")

            # Add Early Stopping to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increased patience

            # Increased epochs
            model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.1,
                      callbacks=[early_stopping])
            model.save(model_filename)
            st.write("Trained and saved new model for this stock.")

        # Make Predictions on Test Data
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(
            np.concatenate((predicted_stock_price, np.zeros_like(predicted_stock_price)),
                           axis=1))[:, 0]  # Inverse transform only the Close price
        y_test_actual = scaler.inverse_transform(
            np.concatenate((y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))), axis=1))[:, 0]

        # Calculate Error Metrics
        mse = mean_squared_error(y_test_actual, predicted_stock_price)
        mae = mean_absolute_error(y_test_actual, predicted_stock_price)
        r2 = r2_score(y_test_actual, predicted_stock_price)
        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"R² Score: {r2:.4f}")

        # Market Sentiment Analysis
        overall_sentiment, news_data = get_news_sentiment(ticker)
        st.subheader("Recent News Headlines")
        for title, score in news_data:
            st.write(f"- {title} (Sentiment Score: {score:.4f})")
        st.write(f"Overall Market Sentiment: {overall_sentiment:.4f}")

        sentiment_adjustment = max(0.9, min(1.1, 1 + (overall_sentiment / 15)))  # Adjust more conservatively

        # Predict Future Prices (Next 7 Days)
        future_days = 7
        last_100_days = scaled_data[-time_step:].reshape(1, time_step, 2)  # Use the last 100 days of scaled data
        future_predictions = []
        for _ in range(future_days):
            pred = model.predict(last_100_days)[0][0]
            future_predictions.append(pred)
            last_100_days = np.append(last_100_days[:, 1:, :], [[[pred, scaled_data[-1, 1]]]],
                                      axis=1)  # important
        future_predictions = scaler.inverse_transform(
            np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((future_days, 1))), axis=1))[:, 0]
        adjusted_predictions = future_predictions * sentiment_adjustment

        # Display Future Predictions Graph
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode="lines+markers",
                                        name="Original Prediction"))
        fig_future.add_trace(go.Scatter(x=future_dates, y=adjusted_predictions.flatten(), mode="lines+markers",
                                        name="Sentiment-Adjusted Prediction", line=dict(dash='dot')))
        fig_future.update_layout(title="Future Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_future)
    else:
        st.warning("Invalid stock ticker! Please enter a valid one.")