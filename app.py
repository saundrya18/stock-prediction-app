import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3
import hashlib
import re
import random
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from email.mime.text import MIMEText
import secrets
import openai
from twilio.rest import Client
from streamlit_chat import message
import google.generativeai as genai

genai.configure(api_key=st.secrets["gemini_api_key"])

# Clear chatbot response when the app refreshes
if "chatbot_response" in st.session_state:
    del st.session_state["chatbot_response"]



# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Database Setup for Users
conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        mobile TEXT,
        email TEXT UNIQUE,
        username TEXT UNIQUE,
        password TEXT,
        otp TEXT,
        otp_timestamp REAL,
        reset_token TEXT,
        reset_token_timestamp REAL
    )
""")
try:
    cursor.execute("ALTER TABLE users ADD COLUMN preferred_sectors TEXT")
except sqlite3.OperationalError:
    pass

cursor.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        stock_ticker TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
""")
    # Create polls table (if it doesn't exist)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS polls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        stock_ticker TEXT,
        vote TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
""")

conn.commit()

def message(content, is_user):
    with st.chat_message("user" if is_user else "assistant"):
        st.markdown(content)

# Authentication Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    return user and user[0] == hash_password(password)
def is_valid_mobile(mobile):
    return re.match(r'^\d{10}$', mobile)

def is_mobile_duplicate(mobile):
    cursor.execute("SELECT mobile FROM users WHERE mobile = ?", (mobile,))
    return cursor.fetchone() is not None

def is_strong_password(password):
    return len(password) >= 8 and re.search(r'[A-Z]', password) and re.search(r'[a-z]', password) and re.search(r'[0-9]', password)

def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email)

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp(mobile, otp):
    st.write(f"OTP for {mobile}: {otp}")  # Print to screen

def verify_otp(username, user_otp):
    try:
        cursor.execute("SELECT otp, otp_timestamp FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        if result:
            stored_otp, timestamp = result
            stored_otp = str(stored_otp).strip()  # Ensure OTP is a string and trimmed
            user_otp = user_otp.strip()
            st.write(f"Stored OTP: {str(stored_otp).strip()}, Entered OTP: {user_otp.strip()}") #debug print
            st.write(f"Timestamp: {timestamp}, Current Time: {time.time()}") #debug print
            if str(stored_otp).strip() == user_otp.strip() and time.time() - timestamp < 300:
                return True
        return False
    except Exception as e:
        st.error(f"An error occured during otp verification: {e}")
        return False

def generate_reset_token():
    return secrets.token_urlsafe(32)

def send_reset_email(email, token):
    reset_link = f"https://stock-prediction-app-tq3fp8vwq795y7coijbaeu.streamlit.app?token={token}"

    msg = MIMEMultipart()
    msg["From"] = "kakkanair007@gmail.com"
    msg["To"] = email
    msg["Subject"] = "🔐 Password Reset Request"

    body = f"""
    Hi there,

    We received a request to reset your password.

    Click the link below to reset it:
    {reset_link}

    If you didn't request this, just ignore this email.

    — Stock Prediction App Team
    """
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login("kakkanair007@gmail.com", "kbyb awgd jxya dygx")
            server.sendmail(msg["From"], msg["To"], msg.as_string())
        st.success("Password reset email sent successfully! 📬")
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        st.error("Please contact support if this problem persists.")


def verify_reset_token(token):
    cursor.execute("SELECT username, reset_token_timestamp FROM users WHERE reset_token = ?", (token,))
    result = cursor.fetchone()
    if result and (time.time() - result[1] < 3600):
        return result[0]
    return None

def reset_password(username, new_password):
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    cursor.execute("UPDATE users SET password = ?, reset_token = NULL WHERE username = ?", (hashed_password, username))
    conn.commit()

def register_user(name, mobile, email, username, password, otp,sectors):
    if not is_valid_mobile(mobile):
        return "Invalid mobile number."

    if is_mobile_duplicate(mobile):
        return "Mobile number already exists."

    if not is_valid_email(email):
        return "Invalid email address."

    try:
        cursor.execute("INSERT INTO users (name, mobile, email, username, password, otp, otp_timestamp,preferred_sectors) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (name, mobile, email, username, hash_password(password), str(otp), time.time(), sectors))
        conn.commit()
        st.write(f"User registered with username: {username}")
        return True
    except sqlite3.IntegrityError:
        return "Username or Email already exists."
def clean_company_name(name):
    match = re.match(r"(.+?) \((The)\)", name)
    if match:
        return f"The {match.group(1)}"
    return name
# Session Management
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.otp_verified = False
    st.session_state.otp_sent = False

# Force logout if a reset token is in the URL
if "token" in st.query_params:
    st.session_state.authenticated = False
    st.session_state.username = ""

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.otp_verified = False
    st.session_state.otp_sent = False
    st.session_state.rerun = True

def add_to_watchlist(user_id, ticker):
    cursor.execute("INSERT INTO watchlist (user_id, stock_ticker) VALUES (?, ?)", (user_id, ticker))
    conn.commit()

def remove_from_watchlist(user_id, ticker):
    cursor.execute("DELETE FROM watchlist WHERE user_id = ? AND stock_ticker = ?", (user_id, ticker))
    conn.commit()

def get_watchlist(user_id):
    cursor.execute("SELECT stock_ticker FROM watchlist WHERE user_id = ?", (user_id,))
    return [row[0] for row in cursor.fetchall()]


# Login & Signup UI
if st.session_state.get('rerun'):
    st.session_state.rerun = False
    st.rerun() 
if not st.session_state.authenticated:
    token = st.query_params.get("token", "")
    if token:
        username = verify_reset_token(token)
        if username:
            st.title("🔐 Reset Your Password")
            new_password = st.text_input("New Password", type="password", key="reset_password")
            if st.button("Reset Password"):
                reset_password(username, new_password)
                st.success("✅ Password reset successful! You can now log in.")
                st.stop()
        else:
            st.error("Invalid or expired reset token.")
            st.stop()
    if 'reset_link_sent' not in st.session_state:
        st.session_state.reset_link_sent = False
    st.title("🔐 Stock Prediction App - Login")
    choice = st.radio("Choose an option", ["Login", "Sign Up", "Forgot Password"])
    
    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    
    elif choice == "Sign Up":
        name = st.text_input("Full Name")
        mobile = st.text_input("Mobile Number")
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer Goods", "Utilities", "Industrials", "Real Estate"]
        preferred_sectors = st.multiselect("Preferred Sectors", sectors)


        if st.button("Register") and not st.session_state.otp_sent:
            if not is_strong_password(password):
                st.error("Password must be at least 8 characters long and include uppercase, lowercase, and numbers.")
                st.stop()
            # Store user details in session state temporarily
            st.session_state.temp_user = {
                "name": name,
                "mobile": mobile,
                "email": email,
                "username": username,
                "password": password,
                "sectors": ",".join(preferred_sectors)
            }

            otp = generate_otp()
            st.session_state.current_otp = otp
            st.session_state.otp_timestamp = time.time()
            st.write(f"Generated OTP (Before Registration): {otp}")
            
            #format the number to E.164.
            mobile = "+"+re.sub(r'[^0-9]','', mobile) #remove any non numerical character, and add the + at the begining.
            send_otp(mobile, otp)
            st.session_state.otp_sent = True
            
        if st.session_state.otp_sent:
            user_otp = st.text_input("Enter OTP")
            if st.button("Verify OTP"):
                if (str(st.session_state.current_otp).strip() == user_otp.strip() and 
                    (time.time() - st.session_state.otp_timestamp < 300)):
                    user_data = st.session_state.temp_user
                    result = register_user(
                        user_data["name"],
                        user_data["mobile"],
                        user_data["email"],
                        user_data["username"],
                        user_data["password"],
                        st.session_state.current_otp,
                        user_data["sectors"]
                    )
                    if result is True:
                        st.session_state.otp_verified = True
                        st.success("User registered successfully! Please log in.")
                        del st.session_state.temp_user
                        del st.session_state.current_otp
                    else:
                        st.error(result)
                else:
                    st.error("Invalid OTP")
        if st.session_state.otp_verified: #add this conditional.
            st.write("Please proceed to login.")
        
    elif choice == "Forgot Password":
        email = st.text_input("Enter your registered email")
        if st.button("Send Reset Link"):
            cursor.execute("SELECT username FROM users WHERE email = ?", (email,))
            result = cursor.fetchone()
            if result:
                token = generate_reset_token()
                send_reset_email(email, token)
                cursor.execute("UPDATE users SET reset_token = ?, reset_token_timestamp = ? WHERE email = ?", (token, time.time(), email))
                conn.commit()
                st.session_state.reset_link_sent = True
            else:
                st.error("Email not found.")
            if st.session_state.reset_link_sent: #show a message if the reset link was sent.
                st.write("A password reset link has been sent to your email. Please check your inbox.")

        token = st.query_params.get("token", "")
        username = verify_reset_token(token)
        if username:
            new_password = st.text_input("New Password", type="password")
            if st.button("Reset Password"):
                reset_password(username, new_password)
                st.success("Password reset successful!")
        elif token: #only show the error when a token is present in the url.
            st.error("Invalid or expired reset token.")



# --- STOCK PREDICTION APP STARTS HERE ---
else:
    st.title("📈 Stock Price Prediction App with AI & User Sentiment")
    if "ticker" not in st.session_state:
        st.session_state.ticker = "AAPL"  # Default stock

    ticker = st.text_input("🔍 Enter Stock Ticker (e.g., AAPL, TSLA):", st.session_state.ticker)

    

    st.session_state.messages = []

    for msg in st.session_state.messages:
        message(msg["content"], is_user=(msg["role"] == "user"))

   
    # User Input
    user_input = st.chat_input("Ask me anything about stocks...")


    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        message(user_input, is_user=True)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(user_input)
    

        reply = response.text
        
        # Display bot message
        message(reply, is_user=False)


    if ticker and ticker != st.session_state.ticker:
        st.session_state.ticker = ticker
        st.rerun()
    for msg in st.session_state.messages:
        message(msg["content"], is_user=(msg["role"] == "user"))

    cursor.execute("SELECT id FROM users WHERE username = ?", (st.session_state.username,))
    user_id = cursor.fetchone()
    if user_id:
        user_id = user_id[0]

    # Logout Button
    st.sidebar.button("Logout", on_click=logout)
    # Sidebar separator
    st.sidebar.markdown("---")

    # Toggle profile visibility
    if "show_profile" not in st.session_state:
        st.session_state.show_profile = False

    if st.sidebar.button("👤 Profile"):
        st.session_state.show_profile = not st.session_state.show_profile

    # Profile section appears only when toggled on
    if st.session_state.show_profile:
        st.sidebar.subheader("Your Profile")

        cursor.execute("SELECT id, name, email, mobile, preferred_sectors FROM users WHERE username = ?", (st.session_state.username,))
        profile_data = cursor.fetchone()

        if profile_data:
            user_id, name, email, mobile, sectors = profile_data
            updated_email = st.sidebar.text_input("Email", value=email)
            updated_mobile = st.sidebar.text_input("Mobile", value=mobile)
            sector_options = ["Technology", "Healthcare", "Finance", "Energy", "Consumer Goods", "Utilities", "Industrials", "Real Estate"]
            updated_sectors = st.sidebar.multiselect("Preferred Sectors", sector_options, default=[s.strip() for s in (sectors or "").split(",") if s.strip()])

            if st.sidebar.button("💾 Update Profile"):
                try:
                    cursor.execute("""
                        UPDATE users 
                        SET email = ?, mobile = ?, preferred_sectors = ?
                        WHERE id = ?
                    """, (updated_email, updated_mobile, ",".join(updated_sectors), user_id))
                    conn.commit()
                    st.sidebar.success("Profile updated successfully! ✅")
                except Exception as e:
                    st.sidebar.error(f"Error updating profile: {e}")

    st.sidebar.subheader("📈 Trending Stocks in Your Sectors")
    # Get user’s sector preferences
    cursor.execute("SELECT preferred_sectors FROM users WHERE id = ?", (user_id,))
    sector_result = cursor.fetchone()
    if sector_result and sector_result[0]:
        user_sectors = [s.strip() for s in sector_result[0].split(",") if s.strip()]

        sector_to_stocks = {
            "Technology": ["AAPL", "MSFT", "NVDA"],
            "Healthcare": ["JNJ", "PFE", "MRK"],
            "Finance": ["JPM", "BAC", "WFC"],
            "Energy": ["XOM", "CVX", "SLB"],
            "Consumer Goods": ["PG", "KO", "PEP"],
            "Utilities": ["NEE", "DUK", "SO"],
            "Industrials": ["UNP", "GE", "CAT"],
            "Real Estate": ["PLD", "AMT", "CCI"]
        }

        for sector in user_sectors:
            st.sidebar.markdown(f"**{sector}**")
            stocks = sector_to_stocks.get(sector, [])
            for stock in stocks:
                stock_info = yf.Ticker(stock).info
                stock_name = stock_info.get("shortName", stock)
                stock_name = clean_company_name(stock_info.get("shortName", stock))
                stock_data = yf.Ticker(stock).history(period="1mo")

                if not stock_data.empty:
                    st.sidebar.write(f"📊 {stock_name} ({stock})")
                    st.sidebar.line_chart(stock_data["Close"])
                else:
                    st.sidebar.write(f"{stock_name} ({stock}): No recent data available.")
    else:
        st.sidebar.info("You haven’t selected any sector preferences yet.")
    

    st.sidebar.subheader("📌 Your Watchlist")

    # Get user ID
    cursor.execute("SELECT id FROM users WHERE username = ?", (st.session_state.username,))
    user_id = cursor.fetchone()
    if user_id:
        user_id = user_id[0]
        watchlist_ticker = st.sidebar.text_input("🔍 Enter Stock Ticker:", "AAPL")

        if st.sidebar.button("➕ Add"):
            if watchlist_ticker:
                add_to_watchlist(user_id, watchlist_ticker)
                st.sidebar.success(f"{watchlist_ticker} added to your watchlist!")
            else:
                st.sidebar.warning("Please enter a ticker to add to the watchlist.")

        if st.sidebar.button("❌ Remove"):
            if watchlist_ticker:
                remove_from_watchlist(user_id, watchlist_ticker)
                st.sidebar.warning(f"{watchlist_ticker} removed from your watchlist!")
            else:
                st.sidebar.warning("Please enter a ticker to remove from the watchlist.")

        watchlist = get_watchlist(user_id)
        if watchlist:
            st.sidebar.write("✅ " + ", ".join(watchlist))
        else:
            st.sidebar.write("Your watchlist is empty.")

        st.sidebar.subheader("📊 Watchlist Stock Prices")
        for stock in watchlist:
            stock_data = yf.Ticker(stock).history(period="1mo")
            if not stock_data.empty:
                st.sidebar.write(f"### {stock}")
                st.sidebar.line_chart(stock_data['Close'])
            else:
                st.sidebar.write(f"No data for {stock}.")

    if st.session_state.ticker:
        try:
            stock = yf.Ticker(st.session_state.ticker) # Use st.session_state.ticker here
            data = stock.history(period="5y").drop(columns=["Dividends", "Stock Splits"], errors='ignore')
            if data.empty:
                raise ValueError("Invalid stock ticker or no data available.")
        except:
            st.error("⚠ Invalid stock ticker. Please enter a valid symbol.")
            st.stop()

        st.subheader(f"📊 Stock Price Data for {st.session_state.ticker}") # Use st.session_state.ticker here
        st.write(data.tail())


        # Plot Stock Prices
        fig_hist = go.Figure([go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Closing Price")])
        fig_hist.update_layout(title=f"{ticker} Stock Price (Last 5 Years)", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_hist)

        # Actual vs. Predicted Stock Prices
        st.subheader("📉 Actual vs Predicted Stock Prices")
        st.write("This graph shows the actual stock prices compared to the predicted stock prices over the last 100 days.")

        sequence_length = 60
        recent_data = data["Close"].values[-(100 + sequence_length):]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_recent_data = scaler.fit_transform(recent_data.reshape(-1, 1))

        X_test = []
        for i in range(100):
            X_test.append(scaled_recent_data[i:i+sequence_length])
        X_test = np.array(X_test)

        # Load model and make predictions
        if os.path.exists("stock_lstm_model.h5"):
            model = load_model("stock_lstm_model.h5")
            predicted_prices = model.predict(X_test)
            predicted_prices = predicted_prices[:, 0]  # Take only the first predicted day from each 7-day output
            predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1)).flatten()
        else:
            st.error("Model file not found. Please train the model first.")
            st.stop()

        actual_prices = recent_data[sequence_length:]

        accuracy_threshold = 0.05  # 5%
        accurate_predictions = np.abs(predicted_prices - actual_prices) / actual_prices <= accuracy_threshold
        accuracy_percentage = np.sum(accurate_predictions) / len(actual_prices) * 100
        fig_actual_vs_pred = go.Figure()
        fig_actual_vs_pred.add_trace(go.Scatter(x=data.index[-100:], y=actual_prices, mode="lines", name="Actual Prices"))
        fig_actual_vs_pred.add_trace(go.Scatter(x=data.index[-100:], y=predicted_prices, mode="lines", name="Predicted Prices"))
        fig_actual_vs_pred.update_layout(title="Actual vs Predicted Stock Prices", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_actual_vs_pred)
        st.success(f"✅ Prediction Accuracy (within ±5%) over last 100 days: **{accuracy_percentage:.2f}%**")
        
        
        # Sentiment Analysis from News
        st.subheader("📰 Market Sentiment Analysis")
        newsapi = NewsApiClient(api_key='c5a1e5e2f8f74a2c9a4e480fa92b2bef')  
        
        try:
            headlines = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
            sentiment_scores = []
            for article in headlines['articles']:
                score = analyzer.polarity_scores(article['title'])['compound']
                sentiment_scores.append(score)
                st.markdown(f"[{article['title']}]({article['url']})", unsafe_allow_html=True)
                st.write(f"💬 Sentiment Score: {score:.2f}")
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            sentiment_color = "green" if avg_sentiment >= 0 else "red"
            st.markdown(f"""<h4>📊 <b>Average Sentiment Score:</b> <span style='color:{sentiment_color}'>{avg_sentiment:.2f}</span></h4>""", unsafe_allow_html=True)

        except:
            st.error("No latest news for the provided stock")
            avg_sentiment = 0

        # --- STOCK PRICE PREDICTION ---
        st.subheader("🔮 Predicting Next 7 Days of Stock Prices")
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data[['Close']])

        # Prepare Data for LSTM Model
        def create_sequences(data, seq_length=60):
            sequences = []
            labels = []
            for i in range(len(data) - seq_length - 7):
                sequences.append(data[i:i+seq_length])
                labels.append(data[i+seq_length:i+seq_length+7])
            return np.array(sequences), np.array(labels)

        seq_length = 60
        X, y = create_sequences(scaled_data)
        
        if os.path.exists("stock_lstm_model.h5"):
            model = load_model("stock_lstm_model.h5")
        else:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(7)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=10, batch_size=16)
            model.save("stock_lstm_model.h5")
        last_60_days = scaled_data[-seq_length:]
        predicted_prices = model.predict(last_60_days.reshape(1, seq_length, 1))[0]
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1)).flatten()

        # Sentiment-Adjusted Prediction
        sentiment_adjustment = avg_sentiment * 0.02 * predicted_prices
        sentiment_adjusted_prices = predicted_prices + sentiment_adjustment

        # Plot Predictions
        future_dates = pd.date_range(start=data.index[-1], periods=8, freq='B')[1:]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode="lines", name="Predicted Prices"))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=sentiment_adjusted_prices, mode="lines", name="Sentiment-Adjusted Prices"))
        fig_pred.update_layout(title="7-Day Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_pred)


        # Poll Feature
        st.subheader("📢 Would you buy this stock?")
        poll_choice = st.radio("Select an option:", ["Yes", "No"])
        if st.button("Submit Vote"):
            # Get user ID
            cursor.execute("SELECT id FROM users WHERE username = ?", (st.session_state.username,))
            user_id = cursor.fetchone()
            if user_id:
                user_id = user_id[0]

                # Insert vote into polls table
                cursor.execute("INSERT INTO polls (user_id, stock_ticker, vote) VALUES (?, ?, ?)", (user_id, ticker, poll_choice))
                conn.commit()

                # Retrieve and display updated poll results
                cursor.execute("SELECT vote, COUNT(*) FROM polls WHERE stock_ticker = ? GROUP BY vote", (ticker,))
                results = cursor.fetchall()
                total_votes = 0
                vote_counts = {"Yes": 0, "No": 0}

                for vote, count in results:
                    total_votes += count
                    vote_counts[vote] = count

                if total_votes > 0:
                    st.write(f"Yes: {vote_counts['Yes'] / total_votes * 100:.2f}%")
                    st.write(f"No: {vote_counts['No'] / total_votes * 100:.2f}%")
                else:
                    st.write("No votes yet.")
 