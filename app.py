#  _____________________________________________________________________________________________________
# murge the all approch1
# # 📦 Imports & Setup
# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import requests
# import os
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout

# # Load GROQ API key
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# # 🧠 GROQ Class
# class GroqLLM:
#     def __init__(self, api_key, model="llama3-70b-8192"):
#         self.api_key = api_key
#         self.model = model
#         self.base_url = "https://api.groq.com/openai/v1/chat/completions"

#     def complete(self, prompt):
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         data = {
#             "model": self.model,
#             "messages": [
#                 {"role": "system", "content": "You are a financial assistant that provides expert-level investment analysis."},
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.7,
#             "max_tokens": 1024
#         }
#         response = requests.post(self.base_url, headers=headers, json=data)
#         if not response.ok:
#             raise Exception(f"{response.status_code} Error: {response.text}")
#         return response.json()["choices"][0]["message"]["content"]

# groq_llm = GroqLLM(api_key=groq_api_key)

# # ---------------- STOCK HELPERS ----------------
# def fetch_stock_price(ticker):
#     stock = yf.Ticker(ticker)
#     data = stock.history(period='1d')
#     return data['Close'].iloc[-1] if not data.empty else None

# def calculate_profit_loss(investment, buy_price, current_price):
#     shares = investment / buy_price
#     return (current_price - buy_price) * shares

# def assess_risk(profit_loss):
#     if profit_loss > 1000:
#         return "High"
#     elif profit_loss > 0:
#         return "Medium"
#     else:
#         return "Low"

# def fetch_stock_news():
#     url = "https://www.moneycontrol.com/news/business/markets/"
#     headers = {"User-Agent": "Mozilla/5.0"}
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, "html.parser")
#         articles = soup.find_all("li", class_="clearfix")
#         news = [(a.find("h2").text.strip(), a.find("a")["href"]) for a in articles[:5]]
#         return news
#     else:
#         return [("Failed to fetch news", "#")]

# def load_stock_data(ticker, start, end):
#     df = yf.download(ticker, start=start, end=end)
#     return df[['Close']]

# def preprocess_data(df):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
#     return scaled, scaler

# def create_sequences(data, seq_len=60):
#     X, y = [], []
#     for i in range(seq_len, len(data)):
#         X.append(data[i-seq_len:i, 0])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)

# def build_lstm_model(input_shape):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(25),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # ---------------- STREAMLIT UI ----------------
# st.set_page_config(page_title="📈 Stock Virtual Analyst", layout="wide")
# st.sidebar.title("📊 Equity Research Dashboard")
# menu = st.sidebar.radio("Choose an option:", [
#     "Portfolio Tracker",
#     "Stock Price Prediction",
#     "Live Stock Market News",
#     "AI Investment Report"
# ])

# if menu == "Portfolio Tracker":
#     st.title("📈 Portfolio Tracker")
#     company = st.text_input("Enter Stock Symbol (e.g., TCS.NS):")
#     investment = st.number_input("Investment Amount (INR):", min_value=1000.0)
#     buy_price = st.number_input("Buy Price (INR):", min_value=1.0)

#     if st.button("Analyze Portfolio"):
#         current_price = fetch_stock_price(company)
#         if current_price:
#             profit_loss = calculate_profit_loss(investment, buy_price, current_price)
#             st.write(f"Current Price: ₹{current_price:.2f}")
#             st.write(f"Profit/Loss: ₹{profit_loss:.2f}")
#             st.write(f"Risk Level: {assess_risk(profit_loss)}")
#             st.write("Advice:", "Hold" if profit_loss >= 0 else "Sell")
#         else:
#             st.error("Could not fetch stock data.")

# elif menu == "Stock Price Prediction":
#     st.title("📊 LSTM Stock Price Prediction")
#     ticker = st.text_input("Enter Stock Ticker:", "TCS.NS")
#     start_date, end_date = "2020-01-01", "2024-01-01"

#     if st.button("Predict Future Price"):
#         df = load_stock_data(ticker, start_date, end_date)
#         if df.empty:
#             st.error("Invalid stock ticker or no data.")
#         else:
#             scaled_data, scaler = preprocess_data(df)
#             X, y = create_sequences(scaled_data)
#             X = np.reshape(X, (X.shape[0], X.shape[1], 1))
#             model = build_lstm_model((X.shape[1], 1))
#             model.fit(X, y, epochs=5, batch_size=32, verbose=0)
#             future_input = scaled_data[-60:].reshape(1, 60, 1)
#             prediction = model.predict(future_input)
#             predicted_price = scaler.inverse_transform(prediction)
#             st.success(f"📈 Predicted Future Price: ₹{predicted_price[0][0]:.2f}")

# elif menu == "Live Stock Market News":
#     st.title("📰 Stock Market News")
#     for i, (headline, link) in enumerate(fetch_stock_news(), start=1):
#         st.markdown(f"{i}. [{headline}]({link})")

# elif menu == "AI Investment Report":
#     st.title("🤖 AI Investment Report Generator")
#     symbols = st.text_input("Enter stock symbols (comma-separated):", "AAPL, MSFT, GOOGL")
#     if st.button("Generate Report") and symbols:
#         with st.spinner("Generating analysis..."):
#             prompt = f"""
#                 You're an expert financial analyst. Give a detailed investment analysis
#                 and strategic recommendation for these companies: {symbols}.
#                 Include trends, risks, growth potential, and a final investment decision.
#             """
#             try:
#                 result = groq_llm.complete(prompt)
#                 st.markdown(result)
#                 st.success("✅ Report generated!")
#                 st.download_button("📥 Download Report", result, file_name="investment_report.md")
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")
# import streamlit as st
# import yfinance as yf
# import numpy as np
# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# import os
# from dotenv import load_dotenv

# # ---------------- Load GROQ API Key ----------------
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# # ---------------- GROQ API CLASS ----------------
# class GroqLLM:
#     def __init__(self, api_key, model="llama3-70b-8192"):
#         self.api_key = api_key
#         self.model = model
#         self.base_url = "https://api.groq.com/openai/v1/chat/completions"

#     def complete(self, prompt):
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         data = {
#             "model": self.model,
#             "messages": [
#                 {"role": "system", "content": "You are a financial assistant that provides expert-level investment analysis."},
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.7,
#             "max_tokens": 1024
#         }
#         response = requests.post(self.base_url, headers=headers, json=data)
#         if not response.ok:
#             raise Exception(f"{response.status_code} Error: {response.text}")
#         return response.json()["choices"][0]["message"]["content"]

# # Initialize LLM
# groq_llm = GroqLLM(api_key=groq_api_key)

# # ---------------- Streamlit App ----------------
# st.set_page_config(page_title="AI Investment Dashboard", layout="wide")
# st.sidebar.title("📊 Investment Dashboard")
# menu = st.sidebar.selectbox("📂 Select Module", ["🏠 Home", "📈 Portfolio Tracker", "📊 Stock Price Prediction", "📰 Market News", "🤖 AI Investment Report"])

# # ---------------- Helper Functions ----------------
# def fetch_stock_price(ticker):
#     try:
#         stock = yf.Ticker(ticker)
#         data = stock.history(period='1d')
#         return data['Close'].iloc[-1] if not data.empty else None
#     except Exception as e:
#         st.error(f"Error fetching stock price: {e}")
#         return None

# def calculate_profit_loss(investment, buy_price, current_price):
#     shares = investment / buy_price
#     return (current_price - buy_price) * shares

# def assess_risk(profit_loss):
#     if profit_loss > 1000:
#         return "High"
#     elif profit_loss > 0:
#         return "Medium"
#     else:
#         return "Low"

# def fetch_stock_news():
#     try:
#         url = "https://www.moneycontrol.com/news/business/markets/"
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(url, headers=headers)
#         soup = BeautifulSoup(response.text, "html.parser")
#         articles = soup.find_all("li", class_="clearfix")
#         news_list = []
#         for article in articles[:5]:
#             headline = article.find("h2").text.strip()
#             link = article.find("a")["href"]
#             news_list.append((headline, link))
#         return news_list
#     except Exception as e:
#         return [("Failed to fetch news", "#")]

# def load_stock_data(ticker, start, end):
#     df = yf.download(ticker, start=start, end=end)
#     return df[['Close']] if 'Close' in df else pd.DataFrame()

# def preprocess_data(df):
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
#     return scaled, scaler

# def create_sequences(data, seq_length=60):
#     X, y = [], []
#     for i in range(seq_length, len(data)):
#         X.append(data[i-seq_length:i, 0])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)

# def build_lstm_model(input_shape):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(25),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # ---------------- Home Page ----------------
# if menu == "🏠 Home":
#     st.title("💼 AI-Powered Investment Analyst")
#     st.markdown("""
#     Welcome to the **AI Investment Dashboard**. This tool helps you:

#     - 📈 Track your **portfolio**
#     - 📊 Predict **stock prices**
#     - 📰 Get **market news**
#     - 🤖 Generate **AI investment reports**
#     """)

# # ---------------- Portfolio Tracker ----------------

# elif menu == "📈 Portfolio Tracker":
#     st.title("📈 Multi-Stock Portfolio Tracker")

#     st.markdown("### ➕ Add Stocks to Your Portfolio")
#     with st.form(key="portfolio_form"):
#         tickers = st.text_area("Enter Stock Symbols (comma-separated, e.g., TCS.NS, INFY.NS, RELIANCE.NS):").split(",")
#         investments = st.text_area("Investment Amounts (comma-separated, match order):").split(",")
#         buy_prices = st.text_area("Buy Prices per Share (comma-separated, match order):").split(",")
#         submit = st.form_submit_button("Analyze Portfolio")

#     if submit:
#         if not (len(tickers) == len(investments) == len(buy_prices)):
#             st.error("⚠️ Please ensure all fields have the same number of entries.")
#         else:
#             tickers = [t.strip().upper() for t in tickers]
#             investments = list(map(float, investments))
#             buy_prices = list(map(float, buy_prices))

#             total_profit_loss = 0
#             data = []
#             pie_data = []

#             for symbol, inv, bp in zip(tickers, investments, buy_prices):
#                 current = fetch_stock_price(symbol)
#                 if current:
#                     profit = calculate_profit_loss(inv, bp, current)
#                     risk = assess_risk(profit)
#                     advice = "Hold" if profit >= 0 else "Sell"
#                     total_profit_loss += profit
#                     data.append({
#                         "Ticker": symbol,
#                         "Investment": f"₹{inv:.2f}",
#                         "Buy Price": f"₹{bp:.2f}",
#                         "Current Price": f"₹{current:.2f}",
#                         "Profit/Loss": f"₹{profit:.2f}",
#                         "Risk": risk,
#                         "Advice": advice
#                     })
#                     pie_data.append({"Stock": symbol, "Profit/Loss": profit})

#                 else:
#                     st.warning(f"⚠️ Could not fetch data for {symbol}")

#             if data:
#                 df = pd.DataFrame(data)
#                 st.dataframe(df)
#                 st.markdown(f"### 📊 Total Portfolio Profit/Loss: ₹{total_profit_loss:.2f}")

#                 # Pie Chart: Profit/Loss Distribution
#                 pie_df = pd.DataFrame(pie_data)
#                 pie_df = pie_df[pie_df["Profit/Loss"] != 0]  # Optional: hide zero profit/loss stocks
#                 import matplotlib.pyplot as plt
#                 fig, ax = plt.subplots()
#                 ax.pie(pie_df['Profit/Loss'], labels=pie_df['Stock'], autopct='%1.1f%%', startangle=90)
#                 ax.axis('equal')
#                 st.subheader("📉 Profit/Loss Distribution by Stock")
#                 st.pyplot(fig)

# # ---------------- Stock Prediction ----------------
# elif menu == "📊 Stock Price Prediction":
#     st.title("📊 Stock Price Prediction using LSTM")
#     ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS):", value="TCS.NS")
#     if st.button("Predict Price"):
#         df = load_stock_data(ticker, "2020-01-01", "2024-01-01")
#         if df.empty:
#             st.error("No data found.")
#         else:
#             scaled_data, scaler = preprocess_data(df)
#             X, y = create_sequences(scaled_data)
#             X = X.reshape((X.shape[0], X.shape[1], 1))
#             model = build_lstm_model((X.shape[1], 1))
#             with st.spinner("Training LSTM model..."):
#                 model.fit(X, y, epochs=5, batch_size=32, verbose=0)
#             prediction_input = scaled_data[-60:].reshape(1, 60, 1)
#             prediction = model.predict(prediction_input)
#             predicted_price = scaler.inverse_transform(prediction)
#             st.success(f"📈 Predicted Future Price: ₹{predicted_price[0][0]:.2f}")

# # ---------------- Market News ----------------
# elif menu == "📰 Market News":
#     st.title("📰 Market News")
#     news = fetch_stock_news()
#     for i, (title, link) in enumerate(news):
#         st.markdown(f"{i+1}. [{title}]({link})")

# # ---------------- AI Investment Report ----------------
# elif menu == "🤖 AI Investment Report":
#     st.title("🤖 AI Investment Report")
#     example = "AAPL, MSFT, GOOGL"
#     symbols = st.text_input("Enter Stock Symbols (comma-separated):", value=example)
#     if st.button("Generate Report") and symbols:
#         with st.spinner("Generating report..."):
#             prompt = f"""
#             You're an expert financial analyst. Give a detailed investment analysis and strategic recommendation 
#             for the following companies: {symbols}. Include market trends, financial health, growth prospects, 
#             risks, and a final recommendation.
#             """
#             try:
#                 report = groq_llm.complete(prompt)
#                 st.markdown(report)
#                 st.success("✅ Report Ready!")
#                 st.download_button("📥 Download Report", report, file_name="investment_report.md")
#             except Exception as e:
#                 st.error(f"Error: {e}")



import os
os.environ["YFINANCE_USE_REQUESTS"] = "1"

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import plotly.graph_objs as go
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from dotenv import load_dotenv

# Load GROQ API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")



# GROQ LLM Class
class GroqLLM:
    def __init__(self, api_key, model="llama3-70b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def complete(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial assistant that provides expert-level investment analysis."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        if not response.ok:
            raise Exception(f"{response.status_code} Error: {response.text}")
        return response.json()["choices"][0]["message"]["content"]

groq_llm = GroqLLM(api_key=groq_api_key)

# Streamlit UI
st.set_page_config(page_title="AI Investment Dashboard", layout="wide")
st.sidebar.title("📊 Investment Dashboard")
menu = st.sidebar.selectbox("📂 Select Module", [
    "🏠 Home",
    "📈 Portfolio Tracker",
    "📊 Stock Price Prediction",
    "📰 Market News",
    "🤖 AI Investment Report",
    "📌 Best Stocks to Buy",
    "📋 Stock Screener",
    "💬 Chat with Analyst",
    "📊 2-Year Stock Chart",
    "₿ Live Crypto Charts"
])

# Helper Functions
def fetch_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        return data['Close'].iloc[-1] if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching stock price: {e}")
        return None



def calculate_profit_loss(investment, buy_price, current_price):
    shares = investment / buy_price
    return (current_price - buy_price) * shares

def assess_risk(profit_loss):
    if profit_loss > 1000:
        return "High"
    elif profit_loss > 0:
        return "Medium"
    else:
        return "Low"

def fetch_stock_news():
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return [(f"Failed to fetch news. Status code: {response.status_code}", "#")]

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("li", class_="clearfix")

        if not articles:
            return [("No articles found. Website structure may have changed.", "#")]

        news_list = []
        for article in articles[:5]:
            h2_tag = article.find("h2")
            if not h2_tag:
                continue
            headline = h2_tag.text.strip()
            link = article.find("a")["href"]
            news_list.append((headline, link))

        if not news_list:
            return [("No headlines found. Page structure changed?", "#")]

        return news_list
    except Exception as e:
        return [(f"Exception occurred: {str(e)}", "#")]



def load_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']] if 'Close' in df else pd.DataFrame()



def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled, scaler

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------- Home Page ----------------
if menu == "🏠 Home":
    st.title("💼 AI-Powered Investment Analyst")
    st.markdown("""
    Welcome to the **AI Investment Dashboard**. This tool helps you:

    - 📈 Track your **portfolio**
    - 📊 Predict **stock prices**
    - 📰 Get **market news**
    - 🤖 Generate **AI investment reports**
    - 📌 Get **best stock recommendations**
    - 📋 Scan **top NIFTY stocks**
    - 💬 Chat with **Analyst**
    - 📊 2-Year Stock **Chart**
    - ₿ Live **Crypto Charts**
    """)

# ---------------- Portfolio Tracker ----------------
elif menu == "📈 Portfolio Tracker":
    st.title("📈 Multi-Stock Portfolio Tracker")
    st.markdown("### ➕ Add Stocks to Your Portfolio")
    with st.form(key="portfolio_form"):
        tickers = st.text_area("Enter Stock Symbols (comma-separated, e.g., TCS.NS, INFY.NS, RELIANCE.NS):").split(",")
        investments = st.text_area("Investment Amounts (comma-separated, match order):").split(",")
        buy_prices = st.text_area("Buy Prices per Share (comma-separated, match order):").split(",")
        submit = st.form_submit_button("Analyze Portfolio")

    if submit:
        if not (len(tickers) == len(investments) == len(buy_prices)):
            st.error("⚠️ Please ensure all fields have the same number of entries.")
        else:
            tickers = [t.strip().upper() for t in tickers]
            investments = list(map(float, investments))
            buy_prices = list(map(float, buy_prices))

            total_profit_loss = 0
            data = []
            pie_data = []

            for symbol, inv, bp in zip(tickers, investments, buy_prices):
                current = fetch_stock_price(symbol)
                if current:
                    profit = calculate_profit_loss(inv, bp, current)
                    risk = assess_risk(profit)
                    advice = "Hold" if profit >= 0 else "Sell"
                    total_profit_loss += profit
                    data.append({
                        "Ticker": symbol,
                        "Investment": f"₹{inv:.2f}",
                        "Buy Price": f"₹{bp:.2f}",
                        "Current Price": f"₹{current:.2f}",
                        "Profit/Loss": f"₹{profit:.2f}",
                        "Risk": risk,
                        "Advice": advice
                    })
                    pie_data.append({"Stock": symbol, "Profit/Loss": profit})

                else:
                    st.warning(f"⚠️ Could not fetch data for {symbol}")

            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
                st.markdown(f"### 📊 Total Portfolio Profit/Loss: ₹{total_profit_loss:.2f}")

                # Pie Chart
                pie_df = pd.DataFrame(pie_data)
                pie_df = pie_df[pie_df["Profit/Loss"] != 0]
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.pie(pie_df['Profit/Loss'], labels=pie_df['Stock'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.subheader("📉 Profit/Loss Distribution by Stock")
                st.pyplot(fig)

# ---------------- Stock Prediction ----------------
elif menu == "📊 Stock Price Prediction":
    st.title("📊 Stock Price Prediction using LSTM")
    ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS):", value="TCS.NS")
    if st.button("Predict Price"):
        df = load_stock_data(ticker, "2020-01-01", "2024-01-01")
        if df.empty:
            st.error("No data found.")
        else:
            scaled_data, scaler = preprocess_data(df)
            X, y = create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            model = build_lstm_model((X.shape[1], 1))
            with st.spinner("Training LSTM model..."):
                model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            prediction_input = scaled_data[-60:].reshape(1, 60, 1)
            prediction = model.predict(prediction_input)
            predicted_price = scaler.inverse_transform(prediction)
            st.success(f"📈 Predicted Future Price: ₹{predicted_price[0][0]:.2f}")

# ---------------- Market News ----------------
elif menu == "📰 Market News":
    st.title("📰 Market News")
    news = fetch_stock_news()
    for i, (title, link) in enumerate(news):
        st.markdown(f"{i+1}. [{title}]({link})")

# ---------------- AI Investment Report ----------------
elif menu == "🤖 AI Investment Report":
    st.title("🤖 AI Investment Report")
    example = "AAPL, MSFT, GOOGL"
    symbols = st.text_input("Enter Stock Symbols (comma-separated):", value=example)
    if st.button("Generate Report") and symbols:
        with st.spinner("Generating report..."):
            prompt = f"""
            You're an expert financial analyst. Give a detailed investment analysis and strategic recommendation 
            for the following companies: {symbols}. Include market trends, financial health, growth prospects, 
            risks, and a final recommendation.
            """
            try:
                report = groq_llm.complete(prompt)
                st.markdown(report)
                st.success("✅ Report Ready!")
                st.download_button("📥 Download Report", report, file_name="investment_report.md")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- Best Stocks to Buy ----------------
elif menu == "📌 Best Stocks to Buy":
    st.title("📌 AI-Picked Best Stocks to Buy Now")

    if st.button("🔍 Get Recommendations"):
        with st.spinner("Analyzing market data and trends..."):
            prompt = """
            As an expert financial advisor, analyze current global market conditions, economic indicators, and 
            company fundamentals to provide a list of 5-7 top stocks to buy right now. Ensure diversity across 
            sectors. For each stock, include:
            - Ticker and Company Name
            - Sector
            - Reason for Recommendation
            - Short-Term and Long-Term Outlook

            Keep your recommendations updated with April 2025 trends and macroeconomic factors.
            """
            try:
                recommendations = groq_llm.complete(prompt)
                st.markdown(recommendations)
                st.success("✅ Recommendations Ready!")
                st.download_button("📥 Download Stock List", recommendations, file_name="best_stocks_to_buy.md")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ---------------- Stock Screener ----------------
elif menu == "📋 Stock Screener":
    st.title("📋 NIFTY Stock Screener")
    st.markdown("Analyze and rank top NIFTY stocks by valuation and technical indicators.")

    nifty_tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "ITC.NS", "LT.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "SBIN.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "AXISBANK.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "NTPC.NS", "NESTLEIND.NS", "ADANIENT.NS", "TITAN.NS",
    "ONGC.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "COALINDIA.NS", "TECHM.NS",
    "BAJAJ-AUTO.NS", "DIVISLAB.NS", "GRASIM.NS", "UPL.NS", "CIPLA.NS"
    ]

    screener_data = []

    for symbol in nifty_tickers:
        stock = yf.Ticker(symbol)
        info = stock.info
        try:
            price = info.get("currentPrice", 0)
            pe = info.get("trailingPE", 0)
            pb = info.get("priceToBook", 0)
            dividend_yield = info.get("dividendYield", 0) or 0

            df = stock.history(period="1y")
            if not df.empty:
                sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
                sma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
                trend = "Bullish" if sma_50 > sma_200 else "Bearish"
            else:
                sma_50 = sma_200 = trend = "N/A"

            score = (1/(pe+1e-5)) + (dividend_yield * 100) - pb + (1 if trend == "Bullish" else 0)
            screener_data.append([symbol, price, pe, pb, dividend_yield, trend, round(score, 2)])
        except Exception as e:
            continue

    df_screener = pd.DataFrame(screener_data, columns=["Ticker", "Price", "P/E", "P/B", "Dividend Yield", "Trend", "Score"])
    df_screener = df_screener.sort_values(by="Score", ascending=False)
    st.dataframe(df_screener)

    st.download_button("📥 Download Screener Results", df_screener.to_csv(index=False), file_name="nifty_stock_screener.csv")

# ---------------- Chat Interface ----------------
elif menu == "💬 Chat with Analyst":
    st.title("💬 Chat with AI Financial Analyst")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    user_input = st.chat_input("Ask a question about stocks, sectors, or investing...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.spinner("Analyzing..."):
                prompt = f"You are a financial expert. {user_input}"
                response = groq_llm.complete(prompt)

            st.session_state.chat_history.append(("assistant", response))
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------------- 2-Year Highlight Chart ----------------
elif menu == "📊 2-Year Stock Chart":
    st.title("📊 2-Year Stock Chart ")

    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, INFY.NS):", "INFY.NS")

    if st.button("Show Chart"):
        data = yf.download(ticker, period="2y", interval="1d")

        if data.empty:
            st.warning("No data found for this ticker.")
        else:
            # Add SMA 50
            data['SMA50'] = data['Close'].rolling(window=50).mean()

            fig = go.Figure()

            # Full candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlesticks",
                increasing_line_color='green',
                decreasing_line_color='red'
            ))

            # SMA 50 line
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA50'],
                line=dict(color='blue', width=1),
                name="SMA 50"
            ))

            fig.update_layout(
                title=f"{ticker} - 2 Year Chart ",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(x=0.01, y=0.99),
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

# ---------------- Live Crypto Charts ----------------
elif menu == "₿ Live Crypto Charts":
    st.title("📉 Live Crypto Chart")

    import yfinance as yf
    import matplotlib.pyplot as plt

    crypto_symbol = st.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"])
    period = st.selectbox("Select Time Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=2)

    if st.button("Show Chart"):
        data = yf.download(crypto_symbol, period=period, interval='1d')

        if data.empty:
            st.warning("No data found. Try a different coin or time period.")
        else:
            st.subheader(f"{crypto_symbol} Closing Price Chart")
            plt.figure(figsize=(10, 4))
            plt.plot(data.index, data['Close'], color='orange', linewidth=2)
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.title(f"{crypto_symbol} Price Over {period}")
            plt.grid(True)
            st.pyplot(plt)
