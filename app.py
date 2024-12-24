from flask import Flask, jsonify, request, send_file, render_template, redirect, url_for
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def fetch_data(url, csv_file):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://finance.yahoo.com/",
        "Connection": "keep-alive",
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from Yahoo Finance")

    soup = BeautifulSoup(response.content, 'lxml')
    all_info = soup.find_all('tr', class_='yf-j5d1ld')

    if not all_info:
        raise Exception("No data found for the provided URL")

    dates, open_prices, high_prices, low_prices, close_prices, adj_close_prices, volumes = [], [], [], [], [], [], []

    for row in all_info[1:]:
        row_data = row.find_all('td', class_='yf-j5d1ld')
        if len(row_data) < 7:
            continue

        try:
            date = datetime.strptime(row_data[0].text.strip(), '%b %d, %Y')
            dates.append(date.strftime('%m/%d/%Y'))
            open_prices.append(float(row_data[1].text.replace(',', '').strip()))
            high_prices.append(float(row_data[2].text.replace(',', '').strip()))
            low_prices.append(float(row_data[3].text.replace(',', '').strip()))
            close_prices.append(float(row_data[4].text.replace(',', '').strip()))
            adj_close_prices.append(float(row_data[5].text.replace(',', '').strip()))
            volumes.append(int(row_data[6].text.replace(',', '').strip()))
        except ValueError:
            continue

    if not dates:
        raise Exception("No valid data could be extracted from the page")

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Adjusted Close", "Volume"])
        for i in range(len(dates)):
            writer.writerow([dates[i], open_prices[i], high_prices[i], low_prices[i], close_prices[i], adj_close_prices[i], volumes[i]])

@app.route('/search_companies', methods=['POST'])
def search_companies():
    query = request.form.get('search_query')
    if not query:
        return render_template('index.html', companies=[], fetch_error=None, search_error="Query is required")

    api_url = f"https://financialmodelingprep.com/api/v3/search?query={query}&apikey={API_KEY}"
    response = requests.get(api_url)

    if response.status_code != 200:
        return render_template('index.html', companies=[], fetch_error=None, search_error="Failed to fetch data from the API")

    companies = response.json()
    # Ensure all required fields are passed to the template
    processed_companies = [
        {
            'symbol': company.get('symbol'),
            'name': company.get('name'),
            'currency': company.get('currency'),
            'stockExchange': company.get('stockExchange'),
            'exchangeShortName': company.get('exchangeShortName')
        } for company in companies
    ]
    return render_template('index.html', companies=processed_companies, fetch_error=None, search_error=None)

@app.route('/fetch_and_save', methods=['POST'])
def fetch_and_save():
    try:
        ticker = request.form.get('ticker', '').upper()
        if not ticker:
            return jsonify({"error": "Ticker symbol is required"}), 400

        end_timestamp = int(datetime.now().timestamp())
        ten_years_ago = datetime.now().replace(year=datetime.now().year - 10)
        start_timestamp_10_years = int(ten_years_ago.timestamp())

        recent_url = f"https://finance.yahoo.com/quote/{ticker}/history/?period1={end_timestamp - 31536000}&period2={end_timestamp}"
        fetch_data(recent_url, "stock_data.csv")

        ten_years_url = f"https://finance.yahoo.com/quote/{ticker}/history/?period1={start_timestamp_10_years}&period2={end_timestamp}"
        fetch_data(ten_years_url, "all_years_stock_data.csv")

        return redirect(url_for('show_analysis'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/show_analysis')
def show_analysis():
    try:
        HAL = pd.read_csv("stock_data.csv")
        HAL = HAL.iloc[::-1]

        ma_day = [10, 20, 50]
        for ma in ma_day:
            column_name = f"MA for {ma} days"
            HAL[column_name] = HAL['Adjusted Close'].rolling(window=ma).mean()

        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        HAL['RSI'] = calculate_rsi(HAL['Adjusted Close'], window=14).dropna()

        def ema(series, span):
            return series.ewm(span=span, adjust=False).mean()

        def calculate_macd(df):
            ema12 = ema(df['Adjusted Close'], span=12)
            ema26 = ema(df['Adjusted Close'], span=26)
            macd = ema12 - ema26
            macd_signal = ema(macd, span=9)
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist

        macd, macd_signal, macd_hist = calculate_macd(HAL)

        def calculate_obv(df):
            obv = [0]
            for i in range(1, len(df)):
                if df['Adjusted Close'][i] > df['Adjusted Close'][i - 1]:
                    obv.append(obv[-1] + df['Volume'][i])
                elif df['Adjusted Close'][i] < df['Adjusted Close'][i - 1]:
                    obv.append(obv[-1] - df['Volume'][i])
                else:
                    obv.append(obv[-1])
            return pd.Series(obv, index=df.index)

        HAL['OBV'] = calculate_obv(HAL)

        plots = []

        def save_plot(fig):
            plot_bytes = io.BytesIO()
            fig.savefig(plot_bytes, format='png', bbox_inches='tight')
            plot_bytes.seek(0)
            plot_base64 = base64.b64encode(plot_bytes.getvalue()).decode('utf8')
            plots.append(plot_base64)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 6))
        HAL.plot(x='Date', y='Adjusted Close', ax=ax)
        ax.set_title("Adjusted Close Price")
        save_plot(fig)

        fig, ax = plt.subplots(figsize=(14, 6))
        HAL.plot(x='Date', y='Volume', ax=ax)
        ax.set_title("Volume")
        save_plot(fig)

        fig, ax = plt.subplots(figsize=(14, 6))
        HAL.plot(x='Date', y=['Adjusted Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'], ax=ax)
        ax.set_title("Moving Averages")
        save_plot(fig)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(HAL['Date'], HAL['RSI'], label='RSI')
        ax.axhline(80, color='red', linestyle='--', label='Overbought')
        ax.axhline(20, color='red', linestyle='--', label='Oversold')
        ax.set_title("RSI")
        ax.legend()
        save_plot(fig)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(HAL['Date'], macd, label='MACD', color='green')
        ax.plot(HAL['Date'], macd_signal, label='Signal Line', color='orange')
        ax.bar(HAL['Date'], macd_hist, label='MACD Histogram', color='purple', alpha=0.5)
        ax.axhline(0, color='red', linestyle='--', label='Base Line')
        ax.set_title("MACD Analysis")
        ax.legend()
        save_plot(fig)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(HAL['Date'], HAL['OBV'], color='orange', label='OBV')
        ax.set_title("On-Balance Volume")
        ax.legend()
        save_plot(fig)

        return render_template('analysis.html', plots=plots)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

lr = None

lr = None
train_score = None
test_score = None

@app.route('/train', methods=['GET'])
def train_model():
    try:
        # Load and preprocess data
        data = pd.read_csv("all_years_stock_data.csv")
        data.dropna(inplace=True)

        # Feature engineering
        data['Open_Close'] = (data['Open'] - data['Adjusted Close']) / data['Open']
        data['High_Low'] = (data['High'] - data['Low']) / data['Low']
        data['Returns'] = data['Adjusted Close'].pct_change()
        data.dropna(inplace=True)

        # Prepare training and testing data
        X = np.array(data['Open']).reshape(-1, 1)
        Y = np.array(data['Adjusted Close']).reshape(-1, 1)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.6, random_state=42)

        # Train Linear Regression model
        global lr, train_score, test_score
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        train_score = round(lr.score(x_train, y_train), 5)
        test_score = round(lr.score(x_test, y_test), 5)

        # Render the template with accuracy values
        return render_template('prediction_form.html', accuracy_train=f"{train_score}", accuracy_test=f"{test_score}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global train_score, test_score
        if lr is None:
            return jsonify({"error": "Model is not trained yet. Train the model first."}), 400

        open_price = float(request.form['open_price'])
        predicted_close = lr.predict(np.array([[open_price]]))[0][0]

        return render_template('prediction_form.html', prediction=f"Predicted Adjusted Close Price: {predicted_close:.2f}", accuracy_train=f"{train_score}", accuracy_test=f"{test_score}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


