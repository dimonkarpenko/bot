import os
import time
import logging
import sqlite3
import numpy as np
import telebot
import requests
import threading
from dotenv import load_dotenv
from machine_learning.ml_models import (
    fetch_historical_data,
    prepare_data,
    build_lstm_model,
    build_xgboost_model,
    integrate_with_trading,
)
from services.binance_client import BinanceClient
from indicators.signal_generator import generate_signal
from risk_managment.risk_managment import RiskManagement

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", ".env")
load_dotenv(dotenv_path)

# Initialize Binance Client and other components
binance_client = BinanceClient(use_testnet=True)
risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)
TELEGRAM_BOT_TOKEN = os.getenv("Tg_Bot_API")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# User state storage
data_lock = threading.Lock()
user_data = {}

# Database setup
conn = sqlite3.connect("trading_logs.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        symbol TEXT,
        action TEXT,
        position_size REAL,
        entry_price REAL,
        result REAL
    )"""
)
conn.commit()

def log_trade(symbol, action, position_size, entry_price, result):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO trades (timestamp, symbol, action, position_size, entry_price, result) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, symbol, action, position_size, entry_price, result),
    )
    conn.commit()

def is_valid_symbol(symbol):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        symbols = [s["symbol"] for s in data["symbols"]]
        return symbol in symbols
    except Exception as e:
        logger.error(f"Error while validating symbol: {e}")
        return False

@bot.message_handler(commands=["start"])
def start_command(message):
    chat_id = message.chat.id
    with data_lock:
        user_data[chat_id] = {"crypto_pair": None, "trading_amount": None}
    bot.send_message(chat_id, "Welcome! Please enter a cryptocurrency pair (e.g., BTCUSDT).")

@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get("crypto_pair") is None)
def set_crypto_pair(message):
    chat_id = message.chat.id
    crypto_pair = message.text.upper().strip()

    if not is_valid_symbol(crypto_pair):
        bot.send_message(chat_id, "Invalid cryptocurrency pair. Please enter a valid pair supported on Binance.")
        return

    with data_lock:
        user_data[chat_id]["crypto_pair"] = crypto_pair

    bot.send_message(chat_id, f"Your chosen pair is: {crypto_pair}")
    bot.send_message(chat_id, "Now, please enter the trading amount.")

@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get("crypto_pair") is not None and user_data[message.chat.id].get("trading_amount") is None)
def set_trading_amount(message):
    chat_id = message.chat.id
    try:
        trading_amount = int(message.text.strip())
        if trading_amount <= 0:
            raise ValueError("Amount must be greater than 0.")

        with data_lock:
            user_data[chat_id]["trading_amount"] = trading_amount

        crypto_pair = user_data[chat_id]["crypto_pair"]
        if crypto_pair is None:
            bot.send_message(chat_id, "Please set a valid cryptocurrency pair first.")
            return

        current_price = safe_get_price(crypto_pair)
        if current_price is None:
            bot.send_message(chat_id, "Failed to fetch the current price. Please try again later.")
            return

        bot.send_message(chat_id, f"Trading details:\nPair: {crypto_pair}\nCurrent Price: {current_price:.5f}\nTrading Amount: {trading_amount}")
        logger.info(f"User {chat_id}: Pair: {crypto_pair}, Amount: {trading_amount}, Price: {current_price}.")
    except ValueError:
        bot.send_message(chat_id, "Please enter a valid numerical amount greater than 0.")
    except Exception as e:
        logger.error(f"Error in set_trading_amount: {e}")
        bot.send_message(chat_id, "An error occurred. Please try again.")


@bot.message_handler(commands=["start_trading"])
def start_trading(message):
    chat_id = message.chat.id

    with data_lock:
        crypto_pair = user_data.get(chat_id, {}).get("crypto_pair")
        trading_amount = user_data.get(chat_id, {}).get("trading_amount")

    if crypto_pair is None:
        bot.send_message(chat_id, "Please set a cryptocurrency pair using the /start command.")
        return

    if trading_amount is None:
        bot.send_message(chat_id, "Please set a trading amount before starting.")
        return

    try:
        bot.send_message(chat_id, "Trading bot is starting! Trading initiated...")
        logger.info(f"Trading started for {chat_id} with pair {crypto_pair} and amount {trading_amount}")
        trading_loop(chat_id)
    except Exception as e:
        logger.error(f"Error in start_trading: {e}")
        bot.send_message(chat_id, "An error occurred while starting the trading bot. Please try again.")

def safe_get_price(crypto_pair):
    if crypto_pair and is_valid_symbol(crypto_pair):
        try:
            return binance_client.get_current_price(crypto_pair)
        except Exception as e:
            logger.error(f"Error fetching price for {crypto_pair}: {e}")
            return None
    else:
        logger.error(f"Invalid or missing crypto pair: {crypto_pair}")
        return None


def trading_loop(chat_id):
    
    logger.info(f"Starting trading loop for chat_id {chat_id}")
    while True:
        try:
            with data_lock:
                crypto_pair = user_data.get(chat_id, {}).get("crypto_pair")

            if not crypto_pair:
                logger.warning(f"Crypto pair for {chat_id} is not set. Skipping iteration.")
                time.sleep(60)
                continue

            # Fetch historical data and check for empty DataFrame
            logger.info(f"Fetching historical data for {crypto_pair}")
            df = fetch_historical_data(symbol=crypto_pair, interval="1h", lookback="3 month ago UTC")
            if df is None or df.empty:
                logger.warning(f"No data fetched for {crypto_pair}. Skipping iteration.")
                time.sleep(60)
                continue


            logger.info(f"Data fetched for {crypto_pair}, shape: {df.shape}")
            X, y, scaler = prepare_data(df)

            logger.info(f"Prepared data: X shape = {X.shape}, y shape = {y.shape}")


            if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
                logger.warning(f"Insufficient data for training models. Skipping iteration.")
                time.sleep(60)
                continue

            lstm_model = build_lstm_model((X.shape[1], 1))
            lstm_model.fit(X, y, epochs=10, batch_size=32)

            # Prepare data for XGBoost model
            df["returns"] = df["close"].pct_change()
            df["volume"] = df["volume"].shift(-1)
            df.dropna(inplace=True)
            X_xgb = df[["returns", "volume"]].values
            y_xgb = np.where(df["returns"] > 0, 1, 0)

            if len(X_xgb) == 0 or len(y_xgb) == 0:
                logger.warning(f"Insufficient data for XGBoost. Skipping iteration.")
                time.sleep(60)
                continue

            xgboost_model = build_xgboost_model(X_xgb, y_xgb)

            try:
                integrate_with_trading(symbol=crypto_pair, df, lstm_model, xgboost_model, scaler)
            except Exception as e:
                logger.error(f"Error in integrate_with_trading: {e}")
                time.sleep(60)
                continue

            time.sleep(60)

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    logger.info("Starting trading bot.")
    try:
        telegram_thread = threading.Thread(target=bot.polling, kwargs={"none_stop": True, "timeout": 20, "interval": 1})
        telegram_thread.start()
        logger.info("Telegram bot started.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
