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
from config.logger_config import logger

# Завантаження змінних середовища з .env файлу
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", ".env")
load_dotenv(dotenv_path)

# Ініціалізація Binance Client та інших компонентів
binance_client = BinanceClient(use_testnet=True)
risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)
TELEGRAM_BOT_TOKEN = os.getenv("Tg_Bot_API")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Словник для збереження стану користувачів
user_data = {}

# Підключення до бази даних для логів угод
conn = sqlite3.connect("trading_logs.db")
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

# Функція для логування торгів
def log_trade(symbol, action, position_size, entry_price, result):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO trades (timestamp, symbol, action, position_size, entry_price, result) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, symbol, action, position_size, entry_price, result),
    )
    conn.commit()

# Функція для перевірки валідності криптовалютної пари
def is_valid_symbol(symbol):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        symbols = [s["symbol"] for s in data["symbols"]]
        return symbol in symbols
    except Exception as e:
        logger.error(f"Помилка під час перевірки пари на Binance: {e}")
        return False

# Обробник команди /start
@bot.message_handler(commands=["start"])
def start_command(message):
    chat_id = message.chat.id
    user_data[chat_id] = {"crypto_pair": None}
    bot.send_message(chat_id, "Ласкаво просимо! Введіть криптовалютну пару (наприклад, BTCUSDT).")

# Обробник для встановлення криптовалютної пари
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get("crypto_pair") is None)
def set_crypto_pair(message):
    chat_id = message.chat.id
    crypto_pair = message.text.upper()

    if not is_valid_symbol(crypto_pair):
        bot.send_message(chat_id, "Будь ласка, введіть коректну криптовалютну пару, що підтримується на Binance.")
        return

    user_data[chat_id]["crypto_pair"] = crypto_pair
    bot.send_message(chat_id, f"Ваша обрана пара: {crypto_pair}")
    bot.send_message(chat_id, "Тепер введіть суму для торгівлі.")
    return crypto_pair

# Функція для безпечного отримання ціни криптовалюти
def safe_get_price(crypto_pair):
    try:
        return binance_client.get_current_price(crypto_pair)
    except Exception as e:
        logger.error(f"Помилка при отриманні ціни для {crypto_pair}: {e}")
        return None

# Обробник для введення суми торгівлі
@bot.message_handler(func=lambda message: user_data.get(message.chat.id, {}).get("crypto_pair") is not None)
def set_trading_amount(message):
    chat_id = message.chat.id
    try:
        trading_amount = int(message.text)
        if trading_amount <= 0:
            bot.send_message(chat_id, "Сума має бути більше 0.")
            return

        user_data[chat_id]["trading_amount"] = trading_amount
        crypto_pair = user_data[chat_id]["crypto_pair"]
        current_price = safe_get_price(crypto_pair)

        if current_price is None:
            bot.send_message(chat_id, "Не вдалося отримати поточну ціну. Спробуйте пізніше.")
            return

        bot.send_message(chat_id, f"Обрана пара: {crypto_pair}\nПоточна ціна: {current_price:.5f}\nСума для торгівлі: {trading_amount}\nПочинаємо торг!")
        logger.info(f"Торгова пара: {crypto_pair}, сума для торгівлі: {trading_amount}, поточна ціна: {current_price}")
        trading_loop(crypto_pair)
    except ValueError:
        bot.send_message(chat_id, "Будь ласка, введіть коректну числову суму.")

# Основна функція для запуску торгового циклу
def trading_loop(crypto_pair):
    logger.info("Start Trading!!!")
    logger.info(crypto_pair)
    while True:
        try:
            # Отримання нових даних
            df = fetch_historical_data(symbol=crypto_pair, interval="1h", lookback="60 hours ago UTC")
            if df is None:
                time.sleep(60)
                continue

            # Підготовка даних для моделювання
            X, y, scaler = prepare_data(df)

            # Навчання LSTM та XGBoost
            lstm_model = build_lstm_model((X.shape[1], 1))
            lstm_model.fit(X, y, epochs=10, batch_size=32)

            df["returns"] = df["close"].pct_change()
            df["volume"] = df["volume"].shift(-1)
            df.dropna(inplace=True)
            X_xgb = df[["returns", "volume"]].values
            y_xgb = np.where(df["returns"] > 0, 1, 0)
            xgboost_model = build_xgboost_model(X_xgb, y_xgb)

            # Інтеграція з торговою логікою
            integrate_with_trading(df, lstm_model, xgboost_model, scaler)
            
            # Затримка між ітераціями
            time.sleep(60)
        
        except Exception as e:
            logger.error(f"Помилка в циклі: {e}")
            time.sleep(60)

if __name__ == "__main__":
    logger.info("Запуск торгового циклу.")
    try:
        # Запуск бота в окремому потоці
        telegram_thread = threading.Thread(target=bot.polling, kwargs={"none_stop": True, "timeout": 20, "interval": 1})
        telegram_thread.start()
        logger.info("Telegram бот запущено.")

        # Запуск торгового циклу
        trading_loop()

    except Exception as e:
        logger.error(f"Помилка: {e}")

