from machine_learning.ml_models import (
    fetch_historical_data,
    prepare_data,
    build_lstm_model,
    build_xgboost_model,
    integrate_with_trading
)
from services.binance_client import BinanceClient
from indicators.signal_generator import generate_signal
from risk_managment.risk_managment import RiskManagement
from config.logger_config import logger
import numpy as np
import pandas as pd
import time
import sqlite3

# Ініціалізація Binance Client
binance_client = BinanceClient(use_testnet=True)

# Ініціалізація ризик-менеджменту
risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)

# Підключення до бази даних для логів угод
conn = sqlite3.connect('trading_logs.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        symbol TEXT,
        action TEXT,
        position_size REAL,
        entry_price REAL,
        result REAL
    )
''')
conn.commit()

def log_trade(symbol, action, position_size, entry_price, result):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(
        'INSERT INTO trades (timestamp, symbol, action, position_size, entry_price, result) VALUES (?, ?, ?, ?, ?, ?)',
        (timestamp, symbol, action, position_size, entry_price, result)
    )
    conn.commit()

def trading_loop():
    # Підготовка даних
    df = fetch_historical_data(symbol="DOGEUSDT", interval="1h", lookback="1 month ago UTC")
    if df is None:
        logger.error("Не вдалося завантажити дані для навчання.")
        return

    X, y, scaler = prepare_data(df)
    
    # Навчання LSTM
    lstm_model = build_lstm_model((X.shape[1], 1))
    lstm_model.fit(X, y, epochs=10, batch_size=32)
    logger.info("LSTM модель навчена.")
    
    # Навчання XGBoost
    df["returns"] = df["close"].pct_change()
    df["volume"] = df["volume"].shift(-1)
    df.dropna(inplace=True)
    X_xgb = df[["returns", "volume"]].values
    y_xgb = np.where(df["returns"] > 0, 1, 0)
    xgboost_model = build_xgboost_model(X_xgb, y_xgb)
    logger.info("XGBoost модель навчена.")
    
    while True:
        try:
            # Отримання нових даних
            df = fetch_historical_data(symbol="DOGEUSDT", interval="1h", lookback="60 hours ago UTC")
            if df is None:
                continue
            
            # Інтеграція з торговою логікою
            integrate_with_trading(df, lstm_model, xgboost_model, scaler)

            # Затримка між ітераціями
            time.sleep(60)
        
        except Exception as e:
            logger.error(f"Помилка в циклі: {e}")
            time.sleep(60)

if __name__ == "__main__":
    trading_loop()


# from machine_learning.ml_models import fetch_historical_data, integrate_with_trading
# from services.binance_client import BinanceClient
# from indicators.signal_generator import generate_signal
# from risk_managment.risk_managment import RiskManagement
# from config.logger_config import logger
# import time
# import sqlite3

# # class RiskManagement:
#     # def __init__(self, account_balance, risk_per_trade=0.01, max_trade_size=0.1):
#     #     """
#     #     Ініціалізація модуля ризик-менеджменту.

#     #     :param account_balance: Баланс рахунку.
#     #     :param risk_per_trade: Частка ризику на одну угоду (0.01 = 1% від балансу).
#     #     :param max_trade_size: Максимальна частка балансу, яку можна використати на угоду.
#     #     """
#     #     self.account_balance = account_balance
#     #     self.risk_per_trade = risk_per_trade
#     #     self.max_trade_size = max_trade_size

#     # def calculate_position_size(self, entry_price, stop_loss_price):
#     #     """
#     #     Розрахунок розміру позиції на основі ризику.

#     #     :param entry_price: Ціна входу в угоду.
#     #     :param stop_loss_price: Ціна стоп-лосс.
#     #     :return: Рекомендований розмір позиції.
#     #     """
#     #     risk_amount = self.account_balance * self.risk_per_trade
#     #     stop_loss_distance = abs(entry_price - stop_loss_price)
#     #     position_size = risk_amount / stop_loss_distance
#     #     max_position_size = self.account_balance * self.max_trade_size / entry_price

#     #     return min(position_size, max_position_size)

#     # def calculate_take_profit_price(self, entry_price, risk_reward_ratio=2):
#     #     """
#     #     Розрахунок тейк-профіт ціни на основі співвідношення ризик/прибуток.

#     #     :param entry_price: Ціна входу.
#     #     :param risk_reward_ratio: Співвідношення ризик/прибуток.
#     #     :return: Ціна тейк-профіту.
#     #     """
#     #     stop_loss_distance = entry_price * self.risk_per_trade
#     #     return entry_price + (stop_loss_distance * risk_reward_ratio)

#     # def trailing_stop(self, current_price, trailing_percentage):
#     #     """
#     #     Розрахунок ціни трейлінг-стопу.

#     #     :param current_price: Поточна ціна активу.
#     #     :param trailing_percentage: Відсоток трейлінг-стопу (0.05 = 5%).
#     #     :return: Ціна трейлінг-стопу.
#     #     """
#     #     trailing_stop_price = current_price * (1 - trailing_percentage)
#     #     return trailing_stop_price

#     # def update_account_balance(self, trade_result):
#     #     """
#     #     Оновлення балансу рахунку після угоди.

#     #     :param trade_result: Результат угоди (позитивний чи негативний).
#     #     """
#     #     self.account_balance += trade_result

# # Ініціалізація Binance Client
# binance_client = BinanceClient(use_testnet=True)

# # Ініціалізація ризик-менеджменту
# risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)

# # Підключення до бази даних для логів угод
# conn = sqlite3.connect('trading_logs.db')
# cursor = conn.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT, action TEXT, position_size REAL, entry_price REAL, result REAL)''')
# conn.commit()

# def log_trade(symbol, action, position_size, entry_price, result):
#     timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#     cursor.execute('INSERT INTO trades (timestamp, symbol, action, position_size, entry_price, result) VALUES (?, ?, ?, ?, ?, ?)', (timestamp, symbol, action, position_size, entry_price, result))
#     conn.commit()


# def trading_loop():
    
#     lstm_model = build_lstm_model((X.shape[1], 1))
#     lstm_model.fit(X, y, epochs=10, batch_size=32)

#         # Prepare features for XGBoost
#     df["returns"] = df["close"].pct_change()
#     df["volume"] = df["volume"].shift(-1)
#     df.dropna(inplace=True)
#     X_xgb = df[["returns", "volume"]].values
#     y_xgb = np.where(df["returns"] > 0, 1, 0)

#         # Train XGBoost model
#     xgboost_model = build_xgboost_model(X_xgb, y_xgb)

#     while True:
#         try:
#             # Збір ринкових даних
#             candles = binance_client.get_candlestick_data(symbol="DOGEUSDT", interval="1h", limit=1000)
#             close_prices = [float(candle[4]) for candle in candles]  # Ціна закриття
#             logger.info(f"Отримані дані про свічки: {close_prices}")


#             # if signal in ["buy", "sell"]:
#             #     # Управління ризиками
#             #     entry_price = close_prices[-1]
#             #     stop_loss_price = entry_price * (1 - risk_manager.risk_per_trade)
#             #     position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price)
#             #     take_profit_price = risk_manager.calculate_take_profit_price(entry_price)

#             #     logger.info(f"Сигнал: {signal}, Ціна входу: {entry_price}, Стоп-лосс: {stop_loss_price}, Тейк-профіт: {take_profit_price}, Розмір позиції: {position_size}")

#             #     # Виконання угоди
#             #     if signal == "buy":
#             #         binance_client.place_order("DOGEUSDT", "BUY", position_size, entry_price)
#             #     elif signal == "sell":
#             #         binance_client.place_order("DOGEUSDT", "SELL", position_size, entry_price)

#             #     # Симуляція прибутку та оновлення балансу
#             #     trade_result = 50  # Припустимо, отриманий прибуток
#             #     risk_manager.update_account_balance(trade_result)

#             #     # Логування угоди
#             #     log_trade("DOGEUSDT", signal, position_size, entry_price, trade_result)
#             #     logger.info(f"Оновлений баланс рахунку: {risk_manager.account_balance:.2f}")

#             # Затримка між ітераціями
#             time.sleep(60)

#         except Exception as e:
#             logger.error(f"Помилка: {e}")
#             time.sleep(60)

# if __name__ == "__main__":
#     trading_loop()
