import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from services.binance_client import BinanceClient
from indicators.signal_generator import generate_signal
from risk_managment.risk_managment import RiskManagement
from config.logger_config import logger

from services.binance_client import BinanceClient



# Binance API setup
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
client = Client(API_KEY, API_SECRET)

# Ініціалізація Binance Client
binance_client = BinanceClient(use_testnet=True)


# Ініціалізація ризик-менеджменту
risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)

# Function to fetch historical market data
def fetch_historical_data(symbol="DOGEUSDT", interval="1h", lookback="1 month ago UTC"):
    try:
        klines = client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                           "close_time", "quote_asset_volume", "number_of_trades", 
                                           "taker_buy_base", "taker_buy_quote", "ignore"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Data Preprocessing
def prepare_data(df, feature_column="close", look_back=60):
    data = df[feature_column].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# LSTM Model for Price Prediction
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# XGBoost for Signal Classification
def build_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = np.where(y_train > 0, 1, 0)

    model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred)}")
    return model

# Integration with Trading Loop
def integrate_with_trading(df, lstm_model, xgboost_model, scaler, look_back=60):
    latest_data = df[-look_back:]  # Останні 60 значень
    if len(latest_data) < look_back:
        print("Не вистачає даних для прогнозування.")
        return
    
    scaled_latest = scaler.transform(latest_data["close"].values.reshape(-1, 1))
    lstm_input = np.reshape(scaled_latest, (1, look_back, 1))

    # Прогнозування ціни
    try:
        predicted_price = scaler.inverse_transform(lstm_model.predict(lstm_input))
        print(f"Predicted Price: {predicted_price[0][0]}")
    except Exception as e:
        print(f"Error predicting price: {e}")
        return

    # Генерація сигналу
    df["returns"] = df["close"].pct_change()
    df["volume"] = df["volume"].shift(-1)
    
    df.dropna(inplace=True)

    X_xgb = df[["returns", "volume"]].values
    y_xgb = np.where(df["returns"] > 0, 1, 0)

    signals = xgboost_model.predict(X_xgb[-1].reshape(1, -1))  # Передаємо останні дані для прогнозу

    print(f"Signal from XGBoost: {signals[0]}")  # Перевірка сигналу

    # Логіка прийняття рішення
    price_threshold = 0.002  # 0.2% зміна ціни як поріг

    signal = ''

    if signals[0] == 1 and df["close"].iloc[-1] < predicted_price[0][0] :
        signal = 'buy'
        print("Buy")
    elif signals[0] == 0 and df["close"].iloc[-1] > predicted_price[0][0]:
        signal = 'sell'
        print("Sell")
    else:
        print("Hold it")

    logger.info(signal)

    if signal in ["buy", "sell"]:
        print("Start trading...")
        
                # Управління ризиками
        entry_price = df["close"].iloc[-1]
        stop_loss_price = entry_price * (1 - risk_manager.risk_per_trade)
        position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price)
        take_profit_price = risk_manager.calculate_take_profit_price(entry_price)

        logger.info(f"Сигнал: {signal}, Ціна входу: {entry_price}, Стоп-лосс: {stop_loss_price}, Тейк-профіт: {take_profit_price}, Розмір позиції: {position_size}")

                # Виконання угоди
        if signal == "buy":
            binance_client.place_order("DOGEUSDT", "BUY", position_size, entry_price)
        elif signal == "sell":
            binance_client.place_order("DOGEUSDT", "SELL", position_size, entry_price)

                # Симуляція прибутку та оновлення балансу
        trade_result = 50  # Припустимо, отриманий прибуток
        risk_manager.update_account_balance(trade_result)

                # Логування угоди
        log_trade("DOGEUSDT", signal, position_size, entry_price, trade_result)
        logger.info(f"Оновлений баланс рахунку: {risk_manager.account_balance:.2f}")



# Main loop for training models and trading
# def trading_loop():
#     try:
#         # Fetch historical data and train models
#         df = fetch_historical_data()
#         if df is None:
#             return
        
#         X, y, scaler = prepare_data(df)

#         # Train LSTM model
        # lstm_model = build_lstm_model((X.shape[1], 1))
        # lstm_model.fit(X, y, epochs=10, batch_size=32)

        # # Prepare features for XGBoost
        # df["returns"] = df["close"].pct_change()
        # df["volume"] = df["volume"].shift(-1)
        # df.dropna(inplace=True)
        # X_xgb = df[["returns", "volume"]].values
        # y_xgb = np.where(df["returns"] > 0, 1, 0)

        # # Train XGBoost model
        # xgboost_model = build_xgboost_model(X_xgb, y_xgb)

#         # Start trading loop
#         while True:
    #         df = fetch_historical_data()
    #         if df is None:
    #             continue
    #         integrate_with_trading(df, lstm_model, xgboost_model, scaler)
    #         time.sleep(60)  # Пауза між кожним циклом
    # except KeyboardInterrupt:
    #     print("Trading loop stopped.")

# trading_loop()
