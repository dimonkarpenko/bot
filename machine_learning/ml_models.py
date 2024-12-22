import numpy as np
import pandas as pd
import logging
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from services.binance_client import BinanceClient
from indicators.signal_generator import generate_signal
from risk_managment.risk_managment import RiskManagement
from config.logger_config import logger

# Binance API setup
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
client = Client(API_KEY, API_SECRET)

# Ініціалізація Binance Client
binance_client = BinanceClient(use_testnet=True)

# Ініціалізація ризик-менеджменту
risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)

# Function to fetch historical market data
def fetch_historical_data(symbol, interval="1h", lookback="1 year ago UTC"):
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
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def prepare_data(df, feature_column="close", look_back=120):
    if df.empty or len(df) < look_back:
        logger.error(f"Недостатньо даних для підготовки. Потрібно мінімум {look_back} значень.")
        return None, None, None

    if df[feature_column].isnull().all():
        logger.error(f"Стовпець {feature_column} містить лише NaN або порожні значення.")
        return None, None, None

    try:
        data = df[feature_column].dropna().values.reshape(-1, 1)
        logger.info(f"Кількість даних для підготовки: {len(data)}")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        logger.info(f"Scaled data: {scaled_data[:5]}...")

        if len(scaled_data) <= look_back:
            logger.error(f"Недостатньо даних для створення X та y. Довжина scaled_data: {len(scaled_data)}, look_back: {look_back}.")
            return None, None, None

        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
            y.append(scaled_data[i, 0])

        if len(X) == 0 or len(y) == 0:
            logger.error("X або y залишилися порожніми після циклу. Перевірте дані та логіку створення.")
            return None, None, None

        X = np.array(X)
        y = np.array(y)
        logger.info(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaler
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None, None


# LSTM Model for Price Prediction
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))  # збільшено кількість одиниць
    model.add(LSTM(units=100, return_sequences=False))  # збільшено кількість одиниць
    model.add(Dense(units=50))  # збільшено кількість одиниць
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# XGBoost for Signal Classification
def build_xgboost_model(X, y):
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        logger.error("Недостатньо даних для навчання XGBoost моделі.")
        return None

    try:
        logger.info("Building XGBoost model")
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
        logger.info(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred)}")
        return model
    except Exception as e:
        logger.error(f"Error building XGBoost model: {e}")
        return None

# Integration with Trading Loop
# Integration with Trading Loop
def integrate_with_trading(df, symbol, lstm_model, xgboost_model, scaler, look_back=120):
    if df.empty or len(df) < look_back:
        logger.error("Не вистачає даних для прогнозування.")
        return

    try:
        latest_data = df[-look_back:]
        scaled_latest = scaler.transform(latest_data["close"].values.reshape(-1, 1))
        lstm_input = np.reshape(scaled_latest, (1, look_back, 1))

        predicted_price = scaler.inverse_transform(lstm_model.predict(lstm_input))
        logger.info(f"Predicted Price: {predicted_price[0][0]}")

        df["returns"] = df["close"].pct_change()
        df["volume"] = df["volume"].shift(-1)
        df.dropna(inplace=True)

        if len(df) == 0:
            logger.error("DataFrame пустий після обробки.")
            return

        X_xgb = df[["returns", "volume"]].values
        signals = xgboost_model.predict(X_xgb[-1].reshape(1, -1))

        price_threshold = 0.0002
        current_price = df["close"].iloc[-1]
        predicted = predicted_price[0][0]

        if signals[0] == 1 and current_price < predicted * (1 - price_threshold):
            signal = 'buy'
        elif signals[0] == 0 and current_price > predicted * (1 + price_threshold):
            signal = 'sell'
        else:
            signal = 'hold'

        logger.info(f"Signal: {signal}, Current Price: {current_price}, Predicted Price: {predicted}")

        if signal in ["buy", "sell"]:
            entry_price = current_price
            stop_loss_price = entry_price * (1 - risk_manager.risk_per_trade)
            position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price)
            take_profit_price = risk_manager.calculate_take_profit_price(entry_price)

            logger.info(f"Сигнал: {signal}, Entry Price: {entry_price}, Stop Loss: {stop_loss_price}, Take Profit: {take_profit_price}, Position Size: {position_size}")

            if signal == "buy":
                order = binance_client.place_order(symbol, "BUY", position_size, entry_price)
            elif signal == "sell":
                order = binance_client.place_order(symbol, "SELL", position_size, entry_price)

            if order:
                order_id = order['orderId']
                trade_result = binance_client.get_trade_result(symbol, order_id)

                if trade_result:
                    binance_client.account_status()
                    logger.info(f"Оновлений баланс після угоди: {trade_result}")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}")
