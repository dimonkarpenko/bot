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
def fetch_historical_data(symbol, interval="1d", lookback="1 year ago UTC"):
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

# Функція для передбачення ціни за допомогою LSTM# Модель LSTM
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dense(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Функція для передбачення ціни за допомогою LSTM
def predicted_lstm(lstm_model, df, look_back=120):
    try:
        if lstm_model is None:
            logger.error("Модель LSTM не визначена.")
            return None
        
        if df.empty or len(df) < look_back:
            logger.error("Недостатньо даних для передбачення LSTM.")
            return None

        latest_data = df[-look_back:]["close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_latest = scaler.fit_transform(latest_data)
        lstm_input = np.reshape(scaled_latest, (1, look_back, 1))

        predicted_price_of_lstm = lstm_model.predict(lstm_input)
        predicted_price_of_lstm = scaler.inverse_transform(predicted_price_of_lstm)

        logger.info(f"Передбачена ціна за допомогою LSTM: {predicted_price_of_lstm[0][0]}")
        return predicted_price_of_lstm[0][0]

    except Exception as e:
        logger.error(f"Помилка в передбаченні LSTM: {e}")
        return None


# Функція для передбачення ціни за допомогою XGBoost
def predicted_xgboost(xgboost_model, df):
    try:
        if df.empty:
            logger.error("Недостатньо даних для передбачення XGBoost.")
            return None

        df["returns"] = df["close"].pct_change()
        df.dropna(inplace=True)
        X_xgb = df[["returns", "volume"]].values

        predicted_price_of_xgb = model.predict(X_xgb[-1].reshape(1, -1))

        logger.info(f"Передбачена ціна за допомогою XGBoost: {predicted_price_of_xgb[0]}")
        return predicted_price_of_xgb[0]

    except Exception as e:
        logger.error(f"Error in XGBoost prediction: {e}")
        return None

# Загальна функція для передбачення
def check_predicted(lstm_model, xgboost_model, df, look_back=120):
    try:
        # Check if there is enough data for predictions
        if len(df) < look_back:
            logger.error(f"Insufficient data for prediction. Need at least {look_back} data points.")
            return None
        
        # Предсказание цены с использованием LSTM
        predicted_lstm_price = predicted_lstm(lstm_model, df, look_back)
        
        # Предсказание цены с использованием XGBoost
        predicted_xgboost_price = predicted_xgboost(xgboost_model, df)

        if predicted_lstm_price is None or predicted_xgboost_price is None:
            logger.error("Не вдалося здійснити передбачення з однієї або обох моделей.")
            return None

        # Вычисление среднего значения предсказанных цен
        predicted_price_of_crypto = (predicted_lstm_price + predicted_xgboost_price) / 2

        logger.info(f"Середнє передбачення ціни: {predicted_price_of_crypto}")

        # Сохранение результата в CSV
        result_df = pd.DataFrame({
            "predicted_price": [predicted_price_of_crypto],
            "timestamp": [pd.to_datetime('now')]
        })
        result_df.to_csv("predicted_prices.csv", mode="a", header=False, index=False)

        return predicted_price_of_crypto

    except Exception as e:
        logger.error(f"Error in combined prediction: {e}")
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
