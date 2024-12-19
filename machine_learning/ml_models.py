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

# Binance API setup
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
client = Client(API_KEY, API_SECRET)

# Function to fetch historical market data
def fetch_historical_data(symbol="BTCUSDT", interval="1h", lookback="1 month ago UTC"):
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
    # Переконайтеся, що мітки цілі contain лише 0 та 1

    # Заміна значення -1 на 0 для numpy масиву
    X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

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
    latest_data = df[-look_back:]
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
    df.dropna(inplace=True)
    features = df[["returns", "volume"]].values
    signals = xgboost_model.predict(features[-1].reshape(1, -1))

    # Логіка прийняття рішення
    if signals[0] == 1 and df["close"].iloc[-1] < predicted_price[0][0]:
        print("Buy Signal")
    elif signals[0] == -1 and df["close"].iloc[-1] > predicted_price[0][0]:
        print("Sell Signal")
    else:
        print("Hold")

# Fetch historical data and train models
df = fetch_historical_data()
X, y, scaler = prepare_data(df)

# Train LSTM model
lstm_model = build_lstm_model((X.shape[1], 1))
lstm_model.fit(X, y, epochs=10, batch_size=32)

# Prepare features for XGBoost
df["returns"] = df["close"].pct_change()
df.dropna(inplace=True)
X_xgb = df[["returns", "volume"]].values
y_xgb = np.where(df["returns"] > 0, 1, 0)

# Train XGBoost model
xgboost_model = build_xgboost_model(X_xgb, y_xgb)

# Uncomment to start trading loop
    # Integrate with trading
def trading_loop():
    while True:
        df = fetch_historical_data()
        integrate_with_trading(df, lstm_model, xgboost_model, scaler)
        time.sleep(60)
trading_loop()
#  trading_loop()
