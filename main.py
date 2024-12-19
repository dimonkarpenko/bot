from machine_learning.ml_models import fetch_historical_data, integrate_with_trading
from services.binance_client import BinanceClient
from risk_managment.risk_managment import RiskManagement
from indicators.signal_generator import generate_signal
import time

def trading_loop():
    # Ініціалізація Binance Client
    binance_client = BinanceClient(use_testnet=True)

    # Ініціалізація ризик-менеджменту
    risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)

    # Ініціалізація моделей для машинного навчання
    lstm_model = None  # Вставте ваш попередньо навчену модель LSTM
    xgboost_model = None  # Вставте ваш попередньо навчену модель XGBoost
    scaler = None  # Скалер для нормалізації даних


    while True:
        try:
            # Отримання даних про свічки
            candles = binance_client.get_candlestick_data(symbol="XRPUSDT", interval="1m", limit=50)
            if candles is not None:
                close_prices = [float(candle[4]) for candle in candles]
            else:
                print("Не вдалося отримати дані про свічки.")

            # Генерація сигналу
            signal = generate_signal(close_prices)
            print(f"Згенерований сигнал: {signal}")

            if signal in ["buy", "sell"]:
                entry_price = close_prices[-1]
                stop_loss_price = risk_manager.calculate_stop_loss(entry_price)
                position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price)
                take_profit_price = risk_manager.calculate_take_profit_price(entry_price)

                print(f"Сигнал: {signal}")
                print(f"Ціна входу: {entry_price}")
                print(f"Стоп-лосс: {stop_loss_price}")
                print(f"Тейк-профіт: {take_profit_price}")
                print(f"Розмір позиції: {position_size}")

                if signal == "buy":
                    binance_client.place_order("XRPUSDT", "BUY", position_size, entry_price)
                elif signal == "sell":
                    binance_client.place_order("XRPUSDT", "SELL", position_size, entry_price)

                # Оновлення балансу (для тестування)
                trade_result = 50
                risk_manager.update_account_balance(trade_result)
                print(f"Оновлений баланс рахунку: {risk_manager.account_balance:.2f}")

            # Затримка між ітераціями
            time.sleep(60)

        except Exception as e:
            print(f"Помилка: {e}")
            time.sleep(60)

    try:
        # Fetch historical data and train models
        df = fetch_historical_data()
        if df is None:
            return
        
        X, y, scaler = prepare_data(df)

        # Train LSTM model
        lstm_model = build_lstm_model((X.shape[1], 1))
        lstm_model.fit(X, y, epochs=10, batch_size=32)

        # Prepare features for XGBoost
        df["returns"] = df["close"].pct_change()
        df["volume"] = df["volume"].shift(-1)
        df.dropna(inplace=True)
        X_xgb = df[["returns", "volume"]].values
        y_xgb = np.where(df["returns"] > 0, 1, 0)

        # Train XGBoost model
        xgboost_model = build_xgboost_model(X_xgb, y_xgb)

        # Start trading loop
        while True:
            df = fetch_historical_data()
            if df is None:
                continue
            integrate_with_trading(df, lstm_model, xgboost_model, scaler)
            time.sleep(60)  # Пауза між кожним циклом
    except KeyboardInterrupt:
        print("Trading loop stopped.")


if __name__ == "__main__":
    trading_loop()
