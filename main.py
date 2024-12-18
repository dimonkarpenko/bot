from machine_learning.ml_models import fetch_historical_data, integrate_with_trading
from services.binance_client import BinanceClient
from signal_generator import generate_signal

if __name__ == "__main__":
    # Ініціалізація Binance Client
    binance_client = BinanceClient(use_testnet=True)

    # Отримання даних про свічки
    candles = binance_client.get_candlestick_data(symbol="DOGEUSDT", interval="1m", limit=50)
    close_prices = [float(candle[4]) for candle in candles]  # Ціна закриття

    # Генерація сигналу
    signal = generate_signal(close_prices)
    print(f"Згенерований сигнал: {signal}")

