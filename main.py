from machine_learning.ml_models import fetch_historical_data, integrate_with_trading
from services.binance_client import BinanceClient
from risk_managment.risk_management import RiskManagement
from signal_generator import generate_signal
import time

def trading_loop():
    # Ініціалізація Binance Client
    binance_client = BinanceClient(use_testnet=True)

    # Ініціалізація ризик-менеджменту
    risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)

    while True:
        try:
            # Отримання даних про свічки
            candles = binance_client.get_candlestick_data(symbol="DOGEUSDT", interval="1m", limit=50)
            close_prices = [float(candle[4]) for candle in candles]  #Ціна закриття

            # Генерація сигналу
            signal = generate_signal(close_prices)
            print(f"Згенерований сигнал: {signal}")

            if signal in ["buy", "sell"]:
                # Розрахунок позиції
                entry_price = close_prices[-1]
                stop_loss_price = risk_manager.calculate_stop_loss(entry_price, risk_percentage=0.02)
                position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price)
                take_profit_price = risk_manager.calculate_take_profit_price(entry_price)

                print(f"Сигнал: {signal}")
                print(f"Ціна входу: {entry_price}")
                print(f"Стоп-лосс: {stop_loss_price}")
                print(f"Тейк-профіт: {take_profit_price}")
                print(f"Розмір позиції: {position_size}")

                # Виконання угоди
                if signal == "buy":
                    binance_client.place_order("DOGEUSDT", "BUY", position_size, entry_price)
                elif signal == "sell":
                    binance_client.place_order("DOGEUSDT", "SELL", position_size, entry_price)

                # Оновлення балансу (для тестування)
                trade_result = 50  # Симуляція прибутку
                risk_manager.update_account_balance(trade_result)
                print(f"Оновлений баланс рахунку: {risk_manager.account_balance:.2f}")

            # Затримка між ітераціями
            time.sleep(60)

        except Exception as e:
            print(f"Помилка: {e}")
            time.sleep(60)

if __name__ == "__main__":
    trading_loop()
