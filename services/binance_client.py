from binance.client import Client
from dotenv import load_dotenv
import os

import hmac
import hashlib
import time
import requests

# Завантаження змінних середовища з .env файлу
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', '.env')
load_dotenv(dotenv_path)


API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')


class BinanceClient:
    """
    Клас для взаємодії з Binance API: отримання ринкових даних, виконання угод та моніторинг рахунку.
    """

    def __init__(self, use_testnet = True):
        # Ініціалізація Binance API клієнта
        api_key = API_KEY
        api_secret = API_SECRET
        self.client = Client(api_key, api_secret)

        if use_testnet:
            self.client.API_URL = "https://testnet.binance.vision/api" #Remove in the end

    # 1. Отримання ринкових даних
    def get_candlestick_data(self, symbol, interval='1m', limit=10):
        """
        Отримує історичні дані про свічки для певної торгової пари.
        
        :param symbol: Торгова пара (наприклад, 'BTCUSDT').
        :param interval: Інтервал свічок (наприклад, '1m', '5m', '1h').
        :param limit: Кількість свічок, які потрібно отримати.
        :return: Список даних про свічки.
        """
        try:
            candlesticks = self.client.get_historical_klines(
                symbol=symbol, 
                interval=interval, 
                start_str=f"{limit} min ago UTC"
            )
            return candlesticks
        except Exception as e:
            print(f"Error fetching candlestick data: {e}")
            return None

# # Приклад використання
# candlestick_data = fetch_candlestick_data("BTCUSDT", "1m")
# if candlestick_data:
#     for item in candlestick_data:
#         print(item)

    def get_open_orders(self, symbol=None):
        """
        Отримує відкриті ордери для певної торгової пари або для всіх.

        :param symbol: Торгова пара (опціонально).
        :return: Список відкритих ордерів.
        """
        try:
            if symbol:
                return self.client.get_open_orders(symbol=symbol)
            else:
                return self.client.get_open_orders()
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return None

    # # 2. Виконання угод (Buy/Sell)
    # def place_order(self, symbol, side, quantity):
    #     """
    #     Виконує ринковий ордер на купівлю або продаж.

    #     :param symbol: Торгова пара (наприклад, 'BTCUSDT').
    #     :param side: Сторона угоди ('buy' або 'sell').
    #     :param quantity: Кількість активу для ордера.
    #     :return: Результат виконання ордера.
    #     """
    #     try:
    #         if side.lower() == 'buy':
    #             return self.client.order_market_buy(symbol=symbol, quantity=quantity)
    #         elif side.lower() == 'sell':
    #             return self.client.order_market_sell(symbol=symbol, quantity=quantity)
    #         else:
    #             raise ValueError("Side must be 'buy' or 'sell'")
    #     except Exception as e:
    #         print(f"Error placing order: {e}")
    #         return None
    def place_order(self, symbol, side, quantity, price=None):
        # # """Виконує ордер"""
        # endpoint = f"{self.base_url}/v3/order"
        api_key = API_KEY
        api_secret = API_SECRET
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",  # Змініть на LIMIT, якщо необхідно
            "quantity": quantity,
            "timestamp": int(time.time() * 1000)
        }
        query_string = "&".join([f"{key}={value}" for key, value in params.items()])
        signature = hmac.new(self.api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        headers = {"X-MBX-APIKEY": self.api_key}
        params["signature"] = signature

        response = requests.post(endpoint, headers=headers, params=params)
        if response.status_code == 200:
            print(f"Ордер успішно виконано: {response.json()}")
        else:
            print(f"Помилка виконання ордера: {response.json()}")

    # 3. Моніторинг статусу рахунку
    def account_status(self):
        """
        Отримує інформацію про статус рахунку.

        :return: Дані про баланс рахунку.
        """
        try:
            account_info = self.client.get_account()
            return account_info
        except Exception as e:
            print(f"Error fetching account status: {e}")
            return None
    
    def fetch_candlestick_data(symbol, interval, limit=100):
        url = f'https://testnet.binance.vision/api/v1/klines'
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()  # Перевірка на помилки HTTP
            
            if response.status_code == 200:
                data = response.json()
                if data:  # Перевірка, чи дані не є None або порожніми
                    return data
                else:
                    print("Не отримано жодних даних.")
                    return []
            else:
                print(f"Помилка при отриманні даних: {response.status_code}")
                return []
        except requests.exceptions.Timeout:
            print("Виникла помилка тайм-ауту. Повторна спроба...")
            time.sleep(5)
            return fetch_candlestick_data(symbol, interval, limit)  # Повторна спроба
        except requests.exceptions.RequestException as e:
            print(f"Помилка запиту: {e}")
            return []


# # Приклад використання класу BinanceClient
# if __name__ == "__main__":
#     # Ініціалізація клієнта
#     binance_client = BinanceClient()
    
#     # Отримання даних про свічки
#     print("Candlestick Data:")
#     candles = binance_client.get_candlestick_data(symbol="BTCUSDT", interval="1m", limit=5)
#     print(candles)

#     # Отримання відкритих ордерів
#     print("\nOpen Orders:")
#     open_orders = binance_client.get_open_orders(symbol="BTCUSDT")
#     print(open_orders)

#     # Виконання купівлі (будьте обережні при реальному виконанні!)
#     # print("\nPlacing Buy Order:")
#     # buy_order = binance_client.place_order(symbol="BTCUSDT", side="buy", quantity=0.001)
#     # print(buy_order)

#     # Моніторинг статусу рахунку
#     print("\nAccount Status:")
#     account = binance_client.account_status()
#     print(account)
