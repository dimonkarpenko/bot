from binance.client import Client
from dotenv import load_dotenv
import os
import time
import logging
import requests

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Завантаження змінних середовища з .env файлу
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', '.env')
load_dotenv(dotenv_path)

API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Функція для логування ордерів
def log_trade(order_data):
    """
    Логування даних про ордер.
    
    :param order_data: Дані про ордер.
    """
    logging.info(f"Торгова операція: {order_data}")

class BinanceClient:
    """
    Клас для взаємодії з Binance API: отримання ринкових даних, виконання угод та моніторинг рахунку.
    """
    def __init__(self, use_testnet=True):
        # Замість API_KEY і API_SECRET вставте свої ключі або використовуйте змінні оточення
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        self.client = Client(api_key, api_secret)
        
        # Встановлення URL залежно від типу середовища (тестове чи реальне)
        if use_testnet:
            self.client.API_URL = 'https://testnet.binance.vision/api'
        else:
            self.client.API_URL = 'https://api.binance.com/api'

    def get_current_price(self, symbol):
        """Отримати поточну ціну для заданого символу."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            raise RuntimeError(f"Помилка при отриманні ціни для {symbol}: {e}")

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

    # 2. Отримання відкритих ордерів
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

    # 3. Виконання ордера
    def place_order(self, symbol, side, quantity, price=None, order_type="MARKET"):
        """
        Виконує ордер на Binance.

        :param symbol: Тікер торгової пари (наприклад, 'BTCUSDT').
        :param side: Напрямок угоди ('BUY' або 'SELL').
        :param quantity: Кількість активу.
        :param price: Ціна для лімітного ордера (необов'язково для ринкового ордера).
        :param order_type: Тип ордера ('MARKET', 'LIMIT').
        """
        try:
            # Отримання точності для торгової пари
            symbol_info = self.client.get_symbol_info(symbol)
            step_size = None
            for filter in symbol_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    break
            
            # Округлення кількості до підтримуваної точності
            quantity = self._round_to_step_size(quantity, step_size)

            # Створення ордера
            order = self.client.create_order(
                symbol=symbol,
                side=side.upper(),
                type=order_type,
                quantity=quantity,
                price=price if order_type == "LIMIT" else None,
                timeInForce="GTC" if order_type == "LIMIT" else None
            )
            
            # Логування ордера
            logger(order)

            print(f"Ордер успішно виконано: {order}")
            return order
        except Exception as e:
            print(f"Помилка виконання ордера: {e}")
            return None

    def _round_to_step_size(self, quantity, step_size):
        """
        Округлює кількість до підтримуваної точності для даної торгової пари.
        
        :param quantity: Кількість активу.
        :param step_size: Крок для округлення.
        :return: Округлена кількість.
        """
        return round(quantity / step_size) * step_size

    # 4. Моніторинг статусу рахунку
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

    # 5. Отримання результату угоди
    def get_trade_result(self, symbol, order_id):
        """
        Отримує результат виконаного ордера, тобто прибуток або збиток.

        :param symbol: Торгова пара.
        :param order_id: ID ордера.
        :return: Результат угоди.
        """
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            # Логіка для розрахунку прибутку чи збитку (залежно від типу ордера)
            # Припустимо, що ми працюємо з ринковими ордерами, тому розрахуємо прибуток на основі поточної ціни
            current_price = self.client.get_symbol_ticker(symbol=symbol)["price"]
            entry_price = float(order["fills"][0]["price"])  # Ціна виконання ордера
            trade_result = (float(current_price) - entry_price) * float(order["executedQty"])  # Прибуток/збиток

            print(f"Результат угоди: {trade_result}")
            return trade_result
        except Exception as e:
            print(f"Помилка при отриманні результату угоди: {e}")
            return None
    
    # def get_current_price(self, symbol):
    #     try:
    #         url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
    #         response = requests.get(url)
    #         response.raise_for_status()
    #         return float(response.json()['price'])
    #     except requests.exceptions.RequestException as e:
    #         logger.error(f"Помилка під час отримання ціни для {symbol}: {e}")
    #         return None



# # Виконання реальної угоди
# if __name__ == "__main__":
#     # Ініціалізація клієнта
#     binance_client = BinanceClient()

#     # Параметри угоди
#     signal = "buy"  # або "sell"
#     symbol = "DOGEUSDT"
#     position_size = 100  # Кількість DOGE для покупки/продажу
#     entry_price = 0.32384  # Ціна входу

#     # Виконання угоди
#     if signal == "buy":
#         order = binance_client.place_order(symbol, "BUY", position_size, entry_price)
#     elif signal == "sell":
#         order = binance_client.place_order(symbol, "SELL", position_size, entry_price)

#     # Якщо ордер виконаний, отримуємо результат угоди
#     if order:
#         order_id = order['orderId']
#         trade_result = binance_client.get_trade_result(symbol, order_id)
        
#         # Оновлення балансу
#         if trade_result:
#             binance_client.account_status()  # Тут можна оновити баланс рахунку на основі результату угоди
#             print(f"Оновлений баланс після угоди: {trade_result}")
