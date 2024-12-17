from services.binance_client import BinanceClient

if __name__ == "__main__":
    # Ініціалізація клієнта Binance у Testnet режимі
    binance_client = BinanceClient(use_testnet=True)

    # Тестові операції
    print("Отримання свічок:")
    candles = binance_client.get_candlestick_data(symbol="BTCUSDT", interval="1m", limit=5)
    print(candles)

    print("\nВідкриті ордери:")
    open_orders = binance_client.get_open_orders()
    print(open_orders)

    print("\nТестова купівля BTC:")
    order = binance_client.place_order(symbol="BTCUSDT", side="buy", quantity=0.001)
    print(order)

    print("\nСтатус рахунку:")
    account = binance_client.account_status()
    print(account)
