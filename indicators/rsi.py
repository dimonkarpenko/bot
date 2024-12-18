import pandas as pd

def calculate_rsi(data, period=14):
    """
    Обчислює RSI індикатор.

    :param data: Список цін закриття.
    :param period: Період RSI.
    :return: RSI значення.
    """
    df = pd.DataFrame(data, columns=["close"])
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.tolist()
