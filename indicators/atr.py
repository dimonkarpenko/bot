import pandas as pd

def calculate_atr(data, period=14):
    """
    Обчислює ATR індикатор.

    :param data: Список свічок (відкрита, висока, низька, закрита).
    :param period: Період ATR.
    :return: ATR значення.
    """
    df = pd.DataFrame(data, columns=["open", "high", "low", "close"])
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.tolist()
