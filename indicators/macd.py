import pandas as pd

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Обчислює MACD індикатор.

    :param data: Список цін закриття.
    :param fast_period: Швидкий EMA.
    :param slow_period: Повільний EMA.
    :param signal_period: Період сигналу.
    :return: MACD, сигнал.
    """
    df = pd.DataFrame(data, columns=["close"])
    ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd.tolist(), signal.tolist()
