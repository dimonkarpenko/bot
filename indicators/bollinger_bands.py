import pandas as pd

def calculate_bollinger_bands(data, period=20, num_std_dev=2):
    """
    Обчислює Bollinger Bands.

    :param data: Список цін закриття.
    :param period: Період середнього.
    :param num_std_dev: Кількість стандартних відхилень.
    :return: Верхня смуга, середня лінія, нижня смуга.
    """
    df = pd.DataFrame(data, columns=["close"])
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    upper_band = sma + (std * num_std_dev)
    lower_band = sma - (std * num_std_dev)
    return upper_band.tolist(), sma.tolist(), lower_band.tolist()
