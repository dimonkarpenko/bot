import pandas as pd

def calculate_ema(data, period):
    """
    Обчислює експоненціальну ковзну середню (EMA).

    :param data: Список цін закриття.
    :param period: Період EMA.
    :return: EMA значення.
    """
    df = pd.DataFrame(data, columns=["close"])
    ema = df["close"].ewm(span=period, adjust=False).mean()
    return ema.tolist()
