from indicators.ema import calculate_ema
from indicators.macd import calculate_macd
from indicators.rsi import calculate_rsi

def generate_signal(data):
    """
    Генерує сигнал на основі технічних індикаторів.

    :param data: Дані про свічки (закрита ціна).
    :return: Сигнал ('buy', 'sell', 'hold').
    """
    ema = calculate_ema(data, period=20)
    macd, signal_line = calculate_macd(data)
    rsi = calculate_rsi(data)

    # Комбінування сигналів
    if rsi[-1] < 30 and macd[-1] > signal_line[-1] and data[-1] > ema[-1]:
        return "buy"
    elif rsi[-1] > 70 and macd[-1] < signal_line[-1] and data[-1] < ema[-1]:
        return "sell"
    else:
        return "hold still"
