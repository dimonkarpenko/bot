from indicators.ema import calculate_ema
from indicators.macd import calculate_macd
from indicators.rsi import calculate_rsi
from config.logger_config import logger


import numpy as np

# Функція для динамічних порогів для RSI
def dynamic_rsi_threshold(rsi, period=14):
    """
    Генерує динамічні пороги для RSI.
    
    :param rsi: Список значень RSI.
    :param period: Період для розрахунку середнього значення RSI.
    :return: Динамічні нижній і верхній пороги.
    """
    rsi_mean = np.mean(rsi[-period:])  # Середнє значення RSI за період
    dynamic_lower = rsi_mean - 10      # Динамічний нижній поріг для перепроданості
    dynamic_upper = rsi_mean + 10      # Динамічний верхній поріг для перекупленості
    return dynamic_lower, dynamic_upper

# Функція для динамічного порогу для MACD
def dynamic_macd_threshold(macd, signal_line):
    """
    Генерує динамічний поріг для MACD.
    
    :param macd: Список значень MACD.
    :param signal_line: Список значень сигнальної лінії.
    :return: Динамічний поріг для MACD.
    """
    macd_histogram = macd[-1] - signal_line[-1]  # Різниця між MACD і сигнальною лінією
    dynamic_threshold = np.mean(macd_histogram)  # Середнє значення гістограми
    return dynamic_threshold

# Функція для динамічного порогу для EMA
def dynamic_ema_threshold(data, ema_short, ema_long):
    """
    Генерує динамічний поріг для EMA.
    
    :param data: Список даних про свічки.
    :param ema_short: Короткострокова EMA (наприклад, 10).
    :param ema_long: Довгострокова EMA (наприклад, 50).
    :return: Різниця між EMA короткостроковою і довгостроковою.
    """
    ema_diff = ema_short[-1] - ema_long[-1]  # Різниця між EMA(10) і EMA(50)
    return ema_diff

# Основна функція для генерації сигналу
def generate_signal(data):
    """
    Генерує сигнал на основі технічних індикаторів з динамічними порогами.

    :param data: Дані про свічки (закрита ціна).
    :return: Сигнал ('buy', 'sell', 'hold').
    """
    # Розрахунок технічних індикаторів
    ema_short = calculate_ema(data, period=10)
    ema_long = calculate_ema(data, period=50)
    macd, signal_line = calculate_macd(data)
    rsi = calculate_rsi(data)

    logger.info(f"EMA_short: {ema_short[-1]}, EMA_long: {ema_long[-1]}, MACD: {macd[-1]}, Signal Line: {signal_line[-1]}, RSI: {rsi[-1]}")

    # Динамічні пороги для RSI, MACD та EMA
    dynamic_rsi_lower, dynamic_rsi_upper = dynamic_rsi_threshold(rsi)
    dynamic_macd_threshold_val = dynamic_macd_threshold(macd, signal_line)
    dynamic_ema_diff = dynamic_ema_threshold(data, ema_short, ema_long)

    # Перевірка умов для сигналу на покупку
    if rsi[-1] < dynamic_rsi_lower and macd[-1] > signal_line[-1] and data[-1] > ema_short[-1] and dynamic_macd_threshold_val > 0 and dynamic_ema_diff > 0:
        logger.info("Умови виконані для сигналу 'buy'")
        return "buy"
    
    # Перевірка умов для сигналу на продаж
    elif rsi[-1] > dynamic_rsi_upper and macd[-1] < signal_line[-1] and data[-1] < ema_short[-1] and dynamic_macd_threshold_val < 0 and dynamic_ema_diff < 0:
        logger.info("Умови виконані для сигналу 'sell'")
        return "sell"
    
    # Якщо умови не виконуються, сигнал 'hold'
    else:
        logger.info("Сигнал 'hold' іфв через невідповідність умов")
        return "hold"



# def generate_signal(data):
#     """
#     Генерує сигнал на основі технічних індикаторів.

#     :param data: Дані про свічки (закрита ціна).
#     :return: Сигнал ('buy', 'sell', 'hold').
#     """
#     ema = calculate_ema(data, period=20)
#     macd, signal_line = calculate_macd(data)
#     rsi = calculate_rsi(data)

#     logger.info(f"EMA: {ema[-1]}, MACD: {macd[-1]}, Signal Line: {signal_line[-1]}, RSI: {rsi[-1]}")

#     # Знижені пороги для тестування
#     if rsi[-1] < 40 and macd[-1] > signal_line[-1] and data[-1] > ema[-1]:
#         logger.info("Умови виконані для сигналу 'buy'")
#         return "buy"
#     elif rsi[-1] > 60 and macd[-1] < signal_line[-1] and data[-1] < ema[-1]:
#         logger.info("Умови виконані для сигналу 'sell'")
#         return "sell"
#     else:
#         logger.info("Сигнал 'hold' через невідповідність умов")
#         return "hold"
