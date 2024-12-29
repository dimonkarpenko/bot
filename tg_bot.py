# import os
# import logging
# import threading
# import requests
# from dotenv import load_dotenv
# import telebot
# from services.binance_client import BinanceClient
# from machine_learning.ml_models import (
#     fetch_historical_data,
#     prepare_data,
#     build_lstm_model,
#     build_xgboost_model,
#     integrate_with_trading,
# )
# from indicators.signal_generator import generate_signal
# from risk_managment.risk_managment import RiskManagement


# # # Додаткові функції Telegram-бота


# # @bot.message_handler(commands=["time_rate"])
# # def time_rate(message):
# #     chat_id = message.chat.id
# #     try:
# #         with data_lock:
# #             trading_data = user_data.get(chat_id, {})
        
# #         # Додайте логіку отримання даних про стан
# #         bot_active = "Активний" if trading_data else "Неактивний"
# #         total_trades = len(trading_data.get("trades", [])) if "trades" in trading_data else 0
# #         total_profit = sum(trade.get("result", 0) for trade in trading_data.get("trades", [])) if "trades" in trading_data else 0
        
# #         response = f"""
# #         Стан торгового бота:
# #         - Статус: {bot_active}
# #         - Кількість угод: {total_trades}
# #         - Загальний заробіток: {total_profit:.2f} USD
# #         """
# #         bot.send_message(chat_id, response)
# #     except Exception as e:
# #         logger.error(f"Error in check_status: {e}")
# #         bot.send_message(chat_id, "Не вдалося отримати статус. Спробуйте ще раз.")


# @bot.message_handler(commands=["predicted_price"])
# def predicted_price(message):
   
#     chat_id = message.chat.id
#     try:
#         with data_lock:
#             trading_data = user_data.get(chat_id, {})
        
#         crypto_pair = trading_data.get("crypto_pair")
        
#         if not crypto_pair:
#             bot.send_message(chat_id, "Спочатку задайте торгову пару за допомогою команди /start.")
#             return
        
#         # Отримання історичних даних
#         df = fetch_historical_data(symbol=crypto_pair, interval="30m", lookback="1 year ago UTC")
#         if df is None or df.empty:
#             bot.send_message(chat_id, "Не вдалося отримати історичні дані для цієї пари. Спробуйте пізніше.")
#             return 
            
#         predicted_price = integrate_with_trading(
#                 df=df,
#             symbol=crypto_pair,
#             lstm_model=lstm_model,  # Попередньо створена модель
#             scaler=scaler  # Попередньо створений scaler
#         )

#         if predicted_price is not None:
#             bot.send_message(chat_id, f"Передбачена ціна для {crypto_pair}: {predicted_price:.2f} USD")
#         else:
#             bot.send_message(chat_id, "Не вдалося отримати передбачену ціну.")
            
#     except Exception as e:
#         logger.error(f"Error in check_status: {e}")
#         bot.send_message(chat_id, "Не вдалося отримати статус. Спробуйте ще раз.")



# # @bot.message_handler(commands=["status"])
# # def check_status(message):
# #     """Перевірка стану торгового бота."""
# #     chat_id = message.chat.id
# #     try:
# #         with data_lock:
# #             trading_data = user_data.get(chat_id, {})
        
# #         # Додайте логіку отримання даних про стан
# #         bot_active = "Активний" if trading_data else "Неактивний"
# #         total_trades = len(trading_data.get("trades", [])) if "trades" in trading_data else 0
# #         total_profit = sum(trade.get("result", 0) for trade in trading_data.get("trades", [])) if "trades" in trading_data else 0
        
# #         response = f"""
# #         Стан торгового бота:
# #         - Статус: {bot_active}
# #         - Кількість угод: {total_trades}
# #         - Загальний заробіток: {total_profit:.2f} USD
# #         """
# #         bot.send_message(chat_id, response)
# #     except Exception as e:
# #         logger.error(f"Error in check_status: {e}")
# #         bot.send_message(chat_id, "Не вдалося отримати статус. Спробуйте ще раз.")

# # @bot.message_handler(commands=["balance"])
# # def get_balance(message):
# #     """Показує баланс Binance."""
# #     chat_id = message.chat.id
# #     try:
# #         balance = binance_client.get_account_balance()  # Використовуйте ваш метод для отримання балансу
# #         response = "\n".join([f"{asset}: {amount}" for asset, amount in balance.items()])
# #         bot.send_message(chat_id, f"Ваш баланс:\n{response}")
# #     except Exception as e:
# #         logger.error(f"Error in get_balance: {e}")
# #         bot.send_message(chat_id, "Не вдалося отримати баланс. Перевірте підключення до Binance.")

# # @bot.message_handler(commands=["buy"])
# # def buy_crypto(message):
# #     """Купівля криптовалюти."""
# #     try:
# #         params = message.text.split()
# #         if len(params) != 3:
# #             bot.send_message(message.chat.id, "Формат: /buy <symbol> <quantity>")
# #             return
# #         symbol, quantity = params[1].upper(), float(params[2])
# #         result = binance_client.buy(symbol, quantity)  # Реалізуйте метод покупки
# #         bot.send_message(message.chat.id, f"Купівля {quantity} {symbol} виконана. Результат: {result}")
# #     except Exception as e:
# #         logger.error(f"Error in buy_crypto: {e}")
# #         bot.send_message(message.chat.id, "Не вдалося виконати покупку. Спробуйте ще раз.")

# # @bot.message_handler(commands=["sell"])
# # def sell_crypto(message):
# #     """Продаж криптовалюти."""
# #     try:
# #         params = message.text.split()
# #         if len(params) != 3:
# #             bot.send_message(message.chat.id, "Формат: /sell <symbol> <quantity>")
# #             return
# #         symbol, quantity = params[1].upper(), float(params[2])
# #         result = binance_client.sell(symbol, quantity)  # Реалізуйте метод продажу
# #         bot.send_message(message.chat.id, f"Продаж {quantity} {symbol} виконана. Результат: {result}")
# #     except Exception as e:
# #         logger.error(f"Error in sell_crypto: {e}")
# #         bot.send_message(message.chat.id, "Не вдалося виконати продаж. Спробуйте ще раз.")

# # @bot.message_handler(commands=["positions"])
# # def show_positions(message):
# #     """Показує відкриті позиції."""
# #     try:
# #         positions = binance_client.get_open_positions()  # Реалізуйте метод отримання відкритих позицій
# #         response = "\n".join([f"{pos['symbol']}: {pos['quantity']} @ {pos['entry_price']}" for pos in positions])
# #         bot.send_message(message.chat.id, f"Відкриті позиції:\n{response}")
# #     except Exception as e:
# #         logger.error(f"Error in show_positions: {e}")
# #         bot.send_message(message.chat.id, "Не вдалося отримати відкриті позиції.")

# # @bot.message_handler(commands=["close_position"])
# # def close_position(message):
# #     """Закриває позицію за символом."""
# #     try:
# #         params = message.text.split()
# #         if len(params) != 2:
# #             bot.send_message(message.chat.id, "Формат: /close_position <symbol>")
# #             return
# #         symbol = params[1].upper()
# #         result = binance_client.close_position(symbol)  # Реалізуйте метод закриття позиції
# #         bot.send_message(message.chat.id, f"Позиція {symbol} закрита. Результат: {result}")
# #     except Exception as e:
# #         logger.error(f"Error in close_position: {e}")
# #         bot.send_message(message.chat.id, "Не вдалося закрити позицію. Спробуйте ще раз.")

# # @bot.message_handler(commands=["set_risk"])
# # def set_risk(message):
# #     """Встановлення рівня ризику."""
# #     try:
# #         params = message.text.split()
# #         if len(params) != 2:
# #             bot.send_message(message.chat.id, "Формат: /set_risk <value>")
# #             return
# #         risk_value = float(params[1])
# #         risk_manager.set_risk(risk_value)  # Реалізуйте метод налаштування ризику
# #         bot.send_message(message.chat.id, f"Рівень ризику встановлено на {risk_value}")
# #     except Exception as e:
# #         logger.error(f"Error in set_risk: {e}")
# #         bot.send_message(message.chat.id, "Не вдалося встановити рівень ризику.")

# # # Інші обробники реалізуйте аналогічно


# # def get_user_data():
# #     """Функція для отримання поточного стану користувачів."""
# #     with data_lock:
# #         return user_data

# # def run_bot():
# #     """Запуск Telegram-бота."""
# #     try:
# #         logger.info("Telegram bot is starting.")
# #         bot.polling(none_stop=True, timeout=20, interval=1)
# #     except Exception as e:
# #         logger.error(f"Error starting Telegram bot: {e}")
