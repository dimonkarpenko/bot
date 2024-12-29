

# # Load environment variables
# dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", ".env")
# load_dotenv(dotenv_path)

# # Initialize Binance Client and other components
# binance_client = BinanceClient(use_testnet=True)
# risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)
# TELEGRAM_BOT_TOKEN = os.getenv("Tg_Bot_API")
# bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# # Logging configuration
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger()

# # User state storage
# data_lock = threading.Lock()
# user_data = {}

# # Database setup
# conn = sqlite3.connect("trading_logs.db", check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute(
#     """CREATE TABLE IF NOT EXISTS trades (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         timestamp TEXT,
#         symbol TEXT,
#         action TEXT,
#         position_size REAL,
#         entry_price REAL,
#         result REAL
#     )"""
# )
# conn.commit()


# def trading_loop():
#     """Основний торговий цикл."""
#     logger.info("Starting trading loop.")
#     while True:
#         try:
#             user_data = get_user_data()
#             for chat_id, user_info in user_data.items():
#                 crypto_pair = user_info.get("crypto_pair")
#                 trading_amount = user_info.get("trading_amount")

#                 if not crypto_pair or not trading_amount:
#                     logger.info(f"Missing data for chat_id {chat_id}. Skipping.")
#                     continue

#                 # Fetch historical data and check for empty DataFrame
#                 logger.info(f"Fetching historical data for {crypto_pair}")
#                 df = fetch_historical_data(symbol=crypto_pair, interval="1h", lookback="3 month ago UTC")
#                 if df is None or df.empty:
#                     logger.warning(f"No data fetched for {crypto_pair}. Skipping iteration.")
#                     continue

#                 logger.info(f"Data fetched for {crypto_pair}, shape: {df.shape}")
#                 X, y, scaler = prepare_data(df)

#                 if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
#                     logger.warning(f"Insufficient data for training models. Skipping iteration.")
#                     continue

#                 # Train LSTM model
#                 lstm_model = build_lstm_model((X.shape[1], 1))
#                 lstm_model.fit(X, y, epochs=10, batch_size=32)

#                 # Prepare data for XGBoost model
#                 df["returns"] = df["close"].pct_change()
#                 df["volume"] = df["volume"].shift(-1)
#                 df.dropna(inplace=True)
#                 X_xgb = df[["returns", "volume"]].values
#                 y_xgb = np.where(df["returns"] > 0, 1, 0)

#                 if len(X_xgb) == 0 or len(y_xgb) == 0:
#                     logger.warning(f"Insufficient data for XGBoost. Skipping iteration.")
#                     continue

#                 xgboost_model = build_xgboost_model(X_xgb, y_xgb)

#                 try:
#                     integrate_with_trading(
#                         symbol=crypto_pair,
#                         df=df,
#                         lstm_model=lstm_model,
#                         xgboost_model=xgboost_model,
#                         scaler=scaler,
#                     )
#                 except Exception as e:
#                     logger.error(f"Error in integrate_with_trading for {chat_id}: {e}")
#                     continue

#             # Інтервал між ітераціями
#             time.sleep(60)

#         except Exception as e:
#             logger.error(f"Error in trading loop: {e}")
#             time.sleep(60)
