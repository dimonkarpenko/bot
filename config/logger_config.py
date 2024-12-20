import logging

# Налаштування логера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='trading.log', filemode='a')
logger = logging.getLogger()
