class RiskManagement:
    def __init__(self, account_balance, risk_per_trade=0.01, max_trade_size=0.1):
        """
        Ініціалізація модуля ризик-менеджменту.

        :param account_balance: Баланс рахунку.
        :param risk_per_trade: Частка ризику на одну угоду (0.01 = 1% від балансу).
        :param max_trade_size: Максимальна частка балансу, яку можна використати на угоду.
        """
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_trade_size = max_trade_size

    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Розрахунок розміру позиції на основі ризику.

        :param entry_price: Ціна входу в угоду.
        :param stop_loss_price: Ціна стоп-лосс.
        :return: Рекомендований розмір позиції.
        """
        risk_amount = self.account_balance * self.risk_per_trade
        stop_loss_distance = abs(entry_price - stop_loss_price)
        position_size = risk_amount / stop_loss_distance
        max_position_size = self.account_balance * self.max_trade_size / entry_price

        return min(position_size, max_position_size)

    def calculate_take_profit_price(self, entry_price, risk_reward_ratio=2):
        """
        Розрахунок тейк-профіт ціни на основі співвідношення ризик/прибуток.

        :param entry_price: Ціна входу.
        :param risk_reward_ratio: Співвідношення ризик/прибуток.
        :return: Ціна тейк-профіту.
        """
        stop_loss_distance = entry_price * self.risk_per_trade
        return entry_price + (stop_loss_distance * risk_reward_ratio)

    def trailing_stop(self, current_price, trailing_percentage):
        """
        Розрахунок ціни трейлінг-стопу.

        :param current_price: Поточна ціна активу.
        :param trailing_percentage: Відсоток трейлінг-стопу (0.05 = 5%).
        :return: Ціна трейлінг-стопу.
        """
        trailing_stop_price = current_price * (1 - trailing_percentage)
        return trailing_stop_price

    def update_account_balance(self, trade_result):
        """
        Оновлення балансу рахунку після угоди.

        :param trade_result: Результат угоди (позитивний чи негативний).
        """
        self.account_balance += trade_result


# # Приклад використання
# if __name__ == "__main__":
#     # Ініціалізація ризик-менеджменту
#     risk_manager = RiskManagement(account_balance=1000, risk_per_trade=0.02)

#     # Розрахунок позиції
#     entry_price = 100
#     stop_loss_price = 95
#     position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price)
#     print(f"Рекомендований розмір позиції: {position_size:.2f}")

#     # Розрахунок тейк-профіту
#     take_profit_price = risk_manager.calculate_take_profit_price(entry_price)
#     print(f"Ціна тейк-профіту: {take_profit_price:.2f}")

#     # Розрахунок трейлінг-стопу
#     current_price = 110
#     trailing_stop_price = risk_manager.trailing_stop(current_price, trailing_percentage=0.05)
#     print(f"Ціна трейлінг-стопу: {trailing_stop_price:.2f}")

#     # Оновлення балансу після угоди
#     trade_result = 50  # Припустимо, отриманий прибуток
#     risk_manager.update_account_balance(trade_result)
#     print(f"Оновлений баланс рахунку: {risk_manager.account_balance:.2f}")
