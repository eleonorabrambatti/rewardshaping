class BSpolicy:
    def __init__(self, s):
        self.base_stock_level = s

    def act(self, current_inventory_level):
        order_quantity = max(0, self.base_stock_level - current_inventory_level)
        return order_quantity