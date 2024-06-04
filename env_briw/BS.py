class BSpolicy:
    def _init_(self, s):
        self.base_stock_level = s

    def act(self, current_inventory_level):
        if isinstance(self.base_stock_level, list):
            order_quantity = max(
                0, self.base_stock_level[0] - current_inventory_level)
        else:
            order_quantity = max(
                0, self.base_stock_level - current_inventory_level)
        return order_quantity