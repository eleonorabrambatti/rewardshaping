class sQpolicy:
    def __init__(self, s, q):
        self.s = s
        self.q = q

    def act(self, current_inventory_level):
        if current_inventory_level < self.s:
            order_quantity = self.q
        else:
            order_quantity = 0

        return order_quantity
