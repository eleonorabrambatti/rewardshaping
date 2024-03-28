class sQPolicy:
    """
    The (s, Q)-policy can be expressed by a rule, which can be
    summarized as follows: at each time step t, the current stock level
    is compared to the reorder point s. If the stock level falls below the
    reorder point s, then the (s, Q)-policy orders Q units of product;
    otherwise, it does not take any action.
    """
 
    def __init__(self, factory_s, factory_Q):
        """
        Initialize the SQPolicy class with the factory s and Q values.
        """
        self.factory_s = factory_s
        self.factory_Q = factory_Q
 
    def select_action(self, state):
        """
        Select an action based on the current state.
        """
        # If the available stock is less than the factory reorder point,
        # order the factory Q units.
        if state["stock"].available.sum() < self.factory_s:
            action = self.factory_Q
        else:
            action = 0
 
        return action