from random import random
 
from  supply_chain_env.schemas import StockSchema
from  supply_chain_env.sized_fifo import SizedFIFO
 
 
class Stock:
    """Stock class.
 
    Attributes:
        stock_schema: Stock schema.
        available: Available items.
        in_transit: Items in transit.
 
    Methods:
        get_availability(): Returns the number of available items.
        get_in_transit(): Returns the number of items in transit.
        get_total(): Returns the total number of items.
        retrieve(items): Retrieves items from the stock. If there are not
            enough items, it retrieves all the available items.
        advance(new_in_transit): Advances the stock by one time step. Returns
            the number of items that expired and the number of items
            that exceed the stock capacity.
    """
 
    def __init__(
        self,
        stock_schema: StockSchema,
        available: SizedFIFO = None,
        in_transit: SizedFIFO = None,
    ):
        self.stock_schema = stock_schema
        self.available = available or SizedFIFO(
            [0] * stock_schema.item_schema.expiration_time
        )
        self.in_transit = in_transit or SizedFIFO(
            [0] * stock_schema.item_schema.lead_time
        )
        if sum(self.available) > self.stock_schema.capacity:
            raise ValueError("stock must be less than or equal to capacity")
        if len(self.available) != stock_schema.item_schema.expiration_time:
            raise ValueError("available length must be equal to expiration time")
        if len(self.in_transit) != stock_schema.item_schema.lead_time:
            raise ValueError("transit length must be equal to lead time")
 
    def __repr__(self) -> str:
        return f"Stock(avail={self.available}, transit={self.in_transit})"
 
    def sum(self) -> int:
        return sum(self.available)
 
    def retrieve(self, items: int) -> int:
        if items < 0:
            raise ValueError("items must be non-negative")
 
        items_retrieved = 0
 
        available_items = sum(self.available)
        if items >= available_items:
            items_retrieved = available_items
            for i in range(len(self.available)):
                self.available[i] = 0
            return items_retrieved
 
        cur_idx = len(self.available) - 1
        while items_retrieved < items:
            if self.available[cur_idx] > items - items_retrieved:
                self.available[cur_idx] -= items - items_retrieved
                items_retrieved = items
            else:
                items_retrieved += self.available[cur_idx]
                self.available[cur_idx] = 0
            cur_idx -= 1
        return items_retrieved
 
    def advance(self, new_in_transit: int) -> tuple[int, int]:
        if new_in_transit < 0:
            raise ValueError("new_in_transit must be non-negative")
        arrived = self.in_transit.insert(new_in_transit)
 
        """"
        for _ in range(arrived):
            if random() < self.stock_schema.lost_items_probability:
                arrived -= 1
        """
        arrived = round(
            arrived * (1 - random() * self.stock_schema.lost_items_probability)
        )
 
        expired = self.available.insert(arrived)
        excess = 0
        available_items = sum(self.available)
        if available_items > self.stock_schema.capacity:
            excess = available_items - self.stock_schema.capacity
            self.retrieve(excess)
        return expired, excess