import numpy as np
 
import gymnasium
from gymnasium.spaces import Box
 
from schemas import SupplyChainSchema
from sized_fifo import SizedFIFO
from stock import Stock
from time_series import get_time_series
 
 
class SupplyChainEnvironment(gymnasium.Env):
    """Supply chain Gymnasium environment.
 
    Methods:
        reset(): Resets the environment.
        get_state(): Returns the current state.
        step(action): Advances the environment by one time step.
    """
 
    INF: int = 1e9
 
    def __init__(self, config: str):
        if isinstance(config, str):
            config = SupplyChainSchema.parse_file(config)
        elif isinstance(config, dict):
            config = SupplyChainSchema(**config)
        self._stock = None
        self._demand = None
        self._demand_history = None
        self._forecast = None
        self._forecast_history = None
        self._action_space = None
        self._observation_space = None
        self._action_history = None
        self._supply_chain_schema = config
        self.time = 0
        self.reset()
 
    def __repr__(self) -> str:
        """ il tipo di dato restituito da una funzione o un metodo """
        return (
            f"SupplyChain(t={self.time}, "
            f"d={self.demand[self.time]}, "
            f"f_t={self.forecast[self.time]}, "
            f"f_l={self.forecast[self.time + self._supply_chain_schema.stock_schema.item_schema.lead_time]}, "
            f"s={self.stock})"
        )
 
    def __str__(self) -> str:
        return self.__repr__()
 
    @property
    def stock(self) -> Stock:
        if self._stock is None:
            raise ValueError("Stock is not initialized. Call reset() first.")
        return self._stock
 
    @property
    def demand(self) -> np.array:
        if self._demand is None:
            raise ValueError("Demand is not initialized. Call reset() first.")
        return self._demand
 
    @property
    def demand_history(self) -> SizedFIFO:
        if self._demand_history is None:
            raise ValueError("Demand history is not initialized. Call reset() first.")
        return self._demand_history
 
    @property
    def forecast(self) -> np.array:
        if self._forecast is None:
            raise ValueError("Forecast is not initialized. Call reset() first.")
        return self._forecast
 
    @property
    def forecast_history(self) -> SizedFIFO:
        if self._forecast_history is None:
            raise ValueError("Forecast history is not initialized. Call reset() first.")
        return self._forecast_history
 
    @property
    def action_history(self) -> SizedFIFO:
        if self._action_history is None:
            raise ValueError("Action history is not initialized. Call reset() first.")
        return self._action_history
 
    @property
    def action_space(self) -> Box:
        if self._action_space is None:
            raise ValueError("Action space is not initialized. Call reset() first.")
        return self._action_space
 
    @property
    def observation_space(self) -> Box:
        if self._observation_space is None:
            raise ValueError(
                "Observation space is not initialized. Call reset() first."
            )
        return self._observation_space
 
    def get_config(self) -> dict:
        return self._supply_chain_schema.dict()
 
    @staticmethod
    def _get_price(price_dict: dict[int, float], num_orders: int) -> float:
        if num_orders <= 0:
            return 0.0
        for key in sorted(price_dict, reverse=True):
            if num_orders >= key:
                return price_dict[key]
        raise ValueError("Price dict is empty.")
 
    def reset(self, seed=None, options=None):
        initial_stock = self._supply_chain_schema.initial_stock
        if initial_stock is not None:
            initial_stock = SizedFIFO(initial_stock)
        in_transit = self._supply_chain_schema.initial_in_transit
        if in_transit is not None:
            in_transit = SizedFIFO(in_transit)
        self._stock = Stock(
            stock_schema=self._supply_chain_schema.stock_schema,
            available=initial_stock,
            in_transit=in_transit,
        )
        self._demand = get_time_series(
            mode=self._supply_chain_schema.demand_fn,
            args=self._supply_chain_schema.demand_fn_args,
            length=self._supply_chain_schema.max_time + 1,
        )
        self._demand_history = SizedFIFO(
            [0] * self._supply_chain_schema.len_demand_history
        )
        self._forecast = get_time_series(
            mode=self._supply_chain_schema.forecast_fn,
            args=self._supply_chain_schema.forecast_fn_args,
            length=self._supply_chain_schema.max_time
            + self._supply_chain_schema.stock_schema.item_schema.lead_time
            + self._supply_chain_schema.len_forecast_history
            + 1,
        )
        self.time = 0
        self._forecast_history = SizedFIFO(
            self._forecast[: self._supply_chain_schema.len_forecast_history][::-1]
        )
        self._action_space = Box(
            0,
            self._supply_chain_schema.max_orders,
            shape=(1,),
            dtype=np.int64,
        )
        if self._supply_chain_schema.scalar_state:
            state_len = 3
            self._observation_space = Box(
                low=np.array([-self.INF] * state_len),
                high=np.array([self.INF] * state_len),
                dtype=np.int64,
            )
        else:
            state_len = (
                1
                + len(self.stock.available)
                + len(self.stock.in_transit)
                + len(self.demand_history)
                + len(self.forecast_history)
            )
            self._observation_space = Box(
                low=np.array([0] * state_len),
                high=np.array(
                    np.concatenate(
                        [
                            [self._supply_chain_schema.max_time],
                            [self._supply_chain_schema.stock_schema.capacity]
                            * len(self.stock.available),
                            [self.INF] * len(self.stock.in_transit),
                            [self.INF] * len(self.demand_history),
                            [self.INF] * len(self.forecast_history),
                        ]
                    )
                ),
                dtype=np.int64,
            )
        self._action_history = SizedFIFO(
            [0] * self._supply_chain_schema.stock_schema.item_schema.lead_time
        )
        return self.get_state(), {}
 
    def get_state(self) -> np.ndarray:
        if self._supply_chain_schema.scalar_state:
            stock_copy = self.stock.available.copy()
            inventory_evolution = []
            for _ in range(
                self._supply_chain_schema.stock_schema.item_schema.lead_time
            ):
                inventory_evolution.append(sum(stock_copy))
                stock_copy.insert(0)
 
            forecast_evolution = self._forecast[
                self.time : self.time + len(inventory_evolution)
            ]
 
            inventory_demand_evolution = [
                inventory_evolution[0] - forecast_evolution[0]
            ]
 
            for i in range(1, len(inventory_evolution)):
                inventory_demand_evolution.append(
                    min(inventory_demand_evolution[i - 1], inventory_evolution[i])
                    - forecast_evolution[i]
                )
 
            forecast_prediction = self._forecast[
                self.time
                + self._supply_chain_schema.stock_schema.item_schema.lead_time : self.time
                + 2 * self._supply_chain_schema.stock_schema.item_schema.lead_time
            ]
 
            return np.array(
                [
                    inventory_demand_evolution[-1] + sum(self._action_history),
                    sum(forecast_prediction),
                    self.time,
                ]
            )
        else:
            return np.array(
                [
                    self.time,
                    *self.stock.available.queue,
                    *self.stock.in_transit.queue,
                    *self.demand_history.queue,
                    *self.forecast_history.queue,
                ]
            )
 
    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        action = np.around(action)
        if action < 0 or action > self._supply_chain_schema.max_orders:
            raise ValueError(
                "action must be greater than zero and less than or equal to max_orders"
            )
 
        # evaluate the number of items ordered
        ordered_items = action * self._supply_chain_schema.items_per_order
        self._action_history.insert(ordered_items)
 
        # advance stock
        expired_items, excess_items = self.stock.advance(ordered_items)
 
        # evaluate unresolved demand
        cur_demand = self.demand[self.time]
        cur_forecast = self.forecast[
            self.time
            + self._supply_chain_schema.stock_schema.item_schema.lead_time
            + self._supply_chain_schema.len_forecast_history
        ]
        self.demand_history.insert(cur_demand)
        self.forecast_history.insert(cur_forecast)
        sold_items = self.stock.retrieve(cur_demand)
        unresolved_demand = max(0, cur_demand - sold_items)
 
        # evaluate reward
        reward = 0.0
 
        reward += (
            self._supply_chain_schema.stock_schema.item_schema.selling_price
            * sold_items
        )
        reward -= self._get_price(
            self._supply_chain_schema.stock_schema.item_schema.order_cost, action
        )
        reward -= (
            unresolved_demand
            * self._supply_chain_schema.stock_schema.item_schema.penalty_cost
        )
        reward -= self.stock.sum() * self._supply_chain_schema.stock_schema.storage_cost
        reward -= (
            expired_items
            * self._supply_chain_schema.stock_schema.item_schema.expiration_cost
        )
 
        # advance time
        self.time += 1
 
        return (
            self.get_state(),
            reward,
            self.time >= self._supply_chain_schema.max_time,
            False,
            {
                "logs": [
                    sum(self.stock.available),
                    sum(self.stock.in_transit),
                    expired_items,
                    unresolved_demand,
                    self.get_state(),
                ]
            },
        )
 
    def render(self):
        pass