import numpy as np
from random import random
import gymnasium
from gymnasium.spaces import Box
 
from supply_chain_env.schemas import SupplyChainSchema
from supply_chain_env.sized_fifo import SizedFIFO
from supply_chain_env.schemas import StockSchema
from supply_chain_env.schemas import SupplyChainBaseSchema
from supply_chain_env.demand import get_time_series
import collections
# Python logging
import logging
logging.basicConfig()
logger = logging.getLogger('LOGGING_Briw')
logger.setLevel(logging.INFO)


from supply_chain_env.demand import get_base_config

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
            config = SupplyChainSchema(**config) # Nel caso specifico di SupplyChainSchema(**config), **config viene utilizzato per estrarre gli elementi dal dizionario config e passarli come argomenti di parole chiave alla classe SupplyChainSchema. Ciò significa che ogni chiave nel dizionario config diventa un nome di argomento e il valore corrispondente diventa il valore di quell'argomento. 
        self._stock = None
        self._demand = None
        self._demand_history = None
        self._forecast = None
        self._forecast_history = None
        self._action_space = None
        self._observation_space = None
        self._action_history = None
        self._supply_chain_schema = config
        self._time = 0
        ################################################################################################################################################
        self._product_types_num=2
        self._central_warehouses_num=2
        self._local_warehouses_num=2
        self._factory=None
        self._T=None
        self._lead_times=None
        self._len_demand_history=None
        self._lead_times_len=None
        self._d_max= None #maximum demand value
        self._d_var= None
        self._sale_prices=None
        self._excess_demand=None
        self._demand_history_len=None
        self._stock_schema= None
        self._available = None
        self._in_transit = None
        ################################################################################################################################################
 
    def __repr__(self) -> str:
        """ il tipo di dato restituito da una funzione o un metodo """
        return (
            f"SupplyChain(t={self.time}, "
            f"d={self.demand[self.time]}, "
            f"f_t={self.forecast[self.time]}, "
            f"f_l={self.forecast[self.time + self._supply_chain_schema.stock_schema.item_schema.lead_time]}) "
            f"Stock = available: {self._available}, in transit: {self._in_transit})"
        )
 
    def __str__(self) -> str:
        return self.__repr__()
 
 
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
        return self._supply_chain_schema.dict() # va a prendere max_time: int = Field(ge=0) len_demand_history: int = Field(ge=0) len_forecast_history: int = Field(ge=0) stock_schema: StockSchema
           #  self._supply_chain_schema = config
 
    @staticmethod
    def _get_price(price_dict: dict[int, float], num_orders: int) -> float:
        if num_orders <= 0:
            return 0.0
        for key in sorted(price_dict, reverse=True):
            if num_orders >= key:
                return price_dict[key]
        raise ValueError("Price dict is empty.")

 
    def sum(self) -> int:
        return sum(self._available)
    
    def retrieve(self, items: int) -> int:
        if items < 0:
            raise ValueError("items must be non-negative")
 
        items_retrieved = 0
 
        available_items = sum(self._available)
        if items >= available_items:
            items_retrieved = available_items
            for i in range(len(self._available)):
                self._available[i] = 0
            return items_retrieved
 
        cur_idx = len(self._available) - 1
        while items_retrieved < items:
            if self._available[cur_idx] > items - items_retrieved:
                self._available[cur_idx] -= items - items_retrieved
                items_retrieved = items
            else:
                items_retrieved += self._available[cur_idx]
                self._available[cur_idx] = 0
            cur_idx -= 1
        return items_retrieved
 
    def advance(self, new_in_transit: int) -> tuple[int, int]:
        if new_in_transit < 0:
            raise ValueError("new_in_transit must be non-negative")
        arrived = self._in_transit.insert(new_in_transit)
 
        """"
        for _ in range(arrived):
            if random() < self.stock_schema.lost_items_probability:
                arrived -= 1
        """
        arrived = round(
            arrived * (1 - random() * self._supply_chain_schema.stock_schema.lost_items_probability)
        )
 
        expired = self._available.insert(arrived)
        excess = 0
        available_items = sum(self._available)
        if available_items > self._supply_chain_schema.stock_schema.capacity: # se e ' di piu' di quelo che puo' mantenere il magazzino
            excess = available_items - self._supply_chain_schema.stock_schema.capacity
            self.retrieve(excess) # viene restituita la quantita' totale degli articoli prelevati dallinventario
        return expired, excess
    
    def reset(self):
        
        initial_stock = self._supply_chain_schema.initial_stock
        if initial_stock is not None:
            initial_stock = SizedFIFO(initial_stock)
        in_transit = self._supply_chain_schema.initial_in_transit
        if in_transit is not None:
            in_transit = SizedFIFO(in_transit)
        self._stock_schema = self._supply_chain_schema.stock_schema
        self._available = SizedFIFO( [0] * self._stock_schema.item_schema.expiration_time)
        self._in_transit = SizedFIFO([0] * self._stock_schema.item_schema.lead_time)
            
        self._demand = get_time_series(
            mode=self._supply_chain_schema.demand_fn,
            args=self._supply_chain_schema.demand_fn_args,
            length=self._supply_chain_schema.max_time + 1,
        )
        self._demand_history = SizedFIFO(
            [0] * self._supply_chain_schema.len_demand_history
        ) # la lunghezza del vettore e' zero e comunque sarebbe inizializzato a zero ma in questo caso e' vuoto
        self._forecast = get_time_series(
            mode=self._supply_chain_schema.forecast_fn,
            args=self._supply_chain_schema.forecast_fn_args,
            length=self._supply_chain_schema.max_time
            + self._supply_chain_schema.stock_schema.item_schema.lead_time
            + self._supply_chain_schema.len_forecast_history
            + 1,
        ) #perche' scalar_state e' settato su true quindi la lunghezza sara' 239 + 12 + 0 + 1
        self.time = 0
        self._forecast_history = SizedFIFO(
            self._forecast[: self._supply_chain_schema.len_forecast_history][::-1]
        ) # taglia la lista di valori contenuti in forecast alla lunghezza di len forecast history e poi  gira la lista al contrario, cioe' parte dal fondo con i valori
        self._action_space = Box(
            0,
            self._supply_chain_schema.max_orders,
            shape=(1,),
            dtype=np.int64,
        ) # si tratta di un solo elemento che contiene un valore possibile tra i 12 cioe' un solo valore tra 0 e 12
       
        state_len = (
                1
                + len(self._available)
                + len(self._in_transit)
                + len(self.demand_history)
                + len(self.forecast_history)
            ) # se imposto a falso lo scalar state e' 12 + 12 + 0 + 12 
        self._observation_space = Box( # 
                low=np.array([0] * state_len), # viene definito il limite inferiore per l o stato, puo' essere al minimo definito come state len con le caratteristiche sopra specificate
                high=np.array(# al massimo lo stato pu' avere le seguenti carattteristiche
                    np.concatenate(
                        [
                            [self._supply_chain_schema.max_time], # non posso avere piu' di questo a livello di tempo (si tratta del mio time horizon) 
                            [self._supply_chain_schema.stock_schema.capacity] # lo stock disponibile non puo' essere superiore alla capacita' che ho in ogni magazzino 
                            * len(self._available),
                            [self.INF] * len(self._in_transit), # lo stock che ho in transito non puo' superare questo limite
                            [self.INF] * len(self.demand_history),
                            [self.INF] * len(self.forecast_history),
                        ]
                    )
                ),
                dtype=np.int64,
            )
        
        self._action_history = SizedFIFO(
            [0] * self._supply_chain_schema.stock_schema.item_schema.lead_time # action history ha la lunghezza di lead time e la inizializza a zero
        )

        print('qui stampo il reset dello stato')
        print(np.array(
                [
                    self.time,
                    *self._available.queue,
                    *self._in_transit.queue,
                    *self.demand_history.queue,
                    *self.forecast_history.queue,
                ]))
        return np.array( #questo return mi serve solo per stampare lo stato finale dopo che ha agito lo step 
                [
                    self.time,
                    *self._available.queue, # chiama la funzione queue che fa una copia  di _queue cioe' dell'oggetto della classe SizedFifo con tutte le caratteristiche annesse,available quindi viene gestito secondo le specifiche della classe
                    *self._in_transit.queue, # chiama la funzione queue che fa una copia  di _queue cioe' dell'oggetto della classe SizedFifo con tutte le caratteristiche annesse, in transit quindi viene gestito secondo le specifiche della classe
                    *self.demand_history.queue,
                    *self.forecast_history.queue,
                ]
            )
        #return self.get_state(), {}
          
    
    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        action = np.around(action) # arrotonda l'azione all'intero piu' vicino
        if action < 0 or action > self._supply_chain_schema.max_orders: # azione problematica controllata
            raise ValueError(
                "action must be greater than zero and less than or equal to max_orders"
            )
 
        # evaluate the number of items ordered
        ordered_items = action * self._supply_chain_schema.items_per_order # numero di prodotti ordinati, quante volte ordino tot items per oder, quinti rdini faccio di un tipo di prodotto
        self._action_history.insert(ordered_items) # lo aggiungo ad action_history
 
        # advance stock
        expired_items, excess_items = self.advance(ordered_items) # gli expired items vengono poi usati per calcolare i costi e rappresentano le quantita' di prodotto che sono troppo vecchie e vengono quindi scartate
 
        # evaluate unresolved demand
        cur_demand = self.demand[self.time]
        cur_forecast = self.forecast[
            self.time
            + self._supply_chain_schema.stock_schema.item_schema.lead_time # 
            + self._supply_chain_schema.len_forecast_history
        ]
        self.demand_history.insert(cur_demand)
        self.forecast_history.insert(cur_forecast)
        sold_items = self.retrieve(cur_demand) #quantita' effettiva di articoli venduti: gestione nel caso in cui la richeista supera la dispobnibilita' o e' inferiore a tale disponibilit' 
        unresolved_demand = max(0, cur_demand - sold_items) # quanto nn sono riuscito a soddifare
 
        # evaluate reward
        reward = 0.0
 
        reward += (
            self._supply_chain_schema.stock_schema.item_schema.selling_price
            * sold_items
        ) # quello che guadagno vendendo ciascun prodotto
        reward -= self._get_price(
            self._supply_chain_schema.stock_schema.item_schema.order_cost, action
        )
        reward -= (
            unresolved_demand
            * self._supply_chain_schema.stock_schema.item_schema.penalty_cost
        )
        reward -= self.sum() * self._supply_chain_schema.stock_schema.storage_cost
        reward -= (
            expired_items
            * self._supply_chain_schema.stock_schema.item_schema.expiration_cost
        ) # tolgo tutti i costi
 
        # advance time
        self.time += 1

        state = np.array( #questo return mi serve solo per stampare lo stato finale dopo che ha agito lo step 
                [
                    self.time,
                    *self._available.queue, # chiama la funzione queue che fa una copia  di _queue cioe' dell'oggetto della classe SizedFifo con tutte le caratteristiche annesse,available quindi viene gestito secondo le specifiche della classe
                    *self._in_transit.queue, # chiama la funzione queue che fa una copia  di _queue cioe' dell'oggetto della classe SizedFifo con tutte le caratteristiche annesse, in transit quindi viene gestito secondo le specifiche della classe
                    *self.demand_history.queue,
                    *self.forecast_history.queue,
                ]
            )
 
        return (
            state,
            reward,
            self.time >= self._supply_chain_schema.max_time,
            False,
            {
                "logs": [
                    sum(self._available),
                    sum(self._in_transit),
                    expired_items,
                    unresolved_demand
                ]
            },
        )
 
    def render(self):
        pass