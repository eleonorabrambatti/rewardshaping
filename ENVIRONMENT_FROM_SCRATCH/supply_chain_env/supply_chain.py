import numpy as np
 
import gymnasium
from gymnasium.spaces import Box
 
from supply_chain_env.schemas import SupplyChainSchema
from supply_chain_env.sized_fifo import SizedFIFO
from supply_chain_env.stock import Stock
from supply_chain_env.time_series import get_time_series
from supply_chain_env.State_Action import State, Action
import collections
# Python logging
import logging
logging.basicConfig()
logger = logging.getLogger('LOGGING_Briw')
logger.setLevel(logging.INFO)
import random

from supply_chain_env.base_demand import get_base_config

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
        self.time = 0
        ################################################################################################################################################
        self._product_types_num=None
        self._central_warehouses_num=None
        self._local_warehouses_num=None
        self._factory=None
        self._T=None
        self.lead_times=None
        self._lead_times_len=None
        self._d_max= None #maximum demand value
        self._d_var= None
        self._sale_prices=None
        self.excess_demand=None
        self._demand_history_len=None
        ################################################################################################################################################
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

 
    """ def reset(self, seed=None, options=None):
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
        if self._supply_chain_schema.scalar_state:
            state_len = 3
            self._observation_space = Box(
                low=np.array([-self.INF] * state_len),
                high=np.array([self.INF] * state_len),
                dtype=np.int64,
            ) # Questo vettore sarà composto da 3 elementi, ciascuno dei quali può assumere valori compresi tra meno infinito e più infinito (-np.inf a np.inf).
        else:
            state_len = (
                1
                + len(self.stock.available)
                + len(self.stock.in_transit)
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
                            * len(self.stock.available),
                            [self.INF] * len(self.stock.in_transit), # lo stock che ho in transito non puo' superare questo limite
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
        return self.get_state(), {}
  """
    def reset(self, seed=None):
        if seed:
            self.demand_random_generator = np.random.default_rng(seed=seed) # definisco il seed per generare la domanda in modo randomco ma con riproducibilita'
        self.lead_times = collections.deque(maxlen=self._supply_chain_schema.stock_schema.item_schema.lead_time) # crea un vetttore di dimensione lead time che in qusto caso e' 12
        self.demand_history = collections.deque(maxlen=self._supply_chain_schema.len_demand_history)
        logger.debug(f"\n--- SupplyChainEnvironment --- reset"
                     f"\nlead_times_len is "
                     f"{self._supply_chain_schema.stock_schema.item_schema.lead_time}"
                     f"\ndemand_history_len is "
                     f"{self._supply_chain_schema.len_demand_history}"
                     f"\nlead_times is "
                     f"{self.lead_times}"
                     f"\ndemand_history is "
                     f"{self.demand_history}"
                     )
        if self._supply_chain_schema.stock_schema.item_schema.lead_time > 0: #creo la matrice che conterra' il numero di prodotti in transito per ciascun warehouse (central e local)
            for l in range(self._supply_chain_schema.stock_schema.item_schema.lead_time):
                self.lead_times.appendleft(np.zeros(
                    (self._central_warehouses_num+self._local_warehouses_num, self._product_types_num),
                    dtype=np.int32))

        for d in range(self._supply_chain_schema.len_demand_history):
            self.demand_history.append(np.zeros(
                (self._central_warehouses_num+self._local_warehouses_num, self._product_types_num),
                dtype=np.int32))
        self.time = 0

        logger.debug(f"\nlead_times is "
                     f"{self.lead_times}"
                     f"\ndemand_history is "
                     f"{self.demand_history}"
                     f"\nt is "
                     f"{self.time}")
        # definition of the state space
        state_len = (
                1
                + self._supply_chain_schema.stock_schema.item_schema.expiration_time
                + len(self.lead_times)
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
                            * self._supply_chain_schema.stock_schema.item_schema.expiration_time,
                            [self.INF] * len(self.lead_times), # lo stock che ho in transito non puo' superare questo limite
                            [self.INF] * len(self.demand_history),
                            [self.INF] * len(self.forecast_history),
                        ]
                    )
                ),
                dtype=np.int64,
            )
        # definition of the action space
        self._action_space = Box(
            0,
            self._supply_chain_schema.max_orders,
            shape=(1,),
            dtype=np.int64,
        ) # si tratta di un solo elemento che contiene un valore possibile tra i 12 cioe' un solo valore tra 0 e 12
        self._demand = get_time_series(
            mode=self._supply_chain_schema.demand_fn,
            args=self._supply_chain_schema.demand_fn_args,
            length=self._supply_chain_schema.max_time + 1,
        )
        self._forecast = get_time_series(
            mode=self._supply_chain_schema.forecast_fn,
            args=self._supply_chain_schema.forecast_fn_args,
            length=self._supply_chain_schema.max_time
            + self._supply_chain_schema.stock_schema.item_schema.lead_time
            + self._supply_chain_schema.len_forecast_history
            + 1,
        ) #perche' scalar_state e' settato su true quindi la lunghezza sara' 239 + 12 + 0 + 1

    """ def get_state(self) -> np.ndarray:
        if self._supply_chain_schema.scalar_state: # se la condizione e' true
            stock_copy = self.stock.available.copy()
            inventory_evolution = []
            for _ in range(
                self._supply_chain_schema.stock_schema.item_schema.lead_time # per 12 volte 
            ):
                inventory_evolution.append(sum(stock_copy))
                print('qui stampo come evolve l inventario')
                print(inventory_evolution)
                stock_copy.insert(0)
 
            forecast_evolution = self._forecast[
                self.time : self.time + len(inventory_evolution)
            ]
            print('qui stampo come evolve il forecast')
            print(forecast_evolution)
            inventory_demand_evolution = [
                inventory_evolution[0] - forecast_evolution[0]
            ]
            print('qui stampo come evolve l inventory_demand_evolution')
            print(inventory_demand_evolution)

            for i in range(1, len(inventory_evolution)):
                inventory_demand_evolution.append(
                    min(inventory_demand_evolution[i - 1], inventory_evolution[i])
                    - forecast_evolution[i]
                )
            print('qui stampo come evolve l inventory_demand_evolution')
            print(inventory_demand_evolution)
 
            forecast_prediction = self._forecast[
                self.time
                + self._supply_chain_schema.stock_schema.item_schema.lead_time : self.time
                + 2 * self._supply_chain_schema.stock_schema.item_schema.lead_time
            ]
            print('qui stampo come evolve il forecast predction')
            print(forecast_prediction)
            print(np.array(
                [
                    inventory_demand_evolution[-1] + sum(self._action_history),
                    sum(forecast_prediction),
                    self.time,
                ]
            ))
            return np.array(
                [
                    inventory_demand_evolution[-1] + sum(self._action_history),
                    sum(forecast_prediction),
                    self.time,
                ]
            )
        else:
            print('qui stampo come evolve l inventario')
            print(np.array(
                [
                    self.time,
                    *self.stock.available.queue,
                    *self.stock.in_transit.queue,
                    *self.demand_history.queue,
                    *self.forecast_history.queue,
                ]))
            return np.array( #questo return mi serve solo per stampare lo stato finale dopo che ha agito lo step 
                [
                    self.time,
                    *self.stock.available.queue, # chiama la funzione queue che fa una copia  di _queue cioe' dell'oggetto della classe SizedFifo con tutte le caratteristiche annesse,available quindi viene gestito secondo le specifiche della classe
                    *self.stock.in_transit.queue, # chiama la funzione queue che fa una copia  di _queue cioe' dell'oggetto della classe SizedFifo con tutte le caratteristiche annesse, in transit quindi viene gestito secondo le specifiche della classe
                    *self.demand_history.queue,
                    *self.forecast_history.queue,
                ]
            ) """

        #definition of the step function
            # N.B the state is not defined yet
         
    def step(self, state, action, sequence_idx):
            
            if (self.time < len(self.lead_times)): # preparo la matrice delle domande
                demands = np.zeros(
                    (self._central_warehouses_num+self._local_warehouses_num, self._product_types_num),
                    dtype=np.int32)
            else:
                config_dict = get_base_config(sequence_idx)
                base_demand = config_dict["demand_fn_args"]["base"]
                demands = random.choice(base_demand)

            logger.debug(f"\n--- SupplyChainEnvironment --- step"
                     f"\nstate is "
                     f"{state}"
                     f"\nstate.factory_stocks is "
                     f"{state.factory_stocks}"
                     f"\nstate.distr_warehouses_stocks is "
                     f"{state.distr_warehouses_stocks}"
                     f"\naction is "
                     f"{action}"
                     f"\naction.production_level is "
                     f"{action.production_level}"
                     f"\naction.shipped_stocks is "
                     f"{action.shipped_stocks}"
                     f"\ndemands is "
                     f"{demands}")

        # next state
            next_state = State(self._product_types_num, self._central_warehouses_num+self._local_warehouses_num,
                           self._T,
                           list(self.lead_times), list(self.demand_history),
                           self.time+1)

            if self.lead_times_len > 0:
            # next state (distribution warehouses)
                distr_warehouses_stocks = np.minimum(
                    np.add(state.distr_warehouses_stocks,
                       self.lead_times[len(self.lead_times)-1]),
                    self.storage_capacities[1:])
                next_state.distr_warehouses_stocks = np.subtract(
                    distr_warehouses_stocks,
                    demands)

            
    
""" def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        action = np.around(action) # arrotonda l'azione all'intero piu' vicino
        if action < 0 or action > self._supply_chain_schema.max_orders: # azione problematica controllata
            raise ValueError(
                "action must be greater than zero and less than or equal to max_orders"
            )
 
        # evaluate the number of items ordered
        ordered_items = action * self._supply_chain_schema.items_per_order # numero di prodotti ordinati, quante volte ordino tot items per oder, quinti rdini faccio di un tipo di prodotto
        self._action_history.insert(ordered_items) # lo aggiungo ad action_history
 
        # advance stock
        expired_items, excess_items = self.stock.advance(ordered_items) # gli expired items vengono poi usati per calcolare i costi e rappresentano le quantita' di prodotto che sono troppo vecchie e vengono quindi scartate
 
        # evaluate unresolved demand
        cur_demand = self.demand[self.time]
        cur_forecast = self.forecast[
            self.time
            + self._supply_chain_schema.stock_schema.item_schema.lead_time # 
            + self._supply_chain_schema.len_forecast_history
        ]
        self.demand_history.insert(cur_demand)
        self.forecast_history.insert(cur_forecast)
        sold_items = self.stock.retrieve(cur_demand) #quantita' effettiva di articoli venduti: gestione nel caso in cui la richeista supera la dispobnibilita' o e' inferiore a tale disponibilit' 
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
        reward -= self.stock.sum() * self._supply_chain_schema.stock_schema.storage_cost
        reward -= (
            expired_items
            * self._supply_chain_schema.stock_schema.item_schema.expiration_cost
        ) # tolgo tutti i costi
 
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
        pass """