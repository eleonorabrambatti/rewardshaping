
import numpy as np
from itertools import chain
import logging
logging.basicConfig()
logger = logging.getLogger('LOGGING_Briw')
logger.setLevel(logging.INFO)

class State:
    """
    We choose the state vector to include all current stock levels for each
    warehouse and product type, plus the last demand values.
    """

    def __init__(self, product_types_num, distr_warehouses_num, T,
                 lead_times, demand_history, t=0):
        self.product_types_num = product_types_num
        self.factory_stocks = np.zeros(
            (self.product_types_num,),
            dtype=np.int32)
        self.distr_warehouses_num = distr_warehouses_num
        self.distr_warehouses_stocks = np.zeros(
            (self.distr_warehouses_num, self.product_types_num),
            dtype=np.int32)
        self.T = T
        self.lead_times = lead_times
        self.demand_history = demand_history
        self.t = t

        logger.debug(f"\n--- State --- __init__"
                     f"\nproduct_types_num is "
                     f"{self.product_types_num}"
                     f"\nfactory_stocks is "
                     f"{self.factory_stocks}"
                     f"\ndistr_warehouses_num is "
                     f"{self.distr_warehouses_num}"
                     f"\ndistr_warehouses_stocks is "
                     f"{self.distr_warehouses_stocks}"
                     f"\nT is "
                     f"{self.T}"
                     f"\nlead_times is "
                     f"{self.lead_times}"
                     f"\ndemand_history is "
                     f"{self.demand_history}"
                     f"\nt is "
                     f"{self.t}")

    def to_array(self):
        if len(self.lead_times) > 0:
            logger.debug(
                f"\n--- State --- to_array"
                f"\nnp.concatenate is "
                f"""{np.concatenate((
                    self.factory_stocks,
                    np.hstack(list(chain(*chain(*self.lead_times)))),
                    self.distr_warehouses_stocks.flatten(),
                    np.hstack(list(chain(*chain(*self.demand_history)))),
                    [self.t]))}""")

            return np.concatenate((
                self.factory_stocks,
                np.hstack(list(chain(*chain(*self.lead_times)))),
                self.distr_warehouses_stocks.flatten(),
                np.hstack(list(chain(*chain(*self.demand_history)))),
                [self.t]))
        else:
            logger.debug(
                f"\n--- State --- to_array"
                f"\nnp.concatenate is "
                f"""{np.concatenate((
                     self.factory_stocks,
                     self.distr_warehouses_stocks.flatten(),
                     np.hstack(list(chain(*chain(*self.demand_history)))),
                     [self.t]))}""")

            return np.concatenate((
                self.factory_stocks,
                self.distr_warehouses_stocks.flatten(),
                np.hstack(list(chain(*chain(*self.demand_history)))),
                [self.t]))

    def stock_levels(self):
        logger.debug(f"\n--- State --- stock_levels"
                     f"\nnp.concatenate is "
                     f"""{np.concatenate((
                         self.factory_stocks,
                         self.distr_warehouses_stocks.flatten()))}""")

        return np.concatenate((
            self.factory_stocks,
            self.distr_warehouses_stocks.flatten()))
    
class Action:
    """
    The action vector consists of production and shipping controls.
    """

    def __init__(self, product_types_num, distr_warehouses_num):
        self.production_level = np.zeros(
            (product_types_num,),
            dtype=np.int32)
        self.shipped_stocks = np.zeros(
            (distr_warehouses_num, product_types_num),
            dtype=np.int32)

        logger.debug(f"\n--- Action --- __init__"
                     f"\nproduction_level is "
                     f"{self.production_level}"
                     f"\nshipped_stocks is "
                     f"{self.shipped_stocks}")