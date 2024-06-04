import numpy as np
from collections import Counter

mean_demand = 3
coef_of_var = 0.7
shape = 1 / (coef_of_var ** 2)
scale = mean_demand / shape

        
demand = np.round(np.random.gamma(shape, scale, 10000)).astype(int)
print(demand)
demand_counter = Counter(demand)
print(demand_counter)

