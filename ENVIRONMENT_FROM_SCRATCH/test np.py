import numpy as np
from BaseStockGY_Config import BaseStockGYConfig 

# Configuration directly defined
config = {
    'm': 3,
    'L': 2,
    'Alpha': 0,
    'beta': 0,
    'p1': 0,
    'p2': 0,
    'c': 0,
    'h': 0.5,
    'b1': 20,
    'b2': 0,
    'w': 2,
    'mean_demand': 4,
    'coef_of_var': 0.001
}

# Initialize the environment with the provided configuration
env = BaseStockGYConfig(config)
env.seed(42)  # Set seed for reproducibility

# Set base stock level to 50 and reset the environment
env.base_stock_level = 50
env.reset()

done = False
while not done:
    _, reward, done, info = env.Simulate_step()  # Simulate a step using the predefined method


