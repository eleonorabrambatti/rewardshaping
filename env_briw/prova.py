import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList,BaseCallback
from ppo_sb3 import PPO   # Adjusted import for your environment

import gym
import gymnasium
shape =float( 1 / (0.38 ** 2))
scale= float(3 / shape)
 
# Set seed for reproducibility
np.random.seed(42)

# Generate 10,000 values from gamma distribution and find the maximum
samples = np.random.gamma(shape, scale, 10000)
print(f'samples: {samples}')
max_demand = np.max(samples)
print(f'max_demand: {round(max_demand)}')