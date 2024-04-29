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
 
gym.envs.register(
    id='Pippo-v0',
    entry_point='ppo_env:InventoryEnvConfig',  # Sostituisci 'your_module_name' con il nome del tuo modulo Python
)

# List all available Gymnasium environments
env_ids = [env.id for env in gym.envs.registry.all()]
 
print("Available Gymnasium environments:")
for env_id in env_ids:
    print(env_id)