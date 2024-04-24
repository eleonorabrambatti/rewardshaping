import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList,BaseCallback
from ppo_sb3 import PPO   # Adjusted import for your environment

# Adjust policy_kwargs and learning rate if needed
policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])])
# In this case, it's specifying a feedforward neural network with two hidden layers for both the policy (pi) 
# and the value function (vf). Each hidden layer has 32 units. This architecture configuration can be adjusted as needed.
learning_rate = 1e-4

model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=learning_rate, n_steps=4096, batch_size=64,
                n_epochs=10, clip_range=0.1, gamma=0.99, ent_coef=0.01)