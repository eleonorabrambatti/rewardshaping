import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import InventoryEnvGYConfig
import pandas as pd
import os

from env import InventoryEnvGYConfig

import eval_ppo
import eval_dqn
from stable_baselines3 import PPO
from stable_baselines3 import DQN
import numpy as np
from scipy.optimize import show_options
from openpyxl import load_workbook
import json
excel_path = r'..\rewardshaping\configurations.xlsx'
config_json_path = r'..\rewardshaping\config.json'
df_configurations = pd.read_excel(excel_path, engine='openpyxl')
configurations = df_configurations.to_dict('records')
        # Leggere il file di configurazione JSON
with open(config_json_path, 'r') as file:
    json_config = json.load(file)

ppo = True
dqn=False

steps = list(range(1, 11))
# Define the search space for the GridSampler
search_space = {
    'learning_rate': [1e-5, 1e-4],
    'n_steps': [ 2048, 4096],
    'batch_size': [64,128],
    'n_epochs': [5,10],
    'buffer_size': [1000],
    'gradient_steps': [1],
    'target_update_interval': [1000]
}
# Define the objective function
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 3e-4,3e-3, step=0.00001)
    n_steps = trial.suggest_float('n_steps', 2048, 4096, step=1024)
    batch_size = trial.suggest_categorical('batch_size', [64,128])
    n_epochs = trial.suggest_float('n_epochs', 5, 10, step=5)
    buffer_size = trial.suggest_categorical('buffer_size', [1000])
    gradient_steps = trial.suggest_categorical('gradient_steps', [1])
    target_update_interval = trial.suggest_categorical('target_update_interval', [1000])

    # Use a single configuration for demonstration
    config = configurations[0]
    config_details = "_".join([f"{k}_{v}" for k, v in config.items() if k != 'configuration'])
    output_dir = f'{config_details}'
    env = InventoryEnvGYConfig(config, json_config)
    total_timesteps = 200000

    if ppo:
        # Train the model
        model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps,
                    batch_size=batch_size, n_epochs=n_epochs, verbose=0)
        model.learn(total_timesteps=total_timesteps)

        # Evaluate the model
        mean_reward = eval_ppo.evaluate_policy_and_log_detailed_metrics_2(model, env, n_eval_episodes=100)
        
    elif dqn:
        model = DQN('MlpPolicy', env, learning_rate=learning_rate, batch_size=batch_size, buffer_size=buffer_size, gradient_steps=gradient_steps,target_update_interval=target_update_interval,
                exploration_final_eps=0.05)
        model.learn(total_timesteps=total_timesteps)

        # Evaluate the model
        mean_reward = eval_dqn.evaluate_policy_and_log_detailed_metrics_2(model, env, n_eval_episodes=100)
        print(mean_reward)
    return mean_reward

# Create a study with GridSampler and MedianPruner
sampler = optuna.samplers.GridSampler(search_space)
pruner = optuna.pruners.SuccessiveHalvingPruner()

study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
study.optimize(objective, n_jobs=-1, show_progress_bar=True)


# Print the best hyperparameters
print('Best hyperparameters: ', study.best_params)