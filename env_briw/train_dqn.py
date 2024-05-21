import os
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train_dqn_model(config_index, env, policy_kwargs, learning_rate, total_timesteps=30000):
    # Loop through each configuration

    env.seed(42)  # Setting environment seed

    model = DQN('MlpPolicy', env)

    # Callbacks setup
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000,
                                 deterministic=False, render=False)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path='./logs/', name_prefix='dqn_model')
    callback = CallbackList([eval_callback, checkpoint_callback])

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the trained model
    model.save(f"./logs/ppo_model_{config_index}")

    return model


def save_logs():
    # Load the logs and save the plot
    logs = np.load('./logs/evaluations.npz')
    timesteps = logs['timesteps']
    results = logs['results']
    # Salva i timesteps e i risultati in due file separati in formato pkl
    timesteps_filename = 'timesteps.pkl'
    results_filename = 'results.pkl'
    # Salvataggio dei timesteps
    with open(timesteps_filename, 'wb') as f:
        pickle.dump(timesteps, f)

    # Salvataggio dei risultati
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
