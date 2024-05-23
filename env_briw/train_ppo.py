import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train_ppo_model(env, learning_rate, total_timesteps, full_path, n_steps, batch_size, n_epochs):
    # Loop through each configuration

    env.seed(42)  # Setting environment seed

    model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                n_epochs=n_epochs)

    # Callbacks setup
    eval_callback = EvalCallback(env, best_model_save_path=f'./{full_path}/logs/', log_path=f'./{full_path}/logs/', eval_freq=1000,
                                    deterministic=False, render=False)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000, save_path=f'./{full_path}/logs/', name_prefix='ppo_model')
    callback = CallbackList([eval_callback, checkpoint_callback])

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the trained model
    model_path = os.path.join(full_path, f"./logs/ppo_model")
    model.save(model_path)

    return model

def save_logs(output_dir):
    subdir = 'pickle_file'
    full_path = os.path.join(output_dir, subdir)
    os.makedirs(full_path, exist_ok=True)
    # Load the logs and save the plot
    logs = np.load(f'./{output_dir}/logs/evaluations.npz')
    timesteps = logs['timesteps']
    results = logs['results']
    # Salva i timesteps e i risultati in due file separati in formato pkl
    timesteps_filename = os.path.join(full_path, 'timesteps.pkl')
    results_filename = os.path.join(full_path, 'results.pkl')
    # Salvataggio dei timesteps
    with open(timesteps_filename, 'wb') as f:
        pickle.dump(timesteps, f)

    # Salvataggio dei risultati
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)