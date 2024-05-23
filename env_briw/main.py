import pandas as pd
import os
from env import InventoryEnvGYConfig
import train_ppo
import train_BS
import train_sQ
import train_dqn
import eval_ppo
import eval_BS
import eval_dqn
import plot
import BS
import sQ
import time
from stable_baselines3 import PPO
from stable_baselines3 import DQN
import scipy.optimize as optimize
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import show_options
from openpyxl import load_workbook

# Set the following flags to True to run the corresponding algorithm

import scipy.optimize as optimize
import numpy as np
from scipy.optimize import Bounds


ppo = False
dqn = False
bs = True
sq = False


train = True
plot_train = False
eval = False
plot_eval = False


def main():
    # Load configurations from an Excel file
    excel_path = r'../rewardshaping/configurations_ppo.xlsx'
    df_configurations = pd.read_excel(excel_path, engine='openpyxl')
    configurations = df_configurations.to_dict('records')

    # Adjust policy_kwargs and learning rate if needed
    policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    learning_rate = 1e-4
    steps = list(range(1, 11))
    for i, config in enumerate(configurations):
        config_details = "_".join(
            [f"{k}_{v}" for k, v in config.items() if k != 'configuration'])
        output_dir = f'{config_details}'
        env = InventoryEnvGYConfig(config)
        total_timesteps = 20000
        if ppo:

            n_steps = 500
            batch_size = 50
            n_epochs = 10

            subdir = 'ppo'
            subdir_1 = f'lr_{learning_rate}_n_steps_{n_steps}_batch_size_{batch_size}_n_epochs_{n_epochs}'
            full_path = os.path.join(output_dir, subdir, subdir_1)
            os.makedirs(full_path, exist_ok=True)
            # Train and evaluate the model
            if train:
                model = train_ppo.train_ppo_model(
                    env, learning_rate, total_timesteps, full_path, n_steps, batch_size, n_epochs)
            if plot_train:
                train_ppo.save_logs(full_path)
                plot.plot_reward_convergence(full_path)

            if eval:
                model_path = os.path.join(full_path, f"./logs/ppo_model")

                model = PPO.load(model_path)
                eval_ppo.evaluate_policy_and_log_detailed_metrics(
                    model, env, full_path, n_eval_episodes=20)

            if plot_eval:
                eval_ppo.save_metrics_to_dataframe(full_path, config_details)
                plot.plot_episodes_metrics(steps, full_path)

        elif dqn:

            batch_size = 32
            buffer_size = 1000
            gradient_steps = 1
            target_update_interval = 1000

            subdir = 'dqn'
            subdir_1 = f'lr_{learning_rate}_batch_size_{batch_size}_buffer_size_{buffer_size}_gradient_steps_{gradient_steps}_target_update_interval_{target_update_interval}'
            full_path = os.path.join(output_dir, subdir, subdir_1)
            os.makedirs(full_path, exist_ok=True)

            if train:
                model = train_dqn.train_dqn_model(
                    i, env, policy_kwargs, learning_rate, total_timesteps)
                train_dqn.save_logs()
                # timesteps, results = plot.load_results_and_timesteps(
                #    'timesteps.pkl', 'results.pkl')
                # plot.plot_reward_convergence(
                #    timesteps, results, config_details)
            else:
                model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                            learning_rate=learning_rate, n_steps=32, batch_size=32,
                            n_epochs=10, clip_range=0.1, gamma=0.99, ent_coef=0.01)
                model = DQN.load(f"./logs/ppo_model_{i}")

        elif bs:
            if train:

                bnds = [(5, 25)]
                initial_guess = 9
                res = optimize.minimize(train_BS.fun, initial_guess, args=(
                    env,), method='Powell', bounds=bnds)
                print("Best s:", np.around(res.x[0]))

                print("Best average reward:", -res.fun)
                # res = optimize.minimize(train_BS.fun,np.array([10,]), args=(env,), method='Powell', bounds=bnds)
                # base_stock_level, levels, rewards = train_BS.train_bs_policy(
                #    env, 5, 25, total_timesteps)
                # plot.plot_rewards_per_bs_level(levels, rewards, config_details)
                base_stock_level = np.around(res.x[0])
                df_configurations.at[i,
                                     'base_stock_level'] = np.around(res.x[0])
                df_configurations.to_excel(
                    excel_path, index=False, engine='openpyxl')
            else:
                base_stock_level = config['base_stock_level']
            if eval:
                bs_class = BS.BSpolicy(base_stock_level)
                avg_reward, std_reward, avg_metrics, episodes_metrics = eval_BS.evaluate_policy_and_log_detailed_metrics(
                    env, bs_class, n_eval_episodes=20)
                eval_BS.save_metrics_to_dataframe(
                    avg_metrics, config_details, avg_reward, std_reward, filename='evaluation_metrics_bs.csv')
                plot.plot_episodes_metrics(
                    episodes_metrics, config_details, steps)
        elif sq:
            if train:

                # Definiamo i limiti per s e Q
                # Cambia questi limiti in base al tuo problema
                bnds = [(0, 10), (0, 10)]

                # Eseguiamo l'ottimizzazione
                # Cambia la stima iniziale in base al tuo problema
                initial_guess = (5, 5)
                res = optimize.minimize(train_sQ.fun, initial_guess, args=(
                    env,), method='Powell', bounds=bnds, )

                print("Best s:", res.x[0])
                print("Best Q:", res.x[1])
                print("Best average reward:", -res.fun)

                # best_s, best_Q, levels, rewards = train_sQ.train_sQ_policy(
                #    env, 5, 5, 1, 4, total_timesteps)
                # plot.plot_rewards_per_sq_level(levels, rewards, config_details)
            else:
                base_stock_level = config['base_stock_level']
            # if eval:
            #    bs_class = sQ.sQpolicy(base_stock_level)
            #    avg_reward, std_reward, avg_metrics, episodes_metrics = eval_BS.evaluate_policy_and_log_detailed_metrics(
            #        env, bs_class, n_eval_episodes=20)
            #    eval_BS.save_metrics_to_dataframe(
            #        avg_metrics, config_details, avg_reward, std_reward, filename='evaluation_metrics_bs.csv')
            #    plot.plot_episodes_metrics(
            #        episodes_metrics, config_details, steps) """
