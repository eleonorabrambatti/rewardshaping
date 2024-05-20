import pandas as pd
import numpy as np
from env import InventoryEnvGYConfig
import train_ppo
import train_BS
import eval_ppo
import eval_BS
import plot
import BS
from stable_baselines3 import PPO


ppo = True
bs = False
sQ = False

train = False
eval = True

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
        config_details = "_".join([f"{k}_{v}" for k, v in config.items() if k != 'configuration'])
        env = InventoryEnvGYConfig(config)
        total_timesteps = 10000
        if ppo:
            # Train and evaluate the model
            if train:
                model = train_ppo.train_ppo_model(i, env, policy_kwargs, learning_rate, total_timesteps)
                train_ppo.save_logs()
                timesteps, results = plot.load_results_and_timesteps('timesteps.pkl', 'results.pkl')
                plot.plot_reward_convergence(timesteps, results, config_details)
            else:
                model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=learning_rate, n_steps=32, batch_size=32,
                n_epochs=10, clip_range=0.1, gamma=0.99, ent_coef=0.01)
                model = PPO.load(f"./logs/ppo_model_{i}")
            if eval:
                avg_reward, std_reward, avg_metrics, episodes_metrics = eval_ppo.evaluate_policy_and_log_detailed_metrics(model, env, n_eval_episodes=20)
                eval_ppo.save_metrics_to_dataframe(avg_metrics, config_details, avg_reward, std_reward, filename='evaluation_metrics_ppo.csv')
                plot.plot_episodes_metrics(episodes_metrics, config_details, steps)
        elif bs:
            if train:
                base_stock_level = train_BS.train_bs_policy(env, 5, 25, total_timesteps)
            else:
                base_stock_level = config['base_stock_level']
            bs_class = BS.BSpolicy(base_stock_level)
            avg_reward, std_reward, avg_metrics, episodes_metrics = eval_BS.evaluate_policy_and_log_detailed_metrics(env, bs_class, n_eval_episodes=20)
            eval_BS.save_metrics_to_dataframe(avg_metrics, config_details, avg_reward, std_reward, filename='evaluation_metrics_bs.csv')
            train_BS.save_logs()
            timesteps, results = plot.load_results_and_timesteps('timesteps.pkl', 'results.pkl')
            plot.plot_reward_convergence(timesteps, results, config_details)
            plot.plot_episodes_metrics(episodes_metrics, config_details, steps)
main()