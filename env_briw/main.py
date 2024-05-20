import pandas as pd
from env import InventoryEnvGYConfig
import train_ppo
import train_BS
import train_sQ
import train_dqn
import eval_ppo
import eval_BS
import plot
import BS
import sQ
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from openpyxl import load_workbook


ppo = False
bs = False
sq = False
dqn = True
train = True
# eval = True


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
        env = InventoryEnvGYConfig(config)
        total_timesteps = 1000
        if ppo:
            # Train and evaluate the model
            if train:
                model = train_ppo.train_ppo_model(
                    i, env, policy_kwargs, learning_rate, total_timesteps)
                train_ppo.save_logs()
                timesteps, results = plot.load_results_and_timesteps(
                    'timesteps.pkl', 'results.pkl')
                plot.plot_reward_convergence(
                    timesteps, results, config_details)
            else:
                model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                            learning_rate=learning_rate, n_steps=32, batch_size=32,
                            n_epochs=10, clip_range=0.1, gamma=0.99, ent_coef=0.01)
                model = PPO.load(f"./logs/ppo_model_{i}")
            if eval:
                avg_reward, std_reward, avg_metrics, episodes_metrics = eval_ppo.evaluate_policy_and_log_detailed_metrics(
                    model, env, n_eval_episodes=20)
                eval_ppo.save_metrics_to_dataframe(
                    avg_metrics, config_details, avg_reward, std_reward, filename='evaluation_metrics_ppo.csv')
                plot.plot_episodes_metrics(
                    episodes_metrics, config_details, steps)
        if dqn:
            # Train and evaluate the model
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
                base_stock_level, levels, rewards = train_BS.train_bs_policy(
                    env, 5, 25, total_timesteps)
                plot.plot_rewards_per_bs_level(levels, rewards, config_details)
                df_configurations.at[i, 'base_stock_level'] = base_stock_level
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
                best_s, best_Q, levels, rewards = train_sQ.train_sQ_policy(
                    env, 5, 10, 1, 4, total_timesteps)
                # plot.plot_rewards_per_sq_level(levels, rewards, config_details)
            # else:
            #    base_stock_level = config['base_stock_level']
            # if eval:
            #    bs_class = sQ.sQpolicy(base_stock_level)
            #    avg_reward, std_reward, avg_metrics, episodes_metrics = eval_BS.evaluate_policy_and_log_detailed_metrics(
            #        env, bs_class, n_eval_episodes=20)
            #    eval_BS.save_metrics_to_dataframe(
            #        avg_metrics, config_details, avg_reward, std_reward, filename='evaluation_metrics_bs.csv')
            #    plot.plot_episodes_metrics(
            #        episodes_metrics, config_details, steps)


main()
