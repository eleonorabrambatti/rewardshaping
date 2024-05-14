import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from BaseStockGY_Config import BaseStockGYConfig   # Adjusted import for your environment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load configurations from an Excel file
excel_path = r'\Users\mprivitera\Documents\GitHub\rewardshaping\configurations_ppo.xlsx'
df_configurations = pd.read_excel(excel_path, engine='openpyxl')
configurations = df_configurations.to_dict('records')

def evaluate_policy_and_log_detailed_metrics(model, env, n_eval_episodes=10):
    total_rewards = []
    metrics = {
       'expired_green': [],
       'expired_yellow': [],
       'stock_green': [],  # Will be updated to reflect last m elements sum
       'stock_yellow': [],
       'lost_sales_green': [],
       'lost_sales_yellow': [],
       'satisfied_green': [],
       'satisfied_yellow': [],
       'base_stock_level': [],
       'average_reward': [],
       'reward_std': [],
    }

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_rewards = 0
        episode_metrics = {key: [] for key in metrics}

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            print(f'action: {action}')
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            
            # Accumulate metrics for each step
            for key in metrics:
                if key == 'stock_green':
                    # Directly use the 'stock_green' value from info which is now consistent
                    episode_metrics[key].append(info[key])
                else:
                    episode_metrics[key].append(info.get(key, 0))

        total_rewards.append(episode_rewards)
        
        # Calculate and aggregate episode metrics
        for key in metrics:
            metrics[key].append(np.mean(episode_metrics[key]))

    # Calculate average metrics over all episodes
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    #print(f"Average Reward: {avg_reward}, Reward STD: {std_reward}")
     
    return avg_reward, std_reward, avg_metrics



def save_metrics_to_dataframe(metrics, config_details, avg_reward, std_reward, filename='BSevaluation_metrics.csv'):
    metrics['config_details'] = str(config_details)  # Add configuration details for comparison
    print(f"Average Reward before DataFrame: {metrics['average_reward']}")
    print(f"Reward STD before DataFrame: {metrics['reward_std']}")
    metrics['average_reward'] = avg_reward
    metrics['reward_std'] = std_reward
    df = pd.DataFrame([metrics])
    print(df[['average_reward', 'reward_std']])
   
   
    
    # Append if file exists; write otherwise
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
            df.to_csv(filename, mode='a', header=False, index=False)



total_configs = len(configurations)
# Calculate indices for the configurations to visualize
indices_to_visualize = []

# Add two from the beginning
indices_to_visualize.extend([0, 1] if total_configs > 1 else [0])

# Calculate middle indices
if total_configs > 4:
    middle_index1 = total_configs // 3  # Approximately one-third into the list
    middle_index2 = 2 * total_configs // 3  # Approximately two-thirds into the list
    indices_to_visualize.extend([middle_index1, middle_index2])

# Add two from the end
if total_configs > 2:
    indices_to_visualize.extend([total_configs-2, total_configs-1])

# Adjust policy_kwargs and learning rate if needed
policy_kwargs = dict(net_arch=[dict(pi=[16, 16], vf=[16, 16])])
learning_rate = 1e-5

# Loop through each configuration
for config_index, config in enumerate(configurations[-3:], start=len(configurations)-3+1):
    env = BaseStockGYConfig(config) # Initialize environment with current configuration

    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=learning_rate, n_steps=4096, batch_size=128,
                n_epochs=10, clip_range=0.1, gamma=0.99, ent_coef=0.01)

    # Callbacks setup
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='ppo_model')
    callback = CallbackList([eval_callback, checkpoint_callback])

    # Train the model
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Evaluate the trained model
    mean_reward, std_reward, detailed_metrics = evaluate_policy_and_log_detailed_metrics(model, env, n_eval_episodes=20)

    # Generate a unique filename suffix from configuration for saving results
    config_str = "_".join([f"{k}_{v}" for k, v in config.items() if k != 'configuration'])
    metrics_filename = f'BSevaluation_metrics.csv'
    save_metrics_to_dataframe(detailed_metrics, config_details=config_str, avg_reward=mean_reward, std_reward=std_reward, filename=metrics_filename)
    plot_filename = f'reward_convergence_{config_str}.pdf'

    # Load the logs and save the plot
    logs = np.load('./logs/evaluations.npz')
    timesteps = logs['timesteps']
    results = logs['results']
    
    if config_index in indices_to_visualize:
       # Generate and save the plot only for selected configurations
       plt.figure(figsize=(10, 6))
       plt.plot(timesteps, results.mean(axis=1))
       plt.fill_between(timesteps, results.mean(axis=1) - results.std(axis=1), results.mean(axis=1) + results.std(axis=1), alpha=0.3)
       plt.xlabel('Timesteps')
       plt.ylabel('Mean Reward')
       
       # Create a more spaced-out title
       config_str = "_".join([f"{k}_{v}" for k, v in config.items() if k != 'configuration'])
       plt.title(f'Reward Convergence - Config: {config_str}\n', pad=20)  # Add pad for space
       
       plt.grid(True)
       plot_filename = f'reward_convergence_{config_str}.pdf'
       plt.savefig(plot_filename, dpi=300)  # Saves the plot with a dynamic name
       plt.close()  # Close the plot explicitly to free up memory
