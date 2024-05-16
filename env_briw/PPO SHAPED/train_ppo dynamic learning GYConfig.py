import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from InventoryEnvGY_Config import InventoryEnvGYConfig   # Adjusted import for your environment
from InventoryEnvGY_ConfigReshaped import InventoryEnvGYConfigRechaped   # Adjusted import for your reshaped environment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load configurations from an Excel file
excel_path = r'..\rewardshaping\configurations_ppo.xlsx'
df_configurations = pd.read_excel(excel_path, engine='openpyxl')
configurations = df_configurations.to_dict('records')

def evaluate_policy_and_log_detailed_metrics(model, env, n_eval_episodes=10):
    total_rewards = []
    metrics = {
       'expired': [],
       'stock': [],  # Will be updated to reflect last m elements sum
       'lost_sales': [],
       'satisfied': [],
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
            action=np.around(action).astype(int)
            if action != 3:
                print(f'action: {action}')
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            
            # Accumulate metrics for each step
            for key in metrics:
                if key == 'stock':
                    # Directly use the 'stock' value from info which is now consistent
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



def save_metrics_to_dataframe(metrics, config_details, avg_reward, std_reward, filename='evaluation_metrics.csv'):
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
# Adjust policy_kwargs and learning rate if needed
policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])])
learning_rate = 1e-4

# Loop through each configuration
for config_index, config in enumerate(configurations):
    
    env = InventoryEnvGYConfig(config) # Initialize environment with current configuration
    #env = InventoryEnvGYConfigRechaped(config) # Initialize environment with current configuration
    env.seed(42)  # Setting environment seed

    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=learning_rate, n_steps=32, batch_size=32,
                n_epochs=10, clip_range=0.1, gamma=0.99, ent_coef=0.01)

    # Callbacks setup
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000,
                                 deterministic=False, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='ppo_model')
    callback = CallbackList([eval_callback, checkpoint_callback])

    # Train the model
    total_timesteps = 30000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Evaluate the trained model
    mean_reward, std_reward, detailed_metrics = evaluate_policy_and_log_detailed_metrics(model, env, n_eval_episodes=20)

    # Generate a unique filename suffix from configuration for saving results
    config_str = "_".join([f"{k}_{v}" for k, v in config.items() if k != 'configuration'])
    metrics_filename = f'evaluation_metricsSUnshaped.csv'
    save_metrics_to_dataframe(detailed_metrics, config_details=config_str, avg_reward=mean_reward, std_reward=std_reward, filename=metrics_filename)
    plot_filename = f'reward_convergence_{config_str}.pdf'

    # Load the logs and save the plot
    logs = np.load('./logs/evaluations.npz')
    timesteps = logs['timesteps']
    results = logs['results']

    #if config_index in indices_to_visualize:
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