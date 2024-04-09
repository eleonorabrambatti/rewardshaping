import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList,BaseCallback
from env_chaaben import InventoryEnvGYConfig   # Adjusted import for your environment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load configurations from an Excel file
excel_path = r'C:\Users\mprivitera\Documents\GitHub\rewardshaping\ENVIRONMENT_FROM_SCRATCH\configurations.xlsx'
df_configurations = pd.read_excel(excel_path, engine='openpyxl')
configurations = df_configurations.to_dict('records')

def evaluate_policy_and_log_detailed_metrics(model, env, n_eval_episodes):
    total_rewards = []
    metrics = {
        # your metrics initialization
    }
    order_quantities = []  # List to store order quantities for each step

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_rewards = 0
        episode_metrics = {key: [] for key in metrics}
        episode_order_quantities = []  # Store order quantities for this episode

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            
            # Log the order quantity for this step
            if 'order_quantity' in info:
                order_quantity = int(info['order_quantity']) 
                episode_order_quantities.append(order_quantity)

            # Accumulate metrics for each step
            for key in metrics:
                if key in info:
                    episode_metrics[key].append(info[key])

        total_rewards.append(episode_rewards)
        order_quantities.append(episode_order_quantities)  # Add this episode's orders to the list
        
        # Calculate and aggregate episode metrics
        for key in metrics:
            metrics[key].append(np.mean(episode_metrics[key]))

    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    # After evaluation, print or analyze order quantities
    print("Order quantities during evaluation:")
    for ep_num, quantities in enumerate(order_quantities, start=1):
        print(f"Episode {ep_num}: {quantities}")
    
    print(f"Average Reward: {avg_reward}, Reward STD: {std_reward}")
     
    return avg_reward, std_reward, avg_metrics

class WarmupCallback(BaseCallback):
    def __init__(self, warmup_steps=10000, start_prob=1.0, end_prob=0.1, verbose=0):
        super(WarmupCallback, self).__init__(verbose)
        self.warmup_steps = warmup_steps
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.step = 0

    def _on_step(self) -> bool:
        # Gradually decrease the probability of taking a random action
        prob = np.linspace(self.start_prob, self.end_prob, self.warmup_steps)
        current_prob = prob[min(self.step, self.warmup_steps - 1)]
        if np.random.rand() < current_prob:
            # Override the action with a random action
            # Note: You'll need to adjust this depending on how you access the environment
            self.training_env.env_method('override_action_with_random')
        self.step += 1
        return True


def save_metrics_to_dataframe(metrics, config_details, avg_reward, std_reward, filename='evaluation_metrics.csv'):
    metrics['config_details'] = str(config_details)  # Add configuration details for comparison
    print(f'metrics dictionary: {metrics}')
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
policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])])
learning_rate = 1e-5

# Loop through each configuration
for config_index, config in enumerate(configurations):
    env = InventoryEnvGYConfig(config) # Initialize environment with current configuration
    env.reset()

    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                learning_rate=learning_rate, n_steps=4096, batch_size=64,
                n_epochs=10, clip_range=0.1, gamma=0.99, ent_coef=0.01)
    
    #warmup_callback = WarmupCallback(warmup_steps=10000, start_prob=1.0, end_prob=0.1)


    # Callbacks setup
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000,
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='ppo_model')
    callback = CallbackList([eval_callback, checkpoint_callback])

    # Train the model
    total_timesteps = 300000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Evaluate the trained model
    mean_reward, std_reward, detailed_metrics =evaluate_policy_and_log_detailed_metrics(model,
                                    env,n_eval_episodes=2000)

    # Generate a unique filename suffix from configuration for saving results
    config_str = "_".join([f"{k}_{v}" for k, v in config.items() if k != 'configuration'])
    metrics_filename = f'evaluation_metrics.csv'
    """  save_metrics_to_dataframe(detailed_metrics, config_details=config_str, avg_reward=mean_reward,
                              std_reward=std_reward, filename=metrics_filename) """
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
