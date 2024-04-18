import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BaseStockGY_Config import BaseStockGYConfig   # Adjusted import for your environment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load configurations from an Excel file
excel_path = r'C:\Users\mprivitera\Documents\GitHub\rewardshaping\ENVIRONMENT_FROM_SCRATCH\configurations 1.xlsx'
df_configurations = pd.read_excel(excel_path, engine='openpyxl')
configurations = df_configurations.to_dict('records')
def optimize_base_stock(env, min_base_stock, max_base_stock, num_episodes_per_level):
    """
    Find the optimal base stock level that maximizes rewards.
    
    Args:
        env: The inventory management environment.
        min_base_stock (int): The minimum base stock level to evaluate.
        max_base_stock (int): The maximum base stock level to evaluate.
        num_episodes_per_level (int): The number of episodes to simulate for each stock level.
        
    Returns:
        int: The optimal base stock level.
        float: The average reward of the optimal level.
    """
    best_average_reward = -np.inf
    best_base_stock = None
    for base_stock_level in range(min_base_stock, max_base_stock + 1):
        total_reward = 0
        env.reset()  # Reset the environment for a new episode
        for i in range(num_episodes_per_level):
            print(f' num episodes: {i}')
            #env.reset()  # Reset the environment for a new episode
            env.base_stock_level = base_stock_level  # Set the base stock level for this episode
            done = False
            while not done:
                #action = env.action_space.sample()  # Sample an action
                #_, reward, done, _ = env.step(action)  # Take a step in the environment
                _, reward, done, _=env.Simulate_step()
                total_reward += reward
        average_reward = total_reward / num_episodes_per_level
        print(f"Base Stock Level: {base_stock_level}, Average Reward: {average_reward}")
        
        if average_reward > best_average_reward:
            best_average_reward = average_reward
            best_base_stock = base_stock_level
    
    return best_base_stock, best_average_reward

def evaluate_base_stock_performance(env, base_stock_level, n_eval_episodes):
    total_rewards = []
    
    # Initialize storage for reward components
    reward_components_summary = {
        'holding_cost': [],
        'lost_sales_cost': [],
        'expired_stock_cost': [],
        'ordering_cost': [],
         'Satisfied demand': []
    }
    
    # Initialize storage for sum of reward components per episode
    episode_components_sum = {
        'holding_cost': [],
        'lost_sales_cost': [],
        'expired_stock_cost': [],
        'ordering_cost': [],
         'Satisfied demand': []
    }
    
    for episode in range(n_eval_episodes):
        env.reset()
        env.base_stock_level = base_stock_level  # Ensure consistent naming for setting base stock level
        done = False
        episode_rewards = []
        temp_components = {key: [] for key in reward_components_summary}  # Temporary storage for this episode's components
        
        while not done:
            _, reward, done, info = env.Simulate_step()
            episode_rewards.append(reward)
            
            # Extract and temporarily store reward components from the info dictionary for this episode
            for component, value in info['reward_components'].items():
                temp_components[component].append(value)

        # Aggregate total rewards for the episode
        total_rewards.append(sum(episode_rewards))
        
        # Sum and store each component for the episode
        for component in temp_components:
            episode_components_sum[component].append(sum(temp_components[component]))
        
    # Compute average of the components sums and total rewards over all episodes
    for component in reward_components_summary.keys():
        reward_components_summary[component] = np.mean(episode_components_sum[component])

    reward_components_summary['average_reward'] = np.mean(total_rewards)
    reward_components_summary['stdv_reward'] = np.std(total_rewards)

    return reward_components_summary



def save_metrics_to_dataframe(metrics, config_details, avg_reward, std_reward,
                              filename='BSevaluation_metrics.csv'):
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


# Loop through each configuration
for config_index, config in enumerate(configurations):
    env = BaseStockGYConfig(config) # Initialize environment with current configuration
    env.seed(42)


    optimal_base_stock, optimal_reward = optimize_base_stock(env, 0,
                                                             50, 3)
    print(f"Optimal Base Stock Level: {optimal_base_stock}, with an average reward of: {optimal_reward}")
    env.reset()
 
    # Evaluate the trained model
    reward_components_summary = evaluate_base_stock_performance(env, optimal_base_stock,n_eval_episodes=1000)
    print(f"Ereward_components_summary: {reward_components_summary }")
    

    break
    # Generate a unique filename suffix from configuration for saving results
    config_str = "_".join([f"{k}_{v}" for k, v in config.items() if k != 'configuration'])
    metrics_filename = f'BSevaluation_metrics.csv'
    save_metrics_to_dataframe(detailed_metrics, config_details=config_str,
                              avg_reward=mean_reward, std_reward=std_reward,
                              filename=metrics_filename)
    plot_filename = f'reward_convergence_{config_str}.pdf'

    break
    # Load the logs and save the plot
    logs_1 = np.load('./logs_1/evaluations.npz')
    timesteps = logs_1['timesteps']
    results = logs_1['results']
    
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
