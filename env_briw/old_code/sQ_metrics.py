import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sQpolicy_env import BaseStockConfig   # Adjusted import for your environment
import pandas as pd
import matplotlib.pyplot as plt
 
from timeit import default_timer as timer
 
import csv
 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
 
# Load configurations from an Excel file
excel_path = r'..\rewardshaping\configurations_prove.xlsx'
df_configurations = pd.read_excel(excel_path, engine='openpyxl')
configurations = df_configurations.to_dict('records')
def optimize_base_stock(env, min_s, max_s, min_Q, max_Q, num_episodes_per_level):
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
    best_s = None
    best_Q=None
    for s in range(min_s, max_s + 1):

        for q in range(min_Q, max_Q + 1):
            total_reward = 0
            env.reset()  # Reset the environment for a new episode
            for i in range(num_episodes_per_level):
                #print(f' num episodes: {i}')
                #env.reset()  # Reset the environment for a new episode
                env.s = s  # Set the base stock level for this episode
                env.Q = q
                done = False
                while not done:
                    #action = env.action_space.sample()  # Sample an action
                    #_, reward, done, _ = env.step(action)  # Take a step in the environment
                    _, reward, done, _=env.Simulate_step()
                    total_reward += reward
            average_reward = total_reward / num_episodes_per_level
            print(f"s: {s}, q: {q}, Average Reward: {average_reward}")
        
            if average_reward > best_average_reward:
                best_average_reward = average_reward
                best_s = s
                best_Q=q
   
    return best_s, best_Q, best_average_reward
 
def evaluate_base_stock_performance(env, s, Q, n_eval_episodes):
    total_rewards = []
   
    # Initialize storage for reward components
    reward_components_summary = {
        'holding_cost': [],
        'lost_sales_cost': [],
        'expired_stock_cost': [],
        'ordering_cost': [],
        'Satisfied demand': [],
        'best_s' : [],
        'best_Q': [],
        'action' : []
    }
   
    # Initialize storage for sum of reward components per episode
    episode_components_sum = {
        'holding_cost': [],
        'lost_sales_cost': [],
        'expired_stock_cost': [],
        'ordering_cost': [],
        'Satisfied demand': [],
        'best_s' : [],
        'best_Q': [],
        'action' : []
    }
   
    for _ in range(n_eval_episodes):
        env.reset()
        env.s= s  # Ensure consistent naming for setting base stock level
        env.Q= Q
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
    reward_components_summary['action'] = round(float(reward_components_summary['action']/env.done))
    reward_components_summary['best_s'] = int(reward_components_summary['best_s']/env.done)
    reward_components_summary['best_Q'] = int(reward_components_summary['best_Q']/env.done)
    reward_components_summary['average_reward'] = np.mean(total_rewards)
    reward_components_summary['stdv_reward'] = np.std(total_rewards)
 
    return reward_components_summary
 
 
 
def save_metrics_to_dataframe(metrics, config_details, avg_reward, std_reward,
                              filename='evaluation_metrics_BS.csv'):
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
""" indices_to_visualize = []
 
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
  """
# Open CSV file in write mode
with open('evaluation_metrics_BS.csv', mode='w', newline='') as file:
 
    # Define fieldnames based on dictionary keys
    fieldnames = ['Configuration', 'holding_cost', 'lost_sales_cost', 'expired_stock_cost', 'ordering_cost', 'Satisfied demand', 'average_reward', 'stdv_reward','best_s','best_Q','action', 'time']
   
    # Create a CSV writer object
    writer = csv.DictWriter(file, fieldnames=fieldnames)
   
    # Write header row
    writer.writeheader()
   
    # Loop through each configuration
    for config_index, config in enumerate(configurations):
 
        start = timer()
        #print('sto entrando nella nuova configurazione')
        env = BaseStockConfig(config)  # Initialize environment with current configuration
        env.seed(42)
 
        best_s, best_Q, optimal_reward = optimize_base_stock(env, 0, 15, 0, 10, 1000)
        #print(f"Optimal Base Stock Level: {optimal_base_stock}, with an average reward of: {optimal_reward}")
        env.reset()
 
        # Evaluate the trained model
        reward_components_summary = evaluate_base_stock_performance(env, best_s, best_Q, n_eval_episodes=100)
        #print(f"Reward_components_summary: {reward_components_summary}")
        end = timer()
        # Write data to CSV file for current iteration
        writer.writerow({'Configuration': config_index,
                         'holding_cost': reward_components_summary['holding_cost'],
                         'lost_sales_cost': reward_components_summary['lost_sales_cost'],
                         'expired_stock_cost': reward_components_summary['expired_stock_cost'],
                         'ordering_cost': reward_components_summary['ordering_cost'],
                         'Satisfied demand': reward_components_summary['Satisfied demand'],
                         'average_reward': reward_components_summary['average_reward'],
                         'stdv_reward': reward_components_summary['stdv_reward'],
                         'best_s' : reward_components_summary['best_s'],
                         'best_Q' : reward_components_summary['best_Q'],
                         'action' : reward_components_summary['action'],
                         'time' : end - start})
        
        print(end - start) # Time in seconds