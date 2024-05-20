import numpy as np
from BS import BSpolicy
import pickle

def train_bs_policy(env, min_base_stock, max_base_stock, total_timesteps):
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
        bs = BSpolicy(base_stock_level)
        for i in range(total_timesteps):
            #print(f' num episodes: {i}')
            #env.reset()  # Reset the environment for a new episode
            env.base_stock_level = base_stock_level  # Set the base stock level for this episode
            done = False
            while not done:
                action = bs.act(env.stock)  # Sample an action
                #_, reward, done, _ = env.step(action)  # Take a step in the environment
                _, reward, done, _=env.step(action)
                total_reward += reward
        average_reward = total_reward / total_timesteps
        print(f"Base Stock Level: {base_stock_level}, Average Reward: {average_reward}")
       
        if average_reward > best_average_reward:
            best_average_reward = average_reward
            best_base_stock = base_stock_level
   
    return best_base_stock

def save_logs():
    # Load the logs and save the plot
    logs = np.load('./logs/evaluations.npz')
    timesteps = logs['timesteps']
    results = logs['results']
    config_str = logs['config_detail']
    # Salva i timesteps e i risultati in due file separati in formato pkl
    timesteps_filename = 'timesteps.pkl'
    results_filename = 'results.pkl'
    config_filename = 'config_str.pkl'
    # Salvataggio dei timesteps
    with open(timesteps_filename, 'wb') as f:
        pickle.dump(timesteps, f)

    # Salvataggio dei risultati
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

    # Salvataggio delle config
    with open(config_filename, 'wb') as f:
        pickle.dump(config_str, f)