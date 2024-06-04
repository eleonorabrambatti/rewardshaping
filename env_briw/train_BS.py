import numpy as np
from BS import BSpolicy
import os
import pickle


def train_bs_policy(env, output_dir, min_base_stock, max_base_stock, total_timesteps):
    subdir = 'pickle_file'
    full_path = os.path.join(output_dir, subdir)
    os.makedirs(full_path, exist_ok=True)
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
    levels = []
    avg_rewards = []
    for base_stock_level in range(min_base_stock, max_base_stock + 1):
        total_reward = 0
        env.reset()  # Reset the environment for a new episode
        bs = BSpolicy(base_stock_level)
        for _ in range(total_timesteps):
            # print(f' num episodes: {i}')
            env.reset()  # Reset the environment for a new episode
            # Set the base stock level for this episode
            env.base_stock_level = base_stock_level
            done = False
            while not done:
                action = bs.act(env.total_stock)  # Sample an action
                # _, reward, done, _ = env.step(action)  # Take a step in the environment
                _, reward, done, _ = env.step(action)
                total_reward += reward
        average_reward = total_reward / total_timesteps
        print(
            f"Base Stock Level: {base_stock_level}, Average Reward: {average_reward}")
        levels.append(base_stock_level)
        avg_rewards.append(average_reward)
        if average_reward > best_average_reward:
            best_average_reward = average_reward
            best_base_stock = base_stock_level
    
    filename = os.path.join(full_path, f'levels.pkl')
    with open(filename, 'wb') as f:
            pickle.dump(levels, f)
    filename = os.path.join(full_path, f'avg_rewards.pkl')
    with open(filename, 'wb') as f:
            pickle.dump(avg_rewards, f)
    filename = os.path.join(full_path, f'best_base_stock.pkl')
    with open(filename, 'wb') as f:
            pickle.dump(best_base_stock, f)
    return base_stock_level        



def fun(x, env):
    
    levels = []
    avg_rewards = []
    s = np.around(x)

    total_reward = 0
    num_episodes_per_level = 10  # Numero di episodi per livello
    # env.reset()
    # s = np.around(s)
    bs = BSpolicy(s)

    for _ in range(num_episodes_per_level):
        env.reset()
        # Reset dell'ambiente per un nuovo episodio
        # Setta il livello di base stock per questo episodio (s=s) # Imposta il livello di base stock per questo episodio
        env.base_stock_level = s

        done = False
        while not done:
            action = bs.act(env.total_stock)
            _, reward, done, _ = env.step(action)
            total_reward += reward

    average_reward = total_reward / num_episodes_per_level
    # print(f"Base Stock Level: {s}, Average Reward: {average_reward}")
    levels.append(s)
    avg_rewards.append(average_reward)
    # Restituiamo il negativo perché stiamo minimizzando

    return -average_reward