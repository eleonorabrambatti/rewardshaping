import numpy as np
from BS import BSpolicy


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
    levels = []
    avg_rewards = []
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
                action = bs.act(env.total_stock)  # Sample an action
                #_, reward, done, _ = env.step(action)  # Take a step in the environment
                _, reward, done, _=env.step(action)
                total_reward += reward
        average_reward = total_reward / total_timesteps
        print(f"Base Stock Level: {base_stock_level}, Average Reward: {average_reward}")
        levels.append(base_stock_level)
        avg_rewards.append(average_reward)
        if average_reward > best_average_reward:
            best_average_reward = average_reward
            best_base_stock = base_stock_level
   
    return best_base_stock, levels, avg_rewards



def fun(x, env):
    s = x
    total_reward = 0
    num_episodes_per_level = 20000  # Numero di episodi per livello
    env.reset() 
    s=np.around(s)
    bs = BSpolicy(s)
    
    for _ in range(num_episodes_per_level):
          # Reset dell'ambiente per un nuovo episodio
        env.base_stock_level = np.around(s)  # Setta il livello di base stock per questo episodio (s=s) # Imposta il livello di base stock per questo episodio
        done = False
        while not done:
            action = bs.act(env.total_stock)
            _, reward, done, _ = env.step(action)
            total_reward += reward
    
    average_reward = total_reward / num_episodes_per_level
    return -average_reward  # Restituiamo il negativo perché stiamo minimizzando

