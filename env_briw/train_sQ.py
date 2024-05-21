import numpy as np
from sQ import sQpolicy
import numpy as np


def train_sQ_policy(env, min_s, max_s, min_Q, max_Q, num_episodes_per_level):
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
    best_Q = None
    levels = []
    avg_rewards = []
    for s in range(min_s, max_s + 1):

        for q in range(min_Q, max_Q + 1):
            total_reward = 0
            env.reset()  # Reset the environment for a new episode
            sq = sQpolicy(s, q)
            for _ in range(num_episodes_per_level):
                # print(f' num episodes: {i}')
                # env.reset()  # Reset the environment for a new episode
                env.s = s  # Set the base stock level for this episode
                env.Q = q
                done = False
                while not done:
                    action = sq.act(env.total_stock)
                    # action = env.action_space.sample()  # Sample an action
                    # _, reward, done, _ = env.step(action)  # Take a step in the environment
                    _, reward, done, _ = env.step(action)
                    total_reward += reward
            average_reward = total_reward / num_episodes_per_level
            print(f"s: {s}, q: {q}, Average Reward: {average_reward}")
            levels.append(q)
            avg_rewards.append(average_reward)
            if average_reward > best_average_reward:
                best_average_reward = average_reward
                best_s = s
                best_Q = q

    return best_s, best_Q, levels, avg_rewards



def fun(x, env):
    s, q = x
    s=np.around(s)
    q=np.around(q)
    total_reward = 0
    num_episodes_per_level = 1000  # Numero di episodi per livello
    sq = sQpolicy(s, q)
    env.reset()
    for _ in range(num_episodes_per_level):
         # Reset dell'ambiente per un nuovo episodio
        env.s = s
        env.Q = q
        done = False
        while not done:
            action = sq.act(env.total_stock)
            _, reward, done, _ = env.step(action)
            total_reward += reward
    
    average_reward = total_reward / num_episodes_per_level
    return -average_reward  # Restituiamo il negativo perché stiamo minimizzando