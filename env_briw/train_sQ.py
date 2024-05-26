import numpy as np
from sQ import sQpolicy
import numpy as np
import os
import pickle


def train_sQ_policy(env, min_s, max_s, min_Q, max_Q, output_dir, num_episodes_per_level):
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
    best_s = None
    best_Q = None
    levels = []
    avg_rewards = []
    for s in range(min_s, max_s + 1):

        for q in range(min_Q, max_Q + 1):
            total_reward = 0
            # env.reset()  # Reset the environment for a new episode
            sq = sQpolicy(s, q)
            for _ in range(num_episodes_per_level):
                # print(f' num episodes: {i}')
                env.reset()  # Reset the environment for a new episode
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
            levels.append([s, q])
            avg_rewards.append(average_reward)
            if average_reward > best_average_reward:
                best_average_reward = average_reward
                best_s = s
                best_Q = q

    filename = os.path.join(full_path, f'levels.pkl')
    with open(filename, 'wb') as f:
            pickle.dump(levels, f)
    filename = os.path.join(full_path, f'avg_rewards.pkl')
    with open(filename, 'wb') as f:
            pickle.dump(avg_rewards, f)
    filename = os.path.join(full_path, f'best_s.pkl')
    with open(filename, 'wb') as f:
            pickle.dump(best_s, f)
    filename = os.path.join(full_path, f'best_Q.pkl')
    with open(filename, 'wb') as f:
            pickle.dump(best_Q, f)


def fun(x, env):
    s, q = np.around(x)
    # print(s)
    # print(q)
    # s = np.around(s)
    # q = np.around(q)
    total_reward = 0
    num_episodes_per_level = 1000  # Numero di episodi per livello
    # env.reset()
    sq = sQpolicy(s, q)
    env.s = s
    env.Q = q
    for _ in range(num_episodes_per_level):
        env.reset()  # Reset dell'ambiente per un nuovo episodio
        done = False
        while not done:
            action = sq.act(env.total_stock)
            _, reward, done, _ = env.step(action)
            total_reward += reward

    average_reward = total_reward / num_episodes_per_level
    return -average_reward  # Restituiamo il negativo perché stiamo minimizzando
