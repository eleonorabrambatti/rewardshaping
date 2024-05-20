import numpy as np
import pandas as pd
import os


def evaluate_policy_and_log_detailed_metrics(env, bs_class, n_eval_episodes=10):
    total_rewards = []
    metrics = {
        'demand': [],
        'expired_items': [],
        'stock': [],  # Will be updated to reflect last m elements sum
        'in_transit': [],
        'lost_sales': [],
        'satisfied_demand': [],
        'orders': [],
        'average_reward': [],
        'reward_std': [],
    }
    episodes = {key: [] for key in metrics}
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_rewards = 0
        episode_metrics = {key: [] for key in metrics}

        while not done:
            action = bs_class.act(env.total_stock)
            action = np.around(action).astype(int)
            # if action != 3:
            #    print(f'action: {action}')
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

        for key in metrics:
            if episode == 0:
                episodes[key] = episode_metrics[key]
            else:
                episodes[key] = [
                    x+y for x, y in zip(episodes[key], episode_metrics.get(key, [0]*len(episodes[key])))]

        # Calculate and aggregate episode metrics
            metrics[key].append(np.mean(episode_metrics[key]))

    # Calculate average metrics over all episodes
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    # print(f"Average Reward: {avg_reward}, Reward STD: {std_reward}")
    for key in episodes:
        episodes[key] = [x/n_eval_episodes for x in episodes[key]]

    return avg_reward, std_reward, avg_metrics, episodes


def save_metrics_to_dataframe(metrics_dict, config_details, avg_reward, std_reward, filename='evaluation_metrics.csv'):
    metrics_dict['config_details'] = str(config_details)
    metrics_dict['avg_reward'] = avg_reward
    metrics_dict['std_reward'] = std_reward

    df = pd.DataFrame([metrics_dict])

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)
