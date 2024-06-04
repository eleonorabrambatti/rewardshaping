import numpy as np
import pandas as pd
import os
import pickle


def evaluate_policy_and_log_detailed_metrics(env, sq_class, output_dir, n_eval_episodes=10):
    subdir = 'pickle_file'
    full_path = os.path.join(output_dir, subdir)
    os.makedirs(full_path, exist_ok=True)
    total_rewards = []
    metrics = {
        'demand': [],
        'expired_items': [],
        'stock': [],  # Will be updated to reflect last m elements sum
        'in_transit': [],
        'lost_sales': [],
        'satisfied_demand': [],
        'orders': [],
        'reward': [],
        'reward_std': [],
    }
    episodes = {key: [] for key in metrics}
    metrics_sd = {key: [] for key in metrics}
    std_devs = {key: [] for key in metrics}
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_rewards = 0
        episode_metrics = {key: [] for key in metrics}

        while not done:
            action = sq_class.act(env.total_stock)
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
            metrics_sd[key].append(episode_metrics[key])

    # Calculate average metrics over all episodes
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    filename = os.path.join(full_path, f'avg_metrics.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(avg_metrics, f)
    avg_reward = np.mean(total_rewards)
    filename = os.path.join(full_path, f'avg_reward.pkl')
    with open(filename, 'wb') as f:
                pickle.dump(avg_reward, f)
    std_reward = np.std(total_rewards)
    filename = os.path.join(full_path, f'std_reward.pkl')
    with open(filename, 'wb') as f:
                pickle.dump(std_reward, f)
    # print(f"Average Reward: {avg_reward}, Reward STD: {std_reward}")
    for key in episodes:
        episodes[key] = [x/n_eval_episodes for x in episodes[key]]

        if key != 'reward_std':
            value=episodes[key]
            filename = os.path.join(full_path, f'{key}.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(value, f)

        for i in range(0,10):
             values = []
             for k in range(0,len(metrics_sd[key])):
                values.append(metrics_sd[key][k][i])
             std_values = np.std(values)
             std_devs[key].append(std_values)
        
        value=std_devs[key]
        filename = os.path.join(full_path, f'{key}_std.pkl')
        with open(filename, 'wb') as f:
                pickle.dump(value, f)




def save_metrics_to_dataframe(output_dir, config_details):
    subdir = 'pickle_file'
    full_path = os.path.join(output_dir, subdir)
    os.makedirs(full_path, exist_ok=True)

    subdir_avg_metrics = 'pickle_file/avg_metrics.pkl'
    subdir_avg_reward = 'pickle_file/avg_reward.pkl'
    subdir_std_reward = 'pickle_file/std_reward.pkl'
    subdir_time = 'pickle_file/time.pkl'
    avg_metrics_path = os.path.join(output_dir, subdir_avg_metrics)    
    avg_reward_path = os.path.join(output_dir, subdir_avg_reward)
    std_reward_path = os.path.join(output_dir, subdir_std_reward)
    time_path = os.path.join(output_dir, subdir_time)

    with open(avg_metrics_path, 'rb') as file:
        metrics_dict = pickle.load(file)
    with open(avg_reward_path, 'rb') as file:
        avg_reward = pickle.load(file)
    with open(std_reward_path, 'rb') as file:
        std_reward = pickle.load(file)
    with open(time_path, 'rb') as file:
        time = pickle.load(file)

    metrics_dict['config_details'] = str(config_details)
    metrics_dict['avg_reward'] = avg_reward
    metrics_dict['std_reward'] = std_reward
    metrics_dict['time'] = time


    filename='evaluation_metrics_sq.csv'
    df = pd.DataFrame([metrics_dict])
    filename = os.path.join(output_dir, filename)
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)
