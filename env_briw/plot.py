import pickle
import matplotlib.pyplot as plt
import numpy as np
import os



def plot_reward_convergence(output_dir):
    subdir_results = 'pickle_file/results.pkl'
    subdir_episodes = 'pickle_file/timesteps.pkl'
    results_path = os.path.join(output_dir, subdir_results)    
    episodes_path = os.path.join(output_dir, subdir_episodes)

    with open(episodes_path, 'rb') as file:
        episodes = pickle.load(file)
    with open(results_path, 'rb') as file:
        results = pickle.load(file)

    subdir = 'train_graph'
    plot_path = os.path.join(output_dir, subdir)
    os.makedirs(plot_path, exist_ok=True)

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards)
    plt.fill_between(episodes, mean_rewards - std_rewards,
                     mean_rewards + std_rewards, alpha=0.3)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    # Add pad for space
    plt.title(f'Reward Convergence\n', pad=20)
    plt.grid(True)
    plot_filename = os.path.join(plot_path, f'reward_convergence.pdf')
    plt.savefig(plot_filename, dpi=300)


def plot_rewards_per_bs_level(output_dir):
    subdir_levels = 'pickle_file/levels.pkl'
    subdir_rewards = 'pickle_file/avg_rewards.pkl'
    levels_path = os.path.join(output_dir, subdir_levels)    
    rewards_path = os.path.join(output_dir, subdir_rewards)
    with open(levels_path, 'rb') as file:
        levels = pickle.load(file)
    with open(rewards_path, 'rb') as file:
        rewards = pickle.load(file)

    subdir = 'train_graph'
    plot_path = os.path.join(output_dir, subdir)
    os.makedirs(plot_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(levels, rewards)
    plt.xlabel('Base Stock Level')
    plt.ylabel('Average Reward')
    # Add pad for space
    plt.title(f'Average Reward per Base Stock Level\n', pad=20)
    plt.grid(True)
    plot_filename = os.path.join(plot_path, f'reward_per_bs_level.pdf')
    plt.savefig(plot_filename, dpi=300)


def plot_rewards_per_sq_level(levels, rewards, config_details):
    plt.figure(figsize=(10, 6))
    plt.plot(levels, rewards)
    plt.xlabel('Base Stock Level')
    plt.ylabel('Average Reward')
    # Add pad for space
    plt.title(f'rewards per bs level - Config: {config_details}\n', pad=20)
    plt.grid(True)
    plot_filename = f'reward_per_bs_level_{config_details}.pdf'
    plt.savefig(plot_filename, dpi=300)


def plot_episodes_metrics(steps, output_dir):
    subdir_pickle = 'pickle_file'
    components_path = os.path.join(output_dir, subdir_pickle) 

    subdir = 'eval_graph'
    full_path = os.path.join(output_dir, subdir)
    os.makedirs(full_path, exist_ok=True)
    keys = ['demand', 'expired_items', 'stock', 'in_transit', 'lost_sales', 'satisfied_demand', 'orders', 'average_reward']

    for key in keys:
        path = os.path.join(components_path, f'{key}.pkl')
        with open(path, 'rb') as file:
            metric = pickle.load(file)
    
        plt.figure(figsize=(10, 6))
        plt.plot(steps, np.array(metric))
        # plt.fill_between(episodes, np.array(episode_metrics[key]) - np.array(episode_metrics[key]).std(axis=1), np.array(episode_metrics[key]).mean(axis=1) + np.array(episode_metrics[key]).std(axis=1), alpha=0.3)
        plt.xlabel('Steps')
        plt.ylabel(f'{key}')

        plt.title(f'{key}\n', pad=20)  # Add pad for space

        plt.grid(True)

        plot_filename = os.path.join(full_path, f'{key}.pdf')
        # Saves the plot with a dynamic name
        plt.savefig(plot_filename, dpi=300)
        plt.close()  # Close the plot explicitly to free up memory
