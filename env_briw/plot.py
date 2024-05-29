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


def plot_rewards_per_sq_level(output_dir):
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
    plt.xlabel('sQ Level')
    plt.ylabel('Average Reward')
    # Add pad for space
    plt.title(f'Reward per sQ Level\n', pad=20)
    plt.grid(True)
    plot_filename = os.path.join(plot_path, f'reward_per_sq_level.pdf')
    plt.savefig(plot_filename, dpi=300)


def plot_episodes_metrics(steps, output_dir):
    subdir_pickle = 'pickle_file'
    components_path = os.path.join(output_dir, subdir_pickle) 

    subdir = 'eval_graph'
    full_path = os.path.join(output_dir, subdir)
    os.makedirs(full_path, exist_ok=True)
    keys = ['demand', 'expired_items', 'stock', 'in_transit', 'lost_sales', 'satisfied_demand', 'orders', 'reward']
    graph_label_y = {'demand': 'Average Demand', 
                   'expired_items': 'Average Expired Items',
                    'stock': 'Average Stock',
                    'in_transit': 'Average In Transit Items', 
                    'lost_sales': 'Average Lost Sales',
                    'satisfied_demand': 'Average Satisfied Demand',
                    'orders': 'Average Orders',
                    'reward': 'Average Reward'}
    graph_label_title = {'demand': 'Average Demand per Step', 
                   'expired_items': 'Average Expired Items per Step',
                    'stock': 'Average Stock per Step',
                    'in_transit': 'Average In Transit Items per Step', 
                    'lost_sales': 'Average Lost Sales per Step',
                    'satisfied_demand': 'Average Satisfied Demand per Step',
                    'orders': 'Average Orders per Step',
                    'reward': 'Average Reward per Step'}

    fig, axs = plt.subplots(8, 1, figsize=(9, 6*5))
    colors = ['red', 'orange', 'yellow','green', 'blue', 'purple', 'pink','brown']
    for i, key in enumerate(keys):
        path = os.path.join(components_path, f'{key}.pkl')
        with open(path, 'rb') as file:
            metric = pickle.load(file)
        path_std = os.path.join(components_path, f'{key}_std.pkl')
        with open(path_std, 'rb') as file:
            metric_std = pickle.load(file)
        axs[i].plot(steps, np.array(metric), color=colors[i % len(colors)])
        axs[i].fill_between(steps, np.array(metric) - np.array(metric_std), np.array(metric) + np.array(metric_std), alpha=0.3, color=colors[i % len(colors)])
        axs[i].set_xlabel('Steps')
        axs[i].set_ylabel(f'{graph_label_y.get(key,0)}')
        axs[i].set_title(f'{graph_label_title.get(key,0)}\n', pad=5)  # Add pad for space
        axs[i].grid(True)
    plt.subplots_adjust(hspace=0.5)  # Add this line to adjust the vertical spacing between subplots
    plot_filename = os.path.join(full_path, 'metrics_subplot.pdf')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)  # Close the plot explicitly to free up memory
