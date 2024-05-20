import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_results_and_timesteps(timesteps_path, results_path):
    """
    Load timesteps, results, and configuration from files.
    """
    with open(timesteps_path, 'rb') as file:
        timesteps = pickle.load(file)
    with open(results_path, 'rb') as file:
        results = pickle.load(file)
    return timesteps, results


def plot_reward_convergence(episodes, results, config_details):

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards)
    plt.fill_between(episodes, mean_rewards - std_rewards,
                     mean_rewards + std_rewards, alpha=0.3)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    # Add pad for space
    plt.title(f'Reward Convergence - Config: {config_details}\n', pad=20)
    plt.grid(True)
    plot_filename = f'reward_convergence_{config_details}.pdf'
    plt.savefig(plot_filename, dpi=300)


def plot_rewards_per_bs_level(levels, rewards, config_details):
    plt.figure(figsize=(10, 6))
    plt.plot(levels, rewards)
    plt.xlabel('Base Stock Level')
    plt.ylabel('Average Reward')
    # Add pad for space
    plt.title(f'rewards per bs level - Config: {config_details}\n', pad=20)
    plt.grid(True)
    plot_filename = f'reward_per_bs_level_{config_details}.pdf'
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


def plot_episodes_metrics(episode_metrics, config_details, steps):
    for key in episode_metrics.keys():
        if key != 'reward_sd':
            plt.figure(figsize=(10, 6))
            plt.plot(steps, np.array(episode_metrics[key]))
            # plt.fill_between(episodes, np.array(episode_metrics[key]) - np.array(episode_metrics[key]).std(axis=1), np.array(episode_metrics[key]).mean(axis=1) + np.array(episode_metrics[key]).std(axis=1), alpha=0.3)
            plt.xlabel('Steps')
            plt.ylabel(f'{key}')

            plt.title(f'{key} - Config: {config_details}\n',
                      pad=20)  # Add pad for space

            plt.grid(True)
            plot_filename = f'{key}_{config_details}.pdf'
            # Saves the plot with a dynamic name
            plt.savefig(plot_filename, dpi=300)
            plt.close()  # Close the plot explicitly to free up memory
