# Contains some helper plotting functions for q2.py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Function to display the frozen lake
def display_lake(lake):
    N = len(lake)
    
    # Create a grid representation using numbers for visualization
    lake_grid = np.zeros((N, N))
    
    # Assign values based on the type of cell
    for i in range(N):
        for j in range(N):
            if lake[i][j] == 'S':  # Start
                lake_grid[i, j] = 0
            elif lake[i][j] == 'G':  # Goal
                lake_grid[i, j] = 1
            elif lake[i][j] == 'H':  # Hole
                lake_grid[i, j] = 2
            elif lake[i][j] == 'F':  # Frozen (safe path)
                lake_grid[i, j] = 3

    # Create a color map
    cmap = mcolors.ListedColormap(['blue', 'green', 'black', 'lightgray'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Display the lake as an image
    plt.imshow(lake_grid, cmap=cmap, norm=norm)

    # Add grid lines for clarity
    plt.grid(which='major', color='k', linestyle='-', linewidth=2)
    plt.xticks(np.arange(-0.5, N, 1), [])  # Hide x-axis ticks
    plt.yticks(np.arange(-0.5, N, 1), [])  # Hide y-axis ticks

    # Annotate the start and goal positions
    for i in range(N):
        for j in range(N):
            if lake[i][j] == 'S':
                plt.text(j, i, 'S', ha='center', va='center', fontsize=12, color='white')
            elif lake[i][j] == 'G':
                plt.text(j, i, 'G', ha='center', va='center', fontsize=12, color='white')

    # Display the lake as a grid
    plt.title('Frozen Lake')
    plt.show()


# Plot the results of total rewards per episode
def plot_results(rewards_per_episode):
    plt.figure(figsize=(15, 8))
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-learning: Episode vs Total Reward')
    plt.ylim(min(rewards_per_episode) - 0.1, 1.1)
    plt.yticks(np.arange(min(rewards_per_episode), 1.1, step=1))
    plt.show()


# Plot the results of average rewards and steps over previous window_size episodes
def plot_average_metrics(rewards_per_episode, steps_per_episode, window_size=50):
    # Calculate the moving average of rewards and steps
    average_rewards = np.convolve(rewards_per_episode, np.ones(window_size) / window_size, mode='valid')
    average_steps = np.convolve(steps_per_episode, np.ones(window_size) / window_size, mode='valid')

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot average rewards
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(window_size - 1, len(rewards_per_episode)), average_rewards, label='Average Reward (Last 50 Episodes)', color='blue')
    plt.title('Average Reward and Average Steps Over Last 50 Episodes')
    plt.xlabel('Episode Count')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid()

    # Plot average steps
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(window_size - 1, len(steps_per_episode)), average_steps, label='Average Steps (Last 50 Episodes)', color='orange')
    plt.xlabel('Episode Count')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.grid()

    # Show the plots
    plt.tight_layout()
    plt.show()

# Plot the results of average rewards and steps over previous window_size episodes with a varying parameter
def plot_metrics_for_varying_param(rewards_data, steps_data, params, param_name, window_size=50):
    
    # Create a figure for average rewards
    plt.figure(figsize=(12, 6))

    # Plot average rewards for each alpha
    for i, param in enumerate(params):
        # Calculate the moving average of rewards
        average_rewards = np.convolve(rewards_data[i], np.ones(window_size) / window_size, mode='valid')

        # Plot average rewards for this alpha
        plt.plot(np.arange(window_size - 1, len(rewards_data[i])), average_rewards, 
                 label=f'{param_name} = {param}', linestyle='-', alpha=0.7)

    # Adding titles and labels for rewards plot
    plt.title(f'Average Reward Over Last 50 Episodes for Different {param_name} Values')
    plt.xlabel('Episode Count')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid()

    # Show the plot for rewards
    plt.tight_layout()
    plt.show()

    # Create a second figure for average steps
    plt.figure(figsize=(12, 6))

    # Plot average steps for each alpha
    for i, param in enumerate(params):
        # Calculate the moving average of steps
        average_steps = np.convolve(steps_data[i], np.ones(window_size) / window_size, mode='valid')

        # Plot average steps for this alpha
        plt.plot(np.arange(window_size - 1, len(steps_data[i])), average_steps, 
                 label=f'{param_name} = {param}', linestyle='--', alpha=0.7)

    # Adding titles and labels for steps plot
    plt.title(f'Average Steps Over Last 50 Episodes for Different {param_name} Values')
    plt.xlabel('Episode Count')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.grid()

    # Show the plot for steps
    plt.tight_layout()
    plt.show()

