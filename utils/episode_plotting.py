import pandas as pd
import matplotlib.pyplot as plt
import json
import wandb
#import feather


def plot_episode_new(log_dir):
    """
    Plots the episode data from a given log file.

    Parameters:
    log_dir (str): The path to the log file.

    Returns:
    plot: The matplotlib plot object.
    """
    df = pd.read_csv(log_dir) 
    plot = df.plot(subplots=True, figsize=(15, len(df.columns)*3))
    return plot

def plot_training_episode(episode_num: int, monitor_dir='logs/monitor.csv'):
    """
    Plots environment states and chosen actions of any given training episode.
    
    Parameters
    ----------
    episode_num : int
        Number of the episode that will be plotted.
    monitor_dir : str, optional
        Path to the monitor.csv file. Default is 'logs/monitor.csv'.

    Raises
    ------
    AssertionError
        If the requested episode number is invalid.
        If the episode number is less than 0.

    """

    df = pd.read_csv(monitor_dir, header=1)

    assert episode_num < len(df), f"Requested episode number invalid. Max episodes collected: {len(df)-1}"
    assert episode_num >= 0, "Episode number must be larger or equal to 0"

    df['action'] = df['action'].apply(lambda x: json.loads(x))
    df['volumes'] = df['volumes'].apply(lambda x: json.loads(x))

    n_bunkers = len(df['volumes'][0][0])

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    for i in range(n_bunkers):
        vol = [x[i] for x in df['volumes'].loc[episode_num]]
        ax1.plot(vol, label=f"vol{i}")

    actions = [a for a in df['action'].loc[episode_num]]
    ax2.plot(actions, 'bo', label="Actions")

    ax1.legend()
    ax2.legend()
    plt.show()

def get_max_episodes(monitor_dir='logs/monitor.csv'):
    """
    Returns the number of episodes logged in a monitor.csv
    
    Parameters
    ----------
    monitor_dir : str, optional
        The path to the monitor.csv file (default is 'logs/monitor.csv')
    
    Returns
    -------
    int
        The number of episodes logged in the monitor.csv file
    """
    df = pd.read_csv(monitor_dir, header=1)
    return len(df)-1

def plot_volume_and_action_distributions(fig_name = "",monitor_dir='logs/monitor.csv', save_fig = False, plot_wandb = None ,  run = None ):
    """Plots the distribution (histogram) of all volumes/actions observed/performed during training.
    
    Parameters
    ----------
    fig_name : str, optional
        Name of the figure to be saved (default is "").
    monitor_dir : str, optional
        Path to the monitor.csv file (default is 'logs/monitor.csv').
    save_fig : bool, optional
        Flag to save the figure (default is False).
    plot_wandb : bool, optional
        Flag to plot the figure on Weights & Biases (default is None).
    run : str, optional
        Weights & Biases run ID (default is None).
    
    Returns
    -------
    plt : matplotlib.pyplot object
        The generated plot.
    """
    df = pd.read_csv(monitor_dir, header=1)

    df['action'] = df['action'].apply(lambda x: json.loads(x))
    df['volumes'] = df['volumes'].apply(lambda x: json.loads(x))

    n_bunkers = len(df['volumes'][0][0])

    fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(15, 12))

    for i in range(n_bunkers):
        volumes = []
        for j in range(len(df)):
            volumes += [x[i] for x in df['volumes'].loc[j]]

        ax1.hist(volumes, label=f"Volumes, Bunker {i}")

    actions = []
    for j in range(len(df)):
        actions += [a for a in df['action'].loc[j]]

    ax2.hist(actions, label="Actions")

    ax1.legend()
    ax2.legend()
    
    if save_fig:
        # Save plot
        plt.savefig(fig_name + '.jpg', dpi=fig.dpi)
    
    if plot_wandb:
        wandb.run.log({'State variables distributions during training_jpg/actions_histogram_jpb': wandb.Image(plt)})
    
    return plt