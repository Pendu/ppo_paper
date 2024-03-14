import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import matplotlib.pyplot as plt
import wandb


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model that achieves a new best training reward.

    This callback checks the training reward every specified number of steps and saves the model if its performance has improved.
    It is recommended to use this in conjunction with the ``EvalCallback`` for more robust evaluation of the model's performance.

    Parameters
    ----------
    check_freq : int
        How often to check for improvement, in terms of training steps.
    log_dir : str
        Path to the directory where the training logs and model will be saved. This directory must contain the file created by the ``Monitor`` wrapper.
    verbose : int, optional
        Level of verbosity. 0 for no output, 1 for detailed output. Default is 1.

    Attributes
    ----------
    save_path : str
        The path where the best model will be saved.
    best_mean_reward : float
        The highest mean reward achieved by the model so far. Initialized to negative infinity.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        """
        Initialize the callback.

        Parameters
        ----------
        check_freq : int
            How often to check for improvement, in terms of training steps.
        log_dir : str
            Path to the directory where the training logs and model will be saved.
        verbose : int, optional
            Level of verbosity.
        """
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        """
        Initialize the callback by creating the directory where the best model will be saved, if it does not already exist.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every step to check if the current step is a check step and, if so, evaluates the model's performance.

        Returns
        -------
        bool
            Always returns True to continue training.
        """
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-10:])  # make this value smaller
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


class ProgressBarCallback(BaseCallback):
    """
    Callback for displaying a progress bar during training.

    Parameters
    ----------
    pbar : tqdm.pbar
        Progress bar object.

    """

    def __init__(self, pbar):
        """
        Initialize the ProgressBarCallback.

        Parameters
        ----------
        pbar : tqdm.pbar
            Progress bar object.
        """
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        """
        Update the progress bar at each step.
        """
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


class ProgressBarManager(object):
    """
    Manager for the progress bar during training or evaluation.

    This class is designed to be used with a `with` statement, ensuring that the progress bar is properly created and
    destroyed upon completion.

    Parameters
    ----------
    total_timesteps : int
        The total number of timesteps for which the progress bar will track progress.

    Attributes
    ----------
    pbar : tqdm or None
        The progress bar object. Initialized as `None` and set upon entering the context.
    total_timesteps : int
        The total number of timesteps the progress bar will track.
    """

    def __init__(self, total_timesteps):
        """
        Initializes the ProgressBarManager with the total number of timesteps.

        Parameters
        ----------
        total_timesteps : int
            The total number of timesteps for which the progress bar will track progress.
        """
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):
        """
        Creates and returns a progress bar and its associated callback upon entering the context.

        Returns
        -------
        ProgressBarCallback
            The callback associated with the progress bar to be used during training or evaluation.
        """
        self.pbar = tqdm(total=self.total_timesteps)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures proper closure of the progress bar upon exiting the context.
        """
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


class EnvLogger(BaseCallback):
    """
    Callback for logging episodes of the environment during training.

    Parameters
    ----------
    log_frequency : int
        How many episodes to wait before logging an episode. (1 -> log every episode, 5 -> log every 5th episode)
    log_dir : str
        Directory where the logs will be saved.

    """

    def __init__(self, log_frequency, log_dir):
        """
        Initialize the EnvLogger callback.

        Parameters
        ----------
        log_frequency : int
            How many episodes to wait before logging an episode.
        log_dir : str
            Directory where the logs will be saved.
        """
        super(EnvLogger, self).__init__()
        self.log_frequency = log_frequency
        self.log_dir = log_dir
        # create dir if not exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.episode_num = 1

    def _init_callback(self) -> None:
        """
        Initialize the callback by setting up the logging infrastructure.
        """
        bunkers_raw = self.model.env.get_attr("bunker_ids")
        self.bunkers = [x for x, y in bunkers_raw[0]]
        # Create output frame
        self.df = pd.DataFrame(columns=['action', 'reward'] + self.bunkers)

    def _on_step(self):
        """
        Called at every step to log the episode data if the current episode number matches the logging frequency.
        """
        if self.episode_num % self.log_frequency == 0:
            # Check done
            if self.locals['dones'][0]:
                self.save_feather()
                self.episode_num += 1
                return  # Stablebaselines calls reset() before the callback, so this step has invalid values

            # Write action and reward
            row_dict = dict()
            row_dict['action'] = self.locals['actions'][0]
            row_dict['reward'] = self.locals['rewards'][0]
            row_dict['values'] = self.locals['values'][0].detach().numpy()[0]
            row_dict['log_probs'] = self.locals['log_probs'].detach().numpy()[0]

            # Write volumes
            obs = self.locals['new_obs']
            volumes = obs['Volumes'][0]
            for i, bunker in enumerate(self.bunkers):
                row_dict[bunker] = volumes[i]
            self.df = self.df.append(row_dict, ignore_index=True)

        # Count episodes
        if self.locals['dones'][0]:
            self.episode_num += 1
        return

    def save_feather(self):
        """
        Save the logged data to a file and reset the logging dataframe.
        """
        # Save to file
        self.df.to_csv(self.log_dir + f"episode_{self.episode_num}.csv", index=False)
        self.plot_episodic_graphs(path=self.log_dir + f"episode_{self.episode_num}.csv", n_bunkers=len(self.bunkers))
        # Reset logged data
        self.df = self.df[0:0]
        print(f'saved the file for episode_{self.episode_num}')
        return

    def plot_episodic_graphs(self, path=None, n_bunkers=None):
        """
        Plot episodic graphs for the logged data.

        Parameters
        ----------
        path : str, optional
            Path to the CSV file containing the logged data.
        n_bunkers : int, optional
            Number of bunkers to plot data for.
        """
        df_to_plot = pd.read_csv(path).drop(columns=["action"])
        fig, axes = plt.subplots(nrows=n_bunkers + 3, ncols=1, figsize=(15, len(df_to_plot.columns) * 3))
        df_to_plot.plot(subplots=True, ax=axes)
        plt.close()
        wandb.log({f"{self.episode_num}": wandb.Image(fig)})

        print("finished episodic plotting of state variables")