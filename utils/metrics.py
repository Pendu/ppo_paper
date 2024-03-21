import numpy as np
import pandas as pd
import os
from utils.container_utils import *

def calculate_overflow(env=None, volumes=None, actions=None, rewards=None, seed=None, fig_name=None, save_fig=None,
                       color=None,
                       bunker_names=None, fig_dir=None, upload_inf_wandb=None):
    """
    Calculates the overflow percentage and total volume processed for each bunker in the environment.

    Args:
        env (object): The environment object.
        volumes (list): List of volume values for each time step.
        actions (list): List of action values for each time step.
        rewards (list): List of reward values for each time step.
        seed (int): The random seed value.
        fig_name (str): The name of the figure.
        save_fig (bool): Whether to save the figure or not.
        color (str): The color of the figure.
        bunker_names (list): List of bunker names.
        fig_dir (str): The directory to save the figure.
        upload_inf_wandb (bool): Whether to upload the inference to Weights & Biases or not.

    Returns:
        tuple: A tuple containing the overall overflow percentage, total volume processed for all bunkers,
               and a dictionary of emptying volumes for each bunker.
    """
    env_unwrapped = env.unwrapped

    peak_rew_vols = {"C1-10": [19.84],
                        "C1-20": [26.75],
                        # "C1-30": [8.84, 17.70, 26.56],
                        "C1-30": [8.84, 17.70, 26.71],  # 10
                        "C1-40": [8.40, 16.80, 8.40],
                        "C1-50": [4.51, 9.02, 13.53],
                        # "C1-60": [7.19, 14.38, 21.57,28.78], #28.78 #15
                        "C1-60": [7.19, 14.38, 21.57, 14.38],  # 15
                        "C1-70": [8.65, 17.31, 25.96],  # 20
                        "C1-80": [12.37, 24.74],  # 32
                        "C2-10": [27.39],
                        "C2-20": [32],
                        "C2-40": [8.58, 17.17, 25.77],
                        "C2-50": [4.60, 9.21, 13.82],  # TODO #wrong check again
                        "C2-60": [8.58, 17.17, 12.96],
                        "C2-70": [13.22, 17],
                        "C2-80": [28.75],
                        "C2-90": [5.76, 11.50, 17.26]}
    percentage_list = []
    total_overflow_underflow = 0
    total_volume_processed_all_bunkers = 0
    emptying_volumes = {}

    for i in range(env_unwrapped.n_bunkers):
        
        
        volume_peaks = []
        x = np.array(volumes)[:, i]
        actions_1 = actions[:]
        overflow = 0
        underflow = 0
        total_vol_processed = 0

        # Find peaks in x with height greater than 5
        df = pd.DataFrame(x)
        peaks = get_peaks(np.array(volumes)[:, i], actions_1, i)
       
        for j in range(len(x)):
            if j in peaks and j != len(x) - 1:  # and actions_1[j]!=0
                # if x[j - 1] > 3 and x[j] == 0: #to avoid small peaks due to variance
                diff = x[j-1] - peak_rew_vols[env_unwrapped.bunker_ids[i][0]][-1]
                volume_peaks.append(round(x[j-1], 2))
                total_vol_processed += x[j-1]
                if diff < 0:
                    underflow += diff
                else:
                    overflow += diff
                
        emptying_volumes[env_unwrapped.bunker_ids[i][0]] = volume_peaks


        if total_vol_processed:

            percentage = ((overflow - underflow) / total_vol_processed) * 100
            percentage_list.append(percentage)
        else:
            percentage = 0
            percentage_list.append(0)

        total_overflow_underflow += (overflow - underflow)
        total_volume_processed_all_bunkers += total_vol_processed

    if total_volume_processed_all_bunkers:
        # Calculate overall percentage
        overall_percentage = (total_overflow_underflow / total_volume_processed_all_bunkers) * 100
    else:
        overall_percentage = 0
        
    return overall_percentage, total_volume_processed_all_bunkers, emptying_volumes, percentage_list
