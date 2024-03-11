import gymnasium as gym
from gymnasium import error, spaces, utils


# import gym
# from gym import error, spaces, utils
# import sys
# sys.modules["gym"] = gym

import os
import argparse
from datetime import datetime
from distutils.util import strtobool
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import glob
import shutil
import os
from sb3_contrib import TRPO, MaskablePPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from multiprocessing import Process
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList


from utils.callbacks import *

from env import SutcoEnv

from utils.episode_plotting import plot_volume_and_action_distributions, plot_episode_new

import torch
from torch import nn
from torch.multiprocessing import Process
import torch as th

import wandb
from wandb.integration.sb3 import WandbCallback

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# torch.set_num_threads(1)

from math import inf
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from math import inf
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment specific arguments
    parser.add_argument("--train", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, plot inference curves instead of training")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this     experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="enable cuda")
    parser.add_argument("--verbose-logging", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, plot training volume and action distributions")
    parser.add_argument("--log-dir", type=str, default=f'./logs/',
                        help="the name of this     experiment")

    # Environment specific argumentsP
    parser.add_argument("--max-episode-length", type=int, default=1200,
                        help="maximum length of an episode")
    parser.add_argument("--bunkers", nargs="+", type=int, default=[1, 2],
                        help="index values of bunkers for the environment(refer to excel sheet for indices)")
    parser.add_argument("--number-of-presses", type=int, default=1,
                        help="total number of presses in the environment")
    parser.add_argument("--save-inf-fig", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, save the figure after inference")
    parser.add_argument("--use-min-rew", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, use the min reward variant in reward function")
    parser.add_argument("--envlogger", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, use the envlogger callback to save episodic observations and actions")
    parser.add_argument("--envlogger-freq", type=int, default=1,
                        help="frequency of logging env state var during training")

    # wandb specific arguments
    parser.add_argument("--wandb-project-name", type=str, default="Trials",
                        help="the wandb's project name")
    parser.add_argument("--filename-suffix", type=str, default="Trials",
                        help="the wandb's project name")
    parser.add_argument("--track-wandb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights ecand Biases")
    parser.add_argument("--track-local", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")

    # PPO specific arguments
    parser.add_argument("--total-timesteps", type=int, default=2000,
                        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size of the experiment")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="clip range of PPO")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discout factor gamma")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="the coefficient of entropy")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="timesteps per environment per policy rollout")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of epoch when optimizing the surrogate loss")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--NA-Pf", nargs="+", type=int, default=[64, 64],
                        help="number of hidden units in the network architechture of the policy function")
    parser.add_argument("--NA-Vf", nargs="+", type=int, default=[64, 64],
                        help="number of hidden units in the network architechture of the value function")
    parser.add_argument("--act-fun", nargs="+", type=int, default=[0],
                        help="activation functions for policy and value function NNs. 0:sigmoid activation function, 1: tanh activation function, 2: relu activation function ")
    parser.add_argument("--inf-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, changes inference to stochastic")
    parser.add_argument("--CL-step", type=float, default=1,
                        help="Step number of the CL pipeline")

    args = parser.parse_args()

    return args


def inference(args=None, log_dir=None, max_episode_length=None,
              deterministic_policy=True, env=None, shared_list = None, seed = None):  # TODO: remove log_dir param
    """
    :param log_dir: Where to load trained agent
    :param fig_name: Name used for the plots
    :param save_fig: Whether to save the plotted figure or not
    :param deterministic_policy: Whether to sample from deterministic policy or not
    :return:
    """
    env = SutcoEnv(max_episode_length,
                   args.verbose_logging,
                   args.bunkers,
                   args.use_min_rew,
                   args.number_of_presses,
                   args.CL_step)
    env_unwrapped = env.unwrapped

    obs, _ = env.reset()

    counter = 0

    tensorboard_log = log_dir + "/tensorboard/"

    # Create the config file for the PPO model
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": args.total_timesteps,

        "policy_kwargs": dict(
            activation_fn=th.nn.Sigmoid,
            net_arch=dict(pi=args.NA_Pf, vf=args.NA_Vf)),

        # "env_name": "CartPole-v1",
        "batch_size": args.batch_size,
        # "clip_range_vf" : 0.7
    }

    model_old = PPO.load(log_dir + "best_model.zip", env=env)  # TODO: log this onto wandb

    model = MaskablePPO("MultiInputPolicy",
                        env,
                        seed=seed,
                        verbose=0,
                        ent_coef=args.ent_coef,
                        gamma=args.gamma,
                        #             n_steps=args.n_steps,
                        # tensorboard_log="./ppo_sutco_tensorboard/"
                        tensorboard_log=tensorboard_log,
                        target_kl=args.target_kl,
                        policy_kwargs=config["policy_kwargs"]
                        )

    model.policy.load_state_dict(model_old.policy.state_dict())

    # state variables (volumes), actions, and rewards
    volumes = []
    actions = []
    rewards = []
    t_p1 = []
    t_p2 = []
    T1_normalized = []
    T2_normalized = []
    action_masks_list = []
    bunkers_being_emptied = []

    step = 0

    episode_length = 0

    # Run episode
    while True:
        episode_length += 1
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=deterministic_policy)
        actions.append(action)
        #obs, reward, done, info = env.step(action)
        obs, reward, done, truncated, info = env.step(action)
        volumes.append(obs["Volumes"].copy())
        rewards.append(reward)
        value_1 = [
            0 if env.times['Time presses will be free'][0].item() is None or env.times['Time presses will be free'][
                0].item() == 'null' else env.times['Time presses will be free'][0].item()]

        value_2 = [
            0 if env.times['Time presses will be free'][1].item() is None or env.times['Time presses will be free'][
                1].item() == 'null' else env.times['Time presses will be free'][1].item()]

        t_p1.append(value_1[0])
        t_p2.append(value_2[0])
        T1_normalized.append(obs["Time presses will be free normalized"][0])
        T2_normalized.append(obs["Time presses will be free normalized"][1])
        action_masks_list.append(action_masks)
        bunkers_being_emptied.append(obs["Bunkers being emptied"].copy())
        if done:
            break
        step += 1

    print("Episodic reward: ", sum(rewards))

    dict_final = {
        "seed": seed,
        "episodic_cum_reward":sum(rewards),
    }
    

    shared_list.append(dict_final)

    import pandas as pd

    # Create a list of bunker names
    bunker_names = []
    for i in range(env_unwrapped.n_bunkers):
        bunker_names.append(env_unwrapped.bunker_ids[i][0])

    # Create a dictionary to store the volumes for each bunker
    volumes_dict = {bunker_name: [] for bunker_name in bunker_names}

    # Iterate over the volumes and separate them into different columns
    for volume in volumes:
        for i, bunker_name in enumerate(bunker_names):
            volumes_dict[bunker_name].append(volume[i])

    # Create a DataFrame with the separated volumes
    df = pd.DataFrame({
        "Actions": actions,
        "Action Masks": action_masks_list,
        "Rewards": rewards,
        "Bunkers being emptied": bunkers_being_emptied,
        "Time Press 1": t_p1,
        "Time Press 2": t_p2,
        "T1 normalized": T1_normalized,
        "T2 normalized": T2_normalized,
        **volumes_dict  # Unpack the volumes_dict to add separate columns for each bunker
    })

    df.to_csv(log_dir + "data.csv", index=False)
    
    return volumes, actions, rewards, t_p1, t_p2


def plot_local(env=None, volumes=None, actions=None, rewards=None, seed=None, fig_name=None, save_fig=None, color=None,
               bunker_names=None, fig_dir=None, upload_inf_wandb=False, t_p1=None, t_p2=None):
    ## Plot state variables locally ##
    fig = plt.figure(figsize=(25, 25))
    env_unwrapped = env.unwrapped

    fig.suptitle("If else agent on Sutco env with " + fig_name)  # TODO
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    ax4 = fig.add_subplot(514)
    ax5 = fig.add_subplot(515)

    ax1.set_title("Volume")
    ax2.set_title("Action")
    ax3.set_title("Reward")
    ax4.set_title("time press-1")
    ax5.set_title("time press-2")

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()

    ax1.set_ylim(top=40)

    ax2.set_yticks(list(range(env_unwrapped.action_space.n)))

    plt.xlabel("Steps")

    default_color = "#1f77b4"  # Default Matplotlib blue color
    color_code = {
        "C1-20": "#0000FF",  # blue
        "C1-30": "#FFA500",  # orange
        "C1-40": "#008000",  # green
        "C1-60": "#FF00FF",  # fuchsia
        "C1-70": "#800080",  # purple
        "C1-80": "#FF4500",  # orangered
        "C2-10": "#FFFF00",  # yellow
        "C2-20": "#A52A2A",  # brown
        "C2-60": "#D2691E",  # chocolate
        "C2-70": "#20B2AA",  # lightseagreen
        "C2-80": "#87CEEB"  # skyblue
    }
    line_width = 3

        # Plot volumes for each bunker
    for i in range(env_unwrapped.n_bunkers):
        ax1.plot(np.array(volumes)[:, i], linewidth=3,
                 label=env_unwrapped.bunker_ids[i][0],
                 color=color_code[env_unwrapped.bunker_ids[i][0]]
                 )
    ax1.legend()

    # Plot actions
    # x_axis = range(env_unwrapped.episode_length)
    # x_axis = range(env_unwrapped.max_episode_length)
    x_axis = range(len(volumes))
    # print(env_unwrapped.episode_length)
    for i in x_axis:
        if actions[i] == 0:  # Action: "do nothing"
            ax2.scatter(i, actions[i], linewidth=line_width, color=default_color)
        elif actions[i] in range(1, env_unwrapped.n_bunkers + 1):  # Action: "use Press 1"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.bunker_ids[actions[i] - 1][0]],
                        marker="^")
        elif actions[i] in range(env_unwrapped.n_bunkers + 1, env_unwrapped.n_bunkers * 2 + 1):  # Action: "use Press 2"
            ax2.scatter(i,
                        actions[i],
                        linewidth=line_width,
                        color=color_code[env_unwrapped.bunker_ids[actions[i] - env_unwrapped.n_bunkers - 1][0]],
                        marker="x")
        else:
            print("Unrecognised action: ", actions[i])

    # Plot rewards
    ax3.scatter(range(len(rewards)), rewards, linewidth=3,
                color=default_color
                )

    ax3.annotate("Cumul. reward: {:.2f}".format(sum(rewards)), xy=(0.8, 0.8), xycoords='axes fraction', fontsize=12)

    ax4.plot(t_p1, linewidth=3, label="press-1")
    ax5.plot(t_p2, linewidth=3, label="press-2")
    ax4.legend()

    # if "_realinf_" in fig_name:
    # ax3.annotate("Theor.Max Cumul. reward: {:.2f}".format(np.count_nonzero(actions)*25.0), xy=(0.75, 0.7), xycoords='axes fraction', fontsize=12)

    # else:
    #     ax3.annotate("Theoritical Max Cumul. reward: {:.2f}".format(25*len(rewards)), xy=(0.8, 0.8), xycoords='axes fraction', fontsize=14)

    if save_fig:
        # Save plot
        plt.savefig(fig_dir + fig_name + '.jpg', dpi=fig.dpi)
        # plt.savefig('{}/graph.png'.format(fig_dir))

    if upload_inf_wandb:
        # log image into wandb
        wandb.log({"xyz": wandb.Image(fig)})

    plt.show()


def plot_wandb(env=None, volumes=None, actions=None, rewards=None, bunker_names=None):
    ## Plot state variables onto wandb ##

    # append volumes, actions and rewards
    volumes_l = []
    actions_l = []
    rewards_l = []
    num_bunkers = len(volumes[:][0])
    vol_len = len(volumes)

    for j in range(len(volumes[:][0])):
        for index in range(len(volumes)):
            volumes_l.append(volumes[index][j])

    vol_dict = {}
    for i in range(num_bunkers):
        vol_dict['volumes_l_%s' % i] = volumes_l[i * vol_len:(i + 1) * vol_len]

    vol_dict_keys = list(vol_dict.keys())

    y = np.arange(3)
    line_plot_dict = {}
    for i in y:
        line_plot_dict['line_%s' % i] = 'line_plot_%s' % i

    for index in range(len(volumes)):
        actions_l.append(actions[index])
        rewards_l.append(rewards[index])

    ys_list = []
    for i in range(num_bunkers):
        y_values = vol_dict[vol_dict_keys[i]]
        ys_list.append(y_values)

    # plot volumes of all bunkers
    line_plot_dict['line_%s' % 1] = wandb.plot.line_series(
        xs=np.arange(0, len(actions_l)),
        ys=ys_list,
        keys=bunker_names,
        title="Volume of all Bunkers",
        xname="steps")

    # plot actions
    y_values = actions_l
    indices = np.arange(0, len(y_values))
    x_values = indices
    # Set up data to log in custom charts
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    # Create a table with the columns to plot
    table = wandb.Table(data=data, columns=["steps", "actions"])
    # Use the table to populate various custom charts
    # lplot_dict['line_%s' % y[-2]]  = wandb.plot.scatter(table, x='steps', y='actions', title='Actions')
    line_plot_dict['line_%s' % 2] = wandb.plot.scatter(table, x='steps', y='actions', title='Actions')

    # plot rewards
    y_values = rewards_l
    indices = np.arange(0, len(y_values))
    x_values = indices
    # Set up data to log in custom charts
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    # Create a table with the columns to plot
    table = wandb.Table(data=data, columns=["steps", "rewards"])
    # Use the table to populate various custom charts
    # lplot_dict['line_%s' % y[-1]] = wandb.plot.scatter(table, x='steps', y='rewards', title='Rewards')
    line_plot_dict['line_%s' % 3] = wandb.plot.scatter(table, x='steps', y='rewards', title='Rewards')

    # log all the graphs onto wandb board
    wandb.log(line_plot_dict)


def plot_episodic_obs(log_dir_statevar=None, n_bunkers=None):
    episode_files = glob.glob(log_dir_statevar + "/*.csv")
    counter = 1
    for i in episode_files:
        df = pd.read_csv(i).drop(columns=["action"])
        fig, axes = plt.subplots(nrows=n_bunkers + 3, ncols=1, figsize=(15, len(df.columns) * 3))
        df.plot(subplots=True, ax=axes)
        plt.close()
        wandb.log({f"{counter}": wandb.Image(fig)})
        counter += 1
    print("finished episodic plotting of state variables")


def bunker_ids(bunker_indices):
    """
    :param args.bunkers: the indicies of the bunkers used in the environment
    :return: list of actual bunker names
    """

    bunker_ids_all = [
        "C1-10",
        "C1-20",
        "C1-30",
        "C1-40",
        "C1-50",
        "C1-60",
        "C1-70",
        "C1-80",
        "C2-10",
        "C2-20",
        "C2-40",
        "C2-50",
        "C2-60",
        "C2-70",
        "C2-80",
        "C2-90"
    ]

    bunker_ids = []
    for i in bunker_indices:
        bunker_ids.append(bunker_ids_all[i])

    f"The following bunkers have been chosen for the environment {bunker_ids}"

    return bunker_ids


def get_peaks(volumes, actions, bunker):
    """ Find local maxima in a list of volumes based on action 1 and return their indices. """
        
    action_indices = [i for i, value in enumerate(actions) if value == bunker + 1]
    
    zero_indices_volumes = [i for i, value in enumerate(volumes) if value == 0.0]
    
    #overlap = [volumes[i + 1] for i in action_based_maxima_indices]
    
    zero_indices_volumes_overlap_action = [i for i in zero_indices_volumes if actions[i] == bunker + 1]
    print(f"Indices of zero volume with action equal to bunker {bunker} + 1: {zero_indices_volumes_overlap_action}")
    
    # action_based_maxima_indices_final = [action_based_maxima_indices[i] for i in indices_with_zero]
    # print(f"Final action based maxima indices for bunker {bunker}: {action_based_maxima_indices_final}")
    
    filtered_indices = [i for i in zero_indices_volumes_overlap_action if volumes[i-1] and volumes[i-2] > 0]
    
    return filtered_indices


def plot_local_voldiff(env=None, volumes=None, actions=None, rewards=None, seed=None, fig_name=None, save_fig=None,
                       color=None,
                       bunker_names=None, fig_dir=None, upload_inf_wandb=False, shared_list = None, args = None):
    peak_rew_vols = {#"C1-10": [19.84],
        "C1-20": [26.75],
        "C1-30": [26.52],  # 10
        "C1-40": [8.34],
    # "C1-50": [31.53],
        "C1-60": [14.34],  # 15
        "C1-70": [25.93],  # 20
        "C1-80": [24.75],  # 32
        "C2-10": [27.39],
        "C2-20": [32],
        "C2-40": [25.77],
    # "C2-50": [32.23], 
        "C2-60": [12.6],
        "C2-70": [17],
        "C2-80": [28.75],
        "C2-90": [28.79]}

    default_color = "#1f77b4"  # Default Matplotlib blue color
    color_code = {
        "C1-20": "#0000FF",  # blue
        "C1-30": "#FFA500",  # orange
        "C1-40": "#008000",  # green
        "C1-60": "#FF00FF",  # fuchsia
        "C1-70": "#800080",  # purple
        "C1-80": "#FF4500",  # orangered
        "C2-10": "#FFFF00",  # yellow
        "C2-20": "#A52A2A",  # brown
        "C2-60": "#D2691E",  # chocolate
        "C2-70": "#20B2AA",  # lightseagreen
        "C2-80": "#87CEEB"  # skyblue
    }  # TODO: Complete for remaining bunkers

    env_unwrapped = env.unwrapped
    fig = plt.figure(figsize=(20, 30))  # change the size of figure!
    fig.tight_layout()

    percentage_list = []
    total_overflow_underflow = 0
    total_volume_processed_all_bunkers = 0

    for i in range(env_unwrapped.n_bunkers):
    #for i in range(1):

        plt.subplot(env_unwrapped.n_bunkers + 1, 1, i + 1)

        x = np.array(volumes)[:, i]
        actions_1 = actions[:]
        volume_y_old = []
        volume_x = []
        volume_y = []
        overflow = 0
        underflow = 0
        total_vol_processed = 0

        # Find peaks in x with height greater than 5
        df = pd.DataFrame(x)
        peaks = get_peaks(np.array(volumes)[:, i], actions_1, i)

        for j in range(len(x)):
            if j in peaks and j != len(x) - 1:  # and actions_1[j]!=0
                # if x[j - 1] > 3 and x[j] == 0: #to avoid small peaks due to variance
                volume_x.append(j)
                diff = x[j-1] - peak_rew_vols[env_unwrapped.bunker_ids[i][0]][-1]
                total_vol_processed += x[j-1]
                if diff < 0:
                    underflow += diff
                else:
                    overflow += diff
                volume_y.append(diff)
                
        suffix = ""

        fig.suptitle("Ideal volume minus actual volume for bunkers" + suffix)

        plt.plot(volume_x, volume_y, linewidth=3, label=env_unwrapped.bunker_ids[i][0],
                 color=color_code[env_unwrapped.bunker_ids[i][0]], marker='o'
                 )

        # Annotate cumulative underflow and overflow for each subplot
        # plt.annotate(f'cum_underflow_ideal: {underflow}', xy=(0.2, 0.9), xycoords='axes fraction')
        # plt.annotate(f'cum_overflow_ideal: {overflow}', xy=(0.2, 0.75), xycoords='axes fraction')
        plt.annotate(f'cum_overandunderflow_ideal: {(overflow - underflow):.2f}', xy=(0.05, 0.85),
                     xycoords='axes fraction')
        plt.annotate(f'total_vol_processed: {total_vol_processed:.2f}', xy=(0.05, 0.65), xycoords='axes fraction')

        if total_vol_processed:

            percentage = ((overflow - underflow) / total_vol_processed) * 100
            percentage_list.append(percentage)
        else:
            percentage = 0
            percentage_list.append(0)

        total_overflow_underflow += (overflow - underflow)
        total_volume_processed_all_bunkers += total_vol_processed

        plt.annotate(f'% overandunderflow_ideal: {percentage:.2f}', xy=(0.05, 0.45), xycoords='axes fraction')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

        plt.grid()
        plt.legend()

    # Add new subplot for plotting the percentages
    plt.subplot(env_unwrapped.n_bunkers + 1, 1, env_unwrapped.n_bunkers + 1)
    bunker_ids = [b[0] for b in env_unwrapped.bunker_ids]

    if total_volume_processed_all_bunkers:
        # Calculate overall percentage
        overall_percentage = (total_overflow_underflow / total_volume_processed_all_bunkers) * 100
    else:
        overall_percentage = 0

    plt.bar(bunker_ids, percentage_list)
    plt.xlabel('Bunker IDs')
    plt.ylabel('Percentage of Over-and-Underflow')
    plt.title('Percentage of Over-and-Underflow by Bunker (wrt ideal vol)')
    plt.annotate(f'episodic cum_overandunderflow_ideal: {overflow - total_overflow_underflow:.2f}', xy=(0.05, 0.85),
                 xycoords='axes fraction')
    plt.annotate(f'episodic total_vol_processed: {total_volume_processed_all_bunkers:.2f}', xy=(0.05, 0.65),
                 xycoords='axes fraction')
    plt.annotate(f'episodic % Over-and-Underflow : {overall_percentage:.2f}%', xy=(0.05, 0.45),
                 xycoords='axes fraction')
    

    dict_final = {
        "seed": seed,
        "episodic % Over-and-Underflow": round(overall_percentage, 2)
    }

    shared_list.append(dict_final)

    if save_fig:
        # Save plot
        plt.savefig(fig_dir + fig_name + '_voldiff_.jpg', dpi=fig.dpi)
        # plt.savefig(fig_dir+fig_name + 'infwithoutmask_voldiff.jpg', dpi=fig.dpi)
        # plt.savefig('{}/graph.png'.format(fig_dir))

    if upload_inf_wandb:
        # log image into wandb
        wandb.log({"xyz": wandb.Image(fig)})

    plt.show()


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from functools import partial


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh()
        )
        # for param in self.policy_net:
        #     print(param)
        #     param.requires_grad = True
        #     print(param.requires_grad)
        # for param in self.value_net:
        #     print(param)
        #     param.requires_grad = False
        #     print(param.requires_grad)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            # optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            # optimizer_kwargs: Optional[Dict[str, Any]] = None,
            # filter( lambda p: p.requires_grad, self.mlp_extractor.parameters() ),
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # optimizer_class,
            # optimizer_kwargs,
            # Pass remaining arguments to base class
            *args,
            **kwargs
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

    def _build(self, lr_schedule: 1e-3) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # self.optimizer_kwargs = filter( lambda p: p.requires_grad, self.mlp_extractor.parameters())

        # Setup optimizer with initial learning rate
        # self.optimizer = self.optimizer_class(self.parameters(), lr=1e-3, **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.mlp_extractor.parameters()),
                                              lr=1e-3, **self.optimizer_kwargs)


from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from functools import partial

import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F
import torch

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn


def toNP(x):
    return x.detach().numpy()
    # return x.detach().to('cpu').numpy()


class CustomPPO(PPO):
    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            # args,
            # *kwargs
        )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):  # self.batch_size

                # print(rollout_data)
                # self.total_steps_counter +=1

                # train only value network for the first few epochs:
                # if self.counter_temp < 10:
                #     mean_params = dict((key, value) for key, value in self.policy.state_dict().items() if ("policy" in key or "value" in key) )
                #     for key, value in mean_params.items():
                #         value.requires_grad = False
                # self.counter_temp += 1

                # print(self.policy.mlp_extractor.policy_net[0].weight.requires_grad)

                #                 list_temp = []
                #                 list_temp_1 = []
                #                 for name, param in self.policy.state_dict().items():
                #                     list_temp.append(name)
                #                 for item in list_temp:
                #                     if "policy" in item:
                #                         list_temp_1.append(item)

                #                 for name, param in self.policy.state_dict().items():

                #                     if name in list_temp:
                #                         param.requires_grad  = False
                #                         print(param)

                for param in self.policy.mlp_extractor.policy_net.parameters():
                    param.requires_grad = False
                for param in self.policy.action_net.parameters():
                    param.requires_grad = False

                # for param in self.policy.mlp_extractor.parameters():
                #         param.requires_grad  = False

                # self.policy.mlp_extractor.policy_net[0].weight.requirs_grad = False
                before = toNP(self.policy.mlp_extractor.policy_net[0].weight)
                before_v = toNP(self.policy.mlp_extractor.value_net[0].weight)
                # print(np.abs(np.max(before)))
                # print(np.abs(np.max(before_v)))

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                coeff_p = 0.4

                # print("\n values pred are \n", values_pred)

                # print("\n sum of rewards are \n", torch.ones(values_pred.shape))

                # print(F.mse_loss(torch.sum(rollout_data.observations["reward"].flatten()), values_pred))

                # print()

                # Rewards_rollout_buffer = rollout_data.rewards

                # print(Rewards_rollout_buffer)

                # physics_loss = coeff_p * F.mse_loss(torch.ones(values_pred.shape), values_pred)
                # physics_loss =  coeff_p * F.relu(torch.ones(values_pred.shape), values_pred)

                loss = self.ent_coef * entropy_loss + self.vf_coef * value_loss  # + physics_loss

                # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss #+ physics_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                after = toNP(self.policy.mlp_extractor.policy_net[0].weight)
                # print(after)
                # print(np.abs(np.max(after)))

                if np.max(np.abs(before - after)) > 0:
                    print(np.max(np.abs(before - after)))

            if not continue_training:
                break


# -----------------------------------------------------------------------------

def train(seed, args, shared_list):
    args = args
    # get the actual bunker-id from args.bunkers (indices)
    bunker_names = bunker_ids(args.bunkers)

    # name of the run
    prefix = ""
    for i in bunker_names:
        prefix = prefix + i + "_"

    NAVf_prefix = ""
    for i in args.NA_Vf:
        NAVf_prefix = NAVf_prefix + str(i) + "_"

    NAPf_prefix = ""
    for i in args.NA_Pf:
        NAPf_prefix = NAPf_prefix + str(i) + "_"

        # run_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_g_{args.gamma}_nst_{args.n_steps}_NAVf_{args.NA_Vf[0]}_{args.NA_Vf[1]}_NAPf_{args.NA_Pf[0]}_{args.NA_Pf[1]}_bs_{args.batch_size}_mr_{args.use_min_rew}_mulmo_s_{seed}"

        #         run_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_g_{args.gamma}_nst_{args.n_steps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}bs_{args.batch_size}_mr_{args.use_min_rew}_kl_{args.target_kl}_mulmo_s_{seed}_15to23init_{args.ent_coef}_{args.filename_suffix}"
        prefix = "11B"

        run_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_s_{seed}_{args.filename_suffix}"

    # run_name =  f"_presses_{args.number_of_presses}_seed_{args.seed}"

    # create log dir
    # log_dir = './logs/'+ args.exp_name + '_'+ run_name+ '/'
    # log_dir = './logs/c1-30/' + run_name + '/'
    log_dir = args.log_dir + run_name + '/'

    log_dir_statevar = log_dir + 'statevar/'
    # if os.path.isdir(log_dir):
    #    shutil.rmtree(log_dir)
    if os.path.isdir(log_dir_statevar):
        shutil.rmtree(log_dir_statevar)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir_statevar, exist_ok=True)
    print(log_dir)
    print(log_dir_statevar)

    # log the parameters onto tensboard
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )
    # writer.close()

    # seeding
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = args.torch_deterministic
    #     device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # create and wrap the environment
    max_episode_length = args.max_episode_length
    verbose_logging = args.verbose_logging
    env = SutcoEnv(max_episode_length,
                   verbose_logging,
                   args.bunkers,
                   args.use_min_rew,
                   args.number_of_presses,
                   args.CL_step)
    
    
    #env = gym.make("CartPole-v1")


    # env = make_vec_env(lambda: env, n_envs=4, monitor_dir = log_dir)

    # env = VecMonitor(env, log_dir, info_keywords=("action", "volumes"))

    # logs will be saved in log_dir/monitor.csv
    # if verbose_logging:
    #     env = Monitor(env, log_dir, info_keywords=("action", "volumes"))
    # else:
    #     env = Monitor(env, log_dir)

    # Function to create environment and wrap it with Monitor
    def make_env(rank):
        def _init():
            max_episode_length = args.max_episode_length
            verbose_logging = args.verbose_logging
            env = SutcoEnv(max_episode_length,
                           verbose_logging,
                           args.bunkers,
                           args.use_min_rew,
                           args.number_of_presses,
                           args.CL_step)
            env = Monitor(env, log_dir + f"{rank}_")
            return env

        return _init

    # Create multiple environments
    n_envs = 3  # Number of environments to train on
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Create the config file for the PPO model
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": args.total_timesteps,

        "policy_kwargs": dict(
            activation_fn=th.nn.Sigmoid,
            net_arch=dict(pi=args.NA_Pf, vf=args.NA_Vf)),

        # "env_name": "CartPole-v1",
        "batch_size": args.batch_size,
        # "clip_range_vf" : 0.7
    }

    # track the experiment on wandb
    if args.track_wandb:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            # config=vars(args),
            config=config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            reinit=True
        )

        # log the code
        wandb.run.log_code(".")

        # wandb callback
        wanbc = WandbCallback(gradient_save_freq=2000, model_save_path=f"models/{run.id}",
                              verbose=0)
    else:
        wanbc = None

    # instantiate the model

    if args.track_wandb:
        tensorboard_log = f"runs/{run.id}"
    else:
        tensorboard_log = log_dir + "/tensorboard/"
        
        
    print("args.CL_step", args.CL_step)

    if args.CL_step in [1,6]: #if args.CL_step in [1,6]
                
        
        model = PPO("MultiInputPolicy",
                    env,
                    seed=seed,
                    verbose=1,
                    ent_coef=args.ent_coef,
                    gamma=args.gamma,
                    n_steps=args.n_steps,
                    # tensorboard_log="./ppo_sutco_tensorboard/"
                    tensorboard_log=tensorboard_log,
                    target_kl=args.target_kl,
                    policy_kwargs=config["policy_kwargs"],
                    n_epochs=args.n_epochs,
                    learning_rate=args.learning_rate,
                    #lr_schedule= lambda _: 0.0,
                    #clip_range= lambda _: 0.0,
                    )
            
        #model = PPO("MlpPolicy", env, verbose=1)

        
    # print("Activation function:", model.policy.activation_fn)

    # model = PPO("MultiInputPolicy", env, verbose=0,seed=seed,gamma=args.gamma,tensorboard_log=tensorboard_log,n_steps=args.n_steps)

    #     model = PPO(CustomActorCriticPolicy,
    #                 env,
    #                 seed=seed,
    #                 verbose=0,
    #                 ent_coef=args.ent_coef,
    #                 gamma=args.gamma,
    #                 # n_steps=args.n_steps,
    #                 # tensorboard_log="./ppo_sutco_tensorboard/"
    #                 tensorboard_log=tensorboard_log,
    #                 policy_kwargs = config["policy_kwargs"]
    #                 )

    # log_dir_BC = "./Behavioural_Cloning/artificial/PTC1-20_C1-30_C1-60_C1-70_C1-80_p_2_el_200_b_100000_g_0.99_nst_2048_NAVf_64_64_NAPf_64_64_bs_64_mr_True_mulmo_s_0/best_model.zip"
    # log_dir = "./aerorew/C1-20_p_1_el_100_b_1000000_g_0.99_nst_2048_NAVf_64_64_NAPf_64_64_bs_64_mr_True_kl_None_mulmo_s_3_autosavecallback1knot5klikebefore_18to25init_imbalanceloss_entcoef_0.0_withfullaerorew_newgaussian/best_model.zip"

    # log_dir_temp = "./aerorew_1/C1-20_C1-30_C1-60_p_1_el_13_b_500000_g_0.99_nst_2048_NAVf_64_64_NAPf_64_64_bs_64_mr_True_kl_0.01_mulmo_s_0_15to23init_0.0_0press_fixbonus_retrained/best_model.zip"

    # log_dir_temp = "./aerorew_1/C1-20_C1-30_C1-60_p_1_el_13_b_100000_g_0.99_nst_2048_NAVf_64_64_NAPf_64_64_bs_64_mr_True_kl_None_mulmo_s_0_15to23init_0.0_0press_allsamepeaks_actspac2/best_model.zip"

    seed = seed

    # log_dir_temp = f'prefinal_mulbunk_2/C1-20_C1-30_C1-60_C1-70_C1-80_p_2_el_13_b_1000000_NAVf_512_NAPf_512__bs_64_kl_None_s_{seed}_0press_30ts_imabalanceppo_custpeaks_5b2penv_alsoinf/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/C1-20_C1-30_C1-40_C1-60_C1-70_C1-80_C2-10_C2-20_C2-60_C2-70_C2-80_p_2_el_30_b_1500000_NAVf_512_NAPf_512__bs_64_kl_None_s_{seed}_60press_30ts_c2-20-35-restreal_hugenegfor40/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/step2/0to30init/C1-20_C1-30_C1-40_C1-60_C1-70_C1-80_C2-10_C2-20_C2-60_C2-70_C2-80_p_2_el_1500_b_1000000_NAVf_512_NAPf_512__bs_64_kl_None_s_{seed}_30ts_c2-20-32_origrw_normalpress_step2_oto30init/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/step1/unimolowerpeaksforfew_easyrew/redu_actionspace/C1-20_C1-30_C1-40_C1-60_C1-70_C1-80_C2-10_C2-20_C2-60_C2-70_C2-80_p_2_el_25_b_500000_NAVf_512_512_512_512_NAPf_512_512_512_512__bs_64_kl_None_s_{seed}_0press_30ts_Umo_lowpeaksfew_easyR_sigm_redu_actionspace_step1_closetopeakinit_continue/best_model.zip'
    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/C1-20_C1-30_C1-40_C1-60_C1-70_C1-80_C2-10_C2-20_C2-60_C2-70_C2-80_p_2_el_1500_b_200000_NAVf_512_NAPf_512__bs_64_kl_None_s_{seed}_60press_30ts_c2-20-35-restreal_hugenegfor40_step2pipeline/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/5b2p/C1-20_C1-30_C1-60_C1-70_C1-80_p_2_el_13_b_1500000_NAVf_1024_NAPf_1024__bs_64_kl_None_s_{seed}_60press_30ts_realpeaks/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/C1-20_C1-30_C1-40_C1-60_C1-70_C1-80_C2-10_C2-20_C2-60_C2-70_C2-80_p_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_0press_30ts_Umo_lowpeaksfew_sigm_redu_actionspace_step1_closetopeakinit_nsteps_6144_25xdistrew/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/C1-20_C1-30_C1-40_C1-60_C1-70_C1-80_C2-10_C2-20_C2-60_C2-70_C2-80_p_2_el_25_b_1200000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_2_0press_30ts_Umo_lowpeaksfew_sigm_redu_actionspace_step1_closetopeakinit_nsteps_6144_step1a_cost_0.1_num_actions/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/11Bp_2_el_600_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_0_normalPandRW_60ts_nsteps_6144_simplerew_step2/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_0press_30ts_Umo_lowpeaksfew_sigm_redu_actionspace_step1_closetopeakinit_nsteps_6144_simplerew_+-1.0_load0.1costseed2_neg0.1_0.1cost/best_model.zip'

    # log_dir_temp = f'prefinal_mulbunk_2/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/11Bp_2_el_600_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_normalPandRW_60ts_nsteps_6144_simplerew_step2_run2/best_model.zip'

    # if args.CL_step == 1:
    #     log_dir_temp = None
    
    if args.CL_step == 1.5:
        log_dir_temp = f"prefinal_mulbunk_5.5/11b2p/step1/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step1/best_model.zip"

    if args.CL_step == 2:
        # log_dir_temp = f'prefinal_mulbunk_3/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_0press_30ts_Umo_lowpeaksfew_sigm_redu_actionspace_step1_closetopeakinit_nsteps_6144_25xdistrew_repeat/best_model.zip'
        #log_dir_temp = f'trained_models/step1/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step1/best_model.zip'
        log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step1.5/run2/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step1.5_run2/best_model.zip'

    elif args.CL_step == 3:
        # log_dir_temp = f'prefinal_mulbunk_3/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_0press_30ts_Umo_lowpeaksfew_sigm_redu_actionspace_closetopeakinit_nsteps_6144_cost_0.1_num_actions_step1a/best_model.zip'
        log_dir_temp = f'trained_models/step2/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step2/best_model.zip'


    elif args.CL_step == 4:
        # log_dir_temp = f'prefinal_mulbunk_3/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_0press_30ts__nsteps_6144_simplerew_+-1.0__neg0.1_0.1cost_step1b/best_model.zip'
        #log_dir_temp = f'trained_models/step3/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step3/best_model.zip'
        #log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step2/withnewpreciserew/11Bp_2_el_25_b_500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step2_withnewpreciserew/best_model.zip'
        log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step2/withnewpreciserew/11Bp_2_el_25_b_1000000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step2_withnewpreciserew/best_model.zip'



    elif args.CL_step == 5:
        # log_dir_temp = f'prefinal_mulbunk_3/11b2p/step1/unimolowerpeaksforfew/redu_actionspace/11Bp_2_el_600_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_8_normalPandRW_60ts_nsteps_6144_simplerew_step2/best_model.zip'
        #log_dir_temp = f'trained_models/step4/11Bp_2_el_600_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_8_step4/best_model.zip'
        #log_dir_temp = f'prefinal_mulbunk_4/11b2p/step4/11Bp_2_el_600_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step4_otopeakinit/best_model.zip'
        #log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step4/withnewpreciserew/11Bp_2_el_600_b_100000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{10}_step4_withnewpreciserew/best_model.zip'
        #log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step2/withnewpreciserew/11Bp_2_el_25_b_1000000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step2_withnewpreciserew/best_model.zip'
        log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step4/withnewpreciserew/11Bp_2_el_600_b_500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_10_step4_withnewpreciserew_from1milstep2/best_model.zip'

    
    #TODO: change it later
    elif args.CL_step == 6:
        #log_dir_temp = f'trained_models/step1/unimolowerpeaksforfew/redu_actionspace/11Bp_2_el_600_b_500000_NAVf_512_512_NAPf_512_512__bs_64_kl_0.001_s_{seed}_normalPandRW_60ts_nsteps_6144_simplerew_clip0.05_kl0.001_seed8fromprev_step3/best_model.zip'
        log_dir_temp = f'prefinal_mulbunk_4/11b2p/baseline_origgaus/11Bp_2_el_600_b_5000000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_baseline_origgaus/best_model.zip'
        # the above is the best model from step 3 which i used in 15 inferences and also showed to prof.



    # pretrained_model = PPO.load(log_dir_temp)
    # #
    # new_model =  PPO("MultiInputPolicy",
    #             env,
    #             seed=seed,
    #             verbose=0,
    #             ent_coef=args.ent_coef,
    #             gamma=args.gamma,
    #             # n_steps=args.n_steps,
    #             # tensorboard_log="./ppo_sutco_tensorboard/"
    #             tensorboard_log=tensorboard_log,
    #             target_kl=args.target_kl,
    #             policy_kwargs=config["policy_kwargs"]
    #             )

    # new_model.policy.load_state_dict(pretrained_model.policy.state_dict())

    # model = new_model

    if args.CL_step in [1.5, 2, 3,5]: #if args.CL_step in [2, 3,5]:
        print("The value of args.CL_step is", args.CL_step)
        model = PPO.load(log_dir_temp,
                         env=env,
                         tensorboard_log=tensorboard_log,
                         target_kl=args.target_kl,
                         seed=seed,
                         verbose=0,
                         ent_coef=args.ent_coef,
                         gamma=args.gamma,
                         clip_range=args.clip_range,
                         batch_size=args.batch_size,
                         n_steps=args.n_steps,
                         # policy_kwargs = config["policy_kwargs"],
                         print_system_info=True
                         )  # TODO: log this onto wandb

    if args.CL_step == 4:
        model = CustomPPO.load(log_dir_temp,
                               env=env,
                               tensorboard_log=tensorboard_log,
                               target_kl=args.target_kl,
                               # #seed=seed,
                               # verbose=0,
                               # ent_coef=args.ent_coef,
                               # gamma=args.gamma,
                               clip_range=args.clip_range,
                               # batch_size= args.batch_size,
                               # n_steps = args.n_steps,
                               # #policy_kwargs = config["policy_kwargs"],
                               print_system_info=True
                               )  # TODO: log this onto wandb

    # Callbacks

    print("target kl and clip range are set to:", args.target_kl, args.clip_range)

    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000,log_dir=log_dir)

    # train the model

    start = datetime.now()

    with ProgressBarManager(config["total_timesteps"]) as progress_callback:

        if args.track_wandb and args.envlogger:
            callbacks = [progress_callback, auto_save_callback, EnvLogger(args.envlogger_freq, log_dir_statevar)]
        elif args.track_wandb:
            callbacks = [progress_callback, wanbc, auto_save_callback]

        elif args.track_local and args.envlogger:
            callbacks = [progress_callback, auto_save_callback, EnvLogger(args.envlogger_freq, log_dir_statevar)]
        elif args.track_local:
            callbacks = [progress_callback, auto_save_callback]
        # model.learn(total_timesteps=config["total_timesteps"], callback=[progress_callback,wanbc,eval_callback])
        
        #print callbacks used 
        print("Callbacks used are:", callbacks)
        
        model.learn(total_timesteps=config["total_timesteps"],callback=auto_save_callback, progress_bar=True)
        #model.learn(total_timesteps=config["total_timesteps"],callback=callbacks, progress_bar=True)

        #model.learn(total_timesteps=config["total_timesteps"])

        
    #model.learn(total_timesteps=config["total_timesteps"], callback=[auto_save_callback])

    print("Total training time: ", datetime.now() - start)

    # # average episodic reward of best policy
    # # TODO: Not 100% sure about load() and evaluate_policy() and env argument...

    # del the latest model and load the model with best episodic reward
    del model
    model = PPO.load(log_dir + "best_model.zip", env=env) 

    if args.track_wandb:
        model.save(f"models/{run.id}")  # save the best model to models folder locally
        print(f'saved the best model to wandb')
    print("loaded the model with best reward")

    # evaluate the policy
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=True)
    # print("Average episodic reward: {:} \u00B1 {:}".format(mean_reward, std_reward))

    # if not args.track_wandb:
    #     # plot training reward
    #     results_plotter.plot_results([log_dir], args.total_timesteps, results_plotter.X_TIMESTEPS, "PPO Sutco")

    # plot state variables logged during training
    # plot_episode_new(log_dir+"/statevar/")

    # plot training volume and action distributions
    if args.verbose_logging:
        if args.track_wandb:
            plt = plot_volume_and_action_distributions(monitor_dir=log_dir + 'monitor.csv',
                                                       fig_name='train_vol_action_distrib_' + run_name,
                                                       save_fig=False, plot_wandb=args.track_wandb, run=run)

        else:
            run = 0
            plt = plot_volume_and_action_distributions(monitor_dir=log_dir + 'monitor.csv',
                                                       fig_name='train_vol_action_distrib_' + run_name,
                                                       save_fig=False, plot_wandb=args.track_wandb, run=run)
            plt.show()
    else:
        print("No episode plotting.")

    verbose_logging = args.verbose_logging
    env = SutcoEnv(600,
                   verbose_logging,
                   args.bunkers,
                   args.use_min_rew,
                   args.number_of_presses,
                   args.CL_step)

    if verbose_logging:
        env = Monitor(env, log_dir, info_keywords=("action", "volumes"))
    else:
        env = Monitor(env, log_dir)

    # do inference with the best trained agent and plot state variables
    volumes, actions, rewards, t_p1, t_p2 = inference(args=args,
                                                      log_dir=log_dir,
                                                      deterministic_policy=args.inf_deterministic,
                                                      max_episode_length=600,
                                                      env=env,
                                                      shared_list=shared_list,
                                                      seed = seed)

    # plot the state variables from inference

    if args.track_wandb:
        plot_wandb(env,
                   volumes,
                   actions,
                   rewards,
                   bunker_names
                   )

        plot_episodic_obs(log_dir_statevar, n_bunkers=env.unwrapped.n_bunkers)

        # run.finish()

    if args.track_local:
        plot_local(env=env,
                   volumes=volumes,
                   actions=actions,
                   rewards=rewards,
                   seed=seed,
                   fig_name=run_name,
                   save_fig=args.save_inf_fig,
                   color="blue",
                   bunker_names=bunker_names,
                   fig_dir=log_dir,
                   upload_inf_wandb=args.track_wandb,
                   t_p1=t_p1,
                   t_p2=t_p2
                   )

        plot_local_voldiff(env=env,
                           volumes=volumes,
                           actions=actions,
                           rewards=rewards,
                           seed=seed,
                           fig_name=run_name,
                           save_fig=args.save_inf_fig,
                           color="blue",
                           bunker_names=bunker_names,
                           fig_dir=log_dir,
                           upload_inf_wandb=args.track_wandb,
                           shared_list=shared_list,
                           args=args)
        # # from env_aerorew_vectorized_moreinobsspace_fororgiinf import SutcoEnv as SutcoEnvInf
        # # from env_aerorew_vectorized_moreinobsspace_5b1p_forinf import SutcoEnv as SutcoEnvInf
        # #from env_aerorew_vectorized_moreinobsspace_5b2p_forinf import SutcoEnv as SutcoEnvInf
        # from env_aerorew_vectorized_moreinobsspace_11b2p_normalpress_origrw import SutcoEnv as SutcoEnvInf
        #
        # env_real_inf = SutcoEnvInf(600,
        #                            args.verbose_logging,
        #                            args.bunkers,
        #                            args.use_min_rew,
        #                            args.number_of_presses)
        #
        # if verbose_logging:
        #     env_real_inf = Monitor(env_real_inf, log_dir, info_keywords=("action", "volumes"))
        # else:
        #     env_real_inf = Monitor(env_real_inf, log_dir)
        #
        #     # do inference with the best trained agent and plot state variables
        # volumes_real_inf, actions_real_inf, rewards_real_inf = inference(args=args,
        #                                                                  log_dir=log_dir,
        #                                                                  deterministic_policy=args.inf_deterministic,
        #                                                                  max_episode_length=600,
        #                                                                  env=env_real_inf)
        #
        # plot_local(env=env_real_inf,
        #            volumes=volumes_real_inf,
        #            actions=actions_real_inf,
        #            rewards=rewards_real_inf,
        #            seed=seed,
        #            fig_name=run_name + "_realinf_",
        #            save_fig=args.save_inf_fig,
        #            color="blue",
        #            bunker_names=bunker_names,
        #            fig_dir=log_dir,
        #            upload_inf_wandb=args.track_wandb
        #            )
        #
        # plot_local_voldiff(env=env_real_inf,
        #                    volumes=volumes_real_inf,
        #                    actions=actions_real_inf,
        #                    rewards=rewards_real_inf,
        #                    seed=seed,
        #                    fig_name=run_name + "_realinf_",
        #                    save_fig=args.save_inf_fig,
        #                    color="blue",
        #                    bunker_names=bunker_names,
        #                    fig_dir=log_dir,
        #                    upload_inf_wandb=args.track_wandb
        #                    )
        #
        # plot_local(env=env_real_inf,
        #            volumes=volumes_real_inf,
        #            actions=actions_real_inf,
        #            rewards=rewards_real_inf,
        #            seed=seed,
        #            fig_name=run_name + "_realinfscaledvol_",
        #            save_fig=args.save_inf_fig,
        #            color="blue",
        #            bunker_names=bunker_names,
        #            fig_dir=log_dir,
        #            upload_inf_wandb=args.track_wandb
        #            )
        #
        # plot_local_voldiff(env=env_real_inf,
        #                    volumes=volumes_real_inf,
        #                    actions=actions_real_inf,
        #                    rewards=rewards_real_inf,
        #                    seed=seed,
        #                    fig_name=run_name + "_realinfscaledvol_",
        #                    save_fig=args.save_inf_fig,
        #                    color="blue",
        #                    bunker_names=bunker_names,
        #                    fig_dir=log_dir,
        #                    upload_inf_wandb=args.track_wandb
        #                    )

    run.finish()


def plot_local_during_training(seed, args, shared_list):
    bunker_names = bunker_ids(args.bunkers)
    # name of the run
    prefix = ""
    for i in bunker_names:
        prefix = prefix + i + "_"

    NAVf_prefix = ""
    for i in args.NA_Vf:
        NAVf_prefix = NAVf_prefix + str(i) + "_"

    NAPf_prefix = ""
    for i in args.NA_Pf:
        NAPf_prefix = NAPf_prefix + str(i) + "_"

    prefix = "11B"

    run_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_s_{seed}_{args.filename_suffix}"
    
    #TODO: change this to the correct log_dir
    fig_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_s_{seed}_maskinf"

    # create log dir
    # log_dir = './logs/'+ args.exp_name + '_'+ run_name+ '/'
    # log_dir = './logs/c1-30/' + run_name + '/'
    log_dir = args.log_dir + run_name + '/'

    if args.track_wandb:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            # config=vars(args),
            # config=config,
            name=run_name,
            monitor_gym=True,
            save_code=True,
            reinit=True
        )

    max_episode_length = 600
    verbose_logging = args.verbose_logging
    env = SutcoEnv(max_episode_length,
                   verbose_logging,
                   args.bunkers,
                   args.use_min_rew,
                   args.number_of_presses,
                   args.CL_step)

    # env = make_vec_env(lambda: env, n_envs=1, monitor_dir = log_dir)

    # env = VecMonitor(env, log_dir, info_keywords=("action", "volumes"))

    # logs will be saved in log_dir/monitor.csv
    if verbose_logging:
        env = Monitor(env, log_dir, info_keywords=("action", "volumes"))
    else:
        env = Monitor(env, log_dir)

    env_inf = env

    volumes, actions, rewards, t_p1, t_p2 = inference(args=args,
                                                      log_dir=log_dir,
                                                      deterministic_policy=args.inf_deterministic,
                                                      max_episode_length=600,
                                                      env=env,
                                                      shared_list=shared_list,
                                                      seed=seed)

    if args.track_local:
        plot_local(env=env,
                   volumes=volumes,
                   actions=actions,
                   rewards=rewards,
                   seed=seed,
                   fig_name=fig_name,
                   save_fig=args.save_inf_fig,
                   color="blue",
                   bunker_names=bunker_names,
                   fig_dir=log_dir,
                   upload_inf_wandb=args.track_wandb,
                   t_p1=t_p1,
                   t_p2=t_p2
                   )

        plot_local_voldiff(env=env,
                           volumes=volumes,
                           actions=actions,
                           rewards=rewards,
                           seed=seed,
                           fig_name=fig_name,
                           save_fig=args.save_inf_fig,
                           color="blue",
                           bunker_names=bunker_names,
                           fig_dir=log_dir,
                           upload_inf_wandb=args.track_wandb,
                           args=args,
                           shared_list=shared_list
                           )
                           
    if args.train:
        run.finish()


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')

    # get the arguments
    args = parse_args()

    with Manager() as manager:
        shared_list = manager.list()  # Create a shared list
        processes = []
        seeds = range(args.seed)

        if args.train:

            for s in seeds:
                p = Process(target=train, args=(s, args,shared_list))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()
        else:

            for s in seeds:
                p = Process(target=plot_local_during_training, args=(s, args,shared_list))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

        #At this point, shared_list contains all the results
        #Convert shared_list to a DataFrame
        df_final = pd.DataFrame(list(shared_list))

        # Grouping by 'seed'
        merged_df = df_final.groupby('seed').agg({
            'episodic_cum_reward': 'first',
            'episodic % Over-and-Underflow': 'first'
        }).reset_index()

        # Sort the DataFrame by 'episodic_cum_reward' in descending order
        sorted_df = merged_df.sort_values('episodic_cum_reward', ascending=False)

        # Get the top 5 rows from the sorted DataFrame
        top_5_df = sorted_df.head(5)

        # Print the seed numbers corresponding to the top 5 highest 'episodic_cum_reward' values
        top_5_seeds = top_5_df['seed'].tolist()
        print("Seed numbers and corresponding reward values for the top 5 highest episodic_cum_reward:")
        print(top_5_df)

        prefix = "11B"

        NAVf_prefix = ""
        for i in args.NA_Vf:
            NAVf_prefix = NAVf_prefix + str(i) + "_"

        NAPf_prefix = ""
        for i in args.NA_Pf:
            NAPf_prefix = NAPf_prefix + str(i) + "_"


        run_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_{args.filename_suffix}_allseeds"


        # Create the log_dir and allseeds_dir if they don't exist
        log_dir = os.path.join(args.log_dir, run_name)
        allseeds_dir = os.path.join(log_dir)
        os.makedirs(allseeds_dir, exist_ok=True)

        # Save the DataFrame to a CSV file
        filename = f'Result_step_{args.CL_step}_seed_{args.seed}.csv'
        sorted_df.to_csv(os.path.join(allseeds_dir, filename), index=False)

        print("The final csv file is saved")
        
        
# if __name__ == '__main__':

#     # get the arguments
#     args = parse_args()

#     seeds = range(args.seed)
    
#     shared_list = []
    
#     seed = seeds[0]
    
    
#     train(seed, args, shared_list)


