
import argparse
import os
from distutils.util import strtobool
from env import SutcoEnv
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
import pandas as pd
import torch as th
import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks


def parse_args():
    """
    Parse the arguments from the command line
    
    Returns
    -------
    args : Namespace
        The arguments passed to the training script, containing hyperparameters and environment settings.
    """
    
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
              deterministic_policy=True, env=None, shared_list = None, seed = None): 
    """
    Perform inference with the trained model on the environment.

    Parameters
    ----------
    args : Namespace
        The arguments passed to the training script, containing hyperparameters and environment settings.
    log_dir : str
        The directory where logs and the trained model are stored.
    max_episode_length : int
        The maximum length of an episode.
    deterministic_policy : bool
        Whether to use a deterministic policy during inference.
    env : gym.Env
        The environment to perform inference on.
    shared_list : multiprocessing.Manager().list
        A shared list for multiprocessing, used to store results from parallel environments.
    seed : int
        The random seed for environment and model reproducibility.

    Returns
    -------
    volumes : list
        A list of volumes observed during the episode.
    actions : list
        A list of actions taken during the episode.
    rewards : list
        A list of rewards received during the episode.
    t_p1 : list
        A placeholder list, reserved for future use.
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

    model_old = PPO.load(log_dir + "best_model.zip", env=env) 

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