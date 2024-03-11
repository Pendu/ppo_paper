import os
import argparse
from datetime import datetime
from distutils.util import strtobool
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from multiprocessing import Process, Manager
import warnings
warnings.filterwarnings('ignore')
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import glob
import shutil
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
from utils.callbacks import *
from utils.episode_plotting import plot_volume_and_action_distributions, plot_episode_new
import torch
from torch import nn
from torch.multiprocessing import Process
import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from functools import partial
from math import inf
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from math import inf
import numpy as np
from utils.inference_plotting import *
from models.optimal_analytic_agent import *

# Import the environment
from env import SutcoEnv


os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)





def inference(args=None, log_dir=None, max_episode_length=None,
              deterministic_policy=True, env=None, seed = None, plot_local = None, fig_name= None, results_path = None):
    """
    Run inference using a trained agent.

    Args:
        args (object): Optional arguments.
        log_dir (str): Directory where the trained agent is saved.
        max_episode_length (int): Maximum length of an episode.
        deterministic_policy (bool): Whether to sample from a deterministic policy or not.
        env (object): Environment object.
        seed (int): Random seed.
        plot_local (bool): Whether to plot local inference results or not.
        fig_name (str): Name used for the plots.
        results_path (str): Path to save the results.

    Returns:
        tuple: A tuple containing volumes, actions, rewards, t_p1, and t_p2.
    """
    
    env = env

    obs, _ = env.reset()
        
    
    counter = 0


    #log_dir_best = f'prefinal_mulbunk_4/11b2p/step5/11Bp_2_el_600_b_500000_NAVf_512_512_NAPf_512_512__bs_64_kl_0.001_s_8_step4/'
    tensorboard_log = log_dir + "/tensorboard/"

    # Create the config file for the PPO model
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": args.total_timesteps,

        "policy_kwargs": dict(
            activation_fn=th.nn.Sigmoid,
            net_arch=[dict(pi=args.NA_Pf, vf=args.NA_Vf)]),

        # "env_name": "CartPole-v1",
        "batch_size": args.batch_size,
        # "clip_range_vf" : 0.7
    }



    #model_old = PPO.load(log_dir + "best_model.zip", env=env)  
    model_old = PPO.load(log_dir + "best_model.zip", env=None)
    #model_old = PPO.load(log_dir + "best_model.zip", env=env)
    model_old.observation_space = env.observation_space
    
    


    
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

    step = 0

    episode_length = 0

    # Run episode
    while True:
        episode_length += 1
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=deterministic_policy)
        #action, _ = model.predict(obs, deterministic=deterministic_policy)
        actions.append(action)
        obs, reward, done, truncated, info = env.step(action)
        #obs, reward, done, info = env.step(action)

        volumes.append(obs["Volumes"].copy())
        rewards.append(reward)
        value_1 = [
            0 if env.times['Time presses will be free'][0].item() is None or env.times['Time presses will be free'][
                0].item() == 'null' else env.times['Time presses will be free'][0].item()]
        # print(value_1[0])

        value_2 = [
            0 if env.times['Time presses will be free'][1].item() is None or env.times['Time presses will be free'][
                1].item() == 'null' else env.times['Time presses will be free'][1].item()]

        # print(value_2[0])
        t_p1.append(value_1[0])
        t_p2.append(value_2[0])
        if done:
            break
        step += 1

    print("Episodic reward: ", sum(rewards))
    
    
    if plot_local:
        plot_local_inference(env=env,
                volumes=volumes,
                actions=actions,
                rewards=rewards,
                seed=seed,
                fig_name=fig_name,
                save_fig=args.save_inf_fig,
                color="blue",
                fig_dir=log_dir,
                upload_inf_wandb=args.track_wandb,
                t_p1=t_p1,
                t_p2=t_p2,
                results_path = results_path)


    return volumes, actions, rewards, t_p1, t_p2

def inference_optimal_analytic(args = None,log_dir=None, max_episode_length=None,
              deterministic_policy=True, env_input = None, model_input = None, plot_local = None, fig_name = None, seed = None, save_inf_fig = None, results_path = None):  # TODO: remove log_dir param
    
    """
    Run inference on a trained agent (best model)
    
    :param args: Argparse parameters from console
    :param log_dir: From where to load trained agent
    :param max_episode_length: Maximum episode length for inference 
    :param fig_name: Name used for the plots
    :param deterministic_policy: Whether to sample from deterministic policy or stochastic
    :param env_input: The environment object used for inference
    :param model_input: The trained model used for inference
    :param plot_local: Whether to plot the results locally
    :param seed: Seed value for reproducibility
    :param save_inf_fig: Whether to save the inference plots
    :param results_path: Path to save the inference results
    
    :return: The volumes, actions, rewards, t_p1, and t_p2 from the inference
    
    """
    
    # Toggle verbose logging to save parameters for each episode 
    verbose_logging = False
    
    
    # Instantiate a new environment, instead of using the environment object used for training
    env = env_input
        
    # Reset the environment    
    obs,_ = env.reset()
        
    # Load the trained model(best model) from the log directory
    model = model_input
    
    # Initialize variables 
    volumes = []
    actions = []
    rewards = []
    clash_counters = []
    values = []
    log_probs = []
    entropies = []
    t_p1 = []
    t_p2 = []
    
    

    step = 0
    episode_length = 0
    
    

    # Run inference on a maximum of max_episode_length steps or /
    # / on a fewer steps, incase the episode terminates prematurely
    while True:
        episode_length += 1 #TODO: check why this is necessary
        #action_masks = get_action_masks(env)
        #action_masks = env.action_masks()
        #action, _states = model.predict(obs, action_masks=action_masks, deterministic=deterministic_policy)
        action, clash_counter = model.predict(obs)
        #print(obs,action)
        obs, reward, done, truncated, info = env.step(action)
        #obs, reward, done, info = env.step(action)

        #action_probability, value, log_prob, entropy = predict_proba(model,obs, action)
        #print(action)
        #print(action_probability)
        actions.append(action)
        #print(obs["Volumes"])
        #volumes.append(obs.copy())
        volumes.append(obs["Volumes"].copy())
        #volumes.append(obs[:5]) 
        rewards.append(reward)
        clash_counters.append(clash_counter)
        
        value_1 = [0 if env.times['Time presses will be free'][0].item() is None or env.times['Time presses will be free'][0].item() == 'null' else env.times['Time presses will be free'][0].item()]
        #print(value_1[0])

        value_2 = [0 if env.times['Time presses will be free'][1].item() is None or env.times['Time presses will be free'][1].item() == 'null' else env.times['Time presses will be free'][1].item()]
        
        #print(value_2[0])
        t_p1.append(value_1[0])
        t_p2.append(value_2[0])
           
        if done:
            break
        step += 1

    #print("Episodic reward: ", sum(rewards))
    #print(actions)
    
    if plot_local:
        plot_local_inference(env=env,
                volumes=volumes,
                actions=actions,
                rewards=rewards,
                seed=seed,
                fig_name=fig_name,
                save_fig=save_inf_fig,
                color="blue",
                fig_dir=log_dir,
                upload_inf_wandb=False,
                t_p1=t_p1,
                t_p2=t_p2,
                results_path = results_path)
                

    return volumes, actions, rewards, t_p1, t_p2

def average_inference(seed, args, shared_list, plot_local = None, rollouts = None, fig_name = None, n_rollouts = None, results_path = None):

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

    prefix = f"{len(args.bunkers)}B" #"11B" 

    run_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_s_{seed}_{args.filename_suffix}"

    if not rollouts:
        fig_name = fig_name
    else:
        fig_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_s_{seed}"

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

    #set the random seed
    #np.random.seed(0)
    print(f"seed: {seed}")
    
    
    def make_env():
        env = SutcoEnv(max_episode_length,
                   verbose_logging,
                   args.bunkers,
                   args.use_min_rew,
                   args.number_of_presses,
                   args.CL_step)
        return env

    env = make_env()

    # env = make_vec_env(lambda: env, n_envs=1, monitor_dir = log_dir)

    # env = VecMonitor(env, log_dir, info_keywords=("action", "volumes"))

    # logs will be saved in log_dir/monitor.csv
    if verbose_logging:
        env = Monitor(env, log_dir, info_keywords=("action", "volumes"))
    else:
        env = Monitor(env, log_dir)

    env_inf = env

    eplens = []
    sum_rewards = []
    overflow_list = []
    press_1_utilization = []
    press_2_utilization = []
    total_volume_processed = []
    safety_voilations = []
    total_press_utilization = []
    emptying_volumes_rollouts = []

    #plot_local = plot_local
    if rollouts:
        n_rollouts = n_rollouts
    else:
        n_rollouts = 1
        seed = seed

    for i in tqdm(range(n_rollouts)):

        #set random seed
        np.random.seed(i)
        
        # print agent type and rollout seed
        print(f"Agent type is : {args.agent_type}; Agent's seed is {seed}; and seed for rollout is: {i}")
                
        volumes, actions, rewards, t_p1, t_p2 = inference(args=args,
                                                        log_dir=log_dir,
                                                        deterministic_policy=args.inf_deterministic,
                                                        max_episode_length=600,
                                                        env=env,
                                                        seed=seed,
                                                        plot_local = plot_local,
                                                        fig_name=fig_name,
                                                        results_path = results_path)
        
        overflow,total_volume_processed_all_bunkers,emptying_volumes  = calculate_overflow(env=env,
                           volumes=volumes,
                           actions=actions,
                           rewards=rewards,
                           seed=seed,
                           fig_name=fig_name,
                           save_fig=args.save_inf_fig,
                           color="blue",
                           bunker_names=bunker_names,
                           fig_dir=log_dir,
                           upload_inf_wandb=args.track_wandb
                           )
        
        overflow_list.append(overflow)
        
        sum_rewards.append(sum(rewards))
    
        eplens.append(len(volumes))
        
        safety_voilations.append(1 if len(volumes) < 600 else 0)
        
        total_volume_processed.append(total_volume_processed_all_bunkers)
        
        press_1_utilization.append(sum(t_p1))
        
        press_2_utilization.append(sum(t_p2))
        
        total_press_utilization.append(sum(t_p1)+sum(t_p2))
        
        emptying_volumes_rollouts.append(emptying_volumes)
    
    list_original = emptying_volumes_rollouts

    # Create a new dictionary to combine all values
    dict_new = {}

    # Loop through each dictionary in the list
    for d in list_original:
        # Loop through each key, value pair in the dictionary
        for key, value in d.items():
            # If the key is already in dict_new, extend the existing list with the new values
            if key in dict_new:
                dict_new[key].extend(value)
            # If the key is not in dict_new, add it with its values
            else:
                dict_new[key] = value
    
    dict_final = {
        "seed": seed,
        "average_inference_reward": np.mean(sum_rewards),
        "standard_dev_in_inference_reward": np.std(sum_rewards),
        "average_inference_episode_length": np.mean(eplens),
        "average_inference_overflow": np.mean(overflow_list),
        "standard_dev_in_inference_overflow": np.std(overflow_list),
        "emptying actions": len(actions)-actions.count(0),
        "reward per emptying action": np.mean(sum_rewards)/(len(actions)-actions.count(0)),
        "average_total_volume_processed": np.mean(total_volume_processed),
        "average_press_1_utilization": np.mean(press_1_utilization),
        "average_press_2_utilization": np.mean(press_2_utilization),
        "safety_voilations": np.sum(safety_voilations),
        "average_total_press_utilization": np.mean(total_press_utilization)
    }

    shared_list.append(dict_final)
    
    return dict_new,  pd.DataFrame(list(shared_list))

def average_inference_optimal_analytic(seed, args, shared_list, plot_local = None, rollouts = None, fig_name = None, n_rollouts = None, results_path = None):

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

    prefix = f"11B" #"11B"

    run_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_s_{seed}_{args.filename_suffix}"

    if not rollouts:
        fig_name = fig_name
    else:
        fig_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_s_{seed}"

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

    # logs will be saved in log_dir/monitor.csv
    if verbose_logging:
        env = Monitor(env, log_dir, info_keywords=("action", "volumes"))
    else:
        env = Monitor(env, log_dir)

    env_inf = env

    eplens = []
    sum_rewards = []
    overflow_list = []
    press_1_utilization = []
    press_2_utilization = []
    total_volume_processed = []
    safety_voilations = []
    total_press_utilization = []
    emptying_volumes_rollouts = []

    model =  optimal_analytic_agent(n_bunkers=env.n_bunkers, env=env)

    plot_local = plot_local
    
    if rollouts:
        n_rollouts = n_rollouts
    else:
        n_rollouts = 1
        seed = seed
    
    for i in tqdm(range(n_rollouts)):
        
        #set random seed
        np.random.seed(i)
        
        # print agent type and rollout seed
        print(f"Agent type is : {args.agent_type}; Agent's seed is {seed}; and seed for rollout is: {i}")


        volumes, actions, rewards, t_p1, t_p2 = inference_optimal_analytic(args=None,
                                      model_input=model,
                                      deterministic_policy=True,
                                      max_episode_length=600,
                                      env_input=env,
                                      plot_local = plot_local,
                                      fig_name=fig_name,
                                      seed=seed,
                                      save_inf_fig=True,
                                      results_path = results_path)
        
        overflow, total_volume_processed_all_bunkers,emptying_volumes = calculate_overflow(env=env,
                           volumes=volumes,
                           actions=actions,
                           rewards=rewards,
                           seed=seed,
                           fig_name=fig_name,
                           save_fig=args.save_inf_fig,
                           color="blue",
                           bunker_names=bunker_names,
                           fig_dir=log_dir,
                           upload_inf_wandb=args.track_wandb
                           )
        print(f"episodic reward is : {sum(rewards)}")

    
                
        overflow_list.append(overflow)
        
        sum_rewards.append(sum(rewards))
        
    
        eplens.append(len(volumes))
        
        press_1_utilization.append(sum(t_p1))
        
        press_2_utilization.append(sum(t_p2))
        
        total_volume_processed.append(total_volume_processed_all_bunkers)
        
        safety_voilations.append(1 if len(volumes) < 600 else 0)    
        
        total_press_utilization.append(sum(t_p1)+sum(t_p2))
        
        emptying_volumes_rollouts.append(emptying_volumes)
        
        
    list_original = emptying_volumes_rollouts

    # Create a new dictionary to combine all values
    dict_new = {}

    # Loop through each dictionary in the list
    for d in list_original:
        # Loop through each key, value pair in the dictionary
        for key, value in d.items():
            # If the key is already in dict_new, extend the existing list with the new values
            if key in dict_new:
                dict_new[key].extend(value)
            # If the key is not in dict_new, add it with its values
            else:
                dict_new[key] = value
    
    dict_final = {
        "seed": seed,
        "average_inference_reward": np.mean(sum_rewards),
        "standard_dev_in_inference_reward": np.std(sum_rewards),
        "average_inference_episode_length": np.mean(eplens),
        "average_inference_overflow": np.mean(overflow_list),
        "standard_dev_in_inference_overflow": np.std(overflow_list),
        "emptying actions": len(actions)-actions.count(0),
        "reward per emptying action": np.mean(sum_rewards)/(len(actions)-actions.count(0)),
        "average_total_volume_processed": np.mean(total_volume_processed),
        "average_press_1_utilization": np.mean(press_1_utilization),
        "average_press_2_utilization": np.mean(press_2_utilization),
        "safety_voilations": np.sum(safety_voilations),
        "average_total_press_utilization": np.mean(total_press_utilization)
    }
    
    

    shared_list.append(dict_final)
    
    return dict_new,  pd.DataFrame(list(shared_list))
