import gymnasium as gym
from gymnasium import error, spaces, utils

import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from multiprocessing import Process, Manager
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
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList


from utils.callbacks import *

from env import SutcoEnv


import torch
from torch import nn
import torch as th

import wandb
from wandb.integration.sb3 import WandbCallback

from math import inf
from scipy.signal import find_peaks, peak_prominences


from typing import Any, Dict, List, Optional, Tuple, Type, Union
from functools import partial

import warnings


from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, \
    MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn



from utils.train_utils import parse_args, inference
from utils.container_utils import get_peaks, bunker_ids
from utils.train_plotting import plot_episodic_obs, plot_wandb, plot_local_during_training, plot_local_voldiff
from utils.inference_plotting import plot_local_inference
from utils.train_custom_policies import CustomPPO


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

        prefix = "11B"

        run_name = f"{prefix}p_{args.number_of_presses}_el_{args.max_episode_length}_b_{args.total_timesteps}_NAVf_{NAVf_prefix}NAPf_{NAPf_prefix}_bs_{args.batch_size}_kl_{args.target_kl}_s_{seed}_{args.filename_suffix}"

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


    # create and wrap the environment
    max_episode_length = args.max_episode_length
    verbose_logging = args.verbose_logging
    env = SutcoEnv(max_episode_length,
                   verbose_logging,
                   args.bunkers,
                   args.use_min_rew,
                   args.number_of_presses,
                   args.CL_step)


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

    seed = seed
    
    if args.CL_step == 1.5:
        log_dir_temp = f"prefinal_mulbunk_5.5/11b2p/step1/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step1/best_model.zip"

    if args.CL_step == 2:
        log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step1.5/run2/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step1.5_run2/best_model.zip'

    elif args.CL_step == 3:
        log_dir_temp = f'trained_models/step2/11Bp_2_el_25_b_1500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step2/best_model.zip'


    elif args.CL_step == 4:
        log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step2/withnewpreciserew/11Bp_2_el_25_b_1000000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_step2_withnewpreciserew/best_model.zip'



    elif args.CL_step == 5:
        log_dir_temp = f'prefinal_mulbunk_5.5/11b2p/step4/withnewpreciserew/11Bp_2_el_600_b_500000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_10_step4_withnewpreciserew_from1milstep2/best_model.zip'

    
    elif args.CL_step == 6:
        log_dir_temp = f'prefinal_mulbunk_4/11b2p/baseline_origgaus/11Bp_2_el_600_b_5000000_NAVf_512_512_NAPf_512_512__bs_64_kl_None_s_{seed}_baseline_origgaus/best_model.zip'
        # the above is the best model from step 3 which i used in 15 inferences and also showed to prof.


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
                         )  

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
                               ) 

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


        

    print("Total training time: ", datetime.now() - start)

    # del the latest model and load the model with best episodic reward
    del model
    model = PPO.load(log_dir + "best_model.zip", env=env) 

    if args.track_wandb:
        model.save(f"models/{run.id}")  # save the best model to models folder locally
        print(f'saved the best model to wandb')
    print("loaded the model with best reward")


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

    results_path = log_dir 
    if args.track_local:
        plot_local_inference(env=env,
                volumes=volumes,
                actions=actions,
                rewards=rewards,
                seed=seed,
                fig_name=run_name,
                save_fig=args.save_inf_fig,
                color="blue",
                fig_dir=log_dir,
                upload_inf_wandb=args.track_wandb,
                t_p1=t_p1,
                t_p2=t_p2,
                results_path = results_path)
        
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


