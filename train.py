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

from math import inf
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from math import inf
import numpy as np

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


from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from functools import partial


from utils.train_utils import parse_args, inference
from utils.container_utils import get_peaks, bunker_ids
from utils.train_plotting import plot_episodic_obs, plot_wandb, plot_local_during_training, plot_local_voldiff
from utils.inference_plotting import plot_local_inference
from utils.train_custom_policies import CustomPPO



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
        #model.learn(total_timesteps=config["total_timesteps"],callback=callbacks, progress_bar=True)

        #model.learn(total_timesteps=config["total_timesteps"])

        
    #model.learn(total_timesteps=config["total_timesteps"], callback=[auto_save_callback])

    print("Total training time: ", datetime.now() - start)

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


