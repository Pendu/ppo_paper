import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import glob
from utils.inference_utils import inference
from utils.container_utils import get_peaks


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

    volumes, actions, rewards, t_p1, t_p2, vol_dev_list = inference(args=args,
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
    }  

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

    #plt.show()