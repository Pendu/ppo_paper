import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from statsmodels.distributions.empirical_distribution import ECDF
from utils.metrics import *
from utils.container_utils import *

# Read the parameters from the config file 
params = pd.read_csv('configs/RW_param_all_bunkers_orig.csv').set_index('bunker_id')

# Plot ECDF for a single agent type
def plot_ecdf_volume(emptying_volumes=None, agent_type=None):
    """
    Plot the empirical cumulative distribution function (ECDF) of emptying volumes.

    Parameters:
    emptying_volumes (dict): A dictionary containing the emptying volumes for different scenarios.
    agent_type (str): The type of agent.

    Returns:
    None
    """
    emptying_volumes = emptying_volumes
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
    
    # Plot ECDFs
    plt.rcParams.update({"font.size": 13})
    color_code = get_color_code()
    line_width = 3
    for k, v in emptying_volumes.items():
        ecdf = ECDF(v)
        plt.figure()
        plt.title(k)
        plt.plot(ecdf.x, ecdf.y, color=color_code[k], linewidth=line_width)
        plt.grid()
        plt.xlim(-1, 40)
        plt.ylim(0, 1)
        plt.xlabel("Emptying volumes")

        # Annotate optimal volumes
        optimal_vols = peak_rew_vols[k]
        plt.vlines(
            x=optimal_vols[0],
            ymin=0,
            ymax=1,
            label="Ideal vol.",
            colors="grey",
            linewidth=line_width,
        )
        if len(optimal_vols) > 1:
            plt.vlines(
                x=optimal_vols[1:],
                ymin=0,
                ymax=1,
                ls=":",
                label="local opt.",
                colors="grey",
                linewidth=line_width,
            )

        # Add fill rate
        plt.annotate(
            #"Fill rate": round(params.loc[k]['mu'],3),
            "Fill rate: {:.2e}".format(params.loc[k]['mu']),
            xy=(0.03, 0.5),
            xycoords="axes fraction",
        )

        plt.legend(loc=2)


        save_dir = "ecdfplots/"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + "ecdf" + "_" + k + ".%s" % format, dpi="figure", format="png")

# Plot ECDF for multiple agent types
def plot_ecdf_volume_allagents(emptying_volumes=None, results_path=None, labels=None):
    """
    Plots the ECDF (Empirical Cumulative Distribution Function) for emptying volumes of different agents.

    Parameters:
    emptying_volumes (dict): A dictionary containing the emptying volumes for each agent.
                            The keys are agent names and the values are lists of emptying volumes.
                            Each list should contain three values representing the emptying volumes
                            for PPO, TRPO, and DQN agents respectively.
    results_path (str): The path to the directory where the plot will be saved.
    labels (list): A list of labels for the agents. The order should match the order of emptying volumes.

    Returns:
    None
    """    
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

    # Plot ECDFs
    plt.rcParams.update({"font.size": 13})
    color_code = get_color_code()
    line_width = 1
    for k, v in emptying_volumes.items():
        ecdf_ppo = ECDF(v[0])
        ecdf_trpo = ECDF(v[1])
        ecdf_dqn = ECDF(v[2])
        plt.figure()
        plt.title(k)
        plt.plot(
            ecdf_ppo.x,
            ecdf_ppo.y,
            color=color_code["C1-20"],
            label=labels[0],
            linewidth=line_width,  # Consider slightly increasing for emphasis
            marker='o',  # Adding a marker can help distinguish this line
            markevery=5,  # Adjust as needed to control frequency of markers
            markersize=2,  # Adjust size for visibility
        )

        plt.plot(
            ecdf_trpo.x,
            ecdf_trpo.y,
            color=color_code["C1-30"],
            label=labels[1],
            linewidth=line_width,  # Slightly thicker line for distinction
            ls="-.",  # Keep the dash-dot style but make it thicker or add markers
            marker='s',  # Square markers can differentiate this line further
            markevery=5,  # Same here, adjust as needed
            markersize=2,  # Adjust size for visibility
        )

        plt.plot(
            ecdf_dqn.x,
            ecdf_dqn.y,
            color=color_code["C1-40"],
            label=labels[2],
            linewidth=line_width,  # Making this the thickest line
            ls="--",  # Keeping dashed style, consider adding markers for more distinction
            marker='^',  # Triangle markers for further distinction
            markevery=5,  # Adjust as needed
            markersize=2,  # Adjust size for visibility
        )

        plt.grid()
        plt.xlim(-1, 40)
        plt.ylim(0, 1)
        plt.xlabel("Emptying volumes")

        optimal_vols = peak_rew_vols[k]
        plt.vlines(
            x=optimal_vols[0],
            ymin=0,
            ymax=1,
            label="Ideal vol.",
            colors="red",
            linewidth=line_width+1,
        )
        if len(optimal_vols) > 1:
            plt.vlines(
                x=optimal_vols[1:],
                ymin=0,
                ymax=1,
                ls=":",
                label="local opt.",
                colors="grey",
                linewidth=line_width,
            )

        # Add fill rate
        plt.annotate(
            #"Fill rate": round(params.loc[k]['mu'],3),
            "Fill rate: {:.2e}".format(params.loc[k]['mu']),
            xy=(0.03, 0.5),
            xycoords="axes fraction",
        )

        #plt.legend(loc=2)
        plt.legend(loc='best', fancybox=True).get_frame().set_alpha(0.5)



        save_dir = results_path+"ecdfplots/allagents/"
        os.makedirs(save_dir, exist_ok=True)
        format = "png"
        #plt.savefig(save_dir + "ecdf" + "_" + k + ".%s" % format, dpi="figure", format=format)
        plt.savefig(save_dir + "ecdf" + "_" + k + ".%s" % format, dpi="figure", format=format, bbox_inches='tight')
        
def plot_local_inference(env=None, volumes=None, actions=None, rewards=None, seed=None, fig_name=None, save_fig=None, color=None, fig_dir=None, upload_inf_wandb=None, t_p1=None, t_p2=None, results_path = None):
    """
    Plot local inference for state variables.

    Args:
        env (object): The environment object.
        volumes (list): List of volume values.
        actions (list): List of action values.
        rewards (list): List of reward values.
        seed (int): Random seed value.
        fig_name (str): Name of the figure.
        save_fig (bool): Whether to save the figure.
        color (str): Color code for the plot.
        fig_dir (str): Directory to save the figure.
        upload_inf_wandb (bool): Whether to upload the inference to wandb.
        t_p1 (list): List of time values for press-1.
        t_p2 (list): List of time values for press-2.
        results_path (str): Path to save the results.

    Returns:
        None
    """
    ## Plot state variables locally ##
    
    plt.rcParams.update({"font.size": 20})

    fig = plt.figure(figsize=(15, 18))
    fig.subplots_adjust(hspace=0.4)

    env_unwrapped = env.unwrapped

    fig.suptitle(fig_name)  # TODO
    ax1 = fig.add_subplot(511)
    plt.xlim(left=-10, right=env.max_episode_length + 10)

    ax2 = fig.add_subplot(512)
    plt.xlim(left=-10, right=env.max_episode_length + 10)

    ax3 = fig.add_subplot(513)
    plt.xlim(left=-10, right=env.max_episode_length + 10)

    ax4 = fig.add_subplot(514)
    plt.xlim(left=-10, right=env.max_episode_length + 10)

    ax5 = fig.add_subplot(515)
    plt.xlim(left=-10, right=env.max_episode_length + 10)


    ax1.set_title("Volume")
    ax2.set_title("Action")
    ax3.set_title("Reward")
    ax4.set_title("Time press-1")
    ax5.set_title("Time press-2")

    ax1.grid()
    ax2.grid()
    #ax3.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()

    ax1.set_ylim(top=40)

    ax2.set_yticks(np.arange(0, env_unwrapped.action_space.n, 2))

    plt.xlabel("Time step")

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
    # TODO: Complete for remaining bunkers
    line_width = 3

        # Plot volumes for each bunker
    for i in range(env_unwrapped.n_bunkers):
        ax1.plot(np.array(volumes)[:, i], linewidth=3,
                 label=env_unwrapped.bunker_ids[i][0],
                 color=color_code[env_unwrapped.bunker_ids[i][0]]
                 )
    #ax1.legend()

    # Plot actions
    x_axis = range(len(volumes))
    for i in x_axis:
        if actions[i] == 0:  # Action: "do nothing"
            ax2.scatter(i, actions[i], linewidth=line_width, color=default_color, alpha=0)
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
        ax3.scatter(i, rewards[i], linewidth=3,
                    color=color_code[env_unwrapped.bunker_ids[actions[i] - 1][0]],
                    clip_on=False
                    )

    #ax3.annotate("Cumul. reward: {:.2f}".format(sum(rewards)), xy=(0.81, 1.02), xycoords='axes fraction', fontsize=12)

    ax4.plot(t_p1, linewidth=3, label="press-1")
    ax5.plot(t_p2, linewidth=3, label="press-2")
    ax4.legend()
    ax5.legend()
    # Add legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    if save_fig:
        # Save plot
        plt.savefig(results_path+fig_name + '.jpg', dpi=fig.dpi) 
        # plt.savefig('{}/graph.png'.format(fig_dir))

    if upload_inf_wandb:
        # log image into wandb
        wandb.log({"xyz": wandb.Image(fig)})
        
        
        
        
def plot_vol_deviation(env=None, volumes=None, actions=None, rewards=None, seed=None, fig_name=None, save_fig=None,
                       color=None,
                       bunker_names=None, fig_dir=None, upload_inf_wandb=False, shared_list = None, args = None, results_path = None):
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
    fig = plt.figure(figsize=(15, 12))  # change the size of figure!
    fig.tight_layout()

    percentage_list = []
    total_overflow_underflow = 0
    total_volume_processed_all_bunkers = 0

    for i in range(env_unwrapped.n_bunkers):
    #for i in range(1):

        #plt.subplot(env_unwrapped.n_bunkers + 1, 1, i + 1)

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

        #fig.suptitle("Ideal volume minus actual volume for bunkers" + suffix)

        #plt.plot(volume_x, volume_y, linewidth=3, label=env_unwrapped.bunker_ids[i][0],
        #         color=color_code[env_unwrapped.bunker_ids[i][0]], marker='o'
        #         )

        # Annotate cumulative underflow and overflow for each subplot
        # plt.annotate(f'cum_underflow_ideal: {underflow}', xy=(0.2, 0.9), xycoords='axes fraction')
        # plt.annotate(f'cum_overflow_ideal: {overflow}', xy=(0.2, 0.75), xycoords='axes fraction')
        #plt.annotate(f'cum_overandunderflow_ideal: {(overflow - underflow):.2f}', xy=(0.05, 0.85),
        #             xycoords='axes fraction')
        #plt.annotate(f'total_vol_processed: {total_vol_processed:.2f}', xy=(0.05, 0.65), xycoords='axes fraction')

        if total_vol_processed:

            percentage = ((overflow - underflow) / total_vol_processed) * 100
            percentage_list.append(percentage)
        else:
            percentage = 0
            percentage_list.append(0)

        total_overflow_underflow += (overflow - underflow)
        total_volume_processed_all_bunkers += total_vol_processed

        #plt.annotate(f'% overandunderflow_ideal: {percentage:.2f}', xy=(0.05, 0.45), xycoords='axes fraction')

        #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)

        #plt.grid()
        #plt.legend()
        
        
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
    

    if save_fig:
        # Save plot
        #plt.savefig(results_path + fig_name + '_voldiff_.jpg', dpi=fig.dpi)
        # plt.savefig(fig_dir+fig_name + 'infwithoutmask_voldiff.jpg', dpi=fig.dpi)
        # plt.savefig('{}/graph.png'.format(fig_dir))
        pass 

    if upload_inf_wandb:
        # log image into wandb
        wandb.log({"xyz": wandb.Image(fig)})

    #plt.show()
    
    return percentage_list




