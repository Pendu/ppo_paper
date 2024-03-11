#from utils.inference_utils import average_inference, average_inference_optimal_analytic
from utils.inference_utils import average_inference, average_inference_optimal_analytic

from utils.inference_plotting import *
import json
from argparse import Namespace
from tqdm import tqdm
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd



# Read args from configs folder and convert to Namespace
with open('configs/ppo_volume_criteria.json') as f:
    args2 = json.load(f)
with open('configs/ppo.json') as f:
    args3 = json.load(f)
with open('configs/optimal_analytic.json') as f:
    args4 = json.load(f)
    
# Convert dictionaries to Namespace objects
args2_ns = Namespace(**args2)
args3_ns = Namespace(**args3)
args4_ns = Namespace(**args4)


# Create empty lists to store the emptying volumes and dataframes for each agent
emptying_volumes_ppo_volume_criteria_list= []
emptying_volumes_ppo_list = []
emptying_volume_optimal_analytic_list = []
df_ppo_volume_criteria_list = []
df_ppo_list = []
df_optimal_analytic_list = []

# create a directory to save all the results
if not os.path.exists('results'):
    os.makedirs('results')
    results_path = 'results/'
else:
    results_path = 'results/'
    

################################################################################# Run inference for 15 agents ############################################################################################################

#check if the dataframes already exist locally, if not run inference for 15 agents
if not os.path.exists(results_path+'df_ppo_allagents.csv'):

    for i in tqdm(range(15)):
        
        emptying_volumes_ppo_volume_criteria, df_ppo_volume_criteria= average_inference(seed=i , args=args2_ns, shared_list=[], plot_local=False, rollouts=True, n_rollouts=15)
        emptying_volumes_ppo, df_ppo = average_inference(seed=i, args=args3_ns, shared_list=[], plot_local=False, rollouts=True, n_rollouts=15) #seed 7 has least press utilization and seed 9 has least overflow
        emptying_volume_optimal_analytic, df_optimal_analytic= average_inference_optimal_analytic(seed=i, args=args4_ns, shared_list=[], plot_local=False, rollouts=True, n_rollouts=15)
        
        #append the emptying volumes and dataframes to the lists
        emptying_volumes_ppo_volume_criteria_list.append(emptying_volumes_ppo_volume_criteria)
        df_ppo_volume_criteria_list.append(df_ppo_volume_criteria)
        emptying_volumes_ppo_list.append(emptying_volumes_ppo)
        df_ppo_list.append(df_ppo)
        emptying_volume_optimal_analytic_list.append(emptying_volume_optimal_analytic)
        df_optimal_analytic_list.append(df_optimal_analytic)
    
    
if not os.path.exists(results_path+'df_ppo_allagents.csv'):

    #concatenate the dataframes in the lists to create a single dataframe for each agent
    df_ppo_allagents = pd.concat(df_ppo_list)
    df_ppo_volume_criteria_allagents = pd.concat(df_ppo_volume_criteria_list)
    df_optimal_analytic_allagents = pd.concat(df_optimal_analytic_list)


# check if all agents are already saved locally, if not save them
if not os.path.exists(results_path+'df_ppo_allagents.csv'):
    
    
    print("Saving all agents data locally...")
    
    # save the dataframes as csv
    df_ppo_allagents.to_csv(results_path+'df_ppo_allagents.csv', index=False)
    df_ppo_volume_criteria_allagents.to_csv(results_path+'df_ppo_volume_criteria_allagents.csv', index=False)
    df_optimal_analytic_allagents.to_csv(results_path+'df_optimal_analytic_allagents.csv', index=False)
    # save the emptying volumes as JSON
    with open(results_path+'emptying_volumes_ppo_list.json', 'w') as file:
        json.dump(emptying_volumes_ppo_list, file)
    with open(results_path+'emptying_volumes_ppo_volume_criteria_list.json', 'w') as file:
        json.dump(emptying_volumes_ppo_volume_criteria_list, file)
    with open(results_path+'emptying_volume_optimal_analytic_list.json', 'w') as file:
        json.dump(emptying_volume_optimal_analytic_list, file)

else:
    
    print("All agents data already saved locally, loading data from local files...")
    
    #load all agents data
    df_ppo_allagents = pd.read_csv(results_path+'df_ppo_allagents.csv')
    df_ppo_volume_criteria_allagents = pd.read_csv(results_path+'df_ppo_volume_criteria_allagents.csv')
    df_optimal_analytic_allagents = pd.read_csv(results_path+'df_optimal_analytic_allagents.csv')
    
    # Load from JSON
    with open(results_path+'emptying_volumes_ppo_list.json', 'r') as file:
        emptying_volumes_ppo_list = json.load(file)
    with open(results_path+'emptying_volumes_ppo_volume_criteria_list.json', 'r') as file:
        emptying_volumes_ppo_volume_criteria_list = json.load(file)
    with open(results_path+'emptying_volume_optimal_analytic_list.json', 'r') as file:
        emptying_volume_optimal_analytic_list = json.load(file)

################################################################################# Plotting best agents emptying volumes ############################################################################################################
print("plotting best agents emptying volumes")
print("\n")


# set the best seeds for each agent manually
ppo_best_seed = 7
ppo_volume_criteria_best_seed = 11 
optimal_analytic_best_seed = 4 

# create a dictionary with the best emptying volumes for each agent
emptying_volumes = {}
emptying_volumes["ppo"] = emptying_volumes_ppo_list[ppo_best_seed] 
emptying_volumes["ppo_volume_criteria"] = emptying_volumes_ppo_volume_criteria_list[ppo_volume_criteria_best_seed] 
emptying_volumes["optimal_analytic"] = emptying_volume_optimal_analytic_list[optimal_analytic_best_seed] 
emptying_volumes_best_agents = {k: [emptying_volumes["ppo"][k], emptying_volumes["ppo_volume_criteria"][k], emptying_volumes["optimal_analytic"][k]] for k in emptying_volumes["ppo"].keys()}

# save the emptying volumes for the best agents as JSON
with open(results_path+'emptying_volumes_best_agents.json', 'w') as file:
    json.dump(emptying_volumes_best_agents, file)

# check if plots already exist locally, if not plot them and save them
if not os.path.exists(results_path+"ecdfplots/allagents/"+'ecdf_C1-20.png'):
    plot_ecdf_volume_allagents(emptying_volumes = emptying_volumes_best_agents, results_path = results_path, labels = ["PPO-CL", "PPO-Volume Criteria", "Optimal Analytic Agent"])
else:
    print("ECDF plots for best agents already saved locally")
    
    
################################################################################# Plotting best agents inference graphs ############################################################################################################

#check if the inference graphs already exist locally, if not plot them
if not os.path.exists(results_path+'Inference of Baseline PPO agent on test environment.png'):

    # plot infercence graphs for best agents                        
    _ , _ = average_inference(seed=ppo_volume_criteria_best_seed , args=args2_ns,shared_list=[], plot_local=True, rollouts=False, fig_name="Inference of PPO-volume criteria on test environment",n_rollouts=1, results_path=results_path) 
    _ , _ = average_inference(seed=ppo_best_seed , args=args3_ns, shared_list=[], plot_local=True, rollouts=False, fig_name="Inference of PPO-CL agent on test environment",n_rollouts=1, results_path=results_path)
    _ , _ = average_inference_optimal_analytic(seed=optimal_analytic_best_seed, args=args4_ns, shared_list=[], plot_local=True, rollouts=False, fig_name="Inference of Optical Analytic agent on test environment",n_rollouts=1, results_path=results_path)
else:
    print("Inference graphs for best agents already saved locally")



################################################################################# Create a dataframe with the best agents ############################################################################################################

# read from the df_ppo_allagents dataframe the seed value with ppo_best_seed into df_ppo
df_ppo = df_ppo_allagents[df_ppo_allagents['seed'] == ppo_best_seed]
df_ppo_volume_criteria = df_ppo_volume_criteria_allagents[df_ppo_volume_criteria_allagents['seed'] == ppo_volume_criteria_best_seed]
df_optimal_analytic = df_optimal_analytic_allagents[df_optimal_analytic_allagents['seed'] == optimal_analytic_best_seed]

# Rename seeds based on file - to indicate best or median agents
df_ppo.loc[df_ppo['seed'] == ppo_best_seed, 'seed'] = 'PPO-CL'
df_ppo_volume_criteria.loc[df_ppo_volume_criteria['seed'] == ppo_volume_criteria_best_seed, 'seed'] = 'PPO-Volume Criteria'
df_optimal_analytic.loc[df_optimal_analytic['seed'] == optimal_analytic_best_seed, 'seed'] = 'Optimal Analytic Agent'

# Select specific rows based on the new seed names
ppo_baseline_best = df_ppo_volume_criteria[df_ppo_volume_criteria['seed'] =='PPO-Volume Criteria']
ppo_optimal_analytic_best = df_optimal_analytic[df_optimal_analytic['seed'] == 'Optimal Analytic Agent']
ppo_best = df_ppo[df_ppo['seed'] =='PPO-CL']


# Combine selected rows into a new DataFrame
combined_df_bestseeds = pd.concat([ppo_baseline_best, ppo_best, ppo_optimal_analytic_best])


# Simplify the DataFrame by combining average and std dev columns, and removing unnecessary columns
combined_df_bestseeds['Average Inference Reward (±Std Dev)'] = round(combined_df_bestseeds['average_inference_reward'],2).astype(str) + " (±" + round(combined_df_bestseeds['standard_dev_in_inference_reward'],2).astype(str) + ")"
combined_df_bestseeds['Average Inference Overflow (±Std Dev)'] = round(combined_df_bestseeds['average_inference_overflow'],2).astype(str) + " (±" + round(combined_df_bestseeds['standard_dev_in_inference_overflow'],2).astype(str) + ")"


# Remove unncessary columns
final_df_bestseeds = combined_df_bestseeds.drop(columns=[
    'average_inference_reward', 
    'standard_dev_in_inference_reward', 
    'average_inference_overflow',
    'standard_dev_in_inference_overflow', 
    'average_inference_episode_length',
    'emptying actions',
    'reward per emptying action', 
    'average_press_1_utilization', 
    'average_press_2_utilization',
    'average_total_volume_processed'
    ])

print("\n")
final_df_bestseeds.drop(columns=['safety_voilations'], inplace=True)
final_df_bestseeds.rename(columns={'average_total_press_utilization':'Average Total Press Utilization'}, inplace=True)


#save the final_df_bestseeds as csv
final_df_bestseeds.to_csv(results_path+'final_df_bestseeds.csv', index=False)
final_df_bestseeds.rename(columns={'Seed':'Agent'}, inplace=True)

print("final_df_bestseeds saved as csv in "+results_path+'final_df_bestseeds.csv')

print(tabulate(final_df_bestseeds, headers= 'keys', tablefmt='fancy_grid', showindex=False))


print("\n")


################################################################################### Plot graph of Average Total Press Utilization Across Best Agents #######################################################################

df_placeholder = pd.read_csv(results_path+'final_df_bestseeds.csv')


# Re-applying the plot style with exact details as previously mentioned
plt.rcParams.update({"font.size": 20})  # Adjusting font size globally

fig, ax = plt.subplots(figsize=(15, 10))  # Adjusting figure size
ax = sns.barplot(x="seed", y="Average Total Press Utilization", data=df_placeholder, palette="muted")

# Setting the title and labels with the exact specified font sizes
plt.title('Average Total Press Utilization Across Agents', fontsize=20)
plt.xlabel('Agent', fontsize=18)
plt.ylabel('Average Total Press Utilization (secs)', fontsize=18)

# Adjusting y-axis with the specified start and adding space above the highest value for clarity
plt.ylim(60000, max(df_placeholder["Average Total Press Utilization"]) + 5000)

# Highlighting the PPO-CL agent for emphasis in green
for bar, agent in zip(ax.patches, df_placeholder["seed"]):
    if agent == "PPO-CL":
        bar.set_color('green')  # Highlight PPO-CL in green

# Annotating the bars with the numeric values, adjusting for specified font size and placement
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points', fontsize=16)

plt.xticks(rotation=45, fontsize=16)  # Adjusting x-axis labels for readability
plt.yticks(fontsize=16)  # Adjusting y-axis labels font size
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

plt.savefig(results_path+'avg_total_press_utilization_best_agents.png')
print("Average Total Press Utilization for Best Agents saved as avg_total_press_utilization_best_agents.png in "+results_path)



################################################################################## Plot Percentage Safety Violations for 15 agents #######################################################################################
# Total violations
print("\n")
print("Total violations for 15 agents for each agent type")
print("\n")
df_ppo_allagents = pd.read_csv(results_path+'df_ppo_allagents.csv')
df_ppo_volume_criteria_allagents = pd.read_csv(results_path+'df_ppo_volume_criteria_allagents.csv')
df_optimal_analytic_allagents = pd.read_csv(results_path+'df_optimal_analytic_allagents.csv')
safety_voilations_15agents_ppo = df_ppo_allagents['safety_voilations'].sum()
safety_voilations_15agents_volume_criteria = df_ppo_volume_criteria_allagents['safety_voilations'].sum()
safety_voilations_15agents_optimal_analytic = df_optimal_analytic_allagents['safety_voilations'].sum()

# create dataframe with the total voilations for 15 agents with the agent types as first column and the voilations as second column
total_voilations = pd.DataFrame( {'seed': ['PPO-CL', 'PPO-Volume Criteria', 'Optimal Analytic Agent'],
                                        'Percentage safety voilations for 15 agents': [round(safety_voilations_15agents_ppo*100/225,2), round(safety_voilations_15agents_volume_criteria*100/225,2), round(safety_voilations_15agents_optimal_analytic*100/225,2)]
                                        }
                                        )   

#save the total_voilations as csv
total_voilations.to_csv(results_path+'15agents_total_voilations.csv', index=False)
total_voilations = pd.read_csv(results_path+'15agents_total_voilations.csv')

print("15 agents total voilations saved as csv in 15agents_total_voilations.csv")

# Print the dataframe with dashed lines
print(tabulate(total_voilations, headers='keys', tablefmt='fancy_grid', showindex=True))

print("\n")


data = pd.read_csv(results_path+'15agents_total_voilations.csv')

# Adapting the plot design based on the provided settings
plt.rcParams.update({"font.size": 20})  # Update font size as per the provided code

fig, ax = plt.subplots(figsize=(15, 10))  # Adjusting figure size
ax = sns.barplot(x="seed", y="Percentage safety voilations for 15 agents", data=data, palette="muted")

# Setting the title and labels with specified font sizes
plt.title('Percentage Safety Violations Across Agents', fontsize=20)
plt.xlabel('Agent', fontsize=18)
plt.ylabel('Percentage Safety Violations (%)', fontsize=18)

# Adjusting y-axis
plt.ylim(0, max(data["Percentage safety voilations for 15 agents"])+5)  # Adding some space above the highest value for clarity

# Highlighting the PPO-CL agent for emphasis
for bar, agent in zip(ax.patches, data["seed"].values):
    if agent == "PPO-CL":
        bar.set_color('green')  # Highlight PPO-CL in red to draw attention

# Annotating the bars with the numeric values, adjusting the font size
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points', fontsize=16)

plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels


# Save the plot
plot_path = results_path+'avg_total_safety_violations.png'
plt.savefig(plot_path)
print(f"Percentage Safety Violations for 15 agents saved as {plot_path}")


