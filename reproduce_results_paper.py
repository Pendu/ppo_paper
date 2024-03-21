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
        
        emptying_volumes_ppo_volume_criteria, df_ppo_volume_criteria, _, _= average_inference(seed=i , args=args2_ns, shared_list=[], plot_local=False, rollouts=True, n_rollouts=15)
        emptying_volumes_ppo, df_ppo,_ ,_ = average_inference(seed=i, args=args3_ns, shared_list=[], plot_local=False, rollouts=True, n_rollouts=15) #seed 7 has least press utilization and seed 9 has least overflow
        emptying_volume_optimal_analytic, df_optimal_analytic,_, _= average_inference_optimal_analytic(seed=i, args=args4_ns, shared_list=[], plot_local=False, rollouts=True, n_rollouts=15)
        
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
    _ , _, _,_= average_inference(seed=ppo_volume_criteria_best_seed , args=args2_ns,shared_list=[], plot_local=True, rollouts=False, fig_name="Inference of PPO-volume criteria on test environment",n_rollouts=1, results_path=results_path) 
    _ , _ , _,_ = average_inference(seed=ppo_best_seed , args=args3_ns, shared_list=[], plot_local=True, rollouts=False, fig_name="Inference of PPO-CL agent on test environment",n_rollouts=1, results_path=results_path)
    _ , _ , _,_ = average_inference_optimal_analytic(seed=optimal_analytic_best_seed, args=args4_ns, shared_list=[], plot_local=True, rollouts=False, fig_name="Inference of Optical Analytic agent on test environment",n_rollouts=1, results_path=results_path)
else:
    print("Inference graphs for best agents already saved locally")










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
plt.rcParams.update({"font.size": 25})  # Update font size as per the provided code

fig, ax = plt.subplots(figsize=(15, 10))  # Adjusting figure size
ax = sns.barplot(x="seed", y="Percentage safety voilations for 15 agents", data=data, palette={"PPO-CL": "green", "PPO-Volume Criteria": "orange", "Optimal Analytic Agent": "violet"})

# Setting the title and labels with specified font sizes
plt.title('Percentage Safety Violations Across Agents', fontsize=25)
plt.xlabel('Agent', fontsize=23)
plt.ylabel('Percentage Safety Violations (%)', fontsize=23)

# Adjusting y-axis
plt.ylim(0, max(data["Percentage safety voilations for 15 agents"])+5)  # Adding some space above the highest value for clarity

# Annotating the bars with the numeric values, adjusting the font size
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points', fontsize=22)

#plt.xticks(rotation=45, fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels


# Save the plot
plot_path = results_path+'avg_total_safety_violations.png'
plt.savefig(plot_path)
print(f"Percentage Safety Violations for 15 agents saved as {plot_path}")



################################################################################## Plot Bar graph comparing percentage volume deviation for best agents #######################################################################################

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
    
bunker_ids_all = [
    ["C1-20", "C1-2"],
    ["C1-30", "C1-3"],
    ["C1-40", "C1-4"],
    ["C1-60", "C1-6"],
    ["C1-70", "C1-7"],
    ["C1-80", "C1-8"],
    ["C2-10", "C2-1"],
    ["C2-20", "C2-2"],
    ["C2-60", "C2-6"],
    ["C2-70", "C2-7"],
    ["C2-80", "C2-8"],
]
bunker_ids = []
bunker_ids = [b[0] for b in bunker_ids_all]


#check if the inference graphs already exist locally, if not plot them
if not os.path.exists(results_path+'vol_dev_PPO_volcriteria_average.csv'):
    # plot infercence graphs for best agents                        
    _ , _, vol_dev_PPO_volcriteria, emptying_actions_PPO_volcriteria= average_inference(seed=ppo_volume_criteria_best_seed , args=args2_ns,shared_list=[], plot_local=True, rollouts=True, fig_name="Inference of PPO-volume criteria on test environment",n_rollouts=15, results_path=results_path) 
    _ , _ , vol_dev_PPO_CL, emptying_actions_PPO_CL = average_inference(seed=ppo_best_seed , args=args3_ns, shared_list=[], plot_local=True, rollouts=True, fig_name="Inference of PPO-CL agent on test environment",n_rollouts=15, results_path=results_path)
    _, _, vol_dev_optimal_analytic, emptying_actions_optimal_analytic = average_inference_optimal_analytic(seed=optimal_analytic_best_seed, args=args4_ns, shared_list=[], plot_local=True, rollouts=True, fig_name="Inference of Optimal Analytic agent on test environment",n_rollouts=11, results_path=results_path)
    
    # Take average of 15 runs from vol_dev_PPO_CL and vol_dev_PPO_volcriteria
    vol_dev_PPO_CL_average = [sum(x)/len(x) for x in zip(*vol_dev_PPO_CL)]
    vol_dev_PPO_volcriteria_average = [sum(x)/len(x) for x in zip(*vol_dev_PPO_volcriteria)]
    vol_dev_optimal_analytic_average = [sum(x)/len(x) for x in zip(*vol_dev_optimal_analytic)]
    emptying_actions_PPO_volcriteria_average = [sum(x)/len(x) for x in zip(*emptying_actions_PPO_volcriteria)]
    emptying_actions_PPO_CL_average = [sum(x)/len(x) for x in zip(*emptying_actions_PPO_CL)]
    emptying_actions_optimal_analytic_average = [sum(x)/len(x) for x in zip(*emptying_actions_optimal_analytic)]
    
    # save the vol_dev_PPO_CL_average and vol_dev_PPO_volcriteria_average 
    vol_dev_PPO_CL_average = pd.DataFrame(vol_dev_PPO_CL_average, columns=['PPO-CL'])
    vol_dev_PPO_volcriteria_average = pd.DataFrame(vol_dev_PPO_volcriteria_average, columns=['PPO-volume criteria'])
    vol_dev_optimal_analytic_average = pd.DataFrame(vol_dev_optimal_analytic_average, columns=['Optimal Analytic Agent'])
    emptying_actions_PPO_volcriteria_average = pd.DataFrame(emptying_actions_PPO_volcriteria_average, columns=['PPO-volume criteria'])
    emptying_actions_PPO_CL_average = pd.DataFrame(emptying_actions_PPO_CL_average, columns=['PPO-CL'])
    emptying_actions_optimal_analytic_average = pd.DataFrame(emptying_actions_optimal_analytic_average, columns=['Optimal Analytic Agent'])
    
    vol_dev_PPO_volcriteria_average.to_csv(results_path+'vol_dev_PPO_volcriteria_average.csv', index=False)
    vol_dev_PPO_CL_average.to_csv(results_path+'vol_dev_PPO_CL_average.csv', index=False)
    vol_dev_optimal_analytic_average.to_csv(results_path+'vol_dev_optimal_analytic_average.csv', index=False)
    emptying_actions_PPO_volcriteria_average.to_csv(results_path+'emptying_actions_PPO_volcriteria_average.csv', index=False)
    emptying_actions_PPO_CL_average.to_csv(results_path+'emptying_actions_PPO_CL_average.csv', index=False)
    emptying_actions_optimal_analytic_average.to_csv(results_path+'emptying_actions_optimal_analytic_average.csv', index=False)

    print("vol_dev_PPO_volcriteria_average, vol_dev_PPO_CL_average and vol_dev_optimal_analytic_average saved as csv in "+results_path)
else:
    print("Average volume deviation for PPO-CL and PPO-volume criteria already saved locally")
    vol_dev_PPO_CL_average = pd.read_csv(results_path+'vol_dev_PPO_CL_average.csv')
    vol_dev_PPO_volcriteria_average = pd.read_csv(results_path+'vol_dev_PPO_volcriteria_average.csv')
    vol_dev_optimal_analytic_average = pd.read_csv(results_path+'vol_dev_optimal_analytic_average.csv')
    emptying_actions_PPO_volcriteria_average = pd.read_csv(results_path+'emptying_actions_PPO_volcriteria_average.csv')
    emptying_actions_PPO_CL_average = pd.read_csv(results_path+'emptying_actions_PPO_CL_average.csv')
    emptying_actions_optimal_analytic_average = pd.read_csv(results_path+'emptying_actions_optimal_analytic_average.csv')
    






## plot percentage volume deviation
fig, ax = plt.subplots(figsize=(15, 10))  # Adjust to create a single plot for side-by-side bar charts
plt.rcParams.update({"font.size": 20})  # Adjusting font size globally

# Data preparation for side-by-side bar chart
bar_width = 0.25  # width of the bars, adjusted to fit three bars side by side
index = np.arange(len(bunker_ids))  # the label locations

# Plot for vol_dev_PPO_volcriteria
bars1 = ax.bar(index - bar_width, vol_dev_PPO_volcriteria_average['PPO-volume criteria'], bar_width, label='PPO-volume criteria', color='orange')

# Plot for vol_dev_PPO_CL
bars2 = ax.bar(index, vol_dev_PPO_CL_average['PPO-CL'], bar_width, label='PPO-CL', color='green')

# Plot for vol_dev_optimal_analytic
bars3 = ax.bar(index + bar_width, vol_dev_optimal_analytic_average['Optimal Analytic Agent'], bar_width, label='Optimal Analytic Agent', color='violet')


# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Container IDs', fontsize=20)
ax.set_ylabel('Percentage of Volume deviation (%)', fontsize=20)
ax.set_title('Percentage Volume Deviation', fontsize=22)
ax.set_xticks(index)
ax.set_xticklabels(bunker_ids, fontsize=20)

# Position annotations in the top left corner
# plt.annotate(f'Avg. vol dev: PPO-volume criteria : {np.mean(vol_dev_PPO_volcriteria_average["PPO-volume criteria"]):.0f}%', xy=(0.01, 0.95),
#                  xycoords='axes fraction', fontsize=17)
# plt.annotate(f'Avg. vol dev: PPO-CL : {np.mean(vol_dev_PPO_CL_average["PPO-CL"]):.0f}%', xy=(0.01, 0.90),
#                  xycoords='axes fraction', fontsize=17)
# plt.annotate(f'Avg. vol dev: Optimal Analytic Agent : {np.mean(vol_dev_optimal_analytic_average["Optimal Analytic Agent"]):.0f}%', xy=(0.01, 0.85),
#                  xycoords='axes fraction', fontsize=17)

# Position legend in the top right corner
ax.legend(loc='upper right')

fig.tight_layout()

plt.savefig(results_path+'percentage_volume_deviation_comparison.png')
print("Comparison of percentage volume deviation for best agents saved as percentage_volume_deviation_comparison.png in "+results_path)






# # plot emptying actions 
# ## plot emptying actions comparison
# fig, ax = plt.subplots(figsize=(15, 10))  # Adjust to create a single plot for side-by-side bar charts
# plt.rcParams.update({"font.size": 20})  # Adjusting font size globally

# # Data preparation for side-by-side bar chart
# bar_width = 0.25  # width of the bars, adjusted to fit three bars side by side
# index = np.arange(len(bunker_ids))  # the label locations

# # Plot for emptying_actions_PPO_volcriteria
# bars1 = ax.bar(index - bar_width, emptying_actions_PPO_volcriteria_average['PPO-volume criteria'][1:], bar_width, label='PPO-volume criteria', color='orange')

# # Plot for emptying_actions_PPO_CL
# bars2 = ax.bar(index, emptying_actions_PPO_CL_average['PPO-CL'][1:], bar_width, label='PPO-CL', color='green')

# # Plot for emptying_actions_optimal_analytic
# bars3 = ax.bar(index + bar_width, emptying_actions_optimal_analytic_average['Optimal Analytic Agent'][1:], bar_width, label='Optimal Analytic Agent', color='violet')

# # Add some text for labels, title, and custom x-axis tick labels, etc.
# ax.set_xlabel('Container IDs', fontsize=20)
# ax.set_ylabel('Number of Emptying Actions', fontsize=20)
# ax.set_title('Comparison of Emptying Actions', fontsize=22)
# ax.set_xticks(index)
# ax.set_xticklabels(bunker_ids, fontsize=20)

# # Position annotations in the top left corner
# plt.annotate(f'Tot. emptying actions: PPO-volume criteria : {np.sum(emptying_actions_PPO_volcriteria_average["PPO-volume criteria"][1:] ):.2f}', xy=(0.01, 0.95),
#                  xycoords='axes fraction', fontsize=17)
# plt.annotate(f'Tot. emptying actions: PPO-CL : {np.sum(emptying_actions_PPO_CL_average["PPO-CL"][1:]):.2f}', xy=(0.01, 0.90),
#                  xycoords='axes fraction', fontsize=17)
# plt.annotate(f'Tot. emptying actions: Optimal Analytic Agent : {np.sum(emptying_actions_optimal_analytic_average["Optimal Analytic Agent"][1:]):.2f}', xy=(0.01, 0.85),
#                  xycoords='axes fraction', fontsize=17)

# # Position legend in the top right corner
# ax.legend(loc='upper right')

# fig.tight_layout()

# plt.savefig(results_path+'emptying_actions_comparison.png')
# print("Comparison of emptying actions for best agents saved as emptying_actions_comparison.png in "+results_path)


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
#combined_df_bestseeds['Average Inference Reward (±Std Dev)'] = round(combined_df_bestseeds['average_inference_reward'],2).astype(str) + " (±" + round(combined_df_bestseeds['standard_dev_in_inference_reward'],2).astype(str) + ")"
combined_df_bestseeds['Emptying actions'] = combined_df_bestseeds['emptying actions'].astype(str)
#combined_df_bestseeds['Reward per Empyting Action'] = round(combined_df_bestseeds['reward per emptying action'],2).astype(str)


#combined_df_bestseeds['Average Inference Overflow (±Std Dev)'] = round(combined_df_bestseeds['average_inference_overflow'],2).astype(str) + " (±" + round(combined_df_bestseeds['standard_dev_in_inference_overflow'],2).astype(str) + ")"

combined_df_bestseeds['Average Inference Overflow (±Std Dev)'] = combined_df_bestseeds['seed'].map({
    'PPO-Volume Criteria': f"{np.mean(vol_dev_PPO_volcriteria_average['PPO-volume criteria']):.2f} (±{np.std(vol_dev_PPO_volcriteria_average['PPO-volume criteria']):.2f})",
    'PPO-CL': f"{np.mean(vol_dev_PPO_CL_average['PPO-CL']):.2f} (±{np.std(vol_dev_PPO_CL_average['PPO-CL']):.2f})",
    'Optimal Analytic Agent': f"{np.mean(vol_dev_optimal_analytic_average['Optimal Analytic Agent']):.2f} (±{np.std(vol_dev_optimal_analytic_average['Optimal Analytic Agent']):.2f})"
})
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
plt.rcParams.update({"font.size": 25})  # Adjusting font size globally

fig, ax = plt.subplots(figsize=(15, 10))  # Adjusting figure size

# Customizing the palette to highlight specific agents
custom_palette = {"PPO-CL": "green", "PPO-Volume Criteria": "orange", "Optimal Analytic Agent": "violet"}
ax = sns.barplot(x="seed", y="Average Total Press Utilization", data=df_placeholder, palette=custom_palette)

# Setting the title and labels with the exact specified font sizes
plt.title('Average Total Press Utilization Across Agents', fontsize=25)
plt.xlabel('Agent', fontsize=23)
plt.ylabel('Average Total Press Utilization (secs)', fontsize=23)

# Adjusting y-axis with the specified start and adding space above the highest value for clarity
plt.ylim(60000, max(df_placeholder["Average Total Press Utilization"]) + 5000)

# Annotating the bars with the numeric values, adjusting for specified font size and placement
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points', fontsize=22)

#plt.xticks(rotation=45, fontsize=22)  # Adjusting x-axis labels for readability
plt.xticks(fontsize=20)  # Adjusting x-axis labels for readability

plt.yticks(fontsize=20)  # Adjusting y-axis labels font size
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

plt.savefig(results_path+'avg_total_press_utilization_best_agents.png')
print("Average Total Press Utilization for Best Agents saved as avg_total_press_utilization_best_agents.png in "+results_path)


