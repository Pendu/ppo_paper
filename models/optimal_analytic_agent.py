import pandas as pd
import numpy as np
import os
from models.press_models import validy_bunker

df_parameters_P1 = pd.read_csv("configs/Unterprozesse_Presse_1.csv").set_index("bunker")
df_parameters_P2 = pd.read_csv("configs/Unterprozesse_Presse_2.csv").set_index("bunker")


class optimal_analytic_agent():
    def __init__(self, n_bunkers = None, env = None):
        self.n_bunkers = n_bunkers
        self.env = env
    def predict(self, obs = None):
        env = self.env
        n_bunkers = self.n_bunkers
        clash_counter = 0
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
        peak_vols = []
        for i in range(n_bunkers):
            
            peak_vols.append(peak_rew_vols[env.bunker_ids[i][0]][-1])
        
        # Calculate differences and sort bunkers based on the proximity to their peak volumes
        bunker_diffs = [(i, abs(obs["Volumes"][i] - peak_vols[i])) for i in range(env.n_bunkers)]
        sorted_bunkers = sorted(bunker_diffs, key=lambda x: x[1])

        for i, _ in sorted_bunkers:

        #for i in range(env.n_bunkers): 
            
            #press-1 free but not press-2
            if peak_vols[i]-1<obs["Volumes"][i]<peak_vols[i]+1 and env.times["Time presses will be free"][0]==0 and env.times["Time presses will be free"][1]!=0: 
                action = i+1
                clash_counter+=1
                break
                
                #print("i came here")
            elif obs["Volumes"][i]>peak_vols[i]+1 and env.times["Time presses will be free"][0]==0 and env.times["Time presses will be free"][1]!=0:
                action = i+1
                clash_counter+=1
                break

                
            #press-2 free but not press-1
            if peak_vols[i]-1<obs["Volumes"][i]<peak_vols[i]+1 and env.times["Time presses will be free"][1]==0  and env.times["Time presses will be free"][0]!=0:
                action = i+1
                clash_counter+=1
                
                #print("i came here")
            elif obs["Volumes"][i]>peak_vols[i]+1 and env.times["Time presses will be free"][1]==0  and env.times["Time presses will be free"][0]!=0:
                action = i+1
                clash_counter+=1
                
            # both presses are free. choose press-1 or press-2 depending on validity of bunker
            if peak_vols[i]-1<obs["Volumes"][i]<peak_vols[i]+1 and env.times["Time presses will be free"][0]==0 and env.times["Time presses will be free"][1]==0:
                if validy_bunker(env.bunker_ids[i][1], list(df_parameters_P1.index)):
                    action = i+1
                    clash_counter+=1
                    #print("i was here")
                elif validy_bunker(env.bunker_ids[i][1], list(df_parameters_P2.index)):
                    action = i+1
                    clash_counter+=1
                    #print("i was here")
                else:
                    print("i am here in the nomads land")

            elif obs["Volumes"][i]>peak_vols[i]+1 and env.times["Time presses will be free"][0]==0 and env.times["Time presses will be free"][1]==0: 
                if validy_bunker(env.bunker_ids[i][1], list(df_parameters_P1.index)):
                    action = i+1
                    clash_counter+=1
                    #print("no no i was here")

                else:
                    action = i+1
                    clash_counter+=1                    
                    #print("no no i was here")

                
            # no action
            elif not clash_counter:
                action = 0
        return action, clash_counter