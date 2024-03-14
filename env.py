import gymnasium as gym
from gymnasium import error, spaces, utils
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.penalty import reward as freward
from models.press_models import Absolute_time_Press_1, Absolute_time_Press_2
from collections import deque
import numpy as np
import pandas as pd
import torch
import math
from models.random_walk import future_volume_v

class SutcoEnv(gym.Env):

    def __init__(self, max_episode_length, verbose_logging, bunkers, use_minr, number_of_presses, CL_step):
        super(SutcoEnv, self).__init__()
        self.render_mode = "human"
        self.bunker_ids_all = [
            ["C1-10", "C1-1"],
            ["C1-20", "C1-2"],
            ["C1-30", "C1-3"],
            ["C1-40", "C1-4"],
            ["C1-50", "C1-5"],
            ["C1-60", "C1-6"],
            ["C1-70", "C1-7"],
            ["C1-80", "C1-8"],
            ["C2-10", "C2-1"],
            ["C2-20", "C2-2"],
            ["C2-40", "C2-4"],
            ["C2-50", "C2-5"],
            ["C2-60", "C2-6"],
            ["C2-70", "C2-7"],
            ["C2-80", "C2-8"],
            ["C2-90", "C2-9"]
        ]
        self.CL_step = CL_step
        self.bunker_ids = []
        for i in bunkers:
            self.bunker_ids.append(self.bunker_ids_all[i])
        self.bunker_ids = self.bunker_ids
        self.n_bunkers = len(self.bunker_ids)

        self.verbose_logging = verbose_logging
        self.number_of_presses = number_of_presses

        # Define state variable
        # Set initial volumes
        self.min_vol = 0
        self.max_vol = 30
        self.state = {
            "Bunkers being emptied": np.zeros(self.n_bunkers),  # 1 if bunker is being emptied, 0 otherwise
            "reward": np.random.uniform(0, 0, size=self.n_bunkers),
            "Time presses will be free normalized": np.array(2 * [0.]),
            "peak vol": np.array([26.75,  # c1-20
                                  26.52,  # 10 #c1-30
                                  8.34,  # c1-40
                                  14.34,  # 15 #c1-60
                                  25.93,  # 20 #c1-70
                                  24.75,  # 32 #c1-80
                                  27.39,  # c2-10
                                  32,  # c2-20
                                  12.6,  # c2-60
                                  17,  # c2-70
                                  28.75  # c2-80
                                  ]),  # np.array([26.75,26.75,26.75]), #np.array([26.75,10,15]),
        }  # Duration in sec.


        if self.CL_step in [1, 1.5, 2, 3]:
            self.state["Volumes"] = np.array([
                                           np.random.uniform(19,24.5,1)[0], #c1-20
                                           np.random.uniform(19,24.5,1)[0], #c1-30
                                           np.random.uniform(2.5, 6.5, 1)[0], #c1-40
                                           np.random.uniform(9,13,1)[0], #c1-60
                                           np.random.uniform(19,24,1)[0], #c1-70
                                           np.random.uniform(19,23,1)[0],#c1-80
                                           np.random.uniform(22, 26, 1)[0],# c2-10
                                           np.random.uniform(26, 30.5, 1)[0],  # c2-20
                                           np.random.uniform(6, 10, 1)[0],  # c2-60
                                           np.random.uniform(11, 15.5,1)[0],  # c2-70
                                           np.random.uniform(22, 26, 1)[0],  # c2-80

                                           ])
        elif self.CL_step in [4,5]:
            self.state["Volumes"]= np.array([
                                           np.random.uniform(0,24.5,1)[0], #c1-20
                                           np.random.uniform(0,24.5,1)[0], #c1-30
                                           np.random.uniform(0, 6.5, 1)[0], #c1-40
                                           np.random.uniform(0,13,1)[0], #c1-60
                                           np.random.uniform(0,24,1)[0], #c1-70
                                           np.random.uniform(0,23,1)[0],#c1-80
                                           np.random.uniform(0, 26, 1)[0],# c2-10
                                           np.random.uniform(0, 30.5, 1)[0],  # c2-20
                                           np.random.uniform(0, 10, 1)[0],  # c2-60
                                           np.random.uniform(0, 15.5,1)[0],  # c2-70
                                           np.random.uniform(0, 26, 1)[0],  # c2-80

                                           ])
        elif self.CL_step in [6,7]:
            self.state["Volumes"] = np.random.uniform( self.min_vol, self.max_vol, size=self.n_bunkers)


        self.times = {
            "Time presses will be free": np.array(2 * [0.]),
            # # Duration in sec. Index 0: Press 1, index 1: Press 2
            "Time conveyor belts will be free": np.array(2 * [0.]),
            # # Duration in sec. Conveyor belts (cbs) of Press 2. 0: left cb, 1: right cb
            "Time emptying ends": np.array(self.n_bunkers * [0.]),
            # "Bunkers being emptied": np.zeros(self.n_bunkers),  # 1 if bunker is being emptied, 0 otherwise

        }


        self.observation_space = spaces.Dict({
            "Volumes": spaces.Box(low=self.min_vol, high=self.max_vol, shape=(self.n_bunkers,), dtype=np.float32),
            "Bunkers being emptied": spaces.Box(low=0, high=1, shape=(self.n_bunkers,), dtype=np.float32),
            "reward": spaces.Box(low=-10, high=1000, shape=(self.n_bunkers,), dtype=np.float32),
            "Time presses will be free normalized": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
            "peak vol": spaces.Box(low=0, high=40, shape=(self.n_bunkers,), dtype=np.float32),
        })
        


        # Define action space
        if number_of_presses == 1:
            self.action_space = spaces.Discrete(self.n_bunkers + 1)  # (2 * self.n_bunkers + 1)

            # self.action_space = spaces.Discrete(1 + 1 +1 )  # (2 * self.n_bunkers + 1)
        if number_of_presses == 2:
            self.action_space = spaces.Discrete(self.n_bunkers + 1)  # (2 * self.n_bunkers + 1)
            
        if self.CL_step in [1,1.5, 2, 3]:
            self.timestep = 30  # Time step in seconds
        elif self.CL_step in [4, 5, 6,7]:
            self.timestep = 60  # Time step in seconds
        

        # Set maximal length for an episode (in min)
        self.max_episode_length = max_episode_length  # 1200  # 300

        # Variable to compute episode length
        self.episode_length = 0

        self.bunkers_meta_data = pd.read_pickle("configs/bunkers_meta_data.pkl")
        self.bale_sizes = {"C1-10": 11,
                           "C1-20": 27,
                           "C1-30": 12.5,
                           "C1-40": 7.5,
                           "C1-50": 5.5,
                           "C1-60": 8.5,
                           "C1-70": 10.5,
                           "C1-80": 12.5,
                           "C2-10": 18,
                           "C2-20": 35,
                           "C2-40": 6.5,
                           "C2-50": 6,
                           "C2-60": 6.3,
                           "C2-70": 16,
                           "C2-80": 28.5,
                           "C2-90": 6.5}

        # List of bunkers that can be pressed by Press 1 (raw data in Unterprozesse_Presse_1.csv)
        self.bunkers_press_1 = ["C1-20", "C1-30", "C1-40", "C1-60", "C1-70", "C1-80", "C2-20", "C2-50", "C2-70",
                                "C2-80"]

        # List of bunkers that can be pressed by Press 2 (raw data in Unterprozesse_Presse_2.csv)
        self.bunkers_press_2 = ["C1-20", "C1-30", "C1-50", "C1-60", "C1-70", "C1-80", "C2-10", "C2-20", "C2-50",
                                "C2-60", "C2-70", "C2-80"]

        # lists for verbose logging
        self.action_hist = []
        self.volume_hist = []
        self.peak_rew_vols_current = []
        # self.iter_step = 0

        self.peak_rew_vols = {#"C1-10": [19.84],
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


        for i in range(self.n_bunkers):
            self.peak_rew_vols_current.append(self.peak_rew_vols[self.bunker_ids[i][0]][-1])

        self.use_minr = use_minr
        self.action_history = deque(maxlen=10)  # Store the last 10 actions
        self.action_history.append(0)   # Initialize with 0
        self.pending_action = None  # Store the pending action

    def step(self, action):
        # Apply action (update state)
        self.action_history.append(action)

        old_volumes = self.state["Volumes"].copy()  # Copy current volumes. Needed to calculate reward
        press_is_free = False
        self.state[
            "peak vol"] = self.peak_rew_vols_current  

        if action == 0:  # Do nothing
            self.state["Volumes"] = future_volume_v([i[0] for i in self.bunker_ids], self.state["Volumes"],
                                                    self.timestep, self.CL_step)
            # self.iter_step+=1

            for i in range(self.n_bunkers):
                if self.state["Bunkers being emptied"][i] == 1:
                    self.state["Volumes"][i] = old_volumes[
                        i]  # override the volume with previous volume i.e. zero (in the future try zero and check)



        elif action in range(1, self.n_bunkers + 1):  # Use Press 1 or press 2 depending on which is free

            self.state["Volumes"] = future_volume_v([i[0] for i in self.bunker_ids], self.state["Volumes"],
                                                    self.timestep, self.CL_step)


            ## Use - press-1:

            if self.times["Time presses will be free"][0] == 0 and self.bunker_ids[action-1][0] in self.bunkers_press_1: 

                # check if press-1 is free and if the bunker is press-1 compatible
                if self.CL_step in [1, 1.5, 2, 3]:
                    t_pressing_ends = timedelta(0) 
                elif self.CL_step in [4, 5, 6,7]:
                    t_pressing_ends = Absolute_time_Press_1(timedelta(0),
                                        timedelta(seconds=self.times["Time presses will be free"][0]),
                                        self.bunker_ids[action - 1][1],
                                        round(self.state["Volumes"][action - 1]
                                                / self.bale_sizes[self.bunker_ids[action - 1][0]]))


                if t_pressing_ends is not None:  # Emptying is possible

                    press_is_free = True  # Used to calculate reward 
                    t_pressing_ends = t_pressing_ends.total_seconds()
                    # override volume
                    self.state["Volumes"][action - 1] = 0.
                    self.state["Bunkers being emptied"][action - 1] = 1
                    self.times["Time presses will be free"][0] = t_pressing_ends
                    self.state["Time presses will be free normalized"][0] = t_pressing_ends / self.timestep
                    self.times["Time emptying ends"][action - 1] = t_pressing_ends  
                else:  # Emptying is not possible
                    press_is_free = False
                    self.state = self.state
                    print("t_pressing_ends is", t_pressing_ends, "although press-1 is free with time presses will be free normalized = 0 and action is", action)
                    print("self.times['Time presses will be free'][0] is", self.times["Time presses will be free"][0])


            ####Use press-2

            elif self.times["Time presses will be free"][1] == 0 and self.bunker_ids[action-1][0] in self.bunkers_press_2: 

                if self.CL_step in [1,1.5, 2, 3]:

                    res_press_2 = (0, 0, 0)
                elif self.CL_step in [4, 5, 6,7]:
                    res_press_2 = Absolute_time_Press_2(timedelta(0),
                                          timedelta(seconds=self.times["Time presses will be free"][1]),
                                          timedelta(seconds=self.times["Time conveyor belts will be free"][0]),
                                          timedelta(seconds=self.times["Time conveyor belts will be free"][1]),
                                          self.bunker_ids[action - self.n_bunkers - 1][1],
                                          round(self.state["Volumes"][action - self.n_bunkers - 1]
                                                / self.bale_sizes[self.bunker_ids[action - self.n_bunkers - 1][0]]))

                if res_press_2 is None:

                    print("res_press_2 is None although press-2 is free with time presses will be free normalized = 0 with action", action, "empyting is not possible")
                    # Emptying is not possible
                    press_is_free = False
                    self.state = self.state
                else:  # Emptying is possible
                    t_pressing_ends, t_left_cb_free, t_right_cb_free = res_press_2


                    if self.CL_step in [4,5,6,7]:
                        t_pressing_ends = max(0, t_pressing_ends.total_seconds())
                        t_left_cb_free = max(0, t_left_cb_free.total_seconds())
                        t_right_cb_free = max(0, t_right_cb_free.total_seconds())

                    press_is_free = True  
                    
                    # override the volume
                    self.state["Volumes"][action - self.n_bunkers - 1] = 0.

                    self.state["Bunkers being emptied"][action - self.n_bunkers - 1] = 1
                    self.state["Time presses will be free normalized"][1] = t_pressing_ends / self.timestep
                    self.times["Time presses will be free"][1] = t_pressing_ends
                    self.times["Time conveyor belts will be free"][0] = t_left_cb_free
                    self.times["Time conveyor belts will be free"][1] = t_right_cb_free
                    self.times["Time emptying ends"][action - self.n_bunkers - 1] = t_pressing_ends

            # emptying is not possible
            else: 
                press_is_free = False


        # Decrease time counters
        for i in range(len(self.times["Time presses will be free"])):
            self.times["Time presses will be free"][i] = max(0, self.times["Time presses will be free"][i]
                                                             - self.timestep)
            self.state["Time presses will be free normalized"][i] = max(0, self.times["Time presses will be free"][i] / self.timestep- 1)

        for i in range(len(self.times["Time conveyor belts will be free"])):
            self.times["Time conveyor belts will be free"][i] = max(0, self.times["Time conveyor belts will be free"][i]
                                                                    - self.timestep)

        for i in range(self.n_bunkers):
            self.times["Time emptying ends"][i] = max(0, self.times["Time emptying ends"][i]
                                                      - self.timestep)
            # Update bunker status if emptying has ended
            if self.times["Time emptying ends"][i] == 0 and self.state["Bunkers being emptied"][i] == 1:
                self.state["Bunkers being emptied"][i] = 0

        # Increment episode length by 1
        self.episode_length += 1

        # Max volume is 40 for all bunkers
        max_vols = self.bunkers_meta_data.loc[self.bunkers_meta_data["bunker"].isin(np.array(self.bunker_ids)
                                                                                    [:, 1])]["Vol_max"].values

        
        episode_termination_reward = 0
        # Check if episode is done
        if len(np.where(self.state["Volumes"] > max_vols)[0]) > 0 or self.episode_length == self.max_episode_length:

            # End episode
            done = True
            

            if self.episode_length == self.max_episode_length:
                # small positive reward for successful episode 
                episode_termination_reward = 0.2 
            if len(np.where(self.state["Volumes"] > max_vols)[0]) > 0:              
                # huge negative reward for overshooting max volume. happens only once in an episode
                episode_termination_reward = -3

        else:
            # Episode is not done
            done = False
        

        #In the range of [-0.1,1] for all bunkers on which action was not taken
        def distance_reward(current_vol, ideal_vol):

            if current_vol <= ideal_vol:
                dist_rew = 1 - abs(((ideal_vol - current_vol) / ideal_vol)) ** 0.5

            else:
                dist_rew = -0.1 

            return dist_rew

        # Simple Gaussian reward function for PPO Baseline Agent
        if self.CL_step in [6]:
            
            reward = 0
            for i in range(self.n_bunkers):

                a = 0  # Action performed on Bunker i
                if (i == action - 1) or (i == action - self.n_bunkers - 1):
                    a = action
                xyz = freward(a, old_volumes[i], press_is_free, self.bunker_ids[i][0], self.use_minr)
                self.state["reward"][i] = xyz
                reward += xyz
                
        #         if action>0:
        #             reward = reward - 0.1 #penalize every successful action to reduce the number of actions
                    
        # # # Add episode termination reward to the reward
        # reward = reward + episode_termination_reward # 
                
                
        def get_bonus(old_volumes, ideal_vol, xyz):
             
            bonus = 0
            if xyz > 0: 
                if 0.7 + ideal_vol < old_volumes < 1.5 + ideal_vol:  
                    bonus = 3.5

                elif 0.3 + ideal_vol < old_volumes < 0.7 + ideal_vol:
                    bonus = 10.0  

                elif -0.3 + ideal_vol < old_volumes < 0.3 + ideal_vol:
                    bonus = 20.0 

                elif -0.7 + ideal_vol < old_volumes < -0.3 + ideal_vol:
                    bonus = 10.0 

                elif -1.5 + ideal_vol < old_volumes < -0.7 + ideal_vol: 
                    bonus = 3.5

                elif -2.5 + ideal_vol <old_volumes < -1.5 + ideal_vol: 
                    bonus = 2.5

                elif -3.5 + ideal_vol < old_volumes < -2.5 + ideal_vol: 
                    bonus = 2.0

                elif -5.0 + ideal_vol < old_volumes < -3.5 + ideal_vol:
                    bonus = 1.5
                else:
                    bonus = -0.5  # small negative bonus, if it empties in non-ideal range
                    
            return bonus
        
        
        #if self.CL_step in [100]: 
        if self.CL_step in [1]: 

            action_reward = 0  # action reward (only for bunker being emptied, else zero). happens each and every time step of an episode
            bonus = 0  # positve or negative value added to action reward, happens only once or twice and episode
            postional_reward = 0 # position reward (for all bunkers, based on their distance from the ideal volume). happens every time step and has a range of 0 to 1, with 1 at ideal volume. This pushes the agent to
            
            for i in range(self.n_bunkers):
                a = 0 
                
                if (i == action - 1) or (i == action - self.n_bunkers - 1):
                     # action performed on bunker i
                    a = action
                    
                # if  a is 0 xyz will be zero; xyz will be between [-0.1,1] only for bunker on which action was taken
                xyz = freward(a, old_volumes[i], press_is_free, self.bunker_ids[i][0],self.use_minr)  

                ideal_vol = self.peak_rew_vols[self.bunker_ids[i][0]][-1]

                # weightage of action reward is 5 times that of positional reward
                action_reward = action_reward + 5 * xyz + get_bonus(old_volumes[i], ideal_vol,xyz)
                
                # sum of non-action rewards for all bunkers
                postional_reward = postional_reward + distance_reward(self.state["Volumes"][i],ideal_vol)   # this will be in the range of n_bunkers * ([-0.1]+(0,1])

                self.state["reward"][i] = action_reward

            action_and_postional_reward = (postional_reward/11)*25 + action_reward 

            if action: 
                factor = 1
            else: 
                factor = 0

            if self.CL_step in [1,6]:
                neg_mul = 0.0
            elif self.CL_step == 2:
                neg_mul = 0.1
            
            # scale the reward to be in the range of [-0.1,1]      
            reward = (episode_termination_reward + action_and_postional_reward)/51 - neg_mul*factor
            
            
        #if self.CL_step in [1.5]:
        if self.CL_step in [1.5,2,4,5]:
        #if self.CL_step in [1.5,2,4,5,6]:

            reward = 0
            ideal_vol = self.peak_rew_vols[self.bunker_ids[action-1][0]][-1]


            if action > 0:

                if -0.5 +ideal_vol < old_volumes[action-1] < 0.5 + ideal_vol:  # auxillary reward to push bunker towards higher volumes for slow filling bunkers
                    reward = 1.0
                    self.state["reward"][action-1] = reward
                else:
                    reward =  -0.2* abs(((ideal_vol - self.state["Volumes"][i]) / ideal_vol)) ** 0.5

                if self.CL_step in [2]:
                    reward = reward - 0.1 #penalize every successful action to reduce the number of actions
                    

        ## Simple reward function for CL_step 3,4,5 . also use this for rule based agent. 
        if self.CL_step in [3, 4, 5]:
        #if self.CL_step in [6]:

            ideal_vol = self.peak_rew_vols[self.bunker_ids[action-1][0]][-1]

            reward = 0
            press_penalty = 0
            neg_mul = 0.0
            factor = 0


            if action > 0:

                if self.CL_step in [4,5]:
                    neg_mul = 0.0
                elif self.CL_step == [3]:
                    neg_mul = 0.1

                if action: 
                    factor = 1
                else: 
                    factor = 0
                


                if -1.0 +ideal_vol < old_volumes[action-1] < 1.0 + ideal_vol:
                    reward = 1.0
                    self.state["reward"][action-1] = reward
                    reward = reward- neg_mul*factor
                    if action == 0: 
                        print("action is zero but reward given is one. there is a bug in your code")
                else:
                    if self.CL_step==3:
                        reward = -0.1
                    elif self.CL_step in [4,5]:
                        reward = 0.0
                    reward = reward- neg_mul*factor

                reward = reward + press_penalty
                
        if self.CL_step in [7]: #also use this for rule based agent. 

            ideal_vol = self.peak_rew_vols[self.bunker_ids[action-1][0]][-1]

            reward = 0

            if action > 0:


                if -0.5 +ideal_vol < old_volumes[action-1] < 0.5 + ideal_vol:  
                    reward = 1.0
                    self.state["reward"][action-1] = reward
                    reward = reward
                    if action == 0: 
                        print("action is zero but reward given is one. there is a bug in your code")
                else:
                    reward = 0

                reward = reward
            


        # remove 6 from above and add 6 here when you want to test the press penalty
        # if self.CL_step in [6]:

        #     reward = 0
        #     press_penalty = 0

        #     if action: 
        #         if old_volumes[action-1] == 0. or self.times["Time presses will be free"][0] != 0. or self.times["Time presses will be free"][1] != 0.:
        #             press_penalty = -0.1
        #     else:
        #         press_penalty = 0.0

        #     reward = press_penalty

        
            
        # print(f'reward is {reward}')

        # reward = reward + termination_rew

        # Verbose Logging
        info = {}
        if self.verbose_logging:
            self.action_hist.append(action)
            self.volume_hist.append(list(self.state["Volumes"]))
            # Prepare info
            info = {
                "action": self.action_hist,
                "volumes": self.volume_hist
            }

        
        terminated = done
        
        truncated = False
        
        #return  self.state, reward, terminated, truncated, info
        
        return (
            self.state,
            reward,
            terminated,
            truncated,
            info,
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)


        if self.CL_step in [1, 1.5, 2, 3, 4]:
            self.state["Volumes"] = np.array([
                                            np.random.uniform(0,24.5,1)[0], #c1-20
                                            np.random.uniform(0,24.5,1)[0], #c1-30
                                            np.random.uniform(0, 6.5, 1)[0], #c1-40
                                            np.random.uniform(0,13,1)[0], #c1-60
                                            np.random.uniform(0,24,1)[0], #c1-70
                                            np.random.uniform(0,23,1)[0],#c1-80
                                            np.random.uniform(0, 26, 1)[0],# c2-10
                                            np.random.uniform(0, 30.5, 1)[0],  # c2-20
                                            np.random.uniform(0, 10, 1)[0],  # c2-60
                                            np.random.uniform(0, 15.5,1)[0],  # c2-70
                                            np.random.uniform(0, 26, 1)[0],  # c2-80

                                            ])
        elif self.CL_step in [5]:

            #self.state["Volumes"] = np.zeros(self.n_bunkers)# don't do random here try to start from a particular far away value


            self.state["Volumes"]= np.array([
                                            np.random.uniform(0,24.5,1)[0], #c1-20
                                            np.random.uniform(0,24.5,1)[0], #c1-30
                                            np.random.uniform(0, 6.5, 1)[0], #c1-40
                                            np.random.uniform(0,13,1)[0], #c1-60
                                            np.random.uniform(0,24,1)[0], #c1-70
                                            np.random.uniform(0,23,1)[0],#c1-80
                                            np.random.uniform(0, 26, 1)[0],# c2-10
                                            np.random.uniform(0, 30.5, 1)[0],  # c2-20
                                            np.random.uniform(0, 10, 1)[0],  # c2-60
                                            np.random.uniform(0, 15.5,1)[0],  # c2-70
                                            np.random.uniform(0, 26, 1)[0],  # c2-80

                                            ])
        elif self.CL_step in [6,7]:
            self.state["Volumes"] = np.random.uniform( self.min_vol, self.max_vol, size=self.n_bunkers)

        # np.zeros(self.n_bunkers)
        self.times["Time presses will be free"] = np.array(2 * [0.])
        self.state["Time presses will be free normalized"] = np.array(2 * [0.])
        self.times["Time conveyor belts will be free"] = np.array(2 * [0.])
        self.state["Bunkers being emptied"] = np.zeros(self.n_bunkers)
        self.times["Time emptying ends"] = np.array(self.n_bunkers * [0.])
        self.state["reward"] = np.random.uniform(0, 0, size=self.n_bunkers)
        #self.state["press_is_free"] = np.zeros(1)  # np.array([0])
        self.state[
            "peak vol"] = self.peak_rew_vols_current  # np.array([26.75,10,15,20,32]) #np.array([26.75,10,15]) #np.array([26.75,26.75,26.75]) # np.array([26.75,10,15])
        # self.state["ideal rew"] = np.array([175])

        # Reset episode length
        self.episode_length = 0

        self.action_history.clear()  # Clear the action history


        # Reset logging
        self.action_hist = []
        self.volume_hist = []

        info = {}
        
        #return self.state
        return self.state, info 


    def render(self):
        # Implement Visualization
        pass


    def action_masks(self):
        """
        Generates a binary mask indicating the possible actions for the current state.

        Returns:
            np.array: Binary mask indicating the possible actions. Each element represents whether the corresponding action is possible (True) or not (False).
        """
        
        # Initialize the press actions to a default of fault
        if  self.number_of_presses==1:
            possible_actions = [True] * (self.n_bunkers + 1) # For one press
        else:    
            possible_actions = [True] * (self.n_bunkers + 1) # For two presses

            
        #set basic action to be true
        possible_actions[0] = True  # Action 0 (do nothing) is always possible


        # # if number of presses is two
        if self.number_of_presses ==2:


            # Check if each bunker is compatible with available presses
            for i in range(self.n_bunkers):
                bunker_id = self.bunker_ids[i][0]
                can_use_press_1 = bunker_id in self.bunkers_press_1 and self.times["Time presses will be free"][0] <= 0
                can_use_press_2 = bunker_id in self.bunkers_press_2 and self.times["Time presses will be free"][1] <= 0

                # Set action as possible if either press is available for this bunker
                possible_actions[i + 1] = can_use_press_1 or can_use_press_2

            # press-1 and 2 are busy, so cannot use it 
            if self.times["Time presses will be free"][0] > 0 and self.times["Time presses will be free"][1] > 0: 
                for i in range(self.n_bunkers):
                    possible_actions[i+1] = False
                
            # certain bunkers cannot be emptied on press-1 or press-2
            if self.times["Time presses will be free"][0] > 0:
                possible_actions[3] = False
            if self.times["Time presses will be free"][1] > 0:
                possible_actions[7] = False
                possible_actions[9] = False
                
        return np.array(possible_actions)