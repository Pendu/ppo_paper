import os 
import math
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

# print out the current working directory
print(os.getcwd())

# Read parameters
df_parameters_P1 = pd.read_csv("configs/Unterprozesse_Presse_1.csv").set_index("bunker")
df_parameters_P2 = pd.read_csv("configs/Unterprozesse_Presse_2.csv").set_index("bunker")

# bunkers on left/right side
rechte_seite = ["C1-1", "C1-2", "C1-3", "C1-4", "C1-5", "C1-6", "C1-7", "C1-8", "C1-9"]
linke_seite =  ["C2-1", "C2-2", "C2-3", "C2-4", "C2-5", "C2-6", "C2-7", "C2-8", "C2-9"]

def validy_bunker(bunker, list_bunker):
    # check if bunker is able to be pressed by choosen press
    if bunker not in list_bunker:
        return False
    
    else:
        return True

def duration_generator_press_1(bunker: str, number_of_bales: int):
    """
    Return the avereage amount of minutes to discharge the bunker
    """
    t_ges = validy_bunker(bunker, list(df_parameters_P1.index))
    
    # DataFrame mit den Unterprozessen für Presse 1 verwenden
    df_parameters = df_parameters_P1
    
    # Auslesen der entsprechenden Parameter
    t_offset = df_parameters["t_offset"].loc[bunker]
    t_mat_hub = df_parameters["t_mat_hub"].loc[bunker]
    n_hueb_bale = df_parameters["n_Huebe_pro_Ballen"].loc[bunker]
    t_Hub_vor = df_parameters["t_Hub_vor"].loc[bunker]
    t_Hub_zurueck = df_parameters["t_Hub_zurück"].loc[bunker]
    t_WZ_zw_Hueben_pro_Ballen = df_parameters["t_WZ_zw_Hueben_pro_Ballen"].loc[bunker]
    t_Abbinden = df_parameters["t_Abbinden"].loc[bunker]
    
    # Gesamtdauer berechnen
    t_ges = (t_offset + number_of_bales * (t_mat_hub + (n_hueb_bale * (t_Hub_vor + t_Hub_zurueck)) +
                                      t_WZ_zw_Hueben_pro_Ballen + t_Abbinden))
    return t_ges

def validy_time_P1(t_next, t_pr_alt):
    """
    Return, whether this Bunker can be discharged on press 1
    """
    if t_next < t_pr_alt:
        t_bis_fertig = round((t_pr_alt - t_next).total_seconds() / 60, 3)
        if t_bis_fertig > 0:            
            # print Ausgaben         
            # print("\nPress 1 is not free!\nYou have to wait " + str(t_bis_fertig) + " minutes, until press will be ready at " + str(t_pr_alt))
            return False
        else:
            # print("\nPress is still ready since " + str(abs(t_bis_fertig)) + " minutes!")
            return True
    else:
        return True

def Absolute_time_Press_1(t_next, t_pr_alt, bunker: str, n_bales = int):
    
    """
    Return the duration how long my given bunker will need to finish the whole process.
    Calling this functions it will check if the press is free or not, based on given paramters t_next and t_pr_alt.
    To get some feedback the functions generates some prints.
    The return value gives you the datetime when the press will be ready and free again. Use this timestamp/datetime
    for next call of the function. So you can go iterativ through press 1.
        
        Parameter:
                t_next (datetime.datetime): Current time/Datetime when the next bunker should be send to press 1
                t_pr_alt (datetime.datetime): (old, previous) Time/Datetime when the press of previous process will be/was free again
                bunker (str): Shortcut of bunkername ---> e.g. "C1-2"
                n_bales (int): Number of bales you want to press
                
        Returns:
                t_pr_ready (datetime.datetime): (future) Time/Datetime, when the process is finished and the press will be free again
    """
    
    # Check if bunker is allowed to be pressed by press 1
    if validy_bunker(bunker, list(df_parameters_P1.index)) == False:
        print("\nChoose another bunker! Bunker " + bunker + " is not suitable for press 1!")
        return None
    else:
        # check if press is already free
        if validy_time_P1(t_next, t_pr_alt) == False:
            #print("\nWait until press 1 is free!")
            return None
        else:

            t_duration = duration_generator_press_1(bunker = bunker, number_of_bales = n_bales)

            # reshape duration to timedelta
            t_duration_delta = timedelta(minutes = t_duration)

            # determine the datetime, when press 1 will be ready/free after
            t_pr_neu = t_next + t_duration_delta

            return t_pr_neu

def duration_generator_press_2(bunker: str, number_of_bales: int):
    """
    This function calculate the duration of the emptying and pressing process of a bunker
    based on the number of bales and the choosen bunker. The outputs of the functions are
    durations (relative times) and are used to determine timestamps/absolute times for
    future process states (like:
     - when is the pressingprocess finished = press is free,
     - time when it is allowed to pull the next bunker from left or right bunkerside)
    
        Parameter:
                bunker (string): name/shortcut of the bunker which is to be pressed next
                number_of_bales (int): disered number of bales want to be pressed
    
        return:
                t_offset (float64): the duration (in minutes) the band runs from bunker opening to first material in press
                t_press (float64): the duration (in minutes) press needs to press the material of the bunker (from first material in press to finish the last bale)
    
    NOTE:
    Case of fully seriell process!!!
    
    """
    t_ges = validy_bunker(bunker, list(df_parameters_P2.index))
    
    # use dataframe with calculation values of press 2
    df_parameters = df_parameters_P2
    
    # read out important values
    t_offset = df_parameters["t_offset"].loc[bunker] # time the material needs from the bunker to the press
    t_mat_hub = df_parameters["t_mat_hub"].loc[bunker] # time from first material in press to first stroke
    n_hueb_bale = df_parameters["n_Huebe_pro_Ballen"].loc[bunker] # number of strokes
    t_Hub_vor = df_parameters["t_Hub_vor"].loc[bunker] # time for a forward stroke
    t_Hub_zurueck = df_parameters["t_Hub_zurück"].loc[bunker] # time for a backward stroke
    t_WZ_zw_Hueben_pro_Ballen = df_parameters["t_WZ_zw_Hueben_pro_Ballen"].loc[bunker] # waittime between strokes
    t_Abbinden = df_parameters["t_Abbinden"].loc[bunker] # binding time
    
    # calculate duration of pressing process
    t_press = number_of_bales*(t_mat_hub + (n_hueb_bale * (t_Hub_vor + t_Hub_zurueck)) + t_WZ_zw_Hueben_pro_Ballen + t_Abbinden)

    return t_offset, t_press

def validy_time_P2(t_next, t_links_alt, t_rechts_alt, bunker: str):
    """
    Checks, whether the Bunker can be discharged at time t_next
    Input: datetime.datetime
    """
    if bunker in linke_seite:
        t_follow = t_links_alt
        side = "left"
    if bunker in rechte_seite:
        t_follow = t_rechts_alt
        side = "right"
        
    if t_next >= t_follow:
        return True
    else:
        # print("You have to wait until " + str(t_follow) + " to pull bunkers from " + side + " bunkerside!")
        return False 
    
def Absolute_time_Press_2(t_next, t_pr_alt, t_links_alt, t_rechts_alt, bunker: str, n_bales: int):
    
    """
    Calculate the datetime, when given bunker on press 2 is finished. At first this function checks whether
    the press is free or not and if we are able to pull given bunker on the left or right side at given time (t_next).
    
    So if the press is still working (t_next < t_pr_alt) it is possible that we are allowed to pull given bunker
    from the left or right side (if t_next > t_links_alt OR t_next > t_rechts_alt).
    With respect to parellel processes this function returns 3 times/datetimes (see 'Returns').
    The function prints out some informations about times and time it safes while running parallel.
    
        Parameter:
                t_next (datetime.datetime): Current time/datetime when you want to empty given bunker to press 2
                t_links_alt (datetime.datetime): time/datetime, when the previous process allows to send a new bunker from left bunkerside
                t_rechts_alt (datetime.datetime): time/datetime, when the previous process allows to send a new bunker from right bunkerside
                t_pr_alt (datetime.datetime): time/datetime, when the previous process is finished and the press is free
                bunker (str): Shortcut of bunkername ---> e.g. "C1-2"
                n_bales (int): Number of bales you want to press
                
        Returns:
                t_pr_ready (datetime.datetime): time/datetime, when given bunker will be pressed and finished (in the future)
                                                OR old value if given bunker is not able to start process
                t_links_neu (datetime.datetime): time/datetime, when left bunkerside is allowed to start emptying process (in the future)
                                                 OR old value if given bunker is not able to start process
                t_rechts_neu (datetime.datetime): time/datetime, when right bunkerside is allowed to start emptying process (in the future)
                                                  OR old value if given bunker is not able to start process
    """

    # Check if bunker is allowed to be pressed by press 2
    if validy_bunker(bunker, list(df_parameters_P2.index)) == False:
        # print("\nChoose another bunker! Bunker " + bunker + " is not suitable for press 2!")
        return None
    else:
        # load important times from csv
        dt_klappe_zu_to_rdy = timedelta(minutes = df_parameters_P2["t_klappe_zu_to_rdy"].loc[bunker]) # duration of closed shelter to 'bale_is_ready' [minutes]
        dt_US_to_bale = timedelta(minutes = df_parameters_P2["t_close_US_to_rdy"].loc[bunker]) # duration from 'Close_bandpos_at_US' to 'bale_is_ready' [minutes]   
        dt_stop_to_press_to_2 = timedelta(minutes = df_parameters_P2["t_stop_front_press_to_press"].loc[bunker]) # duration of stopp in front of laser to reach press [minutes]
        
        

        # Checks, whether the bunker can be discharged at time t_next
        if validy_time_P2(t_next, t_links_alt, t_rechts_alt, bunker) == False: 
            # print("\nWait until left or right side is able to be emptied")
            return None
        else:
            min_offset, min_press = duration_generator_press_2(bunker = bunker, number_of_bales = n_bales)

            # reshape to timedelta
            dt_offset = timedelta(minutes = min_offset)
            dt_press = timedelta(minutes = min_press)
            t_seriell = t_pr_alt + dt_press + dt_offset

            # Check the sides of the Bunkers
            if t_links_alt > t_rechts_alt:
                last_side = "left"
            else:
                last_side = "right"     
            if bunker in rechte_seite:
                new_side = "right"            
            if bunker in linke_seite:
                new_side = "left"
            if new_side == last_side:
                different_sides = True
            else:
                different_sides = False

            # Maximum time of the press in parallel mode
            dt_parallel_press = (t_pr_alt - t_next)

            if dt_parallel_press>timedelta(minutes = 0):
                if different_sides == True:
                    if new_side == "right":
                        dt_parallel_bands = t_pr_alt - t_next # EDIT
                        t_rechts = t_seriell - dt_US_to_bale
                        t_links = t_seriell - dt_klappe_zu_to_rdy # EDIT

                    if new_side == "left":
                        dt_parallel_bands = t_pr_alt - t_next # EDIT     
                        t_links = t_seriell - dt_US_to_bale
                        t_rechts = t_seriell - dt_klappe_zu_to_rdy # EDIT

                    dt_1 = dt_offset-dt_stop_to_press_to_2
                    if dt_1<timedelta(minutes = 0): 
                        dt_1 = timedelta(minutes = 0)
                    dt_parallel_total = np.asarray([dt_parallel_press, dt_parallel_bands, dt_1]).min()

                else:  #same side or non-parallel                       
                    dt_parallel_total = np.asarray([dt_parallel_press, dt_offset - dt_stop_to_press_to_2]).min()
            else:
                dt_parallel_total = timedelta(minutes =0)

            # seriell or on same side
            if ((dt_parallel_press<=timedelta(minutes = 0)) or (different_sides == False)):     
                if new_side == "left":
                    t_rechts = t_seriell - dt_klappe_zu_to_rdy # max(0, ...)
                    t_links = t_seriell - dt_US_to_bale
                else:
                    t_links = t_seriell - dt_klappe_zu_to_rdy
                    t_rechts = t_seriell - dt_US_to_bale

            # The maximum time to possibly save is t_parallel_press or the amount of time the bands can run parrallely
            t_ges = t_seriell - dt_parallel_total
            t_links = t_links - dt_parallel_total # new lefttime depends on time the process drives in parallel
            t_rechts = t_rechts - dt_parallel_total # new lefttime depends on time the process drives in parallel

            return t_ges, t_links, t_rechts

