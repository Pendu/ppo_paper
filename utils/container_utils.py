def get_color_code():
    """
    Returns a dictionary containing color codes for each container in the environment.

    Returns:
    ----------
    color_code : dict
        A dictionary containing color codes for each container in the environment. Keys are container names and values are color codes represented as hexadecimal strings.

    Note: The user should make sure to define a color for each container contained in the environment.

    """

    color_code = {
        "C1-10": "#872657",  # raspberry
        "C1-20": "#0000FF",  # blue
        "C1-30": "#FFA500",  # orange
        "C1-40": "#008000",  # green
        "C1-50": "#B0E0E6",  # powderblue
        "C1-60": "#FF00FF",  # fuchsia
        "C1-70": "#800080",  # purple
        "C1-80": "#FF4500",  # orangered
        "C2-10": "#DB7093",  # palevioletred
        "C2-20": "#FF8C69",  # salmon1
        "C2-40": "#27408B",  # royalblue4
        "C2-50": "#54FF9F",  # seagreen1
        "C2-60": "#FF3E96",  # violetred1
        "C2-70": "#FFD700",  # gold1
        "C2-80": "#7FFF00",  # chartreuse1
        "C2-90": "#D2691E",  # chocolate
    }
    return color_code

def get_peaks(volumes, actions, bunker):
    """ Find local maxima in a list of volumes based on action 1 and return their indices.
    
    Args:
        volumes (list): List of volumes.
        actions (list): List of actions.
        bunker (int): Bunker value.
        
    Returns:
        list: List of indices of local maxima.
    """
        
    action_indices = [i for i, value in enumerate(actions) if value == bunker + 1]
    
    zero_indices_volumes = [i for i, value in enumerate(volumes) if value == 0.0]
    
    zero_indices_volumes_overlap_action = [i for i in zero_indices_volumes if actions[i] == bunker + 1]
    
    filtered_indices = [i for i in zero_indices_volumes_overlap_action if volumes[i-1] and volumes[i-2] > 0]

    
    return filtered_indices

def bunker_ids(bunker_indices):
    """
    :param args.bunkers: the indicies of the bunkers used in the environment
    :return: list of actual bunker names
    """

    bunker_ids_all = [
        "C1-10",
        "C1-20",
        "C1-30",
        "C1-40",
        "C1-50",
        "C1-60",
        "C1-70",
        "C1-80",
        "C2-10",
        "C2-20",
        "C2-40",
        "C2-50",
        "C2-60",
        "C2-70",
        "C2-80",
        "C2-90"
    ]

    bunker_ids = []
    for i in bunker_indices:
        bunker_ids.append(bunker_ids_all[i])

    f"The following bunkers have been chosen for the environment {bunker_ids}"

    return bunker_ids
