import math
import numpy as np

min_reward = -1e-1


def gaussian_reward_c1_10(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """
    r = 0.

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p = 19.84

        # Height of the peak(s)
        a = 1.

        # Width of the peak(s)
        c = 2.

        if use_minr:
            r = (a - min_reward) * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c)) + min_reward
        else:
            r = (a) * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c))

    return r


def gaussian_reward_c1_20(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p = 26.75  # 15 #5 #26.71

        # Height of the peak(s)
        a = 1

        # Width of the peak(s)
        c = 2

        if use_minr:
            r = (a - min_reward) * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c)) + min_reward
        else:
            r = a * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c))

    return r


def gaussian_reward_c1_30(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p1 = 26.52
        p2 = 17.68

        # Height of the peak(s)
        a1 = 1
        a2 = 0.3  # 0.5  # Original: 0.952

        # Width of the peak(s)
        c1 = 2.5
        c2 = 0.5  # 1

        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward
        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2)) \

    return r


def gaussian_reward_c1_40(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p1 = 8.34 #25.14
        p2 = 16.79
        p3 = 8.34

        # Height of the peak(s)
        a1 = 1
        a2 = 0.5
        a3 = 0.1

        # Width of the peak(s)
        c1 = 2.5
        c2 = 0.5
        c3 = 0.25

        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward

        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2)) \
                + a3 * np.exp(-(current_volume - p3) * (current_volume - p3) / (2. * c3 * c3))

    return r


def gaussian_reward_c1_50(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p1 = 31.53
        p2 = 27.11
        p3 = 22.59
        p4 = 18.07
        p5 = 13.55
        p6 = 9.03
        p7 = 4.51

        # Height of the peak(s)
        a1 = 1.
        a2 = 0.5
        a3 = 0.25
        a4 = 0.125
        a5 = 0.0625
        a6 = 0.03125
        a7 = 0.015625

        # Width of the peak(s)
        c1 = 1
        c2 = 0.5
        c3 = 0.5
        c4 = 0.5
        c5 = 0.5
        c6 = 0.5
        c7 = 0.5

        if use_minr:

            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) 

        else:

            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2)) \
                + a3 * np.exp(-(current_volume - p3) * (current_volume - p3) / (2. * c3 * c3)) \
                + a4 * np.exp(-(current_volume - p4) * (current_volume - p4) / (2. * c4 * c4)) \
                + a5 * np.exp(-(current_volume - p5) * (current_volume - p5) / (2. * c5 * c5)) \
                + a6 * np.exp(-(current_volume - p6) * (current_volume - p6) / (2. * c6 * c6)) \
                + a7 * np.exp(-(current_volume - p7) * (current_volume - p7) / (2. * c7 * c7))

    return r


def gaussian_reward_c1_60(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p1 = 14.34 #28.78
        p2 = 21.61
        p3 = 14.34

        # Height of the peak(s)
        a1 = 1
        a2 = 0.4
        a3 = 0.2

        # Width of the peak(s)
        c1 = 2  # 2.5
        c2 = 0.5
        c3 = 0.25

        if use_minr:

            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward

        else:
            r = (a1) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + (a2) * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2)) \
                + (a3) * np.exp(-(current_volume - p3) * (current_volume - p3) / (2. * c3 * c3)) \

    return r


def gaussian_reward_c1_70(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p1 = 25.93
        p2 = 17.28
        p3 = 8.64

        # Height of the peak(s)
        a1 = 1
        a2 = 0.25  # 0.3  # 0.5
        a3 = 0.1  # 0.1  # 0.25

        # Width of the peak(s)
        c1 = 2.5
        c2 = 0.5  # 1
        c3 = 0.25  # 0.5

        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward

        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2)) \
                + a3 * np.exp(-(current_volume - p3) * (current_volume - p3) / (2. * c3 * c3))

    return r


def gaussian_reward_c1_80(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p1 = 24.75
        p2 = 12.37

        # Height of the peak(s)
        a1 = 1
        a2 = 0.3  # 0.5

        # Width of the peak(s)
        c1 = 2
        c2 = 0.5  # 1

        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward

        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2))

    return r


def gaussian_reward_c2_10(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """
    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p = 27.39  # 15 #5 #26.71

        # Height of the peak(s)
        a = 1

        # Width of the peak(s)
        c = 2

        if use_minr:
            r = (a - min_reward) * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c)) + min_reward
        else:
            r = a * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c))

    return r


def gaussian_reward_c2_20(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p = 32  # 15 #5 #26.71

        # Height of the peak(s)
        a = 1

        # Width of the peak(s)
        c = 2

        if use_minr:
            r = (a - min_reward) * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c)) + min_reward
        else:
            r = a * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c))

    return r


def gaussian_reward_c2_40(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """
        
    r = 0
    
    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward
        
        p1 = 25.77
        
        a1 = 1
        
        c1 = 2
        
        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward
        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2)) \
                + a3 * np.exp(-(current_volume - p3) * (current_volume - p3) / (2. * c3 * c3)) \
                + a4 * np.exp(-(current_volume - p4) * (current_volume - p4) / (2. * c4 * c4)) \
                + a5 * np.exp(-(current_volume - p5) * (current_volume - p5) / (2. * c5 * c5))
    
    return r

def gaussian_reward_c2_50(action, current_volume, press_is_free, use_minr):
    
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """
        
    r = 0
    
    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward
        
        p1 = 32.23
        
        a1 = 1
        
        c1 = 2
        
        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward
        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2)) \
                + a3 * np.exp(-(current_volume - p3) * (current_volume - p3) / (2. * c3 * c3)) \
                + a4 * np.exp(-(current_volume - p4) * (current_volume - p4) / (2. * c4 * c4)) \
                + a5 * np.exp(-(current_volume - p5) * (current_volume - p5) / (2. * c5 * c5))
    
    return r


def gaussian_reward_c2_60(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p1 = 12.6 #25.77
        p2 = 18.9
        p3 = 12.6

        # Height of the peak(s)
        a1 = 1
        a2 = 0.25  # 0.3  # 0.5
        a3 = 0.1  # 0.1  # 0.25

        # Width of the peak(s)
        c1 = 2.5
        c2 = 0.5  # 1
        c3 = 0.25  # 0.5

        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward

        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2)) \
                + a3 * np.exp(-(current_volume - p3) * (current_volume - p3) / (2. * c3 * c3))

    return r


def gaussian_reward_c2_70(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """
    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p1 = 17 #26.46
        p2 = 17

        # Height of the peak(s)
        a1 = 1
        a2 = 0.3  # 0.5

        # Width of the peak(s)
        c1 = 2
        c2 = 0.5  # 1

        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward

        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2))

    return r


def gaussian_reward_c2_80(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """

    r = 0

    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward

        # Peak(s) corresponding to optimal volume(s) to empty at
        p = 28.75  # 15 #5 #26.71

        # Height of the peak(s)
        a = 1

        # Width of the peak(s)
        c = 2

        if use_minr:
            r = (a - min_reward) * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c)) + min_reward
        else:
            r = a * np.exp(-(current_volume - p) * (current_volume - p) / (2. * c * c))

    return r

def gaussian_reward_c2_90(action, current_volume, press_is_free, use_minr):
    """
    Gaussian reward function. Maximal reward values are achieved when emptying at the peak(s) derived empirically
    from the original penalty function. A negative reward is returned when emptying is not possible
    (press is not free) or when trying to empty an empty bunker. 0 is returned if nothing is done (action 0).

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param use_minr: True if the minimum reward value should be used. False otherwise.
    :return: Reward value between min_reward and 1
    """
    
    r = 0
    
    if action > 0:
        if current_volume == 0. or not press_is_free:
            return min_reward
        
        p1 = 28.79
        
        a1 = 1
                
        c1 = 2
        
        if use_minr:
            r = (a1 - min_reward) * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + min_reward
        else:
            r = a1 * np.exp(-(current_volume - p1) * (current_volume - p1) / (2. * c1 * c1)) \
                + a2 * np.exp(-(current_volume - p2) * (current_volume - p2) / (2. * c2 * c2))
    
    return r


def reward(action, current_volume, press_is_free, bunker_id, use_minr):
    """
    General reward function. Depending on bunker_id, calls corresponding Gaussian reward function.

    :param action: Action taken by the agent at time t
    :param current_volume: Bunker volume at time t
    :param press_is_free: True if press is free (emptying is possible) at time t. False otherwise
    :param bunker_id: ID of bunker on which the action is performed
    :param use_minr: Flag indicating whether to use the minimum reward value
    :return: Reward value between min_reward and 1
    """

    # TODO: Add remaining bunkers

    if bunker_id == "C1-10" or bunker_id == "C1-1":
        return gaussian_reward_c1_10(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C1-20" or bunker_id == "C1-2":
        return gaussian_reward_c1_20(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C1-30" or bunker_id == "C1-3":
        return gaussian_reward_c1_30(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C1-40" or bunker_id == "C1-4":
        return gaussian_reward_c1_40(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C1-50" or bunker_id == "C1-5":
        return gaussian_reward_c1_50(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C1-60" or bunker_id == "C1-6":
        return gaussian_reward_c1_60(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C1-70" or bunker_id == "C1-7":
        return gaussian_reward_c1_70(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C1-80" or bunker_id == "C1-8":
        return gaussian_reward_c1_80(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C2-10" or bunker_id == "C2-1":
        return gaussian_reward_c2_10(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C2-20" or bunker_id == "C2-2":
        return gaussian_reward_c2_20(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C2-40" or bunker_id == "C2-4":
        return gaussian_reward_c2_40(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C2-50" or bunker_id == "C2-5":
        return gaussian_reward_c2_50(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C2-60" or bunker_id == "C2-6":
        return gaussian_reward_c2_60(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C2-70" or bunker_id == "C2-7":
        return gaussian_reward_c2_70(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C2-80" or bunker_id == "C2-8":
        return gaussian_reward_c2_80(action, current_volume, press_is_free, use_minr)
    elif bunker_id == "C2-90" or bunker_id == "C2-9":
        return gaussian_reward_c2_90(action, current_volume, press_is_free, use_minr)