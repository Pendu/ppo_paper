import numpy as np
import pandas as pd

# Read parameters
params = pd.read_csv('configs/RW_param_all_bunkers_orig.csv').set_index('bunker_id')
params_step_1 = pd.read_csv('configs/RW_param_all_bunkers_samemu_zerosigma.csv').set_index('bunker_id')

def future_volume_v(bunker_ids, volumes, timestep, CL_step):
    """Applies a random walk to all bunkers at once with a given timestep duration

    Parameters
    ----------
    bunker_ids : list
        List of bunker IDs
    volumes : np.array
        The current bunker volumes as a numpy vector
    timestep : int
        The length of an environment step in seconds
    CL_step : float
        The CL step value

    Returns
    -------
    np.array
        Vector of the new bunker volumes after the random walk
    """
    CL_step = CL_step
    enabled_bunkers = bunker_ids
    if CL_step in [1, 1.5, 2, 3]:
        mus = np.array([params_step_1.loc[bunker]['mu'] for bunker in enabled_bunkers])
        sigmas = np.array([params_step_1.loc[bunker]['sigma'] for bunker in enabled_bunkers])
    elif CL_step in [4,5, 6,7]:
        mus = np.array([params.loc[bunker]['mu'] for bunker in enabled_bunkers])
        sigmas = np.array([params.loc[bunker]['sigma'] for bunker in enabled_bunkers])
    
    rw_mat = np.random.normal(loc=mus, scale=sigmas, size=(timestep, len(volumes)))

    return np.clip(volumes + rw_mat.sum(axis=0), a_min=0, a_max=None)