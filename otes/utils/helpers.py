"""Helper functions for population and fitness manipulation"""

import numpy as np


def TransMat(pop):
    """
    Transform population to numpy arrays of decision variables and objectives.
    
    Parameters
    ----------
    pop : Population
        Pymoo population object
        
    Returns
    -------
    solution : np.ndarray
        Decision variables of shape (n_individuals, n_vars)
    fitness : np.ndarray
        Objective values of shape (n_individuals, n_objs)
    """
    solution = [solution_.get("X") for solution_ in pop]
    fitness = [solution_.get("F") for solution_ in pop]
    return np.array(solution), np.array(fitness)

