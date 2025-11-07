"""Non-dominated sorting and crowding distance utilities"""

import numpy as np


def fast_non_dominated_sort(Y, N):
    """
    Fast non-dominated sorting algorithm.
    
    Parameters
    ----------
    Y : np.ndarray
        Objective values of shape (n_individuals, n_objs)
    N : int
        Number of individuals to select
        
    Returns
    -------
    F : list
        List of fronts, where each front is a list of individual indices
    p_rank : np.ndarray
        Rank of each individual
    len_fi : list
        Length of each front
    """
    m, n = Y.shape
    n_p = np.zeros(m)
    p_rank = np.inf * np.ones(m)
    S_all = []
    F = []
    F_i = []
    len_fi = []
    
    for i in range(m):
        S_p = []
        for j in range(m):
            flag_dom = (Y[i] < Y[j])
            flag_domed = (Y[i] > Y[j])
            if np.sum(flag_domed) == 0:
                S_p.append(j)
            elif np.sum(flag_dom) == 0:
                n_p[i] = n_p[i] + 1
        if n_p[i] == 0:
            p_rank[i] = 1
            F_i.append(i)
        S_all.append(S_p)
        
    F.append(F_i)
    len_fi.append(len(F_i))
    t = 1
    
    while np.sum(np.array(len_fi)) < N:
        Q = []
        for i in F_i:
            for j in S_all[i]:
                n_p[j] = n_p[j] - 1
                if n_p[j] == 0:
                    p_rank[j] = t + 1
                    Q.append(j)
        t = t + 1
        F_i = Q
        F.append(F_i)
        len_fi.append(len(F_i))
        
    return F, p_rank, len_fi


def crowding_distance(Y):
    """
    Calculate crowding distance for diversity preservation.
    
    Parameters
    ----------
    Y : np.ndarray
        Objective values of shape (n_individuals, n_objs)
        
    Returns
    -------
    I_distance : np.ndarray
        Crowding distance for each individual
    """
    l, m = Y.shape
    I_distance = np.zeros(l)
    y_max = np.max(Y, axis=0)
    y_min = np.min(Y, axis=0)
    
    for i in range(m):
        idx_sort = np.argsort(Y[:, i])
        I_distance[idx_sort[0]] = np.inf
        I_distance[idx_sort[l-1]] = np.inf
        for j in range(1, l-1):
            I_distance[idx_sort[j]] = I_distance[idx_sort[j]] + \
                np.abs(Y[idx_sort[j-1], i] - Y[idx_sort[j+1], i]) / (y_max[i] - y_min[i])
                
    return I_distance


def environment_selection(X, Y, N):
    """
    Environment selection using non-dominated sorting and crowding distance.
    
    Parameters
    ----------
    X : np.ndarray
        Decision variables of shape (n_individuals, n_vars)
    Y : np.ndarray
        Objective values of shape (n_individuals, n_objs)
    N : int
        Number of individuals to select
        
    Returns
    -------
    X_selected : np.ndarray
        Selected decision variables
    Y_selected : np.ndarray
        Selected objective values
    idx_next : list
        Indices of selected individuals
    """
    F, p_rank, len_fi = fast_non_dominated_sort(Y, N)
    idx_next = []
    len_ = 0
    i = -1
    
    for i in range(len(F)-1):
        idx_next.extend(F[i])
        len_ = len_ + len_fi[i]
        
    Y_sub = np.array(Y[F[i+1]])
    I_distance = crowding_distance(Y_sub)
    idx_sort = np.argsort(-I_distance)
    
    for j in range(N - len(idx_next)):
        idx_next.append(F[i+1][idx_sort[j]])
    
    return X[idx_next], Y[idx_next], idx_next

