"""Optimal transport-based knowledge transfer methods"""

import ot
import numpy as np
import torch
import torch.nn as nn
from torchmin import Minimizer
from otes.utils.ndsort import environment_selection, fast_non_dominated_sort


class MappingDecision(nn.Module):
    """Linear mapping network for decision variable transfer."""
    
    def __init__(self, input_size, output_size):
        super(MappingDecision, self).__init__()
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.lin(x)


class MappingDecisionNonlinear(nn.Module):
    """Nonlinear mapping network for decision variable transfer."""
    
    def __init__(self, input_size, output_size):
        super(MappingDecisionNonlinear, self).__init__()
        self.lin = nn.Linear(input_size, output_size)
        self.lin1 = nn.Linear(output_size, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        y = self.lin(x)
        y = self.relu(y)
        y = self.lin1(y)
        return y


def train_mapping(model, xs, xt, G, print_loss=False, opt='l-bfgs'):
    """
    Train the mapping model using optimal transport plan.
    
    Parameters
    ----------
    model : nn.Module
        Mapping network
    xs : np.ndarray
        Source decision variables
    xt : np.ndarray
        Target decision variables
    G : np.ndarray
        Optimal transport plan
    print_loss : bool
        Whether to print training loss
    opt : str
        Optimizer type ('l-bfgs' or 'adam')
        
    Returns
    -------
    model : nn.Module
        Trained mapping network
    """
    ms = xs.shape[0]
    mt = xt.shape[0]

    xs_train = torch.tensor(xs, dtype=torch.float)
    xt_train = torch.tensor(xt, dtype=torch.float)
    G = torch.tensor(G, dtype=torch.float)
    xt_train_repeated = xt_train.unsqueeze(0).repeat([ms, 1, 1])
    G = G.unsqueeze(2)
    
    if opt == 'l-bfgs':
        optimizer = Minimizer(model.parameters(),
                            method='l-bfgs',
                            tol=1e-10,
                            max_iter=500,
                            disp=0)    
        def closure():
            optimizer.zero_grad()
            xs_trans = model(xs_train)
            xs_trans_repeated = xs_trans.unsqueeze(1).repeat([1, mt, 1])
            loss_mse = torch.sum(G * (xs_trans_repeated - xt_train_repeated)**2)
            if print_loss:
                print(loss_mse)
            return loss_mse
        optimizer.step(closure=closure)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for i in range(500):
            optimizer.zero_grad()
            xs_trans = model(xs_train)
            xs_trans_repeated = xs_trans.unsqueeze(1).repeat([1, mt, 1])
            loss_mse = torch.sum(G * (xs_trans_repeated - xt_train_repeated)**2)
            loss_mse.backward()
            optimizer.step()
    return model


def optimal_transport(xs, ys, xt, yt, model_style='linear', reg=0.2):
    """
    Perform optimal transport-based knowledge transfer.
    
    Parameters
    ----------
    xs : np.ndarray
        Source decision variables
    ys : np.ndarray
        Source objective values
    xt : np.ndarray
        Target decision variables
    yt : np.ndarray
        Target objective values
    model_style : str
        Mapping model style ('linear' or 'nonlinear')
    reg : float
        Regularization parameter for Sinkhorn algorithm
        
    Returns
    -------
    map_x : nn.Module
        Trained mapping network
    """
    # Normalization
    ys = (ys - np.min(ys, axis=0, keepdims=True)) / \
         (np.max(ys, axis=0, keepdims=True) - np.min(ys, axis=0, keepdims=True) + 1e-10)
    yt = (yt - np.min(yt, axis=0, keepdims=True)) / \
         (np.max(yt, axis=0, keepdims=True) - np.min(yt, axis=0, keepdims=True) + 1e-10)

    # Record the size of the source dataset and the target dataset
    ns, ms = ys.shape
    nt, mt = yt.shape
    ds = xs.shape[1]
    dt = xt.shape[1]
    dmin = np.minimum(ds, dt)

    # Do non-dominated sorting for source and target data
    _, p_rank_source, _ = fast_non_dominated_sort(ys, ys.shape[0])
    _, p_rank_target, _ = fast_non_dominated_sort(yt, yt.shape[0])

    # Calculate distance matrices
    p_rank_source_norm = p_rank_source
    p_rank_target_norm = p_rank_target
    M_Dec = np.sqrt(ot.dist(xs[:, :dmin], xt[:, :dmin]))
    M_Obj = np.sqrt(ot.dist(ys, yt))
    M_ND = np.sqrt(ot.dist(p_rank_source_norm.reshape(-1, 1), 
                           p_rank_target_norm.reshape(-1, 1)))
    M = M_Dec + M_Obj + 0 * M_ND

    # Calculate the optimal transport plan
    mu_s = np.ones(ns) / ns
    mu_t = np.ones(nt) / nt
    G = ot.sinkhorn(mu_s, mu_t, M, reg=reg)

    # Train decision mapping
    if model_style == 'linear':
        map_x = MappingDecision(ds, dt)
        map_x = train_mapping(map_x, xs, xt, G, print_loss=False)
    else:
        map_x = MappingDecisionNonlinear(ds, dt)
        map_x = train_mapping(map_x, xs, xt, G, print_loss=False, opt='adam')
        
    return map_x


class MappingAutoEncoder(nn.Module):
    """Autoencoder mapping network for nonlinear transfer."""
    
    def __init__(self, input_size, output_size):
        super(MappingAutoEncoder, self).__init__()
        self.lin = nn.Linear(input_size, output_size)
        self.lin1 = nn.Linear(output_size, output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        y = self.lin(x)
        return y


def train_autoencoder(model, xs, xt, print_loss=False, opt='l-bfgs'):
    """Train autoencoder model."""
    xs_train = torch.tensor(xs, dtype=torch.float)
    xt_train = torch.tensor(xt, dtype=torch.float)
    
    if opt == 'l-bfgs':
        optimizer = Minimizer(model.parameters(),
                            method='l-bfgs',
                            tol=1e-10,
                            max_iter=500,
                            disp=0)    
        def closure():
            optimizer.zero_grad()
            xs_trans = model(xs_train)
            loss_mse = torch.mean((xs_trans - xt_train)**2)
            if print_loss:
                print(loss_mse)
            return loss_mse
        optimizer.step(closure=closure)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for i in range(500):
            optimizer.zero_grad()
            xs_trans = model(xs_train)
            loss_mse = torch.mean((xs_trans - xt_train)**2)
            loss_mse.backward()
            optimizer.step()
    return model


def autoencoding_linear(xs, ys, xt, yt):
    """
    Linear autoencoding for knowledge transfer.
    
    Parameters
    ----------
    xs : np.ndarray
        Source decision variables
    ys : np.ndarray
        Source objective values
    xt : np.ndarray
        Target decision variables
    yt : np.ndarray
        Target objective values
        
    Returns
    -------
    M : np.ndarray
        Linear transformation matrix
    """
    n_min = np.minimum(ys.shape[0], yt.shape[0])
    xs_nd, ys_nd, idx_s = environment_selection(xs, ys, n_min)
    xt_nd, yt_nd, idx_t = environment_selection(xt, yt, n_min)
    xs_nd_add = np.concatenate((xs_nd, np.ones([xs_nd.shape[0], 1])), axis=1)
    xt_nd_add = np.concatenate((xt_nd, np.ones([xs_nd.shape[0], 1])), axis=1)
    M = np.dot(np.dot(xt_nd_add.T, xs_nd_add), 
               np.linalg.inv(np.dot(xs_nd_add.T, xs_nd_add))) 
    return M


def autoencoding_nonlinear(xs, ys, xt, yt):
    """
    Nonlinear autoencoding for knowledge transfer.
    
    Parameters
    ----------
    xs : np.ndarray
        Source decision variables
    ys : np.ndarray
        Source objective values
    xt : np.ndarray
        Target decision variables
    yt : np.ndarray
        Target objective values
        
    Returns
    -------
    map_x : nn.Module
        Trained autoencoder mapping network
    """
    n_min = np.minimum(ys.shape[0], yt.shape[0])
    xs_nd, ys_nd, idx_s = environment_selection(xs, ys, n_min)
    xt_nd, yt_nd, idx_t = environment_selection(xt, yt, n_min)
    map_x = MappingAutoEncoder(xs.shape[1], xt.shape[1])
    map_x = train_autoencoder(map_x, xs_nd, xt_nd, print_loss=False, opt='adam')
    return map_x

