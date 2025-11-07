"""Modified DTLZ and Inverted DTLZ test problems for multi-objective optimization"""

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


class mDTLZ1:
    """
    Modified DTLZ1 test problem.
    
    Parameters
    ----------
    obj_num : int, default=2
        Number of objectives
    n_var : int, default=10
        Number of decision variables
    delta1 : float, default=1
        Parameter for position-related variables
    delta2 : float, default=0
        Parameter for distance-related variables
    """
    
    def __init__(self, obj_num=2, n_var=10, delta1=1, delta2=0, w=None):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([np.zeros(n_var), np.ones(n_var)])
        self.delta1 = delta1
        self.delta2 = delta2      
        if self.obj_num > 3:
            self.w = get_reference_directions("energy", self.obj_num, 200, seed=1)
        self.pareto_f = self.cal_pareto_f()
        self.hv_ref_point = np.ones(obj_num) * 1.1
        
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim > 3:
            raise Exception("The dimension of the input array should be smaller than 3.")
        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:, 0:M-1]**self.delta1
        x2 = x[:, M-1:self.dim]
        
        triu_mat = np.fliplr(np.triu(np.ones([M-1, M-1]))).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        x1_repeat = x1.reshape(pop_size, 1, -1).repeat(M-1, axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(x1_repeat*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([pop_size, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        f_diversity2 = np.sum((1-x1_repeat)*inv_identity, axis=2)
        f_diversity2 = np.concatenate((np.ones([pop_size, 1]), f_diversity2), axis=1)
        g = ((self.dim - M + 1)*1 + np.sum((x[:, M-1:] - 0.5 - self.delta2)**2 - 
             1*np.cos(2*np.pi*(x[:, M-1:] - 0.5 - self.delta2)), axis=1))
        g = g.reshape(pop_size, 1)

        f = 0.5*f_diversity1*f_diversity2*(1+g)
        return f

    def cal_pareto_f(self):
        if self.obj_num <= 3:
            mc = 10000
            x = np.random.rand(mc, self.obj_num - 1)
            x = x.reshape(mc, 1, -1).repeat(self.obj_num-1, axis=1)
            triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1, self.obj_num-1]))).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
            tril_mat = 1-triu_mat
            f_diversity1 = np.prod(x*triu_mat+tril_mat, axis=2)
            f_diversity1 = np.concatenate((f_diversity1, np.ones([mc, 1])), axis=1)
            inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
            f_diversity2 = np.sum((1-x)*inv_identity, axis=2)
            f_diversity2 = np.concatenate((np.ones([mc, 1]), f_diversity2), axis=1)
            f = 0.5*f_diversity1*f_diversity2
        else:
            f = 0.5*self.w / np.sum(self.w, axis=1, keepdims=True)
        return f
    
    def IGD(self, x):
        """Calculate Inverted Generational Distance."""
        m, n = x.shape
        p, q = self.pareto_f.shape
        x = x.reshape(m, 1, n)
        x = x.repeat(p, axis=1)
        pareto_f = self.pareto_f.reshape(1, p, q)
        pareto_f = pareto_f.repeat(m, axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2, axis=2))
        min_distance = np.min(distance, axis=0)
        igd = np.mean(min_distance)
        return igd
    
    def HV(self, x):
        """Calculate Hypervolume."""
        from pymoo.indicators.hv import HV as HypervolumeIndicator
        hv = HypervolumeIndicator(ref_point=self.hv_ref_point)
        return hv(x)


class mDTLZ2:
    """
    Modified DTLZ2 test problem.
    
    Parameters
    ----------
    obj_num : int, default=2
        Number of objectives
    n_var : int, default=10
        Number of decision variables
    delta1 : float, default=1
        Parameter for position-related variables
    delta2 : float, default=0
        Parameter for distance-related variables
    """
    
    def __init__(self, obj_num=2, n_var=10, delta1=1, delta2=0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([np.zeros(n_var), np.ones(n_var)])
        self.delta1 = delta1
        self.delta2 = delta2     
        if self.obj_num > 3:
            self.w = get_reference_directions("energy", self.obj_num, 200, seed=1)
        self.pareto_f = self.cal_pareto_f()
        self.hv_ref_point = np.ones(obj_num) * 1.1
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            raise Exception("The dimension of the input array should be smaller than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:, 0:M-1]**self.delta1
        x2 = x[:, M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1, M-1]))).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        x1_repeat = x1.reshape(pop_size, 1, -1).repeat(M-1, axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([pop_size, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size, 1]), f_diversity2), axis=1)
        g = 1*np.sum((x[:, M-1:] - 0.5 - self.delta2)**2, axis=1)
        g = g.reshape(pop_size, 1)

        f = f_diversity1*f_diversity2*(1+g)
        return f
    
    def cal_pareto_f(self):
        if self.obj_num <= 3:
            mc = 10000
            x = np.random.rand(mc, self.obj_num - 1)
            x = x.reshape(mc, 1, -1).repeat(self.obj_num-1, axis=1)
            triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1, self.obj_num-1]))).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
            tril_mat = 1-triu_mat
            f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat, axis=2)
            f_diversity1 = np.concatenate((f_diversity1, np.ones([mc, 1])), axis=1)
            inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
            f_diversity2 = np.sin(np.sum(x*inv_identity, axis=2)*np.pi/2)
            f_diversity2 = np.concatenate((np.ones([mc, 1]), f_diversity2), axis=1)
            f = f_diversity1*f_diversity2
        else:
            f = self.w / np.sqrt(np.sum(self.w**2, axis=1, keepdims=True))
        return f

    def IGD(self, x):
        """Calculate Inverted Generational Distance."""
        m, n = x.shape
        p, q = self.pareto_f.shape
        x = x.reshape(m, 1, n)
        x = x.repeat(p, axis=1)
        pareto_f = self.pareto_f.reshape(1, p, q)
        pareto_f = pareto_f.repeat(m, axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2, axis=2))
        min_distance = np.min(distance, axis=0)
        igd = np.mean(min_distance)
        return igd
    
    def HV(self, x):
        """Calculate Hypervolume."""
        from pymoo.indicators.hv import HV as HypervolumeIndicator
        hv = HypervolumeIndicator(ref_point=self.hv_ref_point)
        return hv(x)


class mDTLZ3:
    """
    Modified DTLZ3 test problem.
    
    Parameters
    ----------
    obj_num : int, default=2
        Number of objectives
    n_var : int, default=10
        Number of decision variables
    delta1 : float, default=1
        Parameter for position-related variables
    delta2 : float, default=0
        Parameter for distance-related variables
    """
    
    def __init__(self, obj_num=2, n_var=10, delta1=1, delta2=0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([np.zeros(n_var), np.ones(n_var)])
        self.delta1 = delta1
        self.delta2 = delta2
        if self.obj_num > 3:
            self.w = get_reference_directions("energy", self.obj_num, 200, seed=1)
        self.pareto_f = self.cal_pareto_f()
        self.hv_ref_point = np.ones(obj_num) * 1.1
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            raise Exception("The dimension of the input array should be smaller than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:, 0:M-1]**self.delta1
        x2 = x[:, M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1, M-1]))).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        x1_repeat = x1.reshape(pop_size, 1, -1).repeat(M-1, axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([pop_size, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size, 1]), f_diversity2), axis=1)
        g = ((self.dim - M + 1)*0.1 + np.sum((x[:, M-1:] - 0.5 - self.delta2)**2 - 
             0.1*np.cos(2*np.pi*(x[:, M-1:] - 0.5 - self.delta2)), axis=1))
        g = g.reshape(pop_size, 1)

        f = f_diversity1*f_diversity2*(1+g)
        return f
    
    def cal_pareto_f(self):
        if self.obj_num <= 3:
            mc = 10000
            x = np.random.rand(mc, self.obj_num - 1)
            x = x.reshape(mc, 1, -1).repeat(self.obj_num-1, axis=1)
            triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1, self.obj_num-1]))).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
            tril_mat = 1-triu_mat
            f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat, axis=2)
            f_diversity1 = np.concatenate((f_diversity1, np.ones([mc, 1])), axis=1)
            inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
            f_diversity2 = np.sin(np.sum(x*inv_identity, axis=2)*np.pi/2)
            f_diversity2 = np.concatenate((np.ones([mc, 1]), f_diversity2), axis=1)
            f = f_diversity1*f_diversity2
        else:
            f = self.w / np.sqrt(np.sum(self.w**2, axis=1, keepdims=True))
        return f

    def IGD(self, x):
        """Calculate Inverted Generational Distance."""
        m, n = x.shape
        p, q = self.pareto_f.shape
        x = x.reshape(m, 1, n)
        x = x.repeat(p, axis=1)
        pareto_f = self.pareto_f.reshape(1, p, q)
        pareto_f = pareto_f.repeat(m, axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2, axis=2))
        min_distance = np.min(distance, axis=0)
        igd = np.mean(min_distance)
        return igd
    
    def HV(self, x):
        """Calculate Hypervolume."""
        from pymoo.indicators.hv import HV as HypervolumeIndicator
        hv = HypervolumeIndicator(ref_point=self.hv_ref_point)
        return hv(x)


class mDTLZ4:
    """
    Modified DTLZ4 test problem.
    
    Parameters
    ----------
    obj_num : int, default=2
        Number of objectives
    n_var : int, default=10
        Number of decision variables
    delta1 : float, default=1
        Parameter for position-related variables
    delta2 : float, default=0
        Parameter for distance-related variables
    """
    
    def __init__(self, obj_num=2, n_var=10, delta1=1, delta2=0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([np.zeros(n_var), np.ones(n_var)])
        self.delta1 = delta1
        self.delta2 = delta2
        if self.obj_num > 3:
            self.w = get_reference_directions("energy", self.obj_num, 200, seed=1)
        self.pareto_f = self.cal_pareto_f()
        self.hv_ref_point = np.ones(obj_num) * 1.1
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            raise Exception("The dimension of the input array should be smaller than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:, 0:M-1]**(2*self.delta1)
        x2 = x[:, M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1, M-1]))).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        x1_repeat = x1.reshape(pop_size, 1, -1).repeat(M-1, axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([pop_size, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size, 1]), f_diversity2), axis=1)
        g = np.sum((x[:, M-1:] - 0.5 - self.delta2)**2, axis=1)
        g = g.reshape(pop_size, 1)

        f = f_diversity1*f_diversity2*(1+g)
        return f
    
    def cal_pareto_f(self):
        if self.obj_num <= 3:
            mc = 10000
            x = np.random.rand(mc, self.obj_num - 1)
            x = x.reshape(mc, 1, -1).repeat(self.obj_num-1, axis=1)
            triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1, self.obj_num-1]))).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
            tril_mat = 1-triu_mat
            f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat, axis=2)
            f_diversity1 = np.concatenate((f_diversity1, np.ones([mc, 1])), axis=1)
            inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
            f_diversity2 = np.sin(np.sum(x*inv_identity, axis=2)*np.pi/2)
            f_diversity2 = np.concatenate((np.ones([mc, 1]), f_diversity2), axis=1)
            f = f_diversity1*f_diversity2
        else:
            f = self.w / np.sqrt(np.sum(self.w**2, axis=1, keepdims=True))
        return f

    def IGD(self, x):
        """Calculate Inverted Generational Distance."""
        m, n = x.shape
        p, q = self.pareto_f.shape
        x = x.reshape(m, 1, n)
        x = x.repeat(p, axis=1)
        pareto_f = self.pareto_f.reshape(1, p, q)
        pareto_f = pareto_f.repeat(m, axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2, axis=2))
        min_distance = np.min(distance, axis=0)
        igd = np.mean(min_distance)
        return igd
    
    def HV(self, x):
        """Calculate Hypervolume."""
        from pymoo.indicators.hv import HV as HypervolumeIndicator
        hv = HypervolumeIndicator(ref_point=self.hv_ref_point)
        return hv(x)


class InvDTLZ1:
    """
    Inverted DTLZ1 test problem.
    
    Parameters
    ----------
    obj_num : int, default=2
        Number of objectives
    n_var : int, default=10
        Number of decision variables
    delta1 : float, default=1
        Parameter for position-related variables
    delta2 : float, default=0
        Parameter for distance-related variables
    """
    
    def __init__(self, obj_num=2, n_var=10, delta1=1, delta2=0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([np.zeros(n_var), np.ones(n_var)])
        self.norm_for_hv = np.array([np.zeros(obj_num), 1000*np.ones(obj_num)])
        self.delta1 = delta1
        self.delta2 = delta2
        self.pareto_f = self.cal_pareto_f()
        self.hv_ref_point = np.ones(obj_num) * 0.1
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            raise Exception("The dimension of the input array should be smaller than 3.")
        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:, 0:M-1]**self.delta1
        x2 = x[:, M-1:self.dim]
        
        triu_mat = np.fliplr(np.triu(np.ones([M-1, M-1]))).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        x1_repeat = x1.reshape(pop_size, 1, -1).repeat(M-1, axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(x1_repeat*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([pop_size, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        f_diversity2 = np.sum((1-x1_repeat)*inv_identity, axis=2)
        f_diversity2 = np.concatenate((np.ones([pop_size, 1]), f_diversity2), axis=1)
        g = ((self.dim - M + 1)*1 + np.sum((x[:, M-1:] - 0.5 - self.delta2)**2 - 
             1*np.cos(2*np.pi*(x[:, M-1:] - 0.5 - self.delta2)), axis=1))
        g = g.reshape(pop_size, 1)

        f = -0.5*f_diversity1*f_diversity2*(1-g)
        return f
    
    def cal_pareto_f(self):
        M = self.obj_num
        x_ = 0.5*np.ones([1, self.dim - M + 1]) + self.delta2
        g_ = ((self.dim - M + 1)*1 + np.sum((x_ - 0.5 - self.delta2)**2 - 
              1*np.cos(2*np.pi*(x_ - 0.5 - self.delta2)), axis=1))
        mc = 10000
        x = np.random.rand(mc, self.obj_num - 1)
        x = x.reshape(mc, 1, -1).repeat(self.obj_num-1, axis=1)
        triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1, self.obj_num-1]))).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(x*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([mc, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
        f_diversity2 = np.sum((1-x)*inv_identity, axis=2)
        f_diversity2 = np.concatenate((np.ones([mc, 1]), f_diversity2), axis=1)
        f = -0.5*f_diversity1*f_diversity2*(1 - g_)
        return f

    def set_w(self, w):
        self.w = w

    def IGD(self, x):
        """Calculate Inverted Generational Distance."""
        m, n = x.shape
        p, q = self.pareto_f.shape
        x = x.reshape(m, 1, n)
        x = x.repeat(p, axis=1)
        pareto_f = self.pareto_f.reshape(1, p, q)
        pareto_f = pareto_f.repeat(m, axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2, axis=2))
        min_distance = np.min(distance, axis=0)
        igd = np.mean(min_distance)
        return igd
    
    def HV(self, x):
        """Calculate Hypervolume."""
        from pymoo.indicators.hv import HV as HypervolumeIndicator
        hv = HypervolumeIndicator(ref_point=self.hv_ref_point)
        return hv(x)


class InvDTLZ2:
    """
    Inverted DTLZ2 test problem.
    
    Parameters
    ----------
    obj_num : int, default=2
        Number of objectives
    n_var : int, default=12
        Number of decision variables
    delta1 : float, default=1
        Parameter for position-related variables
    delta2 : float, default=0
        Parameter for distance-related variables
    """
    
    def __init__(self, obj_num=2, n_var=12, delta1=1, delta2=0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([np.zeros(n_var), np.ones(n_var)])
        self.norm_for_hv = np.array([[0, 0], [2, 2]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.pareto_f = self.cal_pareto_f()
        self.hv_ref_point = np.ones(obj_num) * 0.1
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            raise Exception("The dimension of the input array should be smaller than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:, 0:M-1]**self.delta1
        x2 = x[:, M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1, M-1]))).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        x1_repeat = x1.reshape(pop_size, 1, -1).repeat(M-1, axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([pop_size, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size, 1]), f_diversity2), axis=1)
        g = np.sum((x[:, M-1:] - 0.5 - self.delta2)**2, axis=1)
        g = g.reshape(pop_size, 1)

        f = -f_diversity1*f_diversity2*(1-g)
        return f
    
    def cal_pareto_f(self):
        M = self.obj_num
        x_ = 0.5*np.ones([1, self.dim - M + 1]) + self.delta2
        g_ = np.sum((x_ - 0.5 - self.delta2)**2, axis=1)
        mc = 10000
        x = np.random.rand(mc, self.obj_num - 1)
        x = x.reshape(mc, 1, -1).repeat(self.obj_num-1, axis=1)
        triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1, self.obj_num-1]))).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([mc, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
        f_diversity2 = np.sin(np.sum(x*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([mc, 1]), f_diversity2), axis=1)
        f = -f_diversity1*f_diversity2*(1 - g_)
        return f

    def set_w(self, w):
        self.w = w

    def IGD(self, x):
        """Calculate Inverted Generational Distance."""
        m, n = x.shape
        p, q = self.pareto_f.shape
        x = x.reshape(m, 1, n)
        x = x.repeat(p, axis=1)
        pareto_f = self.pareto_f.reshape(1, p, q)
        pareto_f = pareto_f.repeat(m, axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2, axis=2))
        min_distance = np.min(distance, axis=0)
        igd = np.mean(min_distance)
        return igd
    
    def HV(self, x):
        """Calculate Hypervolume."""
        from pymoo.indicators.hv import HV as HypervolumeIndicator
        hv = HypervolumeIndicator(ref_point=self.hv_ref_point)
        return hv(x)


class InvDTLZ3:
    """
    Inverted DTLZ3 test problem.
    
    Parameters
    ----------
    obj_num : int, default=2
        Number of objectives
    n_var : int, default=10
        Number of decision variables
    delta1 : float, default=1
        Parameter for position-related variables
    delta2 : float, default=0
        Parameter for distance-related variables
    """
    
    def __init__(self, obj_num=2, n_var=10, delta1=1, delta2=0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([np.zeros(n_var), np.ones(n_var)])
        self.norm_for_hv = np.array([[0, 0], [2000, 2000]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.pareto_f = self.cal_pareto_f()
        self.hv_ref_point = np.ones(obj_num) * 0.1
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            raise Exception("The dimension of the input array should be smaller than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:, 0:M-1]**(self.delta1)
        x2 = x[:, M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1, M-1]))).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        x1_repeat = x1.reshape(pop_size, 1, -1).repeat(M-1, axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([pop_size, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size, 1]), f_diversity2), axis=1)
        g = ((self.dim - M + 1)*0.1 + np.sum((x[:, M-1:] - 0.5 - self.delta2)**2 - 
             0.1*np.cos(2*np.pi*(x[:, M-1:] - 0.5 - self.delta2)), axis=1))
        g = g.reshape(pop_size, 1)

        f = -f_diversity1*f_diversity2*(1-g)
        return f
    
    def cal_pareto_f(self):
        M = self.obj_num
        x_ = 0.5*np.ones([1, self.dim - M + 1]) + self.delta2
        g_ = ((self.dim - M + 1)*0.1 + np.sum((x_ - 0.5 - self.delta2)**2 - 
              0.1*np.cos(2*np.pi*(x_ - 0.5 - self.delta2)), axis=1))
        mc = 10000
        x = np.random.rand(mc, self.obj_num - 1)
        x = x.reshape(mc, 1, -1).repeat(self.obj_num-1, axis=1)
        triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1, self.obj_num-1]))).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([mc, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
        f_diversity2 = np.sin(np.sum(x*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([mc, 1]), f_diversity2), axis=1)
        f = -f_diversity1*f_diversity2*(1 - g_)
        return f

    def set_w(self, w):
        self.w = w

    def IGD(self, x):
        """Calculate Inverted Generational Distance."""
        m, n = x.shape
        p, q = self.pareto_f.shape
        x = x.reshape(m, 1, n)
        x = x.repeat(p, axis=1)
        pareto_f = self.pareto_f.reshape(1, p, q)
        pareto_f = pareto_f.repeat(m, axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2, axis=2))
        min_distance = np.min(distance, axis=0)
        igd = np.mean(min_distance)
        return igd
    
    def HV(self, x):
        """Calculate Hypervolume."""
        from pymoo.indicators.hv import HV as HypervolumeIndicator
        hv = HypervolumeIndicator(ref_point=self.hv_ref_point)
        return hv(x)


class InvDTLZ4:
    """
    Inverted DTLZ4 test problem.
    
    Parameters
    ----------
    obj_num : int, default=2
        Number of objectives
    n_var : int, default=12
        Number of decision variables
    delta1 : float, default=1
        Parameter for position-related variables
    delta2 : float, default=0
        Parameter for distance-related variables
    """
    
    def __init__(self, obj_num=2, n_var=12, delta1=1, delta2=0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([np.zeros(n_var), np.ones(n_var)])
        self.norm_for_hv = np.array([[0, 0], [2, 2]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.pareto_f = self.cal_pareto_f()
        self.hv_ref_point = np.ones(obj_num) * 0.1
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            raise Exception("The dimension of the input array should be smaller than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:, 0:M-1]**(2*self.delta1)
        x2 = x[:, M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1, M-1]))).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        x1_repeat = x1.reshape(pop_size, 1, -1).repeat(M-1, axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([pop_size, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1, M-1, M-1).repeat(pop_size, axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size, 1]), f_diversity2), axis=1)
        g = np.sum((x[:, M-1:] - 0.5 - self.delta2)**2, axis=1)
        g = g.reshape(pop_size, 1)

        f = -f_diversity1*f_diversity2*(1-g)
        return f
    
    def cal_pareto_f(self):
        M = self.obj_num
        x_ = 0.5*np.ones([1, self.dim - M + 1]) + self.delta2
        g_ = np.sum((x_ - 0.5 - self.delta2)**2, axis=1)
        mc = 10000
        x = np.random.rand(mc, self.obj_num - 1)
        x = x.reshape(mc, 1, -1).repeat(self.obj_num-1, axis=1)
        triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1, self.obj_num-1]))).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat, axis=2)
        f_diversity1 = np.concatenate((f_diversity1, np.ones([mc, 1])), axis=1)
        inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1, self.obj_num-1, self.obj_num-1).repeat(mc, axis=0)
        f_diversity2 = np.sin(np.sum(x*inv_identity, axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([mc, 1]), f_diversity2), axis=1)
        f = -f_diversity1*f_diversity2*(1 - g_)
        return f

    def set_w(self, w):
        self.w = w

    def IGD(self, x):
        """Calculate Inverted Generational Distance."""
        m, n = x.shape
        p, q = self.pareto_f.shape
        x = x.reshape(m, 1, n)
        x = x.repeat(p, axis=1)
        pareto_f = self.pareto_f.reshape(1, p, q)
        pareto_f = pareto_f.repeat(m, axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2, axis=2))
        min_distance = np.min(distance, axis=0)
        igd = np.mean(min_distance)
        return igd
    
    def HV(self, x):
        """Calculate Hypervolume."""
        from pymoo.indicators.hv import HV as HypervolumeIndicator
        hv = HypervolumeIndicator(ref_point=self.hv_ref_point)
        return hv(x)

