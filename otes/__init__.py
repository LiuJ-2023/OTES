"""
OTES: Optimal Transport-based Evolutionary Transfer Optimizer
A transfer optimization framework for evolutionary multi-objective optimization using optimal transport.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from otes.algorithms.nsga2_ot import NSGA2_OT
from otes.algorithms.nsga2_ae import NSGA2_AE
from otes.algorithms.nsga2_motreo import NSGA2_MOTrEO
from otes.algorithms.nsga2_motreo_ot import NSGA2_MOTrEO_OT
from otes.problems.mdtlz import (
    mDTLZ1, mDTLZ2, mDTLZ3, mDTLZ4,
    InvDTLZ1, InvDTLZ2, InvDTLZ3, InvDTLZ4
)

__all__ = [
    'NSGA2_OT',
    'NSGA2_AE',
    'NSGA2_MOTrEO',
    'NSGA2_MOTrEO_OT',
    'mDTLZ1',
    'mDTLZ2', 
    'mDTLZ3',
    'mDTLZ4',
    'InvDTLZ1',
    'InvDTLZ2',
    'InvDTLZ3',
    'InvDTLZ4',
]

