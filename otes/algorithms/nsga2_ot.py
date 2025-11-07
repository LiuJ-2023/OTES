"""NSGA-II with Optimal Transport-based Knowledge Transfer"""

import numpy as np
import warnings

from otes.algorithms.base import GeneticAlgorithmOptimalTransport
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible


# Binary Tournament Selection Function
def binary_tournament(pop, P, algorithm, **kwargs):
    """
    Binary tournament selection for NSGA-II.
    
    Parameters
    ----------
    pop : Population
        Current population
    P : np.ndarray
        Tournament pairs
    algorithm : Algorithm
        Current algorithm instance
        
    Returns
    -------
    S : np.ndarray
        Selected individual indices
    """
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")

        # If at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', 
                          return_random_if_equal=True)

        # Both solutions are feasible
        else:
            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # If rank or domination relation didn't make a decision, compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', 
                              return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


class RankAndCrowdingSurvival(RankAndCrowding):
    """
    Survival selection using rank and crowding distance.
    
    This class is deprecated and maintained for backward compatibility.
    Use RankAndCrowding directly instead.
    """
    
    def __init__(self, nds=None, crowding_func="cd"):
        warnings.warn(
            "RankAndCrowdingSurvival is deprecated and will be removed in future versions; "
            "use RankAndCrowding operator instead.",
            DeprecationWarning, 2
        )
        super().__init__(nds, crowding_func)


class NSGA2_OT(GeneticAlgorithmOptimalTransport):
    """
    NSGA-II with Optimal Transport-based Knowledge Transfer (OTES).
    
    This algorithm extends NSGA-II with the ability to transfer knowledge from
    related source optimization tasks using optimal transport theory. The transfer
    happens periodically based on the transfer_trigger parameter.
    
    Parameters
    ----------
    source_data : tuple
        Tuple of (decision_data, objective_data) from source tasks.
        - decision_data: List of decision variable arrays for each generation
        - objective_data: List of objective value arrays for each generation
    pop_size : int, default=100
        Population size
    transfer_trigger : int, default=10
        Number of generations between knowledge transfer operations
    model_style : str, default='linear'
        Mapping model style ('linear' or 'nonlinear')
    sampling : Sampling, default=FloatRandomSampling()
        Sampling strategy for initialization
    selection : Selection, default=TournamentSelection(func_comp=binary_tournament)
        Selection operator
    crossover : Crossover, default=SBX(eta=15, prob=0.9)
        Crossover operator
    mutation : Mutation, default=PM(eta=20)
        Mutation operator
    survival : Survival, default=RankAndCrowding()
        Survival selection operator
    output : Output, default=MultiObjectiveOutput()
        Output display
    reg : float, default=0.1
        Regularization parameter for optimal transport
        
    Examples
    --------
    >>> from otes import NSGA2_OT
    >>> from pymoo.core.problem import Problem
    >>> import numpy as np
    >>> 
    >>> # Define a problem
    >>> class MyProblem(Problem):
    ...     def __init__(self):
    ...         super().__init__(n_var=10, n_obj=2, xl=0, xu=1)
    ...     def _evaluate(self, x, out, *args, **kwargs):
    ...         out["F"] = np.column_stack([x.sum(axis=1), (1-x).sum(axis=1)])
    >>> 
    >>> # Create source data (simplified example)
    >>> source_decision = [np.random.rand(100, 10) for _ in range(50)]
    >>> source_objective = [np.random.rand(100, 2) for _ in range(50)]
    >>> source_data = (source_decision, source_objective)
    >>> 
    >>> # Initialize algorithm
    >>> algorithm = NSGA2_OT(source_data, pop_size=100, transfer_trigger=10)
    >>> 
    >>> # Solve the problem
    >>> from pymoo.optimize import minimize
    >>> res = minimize(MyProblem(), algorithm, ('n_gen', 50))
    """

    def __init__(self,
                 source_data,
                 pop_size=100,
                 transfer_trigger=10,
                 model_style='linear',
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 reg=0.1,
                 **kwargs):
        
        super().__init__(
            source_data=source_data,
            pop_size=pop_size,
            transfer_trigger=transfer_trigger,
            model_style=model_style,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            reg=reg,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.transfer_trigger = transfer_trigger
        self.source_data = source_data
        self.model_style = model_style
        self.reg = reg

    def _set_optimum(self, **kwargs):
        """Set the optimum to the current non-dominated solutions."""
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


parse_doc_string(NSGA2_OT.__init__)

