"""NSGA-II with Autoencoder-based knowledge transfer"""

import numpy as np
import warnings

from otes.algorithms.base_extended import GeneticAlgorithmAutoEncoding
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


def binary_tournament(pop, P, algorithm, **kwargs):
    """Binary tournament selection for NSGA-II."""
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
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)
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

            # If rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


class RankAndCrowdingSurvival(RankAndCrowding):
    """Deprecated wrapper for RankAndCrowding."""
    
    def __init__(self, nds=None, crowding_func="cd"):
        warnings.warn(
            "RankAndCrowdingSurvival is deprecated and will be removed in version 0.8.*; use RankAndCrowding operator instead.",
            DeprecationWarning, 2
        )
        super().__init__(nds, crowding_func)


class NSGA2_AE(GeneticAlgorithmAutoEncoding):
    """NSGA-II with Autoencoder-based knowledge transfer."""

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
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.transfer_trigger = transfer_trigger
        self.source_data = source_data
        self.model_style = model_style

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


parse_doc_string(NSGA2_AE.__init__)

