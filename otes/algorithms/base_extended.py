"""Extended base classes for additional transfer learning algorithms"""

from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.initialization import Initialization
from pymoo.core.mating import Mating
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from otes.utils.helpers import TransMat
from otes.utils.optimal_transport import autoencoding_linear, autoencoding_nonlinear
from otes.utils.optimal_transport import optimal_transport
from otes.utils.probabilistic_model import MixtureModel
from otes.utils.nlmap import learn_map, model_transform, model_transform_torch
import torch
import numpy as np


class GeneticAlgorithmAutoEncoding(Algorithm):
    """Base genetic algorithm with autoencoder-based knowledge transfer."""

    def __init__(self,
                 source_data=None,
                 pop_size=None,
                 transfer_trigger=10,
                 model_style='linear',
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 mating=None,
                 advance_after_initial_infill=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.advance_after_initial_infill = advance_after_initial_infill
        self.survival = survival
        self.n_offsprings = n_offsprings

        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        if mating is None:
            mating = Mating(selection,
                            crossover,
                            mutation,
                            repair=self.repair,
                            eliminate_duplicates=self.eliminate_duplicates,
                            n_max_iterations=100)
        self.mating = mating

        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills), algorithm=self, **kwargs)

    def _infill(self):
        if (self.n_gen - 1) % self.transfer_trigger == 0:
            # Do autoencoding transfer
            pop_decision, pop_objective = TransMat(self.pop)
            # Use min of current generation index and available source data length
            gen_idx = min(self.n_gen, len(self.source_data[0]) - 1)
            pop_decision_source, pop_objective_source = self.source_data[0][gen_idx], self.source_data[1][gen_idx]
            if self.model_style == 'linear':
                M = autoencoding_linear(pop_decision_source, pop_objective_source, pop_decision, pop_objective)
                # Transfer solutions
                xs_trans = np.dot(np.concatenate((self.source_data[0][-1], np.ones([self.source_data[0][-1].shape[0], 1])), axis=1), M.T)
                xs_trans = np.clip(xs_trans, 0, 1)
                # Define transferred solutions as the offspring
                off = Population.new("X", xs_trans[:, :-1])
            elif self.model_style == 'nonlinear':
                map_x = autoencoding_nonlinear(pop_decision_source, pop_objective_source, pop_decision, pop_objective)
                xs_trans = map_x(torch.tensor(self.source_data[0][-1], dtype=torch.float))
                xs_trans = xs_trans.detach().numpy().astype(float)
                # Define transferred solutions as the offspring
                off = Population.new("X", xs_trans)
        else:
            # Do the mating using the current population
            off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        if len(off) == 0:
            self.termination.force_termination = True
            return

        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    def _advance(self, infills=None, **kwargs):
        pop = self.pop

        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)


class GeneticAlgorithmMOTrEO(Algorithm):
    """Base genetic algorithm with MOTrEO (probabilistic model-based) transfer."""

    def __init__(self,
                 sm_list=None,
                 source_data=None,
                 pop_size=None,
                 transfer_trigger=10,
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 mating=None,
                 advance_after_initial_infill=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.advance_after_initial_infill = advance_after_initial_infill
        self.survival = survival
        self.n_offsprings = n_offsprings
        self.sm_list = sm_list
        self.source_data = source_data
        self.transfer_trigger = transfer_trigger

        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        if mating is None:
            mating = Mating(selection,
                            crossover,
                            mutation,
                            repair=self.repair,
                            eliminate_duplicates=self.eliminate_duplicates,
                            n_max_iterations=100)
        self.mating = mating

        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills), algorithm=self, **kwargs)

    def _infill(self):
        if (self.n_gen - 1) % self.transfer_trigger == 0:
            # NLMap Training
            pop_decision, pop_objective = TransMat(self.pop)
            # Use min of current generation index and available source data length
            gen_idx = min(self.n_gen, len(self.source_data[0]) - 1)
            pop_decision_source, pop_objective_source = self.source_data[0][gen_idx], self.source_data[1][gen_idx]
            map = learn_map(pop_decision_source, pop_decision, 100)

            # Do AMT (Adaptive Model Transfer)
            new_models = [model_transform(self.sm_list[0], map)]
            mm = MixtureModel(new_models)
            mm.createTable(np.array(pop_decision), True, 'mvarnorm')
            mm.EMstacking()
            mm.mutate()
            coefficients = np.around(mm.alpha, decimals=5)

            # Sample solutions
            offspring_A = np.array(mm.sample(self.pop_size))
            offspring_B = np.clip(offspring_A, 0, 1)
            # Define transferred solutions as the offspring
            off = Population.new("X", offspring_B)
        else:
            # Do the mating using the current population
            off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        if len(off) == 0:
            self.termination.force_termination = True
            return

        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    def _advance(self, infills=None, **kwargs):
        pop = self.pop

        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)


class GeneticAlgorithmOptimalTransportMOTrEO(Algorithm):
    """Base genetic algorithm with Optimal Transport + MOTrEO transfer."""

    def __init__(self,
                 sm_list=None,
                 source_data=None,
                 pop_size=None,
                 transfer_trigger=10,
                 model_style='linear',
                 sampling=None,
                 selection=None,
                 crossover=None,
                 mutation=None,
                 survival=None,
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 mating=None,
                 advance_after_initial_infill=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.advance_after_initial_infill = advance_after_initial_infill
        self.survival = survival
        self.n_offsprings = n_offsprings
        self.sm_list = sm_list
        self.source_data = source_data
        self.transfer_trigger = transfer_trigger
        self.model_style = model_style

        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        if mating is None:
            mating = Mating(selection,
                            crossover,
                            mutation,
                            repair=self.repair,
                            eliminate_duplicates=self.eliminate_duplicates,
                            n_max_iterations=100)
        self.mating = mating

        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills), algorithm=self, **kwargs)

    def _infill(self):
        if (self.n_gen - 1) % self.transfer_trigger == 0:
            # Do optimal transport
            pop_decision, pop_objective = TransMat(self.pop)
            # Use min of current generation index and available source data length
            gen_idx = min(self.n_gen, len(self.source_data[0]) - 1)
            pop_decision_source, pop_objective_source = self.source_data[0][gen_idx], self.source_data[1][gen_idx]
            if self.model_style == 'linear':
                map_x = optimal_transport(pop_decision_source, pop_objective_source, pop_decision, pop_objective, model_style='linear')
            elif self.model_style == 'nonlinear':
                map_x = optimal_transport(pop_decision_source, pop_objective_source, pop_decision, pop_objective, model_style='nonlinear')

            # Do AMT (Adaptive Model Transfer)
            new_models = [model_transform_torch(self.sm_list[0], map_x, nSol=self.pop_size)]
            mm = MixtureModel(new_models)
            mm.createTable(np.array(pop_decision), True, 'mvarnorm')
            mm.EMstacking()
            mm.mutate()
            coefficients = np.around(mm.alpha, decimals=5)

            # Sample solutions
            offspring_A = np.array(mm.sample(self.pop_size))
            offspring_B = np.clip(offspring_A, 0, 1)
            # Define transferred solutions as the offspring
            off = Population.new("X", offspring_B)
        else:
            # Do the mating using the current population
            off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        if len(off) == 0:
            self.termination.force_termination = True
            return

        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    def _advance(self, infills=None, **kwargs):
        pop = self.pop

        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)

