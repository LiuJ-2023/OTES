"""Base genetic algorithm with optimal transport"""

from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.initialization import Initialization
from pymoo.core.mating import Mating
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from otes.utils.helpers import TransMat
from otes.utils.optimal_transport import optimal_transport
import torch
import numpy as np


class GeneticAlgorithmOptimalTransport(Algorithm):
    """
    Base genetic algorithm with optimal transport-based knowledge transfer.
    
    This class extends pymoo's Algorithm class to incorporate optimal transport
    for transferring knowledge from source tasks to target tasks.
    """

    def __init__(self,
                 pop_size=None,
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

        # the population size used
        self.pop_size = pop_size

        # whether the algorithm should be advanced after initialization of not
        self.advance_after_initial_infill = advance_after_initial_infill

        # the survival for the genetic algorithm
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # set the duplicate detection class
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
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

        # other run specific data updated whenever solve is called
        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills), 
                                       algorithm=self, **kwargs)

    def _infill(self):
        """
        Generate offspring through either knowledge transfer or mating.
        
        This method checks if it's time to perform knowledge transfer (based on
        transfer_trigger). If yes, it uses optimal transport to transfer solutions
        from source tasks. Otherwise, it performs regular genetic operations.
        """
        if (self.n_gen - 1) % self.transfer_trigger == 0:
            # Perform optimal transport
            pop_decision, pop_objective = TransMat(self.pop)
            # Use min of current generation index and available source data length
            gen_idx = min(self.n_gen, len(self.source_data[0]) - 1)
            pop_decision_source, pop_objective_source = \
                self.source_data[0][gen_idx], self.source_data[1][gen_idx]
            
            if self.model_style == 'linear':
                map_x = optimal_transport(pop_decision_source, pop_objective_source,
                                        pop_decision, pop_objective, 
                                        model_style='linear', reg=self.reg)
            elif self.model_style == 'nonlinear':
                map_x = optimal_transport(pop_decision_source, pop_objective_source,
                                        pop_decision, pop_objective, 
                                        model_style='nonlinear', reg=self.reg)
            
            # Transfer solutions
            xs_trans = map_x(torch.tensor(self.source_data[0][-1], dtype=torch.float))
            xs_trans = xs_trans.detach().numpy().astype(float)
            xs_trans = np.clip(xs_trans, 0, 1)
            
            # Define transferred solutions as the offspring
            off = Population.new("X", xs_trans)
        else:
            # Do the mating using the current population
            off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        # If the mating could not generate any new offspring
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # If not the desired number of offspring could be created
        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    def _advance(self, infills=None, **kwargs):
        """Advance the algorithm by one generation."""
        # The current population
        pop = self.pop

        # Merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        # Execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, 
                                    algorithm=self, **kwargs)

