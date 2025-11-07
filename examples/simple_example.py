"""
Simple example demonstrating the use of OTES (NSGA-II with Optimal Transport).

This example shows how to:
1. Generate source task data
2. Use NSGA-II-OT for transfer optimization
3. Compare with standard NSGA-II
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import sys
import os

# Add parent directory to path to import otes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from otes import NSGA2_OT
from otes.problems import mDTLZ2
from otes.utils import TransMat


class OptimizationProblem(Problem):
    """Wrapper to convert mDTLZ problem to pymoo Problem format."""
    
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance
        super().__init__(
            n_var=problem_instance.dim,
            n_obj=problem_instance.obj_num,
            xl=0.0,
            xu=1.0
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.problem_instance(x)


def generate_source_data(problem, n_generations=50, pop_size=100):
    """
    Generate source task data by running NSGA-II.
    
    Parameters
    ----------
    problem : Problem
        The optimization problem
    n_generations : int
        Number of generations to run
    pop_size : int
        Population size
        
    Returns
    -------
    source_data : tuple
        (decision_data, objective_data) for each generation
    """
    print("Generating source task data...")
    
    # Create pymoo problem wrapper
    pymoo_problem = OptimizationProblem(problem)
    
    # Run NSGA-II
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(
        pymoo_problem,
        algorithm,
        ('n_gen', n_generations),
        save_history=True,
        verbose=False
    )
    
    # Extract data from history
    decision_data = []
    objective_data = []
    
    for gen in res.history:
        pop_decision, pop_objective = TransMat(gen.pop)
        decision_data.append(pop_decision)
        objective_data.append(pop_objective)
    
    # Add final population data
    pop_decision, pop_objective = TransMat(res.pop)
    decision_data.append(pop_decision)
    objective_data.append(pop_objective)
    
    print(f"Generated {len(decision_data)} generations of source data")
    return (decision_data, objective_data)


def run_comparison():
    """Run comparison between NSGA-II and NSGA-II-OT."""
    
    # Define problem
    print("Setting up mDTLZ2 problem...")
    problem = mDTLZ2(obj_num=3, n_var=10, delta1=1, delta2=0)
    pymoo_problem = OptimizationProblem(problem)
    
    # Generate source data (similar task)
    source_problem = mDTLZ2(obj_num=3, n_var=10, delta1=1, delta2=0.1)  # Slightly different
    source_data = generate_source_data(source_problem, n_generations=50, pop_size=100)
    
    # Run standard NSGA-II
    print("\nRunning standard NSGA-II...")
    algorithm_nsga2 = NSGA2(pop_size=100)
    res_nsga2 = minimize(
        pymoo_problem,
        algorithm_nsga2,
        ('n_gen', 50),
        save_history=True,
        verbose=False
    )
    
    # Run NSGA-II-OT
    print("Running NSGA-II-OT with knowledge transfer...")
    algorithm_ot = NSGA2_OT(
        source_data=source_data,
        pop_size=100,
        transfer_trigger=5,  # Transfer every 5 generations
        model_style='linear'
    )
    res_ot = minimize(
        pymoo_problem,
        algorithm_ot,
        ('n_gen', 50),
        save_history=True,
        verbose=False
    )
    
    # Calculate IGD for each generation
    print("\nCalculating performance metrics...")
    igd_nsga2 = []
    igd_ot = []
    
    for gen in res_nsga2.history:
        _, pop_obj = TransMat(gen.pop)
        igd_nsga2.append(problem.IGD(pop_obj))
    
    for gen in res_ot.history:
        _, pop_obj = TransMat(gen.pop)
        igd_ot.append(problem.IGD(pop_obj))
    
    # Plot convergence curves
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    generations = np.arange(1, len(igd_nsga2) + 1)
    plt.plot(generations, igd_nsga2, label='NSGA-II', marker='o', linewidth=2)
    plt.plot(generations, igd_ot, label='NSGA-II-OT (OTES)', marker='s', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('IGD', fontsize=12)
    plt.title('Convergence Comparison on mDTLZ2', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=300)
    print("Saved convergence plot to 'convergence_comparison.png'")
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"NSGA-II Final IGD:    {igd_nsga2[-1]:.6f}")
    print(f"NSGA-II-OT Final IGD: {igd_ot[-1]:.6f}")
    print(f"Improvement:          {((igd_nsga2[-1] - igd_ot[-1]) / igd_nsga2[-1] * 100):.2f}%")
    print("="*60)


if __name__ == "__main__":
    run_comparison()

