"""
Quickstart example for OTES.

This is the simplest possible example to get started with OTES.
It demonstrates basic usage without requiring pre-existing source data.
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from otes import NSGA2_OT, DTLZ2
from otes.utils import TransMat


class SimpleProblem(Problem):
    """A simple 2-objective optimization problem."""
    
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.sum(x**2, axis=1)
        f2 = np.sum((x - 1)**2, axis=1)
        out["F"] = np.column_stack([f1, f2])


def generate_simple_source_data(n_gen=51, pop_size=100, n_var=10, n_obj=2):
    """
    Generate simple random source data for demonstration.
    In practice, this should come from solving related problems.
    """
    print("Generating random source data (for demonstration)...")
    
    # Random decision variables
    decision_data = [np.random.rand(pop_size, n_var) for _ in range(n_gen)]
    
    # Random objectives (normally would come from evaluating a related problem)
    objective_data = [np.random.rand(pop_size, n_obj) for _ in range(n_gen)]
    
    return (decision_data, objective_data)


def main():
    """Run a simple OTES example."""
    
    print("="*60)
    print("OTES Quickstart Example")
    print("="*60)
    
    # Step 1: Define your problem
    problem = SimpleProblem()
    print(f"\n1. Problem defined: {problem.n_var} variables, {problem.n_obj} objectives")
    
    # Step 2: Generate or load source data
    # In real applications, this would come from solving related problems
    source_data = generate_simple_source_data(
        n_gen=51, 
        pop_size=100, 
        n_var=problem.n_var, 
        n_obj=problem.n_obj
    )
    print(f"2. Source data prepared: {len(source_data[0])} generations")
    
    # Step 3: Create OTES algorithm
    algorithm = NSGA2_OT(
        source_data=source_data,
        pop_size=100,
        transfer_trigger=5,  # Transfer knowledge every 5 generations
        model_style='linear'
    )
    print(f"3. OTES algorithm created (transfer every {algorithm.transfer_trigger} generations)")
    
    # Step 4: Run optimization
    print("\n4. Running optimization...")
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 30),
        seed=1,
        verbose=False
    )
    
    # Step 5: Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Number of solutions in Pareto front: {len(res.F)}")
    print(f"Best objective values (first 5):")
    for i, f in enumerate(res.F[:5]):
        print(f"  Solution {i+1}: F1={f[0]:.4f}, F2={f[1]:.4f}")
    
    print("\n" + "="*60)
    print("Quickstart completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Try the 'simple_example.py' for a more complete comparison")
    print("2. Use your own problems and source data")
    print("3. Experiment with different parameters (transfer_trigger, model_style, etc.)")


if __name__ == "__main__":
    main()

