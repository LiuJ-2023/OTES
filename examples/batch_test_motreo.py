"""
MOTrEO Algorithm Comparison Test - MOTrEO vs MOTrEO-OT

This script compares two probabilistic model-based transfer optimization algorithms:
1. MOTrEO - Multi-Objective Transfer Evolutionary Optimization with nonlinear mapping learning
2. MOTrEO-OT - MOTrEO algorithm combined with Optimal Transport

Each test runs 20 times and displays average convergence curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import sys
import os
from datetime import datetime

# Add parent directory to path to import otes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from otes.problems import mDTLZ1, mDTLZ2, mDTLZ3, mDTLZ4, InvDTLZ1, InvDTLZ2, InvDTLZ3, InvDTLZ4
from otes.utils import TransMat

# Import algorithms from OTES package
from otes import NSGA2_MOTrEO, NSGA2_MOTrEO_OT
from otes.utils.probabilistic_model import ProbabilisticModel

# Configure matplotlib
plt.rcParams['font.size'] = 10


class OptimizationProblem(Problem):
    """Wrapper to convert test problem to pymoo Problem format."""
    
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


def generate_source_data(problem, n_generations=50, pop_size=100, seed=None):
    """
    Generate source task data by running NSGA-II.
    
    Returns
    -------
    source_data : tuple
        (decision_data, objective_data) for each generation
    """
    # Create pymoo problem wrapper
    pymoo_problem = OptimizationProblem(problem)
    
    # Run NSGA-II
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(
        pymoo_problem,
        algorithm,
        ('n_gen', n_generations),
        save_history=True,
        verbose=False,
        seed=seed
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
    
    return (decision_data, objective_data)


def build_probabilistic_model(source_data):
    """
    Build probabilistic model from source data.
    
    Parameters
    ----------
    source_data : tuple
        (decision_data, objective_data) from source task
        
    Returns
    -------
    sm_list : list
        List containing a single probabilistic model
    """
    # Use the final generation data
    final_decision = source_data[0][-1]
    
    # Build multivariate normal model
    model = ProbabilisticModel('mvarnorm')
    model.mean = np.mean(final_decision, axis=0)
    model.cov = np.cov(final_decision, rowvar=False)
    model.dim = final_decision.shape[1]
    
    # Add small regularization to ensure positive definite
    model.cov += np.eye(model.dim) * 1e-6
    
    return [model]


def run_single_test(problem_class, problem_params, source_params=None, 
                    n_generations=50, pop_size=100, transfer_trigger=5,
                    seed=None, verbose=True):
    """
    Run a single test comparing MOTrEO and MOTrEO-OT.
    
    Returns
    -------
    results : dict
        Dictionary containing IGD values for both algorithms
    """
    # Create target problem
    target_problem = problem_class(**problem_params)
    pymoo_problem = OptimizationProblem(target_problem)
    
    # Create source problem (slightly different)
    if source_params is None:
        source_params = problem_params.copy()
        if 'delta2' in source_params:
            source_params['delta2'] = source_params.get('delta2', 0) + 0.1
        else:
            source_params['delta2'] = 0.1
    
    source_problem = problem_class(**source_params)
    
    if verbose:
        print(f"  Generating source data...")
    source_data = generate_source_data(source_problem, n_generations=n_generations, 
                                       pop_size=pop_size, seed=seed)
    
    if verbose:
        print(f"  Building probabilistic model...")
    sm_list = build_probabilistic_model(source_data)
    
    # Run MOTrEO
    if verbose:
        print(f"  Running MOTrEO...")
    algorithm_motreo = NSGA2_MOTrEO(
        sm_list=sm_list,
        source_data=source_data,
        pop_size=pop_size,
        transfer_trigger=transfer_trigger
    )
    res_motreo = minimize(
        pymoo_problem,
        algorithm_motreo,
        ('n_gen', n_generations),
        save_history=True,
        verbose=False,
        seed=seed
    )
    
    # Run MOTrEO-OT
    if verbose:
        print(f"  Running MOTrEO-OT...")
    algorithm_motreo_ot = NSGA2_MOTrEO_OT(
        sm_list=sm_list,
        source_data=source_data,
        pop_size=pop_size,
        transfer_trigger=transfer_trigger,
        model_style='linear'
    )
    res_motreo_ot = minimize(
        pymoo_problem,
        algorithm_motreo_ot,
        ('n_gen', n_generations),
        save_history=True,
        verbose=False,
        seed=seed
    )
    
    # Calculate IGD for each generation
    igd_motreo = []
    igd_motreo_ot = []
    
    for gen in res_motreo.history:
        _, pop_obj = TransMat(gen.pop)
        igd_motreo.append(target_problem.IGD(pop_obj))
    
    for gen in res_motreo_ot.history:
        _, pop_obj = TransMat(gen.pop)
        igd_motreo_ot.append(target_problem.IGD(pop_obj))
    
    return {
        'igd_motreo': igd_motreo,
        'igd_motreo_ot': igd_motreo_ot,
        'target_problem': target_problem,
        'problem_name': problem_class.__name__
    }


def run_batch_tests():
    """Run batch tests on 8 problem configurations, each with 20 runs."""
    
    print("="*80)
    print("MOTrEO Algorithm Comparison Test - 8 Test Problems (20 runs each)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define 8 test problem configurations (all 3 objectives)
    test_configs = [
        # ==================== Modified DTLZ Problems ====================
        {
            'problem_class': mDTLZ1,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'mDTLZ1'
        },
        {
            'problem_class': mDTLZ2,
            'problem_params': {'obj_num': 3, 'n_var': 12, 'delta1': 1, 'delta2': 0},
            'name': 'mDTLZ2'
        },
        {
            'problem_class': mDTLZ3,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'mDTLZ3'
        },
        {
            'problem_class': mDTLZ4,
            'problem_params': {'obj_num': 3, 'n_var': 12, 'delta1': 1, 'delta2': 0},
            'name': 'mDTLZ4'
        },
        # ==================== Inverted DTLZ Problems ====================
        {
            'problem_class': InvDTLZ1,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'InvDTLZ1'
        },
        {
            'problem_class': InvDTLZ2,
            'problem_params': {'obj_num': 3, 'n_var': 12, 'delta1': 1, 'delta2': 0},
            'name': 'InvDTLZ2'
        },
        {
            'problem_class': InvDTLZ3,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'InvDTLZ3'
        },
        {
            'problem_class': InvDTLZ4,
            'problem_params': {'obj_num': 3, 'n_var': 12, 'delta1': 1, 'delta2': 0},
            'name': 'InvDTLZ4'
        },
    ]
    
    # Test parameters
    n_generations = 50
    pop_size = 100
    transfer_trigger = 5
    n_runs = 20  # 20 runs per problem
    
    # Store all results
    all_results = []
    
    # 运行每个测试
    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/8] Test Problem: {config['name']}")
        print("-" * 80)
        
        try:
            # Run 20 times and collect results
            igd_motreo_all_runs = []
            igd_motreo_ot_all_runs = []
            
            for run in range(n_runs):
                print(f"  Run {run+1}/{n_runs}...", end='\r')
                
                result = run_single_test(
                    problem_class=config['problem_class'],
                    problem_params=config['problem_params'],
                    n_generations=n_generations,
                    pop_size=pop_size,
                    transfer_trigger=transfer_trigger,
                    seed=run,  # 每次使用不同的种子
                    verbose=False
                )
                
                igd_motreo_all_runs.append(result['igd_motreo'])
                igd_motreo_ot_all_runs.append(result['igd_motreo_ot'])
            
            # Calculate average IGD curves
            avg_igd_motreo = np.mean(igd_motreo_all_runs, axis=0)
            avg_igd_motreo_ot = np.mean(igd_motreo_ot_all_runs, axis=0)
            std_igd_motreo = np.std(igd_motreo_all_runs, axis=0)
            std_igd_motreo_ot = np.std(igd_motreo_ot_all_runs, axis=0)
            
            # Get target problem
            target_problem = config['problem_class'](**config['problem_params'])
            
            result_summary = {
                'igd_motreo': avg_igd_motreo,
                'igd_motreo_ot': avg_igd_motreo_ot,
                'std_motreo': std_igd_motreo,
                'std_motreo_ot': std_igd_motreo_ot,
                'config_name': config['name'],
                'problem_name': config['problem_class'].__name__,
                'target_problem': target_problem
            }
            all_results.append(result_summary)
            
            print(f"  Completed {n_runs} runs                    ")
            
        except Exception as e:
            print(f"  ❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
    
    # Visualize results
    visualize_results(all_results, n_generations)
    
    # Print summary
    print_summary(all_results)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def visualize_results(all_results, n_generations):
    """Visualize convergence curves for all tests."""
    
    print("\nGenerating convergence plots...")
    
    n_tests = len(all_results)
    if n_tests == 0:
        print("  No results available for visualization")
        return
    
    # Create subplots: 2 rows x 4 columns (8 tests)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    generations = np.arange(1, n_generations + 1)
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        
        igd_motreo = result['igd_motreo']
        igd_motreo_ot = result['igd_motreo_ot']
        std_motreo = result['std_motreo']
        std_motreo_ot = result['std_motreo_ot']
        
        # Plot average curves
        ax.plot(generations, igd_motreo, label='MOTrEO', 
                marker='o', markersize=2, linewidth=2, alpha=0.9, color='purple')
        ax.plot(generations, igd_motreo_ot, label='MOTrEO-OT', 
                marker='s', markersize=2, linewidth=2, alpha=0.9, color='orange')
        
        # Add standard deviation shading
        ax.fill_between(generations, igd_motreo - std_motreo, igd_motreo + std_motreo, 
                        alpha=0.15, color='purple')
        ax.fill_between(generations, igd_motreo_ot - std_motreo_ot, igd_motreo_ot + std_motreo_ot, 
                        alpha=0.15, color='orange')
        
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel('IGD', fontsize=11)
        ax.set_title(result['config_name'], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_tests, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'convergence_motreo_test_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Convergence plot saved: {filename}")
    
    plt.close('all')


def print_summary(all_results):
    """Print summary of all test results."""
    
    print("\n" + "="*80)
    print("Test Results Summary (Average of 20 runs)")
    print("="*80)
    print(f"{'Problem':<20} {'MOTrEO':<15} {'MOTrEO-OT':<15}")
    print("-" * 80)
    
    for result in all_results:
        # Display final average IGD
        final_motreo = result['igd_motreo'][-1]
        final_motreo_ot = result['igd_motreo_ot'][-1]
        
        print(f"{result['config_name']:<20} {final_motreo:<15.4f} {final_motreo_ot:<15.4f}")
    
    if all_results:
        # Calculate averages
        avg_motreo = np.mean([r['igd_motreo'][-1] for r in all_results])
        avg_motreo_ot = np.mean([r['igd_motreo_ot'][-1] for r in all_results])
        
        print("-" * 80)
        print(f"{'Average':<20} {avg_motreo:<15.4f} {avg_motreo_ot:<15.4f}")
        
        print("\n" + "="*80)
        print(f"Successfully completed tests: {len(all_results)}/8")
        print(f"Runs per test: 20")
        print(f"Algorithms compared: 2 (MOTrEO, MOTrEO-OT)")
        print("="*80)


if __name__ == "__main__":
    run_batch_tests()

