"""
Batch Test Script - Compare NSGA-II, NSGA-II-OT, and NSGA-II-AE

This script tests 8 different test problem configurations, comparing the performance
of NSGA-II, NSGA-II-OT, and NSGA-II-AE, and generates convergence plots.
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

from otes import NSGA2_OT
from otes.problems import mDTLZ1, mDTLZ2, mDTLZ3, mDTLZ4, InvDTLZ1, InvDTLZ2, InvDTLZ3, InvDTLZ4
from otes.utils import TransMat

# Import algorithms from OTES package
from otes import NSGA2_AE

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
    
    Parameters
    ----------
    problem : Problem
        The optimization problem
    n_generations : int
        Number of generations to run
    pop_size : int
        Population size
    seed : int, optional
        Random seed for reproducibility
        
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


def run_single_test(problem_class, problem_params, source_params=None, 
                    n_generations=50, pop_size=100, transfer_trigger=5,
                    seed=None, verbose=True):
    """
    Run a single test comparing NSGA-II, NSGA-II-OT, and NSGA-II-AE.
    
    Parameters
    ----------
    problem_class : class
        Problem class (mDTLZ1, mDTLZ2, InvDTLZ1, InvDTLZ2, etc.)
    problem_params : dict
        Parameters for target problem
    source_params : dict, optional
        Parameters for source problem (if None, uses problem_params with slight variation)
    n_generations : int
        Number of generations
    pop_size : int
        Population size
    transfer_trigger : int
        Transfer frequency
    seed : int, optional
        Random seed
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : dict
        Dictionary containing IGD values for all three algorithms
    """
    # Create target problem
    target_problem = problem_class(**problem_params)
    pymoo_problem = OptimizationProblem(target_problem)
    
    # Create source problem (slightly different)
    if source_params is None:
        source_params = problem_params.copy()
        source_params['delta1'] = 0.7
        source_params['delta2'] = 0.25
    
    source_problem = problem_class(**source_params)
    
    if verbose:
        print(f"  Generating source data...")
    source_data = generate_source_data(source_problem, n_generations=n_generations, 
                                       pop_size=pop_size, seed=seed)
    
    # Run standard NSGA-II
    if verbose:
        print(f"  Running NSGA-II...")
    algorithm_nsga2 = NSGA2(pop_size=pop_size)
    res_nsga2 = minimize(
        pymoo_problem,
        algorithm_nsga2,
        ('n_gen', n_generations),
        save_history=True,
        verbose=False,
        seed=seed
    )
    
    # Run NSGA-II-OT
    if verbose:
        print(f"  Running NSGA-II-OT...")
    algorithm_ot = NSGA2_OT(
        source_data=source_data,
        pop_size=pop_size,
        transfer_trigger=transfer_trigger,
        model_style='linear'
    )
    res_ot = minimize(
        pymoo_problem,
        algorithm_ot,
        ('n_gen', n_generations),
        save_history=True,
        verbose=False,
        seed=seed
    )
    
    # Run NSGA-II-AE
    if verbose:
        print(f"  Running NSGA-II-AE...")
    algorithm_ae = NSGA2_AE(
        source_data=source_data,
        pop_size=pop_size,
        transfer_trigger=transfer_trigger,
        model_style='linear'
    )
    res_ae = minimize(
        pymoo_problem,
        algorithm_ae,
        ('n_gen', n_generations),
        save_history=True,
        verbose=False,
        seed=seed
    )
    
    # Calculate IGD for each generation
    igd_nsga2 = []
    igd_ot = []
    igd_ae = []
    
    for gen in res_nsga2.history:
        _, pop_obj = TransMat(gen.pop)
        igd_nsga2.append(target_problem.IGD(pop_obj))
    
    for gen in res_ot.history:
        _, pop_obj = TransMat(gen.pop)
        igd_ot.append(target_problem.IGD(pop_obj))
    
    for gen in res_ae.history:
        _, pop_obj = TransMat(gen.pop)
        igd_ae.append(target_problem.IGD(pop_obj))
    
    return {
        'igd_nsga2': igd_nsga2,
        'igd_ot': igd_ot,
        'igd_ae': igd_ae,
        'target_problem': target_problem,
        'problem_name': problem_class.__name__
    }


def run_batch_tests():
    """Run batch tests on 8 problem configurations (4 mDTLZ + 4 invDTLZ), each with 20 runs."""
    
    print("="*80)
    print("OTES Batch Test - 8 Test Problems (20 runs each)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 定义8个测试问题配置（都是3目标，10变量）
    test_configs = [
        # ==================== Modified DTLZ Problems ====================
        # 配置1: mDTLZ1, 3 objectives, 10 variables
        {
            'problem_class': mDTLZ1,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'mDTLZ1'
        },
        # 配置2: mDTLZ2, 3 objectives, 10 variables
        {
            'problem_class': mDTLZ2,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'mDTLZ2'
        },
        # 配置3: mDTLZ3, 3 objectives, 10 variables
        {
            'problem_class': mDTLZ3,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'mDTLZ3'
        },
        # 配置4: mDTLZ4, 3 objectives, 10 variables
        {
            'problem_class': mDTLZ4,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'mDTLZ4'
        },
        # ==================== Inverted DTLZ Problems ====================
        # 配置5: InvDTLZ1, 3 objectives, 10 variables
        {
            'problem_class': InvDTLZ1,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'InvDTLZ1'
        },
        # 配置6: InvDTLZ2, 3 objectives, 10 variables
        {
            'problem_class': InvDTLZ2,
            'problem_params': {'obj_num': 3, 'n_var': 12, 'delta1': 1, 'delta2': 0},
            'name': 'InvDTLZ2'
        },
        # 配置7: InvDTLZ3, 3 objectives, 10 variables
        {
            'problem_class': InvDTLZ3,
            'problem_params': {'obj_num': 3, 'n_var': 10, 'delta1': 1, 'delta2': 0},
            'name': 'InvDTLZ3'
        },
        # 配置8: InvDTLZ4, 3 objectives, 10 variables
        {
            'problem_class': InvDTLZ4,
            'problem_params': {'obj_num': 3, 'n_var': 12, 'delta1': 1, 'delta2': 0},
            'name': 'InvDTLZ4'
        },
    ]
    
    # 测试参数
    n_generations = 50
    pop_size = 100
    transfer_trigger = 5
    n_runs = 20  # 每个问题运行20次
    
    # 存储所有结果
    all_results = []
    
    # 运行每个测试
    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/8] Test Problem: {config['name']}")
        print("-" * 80)
        
        try:
            # 运行20次并收集结果
            igd_nsga2_all_runs = []
            igd_ot_all_runs = []
            igd_ae_all_runs = []
            
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
                
                igd_nsga2_all_runs.append(result['igd_nsga2'])
                igd_ot_all_runs.append(result['igd_ot'])
                igd_ae_all_runs.append(result['igd_ae'])
            
            # 计算平均IGD曲线
            avg_igd_nsga2 = np.mean(igd_nsga2_all_runs, axis=0)
            avg_igd_ot = np.mean(igd_ot_all_runs, axis=0)
            avg_igd_ae = np.mean(igd_ae_all_runs, axis=0)
            std_igd_nsga2 = np.std(igd_nsga2_all_runs, axis=0)
            std_igd_ot = np.std(igd_ot_all_runs, axis=0)
            std_igd_ae = np.std(igd_ae_all_runs, axis=0)
            
            # 获取目标问题用于IGD计算（用于显示）
            target_problem = config['problem_class'](**config['problem_params'])
            
            result_summary = {
                'igd_nsga2': avg_igd_nsga2,
                'igd_ot': avg_igd_ot,
                'igd_ae': avg_igd_ae,
                'std_nsga2': std_igd_nsga2,
                'std_ot': std_igd_ot,
                'std_ae': std_igd_ae,
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
        
        igd_nsga2 = result['igd_nsga2']
        igd_ot = result['igd_ot']
        igd_ae = result['igd_ae']
        std_nsga2 = result['std_nsga2']
        std_ot = result['std_ot']
        std_ae = result['std_ae']
        
        # Plot average curves
        ax.plot(generations, igd_nsga2, label='NSGA-II', 
                marker='o', markersize=2, linewidth=2, alpha=0.9, color='blue')
        ax.plot(generations, igd_ot, label='NSGA-II-OT', 
                marker='s', markersize=2, linewidth=2, alpha=0.9, color='red')
        ax.plot(generations, igd_ae, label='NSGA-II-AE', 
                marker='^', markersize=2, linewidth=2, alpha=0.9, color='green')
        
        # Add standard deviation shading
        ax.fill_between(generations, igd_nsga2 - std_nsga2, igd_nsga2 + std_nsga2, 
                        alpha=0.15, color='blue')
        ax.fill_between(generations, igd_ot - std_ot, igd_ot + std_ot, 
                        alpha=0.15, color='red')
        ax.fill_between(generations, igd_ae - std_ae, igd_ae + std_ae, 
                        alpha=0.15, color='green')
        
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
    filename = f'convergence_batch_test_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Convergence plot saved: {filename}")
    
    plt.close('all')




def print_summary(all_results):
    """Print summary of all test results."""
    
    print("\n" + "="*80)
    print("Test Results Summary (Average of 20 runs)")
    print("="*80)
    print(f"{'Problem':<20} {'NSGA-II':<15} {'NSGA-II-OT':<15} {'NSGA-II-AE':<15}")
    print("-" * 80)
    
    for result in all_results:
        # 显示最终平均IGD
        final_nsga2 = result['igd_nsga2'][-1]
        final_ot = result['igd_ot'][-1]
        final_ae = result['igd_ae'][-1]
        
        print(f"{result['config_name']:<20} {final_nsga2:<15.4f} {final_ot:<15.4f} {final_ae:<15.4f}")
    
    if all_results:
        # 计算平均值
        avg_nsga2 = np.mean([r['igd_nsga2'][-1] for r in all_results])
        avg_ot = np.mean([r['igd_ot'][-1] for r in all_results])
        avg_ae = np.mean([r['igd_ae'][-1] for r in all_results])
        
        print("-" * 80)
        print(f"{'Average':<20} {avg_nsga2:<15.4f} {avg_ot:<15.4f} {avg_ae:<15.4f}")
        
        print("\n" + "="*80)
        print(f"Successfully completed tests: {len(all_results)}/8")
        print(f"Runs per test: 20")
        print(f"Algorithms compared: 3 (NSGA-II, NSGA-II-OT, NSGA-II-AE)")
        print("="*80)


if __name__ == "__main__":
    run_batch_tests()

