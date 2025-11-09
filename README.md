# Optimal Transport-Based Distributional Pairing in Transfer Multiobjective Optimization

OTES is a research-oriented framework for evolutionary multi-objective optimization that leverages transfer optimization and optimal transport. It enables algorithms to reuse knowledge from solved source tasks, accelerating convergence and improving solution quality on new target problems.
**Supplementary Files can also be found in this project**

## Key Features
- **Transfer learning for MOPs**: Reuse historical populations collected from similar optimization tasks.
- **Optimal transport mapping**: Learn decision-space mappings between source and target populations.
- **Multiple transfer strategies**: Built-in support for autoencoder, probabilistic model, and optimal transport hybrids.
- **Self-contained package**: All algorithms and utilities live in the `otes` module—no external repositories required.
- **Benchmark-ready**: Includes modified DTLZ (mDTLZ) and modified inverted DTLZ (mDTLZ$^{-1}$) suites for evaluation.

## Supported Algorithms
| Algorithm | Description |
|-----------|-------------|
| `NSGA-II` | Baseline reference implementation from `pymoo` |
| `NSGA-II-OT` | Optimal transport-based transfer (core OTES method) |
| `NSGA-II-AE` | Autoencoder-based mapping between tasks |
| `NSGA-II-MOTrEO` | Probabilistic model transfer with nonlinear mapping |
| `NSGA-II-MOTrEO-OT` | Hybrid method combining optimal transport with probabilistic transfer |

All algorithms can be imported directly from the package:
```python
from otes import NSGA2_OT, NSGA2_AE, NSGA2_MOTrEO, NSGA2_MOTrEO_OT
```

## Installation
```bash
# Clone the repository
git clone https://github.com/LiuJ-2023/OTES.git
cd OTES

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- `torch`
- `pymoo`
- `POT` (Python Optimal Transport)
- `torchmin`
- `numpy`, `scipy`, `matplotlib`

All dependencies are specified in `requirements.txt`.

## Quick Start
1. **Run the quickstart example**
   ```bash
   cd examples
   python quickstart.py
   ```

2. **Minimal code snippet**
   ```python
   import numpy as np
   from pymoo.core.problem import Problem
   from pymoo.optimize import minimize
   from otes import NSGA2_OT
   from otes.problems import mDTLZ2

   # Wrap the benchmark problem for pymoo
   class PymooProblem(Problem):
       def __init__(self, problem):
           super().__init__(n_var=problem.dim, n_obj=problem.obj_num, xl=0.0, xu=1.0)
           self.problem = problem
       def _evaluate(self, x, out, *args, **kwargs):
           out["F"] = self.problem(x)

   problem = mDTLZ2(obj_num=3, n_var=10)
   pymoo_problem = PymooProblem(problem)

   # Mock source data (replace with real data in practice)
   source_decision = [np.random.rand(100, problem.dim) for _ in range(51)]
   source_objective = [np.random.rand(100, problem.obj_num) for _ in range(51)]
   source_data = (source_decision, source_objective)

   algorithm = NSGA2_OT(source_data=source_data, pop_size=100, transfer_trigger=5)
   result = minimize(pymoo_problem, algorithm, ('n_gen', 50), save_history=True, verbose=True)

   print(f"Solutions found: {len(result.F)}")
   ```

## Batch Experiments
Comprehensive scripts are provided in `examples/` to reproduce benchmark comparisons.

| Script | Algorithms Compared |
|--------|---------------------|
| `batch_test_comprehensive.py` | NSGA-II, NSGA-II-OT, NSGA-II-AE, MOTrEO, MOTrEO-OT |
| `batch_test.py` | NSGA-II, NSGA-II-OT, NSGA-II-AE |
| `batch_test_motreo.py` | MOTrEO, MOTrEO-OT |

Example usage:
```bash
cd examples
python batch_test_comprehensive.py
```
Each script runs 20 independent trials on eight mDTLZ/InvDTLZ benchmarks and saves convergence plots with timestamps.

## Project Structure
```
OTES/
├── otes/
│   ├── algorithms/
│   │   ├── base.py                # Optimal transport GA base class
│   │   ├── base_extended.py       # Additional GA base classes
│   │   ├── nsga2_ot.py            # NSGA-II + OT
│   │   ├── nsga2_ae.py            # NSGA-II + Autoencoder
│   │   ├── nsga2_motreo.py        # NSGA-II + MOTrEO
│   │   └── nsga2_motreo_ot.py     # NSGA-II + MOTrEO + OT
│   ├── problems/
│   │   └── mdtlz.py               # Modified and inverted DTLZ problems
│   └── utils/
│       ├── helpers.py             # Population helpers (TransMat)
│       ├── ndsort.py              # Non-dominated sorting helpers
│       ├── optimal_transport.py   # OT and autoencoder utilities
│       ├── probabilistic_model.py # Probabilistic model utilities
│       └── nlmap.py               # Nonlinear mapping utilities
├── examples/
│   ├── quickstart.py              # Minimal tutorial
│   ├── simple_example.py          # Comparison with baseline NSGA-II
│   ├── batch_test.py              # 3-algorithm benchmark
│   ├── batch_test_motreo.py       # MOTrEO benchmark
│   └── batch_test_comprehensive.py# Full 5-algorithm benchmark
├── requirements.txt
├── test_imports.py
└── setup.py (optional packaging stub)
```

## Generating Source Data
OTES expects source populations in the format `(decision_history, objective_history)` where each entry corresponds to one generation. You can generate them by running any evolutionary algorithm (e.g., NSGA-II) on a related task and recording history:
```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from otes.utils import TransMat

res = minimize(problem, NSGA2(pop_size=100), ('n_gen', 50), save_history=True)
decision_history = []
objective_history = []
for gen in res.history:
    dec, obj = TransMat(gen.pop)
    decision_history.append(dec)
    objective_history.append(obj)
source_data = (decision_history, objective_history)
```

## License
This project is released under the MIT License. See `LICENSE` for details.

## Citation
@ARTICLE{11214262,   
     author={Liu, Jiao and Liu, Weiming and En, Joel Tay Wei and Chen, Caishun and Tan, Puay Siew and Ong, Yew-Soon},  
     journal={IEEE Transactions on Evolutionary Computation},  
     title={Optimal Transport-Based Distributional Pairing in Transfer Multiobjective Optimization},    
     year={2025},    
     volume={},    
     number={},    
     pages={1-1},    
     keywords={Optimization;Sorting;Manufacturing;Convergence;Vectors;Search problems;Linear programming;Faces;Artificial intelligence;Pareto optimization;Transfer optimization;multiobjective optimization;optimal transport;solution representation alignment},  
     doi={10.1109/TEVC.2025.3624132}}
