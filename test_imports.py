"""
Test script to verify OTES installation and imports.
Run this to check if everything is set up correctly.
"""

import sys
import traceback

def test_import(module_name, item_name=None):
    """Test if a module or item can be imported."""
    try:
        if item_name:
            exec(f"from {module_name} import {item_name}")
            print(f"✓ Successfully imported {item_name} from {module_name}")
        else:
            exec(f"import {module_name}")
            print(f"✓ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {item_name or module_name} from {module_name}")
        print(f"  Error: {str(e)}")
        return False

def main():
    """Run all import tests."""
    print("="*60)
    print("OTES Import Test")
    print("="*60)
    
    all_passed = True
    
    # Test external dependencies
    print("\n1. Testing External Dependencies:")
    print("-" * 60)
    deps = [
        'numpy',
        'torch',
        'pymoo',
        'ot',
        'torchmin',
        'matplotlib',
        'scipy'
    ]
    
    for dep in deps:
        if not test_import(dep):
            all_passed = False
    
    # Test OTES core imports
    print("\n2. Testing OTES Core Imports:")
    print("-" * 60)
    
    core_imports = [
        ('otes', None),
        ('otes.algorithms', None),
        ('otes.problems', None),
        ('otes.utils', None),
    ]
    
    for module, item in core_imports:
        if not test_import(module, item):
            all_passed = False
    
    # Test specific classes and functions
    print("\n3. Testing Specific Classes and Functions:")
    print("-" * 60)
    
    specific_imports = [
        ('otes', 'NSGA2_OT'),
        ('otes.problems', 'DTLZ1'),
        ('otes.problems', 'DTLZ2'),
        ('otes.problems', 'DTLZ3'),
        ('otes.problems', 'DTLZ4'),
        ('otes.utils', 'TransMat'),
        ('otes.utils.ndsort', 'fast_non_dominated_sort'),
        ('otes.utils.ndsort', 'environment_selection'),
        ('otes.utils.optimal_transport', 'optimal_transport'),
        ('otes.algorithms.base', 'GeneticAlgorithmOptimalTransport'),
        ('otes.algorithms.nsga2_ot', 'NSGA2_OT'),
    ]
    
    for module, item in specific_imports:
        if not test_import(module, item):
            all_passed = False
    
    # Test creating instances
    print("\n4. Testing Instance Creation:")
    print("-" * 60)
    
    try:
        from otes.problems import DTLZ2
        problem = DTLZ2(obj_num=2, n_var=10)
        print(f"✓ Successfully created DTLZ2 problem instance")
        print(f"  - Dimensions: {problem.dim}")
        print(f"  - Objectives: {problem.obj_num}")
    except Exception as e:
        print(f"✗ Failed to create DTLZ2 instance")
        print(f"  Error: {str(e)}")
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nOTES is correctly installed and ready to use.")
        print("\nNext steps:")
        print("1. Try: python examples/quickstart.py")
        print("2. Read: USAGE.md for detailed guide")
        print("3. Explore: examples/simple_example.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check:")
        print("1. All dependencies are installed: pip install -r requirements.txt")
        print("2. Python version >= 3.7")
        print("3. PYTHONPATH includes the OTES directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())

