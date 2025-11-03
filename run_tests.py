#!/usr/bin/env python3
"""
Test Runner for YOLO Performance Analyzer Unit Tests

This script provides an easy way to run the comprehensive unit tests
for the YOLO performance analyzer script.
"""

import sys
import os
import unittest
import argparse

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Add the tests directory to the Python path for relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

# Add the benchmark_tools directory to the Python path for streamlit_app imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmark_tools'))

def run_all_tests():
    """Run all unit tests."""
    # Import the test module
    from unit.test_yolo_performance_analyzer import (
        TestPerformanceDataProcessor,
        TestPerformanceVisualizer,
        TestPerformanceAnalyzer,
        TestBackwardCompatibilityFunctions,
        TestMainFunction,
        TestIntegrationScenarios
    )
    from integration.test_main_benchmark import TestMainBenchmark
    
    # Import memprofile test classes
    from unit.test_memprofile import (
        TestFormatBytes,
        TestColors,
        TestMemoryProfiler,
        TestProfileMemoryDecorator,
        TestIntegration
    )
    
    # Import attention optimization test classes (pytest-style)
    from unit.test_attention_optimization import (
        TestConfigureYoloEnvironment,
        TestSetupOptimizedYoloEnvironment,
        TestOptimizationLevelRegistry,
        TestAttentionProfiler,
        TestOptimizationImplementations,
        TestCreateOptimizedForward,
        TestPatchFunctions,
        TestIntegrationScenarios as TestIntegrationScenariosAO  # AO = Attention Optimization
    )
    
    # Import streamlit app test classes
    from integration.test_streamlit_app import (
        TestGetCSVFiles,
        TestLoadCSVData,
        TestCreatePlot,
        TestFigToBytes,
        TestMainFunction,
        TestIntegrationScenarios as TestIntegrationScenariosSA  # SA = Streamlit App
    )
    
    # Create a test suite with all unittest-style test classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add unittest-style test classes to the suite
    test_classes = [
        TestPerformanceDataProcessor,
        TestPerformanceVisualizer,
        TestPerformanceAnalyzer,
        TestBackwardCompatibilityFunctions,
        TestMainFunction,
        TestIntegrationScenarios,
        TestMainBenchmark,
        # Memprofile test classes
        TestFormatBytes,
        TestColors,
        TestMemoryProfiler,
        TestProfileMemoryDecorator,
        TestIntegration,
        # Streamlit app test classes
        TestGetCSVFiles,
        TestLoadCSVData,
        TestCreatePlot,
        TestFigToBytes,
        TestMainFunction,
        TestIntegrationScenariosSA
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run unittest tests
    print("Running unittest-style tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    unittest_success = result.wasSuccessful()
    
    # Run pytest-style tests (attention optimization)
    print("\nRunning pytest-style tests (attention optimization)...")
    import subprocess
    import sys
    
    pytest_result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        'tests/unit/test_attention_optimization.py', 
        '-v'
    ], capture_output=True, text=True)
    
    print(pytest_result.stdout)
    if pytest_result.stderr:
        print(pytest_result.stderr, file=sys.stderr)
    
    pytest_success = pytest_result.returncode == 0
    
    # Return overall success
    return unittest_success and pytest_success

def run_specific_test_class(test_class_name):
    """Run tests for a specific test class."""
    from unit.test_yolo_performance_analyzer import (
        TestPerformanceDataProcessor,
        TestPerformanceVisualizer,
        TestPerformanceAnalyzer,
        TestBackwardCompatibilityFunctions,
        TestMainFunction,
        TestIntegrationScenarios
    )
    from integration.test_main_benchmark import TestMainBenchmark
    
    # Import memprofile test classes
    from unit.test_memprofile import (
        TestFormatBytes,
        TestColors,
        TestMemoryProfiler,
        TestProfileMemoryDecorator,
        TestIntegration
    )
    
    # Import attention optimization test classes (pytest-style)
    from unit.test_attention_optimization import (
        TestConfigureYoloEnvironment,
        TestSetupOptimizedYoloEnvironment,
        TestOptimizationLevelRegistry,
        TestAttentionProfiler,
        TestOptimizationImplementations,
        TestCreateOptimizedForward,
        TestPatchFunctions,
        TestIntegrationScenarios as TestIntegrationScenariosAO  # AO = Attention Optimization
    )
    
    # Import streamlit app test classes
    from integration.test_streamlit_app import (
        TestGetCSVFiles,
        TestLoadCSVData,
        TestCreatePlot,
        TestFigToBytes,
        TestMainFunction,
        TestIntegrationScenarios as TestIntegrationScenariosSA  # SA = Streamlit App
    )
    
    # Check if this is a pytest-style test (attention optimization tests)
    pytest_tests = [
        'configure_yolo_env', 'setup_optimized_yolo_env', 'optimization_registry',
        'attention_profiler', 'optimization_implementations', 'create_optimized_forward',
        'patch_functions', 'attention_optimization_integration'
    ]
    
    if test_class_name in pytest_tests:
        # Run pytest-style tests
        import subprocess
        import sys
        
        # Map test class names to pytest test files
        pytest_mapping = {
            'configure_yolo_env': 'tests/unit/test_attention_optimization.py::TestConfigureYoloEnvironment',
            'setup_optimized_yolo_env': 'tests/unit/test_attention_optimization.py::TestSetupOptimizedYoloEnvironment',
            'optimization_registry': 'tests/unit/test_attention_optimization.py::TestOptimizationLevelRegistry',
            'attention_profiler': 'tests/unit/test_attention_optimization.py::TestAttentionProfiler',
            'optimization_implementations': 'tests/unit/test_attention_optimization.py::TestOptimizationImplementations',
            'create_optimized_forward': 'tests/unit/test_attention_optimization.py::TestCreateOptimizedForward',
            'patch_functions': 'tests/unit/test_attention_optimization.py::TestPatchFunctions',
            'attention_optimization_integration': 'tests/unit/test_attention_optimization.py::TestIntegrationScenarios'
        }
        
        test_path = pytest_mapping[test_class_name]
        result = subprocess.run([sys.executable, '-m', 'pytest', test_path, '-v'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    else:
        # Run unittest-style tests
        test_classes = {
            'data_processor': TestPerformanceDataProcessor,
            'visualizer': TestPerformanceVisualizer,
            'analyzer': TestPerformanceAnalyzer,
            'backward_compatibility': TestBackwardCompatibilityFunctions,
            'main': TestMainFunction,
            'integration': TestIntegrationScenarios,
            'main_benchmark': TestMainBenchmark,
            # Memprofile test categories
            'format_bytes': TestFormatBytes,
            'colors': TestColors,
            'memory_profiler': TestMemoryProfiler,
            'profile_memory_decorator': TestProfileMemoryDecorator,
            'memprofile_integration': TestIntegration,
            # Streamlit app test categories
            'get_csv_files': TestGetCSVFiles,
            'load_csv_data': TestLoadCSVData,
            'create_plot': TestCreatePlot,
            'fig_to_bytes': TestFigToBytes,
            'streamlit_main': TestMainFunction,
            'streamlit_integration': TestIntegrationScenariosSA
        }
        
        if test_class_name not in test_classes:
            print(f"Error: Unknown test class '{test_class_name}'")
            print(f"Available test classes: {list(test_classes.keys())}")
            return False
        
        test_class = test_classes[test_class_name]
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()

def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for YOLO Performance Analysis Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all tests
  %(prog)s --class data_processor  # Run only PerformanceDataProcessor tests
  %(prog)s --class visualizer      # Run only PerformanceVisualizer tests
  %(prog)s --class format_bytes    # Run only format_bytes tests
  %(prog)s --class memory_profiler # Run only MemoryProfiler tests
  %(prog)s --class configure_yolo_env  # Run only configure_yolo_environment tests
  %(prog)s --class get_csv_files   # Run only get_csv_files tests
  %(prog)s --list-classes          # List available test classes
        """
    )
    
    parser.add_argument(
        '--class', '-c',
        dest='test_class',
        help='Run tests for a specific test class'
    )
    
    parser.add_argument(
        '--list-classes', '-l',
        action='store_true',
        help='List available test classes'
    )
    
    args = parser.parse_args()
    
    if args.list_classes:
        print("Available test classes:")
        print("  data_processor       - Tests for PerformanceDataProcessor class")
        print("  visualizer          - Tests for PerformanceVisualizer class")
        print("  analyzer            - Tests for PerformanceAnalyzer class")
        print("  backward_compatibility - Tests for backward compatibility functions")
        print("  main                - Tests for main CLI function")
        print("  integration         - Tests for complete workflows")
        print("  main_benchmark      - Tests for main_benchmark module")
        print("")
        print("  Memprofile Tests:")
        print("  format_bytes        - Tests for format_bytes function")
        print("  colors              - Tests for Colors class")
        print("  memory_profiler     - Tests for MemoryProfiler class")
        print("  profile_memory_decorator - Tests for profile_memory decorator")
        print("  memprofile_integration - Tests for complete memprofile workflows")
        print("")
        print("  Attention Optimization Tests:")
        print("  configure_yolo_env  - Tests for configure_yolo_environment function")
        print("  setup_optimized_yolo_env - Tests for setup_optimized_yolo_environ function")
        print("  optimization_registry - Tests for OptimizationLevelRegistry class")
        print("  attention_profiler  - Tests for AttentionProfiler class")
        print("  optimization_implementations - Tests for optimization level implementations")
        print("  create_optimized_forward - Tests for create_optimized_forward function")
        print("  patch_functions     - Tests for ultralytics patching functions")
        print("  attention_optimization_integration - Tests for complete optimization workflows")
        print("")
        print("  Streamlit App Tests:")
        print("  get_csv_files       - Tests for get_csv_files function")
        print("  load_csv_data       - Tests for load_csv_data function")
        print("  create_plot         - Tests for create_plot function")
        print("  fig_to_bytes        - Tests for fig_to_bytes function")
        print("  streamlit_main      - Tests for main Streamlit app function")
        print("  streamlit_integration - Tests for complete Streamlit app workflows")
        return
    
    try:
        if args.test_class:
            print(f"Running tests for: {args.test_class}")
            success = run_specific_test_class(args.test_class)
        else:
            print("Running all YOLO Performance Analyzer unit tests...")
            success = run_all_tests()
        
        if success:
            print("\n✅ All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
