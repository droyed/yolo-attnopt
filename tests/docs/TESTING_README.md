# YOLO Performance Analyzer Unit Tests

This directory contains comprehensive unit tests for the `yolo_performance_analyzer.py` script. The tests cover all functions and methods in the script, including class-based API, backward compatibility functions, and CLI functionality.

## Test Coverage

### 1. PerformanceDataProcessor Class Tests
- **`test_get_model_size_order()`**: Tests model size ordering logic for YOLO models (n, s, m, l, x)
- **`test_process_csv_basic()`**: Tests basic CSV processing functionality
- **`test_process_csv_missing_columns()`**: Tests CSV processing with missing columns
- **`test_process_csv_file_not_found()`**: Tests handling of non-existent CSV files
- **`test_process_csv_empty_file()`**: Tests handling of empty CSV files

### 2. PerformanceVisualizer Class Tests
- **`test_interactive_select_single()`**: Tests single item selection in interactive mode
- **`test_interactive_select_all()`**: Tests "select all" functionality (empty input)
- **`test_interactive_select_all_keyword()`**: Tests "all" keyword selection
- **`test_interactive_select_multiple()`**: Tests multiple item selection
- **`test_interactive_select_empty_list()`**: Tests handling of empty item lists
- **`test_create_plots_basic()`**: Tests basic plot creation functionality
- **`test_create_plots_invalid_approaches()`**: Tests plot creation with invalid approach names
- **`test_create_plots_no_approaches_selected()`**: Tests plot creation with no approaches
- **`test_verify_image_valid()`**: Tests image verification with valid images
- **`test_verify_image_nonexistent()`**: Tests image verification with non-existent files
- **`test_verify_image_too_small()`**: Tests image verification with files that are too small

### 3. PerformanceAnalyzer Class Tests
- **`test_init()`**: Tests analyzer initialization
- **`test_analyze_basic()`**: Tests basic analysis functionality
- **`test_analyze_file_not_found()`**: Tests analysis with non-existent files
- **`test_analyze_invalid_approaches()`**: Tests analysis with invalid approach names

### 4. Backward Compatibility Function Tests
- **`test_get_model_size_order_function()`**: Tests backward compatibility wrapper function
- **`test_interactive_select_function()`**: Tests backward compatibility wrapper function
- **`test_process_performance_metrics_csv_function()`**: Tests backward compatibility wrapper function
- **`test_create_performance_comparison_plots_function()`**: Tests backward compatibility wrapper function
- **`test_verify_output_image_function()`**: Tests backward compatibility wrapper function
- **`test_analyze_performance_metrics_function()`**: Tests backward compatibility wrapper function

### 5. Main CLI Function Tests
- **`test_main_basic()`**: Tests basic main function execution
- **`test_main_with_output_file()`**: Tests main function with output file specified
- **`test_main_non_interactive()`**: Tests main function in non-interactive mode
- **`test_main_file_not_found()`**: Tests main function with non-existent file
- **`test_main_help()`**: Tests main function with --help flag
- **`test_main_version()`**: Tests main function with --version flag

### 6. Integration Tests
- **`test_end_to_end_workflow()`**: Tests complete end-to-end workflow
- **`test_class_based_api_workflow()`**: Tests workflow using class-based API
- **`test_data_processing_edge_cases()`**: Tests data processing with various edge cases

## Running the Tests

### Using the Test Runner (Recommended)

The easiest way to run the tests is using the provided test runner:

```bash
# Run all tests
python run_tests.py

# Run tests for a specific test class
python run_tests.py --class data_processor
python run_tests.py --class visualizer
python run_tests.py --class analyzer
python run_tests.py --class backward_compatibility
python run_tests.py --class main
python run_tests.py --class integration

# List available test classes
python run_tests.py --list-classes
```

### Using unittest Directly

You can also run the tests directly using Python's unittest module:

```bash
# Run all tests
python -m unittest test_yolo_performance_analyzer

# Run a specific test class
python -m unittest test_yolo_performance_analyzer.TestPerformanceDataProcessor

# Run a specific test method
python -m unittest test_yolo_performance_analyzer.TestPerformanceDataProcessor.test_get_model_size_order

# Run with verbose output
python -m unittest -v test_yolo_performance_analyzer
```

### Using pytest

If you have pytest installed, you can also run the tests with pytest:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest test_yolo_performance_analyzer.py

# Run with verbose output
pytest -v test_yolo_performance_analyzer.py

# Run tests for a specific test class
pytest test_yolo_performance_analyzer.py::TestPerformanceDataProcessor
```

## Test Features

### Mocking Strategy
The tests use extensive mocking to:
- Avoid actual file system operations where possible
- Mock matplotlib to prevent display issues during testing
- Mock user input for interactive selection tests
- Create temporary files for testing file operations

### Test Data
The tests create temporary CSV files with sample YOLO performance data:
- Models: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
- Approaches: L0 (Baseline), L1, L2, L3
- Metrics: Memory usage (MB) and Runtime (seconds)

### Error Handling Tests
The tests comprehensively cover error scenarios:
- File not found errors
- Invalid CSV data
- Empty files
- Invalid approach names
- Missing command line arguments
- Invalid user input

### Edge Cases
The tests cover various edge cases:
- Models with unknown size suffixes
- CSV files with missing columns
- Empty item lists for interactive selection
- Very small image files
- Different command line argument combinations

## Dependencies

The tests require the following Python packages:
- `unittest` (built-in)
- `pandas` (for DataFrame operations)
- `numpy` (for numerical operations)
- `matplotlib` (for plotting functionality)
- `tempfile` (built-in, for temporary file operations)
- `unittest.mock` (built-in, for mocking)

## Test Structure

Each test class follows a consistent structure:
- `setUp()`: Creates test fixtures and temporary files
- `tearDown()`: Cleans up test fixtures and temporary files
- Individual test methods: Test specific functionality
- Comprehensive assertions: Verify expected behavior

## Continuous Integration

These tests are designed to be CI/CD friendly:
- No external dependencies beyond the main script's dependencies
- No network operations
- No interactive prompts
- Deterministic results
- Fast execution time

## Adding New Tests

When adding new tests:
1. Follow the existing naming convention: `test_<functionality>()`
2. Use descriptive test method names
3. Include comprehensive docstrings
4. Mock external dependencies appropriately
5. Clean up temporary files in `tearDown()`
6. Test both success and failure scenarios
7. Include edge cases where applicable

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the `benchmark_tools` directory is in your Python path
2. **Missing Dependencies**: Install required packages with `pip install pandas numpy matplotlib`
3. **Permission Errors**: Ensure you have write permissions in the test directory
4. **Matplotlib Display Issues**: Tests mock matplotlib to avoid display problems

### Debug Mode

To run tests in debug mode with more verbose output:

```bash
python -m unittest -v test_yolo_performance_analyzer.TestPerformanceDataProcessor.test_get_model_size_order
```

## Coverage Goals

The current test suite aims to achieve:
- **Line Coverage**: >95%
- **Branch Coverage**: >90%
- **Function Coverage**: 100%
- **Class Coverage**: 100%

## Contributing

When contributing to the test suite:
1. Ensure all new functionality has corresponding tests
2. Update this documentation if adding new test categories
3. Run the full test suite before submitting changes
4. Follow the existing code style and structure
