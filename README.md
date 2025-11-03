# yolo-attnopt


**YOLO-AttnOpt** focuses on optimizing self-attention mechanisms in YOLO for **extreme memory efficiency** without compromising **runtime performance**.
It introduces lightweight attention modules and tensor optimization strategies that significantly reduce the memory footprint during both training and inference.
These optimizations are not limited to YOLO — they can be extended to any application where self-attention is employed.  
The result is a **patched YOLO environment** tailored for **resource-constrained scenarios**.


## Get started

```python

## Setup patched yolo environment
from attention_optimization import setup_optimized_yolo_environ
setup_optimized_yolo_environ(optimize_level=2) # level=2 seems best on memory, perf

## Start using YOLO ...

```

## Installation

Install the package with your desired dependency configuration:

```bash
# Core functionality only
pip install -e .

# With benchmarking tools (includes visualization dependencies)
pip install -e ".[benchmark]"

# With testing dependencies  
pip install -e ".[test]"

# With all optional dependencies
pip install -e ".[benchmark,test]"
```

### Dependencies Overview

**Core Dependencies:**
- `torch>=2.0.0` - PyTorch framework
- `ultralytics>=8.0.0` - YOLO implementation  
- `imagesize>=1.3.0` - Image dimension utilities

**Benchmark Dependencies (optional):**
- `numpy`, `pandas` - Data processing
- `matplotlib`, `seaborn` - Plotting and visualization
- `streamlit` - Web interface
- `Pillow` - Image handling

**Test Dependencies (optional):**
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting
- `unittest-xml-reporting>=3.2.0` - XML test reports
- `coverage>=7.0.0` - Code coverage analysis

## Usage Examples

### Basic YOLO Attention Optimization

```python
from yolo_attnopt.attention_optimization import setup_optimized_yolo_environ
from ultralytics import YOLO

# Configure optimized environment (choose level 0-3)
setup_optimized_yolo_environ(optimize_level=2, debug_mode=True)

# Use YOLO normally - optimizations are now active
model = YOLO('yolo11m.pt')
results = model('path/to/image.jpg')
```

### Memory Profiling

```python
from yolo_attnopt.memprofile import MemoryProfiler, profile_memory

# Using context manager
with MemoryProfiler(name="YOLO Inference") as prof:
    model = YOLO('yolo11x.pt')
    results = model('image.jpg')

# Using decorator
@profile_memory(name="Model Loading")
def load_and_run_model():
    model = YOLO('yolo11l.pt')
    return model('test.jpg')

results = load_and_run_model()
```

### Benchmarking and Analysis

```bash
# Run comprehensive benchmarks
bash benchmark_tools/benchmark_allcombs.sh

# Launch interactive web interface for analysis
streamlit run benchmark_tools/streamlit_app.py

# Command line benchmarking
python benchmark_tools/main_benchmark.py --optimize-level 2 --yolo-modelname yolo11m.pt
```

### Advanced Configuration

```python
import os
from yolo_attnopt.attention_optimization import configure_yolo_environment, patch_ultralytics_attention

# Manual environment configuration
configure_yolo_environment(
    yolo_optimize_level=3,
    yolo_benchmark=True,
    yolo_show_optimization_info=True
)

# Apply optimizations
patch_ultralytics_attention()

# Now use YOLO with custom optimization level
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
```

### Benchmarking

1. Run benchmarking:

```shell
bash benchmark_tools/benchmark_allcombs.sh
```

This generates benchmarked logs that are parsed for plotting in the next step.

2. View benchmark plots on streamlit:

```shell
streamlit run benchmark_tools/streamlit_app.py
```

## Unit testing

The `run_tests.py` script is now a **comprehensive test runner** that serves as a single entry point for testing the entire YOLO performance analysis project. It covers **130+ tests** across:

**YOLO Performance Analysis (Original):**
- PerformanceDataProcessor, PerformanceVisualizer, PerformanceAnalyzer
- Backward compatibility, CLI functionality, integration workflows

**Memory Profiling System (New):**
- format_bytes, Colors, MemoryProfiler, profile_memory decorator
- Complete memprofile workflows and edge cases

**Attention Optimization System (New):**
- Environment configuration, optimization registry, attention profiler
- All optimization levels (0-3), forward method creation, patching functions

**Streamlit Web Application (New):**
- CSV file handling, data loading, plot creation, figure conversion
- Complete web app workflows and error handling

**Benchmark Integration (Original):**
- main_benchmark module functionality

### **🚀 Usage Examples:**

```bash
# List all available test categories
python3 run_tests.py --list-classes

# Run specific test categories
python3 run_tests.py --class format_bytes
python3 run_tests.py --class configure_yolo_env
python3 run_tests.py --class get_csv_files

# Run all tests (130+ tests total)
python3 run_tests.py
```

## Assets and Image Credits

> Image from [Unsplash](https://unsplash.com/photos/a-group-of-people-crossing-a-street-under-a-bridge-iNPI5VlSt4o) (Unsplash License).
> Photographer: Ivan Rohovchenko.
> Image path: `assets/ivan-rohovchenko-iNPI5VlSt4o-unsplash.jpg`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
