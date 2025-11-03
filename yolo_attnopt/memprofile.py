import torch
import gc
import time
from functools import wraps


def format_bytes(bytes_num, default_unit=None):
    """Format bytes to human readable format with optional default unit."""
    if bytes_num < 0:
        return f"-{format_bytes(-bytes_num, default_unit)}"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4
    }
    
    # If default_unit is specified and valid, convert to that unit
    if default_unit is not None and default_unit in units:
        converted_value = bytes_num / unit_multipliers[default_unit]
        return f"{converted_value:.2f} {default_unit}"
    
    # Original auto-scaling behavior if default_unit is None
    for unit in units:
        if bytes_num < 1024.0 or unit == 'TB':
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0

# ANSI color codes
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class MemoryProfiler:
    """A memory profiler for tracking CUDA memory usage."""
    
    def __init__(self, device=None, name=None, use_colors=True, use_symbols=True):
        """Initialize memory profiler.
        
        Args:
            device: CUDA device to track (default: current device)
            name: Optional name for profiling session
            use_colors: Whether to use colorful output
            use_symbols: Whether to use symbolic indicators
        """
        self.device = device if device is not None else torch.device(f'cuda:{torch.cuda.current_device()}')
        self.name = name or "Memory Profile Session"
        self.use_colors = use_colors
        self.use_symbols = use_symbols
        self.start_time = None
        self.reset()
    
    def reset(self):
        """Reset profiler state."""
        self.initial_allocated = 0
        self.peak_allocated = 0
    
    def _clean_environment(self):
        """Ensure clean GPU environment before/after profiling."""
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)
    
    def start(self, name=None):
        """Start memory profiling."""
        # Update name if provided
        if name:
            self.name = name
        
        # Clean environment before starting
        self._clean_environment()
        
        # Record initial state
        self.start_time = time.time()
        self.initial_allocated = torch.cuda.memory_allocated(self.device)
        
        return self
    
    def stop(self, mem_default_unit=None):
        """Stop memory profiling and report results."""
        # Make sure all CUDA operations are complete
        torch.cuda.synchronize()
        
        # Record peak memory usage
        self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
        
        # Calculate pure peak (adjusted for initial memory)
        pure_peak = self.peak_allocated - self.initial_allocated
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        
        # Format the output with color and/or symbols
        if self.use_colors:
            # Create memory profile indicator prefix
            prefix = f"{Colors.PURPLE}{Colors.BOLD}📺 PROF{Colors.ENDC}"
            
            # Format components with color
            block_name = f"{Colors.BLUE}{self.name}{Colors.ENDC}"
            memory_value = f"{Colors.RED}{format_bytes(pure_peak)}{Colors.ENDC}"
            time_value = f"{Colors.GREEN}{elapsed_time:.4f}s{Colors.ENDC}"
            
            if self.use_symbols:
                # Format with symbols and wavy dash separators
                print(f"{prefix} 📌 {block_name} ⌇ 📈 {memory_value} ⌇ ⏱️ {time_value}")
            else:
                # Format without symbols
                print(f"{prefix} {block_name} ⌇ {memory_value} ⌇ {time_value}")
        else:
            # Plain formatting with or without symbols
            if self.use_symbols:
                print(f"📺 PROF 📌 {self.name} ⌇ 📈 {format_bytes(pure_peak, default_unit=mem_default_unit)} ⌇ ⏱️ {elapsed_time:.4f}s")
            else:
                print(f"📺 PROF {self.name} ⌇ {format_bytes(pure_peak, default_unit=mem_default_unit)} ⌇ {elapsed_time:.4f}s")
        
        # Clean up
        self._clean_environment()
        return self
    
    # Context manager support
    def __enter__(self):
        return self.start(self.name)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# Function decorator for profiling
def profile_memory(func=None, *, name=None, use_colors=True, use_symbols=True):
    """Decorator to profile memory usage of a function."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            session_name = name or f"Function: {fn.__name__}"
            with MemoryProfiler(name=session_name, use_colors=use_colors, use_symbols=use_symbols) as _:
                return fn(*args, **kwargs)
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


# Example usage
if __name__ == "__main__":
    # Example 1: Using context manager
    with MemoryProfiler(name="Large Matrix") as prof:
        a = torch.zeros((5000, 5000), device='cuda')
        b = torch.zeros((5000, 5000), device='cuda')
        c = a @ b
    
    # Example 2: Using start/stop
    profiler = MemoryProfiler()
    profiler.start("Addition")
    x = torch.zeros((5000, 5000), device='cuda')
    y = torch.zeros((5000, 5000), device='cuda')
    z = x + y
    profiler.stop()
    
    # Example 3: Using decorator
    @profile_memory(name="Matrix Operations")
    def matrix_ops(size):
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        return a @ b
    
    result = matrix_ops(2000) 