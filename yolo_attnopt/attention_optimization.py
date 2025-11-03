"""
Attention Optimization Patcher for Ultralytics YOLO

This module patches the Attention class in ultralytics to add multiple optimization levels
for benchmarking and memory profiling purposes. It reads configuration from environment
variables directly.

Key improvements:
- Separated optimization implementations into individual functions
- Registry pattern for easy extension
- Cleaner code organization
- Better maintainability

Usage Examples:

    1. Basic usage with environment variables:
        import os
        from attention_optimization import patch_ultralytics_attention
        
        # Set optimization level via environment variable
        os.environ['YOLO_OPTIMIZE_LEVEL'] = '2'
        os.environ['YOLO_SHOW_OPTIMIZATION_INFO'] = 'true'
        os.environ['YOLO_BENCHMARK'] = 'true'
        
        # Apply the patch
        patch_ultralytics_attention()
        
        # Now use YOLO normally - it will have the optimization levels
        from ultralytics import YOLO
        model = YOLO('yolo11m.pt')
    
    2. Simple usage with defaults:
        from patch_attention_optimization_modular_refactored import patch_ultralytics_attention
        
        # Apply the patch with default settings (level 0, no benchmarking)
        patch_ultralytics_attention()
        
        # Use YOLO normally
        from ultralytics import YOLO
        model = YOLO('yolo11m.pt')
"""

import torch
import os
from typing import Callable
from .memprofile import MemoryProfiler


def configure_yolo_environment(yolo_optimize_level: int,
                               yolo_benchmark: bool = False,
                               yolo_show_optimization_info: bool = False) -> bool:
    """Configure YOLO optimization environment variables.

    Validates the optimization level and sets environment variables for YOLO model
    optimization settings. Prints the description of the selected optimization level.

    Args:
        yolo_optimize_level (int): Optimization level for YOLO processing.
            Must be one of the registered levels.
        yolo_benchmark (bool, optional): Whether to enable benchmarking mode.
            Defaults to False.
        yolo_show_optimization_info (bool, optional): Whether to display optimization
            information. Defaults to False.

    Returns:
        bool: True if configuration was successful, False if optimization level
            is invalid.

    Environment Variables Set:
        - YOLO_OPTIMIZE_LEVEL: String representation of optimization level
        - YOLO_BENCHMARK: 'true' or 'false' based on yolo_benchmark
        - YOLO_SHOW_OPTIMIZATION_INFO: 'true' or 'false' based on yolo_show_optimization_info

    Prints:
        Optimization level description if valid level is provided.
    """
    # Reconstruct optimization_levels dict from the registry to avoid duplication
    optimization_levels = {}
    for level, config in OptimizationLevelRegistry.REGISTRY.items():
        optimization_levels[level] = config['description']

    # Validate optimization level
    if yolo_optimize_level not in optimization_levels:
        return False

    # Convert boolean values to string format
    benchmark_val = 'true' if yolo_benchmark else 'false'
    show_info_val = 'true' if yolo_show_optimization_info else 'false'

    # Set environment variables
    os.environ['YOLO_OPTIMIZE_LEVEL'] = str(yolo_optimize_level)
    os.environ['YOLO_BENCHMARK'] = benchmark_val
    os.environ['YOLO_SHOW_OPTIMIZATION_INFO'] = show_info_val

    # Print optimization level description
    if yolo_optimize_level in optimization_levels:
        print(f"[INFO] ✓ Optimization set to Level # {yolo_optimize_level} -> {optimization_levels[yolo_optimize_level]}")

    return True


def setup_optimized_yolo_environ(optimize_level, debug_mode=False):
    """Configure YOLO environment and apply attention optimizations."""
    env_config_success = configure_yolo_environment(
        yolo_optimize_level=optimize_level,
        yolo_benchmark=debug_mode,
        yolo_show_optimization_info=debug_mode,
    )
    passed_flag = patch_ultralytics_attention()
    setup_success = env_config_success and passed_flag

    # Check if setup was successful
    if not setup_success:
        print("[WARNING] YOLO environment setup failed! Check optimization level and ultralytics installation.")

    return setup_success


class OptimizationLevelRegistry:
    """Registry for optimization level implementations."""
    
    # Registry mapping optimization levels to their descriptions and implementations
    REGISTRY = {}
    
    @classmethod
    def initialize_registry(cls):
        """Initialize the registry with optimization level implementations."""
        cls.REGISTRY = {
            0: {
                'description': 'Original method',
                'implementation': level_0_original_implementation
            },
            1: {
                'description': 'Three nested loops method',
                'implementation': level_1_three_nested_loops
            },
            2: {
                'description': 'Two nested loops method with torch matmul',
                'implementation': level_2_two_nested_loops_matmul
            },
            3: {
                'description': 'Two nested loops method with torch matmul and in-site edits',
                'implementation': level_3_two_loops_inplace
            }
        }
    
    @classmethod
    def get_method_description(cls, level: int) -> str:
        """Get description for an optimization level."""
        if level in cls.REGISTRY:
            return cls.REGISTRY[level]['description']
        return f'>>>>>>>> Method # {level} : Unknown optimization level.'
    
    @classmethod
    def print_optimization_info(cls, level: int) -> None:
        """Print information about the optimization method being used."""
        print(cls.get_method_description(level))
    
    @classmethod
    def get_implementation(cls, level: int) -> Callable:
        """Get implementation function for an optimization level."""
        if level in cls.REGISTRY:
            return cls.REGISTRY[level]['implementation']
        return None
    
    @classmethod
    def get_available_levels(cls) -> list:
        """Get list of available optimization levels."""
        return list(cls.REGISTRY.keys())


class AttentionProfiler:
    """Simple context manager for profiling attention operations."""
    
    def __init__(self, enabled: bool, level: int):
        self.enabled = enabled
        self.level = level
        self.profiler = MemoryProfiler(use_colors=False, use_symbols=False) if enabled else None
    
    def __enter__(self):
        if self.enabled:
            self.profiler.start(f"Block - OPTIMIZE LEVEL # {self.level}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.profiler.stop(mem_default_unit='MB')


# Individual optimization level implementations
def level_0_original_implementation(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    scale: float, 
    pe: Callable,
    B: int, 
    C: int, 
    H: int, 
    W: int
) -> torch.Tensor:
    """Original attention implementation."""
    attn = (q.transpose(-2, -1) @ k) * scale
    attn = attn.softmax(dim=-1)
    x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + pe(v.reshape(B, C, H, W))
    return x


def level_1_three_nested_loops(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    scale: float, 
    pe: Callable,
    B: int, 
    C: int, 
    H: int, 
    W: int,
    device: torch.device
) -> torch.Tensor:
    """Three nested loops implementation."""
    R = v.shape[2]
    M, N, K, P = q.shape
    out_dtype = torch.result_type(q, k)    
    var2 = torch.empty((M, N, R, P), dtype=out_dtype, device=device) 
    for i in range(M):
        for j in range(N):
            for t in range(P):
                s1 = ((q[i,j,:,t] @ k[i,j,:,:]) * scale).softmax(dim=0)
                s2 = v[i,j] @ s1
                var2[i,j,:,t] = s2

    return var2.view(B, C, H, W) + pe(v.reshape(B, C, H, W))


def level_2_two_nested_loops_matmul(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    scale: float, 
    pe: Callable,
    B: int, 
    C: int, 
    H: int, 
    W: int,
    device: torch.device
) -> torch.Tensor:
    """Two nested loops implementation with torch matmul."""
    R = v.shape[2]
    M, N, K, P = q.shape
    out_dtype = torch.result_type(q, k)
    var2 = torch.empty((M, N, R, P), dtype=out_dtype, device=device)
    
    # Implementation using PyTorch operations but optimized for GPU
    for i in range(M):
        for j in range(N):
            # Compute q @ k.T for all positions in one go
            attention_weights = torch.matmul(q[i, j].T, k[i, j])  # P x P
            attention_weights = attention_weights * scale
            
            # Apply softmax along appropriate dimension
            attention_probs = torch.nn.functional.softmax(attention_weights, dim=1)
            
            # Compute output for all R dimensions
            var2[i, j] = torch.matmul(v[i,j], attention_probs.T)  # R x P

    return var2.view(B, C, H, W) + pe(v.reshape(B, C, H, W))


def level_3_two_loops_inplace(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    scale: float, 
    pe: Callable,
    B: int, 
    C: int, 
    H: int, 
    W: int
) -> torch.Tensor:
    """Two nested loops implementation with in-place edits."""
    R = v.shape[2]
    M, N, K, P = q.shape
    
    x = pe(v.reshape(B, C, H, W)).reshape(B, N, R, P)            

    # Implementation using PyTorch operations but optimized for GPU
    for i in range(M):
        for j in range(N):
            # Compute q @ k.T for all positions in one go
            attention_weights = torch.matmul(q[i, j].T, k[i, j])  # P x P
            attention_weights = attention_weights * scale
            
            # Apply softmax along appropriate dimension
            attention_probs = torch.nn.functional.softmax(attention_weights, dim=1)
            
            # Compute output for all R dimensions
            x[i,j] += torch.matmul(v[i,j], attention_probs.T)  # R x P
            
    return x.reshape(B, C, H, W)


def create_optimized_forward():
    """Create the optimized forward method with multiple implementation levels."""
    
    def forward(self, x):
        """
        Forward pass of the Attention module with optimization levels.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        # Get configuration from environment variables
        optimize_level = int(os.environ.get('YOLO_OPTIMIZE_LEVEL', '0'))
        benchmark_enabled = os.environ.get('YOLO_BENCHMARK', 'false').lower() == 'true'
        show_info = os.environ.get('YOLO_SHOW_OPTIMIZATION_INFO', 'false').lower() == 'true'
        
        # Extract tensors
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        device = x.device
        
        # Print optimization method info if enabled
        if show_info:
            OptimizationLevelRegistry.print_optimization_info(optimize_level)
            
        # Validate optimization level
        available_levels = OptimizationLevelRegistry.get_available_levels()
        if optimize_level not in available_levels:
            raise ValueError(f'Invalid value for OPTIMIZE_LEVEL: {optimize_level}')
            
        # Get the appropriate implementation function
        implementation = OptimizationLevelRegistry.get_implementation(optimize_level)
        if implementation is None:
            raise ValueError(f'No implementation found for optimization level: {optimize_level}')
        
        # Profile and execute the optimization
        with AttentionProfiler(benchmark_enabled, optimize_level):
            # Call the appropriate implementation with the right parameters
            if optimize_level == 0:
                x = implementation(q, k, v, self.scale, self.pe, B, C, H, W)
            elif optimize_level == 1:
                x = implementation(q, k, v, self.scale, self.pe, B, C, H, W, device)
            elif optimize_level == 2:
                x = implementation(q, k, v, self.scale, self.pe, B, C, H, W, device)
            elif optimize_level == 3:
                x = implementation(q, k, v, self.scale, self.pe, B, C, H, W)
                
        # Apply final projection
        x = self.proj(x)
        return x
    
    return forward


def patch_ultralytics_attention(throw_error=False):
    """
    Patch the ultralytics Attention class to add optimization levels.
    
    This function modifies the forward method of the Attention class to support
    multiple optimization levels for benchmarking purposes. The function reads
    optimization settings directly from environment variables.
    
    Environment variables used (with defaults):
        - YOLO_OPTIMIZE_LEVEL: Optimization level (0-3), defaults to 0
        - YOLO_BENCHMARK: Enable memory profiling ('true' or 'false'), defaults to 'false'
        - YOLO_SHOW_OPTIMIZATION_INFO: Show optimization info ('true' or 'false'), defaults to 'false'
    
    Args:
        throw_error (bool): Whether to raise exceptions on failure. Defaults to False.
    """
    try:
        # Import the module
        from ultralytics.nn.modules.block import Attention
        
        # Store original forward method if not already stored
        if not hasattr(Attention, '_original_forward'):
            Attention._original_forward = Attention.forward
        
        # Apply the patched forward method
        Attention.forward = create_optimized_forward()
        
        print("[INFO] ✓ Ultralytics Attention class successfully patched!")
        return True
        
    except ImportError as e:
        error_msg = f"Failed to import ultralytics: {e}"
        print(f"[ERROR] {error_msg}")
        print("[ERROR] Make sure ultralytics is installed: pip install ultralytics")
        if throw_error:
            raise ImportError(error_msg)
        return False
    except Exception as e:
        error_msg = f"Failed to patch Attention class: {e}"
        print(f"[ERROR] {error_msg}")
        if throw_error:
            raise Exception(error_msg)
        return False


def unpatch_ultralytics_attention():
    """
    Restore the original Attention class forward method.
    """
    try:
        from ultralytics.nn.modules.block import Attention
        
        if hasattr(Attention, '_original_forward'):
            Attention.forward = Attention._original_forward
            delattr(Attention, '_original_forward')
            print("[INFO] Successfully restored original Attention class")
            return True
        else:
            print("[INFO] Attention class was not patched, nothing to restore")
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to restore Attention class: {e}")
        return False


# Initialize the registry after all functions are defined
OptimizationLevelRegistry.initialize_registry()
