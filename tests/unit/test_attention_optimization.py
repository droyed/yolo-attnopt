"""
Comprehensive unit tests for attention_optimization.py

This module contains extensive unit tests for all functions and classes
in the attention optimization script, covering configuration, registry,
profiling, optimization implementations, and patching functionality.
"""

import pytest
import torch
import os
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

# Add the parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo_attnopt.attention_optimization import (
    configure_yolo_environment,
    setup_optimized_yolo_environ,
    OptimizationLevelRegistry,
    AttentionProfiler,
    level_0_original_implementation,
    level_1_three_nested_loops,
    level_2_two_nested_loops_matmul,
    level_3_two_loops_inplace,
    create_optimized_forward,
    patch_ultralytics_attention,
    unpatch_ultralytics_attention
)


class TestConfigureYoloEnvironment:
    """Test suite for configure_yolo_environment function."""
    
    def test_valid_optimization_level(self):
        """Test successful configuration with valid optimization level."""
        # Setup
        level = 2
        benchmark = True
        show_info = False
        
        # Execute
        result = configure_yolo_environment(level, benchmark, show_info)
        
        # Verify
        assert result is True
        assert os.environ.get('YOLO_OPTIMIZE_LEVEL') == '2'
        assert os.environ.get('YOLO_BENCHMARK') == 'true'
        assert os.environ.get('YOLO_SHOW_OPTIMIZATION_INFO') == 'false'
        
        # Cleanup
        for key in ['YOLO_OPTIMIZE_LEVEL', 'YOLO_BENCHMARK', 'YOLO_SHOW_OPTIMIZATION_INFO']:
            os.environ.pop(key, None)
    
    def test_invalid_optimization_level(self):
        """Test failure with invalid optimization level."""
        # Setup
        level = 99
        benchmark = False
        show_info = False
        
        # Execute
        result = configure_yolo_environment(level, benchmark, show_info)
        
        # Verify
        assert result is False
    
    def test_all_boolean_combinations(self):
        """Test all combinations of boolean parameters."""
        test_cases = [
            (1, False, False),
            (1, True, False),
            (1, False, True),
            (1, True, True),
        ]
        
        for level, benchmark, show_info in test_cases:
            result = configure_yolo_environment(level, benchmark, show_info)
            assert result is True
            
            expected_benchmark = 'true' if benchmark else 'false'
            expected_show_info = 'true' if show_info else 'false'
            
            assert os.environ.get('YOLO_BENCHMARK') == expected_benchmark
            assert os.environ.get('YOLO_SHOW_OPTIMIZATION_INFO') == expected_show_info
            
            # Cleanup
            for key in ['YOLO_OPTIMIZE_LEVEL', 'YOLO_BENCHMARK', 'YOLO_SHOW_OPTIMIZATION_INFO']:
                os.environ.pop(key, None)
    
    def test_optimization_level_zero(self):
        """Test configuration with level 0 (original method)."""
        result = configure_yolo_environment(0)
        assert result is True
        assert os.environ.get('YOLO_OPTIMIZE_LEVEL') == '0'


class TestSetupOptimizedYoloEnvironment:
    """Test suite for setup_optimized_yolo_environ function."""
    
    @patch('yolo_attnopt.attention_optimization.configure_yolo_environment')
    @patch('yolo_attnopt.attention_optimization.patch_ultralytics_attention')
    def test_successful_setup(self, mock_patch, mock_configure):
        """Test successful setup with all components working."""
        # Setup
        mock_configure.return_value = True
        mock_patch.return_value = True
        
        # Execute
        result = setup_optimized_yolo_environ(2, debug_mode=True)
        
        # Verify
        assert result is True
        mock_configure.assert_called_once_with(
            yolo_optimize_level=2,
            yolo_benchmark=True,
            yolo_show_optimization_info=True
        )
        mock_patch.assert_called_once()
    
    @patch('yolo_attnopt.attention_optimization.configure_yolo_environment')
    @patch('yolo_attnopt.attention_optimization.patch_ultralytics_attention')
    def test_failed_configuration(self, mock_patch, mock_configure):
        """Test failure when configuration fails."""
        # Setup
        mock_configure.return_value = False
        mock_patch.return_value = True
        
        # Execute
        result = setup_optimized_yolo_environ(99, debug_mode=False)
        
        # Verify
        assert result is False
    
    @patch('yolo_attnopt.attention_optimization.configure_yolo_environment')
    @patch('yolo_attnopt.attention_optimization.patch_ultralytics_attention')
    def test_failed_patching(self, mock_patch, mock_configure):
        """Test failure when patching fails."""
        # Setup
        mock_configure.return_value = True
        mock_patch.return_value = False
        
        # Execute
        result = setup_optimized_yolo_environ(1)
        
        # Verify
        assert result is False


class TestOptimizationLevelRegistry:
    """Test suite for OptimizationLevelRegistry class."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Reset registry to ensure clean state
        OptimizationLevelRegistry.REGISTRY = {}
    
    def test_initialize_registry(self):
        """Test registry initialization."""
        # Execute
        OptimizationLevelRegistry.initialize_registry()
        
        # Verify
        assert len(OptimizationLevelRegistry.REGISTRY) == 4
        for level in [0, 1, 2, 3]:
            assert level in OptimizationLevelRegistry.REGISTRY
            assert 'description' in OptimizationLevelRegistry.REGISTRY[level]
            assert 'implementation' in OptimizationLevelRegistry.REGISTRY[level]
    
    def test_get_method_description_valid(self):
        """Test getting description for valid optimization levels."""
        # Setup
        OptimizationLevelRegistry.initialize_registry()
        
        # Execute & Verify
        assert "Original method" in OptimizationLevelRegistry.get_method_description(0)
        assert "Three nested loops method" in OptimizationLevelRegistry.get_method_description(1)
        assert "Two nested loops method with torch matmul" in OptimizationLevelRegistry.get_method_description(2)
        assert "Two nested loops method with torch matmul and in-site edits" in OptimizationLevelRegistry.get_method_description(3)
    
    def test_get_method_description_invalid(self):
        """Test getting description for invalid optimization level."""
        # Setup
        OptimizationLevelRegistry.initialize_registry()
        
        # Execute
        desc = OptimizationLevelRegistry.get_method_description(99)
        
        # Verify
        assert "Unknown optimization level" in desc
    
    def test_get_implementation_valid(self):
        """Test getting implementation for valid levels."""
        # Setup
        OptimizationLevelRegistry.initialize_registry()
        
        # Execute & Verify
        assert OptimizationLevelRegistry.get_implementation(0) == level_0_original_implementation
        assert OptimizationLevelRegistry.get_implementation(1) == level_1_three_nested_loops
        assert OptimizationLevelRegistry.get_implementation(2) == level_2_two_nested_loops_matmul
        assert OptimizationLevelRegistry.get_implementation(3) == level_3_two_loops_inplace
    
    def test_get_implementation_invalid(self):
        """Test getting implementation for invalid level."""
        # Setup
        OptimizationLevelRegistry.initialize_registry()
        
        # Execute
        impl = OptimizationLevelRegistry.get_implementation(99)
        
        # Verify
        assert impl is None
    
    def test_get_available_levels(self):
        """Test getting list of available levels."""
        # Setup
        OptimizationLevelRegistry.initialize_registry()
        
        # Execute
        levels = OptimizationLevelRegistry.get_available_levels()
        
        # Verify
        assert levels == [0, 1, 2, 3]
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_print_optimization_info(self, mock_stdout):
        """Test printing optimization information."""
        # Setup
        OptimizationLevelRegistry.initialize_registry()
        
        # Execute
        OptimizationLevelRegistry.print_optimization_info(0)
        
        # Verify
        output = mock_stdout.getvalue()
        assert "Original method" in output


class TestAttentionProfiler:
    """Test suite for AttentionProfiler class."""
    
    @patch('yolo_attnopt.attention_optimization.MemoryProfiler')
    def test_profiler_enabled(self, mock_profiler_class):
        """Test profiler when enabled."""
        # Setup
        mock_profiler_instance = Mock()
        mock_profiler_class.return_value = mock_profiler_instance
        
        # Execute
        with AttentionProfiler(enabled=True, level=2) as profiler:
            pass
        
        # Verify
        mock_profiler_class.assert_called_once()
        mock_profiler_instance.start.assert_called_once_with("Block - OPTIMIZE LEVEL # 2")
        mock_profiler_instance.stop.assert_called_once_with(mem_default_unit='MB')
    
    def test_profiler_disabled(self):
        """Test profiler when disabled."""
        # Execute
        with AttentionProfiler(enabled=False, level=1) as profiler:
            pass
        
        # Verify - should not crash and profiler should be None
        assert profiler.profiler is None


class TestOptimizationImplementations:
    """Test suite for optimization level implementations."""
    
    def setup_method(self):
        """Setup common test fixtures."""
        # Create sample tensors for testing
        self.B, self.C, self.H, self.W = 1, 64, 8, 8  # Batch, Channels, Height, Width
        self.N = self.H * self.W  # Number of spatial positions
        self.key_dim = 32
        self.head_dim = 32
        self.num_heads = 2
        
        # Create test tensors
        self.q = torch.randn(self.B, self.num_heads, self.key_dim, self.N)
        self.k = torch.randn(self.B, self.num_heads, self.key_dim, self.N)
        self.v = torch.randn(self.B, self.num_heads, self.head_dim, self.N)
        self.scale = 0.125
        self.device = torch.device('cpu')
        
        # Mock position encoding function
        self.pe = Mock(return_value=torch.zeros_like(self.v.reshape(self.B, self.C, self.H, self.W)))
    
    def test_level_0_original_implementation(self):
        """Test original attention implementation."""
        # Execute
        result = level_0_original_implementation(
            self.q, self.k, self.v, self.scale, self.pe, 
            self.B, self.C, self.H, self.W
        )
        
        # Verify
        assert result.shape == (self.B, self.C, self.H, self.W)
        assert result.dtype == torch.result_type(self.q, self.k)
        # Check that pe was called with the correct tensor shape
        self.pe.assert_called_once()
        call_args = self.pe.call_args[0][0]
        assert call_args.shape == (self.B, self.C, self.H, self.W)
    
    def test_level_1_three_nested_loops(self):
        """Test three nested loops implementation."""
        # Execute
        result = level_1_three_nested_loops(
            self.q, self.k, self.v, self.scale, self.pe,
            self.B, self.C, self.H, self.W, self.device
        )
        
        # Verify
        assert result.shape == (self.B, self.C, self.H, self.W)
        # Check that pe was called with the correct tensor shape
        self.pe.assert_called_once()
        call_args = self.pe.call_args[0][0]
        assert call_args.shape == (self.B, self.C, self.H, self.W)
    
    def test_level_2_two_nested_loops_matmul(self):
        """Test two nested loops with matmul implementation."""
        # Execute
        result = level_2_two_nested_loops_matmul(
            self.q, self.k, self.v, self.scale, self.pe,
            self.B, self.C, self.H, self.W, self.device
        )
        
        # Verify
        assert result.shape == (self.B, self.C, self.H, self.W)
        # Check that pe was called with the correct tensor shape
        self.pe.assert_called_once()
        call_args = self.pe.call_args[0][0]
        assert call_args.shape == (self.B, self.C, self.H, self.W)
    
    def test_level_3_two_loops_inplace(self):
        """Test two loops with in-place operations."""
        # Execute
        result = level_3_two_loops_inplace(
            self.q, self.k, self.v, self.scale, self.pe,
            self.B, self.C, self.H, self.W
        )
        
        # Verify
        assert result.shape == (self.B, self.C, self.H, self.W)
    
    def test_different_tensor_shapes(self):
        """Test implementations with different tensor shapes."""
        test_shapes = [
            (1, 32, 4, 4),  # Smaller
            (2, 128, 16, 16),  # Larger
            (1, 64, 1, 1),  # Single position
        ]
        
        for B, C, H, W in test_shapes:
            N = H * W
            key_dim = 16
            head_dim = 32
            num_heads = 2
            
            # Ensure C matches num_heads * head_dim
            expected_C = num_heads * head_dim
            
            q = torch.randn(B, num_heads, key_dim, N)
            k = torch.randn(B, num_heads, key_dim, N)
            v = torch.randn(B, num_heads, head_dim, N)
            pe = Mock(return_value=torch.zeros(B, expected_C, H, W))
            
            # Test all implementations
            result = level_0_original_implementation(q, k, v, 0.125, pe, B, expected_C, H, W)
            assert result.shape == (B, expected_C, H, W)
    
    def test_different_dtypes(self):
        """Test implementations with different data types."""
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            q = self.q.to(dtype)
            k = self.k.to(dtype)
            v = self.v.to(dtype)
            
            result = level_0_original_implementation(
                q, k, v, self.scale, self.pe, 
                self.B, self.C, self.H, self.W
            )
            
            assert result.shape == (self.B, self.C, self.H, self.W)
    
    def test_scale_parameter_effect(self):
        """Test effect of different scale values."""
        scales = [0.0, 0.1, 1.0, 2.0]
        
        for scale in scales:
            result1 = level_0_original_implementation(
                self.q, self.k, self.v, scale, self.pe,
                self.B, self.C, self.H, self.W
            )
            result2 = level_0_original_implementation(
                self.q, self.k, self.v, scale, self.pe,
                self.B, self.C, self.H, self.W
            )
            
            # Results should be deterministic
            torch.testing.assert_close(result1, result2)


class TestCreateOptimizedForward:
    """Test suite for create_optimized_forward function."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Initialize registry
        OptimizationLevelRegistry.initialize_registry()
    
    @patch.dict(os.environ, {
        'YOLO_OPTIMIZE_LEVEL': '0',
        'YOLO_BENCHMARK': 'false',
        'YOLO_SHOW_OPTIMIZATION_INFO': 'false'
    })
    def test_forward_method_creation(self):
        """Test creation of optimized forward method."""
        # Setup
        mock_attention = Mock()
        mock_attention.qkv = Mock(return_value=torch.randn(1, 3, 96, 64))
        mock_attention.num_heads = 3
        mock_attention.key_dim = 32
        mock_attention.head_dim = 32
        mock_attention.scale = 0.125
        mock_attention.pe = Mock(return_value=torch.zeros(1, 96, 8, 8))
        mock_attention.proj = Mock(return_value=torch.randn(1, 96, 8, 8))
        
        # Execute
        forward_method = create_optimized_forward()
        result = forward_method(mock_attention, torch.randn(1, 96, 8, 8))
        
        # Verify
        assert result is not None
        assert result.shape == (1, 96, 8, 8)
        mock_attention.qkv.assert_called_once()
        mock_attention.proj.assert_called_once()
    
    @patch.dict(os.environ, {
        'YOLO_OPTIMIZE_LEVEL': '1',
        'YOLO_BENCHMARK': 'true',
        'YOLO_SHOW_OPTIMIZATION_INFO': 'true'
    })
    @patch('yolo_attnopt.attention_optimization.AttentionProfiler')
    def test_forward_with_benchmarking(self, mock_profiler):
        """Test forward method with benchmarking enabled."""
        # Setup
        mock_attention = Mock()
        mock_attention.qkv = Mock(return_value=torch.randn(1, 3, 96, 64))
        mock_attention.num_heads = 3
        mock_attention.key_dim = 32
        mock_attention.head_dim = 32
        mock_attention.scale = 0.125
        mock_attention.pe = Mock(return_value=torch.zeros(1, 96, 8, 8))
        mock_attention.proj = Mock(return_value=torch.randn(1, 96, 8, 8))
        
        mock_profiler_instance = Mock()
        mock_profiler.return_value.__enter__ = Mock(return_value=mock_profiler_instance)
        mock_profiler.return_value.__exit__ = Mock(return_value=None)
        
        # Execute
        forward_method = create_optimized_forward()
        forward_method(mock_attention, torch.randn(1, 96, 8, 8))
        
        # Verify
        mock_profiler.assert_called_once()
    
    @patch.dict(os.environ, {'YOLO_OPTIMIZE_LEVEL': '99'})
    def test_forward_with_invalid_level(self):
        """Test forward method with invalid optimization level."""
        # Setup
        mock_attention = Mock()
        mock_attention.num_heads = 3
        mock_attention.key_dim = 32
        mock_attention.head_dim = 32
        mock_attention.qkv = Mock(return_value=torch.randn(1, 3, 96, 64))
        
        # Execute & Verify
        forward_method = create_optimized_forward()
        with pytest.raises(ValueError, match="Invalid value for OPTIMIZE_LEVEL"):
            forward_method(mock_attention, torch.randn(1, 96, 8, 8))


class TestPatchFunctions:
    """Test suite for patching and unpatching functions."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Ensure clean state
        if hasattr(Mock, '_original_forward'):
            delattr(Mock, '_original_forward')
    
    @patch('yolo_attnopt.attention_optimization.create_optimized_forward')
    @patch('ultralytics.nn.modules.block.Attention')
    def test_patch_ultralytics_attention_success(self, mock_attention_class, mock_create_forward):
        """Test successful patching of Attention class."""
        # Setup
        mock_original_forward = Mock()
        mock_attention_class.forward = mock_original_forward
        mock_forward_method = Mock()
        mock_create_forward.return_value = mock_forward_method
        
        # Execute
        result = patch_ultralytics_attention()
        
        # Verify
        assert result is True
        assert hasattr(mock_attention_class, '_original_forward')
        assert mock_attention_class.forward == mock_forward_method
    
    @patch('yolo_attnopt.attention_optimization.create_optimized_forward')
    @patch('ultralytics.nn.modules.block.Attention')
    def test_patch_ultralytics_attention_already_patched(self, mock_attention_class, mock_create_forward):
        """Test patching when already patched."""
        # Setup
        mock_original_forward = Mock()
        mock_attention_class.forward = mock_original_forward
        mock_attention_class._original_forward = Mock()  # Already patched
        mock_forward_method = Mock()
        mock_create_forward.return_value = mock_forward_method
        
        # Execute
        result = patch_ultralytics_attention()
        
        # Verify
        assert result is True
        # Should keep the original _original_forward, not overwrite it
        mock_create_forward.assert_called_once()
    
    @patch('yolo_attnopt.attention_optimization.create_optimized_forward')
    @patch('builtins.__import__', side_effect=ImportError("Module not found"))
    def test_patch_ultralytics_attention_import_error(self, mock_import, mock_create_forward):
        """Test patching with import error."""
        # Execute
        result = patch_ultralytics_attention()
        
        # Verify
        assert result is False
    
    @patch('yolo_attnopt.attention_optimization.create_optimized_forward')
    @patch('builtins.__import__', side_effect=ImportError("Module not found"))
    def test_patch_ultralytics_attention_throw_error(self, mock_import, mock_create_forward):
        """Test patching that throws error when requested."""
        # Execute & Verify
        with pytest.raises(ImportError):
            patch_ultralytics_attention(throw_error=True)
    
    @patch('ultralytics.nn.modules.block.Attention')
    def test_unpatch_ultralytics_attention_success(self, mock_attention_class):
        """Test successful unpatching."""
        # Setup
        mock_original_forward = Mock()
        mock_attention_class._original_forward = mock_original_forward
        mock_attention_class.forward = Mock()
        
        # Execute
        result = unpatch_ultralytics_attention()
        
        # Verify
        assert result is True
        assert mock_attention_class.forward == mock_original_forward
        assert not hasattr(mock_attention_class, '_original_forward')
    
    @patch('ultralytics.nn.modules.block.Attention')
    def test_unpatch_ultralytics_attention_not_patched(self, mock_attention_class):
        """Test unpatching when not previously patched."""
        # Setup
        if hasattr(mock_attention_class, '_original_forward'):
            delattr(mock_attention_class, '_original_forward')
        
        # Execute
        result = unpatch_ultralytics_attention()
        
        # Verify
        assert result is True


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    def setup_method(self):
        """Setup for integration tests."""
        # Initialize registry
        OptimizationLevelRegistry.initialize_registry()
    
    def test_complete_configuration_workflow(self):
        """Test complete configuration and setup workflow."""
        # Execute configuration
        config_result = configure_yolo_environment(2, True, True)
        assert config_result is True
        
        # Execute setup
        setup_result = setup_optimized_yolo_environ(1)
        assert setup_result is True
        
        # Verify environment variables
        assert os.environ.get('YOLO_OPTIMIZE_LEVEL') == '1'
        assert os.environ.get('YOLO_BENCHMARK') == 'false'
        assert os.environ.get('YOLO_SHOW_OPTIMIZATION_INFO') == 'false'
    
    def test_registry_persistence_across_functions(self):
        """Test that registry state persists across different function calls."""
        # Verify registry is initialized
        assert len(OptimizationLevelRegistry.REGISTRY) == 4
        
        # Test registry methods still work
        desc = OptimizationLevelRegistry.get_method_description(0)
        assert "Original method" in desc
        
        impl = OptimizationLevelRegistry.get_implementation(1)
        assert impl == level_1_three_nested_loops
        
        levels = OptimizationLevelRegistry.get_available_levels()
        assert levels == [0, 1, 2, 3]
    
    def test_error_handling_workflow(self):
        """Test error handling across the module."""
        # Test invalid optimization level in configuration
        result = configure_yolo_environment(999)
        assert result is False
        
        # Test invalid level in registry
        desc = OptimizationLevelRegistry.get_method_description(999)
        assert "Unknown optimization level" in desc
        
        impl = OptimizationLevelRegistry.get_implementation(999)
        assert impl is None


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])
