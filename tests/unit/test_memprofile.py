import unittest
import torch
import gc
import time
from unittest.mock import patch, MagicMock, mock_open
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

# Import the module to test
from yolo_attnopt.memprofile import format_bytes, Colors, MemoryProfiler, profile_memory


class TestFormatBytes(unittest.TestCase):
    """Test the format_bytes function."""
    
    def test_basic_byte_conversions(self):
        """Test basic byte unit conversions."""
        # Test bytes
        self.assertEqual(format_bytes(512), "512.00 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        
        # Test kilobytes
        self.assertEqual(format_bytes(1024 * 1024), "1.00 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 2.5), "2.50 MB")
        
        # Test megabytes
        self.assertEqual(format_bytes(1024 * 1024 * 1024), "1.00 GB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 5.75), "5.75 GB")
        
        # Test gigabytes
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 1024), "1.00 TB")
    
    def test_negative_values(self):
        """Test handling of negative values."""
        self.assertEqual(format_bytes(-512), "-512.00 B")
        self.assertEqual(format_bytes(-1024), "-1.00 KB")
        self.assertEqual(format_bytes(-1024 * 1024), "-1.00 MB")
    
    def test_default_unit_parameter(self):
        """Test the default_unit parameter."""
        # Test forcing specific units
        self.assertEqual(format_bytes(2048, default_unit="KB"), "2.00 KB")
        self.assertEqual(format_bytes(1024 * 1024 * 3, default_unit="MB"), "3.00 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 2, default_unit="GB"), "2.00 GB")
        
        # Test invalid default_unit (should fall back to auto-scaling)
        self.assertEqual(format_bytes(1024, default_unit="INVALID"), "1.00 KB")
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Zero bytes
        self.assertEqual(format_bytes(0), "0.00 B")
        
        # Very small values
        self.assertEqual(format_bytes(0.5), "0.50 B")
        
        # Very large values - should show in TB but not cap
        large_value = 1024**5  # Petabyte range
        self.assertEqual(format_bytes(large_value), "1024.00 TB")  # Should show actual value in TB
    
    def test_precision(self):
        """Test decimal precision."""
        # Test that we get proper decimal places
        result = format_bytes(1536)  # 1.5 KB
        self.assertEqual(result, "1.50 KB")


class TestColors(unittest.TestCase):
    """Test the Colors class."""
    
    def test_color_constants_exist(self):
        """Test that all expected color constants are defined."""
        expected_colors = [
            'BLUE', 'GREEN', 'RED', 'YELLOW', 'PURPLE', 
            'CYAN', 'ENDC', 'BOLD'
        ]
        
        for color in expected_colors:
            self.assertTrue(hasattr(Colors, color))
            self.assertIsInstance(getattr(Colors, color), str)
    
    def test_color_codes_format(self):
        """Test that color codes are properly formatted."""
        # ANSI color codes should start with escape character
        self.assertTrue(Colors.BLUE.startswith('\033'))
        self.assertTrue(Colors.GREEN.startswith('\033'))
        self.assertTrue(Colors.RED.startswith('\033'))
        self.assertTrue(Colors.ENDC.startswith('\033'))
    
    def test_color_codes_are_strings(self):
        """Test that all color codes are strings."""
        for attr_name in dir(Colors):
            if not attr_name.startswith('_'):
                attr_value = getattr(Colors, attr_name)
                self.assertIsInstance(attr_value, str)


class TestMemoryProfiler(unittest.TestCase):
    """Test the MemoryProfiler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock CUDA device and operations
        self.mock_device = MagicMock()
        self.mock_cuda = MagicMock()
        
    def test_initialization(self):
        """Test MemoryProfiler initialization."""
        # Test default initialization
        with patch('torch.cuda.current_device', return_value=0):
            with patch('torch.device') as mock_device_class:
                mock_device_instance = MagicMock()
                mock_device_class.return_value = mock_device_instance
                
                profiler = MemoryProfiler()
                
                mock_device_class.assert_called_with('cuda:0')
                self.assertEqual(profiler.device, mock_device_instance)
                self.assertEqual(profiler.name, "Memory Profile Session")
                self.assertTrue(profiler.use_colors)
                self.assertTrue(profiler.use_symbols)
                self.assertIsNone(profiler.start_time)
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        custom_device = MagicMock()
        profiler = MemoryProfiler(
            device=custom_device,
            name="Custom Session",
            use_colors=False,
            use_symbols=False
        )
        
        self.assertEqual(profiler.device, custom_device)
        self.assertEqual(profiler.name, "Custom Session")
        self.assertFalse(profiler.use_colors)
        self.assertFalse(profiler.use_symbols)
    
    def test_reset(self):
        """Test the reset method."""
        profiler = MemoryProfiler()
        profiler.initial_allocated = 1000
        profiler.peak_allocated = 2000
        
        profiler.reset()
        
        self.assertEqual(profiler.initial_allocated, 0)
        self.assertEqual(profiler.peak_allocated, 0)
    
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.reset_peak_memory_stats')
    @patch('gc.collect')
    def test_clean_environment(self, mock_gc, mock_reset, mock_sync, mock_empty):
        """Test the _clean_environment method."""
        profiler = MemoryProfiler()
        profiler._clean_environment()
        
        mock_gc.assert_called_once()
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()
        mock_reset.assert_called_once()
    
    @patch('torch.cuda.memory_allocated')
    @patch('time.time')
    def test_start(self, mock_time, mock_allocated):
        """Test the start method."""
        mock_time.side_effect = [100.0, 100.1]  # start_time and current time
        mock_allocated.return_value = 500
        
        profiler = MemoryProfiler()
        
        with patch.object(profiler, '_clean_environment') as mock_clean:
            result = profiler.start("Test Session")
            
            mock_clean.assert_called_once()
            self.assertEqual(profiler.start_time, 100.0)
            self.assertEqual(profiler.initial_allocated, 500)
            self.assertEqual(profiler.name, "Test Session")
            self.assertEqual(result, profiler)
    
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.max_memory_allocated')
    @patch('time.time')
    def test_stop_with_colors_and_symbols(self, mock_time, mock_max_alloc, mock_sync):
        """Test the stop method with colors and symbols enabled."""
        mock_time.side_effect = [100.0, 105.5]  # start_time and current time
        mock_max_alloc.return_value = 1500
        
        profiler = MemoryProfiler(name="Test Session")  # Set the name during initialization
        profiler.start_time = 100.0
        profiler.initial_allocated = 500
        
        # Capture stdout to test output
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch.object(profiler, '_clean_environment') as mock_clean:
                result = profiler.stop()
                
                mock_sync.assert_called_once()
                mock_max_alloc.assert_called_once()
                mock_clean.assert_called_once()
                
                output = mock_stdout.getvalue()
                self.assertIn("📺 PROF", output)
                # Check for the name content rather than exact string due to color codes
                self.assertIn("Test Session", output)  # Should be present with ANSI color codes
                self.assertIn("📌", output)
                self.assertIn("📈", output)
                self.assertIn("⏱️", output)
                self.assertEqual(result, profiler)
    
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.max_memory_allocated')
    @patch('time.time')
    def test_stop_without_colors_or_symbols(self, mock_time, mock_max_alloc, mock_sync):
        """Test the stop method with colors and symbols disabled."""
        mock_time.side_effect = [100.0, 102.5]
        mock_max_alloc.return_value = 1000
        
        profiler = MemoryProfiler(use_colors=False, use_symbols=False)
        profiler.start_time = 100.0
        profiler.initial_allocated = 500
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch.object(profiler, '_clean_environment') as mock_clean:
                result = profiler.stop()
                
                output = mock_stdout.getvalue()
                self.assertIn("📺 PROF", output)
                self.assertNotIn("📌", output)
                self.assertNotIn("📈", output)
                self.assertNotIn("⏱️", output)
    
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.max_memory_allocated')
    @patch('time.time')
    def test_stop_with_default_unit(self, mock_time, mock_max_alloc, mock_sync):
        """Test the stop method with default_unit parameter."""
        mock_time.side_effect = [100.0, 101.0]
        mock_max_alloc.return_value = 2048  # 2KB
        
        profiler = MemoryProfiler()
        profiler.start_time = 100.0
        profiler.initial_allocated = 0
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch.object(profiler, '_clean_environment'):
                profiler.stop(mem_default_unit="KB")
                
                output = mock_stdout.getvalue()
                self.assertIn("2.00 KB", output)
    
    def test_context_manager_enter(self):
        """Test the __enter__ method for context manager."""
        profiler = MemoryProfiler()
        
        with patch.object(profiler, 'start') as mock_start:
            mock_start.return_value = profiler  # Ensure start returns the profiler
            result = profiler.__enter__()
            
            mock_start.assert_called_once_with(profiler.name)
            self.assertEqual(result, profiler)
    
    def test_context_manager_exit(self):
        """Test the __exit__ method for context manager."""
        profiler = MemoryProfiler()
        
        with patch.object(profiler, 'stop') as mock_stop:
            result = profiler.__exit__(None, None, None)
            
            mock_stop.assert_called_once()
            self.assertFalse(result)  # Should return False
    
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.max_memory_allocated')
    @patch('time.time')
    def test_context_manager_full_workflow(self, mock_time, mock_max_alloc, mock_sync):
        """Test the full context manager workflow."""
        mock_time.side_effect = [100.0, 103.0]
        mock_max_alloc.return_value = 1500
        
        with patch('torch.cuda.memory_allocated', return_value=500):
            with patch.object(MemoryProfiler, '_clean_environment'):
                with patch('sys.stdout', new_callable=io.StringIO):
                    with MemoryProfiler(name="Context Test") as profiler:
                        self.assertIsNotNone(profiler.start_time)
                        self.assertEqual(profiler.name, "Context Test")


class TestProfileMemoryDecorator(unittest.TestCase):
    """Test the profile_memory decorator."""
    
    def test_decorator_without_params(self):
        """Test decorator used without parameters."""
        @profile_memory
        def test_function():
            return "test result"
        
        with patch('sys.stdout', new_callable=io.StringIO):
            result = test_function()
            
        self.assertEqual(result, "test result")
    
    def test_decorator_with_custom_name(self):
        """Test decorator with custom name parameter."""
        @profile_memory(name="Custom Function Name")
        def test_function():
            return "test result"
        
        with patch('sys.stdout', new_callable=io.StringIO):
            result = test_function()
            
        self.assertEqual(result, "test result")
    
    def test_decorator_with_options(self):
        """Test decorator with color and symbol options."""
        @profile_memory(use_colors=False, use_symbols=False)
        def test_function():
            return "test result"
        
        with patch('sys.stdout', new_callable=io.StringIO):
            result = test_function()
            
        self.assertEqual(result, "test result")
    
    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""
        @profile_memory
        def original_function():
            """Original docstring."""
            pass
        
        self.assertEqual(original_function.__name__, "original_function")
        self.assertEqual(original_function.__doc__, "Original docstring.")
    
    def test_decorator_with_args_and_kwargs(self):
        """Test decorator with function that takes arguments."""
        @profile_memory
        def function_with_args(a, b, c=3):
            return a + b + c
        
        with patch('sys.stdout', new_callable=io.StringIO):
            result = function_with_args(1, 2, c=4)
            
        self.assertEqual(result, 7)
    
    def test_decorator_with_exception(self):
        """Test decorator behavior when function raises exception."""
        @profile_memory
        def failing_function():
            raise ValueError("Test exception")
        
        with patch('sys.stdout', new_callable=io.StringIO):
            with self.assertRaises(ValueError):
                failing_function()
    
    def test_decorator_direct_call(self):
        """Test calling decorator directly (not as @decorator)."""
        def regular_function():
            return "direct call result"
        
        decorated = profile_memory(regular_function)
        
        with patch('sys.stdout', new_callable=io.StringIO):
            result = decorated()
            
        self.assertEqual(result, "direct call result")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.max_memory_allocated')
    @patch('torch.cuda.memory_allocated')
    @patch('time.time')
    def test_complete_profiling_workflow(self, mock_time, mock_allocated, mock_max_alloc, mock_sync):
        """Test complete profiling workflow using context manager."""
        mock_time.side_effect = [100.0, 105.0]
        mock_allocated.return_value = 500
        mock_max_alloc.return_value = 2000
        
        with patch.object(MemoryProfiler, '_clean_environment'):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                with MemoryProfiler(name="Integration Test") as profiler:
                    # Simulate some memory allocation
                    pass
                
                output = mock_stdout.getvalue()
                # The name "Integration Test" should be in the output
                self.assertIn("📺 PROF", output)
                # Check for content rather than exact string due to ANSI color codes
                self.assertIn("Integration Test", output)  # Should be present with ANSI color codes
                # (2000-500) = 1500 bytes = 1.46484375 KB ≈ 1.46 KB
                self.assertIn("1.46 KB", output)  # Corrected precision
                self.assertIn("5.0000s", output)
    
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.max_memory_allocated')
    @patch('torch.cuda.memory_allocated')
    @patch('time.time')
    def test_decorator_integration(self, mock_time, mock_allocated, mock_max_alloc, mock_sync):
        """Test decorator integration with mocked CUDA operations."""
        mock_time.side_effect = [100.0, 102.0]
        mock_allocated.return_value = 1000
        mock_max_alloc.return_value = 3000
        
        with patch.object(MemoryProfiler, '_clean_environment'):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                @profile_memory(name="Decorated Function Test")
                def sample_function(x):
                    return x * 2
                
                result = sample_function(5)
                
                self.assertEqual(result, 10)
                
                output = mock_stdout.getvalue()
                self.assertIn("📺 PROF", output)
                # The decorated function should show the custom name
                # Check for content rather than exact string due to ANSI color codes
                self.assertIn("Decorated Function Test", output)  # Should be present with ANSI color codes
                self.assertIn("1.95 KB", output)  # (3000-1000) = 2000 bytes = 1.95KB
    
    def test_multiple_profiler_instances(self):
        """Test multiple profiler instances running concurrently."""
        with patch('torch.cuda.current_device', return_value=0):
            with patch('torch.device') as mock_device_class:
                mock_device_class.return_value = MagicMock()
                
                profiler1 = MemoryProfiler(name="Profiler 1")
                profiler2 = MemoryProfiler(name="Profiler 2")
                
                self.assertEqual(profiler1.name, "Profiler 1")
                self.assertEqual(profiler2.name, "Profiler 2")
                self.assertNotEqual(profiler1, profiler2)
    
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.max_memory_allocated')
    @patch('torch.cuda.memory_allocated')
    @patch('time.time')
    def test_memory_calculation_accuracy(self, mock_time, mock_allocated, mock_max_alloc, mock_sync):
        """Test that memory calculations are accurate."""
        mock_time.side_effect = [100.0, 101.0]
        mock_allocated.return_value = 1024 * 100  # 100KB initial
        mock_max_alloc.return_value = 1024 * 500  # 500KB peak
        
        with patch.object(MemoryProfiler, '_clean_environment'):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                profiler = MemoryProfiler()
                profiler.start()
                profiler.stop()
                
                output = mock_stdout.getvalue()
                # Should show 400KB (500KB - 100KB)
                self.assertIn("400.00 KB", output)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTest(unittest.makeSuite(TestFormatBytes))
    suite.addTest(unittest.makeSuite(TestColors))
    suite.addTest(unittest.makeSuite(TestMemoryProfiler))
    suite.addTest(unittest.makeSuite(TestProfileMemoryDecorator))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
