#!/usr/bin/env python3
"""
Unit tests for YOLO Performance Analyzer

This module contains comprehensive unit tests for all functions and methods
in the yolo_performance_analyzer.py script, including class-based API,
backward compatibility functions, and CLI functionality.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import argparse
from io import StringIO

# Import the modules to test
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_tools.yolo_performance_analyzer import (
    PerformanceDataProcessor,
    PerformanceVisualizer,
    PerformanceAnalyzer,
    get_model_size_order,
    interactive_select,
    process_performance_metrics_csv,
    create_performance_comparison_plots,
    verify_output_image,
    analyze_performance_metrics,
    main
)


class TestPerformanceDataProcessor(unittest.TestCase):
    """Test cases for PerformanceDataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = PerformanceDataProcessor()
        
        # Create sample CSV data for testing
        self.sample_csv_data = """Model,L0_Memory_MB,L0_Time_s,L1_Memory_MB,L1_Time_s,L2_Memory_MB,L2_Time_s,L3_Memory_MB,L3_Time_s
yolo11n.pt,45.2,0.123,42.1,0.145,40.3,0.167,38.9,0.189
yolo11s.pt,89.7,0.234,85.2,0.256,82.1,0.278,79.8,0.301
yolo11m.pt,178.4,0.456,172.1,0.478,168.3,0.501,165.2,0.523
yolo11l.pt,298.6,0.678,291.2,0.701,286.7,0.723,282.1,0.745
yolo11x.pt,456.3,0.890,448.9,0.912,442.1,0.934,437.8,0.956"""
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write(self.sample_csv_data)
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_get_model_size_order(self):
        """Test model size ordering logic."""
        # Test known model sizes
        self.assertEqual(self.processor.get_model_size_order('yolo11n'), 1)
        self.assertEqual(self.processor.get_model_size_order('yolo11s'), 2)
        self.assertEqual(self.processor.get_model_size_order('yolo11m'), 3)
        self.assertEqual(self.processor.get_model_size_order('yolo11l'), 4)
        self.assertEqual(self.processor.get_model_size_order('yolo11x'), 5)
        
        # Test patterns that contain the size indicators
        self.assertEqual(self.processor.get_model_size_order('11n'), 1)
        self.assertEqual(self.processor.get_model_size_order('11s'), 2)
        self.assertEqual(self.processor.get_model_size_order('11m'), 3)
        self.assertEqual(self.processor.get_model_size_order('11l'), 4)
        self.assertEqual(self.processor.get_model_size_order('11x'), 5)
        
        # Test unknown models
        self.assertEqual(self.processor.get_model_size_order('yolo11z'), 999)
        self.assertEqual(self.processor.get_model_size_order('unknown_model'), 999)
        self.assertEqual(self.processor.get_model_size_order(''), 999)
        
        # Test edge cases that don't contain the pattern
        self.assertEqual(self.processor.get_model_size_order('yolo11'), 999)
        self.assertEqual(self.processor.get_model_size_order('yolo'), 999)
    
    def test_process_csv_basic(self):
        """Test basic CSV processing functionality."""
        result_df = self.processor.process_csv(self.temp_csv.name)
        
        # Check DataFrame structure
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 5)  # 5 models
        self.assertEqual(len(result_df.columns), 8)  # 4 approaches × 2 metrics
        
        # Check MultiIndex structure
        self.assertEqual(result_df.columns.nlevels, 2)
        
        # Check model names are cleaned (no .pt extension)
        for model_name in result_df.index:
            self.assertNotIn('.pt', model_name)
        
        # Check that models are sorted by size
        expected_order = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
        self.assertEqual(list(result_df.index), expected_order)
    
    def test_process_csv_missing_columns(self):
        """Test CSV processing with missing columns."""
        # Create CSV with only some columns
        limited_csv_data = """Model,L0_Memory_MB,L0_Time_s,L2_Memory_MB,L2_Time_s
yolo11n.pt,45.2,0.123,40.3,0.167
yolo11s.pt,89.7,0.234,82.1,0.278"""
        
        temp_limited_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_limited_csv.write(limited_csv_data)
        temp_limited_csv.close()
        
        try:
            result_df = self.processor.process_csv(temp_limited_csv.name)
            
            # Should only have 2 approaches (L0 and L2)
            self.assertEqual(len(result_df.columns), 4)  # 2 approaches × 2 metrics
            self.assertEqual(len(result_df), 2)  # 2 models
            
        finally:
            os.unlink(temp_limited_csv.name)
    
    def test_process_csv_file_not_found(self):
        """Test handling of non-existent CSV file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_csv('non_existent_file.csv')
    
    def test_process_csv_empty_file(self):
        """Test handling of empty CSV file."""
        empty_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        empty_csv.write("")
        empty_csv.close()
        
        try:
            with self.assertRaises(pd.errors.EmptyDataError):
                self.processor.process_csv(empty_csv.name)
        finally:
            os.unlink(empty_csv.name)


class TestPerformanceVisualizer(unittest.TestCase):
    """Test cases for PerformanceVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = PerformanceVisualizer()
        
        # Create sample DataFrame for testing
        data = {
            ('Approach #0 [Baseline]', 'Mem (MB)'): [45.2, 89.7, 178.4],
            ('Approach #0 [Baseline]', 'Runtime (s)'): [0.123, 0.234, 0.456],
            ('Approach #1', 'Mem (MB)'): [42.1, 85.2, 172.1],
            ('Approach #1', 'Runtime (s)'): [0.145, 0.256, 0.478]
        }
        self.sample_df = pd.DataFrame(
            data,
            index=['yolo11n', 'yolo11s', 'yolo11m']
        )
    
    @patch('builtins.input', return_value='1')
    def test_interactive_select_single(self, mock_input):
        """Test interactive selection with single item."""
        items = ['Approach #0 [Baseline]', 'Approach #1', 'Approach #2']
        result = self.visualizer.interactive_select(items, "Select one item:")
        
        self.assertEqual(result, ['Approach #0 [Baseline]'])
        mock_input.assert_called_once()
    
    @patch('builtins.input', return_value='')
    def test_interactive_select_all(self, mock_input):
        """Test interactive selection with all items (empty input)."""
        items = ['Approach #0 [Baseline]', 'Approach #1']
        result = self.visualizer.interactive_select(items)
        
        self.assertEqual(result, items)
        mock_input.assert_called_once()
    
    @patch('builtins.input', return_value='all')
    def test_interactive_select_all_keyword(self, mock_input):
        """Test interactive selection with 'all' keyword."""
        items = ['Approach #0 [Baseline]', 'Approach #1']
        result = self.visualizer.interactive_select(items)
        
        self.assertEqual(result, items)
        mock_input.assert_called_once()
    
    @patch('builtins.input', side_effect=['1,2', 'invalid', '1 2'])
    def test_interactive_select_multiple(self, mock_input):
        """Test interactive selection with multiple items."""
        items = ['Approach #0 [Baseline]', 'Approach #1', 'Approach #2']
        
        # First call returns invalid input, second call returns valid input
        result = self.visualizer.interactive_select(items)
        
        # Should return the valid input from the second call
        self.assertEqual(result, ['Approach #0 [Baseline]', 'Approach #1'])
    
    def test_interactive_select_empty_list(self):
        """Test interactive selection with empty list."""
        result = self.visualizer.interactive_select([])
        self.assertEqual(result, [])
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    @patch('builtins.input', return_value='')  # Mock input to return empty (select all)
    def test_create_plots_basic(self, mock_input, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test basic plot creation functionality."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        
        # Mock canvas draw
        mock_fig.canvas.draw = Mock()
        
        # Create a temporary file that will pass verification
        temp_plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_plot_file.write(b'fake plot data\n' * 100)  # Write enough data to pass size check
        temp_plot_file.close()
        
        # Mock the savefig to use our temporary file
        def mock_savefig_func(filename, *args, **kwargs):
            # Copy our temp file to the requested filename
            import shutil
            shutil.copy2(temp_plot_file.name, filename)
        
        mock_savefig.side_effect = mock_savefig_func
        
        try:
            result_fig, result_axes, result_df = self.visualizer.create_plots(
                self.sample_df,
                output_filename='test_plot.png',
                interactive_selection=True,  # Enable interactive selection
                show_plot=False
            )
            
            # Verify the result
            self.assertEqual(result_fig, mock_fig)
            self.assertEqual(result_axes, (mock_ax1, mock_ax2))
            # Use pandas comparison instead of direct equality
            pd.testing.assert_frame_equal(result_df, self.sample_df)
            
            # Verify matplotlib calls
            mock_subplots.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_plot_file.name):
                os.unlink(temp_plot_file.name)
            if os.path.exists('test_plot.png'):
                os.unlink('test_plot.png')
    
    def test_create_plots_invalid_approaches(self):
        """Test plot creation with invalid approach names."""
        with self.assertRaises(ValueError) as context:
            self.visualizer.create_plots(
                self.sample_df,
                interactive_selection=False,
                enabled_approaches=['Invalid Approach']
            )
        
        self.assertIn("Invalid approach names", str(context.exception))
    
    def test_create_plots_no_approaches_selected(self):
        """Test plot creation with no approaches selected."""
        with self.assertRaises(ValueError) as context:
            self.visualizer.create_plots(
                self.sample_df,
                interactive_selection=False,
                enabled_approaches=[]
            )
        
        self.assertIn("At least one approach must be selected", str(context.exception))
    
    def test_verify_image_valid(self):
        """Test image verification with valid image."""
        # Mock PIL verification to avoid complex PNG creation
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img.verify = Mock()
            mock_open.return_value.__enter__.return_value = mock_img
            
            # Create a temporary file that's large enough (>= 1000 bytes)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                # Write enough data to pass the 1000-byte size check
                fake_image_data = b'fake image data\n' * 100  # ~1500 bytes
                temp_img.write(fake_image_data)
                temp_img_path = temp_img.name
            
            try:
                result = self.visualizer.verify_image(temp_img_path)
                self.assertTrue(result)
            finally:
                os.unlink(temp_img_path)
    
    def test_verify_image_nonexistent(self):
        """Test image verification with non-existent file."""
        result = self.visualizer.verify_image('nonexistent.png')
        self.assertFalse(result)
    
    def test_verify_image_too_small(self):
        """Test image verification with file that's too small."""
        # Create a temporary file that's too small
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
            temp_img.write(b'tiny')
            temp_img_path = temp_img.name
        
        try:
            result = self.visualizer.verify_image(temp_img_path)
            self.assertFalse(result)
        finally:
            os.unlink(temp_img_path)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for PerformanceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PerformanceAnalyzer()
        
        # Create sample CSV data
        self.sample_csv_data = """Model,L0_Memory_MB,L0_Time_s,L1_Memory_MB,L1_Time_s
yolo11n.pt,45.2,0.123,42.1,0.145
yolo11s.pt,89.7,0.234,85.2,0.256"""
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write(self.sample_csv_data)
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_init(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer.data_processor, PerformanceDataProcessor)
        self.assertIsInstance(self.analyzer.visualizer, PerformanceVisualizer)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    def test_analyze_basic(self, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test basic analysis functionality."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        mock_fig.canvas.draw = Mock()
        
        # Create a temporary file that will pass verification
        temp_plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_plot_file.write(b'fake plot data\n' * 100)  # Write enough data to pass size check
        temp_plot_file.close()
        
        # Mock the savefig to use our temporary file
        def mock_savefig_func(filename, *args, **kwargs):
            # Copy our temp file to the requested filename
            import shutil
            shutil.copy2(temp_plot_file.name, filename)
        
        mock_savefig.side_effect = mock_savefig_func
        
        try:
            # Mock image verification to return True
            with patch.object(self.analyzer.visualizer, 'verify_image', return_value=True):
                result_df, result_fig, result_axes = self.analyzer.analyze(
                    self.temp_csv.name,
                    output_filename='test_output.png',
                    show_plot=False,
                    interactive_selection=False,
                    enabled_approaches=['Approach #0 [Baseline]', 'Approach #1']
                )
            
            # Verify results
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(result_fig, mock_fig)
            self.assertEqual(result_axes, (mock_ax1, mock_ax2))
            
            # Verify file operations
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_plot_file.name):
                os.unlink(temp_plot_file.name)
            if os.path.exists('test_output.png'):
                os.unlink('test_output.png')
    
    def test_analyze_file_not_found(self):
        """Test analysis with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze('nonexistent.csv')
    
    @patch('matplotlib.pyplot.subplots')
    def test_analyze_invalid_approaches(self, mock_subplots):
        """Test analysis with invalid approach names."""
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze(
                self.temp_csv.name,
                interactive_selection=False,
                enabled_approaches=['Invalid Approach']
            )


class TestBackwardCompatibilityFunctions(unittest.TestCase):
    """Test cases for backward compatibility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample CSV data
        self.sample_csv_data = """Model,L0_Memory_MB,L0_Time_s,L1_Memory_MB,L1_Time_s
yolo11n.pt,45.2,0.123,42.1,0.145
yolo11s.pt,89.7,0.234,85.2,0.256"""
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write(self.sample_csv_data)
        self.temp_csv.close()
        
        # Create sample DataFrame for testing
        data = {
            ('Approach #0 [Baseline]', 'Mem (MB)'): [45.2, 89.7],
            ('Approach #0 [Baseline]', 'Runtime (s)'): [0.123, 0.234],
            ('Approach #1', 'Mem (MB)'): [42.1, 85.2],
            ('Approach #1', 'Runtime (s)'): [0.145, 0.256]
        }
        self.sample_df = pd.DataFrame(
            data,
            index=['yolo11n', 'yolo11s']
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_get_model_size_order_function(self):
        """Test backward compatibility get_model_size_order function."""
        # Test that it returns the same result as the class method
        self.assertEqual(get_model_size_order('yolo11n'), 1)
        self.assertEqual(get_model_size_order('yolo11s'), 2)
        self.assertEqual(get_model_size_order('unknown'), 999)
    
    @patch('builtins.input', return_value='1')
    def test_interactive_select_function(self, mock_input):
        """Test backward compatibility interactive_select function."""
        items = ['Approach #0 [Baseline]', 'Approach #1']
        result = interactive_select(items, "Select item:")
        
        self.assertEqual(result, ['Approach #0 [Baseline]'])
        mock_input.assert_called_once()
    
    def test_process_performance_metrics_csv_function(self):
        """Test backward compatibility process_performance_metrics_csv function."""
        result_df = process_performance_metrics_csv(self.temp_csv.name)
        
        # Should return a DataFrame with the expected structure
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(len(result_df), 2)  # 2 models
        self.assertEqual(len(result_df.columns), 4)  # 2 approaches × 2 metrics
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    def test_create_performance_comparison_plots_function(self, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test backward compatibility create_performance_comparison_plots function."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        mock_fig.canvas.draw = Mock()
        
        # Create a temporary file that will pass verification
        temp_plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_plot_file.write(b'fake plot data\n' * 100)  # Write enough data to pass size check
        temp_plot_file.close()
        
        # Mock the savefig to use our temporary file
        def mock_savefig_func(filename, *args, **kwargs):
            # Copy our temp file to the requested filename
            import shutil
            shutil.copy2(temp_plot_file.name, filename)
        
        mock_savefig.side_effect = mock_savefig_func
        
        try:
            result_fig, result_axes, result_df = create_performance_comparison_plots(
                self.sample_df,
                output_filename='test_plot.png',
                interactive_selection=False,
                enabled_approaches=['Approach #0 [Baseline]', 'Approach #1'],
                show_plot=False
            )
            
            # Verify results
            self.assertEqual(result_fig, mock_fig)
            self.assertEqual(result_axes, (mock_ax1, mock_ax2))
            # Use pandas comparison instead of direct equality
            pd.testing.assert_frame_equal(result_df, self.sample_df)
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_plot_file.name):
                os.unlink(temp_plot_file.name)
            if os.path.exists('test_plot.png'):
                os.unlink('test_plot.png')
    
    def test_verify_output_image_function(self):
        """Test backward compatibility verify_output_image function."""
        # Mock PIL verification to avoid complex PNG creation
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img.verify = Mock()
            mock_open.return_value.__enter__.return_value = mock_img
            
            # Create a temporary file that's large enough (>= 1000 bytes)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                # Write enough data to pass the 1000-byte size check
                fake_image_data = b'fake image data\n' * 100  # ~1500 bytes
                temp_img.write(fake_image_data)
                temp_img_path = temp_img.name
            
            try:
                result = verify_output_image(temp_img_path)
                self.assertTrue(result)
            finally:
                os.unlink(temp_img_path)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    def test_analyze_performance_metrics_function(self, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test backward compatibility analyze_performance_metrics function."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        mock_fig.canvas.draw = Mock()
        
        # Create a temporary file that will pass verification
        temp_plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_plot_file.write(b'fake plot data\n' * 100)  # Write enough data to pass size check
        temp_plot_file.close()
        
        # Mock the savefig to use our temporary file
        def mock_savefig_func(filename, *args, **kwargs):
            # Copy our temp file to the requested filename
            import shutil
            shutil.copy2(temp_plot_file.name, filename)
        
        mock_savefig.side_effect = mock_savefig_func
        
        try:
            # Mock image verification
            with patch('benchmark_tools.yolo_performance_analyzer.PerformanceVisualizer.verify_image', return_value=True):
                result_df, result_fig, result_axes = analyze_performance_metrics(
                    self.temp_csv.name,
                    output_filename='test_output.png',
                    show_plot=False,
                    interactive_selection=False,
                    enabled_approaches=['Approach #0 [Baseline]', 'Approach #1']
                )
            
            # Verify results
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(result_fig, mock_fig)
            self.assertEqual(result_axes, (mock_ax1, mock_ax2))
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_plot_file.name):
                os.unlink(temp_plot_file.name)
            if os.path.exists('test_output.png'):
                os.unlink('test_output.png')


class TestMainFunction(unittest.TestCase):
    """Test cases for main CLI function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample CSV data
        self.sample_csv_data = """Model,L0_Memory_MB,L0_Time_s,L1_Memory_MB,L1_Time_s
yolo11n.pt,45.2,0.123,42.1,0.145
yolo11s.pt,89.7,0.234,85.2,0.256"""
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write(self.sample_csv_data)
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    @patch('sys.argv', ['yolo_performance_analyzer.py', 'dummy_path.csv'])
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    @patch('builtins.input', return_value='')  # Mock input to return empty (select all)
    @patch('builtins.print')  # Mock print to avoid output during tests
    def test_main_basic(self, mock_print, mock_input, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test basic main function execution."""
        # Update the argv patch with the actual temp file path
        import sys
        sys.argv = ['yolo_performance_analyzer.py', self.temp_csv.name]
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        mock_fig.canvas.draw = Mock()
        
        # Mock image verification
        with patch('benchmark_tools.yolo_performance_analyzer.PerformanceVisualizer.verify_image', return_value=True):
            # Capture stdout to avoid interference
            with patch('sys.stdout', StringIO()):
                main()
        
        # Verify that the function completed without errors
        # (main() doesn't return anything, so we just check it didn't raise an exception)
        self.assertTrue(True)  # If we get here, main() executed successfully
    
    @patch('sys.argv', ['yolo_performance_analyzer.py', 'dummy_path.csv', '-o', 'test_output.png'])
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    @patch('builtins.input', return_value='')  # Mock input to return empty (select all)
    @patch('builtins.print')
    def test_main_with_output_file(self, mock_print, mock_input, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test main function with output file specified."""
        # Update the argv patch with the actual temp file path
        import sys
        sys.argv = ['yolo_performance_analyzer.py', self.temp_csv.name, '-o', 'test_output.png']
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        mock_fig.canvas.draw = Mock()
        
        # Mock image verification
        with patch('benchmark_tools.yolo_performance_analyzer.PerformanceVisualizer.verify_image', return_value=True):
            with patch('sys.stdout', StringIO()):
                main()
        
        # Verify savefig was called with the correct filename
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        self.assertIn('test_output.png', args[0])
    
    @patch('sys.argv', ['yolo_performance_analyzer.py', 'dummy_path.csv', '--no-interactive', '--approaches', 'Approach #0 [Baseline]'])
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    @patch('builtins.print')
    def test_main_non_interactive(self, mock_print, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test main function with non-interactive mode."""
        # Update the argv patch with the actual temp file path
        import sys
        sys.argv = ['yolo_performance_analyzer.py', self.temp_csv.name, '--no-interactive', '--approaches', 'Approach #0 [Baseline]']
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        mock_fig.canvas.draw = Mock()
        
        # Mock image verification
        with patch('benchmark_tools.yolo_performance_analyzer.PerformanceVisualizer.verify_image', return_value=True):
            with patch('sys.stdout', StringIO()):
                main()
        
        # Function should complete successfully
        self.assertTrue(True)
    
    @patch('sys.argv', ['yolo_performance_analyzer.py', 'nonexistent.csv'])
    def test_main_file_not_found(self):
        """Test main function with non-existent file."""
        with self.assertRaises(SystemExit) as cm:
            with patch('sys.stderr', StringIO()):
                main()
        
        # Should exit with error code 1
        self.assertEqual(cm.exception.code, 1)
    
    @patch('sys.argv', ['yolo_performance_analyzer.py', '--help'])
    def test_main_help(self):
        """Test main function with --help flag."""
        with self.assertRaises(SystemExit) as cm:
            main()
        
        # Help should exit with code 0
        self.assertEqual(cm.exception.code, 0)
    
    @patch('sys.argv', ['yolo_performance_analyzer.py', '--version'])
    def test_main_version(self):
        """Test main function with --version flag."""
        with self.assertRaises(SystemExit) as cm:
            main()
        
        # Version should exit with code 0
        self.assertEqual(cm.exception.code, 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create comprehensive sample CSV data
        self.comprehensive_csv_data = """Model,L0_Memory_MB,L0_Time_s,L1_Memory_MB,L1_Time_s,L2_Memory_MB,L2_Time_s,L3_Memory_MB,L3_Time_s
yolo11n.pt,45.2,0.123,42.1,0.145,40.3,0.167,38.9,0.189
yolo11s.pt,89.7,0.234,85.2,0.256,82.1,0.278,79.8,0.301
yolo11m.pt,178.4,0.456,172.1,0.478,168.3,0.501,165.2,0.523
yolo11l.pt,298.6,0.678,291.2,0.701,286.7,0.723,282.1,0.745
yolo11x.pt,456.3,0.890,448.9,0.912,442.1,0.934,437.8,0.956"""
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write(self.comprehensive_csv_data)
        self.temp_csv.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    @patch('builtins.input', return_value='')  # Mock input to return empty (select all)
    def test_end_to_end_workflow(self, mock_input, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test complete end-to-end workflow."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        mock_fig.canvas.draw = Mock()
        
        # Create a temporary file that will pass verification
        temp_plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_plot_file.write(b'fake plot data\n' * 100)  # Write enough data to pass size check
        temp_plot_file.close()
        
        # Mock the savefig to use our temporary file
        def mock_savefig_func(filename, *args, **kwargs):
            # Copy our temp file to the requested filename
            import shutil
            shutil.copy2(temp_plot_file.name, filename)
        
        mock_savefig.side_effect = mock_savefig_func
        
        try:
            # Mock image verification
            with patch('benchmark_tools.yolo_performance_analyzer.PerformanceVisualizer.verify_image', return_value=True):
                # Test the complete workflow using the main analyze function
                result_df, result_fig, result_axes = analyze_performance_metrics(
                    self.temp_csv.name,
                    output_filename='integration_test.png',
                    show_plot=False,
                    interactive_selection=True,  # Enable interactive selection
                    enabled_approaches=['Approach #0 [Baseline]', 'Approach #1', 'Approach #2', 'Approach #3']
                )
            
            # Verify the complete workflow
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(len(result_df), 5)  # All 5 models
            self.assertEqual(len(result_df.columns), 8)  # All 4 approaches × 2 metrics
            
            # Check that models are sorted correctly
            expected_order = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
            self.assertEqual(list(result_df.index), expected_order)
            
            # Verify plot generation was attempted
            mock_savefig.assert_called_once()
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_plot_file.name):
                os.unlink(temp_plot_file.name)
            if os.path.exists('integration_test.png'):
                os.unlink('integration_test.png')
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    @patch('builtins.input', return_value='')  # Mock input to return empty (select all)
    def test_class_based_api_workflow(self, mock_input, mock_subplots, mock_close, mock_savefig, mock_show):
        """Test workflow using class-based API."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = mock_fig, (mock_ax1, mock_ax2)
        mock_fig.canvas.draw = Mock()
        
        # Create a temporary file that will pass verification
        temp_plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_plot_file.write(b'fake plot data\n' * 100)  # Write enough data to pass size check
        temp_plot_file.close()
        
        # Mock the savefig to use our temporary file
        def mock_savefig_func(filename, *args, **kwargs):
            # Copy our temp file to the requested filename
            import shutil
            shutil.copy2(temp_plot_file.name, filename)
        
        mock_savefig.side_effect = mock_savefig_func
        
        try:
            # Test class-based API workflow
            analyzer = PerformanceAnalyzer()
            
            # Mock image verification
            with patch.object(analyzer.visualizer, 'verify_image', return_value=True):
                result_df, result_fig, result_axes = analyzer.analyze(
                    self.temp_csv.name,
                    output_filename='class_api_test.png',
                    show_plot=False,
                    interactive_selection=True,  # Enable interactive selection
                    enabled_approaches=['Approach #0 [Baseline]', 'Approach #1']
                )
            
            # Verify the workflow
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(result_fig, mock_fig)
            self.assertEqual(result_axes, (mock_ax1, mock_ax2))
            
            # Verify file operations
            mock_savefig.assert_called_once()
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_plot_file.name):
                os.unlink(temp_plot_file.name)
            if os.path.exists('class_api_test.png'):
                os.unlink('class_api_test.png')
    
    def test_data_processing_edge_cases(self):
        """Test data processing with various edge cases."""
        processor = PerformanceDataProcessor()
        
        # Test with CSV containing only some approaches
        partial_csv_data = """Model,L1_Memory_MB,L1_Time_s,L3_Memory_MB,L3_Time_s
yolo11n.pt,42.1,0.145,38.9,0.189
yolo11s.pt,85.2,0.256,79.8,0.301"""
        
        temp_partial_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_partial_csv.write(partial_csv_data)
        temp_partial_csv.close()
        
        try:
            result_df = processor.process_csv(temp_partial_csv.name)
            
            # Should only have 2 approaches (L1 and L3)
            self.assertEqual(len(result_df.columns), 4)  # 2 approaches × 2 metrics
            
            # Check that approach names are correctly mapped
            expected_approaches = ['Approach #1', 'Approach #3']
            actual_approaches = result_df.columns.levels[0].tolist()
            self.assertEqual(actual_approaches, expected_approaches)
            
        finally:
            os.unlink(temp_partial_csv.name)


if __name__ == '__main__':
    # Run the tests
    unittest.main()
