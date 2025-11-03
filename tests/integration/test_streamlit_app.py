#!/usr/bin/env python3
"""
Unit tests for Streamlit YOLO Performance Metrics Analysis App

This module contains comprehensive unit tests for all functions in the 
streamlit_app.py script, including mocking of Streamlit components and
external dependencies.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import io
from io import BytesIO

# Import the module to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmark_tools'))
from streamlit_app import (
    get_csv_files,
    load_csv_data, 
    create_plot,
    fig_to_bytes,
    main
)


class TestGetCSVFiles(unittest.TestCase):
    """Test cases for get_csv_files function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create sample CSV files
        self.sample_csvs = [
            "performance_metrics_20231101_120000.csv",
            "performance_metrics_20231101_130000.csv", 
            "performance_metrics_baseline.csv",
            "other_file.csv",  # Should be filtered out
            "metrics_20231101.csv"  # Should be filtered out
        ]
        
        for csv_file in self.sample_csvs:
            csv_path = os.path.join(self.results_dir, csv_file)
            with open(csv_path, 'w') as f:
                f.write("Model,L0_Memory_MB,L0_Time_s\n")
                f.write("yolo11n.pt,45.2,0.123\n")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_csv_files_success(self, mock_listdir, mock_exists):
        """Test successful CSV file discovery."""
        mock_exists.return_value = True
        mock_listdir.return_value = self.sample_csvs
        
        csv_files, error = get_csv_files()
        
        self.assertIsNone(error)
        self.assertEqual(len(csv_files), 3)
        self.assertIn("performance_metrics_20231101_120000.csv", csv_files)
        self.assertIn("performance_metrics_20231101_130000.csv", csv_files)
        self.assertIn("performance_metrics_baseline.csv", csv_files)
        # Should not include non-matching files
        self.assertNotIn("other_file.csv", csv_files)
        self.assertNotIn("metrics_20231101.csv", csv_files)
    
    @patch('os.path.exists', return_value=False)
    def test_get_csv_files_directory_not_found(self, mock_exists):
        """Test handling when results directory doesn't exist."""
        csv_files, error = get_csv_files()
        
        self.assertEqual(csv_files, [])
        self.assertIn("Results directory not found", error)
        self.assertIn("performance benchmarks first", error)
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_csv_files_no_matching_files(self, mock_listdir, mock_exists):
        """Test handling when no matching CSV files exist."""
        mock_exists.return_value = True
        mock_listdir.return_value = ["other_file.txt", "data.json", "readme.md"]
        
        csv_files, error = get_csv_files()
        
        self.assertEqual(csv_files, [])
        self.assertIn("No performance metrics CSV files found", error)
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_csv_files_empty_directory(self, mock_listdir, mock_exists):
        """Test handling when results directory is empty."""
        mock_exists.return_value = True
        mock_listdir.return_value = []
        
        csv_files, error = get_csv_files()
        
        self.assertEqual(csv_files, [])
        self.assertIn("No performance metrics CSV files found", error)
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_csv_files_alphabetical_sorting(self, mock_listdir, mock_exists):
        """Test that CSV files are sorted alphabetically."""
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "performance_metrics_zebra.csv",
            "performance_metrics_apple.csv",
            "performance_metrics_banana.csv"
        ]
        
        csv_files, error = get_csv_files()
        
        self.assertIsNone(error)
        self.assertEqual(csv_files, [
            "performance_metrics_apple.csv",
            "performance_metrics_banana.csv", 
            "performance_metrics_zebra.csv"
        ])


class TestLoadCSVData(unittest.TestCase):
    """Test cases for load_csv_data function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create sample CSV data
        self.sample_csv_content = """Model,L0_Memory_MB,L0_Time_s,L1_Memory_MB,L1_Time_s
yolo11n.pt,45.2,0.123,42.1,0.145
yolo11s.pt,89.7,0.234,85.2,0.256"""
        
        self.csv_filename = "test_performance_metrics.csv"
        self.csv_path = os.path.join(self.results_dir, self.csv_filename)
        
        with open(self.csv_path, 'w') as f:
            f.write(self.sample_csv_content)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('streamlit_app.PerformanceDataProcessor')
    def test_load_csv_data_success(self, mock_processor_class):
        """Test successful CSV data loading."""
        # Mock the processor and its process_csv method
        mock_processor = Mock()
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        mock_processor.process_csv.return_value = mock_df
        mock_processor_class.return_value = mock_processor
        
        df, error = load_csv_data(self.csv_filename)
        
        self.assertIsNone(error)
        pd.testing.assert_frame_equal(df, mock_df)
        # Check that it was called with the relative path (as used in the actual function)
        mock_processor.process_csv.assert_called_once_with(f"results/{self.csv_filename}")
    
    @patch('streamlit_app.PerformanceDataProcessor')
    def test_load_csv_data_file_not_found(self, mock_processor_class):
        """Test handling when CSV file doesn't exist."""
        mock_processor = Mock()
        mock_processor.process_csv.side_effect = FileNotFoundError("[Errno 2] No such file or directory: 'results/nonexistent.csv'")
        mock_processor_class.return_value = mock_processor
        
        df, error = load_csv_data("nonexistent.csv")
        
        self.assertIsNone(df)
        self.assertIn("Error loading CSV file", error)
        self.assertIn("No such file or directory", error)
    
    @patch('streamlit_app.PerformanceDataProcessor')
    def test_load_csv_data_malformed_csv(self, mock_processor_class):
        """Test handling when CSV file is malformed."""
        mock_processor = Mock()
        mock_processor.process_csv.side_effect = pd.errors.EmptyDataError("No columns to parse")
        mock_processor_class.return_value = mock_processor
        
        df, error = load_csv_data("malformed.csv")
        
        self.assertIsNone(df)
        self.assertIn("Error loading CSV file", error)
        self.assertIn("No columns to parse", error)
    
    @patch('streamlit_app.PerformanceDataProcessor')
    def test_load_csv_data_generic_exception(self, mock_processor_class):
        """Test handling of generic exceptions during CSV loading."""
        mock_processor = Mock()
        mock_processor.process_csv.side_effect = Exception("Unexpected error")
        mock_processor_class.return_value = mock_processor
        
        df, error = load_csv_data("error.csv")
        
        self.assertIsNone(df)
        self.assertIn("Error loading CSV file", error)
        self.assertIn("Unexpected error", error)


class TestCreatePlot(unittest.TestCase):
    """Test cases for create_plot function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample DataFrame for testing
        data = {
            ('Approach #0 [Baseline]', 'Mem (MB)'): [45.2, 89.7, 178.4],
            ('Approach #0 [Baseline]', 'Runtime (s)'): [0.123, 0.234, 0.456],
            ('Approach #1', 'Mem (MB)'): [42.1, 85.2, 172.1],
            ('Approach #1', 'Runtime (s)'): [0.145, 0.256, 0.478],
            ('Approach #2', 'Mem (MB)'): [40.3, 82.1, 168.3],
            ('Approach #2', 'Runtime (s)'): [0.167, 0.278, 0.501]
        }
        self.sample_df = pd.DataFrame(
            data,
            index=['/path/to/yolo11n.pt', '/path/to/yolo11s.pt', '/path/to/yolo11m.pt']
        )
        
        # Mock matplotlib components
        self.mock_fig = Mock()
        self.mock_ax1 = Mock()
        self.mock_ax2 = Mock()
        self.mock_ax1.bar = Mock()
        self.mock_ax2.bar = Mock()
        self.mock_ax1.text = Mock()
        self.mock_ax2.text = Mock()
        self.mock_ax1.set_ylabel = Mock()
        self.mock_ax2.set_ylabel = Mock()
        self.mock_ax1.legend = Mock()
        self.mock_ax2.set_xlabel = Mock()
        self.mock_ax2.set_xticks = Mock()
        self.mock_ax2.set_xticklabels = Mock()
        self.mock_ax1.grid = Mock()
        self.mock_ax2.grid = Mock()
        self.mock_ax1.set_title = Mock()
        self.mock_ax2.set_title = Mock()
        self.mock_ax1.set_yscale = Mock()
        self.mock_ax2.set_yscale = Mock()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.style.use')
    @patch('seaborn.set_palette')
    @patch('matplotlib.pyplot.rcParams', new_callable=dict)
    def test_create_plot_success(self, mock_rcParams, mock_set_palette, mock_style_use, mock_subplots):
        """Test successful plot creation."""
        mock_subplots.return_value = (self.mock_fig, [self.mock_ax1, self.mock_ax2])
        
        fig, error = create_plot(
            self.sample_df, 
            ['Approach #0 [Baseline]', 'Approach #1'],
            ['/path/to/yolo11n.pt', '/path/to/yolo11s.pt'],
            show_values=False,
            log_scale=False
        )
        
        self.assertIsNone(error)
        self.assertEqual(fig, self.mock_fig)
        
        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_style_use.assert_called_once_with('seaborn-v0_8')
        mock_set_palette.assert_called_once_with("colorblind")
        
        # Verify axis labels and titles
        self.mock_ax1.set_ylabel.assert_called_with('Memory Usage (MB)', fontsize=12, fontweight='bold')
        self.mock_ax2.set_ylabel.assert_called_with('Runtime (seconds)', fontsize=12, fontweight='bold')
        self.mock_ax2.set_xlabel.assert_called_with('YOLO Models', fontsize=12, fontweight='bold')
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.style.use')
    @patch('seaborn.set_palette')
    @patch('matplotlib.pyplot.rcParams', new_callable=dict)
    def test_create_plot_with_log_scale(self, mock_rcParams, mock_set_palette, mock_style_use, mock_subplots):
        """Test plot creation with logarithmic Y-axis scaling."""
        mock_subplots.return_value = (self.mock_fig, [self.mock_ax1, self.mock_ax2])
        
        fig, error = create_plot(
            self.sample_df,
            ['Approach #0 [Baseline]'],
            ['/path/to/yolo11n.pt'],
            show_values=False,
            log_scale=True
        )
        
        self.assertIsNone(error)
        # Verify log scale was applied
        self.mock_ax1.set_yscale.assert_called_with('log')
        self.mock_ax2.set_yscale.assert_called_with('log')
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.style.use')
    @patch('seaborn.set_palette')
    @patch('matplotlib.pyplot.rcParams', new_callable=dict)
    def test_create_plot_with_show_values(self, mock_rcParams, mock_set_palette, mock_style_use, mock_subplots):
        """Test plot creation with value labels on bars."""
        mock_subplots.return_value = (self.mock_fig, [self.mock_ax1, self.mock_ax2])
        
        fig, error = create_plot(
            self.sample_df,
            ['Approach #0 [Baseline]'],
            ['/path/to/yolo11n.pt'],
            show_values=True,
            log_scale=False
        )
        
        self.assertIsNone(error)
        # Verify text annotations were added
        self.mock_ax1.text.assert_called()
        self.mock_ax2.text.assert_called()
    
    @patch('matplotlib.pyplot.subplots')
    def test_create_plot_no_valid_approaches(self, mock_subplots):
        """Test plot creation with no valid approaches."""
        mock_subplots.return_value = (self.mock_fig, [self.mock_ax1, self.mock_ax2])
        
        fig, error = create_plot(
            self.sample_df,
            ['Invalid Approach'],  # Non-existent approach
            ['/path/to/yolo11n.pt'],
            show_values=False,
            log_scale=False
        )
        
        self.assertIsNone(fig)
        self.assertIn("No valid approaches selected or available in data", error)
    
    def test_create_plot_empty_models_list(self):
        """Test plot creation with empty models list."""
        fig, error = create_plot(
            self.sample_df,
            ['Approach #0 [Baseline]'],
            [],  # Empty models list
            show_values=False,
            log_scale=False
        )
        
        # Should return all models when selected_models is empty
        self.assertIsNone(error)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.style.use')
    @patch('seaborn.set_palette')
    @patch('matplotlib.pyplot.rcParams', new_callable=dict)
    def test_create_plot_model_filtering(self, mock_rcParams, mock_set_palette, mock_style_use, mock_subplots):
        """Test plot creation with model filtering."""
        mock_subplots.return_value = (self.mock_fig, [self.mock_ax1, self.mock_ax2])
        
        fig, error = create_plot(
            self.sample_df,
            ['Approach #0 [Baseline]'],
            ['/path/to/yolo11n.pt'],  # Only one model
            show_values=False,
            log_scale=False
        )
        
        self.assertIsNone(error)
        # Should have filtered to only one model
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.style.use')
    @patch('seaborn.set_palette')
    @patch('matplotlib.pyplot.rcParams', new_callable=dict)
    def test_create_plot_exception_handling(self, mock_rcParams, mock_set_palette, mock_style_use, mock_subplots):
        """Test exception handling in plot creation."""
        mock_subplots.side_effect = Exception("Matplotlib error")
        
        fig, error = create_plot(
            self.sample_df,
            ['Approach #0 [Baseline]'],
            ['/path/to/yolo11n.pt'],
            show_values=False,
            log_scale=False
        )
        
        self.assertIsNone(fig)
        self.assertIn("Error creating plot", error)
        self.assertIn("Matplotlib error", error)


class TestFigToBytes(unittest.TestCase):
    """Test cases for fig_to_bytes function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_fig = Mock()
        self.mock_canvas = Mock()
        self.mock_fig.canvas = Mock()
        self.mock_fig.canvas.draw = Mock()
        self.mock_fig.savefig = Mock()
    
    @patch('io.BytesIO')
    def test_fig_to_bytes_png_default_dpi(self, mock_bytesio):
        """Test figure to bytes conversion with default PNG format and DPI."""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_buf.getvalue.return_value = b'fake_image_data'
        
        result = fig_to_bytes(self.mock_fig)
        
        self.assertEqual(result, b'fake_image_data')
        mock_buf.getvalue.assert_called_once()
        mock_buf.seek.assert_called_with(0)
        
        # Verify savefig was called with correct parameters
        self.mock_fig.savefig.assert_called_once()
        args, kwargs = self.mock_fig.savefig.call_args
        self.assertEqual(args[0], mock_buf)
        self.assertEqual(kwargs['format'], 'png')
        self.assertEqual(kwargs['dpi'], 300)
        self.assertTrue(kwargs['bbox_inches'], 'tight')
    
    @patch('io.BytesIO')
    def test_fig_to_bytes_custom_dpi(self, mock_bytesio):
        """Test figure to bytes conversion with custom DPI."""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_buf.getvalue.return_value = b'fake_image_data'
        
        result = fig_to_bytes(self.mock_fig, format='png', dpi=600)
        
        self.assertEqual(result, b'fake_image_data')
        args, kwargs = self.mock_fig.savefig.call_args
        self.assertEqual(kwargs['dpi'], 600)
    
    @patch('io.BytesIO')
    def test_fig_to_bytes_jpeg_format(self, mock_bytesio):
        """Test figure to bytes conversion with JPEG format."""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_buf.getvalue.return_value = b'fake_jpeg_data'
        
        result = fig_to_bytes(self.mock_fig, format='jpeg', dpi=150)
        
        self.assertEqual(result, b'fake_jpeg_data')
        args, kwargs = self.mock_fig.savefig.call_args
        self.assertEqual(kwargs['format'], 'jpeg')
        self.assertEqual(kwargs['dpi'], 150)
    
    @patch('io.BytesIO')
    def test_fig_to_bytes_exception_handling(self, mock_bytesio):
        """Test exception handling in figure to bytes conversion."""
        mock_buf = Mock()
        mock_bytesio.return_value = mock_buf
        mock_buf.getvalue.side_effect = Exception("Buffer error")
        
        # Should raise the exception
        with self.assertRaises(Exception):
            fig_to_bytes(self.mock_fig)
    
    def test_fig_to_bytes_facecolor_and_edgecolor(self):
        """Test that savefig is called with correct facecolor and edgecolor."""
        mock_buf = Mock()
        with patch('io.BytesIO', return_value=mock_buf):
            fig_to_bytes(self.mock_fig)
            
            args, kwargs = self.mock_fig.savefig.call_args
            self.assertEqual(kwargs['facecolor'], 'white')
            self.assertEqual(kwargs['edgecolor'], 'none')


class TestMainFunction(unittest.TestCase):
    """Test cases for main Streamlit app function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create sample CSV files
        self.sample_csv_content = """Model,L0_Memory_MB,L0_Time_s,L1_Memory_MB,L1_Time_s
yolo11n.pt,45.2,0.123,42.1,0.145
yolo11s.pt,89.7,0.234,85.2,0.256"""
        
        self.csv_filename = "performance_metrics_test.csv"
        self.csv_path = os.path.join(self.results_dir, self.csv_filename)
        
        with open(self.csv_path, 'w') as f:
            f.write(self.sample_csv_content)
        
        # Create sample DataFrame
        data = {
            ('Approach #0 [Baseline]', 'Mem (MB)'): [45.2, 89.7],
            ('Approach #0 [Baseline]', 'Runtime (s)'): [0.123, 0.234],
            ('Approach #1', 'Mem (MB)'): [42.1, 85.2],
            ('Approach #1', 'Runtime (s)'): [0.145, 0.256]
        }
        self.sample_df = pd.DataFrame(
            data,
            index=['/path/to/yolo11n.pt', '/path/to/yolo11s.pt']
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('streamlit_app.get_csv_files')
    @patch('streamlit_app.load_csv_data')
    @patch('streamlit_app.create_plot')
    @patch('streamlit_app.fig_to_bytes')
    @patch('os.path.getsize')
    @patch('os.path.getctime')
    def test_main_successful_flow(self, mock_getctime, mock_getsize, mock_fig_to_bytes, mock_create_plot, 
                                 mock_load_csv_data, mock_get_csv_files):
        """Test successful main function flow."""
        # Setup mocks
        mock_get_csv_files.return_value = ([self.csv_filename], None)
        
        # Create a proper MultiIndex DataFrame like the real app expects
        data = {
            ('Approach #0 [Baseline]', 'Mem (MB)'): [45.2, 89.7],
            ('Approach #0 [Baseline]', 'Runtime (s)'): [0.123, 0.234],
            ('Approach #1', 'Mem (MB)'): [42.1, 85.2],
            ('Approach #1', 'Runtime (s)'): [0.145, 0.256]
        }
        mock_df = pd.DataFrame(
            data,
            index=['/path/to/yolo11n.pt', '/path/to/yolo11s.pt']
        )
        mock_load_csv_data.return_value = (mock_df, None)
        
        mock_fig = Mock()
        mock_create_plot.return_value = (mock_fig, None)
        
        mock_plot_bytes = b'fake_plot_data'
        mock_fig_to_bytes.return_value = mock_plot_bytes
        
        # Mock file system operations
        mock_getsize.return_value = 1024
        mock_getctime.return_value = 1234567890
        
        # Mock Streamlit components
        with patch('streamlit.set_page_config'), \
             patch('streamlit.markdown'), \
             patch('streamlit.sidebar.header'), \
             patch('streamlit.sidebar.subheader'), \
             patch('streamlit.sidebar.selectbox') as mock_selectbox, \
             patch('streamlit.sidebar.checkbox') as mock_checkbox, \
             patch('streamlit.sidebar.radio') as mock_radio, \
             patch('streamlit.sidebar.info'), \
             patch('streamlit.spinner'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'), \
             patch('streamlit.subheader'), \
             patch('streamlit.pyplot'), \
             patch('streamlit.expander'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.download_button'), \
             patch('streamlit.columns') as mock_columns:
            
            # Mock UI interactions - setup properly to pass validation
            mock_selectbox.return_value = self.csv_filename
            mock_radio.return_value = "All Models"
            
            # Mock checkbox calls for approaches and models - need to return True for validation
            def checkbox_side_effect(*args, **kwargs):
                if 'approach_' in str(kwargs.get('key', '')):
                    return True  # Return True for approach checkboxes
                elif 'model_' in str(kwargs.get('key', '')):
                    return True  # Return True for model checkboxes  
                elif 'Show values on bars' in str(args):
                    return False
                elif 'logarithmic' in str(args):
                    return False
                return True
            
            mock_checkbox.side_effect = checkbox_side_effect
            
            # Mock columns to return proper context managers
            mock_col1 = Mock()
            mock_col2 = Mock()
            mock_col1.__enter__ = Mock(return_value=mock_col1)
            mock_col1.__exit__ = Mock(return_value=None)
            mock_col2.__enter__ = Mock(return_value=mock_col2)
            mock_col2.__exit__ = Mock(return_value=None)
            mock_columns.return_value = [mock_col1, mock_col2]
            
            # Mock matplotlib components
            with patch('matplotlib.pyplot.close'):
                main()
        
        # Verify the flow
        mock_get_csv_files.assert_called_once()
        mock_load_csv_data.assert_called_once()
        mock_create_plot.assert_called_once()
        mock_fig_to_bytes.assert_called()  # Called twice: once for PNG, once for HD PNG
    
    @patch('streamlit_app.get_csv_files')
    def test_main_no_results_directory(self, mock_get_csv_files):
        """Test main function when results directory doesn't exist."""
        mock_get_csv_files.return_value = ([], "Results directory not found. Please run performance benchmarks first.")
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.markdown'), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.info') as mock_info:
            
            main()
        
        mock_error.assert_called_once()
        mock_info.assert_called_once()
    
    @patch('streamlit_app.get_csv_files')
    @patch('streamlit.set_page_config')
    def test_main_no_csv_files(self, mock_page_config, mock_get_csv_files):
        """Test main function when no CSV files are available."""
        mock_get_csv_files.return_value = ([], None)
        
        with patch('streamlit.error') as mock_error:
            main()
        
        mock_error.assert_called_once_with("No CSV files available to analyze.")
    
    @patch('streamlit_app.get_csv_files')
    @patch('streamlit_app.load_csv_data')
    @patch('os.path.getsize')
    @patch('os.path.getctime')
    def test_main_csv_load_error(self, mock_getctime, mock_getsize, mock_load_csv_data, mock_get_csv_files):
        """Test main function when CSV loading fails."""
        mock_get_csv_files.return_value = ([self.csv_filename], None)
        mock_load_csv_data.return_value = (None, "Error loading CSV file: Invalid format")
        
        # Mock file system operations
        mock_getsize.return_value = 1024
        mock_getctime.return_value = 1234567890
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.markdown'), \
             patch('streamlit.error') as mock_error:
            
            main()
        
        mock_error.assert_called_once()
        args, kwargs = mock_error.call_args
        self.assertIn("Error loading CSV file", args[0])
    
    @patch('streamlit_app.get_csv_files')
    @patch('streamlit_app.load_csv_data')
    @patch('os.path.getsize')
    @patch('os.path.getctime')
    def test_main_empty_dataframe(self, mock_getctime, mock_getsize, mock_load_csv_data, mock_get_csv_files):
        """Test main function when loaded DataFrame is empty."""
        mock_get_csv_files.return_value = ([self.csv_filename], None)
        mock_load_csv_data.return_value = (pd.DataFrame(), None)
        
        # Mock file system operations
        mock_getsize.return_value = 1024
        mock_getctime.return_value = 1234567890
        
        with patch('streamlit.set_page_config'), \
             patch('streamlit.markdown'), \
             patch('streamlit.error') as mock_error:
            
            main()
        
        mock_error.assert_called_once_with("No data available in the selected CSV file.")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create comprehensive test data
        self.comprehensive_csv_content = """Model,L0_Memory_MB,L0_Time_s,L1_Memory_MB,L1_Time_s,L2_Memory_MB,L2_Time_s,L3_Memory_MB,L3_Time_s
yolo11n.pt,45.2,0.123,42.1,0.145,40.3,0.167,38.9,0.189
yolo11s.pt,89.7,0.234,85.2,0.256,82.1,0.278,79.8,0.301
yolo11m.pt,178.4,0.456,172.1,0.478,168.3,0.501,165.2,0.523
yolo11l.pt,298.6,0.678,291.2,0.701,286.7,0.723,282.1,0.745
yolo11x.pt,456.3,0.890,448.9,0.912,442.1,0.934,437.8,0.956"""
        
        self.csv_filename = "comprehensive_performance_metrics.csv"
        self.csv_path = os.path.join(self.results_dir, self.csv_filename)
        
        with open(self.csv_path, 'w') as f:
            f.write(self.comprehensive_csv_content)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.style.use')
    @patch('seaborn.set_palette')
    @patch('matplotlib.pyplot.rcParams', new_callable=dict)
    def test_complete_workflow_with_all_approaches(self, mock_rcParams, mock_set_palette, mock_style_use, mock_subplots):
        """Test complete workflow with all approaches and models."""
        # Create expected DataFrame structure
        data = {
            ('Approach #0 [Baseline]', 'Mem (MB)'): [45.2, 89.7, 178.4, 298.6, 456.3],
            ('Approach #0 [Baseline]', 'Runtime (s)'): [0.123, 0.234, 0.456, 0.678, 0.890],
            ('Approach #1', 'Mem (MB)'): [42.1, 85.2, 172.1, 291.2, 448.9],
            ('Approach #1', 'Runtime (s)'): [0.145, 0.256, 0.478, 0.701, 0.912],
            ('Approach #2', 'Mem (MB)'): [40.3, 82.1, 168.3, 286.7, 442.1],
            ('Approach #2', 'Runtime (s)'): [0.167, 0.278, 0.501, 0.723, 0.934],
            ('Approach #3', 'Mem (MB)'): [38.9, 79.8, 165.2, 282.1, 437.8],
            ('Approach #3', 'Runtime (s)'): [0.189, 0.301, 0.523, 0.745, 0.956]
        }
        expected_df = pd.DataFrame(data, index=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'])
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        
        # Mock data processing
        with patch('benchmark_tools.streamlit_app.PerformanceDataProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_csv.return_value = expected_df
            mock_processor_class.return_value = mock_processor
            
            # Test plot creation
            fig, error = create_plot(
                expected_df,
                ['Approach #0 [Baseline]', 'Approach #1', 'Approach #2', 'Approach #3'],
                ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                show_values=True,
                log_scale=True
            )
            
            # Verify results
            self.assertIsNone(error)
            self.assertEqual(fig, mock_fig)
    
    def test_error_propagation_workflow(self):
        """Test error propagation through the workflow."""
        # Test CSV loading error
        with patch('benchmark_tools.streamlit_app.PerformanceDataProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_csv.side_effect = FileNotFoundError("Test file not found")
            mock_processor_class.return_value = mock_processor
            
            df, error = load_csv_data("nonexistent.csv")
            
            self.assertIsNone(df)
            self.assertIn("Error loading CSV file", error)
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_creation_error_handling(self, mock_subplots):
        """Test comprehensive error handling in plot creation."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        fig, error = create_plot(empty_df, ['Approach #0'], ['Model1'])
        
        # Should handle empty DataFrame gracefully
        self.assertIsNotNone(error)  # Should return an error


if __name__ == '__main__':
    # Run the tests
    unittest.main()
