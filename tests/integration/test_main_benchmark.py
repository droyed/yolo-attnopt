#!/usr/bin/env python3
"""
Unit tests for main_benchmark.py

This module contains minimal unit tests for the main function
in the benchmark_tools.main_benchmark module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the module to test
from benchmark_tools.main_benchmark import main


class TestMainBenchmark(unittest.TestCase):
    """Test cases for main_benchmark main function."""
    
    @patch('benchmark_tools.main_benchmark.setup_optimized_yolo_environ')
    @patch('benchmark_tools.main_benchmark.YOLO')
    @patch('benchmark_tools.main_benchmark.imagesize')
    def test_main_function_basic_execution(self, mock_imagesize, mock_yolo, mock_setup_optimized):
        """Test that main function executes without errors with minimal parameters."""
        # Mock the dependencies
        mock_model_instance = Mock()
        mock_yolo.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.return_value = [Mock()]  # Mock results
        mock_imagesize.get.return_value = (640, 480)  # Mock image dimensions
        
        # Call main function with minimal parameters
        main(
            yolo_modelname='yolo11m.pt',
            image_scale_down=32,
            image_scale_up=32,
            confidence_threshold=0.13,
            save_images=False,
            print_bbox=False,
            image_path='test_image.jpg',  # Provide image path to trigger imagesize.get
            optimize_level=0,
            device='cpu',  # Use CPU to avoid CUDA issues
            image_size=640
        )
        
        # Verify that the mocked functions were called
        mock_setup_optimized.assert_called_once_with(optimize_level=0, debug_mode=True)
        mock_yolo.assert_called_once_with('yolo11m.pt')
        mock_model_instance.to.assert_called_once_with('cpu')
        mock_imagesize.get.assert_called_once_with('test_image.jpg')
        mock_model_instance.assert_called_once()
    
    @patch('benchmark_tools.main_benchmark.setup_optimized_yolo_environ')
    @patch('benchmark_tools.main_benchmark.YOLO')
    def test_main_function_with_random_image(self, mock_yolo, mock_setup_optimized):
        """Test main function with randomly generated image (no image path)."""
        # Mock the dependencies
        mock_model_instance = Mock()
        mock_yolo.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.return_value = [Mock()]  # Mock results
        
        # Call main function without image path (will generate random image)
        main(
            yolo_modelname='yolo11s.pt',
            image_scale_down=16,
            image_scale_up=16,
            confidence_threshold=0.25,
            save_images=True,
            print_bbox=True,
            image_path=None,  # No image path - will generate random image
            optimize_level=1,
            device='cpu',
            image_size=320
        )
        
        # Verify that the mocked functions were called
        mock_setup_optimized.assert_called_once_with(optimize_level=1, debug_mode=True)
        mock_yolo.assert_called_once_with('yolo11s.pt')
        mock_model_instance.to.assert_called_once_with('cpu')
        mock_model_instance.assert_called_once()


if __name__ == '__main__':
    unittest.main()
