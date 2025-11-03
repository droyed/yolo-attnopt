#!/usr/bin/env python3
"""
Performance Metrics Analysis Tool

This script processes YOLO performance metrics CSV files and creates comparison plots.
It can be used both as a module with class-based API or as a command-line tool.
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
# Set backend before importing pyplot to ensure proper headless operation
matplotlib.use('Agg', force=True)  # Use Anti-Grain Geometry backend for file output
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceDataProcessor:
    """
    Handles processing and transformation of YOLO performance metrics data.
    """
    
    @staticmethod
    def get_model_size_order(model_name):
        """Extract size suffix and return sorting priority"""
        size_order = {'n': 1, 's': 2, 'm': 3, 'l': 4, 'x': 5}  # nano, small, medium, large, extra-large
        # Extract the size letter (last character before any numbers)
        for suffix in size_order.keys():
            if f'11{suffix}' in model_name:
                return size_order[suffix]
        return 999  # Unknown models go to the end

    def process_csv(self, csv_file_path):
        """
        Process performance metrics CSV file and create a structured MultiIndex DataFrame.
        
        This method reads a performance metrics CSV file containing YOLO model benchmarks
        and transforms it into a structured format with MultiIndex columns representing
        different optimization approaches and their corresponding memory usage and runtime metrics.
        
        The method performs the following data transformations:
        1. Reads the CSV file into a DataFrame
        2. Extracts and reorganizes performance metrics from different approaches:
           - Approach #0: Maps to L0_Memory_MB and L0_Time_s columns
           - Approach #1: Maps to L2_Memory_MB and L2_Time_s columns  
           - Approach #2: Maps to L3_Memory_MB and L3_Time_s columns
           - Approach #3: Maps to L1_Memory_MB and L1_Time_s columns
        3. Creates a MultiIndex DataFrame with approach names and metric types
        4. Normalizes model names by removing '.pt' file extensions
        5. Sorts models by size using the get_model_size_order utility method
        
        Parameters
        ----------
        csv_file_path : str
            Path to the CSV file containing performance metrics data.
            Expected columns: 'Model', 'L0_Memory_MB', 'L0_Time_s', 'L1_Memory_MB', 
            'L1_Time_s', 'L2_Memory_MB', 'L2_Time_s', 'L3_Memory_MB', 'L3_Time_s'
        
        Returns
        -------
        pandas.DataFrame
            A MultiIndex DataFrame with the following structure:
            - Index: Model names (normalized, without .pt extension), sorted by model size  
            - Columns: MultiIndex with two levels:
              - Level 0: Approach names ('Approach #0', 'Approach #1', 'Approach #2', 'Approach #3')
              - Level 1: Metric types ('Mem (MB)', 'Runtime (s)')
            
            The DataFrame contains memory usage (MB) and runtime (seconds) metrics
            for each model across different optimization approaches.
        
        Notes
        -----
        - The mapping between approach numbers and L-columns is:
          Approach #0 -> L0, Approach #1 -> L2, Approach #2 -> L3, Approach #3 -> L1
        - Model names are automatically cleaned by removing '.pt' extensions
        - Results are sorted by model size using the get_model_size_order method
        """
        df11 = pd.read_csv(csv_file_path)

        # Dynamically detect available optimization levels from column headers
        available_levels = []
        memory_cols = []
        time_cols = []
        for col in df11.columns:
            if col.startswith('L') and col.endswith('_Memory_MB'):
                level = col.split('_')[0][1:]  # Extract level number (e.g., 'L0' -> '0')
                if level.isdigit():
                    available_levels.append(int(level))
                    memory_cols.append(col)
                    time_cols.append(col.replace('_Memory_MB', '_Time_s'))
        
        # Sort levels to maintain consistent order
        level_indices = sorted(range(len(available_levels)), key=lambda i: available_levels[i])
        available_levels = [available_levels[i] for i in level_indices]
        memory_cols = [memory_cols[i] for i in level_indices]
        time_cols = [time_cols[i] for i in level_indices]

        # Create new dataframe
        data = []

        for _, row in df11.iterrows():
            model = row['Model']
            row_data = [model]
            
            # Add data for each available optimization level
            for mem_col, time_col in zip(memory_cols, time_cols):
                row_data.extend([row[mem_col], row[time_col]])
            
            data.append(row_data)

        # Define MultiIndex columns dynamically based on available levels
        tuples = []
        approach_names = {
            0: 'Approach #0 [Baseline]',
            1: 'Approach #1', 
            2: 'Approach #2',
            3: 'Approach #3'
        }
        
        for i, level in enumerate(available_levels):
            if level in approach_names:
                approach_name = approach_names[level]
            else:
                approach_name = f'Approach #L{level}'
            tuples.extend([
                (approach_name, 'Mem (MB)'), 
                (approach_name, 'Runtime (s)')
            ])
        columns = pd.MultiIndex.from_tuples(tuples)

        # Final dataframe
        result_df = pd.DataFrame([row[1:] for row in data], index=[row[0] for row in data], columns=columns)

        # Optional: Normalize model names
        result_df.index = [i.replace('.pt','') for i in result_df.index]

        # Sort by model size
        result_df = result_df.iloc[sorted(range(len(result_df)), key=lambda i: self.get_model_size_order(result_df.index[i]))]
        
        return result_df


class PerformanceVisualizer:
    """
    Handles visualization and plotting of YOLO performance metrics.
    """
    
    @staticmethod
    def interactive_select(items, prompt="Select item(s):"):
        """
        Interactively ask user to select one or more elements from a list.
        
        Args:
            items (list): List of items to choose from
            prompt (str): Prompt message to display
            
        Returns:
            list: A list of all items if no selection is made (pressing Return).
                 For a single selection, the list will contain just one item.
        
        Example:
            fruits = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
            
            # For a single selection
            result = PerformanceVisualizer.interactive_select(fruits, "Select one item:")
            single_item = result[0] if result else None
            
            # For multiple selections
            multiple_items = PerformanceVisualizer.interactive_select(fruits, "Select multiple items:")
            
        Note:
            - Pressing Return with no input selects all items
            - Entering 'all' also selects all items
            - Multiple selections can be made with comma or space-separated numbers
        """
        if not items:
            print("No items to select from.")
            return []
        
        # Display items with indices
        print(f"\n{prompt}")
        for i, item in enumerate(items):
            print(f"  [{i+1}] {item}")
        
        print("\nEnter numbers separated by commas or spaces, or 'all' to select everything.")
        print("For a single selection, just enter one number.")
        print("Press Enter to select all items.")
        
        while True:
            try:
                user_input = input("\nYour selection: ").strip()
                
                # Handle empty input - select all
                if not user_input:
                    return items
                
                # Handle 'all' selection
                if user_input.lower() == 'all':
                    return items
                
                # Parse comma or space separated numbers
                selections = []
                for part in user_input.replace(',', ' ').split():
                    num = int(part)
                    if 1 <= num <= len(items):
                        selections.append(items[num-1])
                    else:
                        print(f"Invalid selection: {num}. Please enter numbers between 1 and {len(items)}.")
                        break
                else:
                    return selections
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas or spaces.")

    def create_plots(self, df, output_filename=None, 
                    figsize=(12, 12), interactive_selection=True, 
                    enabled_approaches=None, show_plot=True, show_values=False, log_scale=False):
        """
        Create interactive comparison plots for YOLO model performance metrics across different approaches.
        
        This method generates vertically stacked bar charts comparing memory usage and runtime performance
        across different optimization approaches for YOLO models. It supports interactive approach
        selection and provides comprehensive visualization with value labels, styling, and customization options.
        All text elements use DejaVu Sans font for consistent, professional typography.
        
        The method performs the following visualization tasks:
        1. Sets up colorblind-friendly plotting style and palette
        2. Extracts model names and approach data from MultiIndex DataFrame
        3. Provides interactive or programmatic approach selection
        4. Creates dual subplot layout with memory and runtime comparisons (stacked vertically with shared x-axis)
        5. Generates grouped bar charts with proper spacing and coloring
        6. Optionally adds value labels on top of each bar for precise readings
        7. Applies professional styling with legends, grids, and titles positioned within plot areas using DejaVu Sans font
        8. Displays and/or saves the final visualization (saving is automatic if output_filename is provided)
        
        Parameters
        ----------
        df : pandas.DataFrame
            MultiIndex DataFrame with performance metrics data.
            Expected structure:
            - Index: Model names (e.g., 'yolo11n', 'yolo11s', etc.)
            - Columns: MultiIndex with approach names and metric types
              - Level 0: Approach names ('Approach #0', 'Approach #1', etc.)
              - Level 1: Metric types ('Mem (MB)', 'Runtime (s)')
        
        output_filename : str, optional
            Filename for saving the generated plot. Supports various formats (.png, .jpg, .pdf, etc.)
            If None, the plot will not be saved. If provided, the plot will be automatically saved.
        
        figsize : tuple of float, default=(12, 12)
            Figure size in inches as (width, height) for the matplotlib figure.
        
        interactive_selection : bool, default=True
            Whether to use interactive approach selection via terminal interface.
            If False, uses enabled_approaches parameter or all approaches if None.
        
        enabled_approaches : list of str, optional
            List of approach names to include in the comparison when interactive_selection=False.
            Must be subset of ['Approach #0 [Baseline]', 'Approach #1', 'Approach #2', 'Approach #3'].
            If None and interactive_selection=False, all approaches are included.
        
        show_plot : bool, default=True
            Whether to display the plot using plt.show().
        
        show_values : bool, default=False
            Whether to display value labels on top of each bar for precise readings.
        
        log_scale : bool, default=False
            Whether to use logarithmic scale for Y-axis. When True, both memory and runtime 
            plots will use log scale for better visualization of large value ranges.
        
        Returns
        -------
        tuple
            A tuple containing:
            - matplotlib.figure.Figure: The generated figure object
            - tuple of matplotlib.axes.Axes: The subplot axes (ax1 for memory on top, ax2 for runtime on bottom, sharing x-axis)
            - pandas.DataFrame: The input DataFrame for reference
        
        Raises
        ------
        ValueError
            If no approaches are selected for comparison or if enabled_approaches contains
            invalid approach names.
        
        Notes
        -----
        - Uses colorblind-friendly color palette: blue (#1f77b4), orange (#ff7f0e), 
          green (#2ca02c), red (#d62728)
        - Memory values are displayed with 1 decimal place precision
        - Runtime values are displayed with 4 decimal place precision  
        - Requires interactive_select method when interactive_selection=True
        - Plot style is set to 'seaborn-v0_8' with colorblind-safe seaborn palette
        - Bar charts include white edge lines and 80% opacity for better visual separation
        - Value labels are optionally positioned 1% above the maximum bar height for readability (controlled by show_values parameter)
        - Subplots share x-axis to eliminate redundant model name labels and create cleaner visualization
        - Subplot titles are positioned within the plot areas with white semi-transparent backgrounds for better readability
        - Plot saving is automatically determined: if output_filename is provided, plot is saved and figure is closed
        - If no output_filename is provided, plot is only displayed (if show_plot=True) and figure remains open for interaction
        - Value labels are hidden by default for cleaner appearance; set show_values=True to display them
        - All text elements (titles, labels, legends, tick labels) use DejaVu Sans font for consistent typography
        - Legend is displayed only in the top subplot to avoid redundancy and reduce visual clutter
        - Log scale can be enabled for both Y-axes to better visualize data with large value ranges
        """
        # Determine whether to save plot based on output_filename
        save_plot = output_filename is not None
        
        # Set the style for better looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("colorblind")  # Use colorblind-friendly palette
        
        # Set DejaVu Sans font for all text elements
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # Extract data for plotting
        models = df.index.tolist()
        # Get basename for model names for cleaner x-axis labels
        model_labels = [os.path.basename(model) for model in models]
        # Get available approaches from the dataframe columns
        all_approaches = df.columns.levels[0].tolist()

        # Approach selection logic
        if interactive_selection:
            # Interactive approach selection
            enabled_approaches = self.interactive_select(all_approaches, 
                                                      "Which approaches would you like to compare?")
        else:
            # Use provided approaches or default to all
            if enabled_approaches is None:
                enabled_approaches = all_approaches
            else:
                # Validate provided approaches
                invalid_approaches = [app for app in enabled_approaches if app not in all_approaches]
                if invalid_approaches:
                    raise ValueError(f"Invalid approach names: {invalid_approaches}. "
                                   f"Valid approaches are: {all_approaches}")

        if not enabled_approaches:
            raise ValueError("At least one approach must be selected!")

        memory_data = {app: df[(app, 'Mem (MB)')] for app in enabled_approaches}
        runtime_data = {app: df[(app, 'Runtime (s)')] for app in enabled_approaches}

        # Setup plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        # Colorblind-friendly palette: blue, orange, green, red
        all_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colorblind-safe colors
        colors = [all_colors[all_approaches.index(app)] for app in enabled_approaches]

        x = np.arange(len(models))
        width = 0.5 / len(enabled_approaches)

        # Plot 1: Memory Usage Comparison
        for i, (approach, values) in enumerate(memory_data.items()):
            ax1.bar(x + i * width, values, width, label=approach, color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)

        ax1.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width * (len(enabled_approaches) - 1) / 2)
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Apply log scale if requested
        if log_scale:
            ax1.set_yscale('log')

        # Add value labels on top of bars (if enabled)
        if show_values:
            for i, (approach, values) in enumerate(memory_data.items()):
                for j, v in enumerate(values):
                    ax1.text(j + i * width, v + max(values) * 0.01, f'{v:.1f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Plot 2: Runtime Comparison
        for i, (approach, values) in enumerate(runtime_data.items()):
            ax2.bar(x + i * width, values, width, label=approach, color=colors[i], alpha=0.8, edgecolor='white', linewidth=1)

        ax2.set_xlabel('YOLO Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width * (len(enabled_approaches) - 1) / 2)
        ax2.set_xticklabels(model_labels)
        ax2.grid(True, alpha=0.3, axis='y')

        print(f"[DEBUG]: Model labels: {model_labels}")
        
        # Apply log scale if requested
        if log_scale:
            ax2.set_yscale('log')

        # Add value labels on top of bars (if enabled)
        if show_values:
            for i, (approach, values) in enumerate(runtime_data.items()):
                for j, v in enumerate(values):
                    ax2.text(j + i * width, v + max(values) * 0.01, f'{v:.4f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add subplot titles within plot areas (centered)
        ax1.text(0.5, 0.95, 'Memory Usage Comparison Across Approaches', 
                 transform=ax1.transAxes, fontsize=14, fontweight='bold', 
                 horizontalalignment='center', verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.text(0.5, 0.95, 'Runtime Comparison Across Approaches', 
                 transform=ax2.transAxes, fontsize=14, fontweight='bold', 
                 horizontalalignment='center', verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Adjust layout and add overall title
        plt.tight_layout()
        fig.suptitle('YOLO Model Performance Comparison', fontsize=16, fontweight='bold', y=0.96)
        plt.subplots_adjust(hspace=0.05, top=0.92)

        # Save the plots if filename is provided (do this BEFORE display/close)
        if save_plot:
            try:
                # Ensure the figure is properly drawn before saving
                fig.canvas.draw()
                plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Plots have been generated and saved as '{output_filename}'")
                
                # Verify file was created and has content
                if os.path.exists(output_filename):
                    file_size = os.path.getsize(output_filename)
                    print(f"File size: {file_size} bytes")
                    if file_size > 1000:  # Reasonable minimum size for a plot
                        print("✅ Plot saved successfully!")
                    else:
                        print("⚠️  Warning: Plot file seems unusually small")
                else:
                    print("❌ Error: Plot file was not created")
                    
            except Exception as e:
                print(f"❌ Error saving plot: {e}")
                print(f"Current matplotlib backend: {matplotlib.get_backend()}")

        # Display the plots (after saving)
        if show_plot:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot: {e}")

        print("\nDataFrame Preview:")
        print(df)
        
        # Close the figure after saving to free memory
        if save_plot:
            plt.close(fig)
        
        return fig, (ax1, ax2), df

    @staticmethod
    def verify_image(image_path):
        """
        Verify that the generated image is valid and readable.
        
        Parameters
        ----------
        image_path : str
            Path to the image file to verify
            
        Returns
        -------
        bool
            True if image is valid, False otherwise
        """
        try:
            from PIL import Image
            
            if not os.path.exists(image_path):
                return False
                
            file_size = os.path.getsize(image_path)
            if file_size < 1000:  # Too small to be a valid plot
                return False
                
            # Try to open and verify the image
            with Image.open(image_path) as img:
                img.verify()
                return True
                
        except Exception:
            return False


class PerformanceAnalyzer:
    """
    Main class that combines data processing and visualization for YOLO performance analysis.
    
    This class provides a high-level interface for analyzing YOLO performance metrics,
    combining CSV data processing with visualization capabilities.
    
    Example:
        # Basic usage
        analyzer = PerformanceAnalyzer()
        result_df, fig, axes = analyzer.analyze('metrics.csv', output_filename='plot.png')
        
        # Advanced usage with custom settings
        analyzer = PerformanceAnalyzer()
        result_df, fig, axes = analyzer.analyze(
            'metrics.csv',
            output_filename='plot.png',
            figsize=(16, 10),
            interactive_selection=False,
            enabled_approaches=['Approach #0 [Baseline]', 'Approach #1'],
            show_values=True,
            log_scale=True  # Enable logarithmic Y-axis scale
        )
    """
    
    def __init__(self):
        """Initialize the analyzer with data processor and visualizer components."""
        self.data_processor = PerformanceDataProcessor()
        self.visualizer = PerformanceVisualizer()
    
    def analyze(self, csv_file_path, output_filename=None, 
               figsize=(12, 12), interactive_selection=True, 
               enabled_approaches=None, show_plot=True, show_values=False, log_scale=False):
        """
        Combined method to process performance metrics CSV and create comparison plots.
        
        This method combines the two-step process of:
        1. Processing the CSV file into a structured DataFrame
        2. Creating visualization plots from the processed data
        
        Parameters
        ----------
        csv_file_path : str
            Path to the CSV file containing performance metrics data.
        output_filename : str, optional
            Filename for saving the generated plot. If None, plot won't be saved.
        figsize : tuple of float, default=(12, 12)
            Figure size in inches as (width, height).
        interactive_selection : bool, default=True
            Whether to use interactive approach selection.
        enabled_approaches : list of str, optional
            List of approach names to include when interactive_selection=False.
        show_plot : bool, default=True
            Whether to display the plot.
        show_values : bool, default=False
            Whether to display value labels on bars.
        log_scale : bool, default=False
            Whether to use logarithmic scale for Y-axis in both plots.
        
        Returns
        -------
        tuple
            A tuple containing:
            - pandas.DataFrame: The processed performance metrics data
            - matplotlib.figure.Figure: The generated figure object
            - tuple of matplotlib.axes.Axes: The subplot axes
        
        Raises
        ------
        FileNotFoundError
            If the CSV file doesn't exist.
        ValueError
            If invalid approaches are specified.
        """
        # Check if CSV file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        print(f"Processing CSV file: {csv_file_path}")
        
        # Step 1: Process the CSV file
        result_df = self.data_processor.process_csv(csv_file_path)
        print(f"Successfully processed {len(result_df)} models from CSV")
        
        # Step 2: Create comparison plots
        fig, axes, df_result = self.visualizer.create_plots(
            result_df, 
            output_filename=output_filename,
            figsize=figsize,
            interactive_selection=interactive_selection,
            enabled_approaches=enabled_approaches,
            show_plot=show_plot,
            show_values=show_values,
            log_scale=log_scale
        )
        
        # Verify the output image if one was saved
        if output_filename and os.path.exists(output_filename):
            if self.visualizer.verify_image(output_filename):
                print("🔍 Image verification: ✅ PASSED")
            else:
                print("🔍 Image verification: ❌ FAILED - Image may be corrupted")
        
        return result_df, fig, axes


# Backward compatibility: provide function-based interface for existing code
def get_model_size_order(model_name):
    """Backward compatibility function."""
    return PerformanceDataProcessor.get_model_size_order(model_name)

def interactive_select(items, prompt="Select item(s):"):
    """Backward compatibility function."""
    return PerformanceVisualizer.interactive_select(items, prompt)

def process_performance_metrics_csv(csv_file_path):
    """Backward compatibility function."""
    processor = PerformanceDataProcessor()
    return processor.process_csv(csv_file_path)

def create_performance_comparison_plots(df, output_filename=None, 
                                      figsize=(12, 12), interactive_selection=True, 
                                      enabled_approaches=None, show_plot=True, show_values=False, log_scale=False):
    """Backward compatibility function."""
    visualizer = PerformanceVisualizer()
    return visualizer.create_plots(df, output_filename, figsize, interactive_selection, 
                                  enabled_approaches, show_plot, show_values, log_scale)

def verify_output_image(image_path):
    """Backward compatibility function."""
    return PerformanceVisualizer.verify_image(image_path)

def analyze_performance_metrics(csv_file_path, output_filename=None, 
                              figsize=(12, 12), interactive_selection=True, 
                              enabled_approaches=None, show_plot=True, show_values=False, log_scale=False):
    """Backward compatibility function."""
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze(csv_file_path, output_filename, figsize, interactive_selection,
                           enabled_approaches, show_plot, show_values, log_scale)


def main():
    """Command-line interface for the performance metrics analysis tool."""
    parser = argparse.ArgumentParser(
        description="Process YOLO performance metrics CSV files and create comparison plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv
  %(prog)s data.csv -o output_plot.png
  %(prog)s data.csv -o plot.png --no-interactive --approaches "Approach #0 [Baseline]" "Approach #1"
  %(prog)s data.csv --figsize 16 10 --show-values
  %(prog)s data.csv -o plot.png --log-scale --show-values
        """
    )
    
    # Required arguments
    parser.add_argument(
        'csv_file',
        help='Path to the CSV file containing performance metrics data'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        dest='output_filename',
        help='Output filename for saving the plot (e.g., plot.png, plot.pdf)'
    )
    
    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[12, 12],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 12 12)'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive approach selection'
    )
    
    parser.add_argument(
        '--approaches',
        nargs='+',
        choices=['Approach #0 [Baseline]', 'Approach #1', 'Approach #2', 'Approach #3'],
        help='Specific approaches to include (only used with --no-interactive)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Don\'t display the plot (useful when only saving to file)'
    )
    
    parser.add_argument(
        '--show-values',
        action='store_true',
        help='Show value labels on top of bars'
    )
    
    parser.add_argument(
        '--log-scale',
        action='store_true',
        help='Use logarithmic scale for Y-axis in both memory and runtime plots'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    
    args = parser.parse_args()
    
    try:
        # Use the class-based API
        analyzer = PerformanceAnalyzer()
        result_df, fig, axes = analyzer.analyze(
            csv_file_path=args.csv_file,
            output_filename=args.output_filename,
            figsize=tuple(args.figsize),
            interactive_selection=not args.no_interactive,
            enabled_approaches=args.approaches,
            show_plot=not args.no_show,
            show_values=args.show_values,
            log_scale=args.log_scale
        )
        
        print(f"\nAnalysis completed successfully!")
        print(f"Processed {len(result_df)} models")
        if args.output_filename:
            print(f"Plot saved to: {args.output_filename}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # If called as script, run command-line interface
    main()
else:
    # If imported as module, you can use both class-based and function-based APIs
    # Example usage with class-based API:
    # analyzer = PerformanceAnalyzer()
    # result_df, fig, axes = analyzer.analyze('metrics.csv', output_filename='plot.png')
    
    # Example usage with backward compatibility functions:
    csv1 = "/home/imerit/Testroom/test102/memfix_3/ultralytics/results/performance_metrics_20250630_193515.csv"
    # result_df, fig, axes = analyze_performance_metrics(csv1, output_filename=None)
    pass
