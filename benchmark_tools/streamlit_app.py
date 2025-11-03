#!/usr/bin/env python3
"""
Streamlit App for YOLO Performance Metrics Analysis

A minimal Streamlit web interface for analyzing YOLO model performance metrics.
Provides interactive approach and model selection with real-time plot generation.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import os
import io
import datetime
try:
    from .yolo_performance_analyzer import PerformanceDataProcessor
except ImportError:
    from yolo_performance_analyzer import PerformanceDataProcessor

# Configure matplotlib for web display
matplotlib.use('Agg')
plt.ioff()  # Turn off interactive mode

# Configure page
st.set_page_config(
    page_title="YOLO Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stRadio > div {
        flex-direction: row;
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def get_csv_files():
    """Get all CSV files from results directory"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return [], "Results directory not found. Please run performance benchmarks first."
    
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('performance_metrics_') and f.endswith('.csv')]
    if not csv_files:
        return [], "No performance metrics CSV files found in results directory."
    
    # Sort by name alphabetically
    csv_files.sort()
    return csv_files, None

def load_csv_data(csv_filename):
    """Load and cache CSV data from selected file"""
    results_dir = "results"
    csv_path = os.path.join(results_dir, csv_filename)
    
    try:
        # Use the analyzer's data processor
        processor = PerformanceDataProcessor()
        df = processor.process_csv(csv_path)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV file: {str(e)}"

def create_plot(df, selected_approaches, selected_models, show_values=False, log_scale=False):
    """Create performance comparison plot with optional logarithmic Y-axis scaling"""
    try:
        # Filter data based on selections
        if selected_models:
            df_filtered = df.loc[selected_models]
        else:
            df_filtered = df
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("colorblind")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # Extract data for plotting
        models = df_filtered.index.tolist()
        # Get basename for model names for cleaner x-axis labels
        model_labels = [os.path.basename(model) for model in models]
        
        # Validate and filter approaches
        available_approaches = df_filtered.columns.levels[0].tolist()
        valid_approaches = [app for app in selected_approaches if app in available_approaches]
        
        if not valid_approaches:
            return None, "No valid approaches selected or available in data."
        
        memory_data = {app: df_filtered[(app, 'Mem (MB)')] for app in valid_approaches}
        runtime_data = {app: df_filtered[(app, 'Runtime (s)')] for app in valid_approaches}
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Color mapping
        all_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        colors = [all_colors[available_approaches.index(app)] for app in valid_approaches]
        
        x = np.arange(len(models))
        width = 0.6 / len(valid_approaches) if len(valid_approaches) > 1 else 0.6
        
        # Memory Usage Plot
        for i, (approach, values) in enumerate(memory_data.items()):
            offset = (i - (len(valid_approaches) - 1) / 2) * width
            ax1.bar(x + offset, values, width, label=approach, color=colors[i], 
                   alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add value labels if requested
            if show_values:
                for j, v in enumerate(values):
                    ax1.text(j + offset, v + max(values) * 0.01, f'{v:.1f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # Apply log scale if requested
        if log_scale:
            ax1.set_yscale('log')
        
        # Runtime Plot
        for i, (approach, values) in enumerate(runtime_data.items()):
            offset = (i - (len(valid_approaches) - 1) / 2) * width
            ax2.bar(x + offset, values, width, label=approach, color=colors[i], 
                   alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add value labels if requested
            if show_values:
                for j, v in enumerate(values):
                    ax2.text(j + offset, v + max(values) * 0.01, f'{v:.4f}', 
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('YOLO Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_labels, rotation=45 if len(models) > 4 else 0)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_title('Runtime Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # Apply log scale if requested
        if log_scale:
            ax2.set_yscale('log')
        
        # Overall title
        #fig.suptitle('YOLO Model Performance Comparison', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        return fig, None
        
    except Exception as e:
        return None, f"Error creating plot: {str(e)}"

def fig_to_bytes(fig, format='png', dpi=300):
    """Convert matplotlib figure to bytes for download"""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return buf.getvalue()

def main():
    """Main Streamlit app"""
    
# No main header - start directly with content
    
    # Get available CSV files
    csv_files, csv_error = get_csv_files()
    
    if csv_error:
        st.error(f"❌ {csv_error}")
        st.info("💡 **How to get data:**\n1. Run `./run_combinations.sh` to generate performance metrics\n2. CSV files will be created in the `results/` directory")
        return
    
    if not csv_files:
        st.error("No CSV files available to analyze.")
        return
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # CSV File Selection
    st.sidebar.subheader("📁 Select CSV File")
    
    # Create display names with file info
    csv_options = []
    results_dir = "results"
    for csv_file in csv_files:
        file_path = os.path.join(results_dir, csv_file)
        file_size = os.path.getsize(file_path)
        file_date = os.path.getctime(file_path)
        date_str = datetime.datetime.fromtimestamp(file_date).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a more readable display name
        display_name = f"{csv_file}"
        csv_options.append((display_name, csv_file, file_size, date_str))
    
    # Create selectbox with file info
    selected_csv_display = st.sidebar.selectbox(
        "Choose a CSV file:",
        options=[opt[0] for opt in csv_options],
        format_func=lambda x: x,
        key="csv_selector"
    )
    
    # Get the actual filename
    selected_csv = next(opt[1] for opt in csv_options if opt[0] == selected_csv_display)
    
    # Show file details
    selected_option = next(opt for opt in csv_options if opt[0] == selected_csv_display)
    st.sidebar.info(f"**File:** {selected_option[1]}\n**Size:** {selected_option[2]:,} bytes\n**Date:** {selected_option[3]}")
    
    # Load data from selected CSV
    with st.spinner("Loading selected CSV data..."):
        df, load_error = load_csv_data(selected_csv)
    
    if load_error:
        st.error(f"❌ {load_error}")
        return
    
    if df is None or df.empty:
        st.error("No data available in the selected CSV file.")
        return
    
    # Get available approaches and models
    available_approaches = df.columns.levels[0].tolist()
    available_models = df.index.tolist()
    
    # Approach Selection
    st.sidebar.subheader("🔧 Select Approaches")
    selected_approaches = []
    
    for approach in available_approaches:
        if st.sidebar.checkbox(approach, value=True, key=f"approach_{approach}"):
            selected_approaches.append(approach)
    
    # Model Selection  
    st.sidebar.subheader("🤖 Select Models")
    model_selection_type = st.sidebar.radio(
        "Selection Type:",
        ["All Models", "Custom Selection"],
        key="model_selection_type"
    )
    
    if model_selection_type == "All Models":
        selected_models = available_models
    else:
        selected_models = []
        for model in available_models:
            if st.sidebar.checkbox(model, value=True, key=f"model_{model}"):
                selected_models.append(model)
    
    # Additional options
    st.sidebar.subheader("⚙️ Display Options")
    show_values = st.sidebar.checkbox("Show values on bars", value=False)
    log_scale = st.sidebar.checkbox("Use logarithmic Y-axis scale", value=False, 
                                   help="Enable log scale for better visualization of large value ranges")
    
    # Validation
    if not selected_approaches:
        st.warning("⚠️ Please select at least one approach.")
        return
        
    if not selected_models:
        st.warning("⚠️ Please select at least one model.")
        return
    
    # Plot takes full width
    # Header with download options
    plot_header_col1, plot_header_col2 = st.columns([3, 1])
    
    with plot_header_col1:
        st.subheader("📊 Performance Comparison")
    
    # Generate plot first to get figure for download buttons
    with st.spinner("Generating plot..."):
        fig, plot_error = create_plot(df, selected_approaches, selected_models, show_values, log_scale)
    
    if plot_error:
        st.error(f"❌ {plot_error}")
        return
    
    if fig:
        with plot_header_col2:
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"yolo_performance_comparison_{timestamp}.png"
            hd_filename = f"yolo_performance_comparison_{timestamp}_HD.png"
            
            # Get download bytes
            plot_bytes = fig_to_bytes(fig, format='png', dpi=300)
            plot_bytes_hd = fig_to_bytes(fig, format='png', dpi=600)
            
            # Download buttons in top right - horizontal layout
            download_btn_col1, download_btn_col2 = st.columns(2)
            
            with download_btn_col1:
                st.download_button(
                    label="📥 PNG",
                    data=plot_bytes,
                    file_name=default_filename,
                    mime="image/png",
                    help="Download as PNG (300 DPI)",
                    width='stretch'
                )
            
            with download_btn_col2:
                st.download_button(
                    label="📥 HD",
                    data=plot_bytes_hd,
                    file_name=hd_filename,
                    mime="image/png",
                    help="Download as HD PNG (600 DPI)",
                    width='stretch'
                )
        
        # Display plot
        st.pyplot(fig, width='stretch')
        
        plt.close(fig)  # Clean up memory
    else:
        st.error("Failed to generate plot.")
    
    # Data preview
    st.subheader("📄 Raw Data Preview")
    with st.expander("View filtered data"):
        if selected_models:
            preview_df = df.loc[selected_models, [col for col in df.columns if col[0] in selected_approaches]]
        else:
            preview_df = df[[col for col in df.columns if col[0] in selected_approaches]]
        st.dataframe(preview_df, width='stretch')

if __name__ == "__main__":
    main()
