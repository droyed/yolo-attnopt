#!/bin/bash

PATH1="assets/ivan-rohovchenko-iNPI5VlSt4o-unsplash.jpg"

# Array of YOLO models
yolo_models=("yolo11n.pt" "yolo11m.pt" "yolo11x.pt")

# Array of optimization levels
#opt_levels=(0 1 2 3)
opt_levels=(0 2 3)

# Add a delay between runs
DELAY=5

# Create logs directory if it doesn't exist
logs_dir="./logs"
mkdir -p "$logs_dir"

# Create a log file with timestamp in the logs directory
log_file="$logs_dir/test_runs_$(date +%Y%m%d_%H%M%S).log"

# Create results directory if it doesn't exist
results_dir="./results"
mkdir -p "$results_dir"

# Create models directory if it doesn't exist
models_dir="./models"
mkdir -p "$models_dir"

# Generate timestamp for CSV files
timestamp=$(date +%Y%m%d_%H%M%S)

# Loop through all combinations
for model in "${yolo_models[@]}"; do
    for level in "${opt_levels[@]}"; do
        echo "Running with model: $model, optimization level: $level" | tee -a "$log_file"

        # Use model from models directory
        model_path="$models_dir/$model"
        python -m benchmark_tools.main_benchmark --image-path $PATH1 --device cuda --yolo-modelname "$model_path" --optimize-level "$level" 2>&1 | tee -a "$log_file"
        
        # Add a small delay between runs (optional)
        sleep $DELAY
        
        echo "----------------------------------------" | tee -a "$log_file"
    done
done

echo "Results saved at - "
echo "$log_file"

# Parse and display performance metrics table
echo ""
echo "=============================================================================================================="
echo "                                    PERFORMANCE METRICS SUMMARY"
echo "=============================================================================================================="
echo ""

# Create dynamic multi-level table header based on tested optimization levels
header_line1="%-15s"
header_line2="%-15s"
header_line3="%-15s"
header_values1=("")
header_values2=("Model")
header_values3=("-----")

for level in "${opt_levels[@]}"; do
    header_line1="$header_line1 %-22s"
    header_line2="$header_line2 %-11s %-10s"
    header_line3="$header_line3 %-11s %-10s"
    header_values1+=("L$level")
    header_values2+=("Memory (MB)" "Time (s)")
    header_values3+=("-----------" "--------")
done

printf "$header_line1\n" "${header_values1[@]}"
printf "$header_line2\n" "${header_values2[@]}"
printf "$header_line3\n" "${header_values3[@]}"

# Declare associative arrays to store data
declare -A memory_data
declare -A time_data
declare -A count_data
current_model=""

# Read the log file line by line to collect data
while IFS= read -r line; do
    # Check if this line contains model name information
    if [[ "$line" =~ yolo_modelname:\ (.+) ]]; then
        current_model=$(echo "$line" | sed -E 's/.*yolo_modelname: (.+)/\1/' | xargs)
    fi
    
    # Check if this is a PROF Block line
    if [[ "$line" =~ ^📺\ PROF\ Block ]]; then
        # Extract optimize level, memory, and time using sed
        opt_level=$(echo "$line" | sed -E 's/.*OPTIMIZE LEVEL # ([0-9]+).*/\1/')
        
        # Handle both MB and GB values and convert to MB for averaging
        if [[ "$line" =~ GB ]]; then
            mem_value=$(echo "$line" | sed -E 's/.*⌇ ([0-9.]+) GB.*/\1/')
            mem_value_mb=$(echo "$mem_value * 1024" | bc -l)
            mem_unit="GB"
        else
            mem_value_mb=$(echo "$line" | sed -E 's/.*⌇ ([0-9.]+) MB.*/\1/')
            mem_unit="MB"
        fi
        
        time_value=$(echo "$line" | sed -E 's/.*⌇ ([0-9.]+)s$/\1/')
        
        # Create keys for associative arrays
        mem_key="${current_model}_${opt_level}_mem"
        time_key="${current_model}_${opt_level}_time"
        count_key="${current_model}_${opt_level}_count"
        unit_key="${current_model}_${opt_level}_unit"
        
        # Accumulate values for averaging
        memory_data[$mem_key]=$(echo "${memory_data[$mem_key]:-0} + $mem_value_mb" | bc -l)
        time_data[$time_key]=$(echo "${time_data[$time_key]:-0} + $time_value" | bc -l)
        count_data[$count_key]=$((${count_data[$count_key]:-0} + 1))
        memory_data[$unit_key]="$mem_unit"
    fi
done < "$log_file"

# Get unique models and sort them
models=$(grep "yolo_modelname:" "$log_file" | sed -E 's/.*yolo_modelname: (.+)/\1/' | sort -u)

# Print averaged data for each model
for model in $models; do
    model=$(echo "$model" | xargs)  # trim whitespace
    
    # Build arrays for the row data
    declare -a row_values=("$model")
    
    # For each optimization level in the tested array  
    for level in "${opt_levels[@]}"; do
        mem_key="${model}_${level}_mem"
        time_key="${model}_${level}_time"
        count_key="${model}_${level}_count"
        unit_key="${model}_${level}_unit"
        
        if [[ ${count_data[$count_key]:-0} -gt 0 ]]; then
            # Calculate averages
            avg_memory=$(echo "scale=2; ${memory_data[$mem_key]} / ${count_data[$count_key]}" | bc -l)
            avg_time=$(echo "scale=4; ${time_data[$time_key]} / ${count_data[$count_key]}" | bc -l)
            
            # Format memory in MB (already converted during parsing)
            mem_display="${avg_memory}"
            
            time_display="${avg_time}"
        else
            mem_display="N/A"
            time_display="N/A"
        fi
        
        row_values+=("$mem_display" "$time_display")
    done
    
    # Print the row with dynamic formatting
    row_format="%-15s"
    for level in "${opt_levels[@]}"; do
        row_format="$row_format %-11s %-10s"
    done
    printf "$row_format\n" "${row_values[@]}"
    unset row_values  # Clear array for next iteration
done

echo ""
echo "=============================================================================================================="

# Generate CSV file
csv_file="$results_dir/performance_metrics_$(date +%Y%m%d_%H%M%S).csv" 

echo ""
echo "Saving table to CSV file: $csv_file"

# Create CSV header dynamically based on optimization levels
csv_header="Model"
for level in "${opt_levels[@]}"; do
    csv_header="$csv_header,L${level}_Memory_MB,L${level}_Time_s"
done
echo "$csv_header" > "$csv_file"

# Generate CSV data for each model (reuse existing data)
for model in $models; do
    model=$(echo "$model" | xargs)  # trim whitespace
    
    # Build CSV row
    csv_row="$model"
    
    # For each optimization level in the tested array
    for level in "${opt_levels[@]}"; do
        mem_key="${model}_${level}_mem"
        time_key="${model}_${level}_time"
        count_key="${model}_${level}_count"
        
        if [[ ${count_data[$count_key]:-0} -gt 0 ]]; then
            # Calculate averages
            avg_memory=$(echo "scale=2; ${memory_data[$mem_key]} / ${count_data[$count_key]}" | bc -l)
            avg_time=$(echo "scale=4; ${time_data[$time_key]} / ${count_data[$count_key]}" | bc -l)
            
            csv_row="$csv_row,$avg_memory,$avg_time"
        else
            csv_row="$csv_row,N/A,N/A"
        fi
    done
    
    # Append row to CSV file
    echo "$csv_row" >> "$csv_file"
done
echo "CSV file saved: $csv_file"
