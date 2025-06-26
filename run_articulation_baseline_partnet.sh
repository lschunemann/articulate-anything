#!/bin/bash

# Activate the conda environment
echo "Activating conda environment: articulate-anything-clean"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate articulate-anything-clean

# Create a list to store all the numbers
numbers=()

# Arrays to track success and failure
failed_numbers=()
success_numbers=()
skipped_numbers=()
success_count=0
failure_count=0
skipped_count=0

# Create log directory if it doesn't exist
log_dir="partnet_articulation_logs"
mkdir -p "$log_dir"

# Find all directories that are just numbers
echo "Scanning for number directories in datasets/partnet-mobility-v0/dataset..."
dataset_dir="datasets/partnet-mobility-v0/dataset"
result_base_dir="results"

if [ ! -d "$dataset_dir" ]; then
    echo "Error: Dataset directory $dataset_dir does not exist"
    exit 1
fi

numbers=()
skipped_numbers=()
skipped_count=0

for dir in "$dataset_dir"/*; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Extract the directory name
        dirname=$(basename "$dir")
        # Check if the directory name is just a number
        if [[ "$dirname" =~ ^[0-9]+$ ]]; then
            # Check if there's a corresponding result directory with URDF files
            result_dir="${result_base_dir}/${dirname}/joint_actor"
            
            # Flag to track if URDF is found
            urdf_found=false
            
            # If the result directory exists
            if [ -d "$result_dir" ]; then
                # Check for URDF files in iter_*/seed_* subdirectories
                if find "$result_dir" -path "*/iter_*/seed_*/*.urdf" -print -quit | grep -q .; then
                    echo "Skipping already processed directory: $dirname (contains .urdf file in results)"
                    skipped_numbers+=("$dirname")
                    ((skipped_count++))
                    urdf_found=true
                fi
            fi
            
            # If no URDF file was found, add to the list of numbers to process
            if [ "$urdf_found" = false ]; then
                echo "Found unprocessed number directory: $dirname"
                numbers+=("$dirname")
            fi
        else
            echo "Skipping non-number directory: $dirname"
        fi
    fi
done

# Find all directories matching the exact pattern
# echo "Scanning for matching directories..."
# for dir in datasets/segmentation_masks/partnet_*_single-view; do
#     # Check if the directory exists
#     if [ -d "$dir" ]; then
#         # Use regex to check for exact pattern match (not containing "grounding")
#         if [[ "$dir" =~ datasets/segmentation_masks/partnet_([0-9]+)_single-view$ ]]; then
#             number=${BASH_REMATCH[1]}
#             echo "Found directory: $dir with number: $number"
#             numbers+=("$number")
#         else
#             echo "Skipping non-matching directory: $dir"
#         fi
#     fi
# done

# Process each number
echo "Found ${#numbers[@]} unprocessed number directories to process"
echo "Skipped ${skipped_count} already processed directories"

for number in "${numbers[@]}"; do
    echo "-----------------------------------------------------"
    echo "Processing prompt number: $number"
    echo "Running: python articulate.py modality=partnet prompt=$number out_dir=results additional_prompt=joint_0"
    echo "-----------------------------------------------------"
    
    # Create log file for this run
    log_file="${log_dir}/articulate_${number}.log"
    
    # Run the Python command directly and redirect output to log file
    python articulate.py modality=partnet prompt=$number out_dir=results additional_prompt=joint_0 > "$log_file" 2>&1
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed number: $number"
        success_numbers+=("$number")
        ((success_count++))
    else
        echo "❌ Failed to process number: $number"
        failed_numbers+=("$number")
        ((failure_count++))
        
        # Check if it's specifically a link placement error
        if grep -q "No solution found for link placement" "$log_file"; then
            echo "   Error: No solution found for link placement"
        else
            echo "   Error: Check log file for details: $log_file"
        fi
    fi
    
    echo "Completed processing for number: $number"
    echo ""
done

# Deactivate the conda environment when done
conda deactivate

# Print summary
echo "===== PROCESSING SUMMARY ====="
echo "Total number directories found: $((${#numbers[@]} + skipped_count))"
echo "Skipped (already processed): $skipped_count"
echo "Attempted to process: ${#numbers[@]}"
echo "Successful: $success_count"
echo "Failed: $failure_count"

# Write summary to file
summary_file="${log_dir}/processing_summary.txt"
{
    echo "===== PROCESSING SUMMARY ====="
    echo "Total number directories found: $((${#numbers[@]} + skipped_count))"
    echo "Skipped (already processed): $skipped_count"
    echo "Attempted to process: ${#numbers[@]}"
    echo "Successful: $success_count"
    echo "Failed: $failure_count"
    echo ""
    
    if [ ${#skipped_numbers[@]} -gt 0 ]; then
        echo "Skipped (already processed) prompt numbers:"
        printf '%s\n' "${skipped_numbers[@]}"
        echo ""
    fi
    
    if [ ${#success_numbers[@]} -gt 0 ]; then
        echo "Successful prompt numbers:"
        printf '%s\n' "${success_numbers[@]}"
        echo ""
    fi
    
    if [ ${#failed_numbers[@]} -gt 0 ]; then
        echo "Failed prompt numbers:"
        printf '%s\n' "${failed_numbers[@]}"
    fi
} > "$summary_file"

if [ ${#failed_numbers[@]} -gt 0 ]; then
    echo "Failed prompt numbers: ${failed_numbers[*]}"
    
    # Write failed numbers to a separate file (one per line for easier processing)
    printf '%s\n' "${failed_numbers[@]}" > "${log_dir}/failed_prompts.txt"
    echo "Failed prompt numbers saved to ${log_dir}/failed_prompts.txt"
fi

# Write skipped numbers to a separate file
if [ ${#skipped_numbers[@]} -gt 0 ]; then
    printf '%s\n' "${skipped_numbers[@]}" > "${log_dir}/skipped_prompts.txt"
    echo "Skipped prompt numbers saved to ${log_dir}/skipped_prompts.txt"
fi

echo "Processing summary saved to $summary_file"
echo "Logs saved to $log_dir directory"
echo "Finished processing all directories"