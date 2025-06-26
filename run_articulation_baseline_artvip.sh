#!/bin/bash

# Activate the conda environment
echo "Activating conda environment: articulate-anything-clean"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate articulate-anything-clean

# Create a list to store all the object IDs
object_ids=()

# Arrays to track success and failure
failed_object_ids=()
success_object_ids=()
skipped_object_ids=()
success_count=0
failure_count=0
skipped_count=0

# Create log directory if it doesn't exist
log_dir="partnet_articulation_logs"
mkdir -p "$log_dir"

# Find all directories matching the ArtVIP pattern
echo "Scanning for ArtVIP directories..."
for dir in datasets/output_views/artvip_*; do
    # Check if the directory exists
    if [ -d "$dir" ]; then
        # Extract the full directory name (ArtVIP_object_id)
        dirname=$(basename "$dir")
        # Check if it matches the ArtVIP pattern
        if [[ "$dirname" =~ ^artvip_.+$ ]]; then
            echo "Found directory: $dir with object ID: $dirname"
            object_ids+=("$dirname")
        else
            echo "Skipping non-matching directory: $dir"
        fi
    fi
done

# Process each object ID
echo "Found ${#object_ids[@]} ArtVIP directories to process"
echo "Skipped ${skipped_count} already processed directories"

for object_id in "${object_ids[@]}"; do
    echo "-----------------------------------------------------"
    echo "Processing object ID: $object_id"
    echo "Running: python articulate.py modality=image prompt=$object_id out_dir=results"
    echo "-----------------------------------------------------"
    
    # Create log file for this run
    log_file="${log_dir}/articulate_${object_id}.log"
    
    # Run the Python command directly and redirect output to log file
    # python articulate.py modality=image prompt=$object_id out_dir=results > "$log_file" 2>&1
    python examples/baseline_artvip_articulate.py $object_id > "$log_file" 2>&1
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✅ Successfully processed object ID: $object_id"
        success_object_ids+=("$object_id")
        ((success_count++))
    else
        echo "❌ Failed to process object ID: $object_id"
        failed_object_ids+=("$object_id")
        ((failure_count++))
        
        # Check if it's specifically a link placement error
        if grep -q "No solution found for link placement" "$log_file"; then
            echo "   Error: No solution found for link placement"
        else
            echo "   Error: Check log file for details: $log_file"
        fi
    fi
    
    echo "Completed processing for object ID: $object_id"
    echo ""
done

# Deactivate the conda environment when done
conda deactivate

# Print summary
echo "===== PROCESSING SUMMARY ====="
echo "Total ArtVIP directories found: $((${#object_ids[@]} + skipped_count))"
echo "Skipped (already processed): $skipped_count"
echo "Attempted to process: ${#object_ids[@]}"
echo "Successful: $success_count"
echo "Failed: $failure_count"

# Write summary to file
summary_file="${log_dir}/processing_summary.txt"
{
    echo "===== PROCESSING SUMMARY ====="
    echo "Total ArtVIP directories found: $((${#object_ids[@]} + skipped_count))"
    echo "Skipped (already processed): $skipped_count"
    echo "Attempted to process: ${#object_ids[@]}"
    echo "Successful: $success_count"
    echo "Failed: $failure_count"
    echo ""
    
    if [ ${#skipped_object_ids[@]} -gt 0 ]; then
        echo "Skipped (already processed) object IDs:"
        printf '%s\n' "${skipped_object_ids[@]}"
        echo ""
    fi
    
    if [ ${#success_object_ids[@]} -gt 0 ]; then
        echo "Successful object IDs:"
        printf '%s\n' "${success_object_ids[@]}"
        echo ""
    fi
    
    if [ ${#failed_object_ids[@]} -gt 0 ]; then
        echo "Failed object IDs:"
        printf '%s\n' "${failed_object_ids[@]}"
    fi
} > "$summary_file"

if [ ${#failed_object_ids[@]} -gt 0 ]; then
    echo "Failed object IDs: ${failed_object_ids[*]}"
    
    # Write failed object IDs to a separate file (one per line for easier processing)
    printf '%s\n' "${failed_object_ids[@]}" > "${log_dir}/failed_object_ids.txt"
    echo "Failed object IDs saved to ${log_dir}/failed_object_ids.txt"
fi

# Write skipped object IDs to a separate file
if [ ${#skipped_object_ids[@]} -gt 0 ]; then
    printf '%s\n' "${skipped_object_ids[@]}" > "${log_dir}/skipped_object_ids.txt"
    echo "Skipped object IDs saved to ${log_dir}/skipped_object_ids.txt"
fi

echo "Processing summary saved to $summary_file"
echo "Logs saved to $log_dir directory"
echo "Finished processing all directories"