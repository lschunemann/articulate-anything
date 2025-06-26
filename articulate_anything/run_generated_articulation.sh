#!/bin/bash

# Check if argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_type>"
    echo "Example: $0 partnet"
    echo "         $0 artvip"
    exit 1
fi

DATASET_TYPE="$1"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate articulate-anything-clean

# Base directory for segmentation masks
BASE_DIR="/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks"

# Results directory (relative to BASE_DIR)
RESULTS_DIR="/home/link/DreMa/third_party/articulate-anything/results/generated"

# Videos directory for checking MP4 files
VIDEOS_DIR="/home/link/DreMa/third_party/articulate-anything/datasets/RLBench/videos"

# Set view type based on dataset type
if [ "$DATASET_TYPE" = "artvip" ]; then
    VIEW_TYPE="multi-view"
    ARTICULATION_SCRIPT="generated_articulation_image.py"
elif [ "$DATASET_TYPE" = "partnet" ]; then
    VIEW_TYPE="single-view"
    # For partnet, the script will be determined during processing based on MP4 existence
else
    echo "Error: Unknown dataset type: $DATASET_TYPE"
    echo "Supported types: partnet, artvip"
    exit 1
fi

# Find all directories matching the pattern based on dataset type and view type
echo "Searching for directories matching pattern: ${DATASET_TYPE}_*_${VIEW_TYPE}"
echo "Base directory: $BASE_DIR"
echo "Results directory: $RESULTS_DIR"

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory $BASE_DIR does not exist"
    exit 1
fi

# Find matching directories, excluding those with "grounding" in the name
MATCHING_DIRS=$(find "$BASE_DIR" -maxdepth 1 -type d -name "${DATASET_TYPE}_*_${VIEW_TYPE}" | grep -v "grounding" | sort)

if [ -z "$MATCHING_DIRS" ]; then
    echo "No directories found matching pattern ${DATASET_TYPE}_*_${VIEW_TYPE} (excluding grounding) in $BASE_DIR"
    exit 1
fi

echo "Found the following matching directories:"
echo "$MATCHING_DIRS"
echo ""

# Counter for tracking progress
count=0
skipped=0
total=$(echo "$MATCHING_DIRS" | wc -l)

# Process each directory
while IFS= read -r dir; do
    ((count++))
    task_name=$(basename "$dir")
    
    echo "[$count/$total] Processing: $task_name"
    echo "Directory: $dir"
    
    # Check if results already exist
    result_dir="$RESULTS_DIR/$task_name"
    if [ -d "$result_dir" ]; then
        echo "⏭️  Results already exist in $result_dir, skipping..."
        ((skipped++))
        echo "----------------------------------------"
        echo ""
        continue
    fi
    
    # Check if the output directory has required files
    output_dir="$dir/output"
    if [ ! -d "$output_dir" ]; then
        echo "Warning: Output directory $output_dir does not exist, skipping..."
        continue
    fi
    
    # Check for mesh files
    mesh_count=$(find "$output_dir" -name "*.obj" -o -name "*.glb" | wc -l)
    if [ "$mesh_count" -eq 0 ]; then
        echo "Warning: No mesh files (.obj or .glb) found in $output_dir, skipping..."
        continue
    fi
    
    echo "Found $mesh_count mesh files in $output_dir"
    
    # Determine which articulation script to use
    if [ "$DATASET_TYPE" = "partnet" ]; then
        # Check if the MP4 file exists for partnet
        mp4_path="$VIDEOS_DIR/${task_name}.mp4"
        if [ -f "$mp4_path" ]; then
            ARTICULATION_SCRIPT="generated_articulation.py"
            echo "Found MP4 file: $mp4_path"
        else
            ARTICULATION_SCRIPT="generated_articulation_image.py"
            echo "MP4 file not found at $mp4_path, using image-based articulation"
        fi
    fi
    
    # Run the appropriate articulation script
    echo "Running articulation for $task_name using $ARTICULATION_SCRIPT..."
    python examples/$ARTICULATION_SCRIPT "$task_name"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $task_name"
    else
        echo "✗ Failed to process $task_name"
    fi
    
    echo "----------------------------------------"
    echo ""
    
done <<< "$MATCHING_DIRS"

echo "Finished processing all directories."
echo "Total found: $total directories"
echo "Skipped (already processed): $skipped directories"
echo "Actually processed: $((total - skipped)) directories"

# Deactivate conda environment
conda deactivate