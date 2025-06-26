#!/bin/bash

# Check if at least one object is provided
if [ $# -eq 0 ]; then
    echo "Error: No objects provided"
    echo "Usage: $0 object1 object2 object3 ..."
    exit 1
fi

# Use conda shell hook to enable conda environment activation in scripts
eval "$(conda shell.bash hook)"

# overwrite existing results
overwrite=false #false

# Track overall success
success_count=0
failure_count=0
failed_objects=()

# Track failures per step
step1_failures=0
step2_failures=0
step3_failures=0
step4_failures=0

# Loop through each object provided as command-line arguments
for object in "$@"; do
    echo "=========================================="
    echo "Processing object: $object"
    echo "=========================================="

    # Check if .obj files already exist in segmentation output directory
    segmentation_output_dir="/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/${object}/output"
    if ls "$segmentation_output_dir"/*.obj 1> /dev/null 2>&1 && [ "$overwrite" != "true" ]; then
        echo "Segmentation .obj files already exist for $object, skipping entire processing pipeline"
        success_count=$((success_count + 1))
        echo "=========================================="
        continue
    fi
    
    # Flag to track if the current object processing failed at any step
    object_failed=false
    
    # Step 1: Check if output already exists, then run render_mesh.py in Blender if needed
    echo "Step 1: Checking for existing render results"
    output_dir="/home/link/DreMa/third_party/articulate-anything/datasets/output_views/${object}"
    if ls "$output_dir"/*.npz 1> /dev/null 2>&1 && [ "$overwrite" != "true" ]; then
        echo "Render results already exist for $object, skipping render_mesh.py"
    else
        echo "Running render_mesh.py"
        if blender --background --python articulate_anything/render_mesh.py -- "$object"; then
            echo "Step 1 completed successfully"
        else
            echo "Step 1 failed for object: $object"
            object_failed=true
            step1_failures=$((step1_failures + 1))
            # Skip to next object
            conda deactivate
            failed_objects+=("$object")
            failure_count=$((failure_count + 1))
            echo "Skipping remaining steps for $object"
            echo "=========================================="
            continue
        fi
    fi

    # Step 1.1: Copy rendered front view to dataset directory instead of front view of original (since mesh is now original)
    # source_dir="/home/link/DreMa/third_party/articulate-anything/datasets/output_views/${object}"
    source_dir="/home/link/DreMa/third_party/articulate-anything/datasets/ArtVIP/multiview_renders"
    target_dir="/home/link/DreMa/third_party/articulate-anything/datasets/partnet-mobility-v0/dataset/${object}"
    # source_file="${source_dir}/render_${object}_6_0001.png"
    object_path=${object#*_}
    object_path=${object_path%_*}
    source_file="${source_dir}/${object_path}/front_view.png"
    target_file="${target_dir}/robot_frontview.png"

    # Create target directory if it doesn't exist
    mkdir -p "$target_dir"
    
    # Copy the file
    if cp "$source_file" "$target_file"; then
        echo "Successfully copied front view image to dataset directory"
    else
        echo "Failed to copy front view image for object: $object"
        echo "Skipping remaining steps for $object"
        echo "=========================================="
        continue
    fi


    # Step 2: Activate articulate-anything-clean environment and run detect_articulated_parts.py
    echo "Step 2: Running detect_articulated_parts_image.py"
    conda activate articulate-anything-clean
    if python articulate_anything/detect_articulated_parts_image.py "$object"; then
        echo "Step 2 completed successfully"
    else
        echo "Step 2 failed for object: $object"
        object_failed=true
        step2_failures=$((step2_failures + 1))
        # Skip to next object
        conda deactivate
        failed_objects+=("$object")
        failure_count=$((failure_count + 1))
        echo "Skipping remaining steps for $object"
        echo "=========================================="
        continue
    fi
    echo "------------------------------------------"

    echo "------------------------------------------"
    
    # Step 3: Still in sam environment, run mesh_segmentation.py
    echo "Step 3: Running mesh_segmentation.py"
    conda activate sam
    # if python articulate_anything/mesh_segmentation.py "$object"; then
    if python articulate_anything/mesh_seg_dino-x.py "$object"; then
        echo "Step 3 completed successfully"
    else
        echo "Step 3 failed for object: $object"
        object_failed=true
        step3_failures=$((step3_failures + 1))
        # Skip to next object
        conda deactivate
        failed_objects+=("$object")
        failure_count=$((failure_count + 1))
        echo "Skipping remaining steps for $object"
        echo "=========================================="
        # break
        continue
    fi
    echo "------------------------------------------"
    
    # Step 4: Still in sam environment, run 2d_to_3d.py
    echo "Step 4: Running 2d_to_3d.py"
    if python articulate_anything/2d_to_3d.py "$object"; then
        echo "Step 4 completed successfully"
    else
        echo "Step 4 failed for object: $object"
        object_failed=true
        step4_failures=$((step4_failures + 1))
        # Mark as failed but continue to cleanup
        failed_objects+=("$object")
        failure_count=$((failure_count + 1))
    fi
    echo "------------------------------------------"
    
    # Deactivate the conda environment
    conda deactivate
    
    # If all steps succeeded for this object
    if [ "$object_failed" = false ]; then
        echo "Successfully completed all processing for: $object"
        success_count=$((success_count + 1))
    else
        echo "Processing completed with errors for: $object"
    fi
    echo "=========================================="
done  

# Print summary
echo "Processing complete!"
echo "Successfully processed: $success_count objects"
echo "Failed: $failure_count objects"

echo ""
echo "Failures by step:"
echo "- Step 1 (detect_articulated_parts.py): $step1_failures failures"
echo "- Step 2 (render_mesh.py): $step2_failures failures"
echo "- Step 3 (mesh_segmentation.py): $step3_failures failures"
echo "- Step 4 (2d_to_3d.py): $step4_failures failures"

if [ ${#failed_objects[@]} -gt 0 ]; then
    echo ""
    echo "Failed objects:"
    for failed in "${failed_objects[@]}"; do
        echo "- $failed"
    done
fi