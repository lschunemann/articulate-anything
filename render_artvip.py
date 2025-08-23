import os
import sys
from pathlib import Path
import traceback
from pxr import Usd, UsdGeom, Gf
import numpy as np
from PIL import Image, ImageDraw

def validate_usd_file(usd_path):
    """Check if a USD file is valid and can be opened"""
    try:
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return False, "Failed to open stage"
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def simple_render_usd(usd_path, output_path, width=512, height=512):
    """Render a simplified front view of a USD file with error handling"""
    # Validate the USD file first
    is_valid, error_msg = validate_usd_file(usd_path)
    if not is_valid:
        print(f"Error with {usd_path}: {error_msg}")
        
        # Create a placeholder error image
        img = Image.new('RGB', (width, height), color='lightgray')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Error: Invalid USD file", fill='red')
        draw.text((10, 30), os.path.basename(usd_path), fill='black')
        
        # Save the error image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return False
    
    try:
        # Open the USD stage
        stage = Usd.Stage.Open(usd_path)
        
        # Create blank image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Get bounding box of the stage
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        root_prim = stage.GetPseudoRoot()
        bound = bbox_cache.ComputeWorldBound(root_prim)
        
        # Correct way to get min and max from BBox3d
        bbox_range = bound.GetRange()
        min_point = bbox_range.GetMin()
        max_point = bbox_range.GetMax()
        
        # Check if we have valid bounds
        if bbox_range.IsEmpty():
            draw.text((10, 10), "Warning: Empty geometry", fill='orange')
            draw.text((10, 30), os.path.basename(usd_path), fill='black')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            return False
        
        # Get the size of the bounding box
        size_x = max_point[0] - min_point[0]
        size_y = max_point[1] - min_point[1]
        size_z = max_point[2] - min_point[2]
        
        # Projection constants - use the largest dimension for scaling
        max_dimension = max(size_x, size_y)
        if max_dimension <= 0:
            # Handle case where dimensions are invalid
            scale = 1.0
        else:
            scale = min(width, height) * 0.8 / max_dimension
        
        center_x, center_y = width/2, height/2
        
        # Center point of the model
        model_center = Gf.Vec3d(
            (min_point[0] + max_point[0]) / 2,
            (min_point[1] + max_point[1]) / 2,
            (min_point[2] + max_point[2]) / 2
        )
        
        # Count of actual meshes rendered
        mesh_count = 0
        
        # Extract mesh data and draw a simplified version
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                faces = mesh.GetFaceVertexIndicesAttr().Get()
                face_counts = mesh.GetFaceVertexCountsAttr().Get()
                
                if points is not None and len(points) > 0:
                    mesh_count += 1
                    
                    # Draw points
                    for point in points:
                        # Center the point relative to model center
                        centered_x = point[0] - model_center[0]
                        centered_y = point[1] - model_center[1]
                        
                        # Project to 2D
                        x = center_x + centered_x * scale
                        y = center_y - centered_y * scale  # Flip Y axis
                        
                        draw.ellipse((x-1, y-1, x+1, y+1), fill='black')
                    
                    # Draw edges if we have face data
                    if faces is not None and face_counts is not None:
                        idx = 0
                        for count in face_counts:
                            face_indices = faces[idx:idx+count]
                            idx += count
                            
                            # Draw edges of the face
                            for i in range(count):
                                pt1 = points[face_indices[i]]
                                pt2 = points[face_indices[(i+1) % count]]
                                
                                # Center the points
                                x1 = center_x + (pt1[0] - model_center[0]) * scale
                                y1 = center_y - (pt1[1] - model_center[1]) * scale
                                x2 = center_x + (pt2[0] - model_center[0]) * scale
                                y2 = center_y - (pt2[1] - model_center[1]) * scale
                                
                                draw.line((x1, y1, x2, y2), fill='blue', width=1)
        
        # If no meshes were found, note this in the image
        if mesh_count == 0:
            draw.text((10, 10), "Warning: No mesh data found", fill='orange')
        else:
            # Add bounding box info for debugging
            draw.text((10, 10), f"Meshes: {mesh_count}", fill='green')
            draw.text((10, 30), f"Size: {size_x:.2f} x {size_y:.2f} x {size_z:.2f}", fill='green')
        
        # Add filename for reference
        draw.text((10, height-20), os.path.basename(usd_path), fill='black')
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return True
    
    except Exception as e:
        print(f"Error rendering {usd_path}: {e}")
        traceback.print_exc()
        
        # Create an error image
        img = Image.new('RGB', (width, height), color='lightgray')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Error: {str(e)[:50]}...", fill='red')
        draw.text((10, 30), os.path.basename(usd_path), fill='black')
        
        # Save the error image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return False

def find_primary_usd_files(base_dir):
    """
    Find primary USD files in object folders.
    Only process the USD file at the object folder level, not in subfolders.
    """
    primary_usd_files = []
    
    # Walking through the directory structure
    for root, dirs, files in os.walk(base_dir):
        # Check if this directory contains USD files
        usd_files = [f for f in files if f.endswith(('.usd', '.usda'))]
        
        if usd_files:
            # This directory contains USD files
            # Find the primary USD file (typically named model_*.usd or model_*.usda)
            primary_file = None
            
            # First, try to find a file starting with "model_"
            for file in usd_files:
                if file.startswith("model_"):
                    primary_file = os.path.join(root, file)
                    break
            
            # If no model_* file found, just use the first USD file
            if primary_file is None and usd_files:
                primary_file = os.path.join(root, usd_files[0])
            
            if primary_file:
                # Get the object folder name
                path_parts = Path(primary_file).parts
                object_folder = path_parts[-2]  # Parent folder of the .usd file
                
                primary_usd_files.append((primary_file, object_folder))
                
                # Skip subdirectories of this directory, as they might contain
                # texture/material USD files we don't want to process
                dirs.clear()  # This modifies os.walk behavior to skip subdirectories
    
    return primary_usd_files

def process_usd_files(base_dir, output_base_dir):
    """Process primary USD files, handling errors gracefully"""
    # Track statistics
    total_files = 0
    successful_renders = 0
    failed_renders = 0
    
    # Create a log file
    log_path = os.path.join(output_base_dir, "render_log.txt")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find primary USD files
    primary_usd_files = find_primary_usd_files(base_dir)
    
    with open(log_path, 'w') as log_file:
        log_file.write("USD Rendering Log\n")
        log_file.write("================\n\n")
        
        for usd_file_path, object_folder in primary_usd_files:
            total_files += 1
            
            # Create output directory with required naming
            output_dir = os.path.join(output_base_dir, f"artvip_{object_folder}_single-view")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "robot_frontview.png")
            
            print(f"Processing {total_files}/{len(primary_usd_files)}: {usd_file_path}")
            log_file.write(f"File: {usd_file_path}\n")
            
            # Try to validate the file first
            is_valid, error_msg = validate_usd_file(usd_file_path)
            if not is_valid:
                log_file.write(f"  Validation failed: {error_msg}\n")
                failed_renders += 1
                
                # Create error image
                img = Image.new('RGB', (512, 512), color='lightgray')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"Error: Invalid USD file", fill='red')
                draw.text((10, 30), f"{error_msg[:50]}...", fill='red')
                draw.text((10, 50), os.path.basename(usd_file_path), fill='black')
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path)
                continue
            
            # Attempt to render
            success = simple_render_usd(usd_file_path, output_path)
            if success:
                log_file.write("  Render successful\n")
                successful_renders += 1
            else:
                log_file.write("  Render failed\n")
                failed_renders += 1
            
            log_file.write("\n")
        
        # Write summary
        log_file.write("\nSummary\n")
        log_file.write("=======\n")
        log_file.write(f"Total USD files: {total_files}\n")
        log_file.write(f"Successful renders: {successful_renders}\n")
        log_file.write(f"Failed renders: {failed_renders}\n")
    
    print("\nRendering complete!")
    print(f"Total USD files: {total_files}")
    print(f"Successful renders: {successful_renders}")
    print(f"Failed renders: {failed_renders}")
    print(f"See log file for details: {log_path}")

# Usage
base_dir = "datasets/ArtVIP/Articulated_objects"
output_base_dir = "rendered_views"
process_usd_files(base_dir, output_base_dir)