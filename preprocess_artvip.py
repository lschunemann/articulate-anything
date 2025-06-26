import os
import json
import bpy
import sys
import traceback
from pathlib import Path
import numpy as np
import math

# This needs to run with: blender --background --python preprocess_artvip_usd.py

def find_primary_usd_file(object_dir):
    """Find the primary USD file in an object directory"""
    usd_files = [f for f in os.listdir(object_dir) if f.endswith(('.usd', '.usda'))]
    
    # First try to find model_*.usd file
    for file in usd_files:
        if file.startswith("model_"):
            return os.path.join(object_dir, file)
    
    # If not found, use the first USD file
    if usd_files:
        return os.path.join(object_dir, usd_files[0])
    
    return None

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w

def extract_joints_from_usd(usd_path, debug=False):
    """Extract joint information from USD file using Blender"""
    # Clear existing scene
    bpy.ops.wm.read_homefile(use_empty=True)
    
    # Import the USD file
    try:
        bpy.ops.wm.usd_import(filepath=usd_path)
    except Exception as e:
        if debug:
            print(f"Error importing USD {usd_path}: {e}")
        return {}
    
    joints = {}
    
    # First look for objects with specific naming patterns indicating joints
    joint_candidates = []
    for obj in bpy.data.objects:
        name_lower = obj.name.lower()
        # Look for objects that might be joints based on naming
        if ("joint" in name_lower or 
            "hinge" in name_lower or 
            "revolute" in name_lower or
            "prismatic" in name_lower or
            "slider" in name_lower or
            "pivot" in name_lower or
            "rotation" in name_lower):
            joint_candidates.append(obj)
    
    # If no explicit joint objects found, try to infer from parent-child relationships
    if not joint_candidates:
        for obj in bpy.data.objects:
            if obj.parent and obj.parent.type == 'EMPTY':
                # Parent empty with child object is often used for joints
                joint_candidates.append(obj.parent)
    
    # Now process the joint candidates
    for obj in joint_candidates:
        joint_name = obj.name
        
        # Try to determine joint type based on constraints or naming
        name_lower = obj.name.lower()
        joint_type = "revolute"  # Default assumption
        
        if ("prismatic" in name_lower or 
            "slider" in name_lower or 
            "linear" in name_lower):
            joint_type = "prismatic"
        
        # Get position (origin)
        location = obj.location
        origin = {
            "xyz": f"{location.x} {location.y} {location.z}",
            "orientation": "0 0 0 1"  # Default
        }
        
        # Get rotation and convert to quaternion
        rotation = obj.rotation_euler
        qx, qy, qz, qw = euler_to_quaternion(rotation.x, rotation.y, rotation.z)
        origin["orientation"] = f"{qx} {qy} {qz} {qw}"
        
        # Determine primary axis
        # Default to X axis
        axis = "1 0 0"
        
        # Check if there's a dominant rotation constraint
        for constraint in obj.constraints:
            if constraint.type == 'LIMIT_ROTATION':
                # Try to determine primary rotation axis from constraints
                x_range = abs(constraint.max_x - constraint.min_x)
                y_range = abs(constraint.max_y - constraint.min_y)
                z_range = abs(constraint.max_z - constraint.min_z)
                
                # The axis with the largest allowed range is likely the joint axis
                if x_range >= y_range and x_range >= z_range:
                    axis = "1 0 0"  # X-axis
                elif y_range >= x_range and y_range >= z_range:
                    axis = "0 1 0"  # Y-axis
                else:
                    axis = "0 0 1"  # Z-axis
        
        # Extract limits
        joint_limit = {"lower": -3.14159, "upper": 3.14159}  # Default
        
        for constraint in obj.constraints:
            if constraint.type == 'LIMIT_ROTATION' and joint_type == 'revolute':
                # Find which axis has the largest range (likely the joint axis)
                x_range = abs(constraint.max_x - constraint.min_x)
                y_range = abs(constraint.max_y - constraint.min_y)
                z_range = abs(constraint.max_z - constraint.min_z)
                
                if x_range >= y_range and x_range >= z_range:
                    joint_limit["lower"] = constraint.min_x
                    joint_limit["upper"] = constraint.max_x
                elif y_range >= x_range and y_range >= z_range:
                    joint_limit["lower"] = constraint.min_y
                    joint_limit["upper"] = constraint.max_y
                else:
                    joint_limit["lower"] = constraint.min_z
                    joint_limit["upper"] = constraint.max_z
            
            elif constraint.type == 'LIMIT_LOCATION' and joint_type == 'prismatic':
                # Find which axis has the largest range (likely the joint axis)
                x_range = abs(constraint.max_x - constraint.min_x)
                y_range = abs(constraint.max_y - constraint.min_y)
                z_range = abs(constraint.max_z - constraint.min_z)
                
                if x_range >= y_range and x_range >= z_range:
                    joint_limit["lower"] = constraint.min_x
                    joint_limit["upper"] = constraint.max_x
                elif y_range >= x_range and y_range >= z_range:
                    joint_limit["lower"] = constraint.min_y
                    joint_limit["upper"] = constraint.max_y
                else:
                    joint_limit["lower"] = constraint.min_z
                    joint_limit["upper"] = constraint.max_z
        
        # Store joint data
        joints[joint_name] = {
            "joint_type": joint_type,
            "joint_origin": origin,
            "joint_axis": axis,
            "joint_limit": joint_limit
        }
        
        if debug:
            print(f"Found joint: {joint_name} (type: {joint_type})")
            print(f"  Origin: {origin}")
            print(f"  Axis: {axis}")
            print(f"  Limits: {joint_limit}")
    
    return joints

def extract_vertices_from_usd(usd_path, max_points=10000, debug=False):
    """Extract vertex information from USD file using Blender"""
    # Clear existing scene
    bpy.ops.wm.read_homefile(use_empty=True)
    
    # Import the USD file
    try:
        bpy.ops.wm.usd_import(filepath=usd_path)
    except Exception as e:
        if debug:
            print(f"Error importing USD {usd_path}: {e}")
        return None
    
    all_vertices = []
    
    # Collect vertices from all mesh objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data and len(obj.data.vertices) > 0:
            # Get object's world matrix to transform vertices to world space
            matrix_world = obj.matrix_world
            
            # Extract vertices in world space
            for vertex in obj.data.vertices:
                world_vertex = matrix_world @ vertex.co
                all_vertices.append((world_vertex.x, world_vertex.y, world_vertex.z))
    
    if not all_vertices:
        if debug:
            print(f"No vertices found in {usd_path}")
        return None
    
    if debug:
        print(f"Extracted {len(all_vertices)} vertices from {usd_path}")
    
    # Downsample if needed
    if len(all_vertices) > max_points:
        import random
        all_vertices = random.sample(all_vertices, max_points)
        if debug:
            print(f"Downsampled to {len(all_vertices)} vertices")
    
    return all_vertices

def preprocess_artvip_dataset(debug=False):
    """Process ArtVIP dataset using Blender to extract data from USD files"""
    GT_ROOT = "datasets/ArtVIP/Articulated_objects"
    PROCESSED_DIR = "datasets/ArtVIP/processed"
    
    # Create base processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Track processing stats
    total_objects = 0
    processed_objects = 0
    objects_with_joints = 0
    
    # Find USD files to process
    usd_files = []
    
    # Walk through the dataset structure
    for category_path in Path(GT_ROOT).iterdir():
        if not category_path.is_dir():
            continue
            
        category = category_path.name
        print(f"Finding objects in category: {category}")
        
        for object_dir in category_path.iterdir():
            if not object_dir.is_dir():
                continue
                
            for object_number_dir in object_dir.iterdir():
                if not object_number_dir.is_dir():
                    continue
                
                total_objects += 1
                
                # Find primary USD file
                usd_file = find_primary_usd_file(object_number_dir)
                
                if usd_file:
                    usd_files.append({
                        "path": usd_file,
                        "category": category,
                        "object_name": object_dir.name,
                        "object_number": object_number_dir.name,
                        "output_dir": os.path.join(PROCESSED_DIR, category, object_dir.name, object_number_dir.name)
                    })
    
    print(f"Found {len(usd_files)} USD files to process out of {total_objects} objects")
    
    # Process each USD file
    for i, usd_info in enumerate(usd_files):
        print(f"Processing {i+1}/{len(usd_files)}: {usd_info['path']}")
        
        # Create output directory
        os.makedirs(usd_info["output_dir"], exist_ok=True)
        
        # Create meta.json
        meta = {
            "model_cat": usd_info["category"],
            "model_name": usd_info["object_number"],
            "id": usd_info["object_number"]
        }
        
        with open(os.path.join(usd_info["output_dir"], "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        # Extract joints
        try:
            joints = extract_joints_from_usd(usd_info["path"], debug=debug)
            
            with open(os.path.join(usd_info["output_dir"], "robot_joints.json"), "w") as f:
                json.dump(joints, f, indent=2)
            
            processed_objects += 1
            if joints:
                objects_with_joints += 1
                print(f"  Processed {usd_info['object_number']}: Found {len(joints)} joints")
                
            # Extract vertices and save for later chamfer distance computation
            vertices = extract_vertices_from_usd(usd_info["path"], debug=debug)
            if vertices:
                vertices_file = os.path.join(usd_info["output_dir"], "vertices.json")
                with open(vertices_file, "w") as f:
                    json.dump(vertices, f)
                print(f"  Saved {len(vertices)} vertices")
            
        except Exception as e:
            print(f"  Error processing {usd_info['path']}: {e}")
            traceback.print_exc()
    
    print(f"Preprocessing complete: {processed_objects}/{total_objects} objects processed")
    print(f"Found joints in {objects_with_joints} objects")

# Add command-line arguments
if __name__ == "__main__":
    import argparse
    
    # Need to get arguments manually since Blender has its own argument parsing
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    
    # Create parser
    parser = argparse.ArgumentParser(description="Preprocess ArtVIP dataset using Blender")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    # Run preprocessing
    preprocess_artvip_dataset(debug=args.debug)
    