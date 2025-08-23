import os
import json
import numpy as np
import math
import traceback
import glob
from collections import defaultdict
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import trimesh

# Import useful functions from original metric.py
from articulate_anything.utils.metric import (
    compute_joint_diff_score_adaptive,
    compute_joint_diff_with_continuous_support,
    analyze_joint_pred_by_type,
    create_specific_failure_result,
    extract_mesh_vertices_from_urdf,
    compute_chamfer_distance_with_alignment,
    extract_part_meshes_from_urdf
)

# Constants
PROCESSED_DIR = "datasets/ArtVIP/processed"
GEN_RESULT_DIR = "results/generated"
BASELINE_RESULT_DIR = "results/image"

def extract_vertices_from_glb(glb_path, debug=False):
    """Extract vertices from a GLB file using trimesh."""
    if not trimesh:
        if debug: print("trimesh is not installed, cannot load GLB.")
        return None
    if not os.path.exists(glb_path):
        if debug: print(f"GLB file not found: {glb_path}")
        return None
    try:
        # Combine all meshes in the scene into one
        scene = trimesh.load(glb_path, force='scene')
        mesh = trimesh.util.concatenate(scene.dump())
        return np.array(mesh.vertices)
    except Exception as e:
        if debug: print(f"Error loading GLB {glb_path}: {e}")
        return None

def load_parts_from_obj_dir(obj_dir, debug=False):
    """Load all .obj files from a directory as separate parts."""
    if not trimesh:
        if debug: print("trimesh is not installed, cannot load OBJ parts.")
        return {}
    if not os.path.isdir(obj_dir):
        if debug: print(f"OBJ directory not found: {obj_dir}")
        return {}
    
    parts = {}
    obj_files = [f for f in os.listdir(obj_dir) if f.endswith('.obj')]
    
    for filename in obj_files:
        filepath = os.path.join(obj_dir, filename)
        try:
            mesh = trimesh.load(filepath, force='mesh')
            if not mesh.is_empty:
                parts[filename] = np.array(mesh.vertices)
        except Exception as e:
            if debug: print(f"Error loading OBJ {filepath}: {e}")
            continue
            
    if debug:
        print(f"Loaded {len(parts)} parts from {obj_dir}")
    return parts

def calculate_bounding_sphere_scale(source_pts, target_pts):
    """Calculates a uniform scale factor to match the bounding sphere of source_pts to target_pts."""
    if source_pts is None or target_pts is None or len(source_pts) == 0 or len(target_pts) == 0:
        return 1.0
    
    source_center = np.mean(source_pts, axis=0)
    source_radius = np.max(np.linalg.norm(source_pts - source_center, axis=1))
    
    target_center = np.mean(target_pts, axis=0)
    target_radius = np.max(np.linalg.norm(target_pts - target_center, axis=1))
    
    if source_radius < 1e-6:
        return 1.0
        
    return target_radius / source_radius

def load_robot_joints(joints_path):
    """Load joint information from robot_joints.json"""
    if not os.path.exists(joints_path):
        return {}
        
    try:
        with open(joints_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {joints_path}: {e}")
        return {}

def load_gt_part_meshes(gt_path, use_surface_samples=False, debug=False):
    """Load ground truth part meshes from the processed directory."""
    if use_surface_samples:
        manifest_path = os.path.join(gt_path, "part_meshes_surface.json")
        parts_dir = os.path.join(gt_path, "parts_surface")
        if not os.path.exists(manifest_path):
            if debug:
                print(f"Warning: Surface-sampled part manifest not found at {manifest_path}. Falling back to full mesh.")
            manifest_path = os.path.join(gt_path, "part_meshes.json")
            parts_dir = os.path.join(gt_path, "parts")
    else:
        manifest_path = os.path.join(gt_path, "part_meshes.json")
        parts_dir = os.path.join(gt_path, "parts")
    
    if not os.path.exists(manifest_path) or not os.path.exists(parts_dir):
        if debug:
            print(f"GT part mesh data not found in {gt_path}")
        return {}
        
    try:
        with open(manifest_path, "r") as f:
            part_manifest = json.load(f)
    except json.JSONDecodeError:
        if debug:
            print(f"Could not decode part manifest: {manifest_path}")
        return {}
        
    gt_parts = {}
    for part_name, part_filename in part_manifest.items():
        part_filepath = os.path.join(parts_dir, part_filename)
        if os.path.exists(part_filepath):
            try:
                with open(part_filepath, "r") as f_part:
                    vertices = np.array(json.load(f_part))
                    gt_parts[part_name] = vertices
            except (json.JSONDecodeError, ValueError) as e:
                if debug:
                    print(f"Could not load part mesh {part_filepath}: {e}")
        elif debug:
            print(f"Part mesh file not found: {part_filepath}")
            
    if debug:
        print(f"Loaded {len(gt_parts)} GT parts for {os.path.basename(gt_path)}")
        
    return gt_parts

def load_vertices(vertices_path):
    """Load vertices from preprocessed vertices.json"""
    if not os.path.exists(vertices_path):
        return None
        
    try:
        with open(vertices_path, "r") as f:
            vertices = json.load(f)
            return np.array(vertices)
    except Exception as e:
        print(f"Error loading vertices from {vertices_path}: {e}")
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

def extract_joints_from_urdf(urdf_path, debug=False):
    """Extract joint information from URDF file"""
    if not os.path.exists(urdf_path):
        return {}
    
    joints = {}
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        for joint_elem in root.findall(".//joint"):
            joint_name = joint_elem.get("name")
            joint_type = joint_elem.get("type")
            
            if not joint_name or not joint_type:
                continue
            
            # Extract origin
            origin = {"xyz": "0 0 0", "orientation": "0 0 0 1"}
            origin_elem = joint_elem.find("origin")
            if origin_elem is not None:
                if "xyz" in origin_elem.attrib:
                    origin["xyz"] = origin_elem.get("xyz")
                if "rpy" in origin_elem.attrib:
                    # Convert RPY to quaternion to match GT format
                    rpy_str = origin_elem.get("rpy")
                    try:
                        roll, pitch, yaw = map(float, rpy_str.split())
                        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
                        origin["orientation"] = f"{qx} {qy} {qz} {qw}"
                    except ValueError:
                        if debug:
                            print(f"Could not parse RPY: {rpy_str}")
                        # Fallback to default if parsing fails
                        origin["orientation"] = "0 0 0 1"
            
            # Extract axis
            axis = "1 0 0"  # Default
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None and "xyz" in axis_elem.attrib:
                axis = axis_elem.get("xyz")

            # Validate axis to prevent division by zero errors later
            try:
                axis_vec = np.array([float(v) for v in axis.split()])
                if len(axis_vec) != 3 or np.linalg.norm(axis_vec) < 1e-6:
                    if debug:
                        print(f"Warning: Invalid or zero-norm axis '{axis}' for joint '{joint_name}' in {urdf_path}. Defaulting to '1 0 0'.")
                    axis = "1 0 0"
            except (ValueError, IndexError):
                if debug:
                    print(f"Warning: Could not parse axis '{axis}' for joint '{joint_name}' in {urdf_path}. Defaulting to '1 0 0'.")
                axis = "1 0 0"
            
            # Extract limits
            joint_limit = {"lower": -3.14, "upper": 3.14}  # Default
            limit_elem = joint_elem.find("limit")
            if limit_elem is not None:
                if "lower" in limit_elem.attrib:
                    joint_limit["lower"] = float(limit_elem.get("lower"))
                if "upper" in limit_elem.attrib:
                    joint_limit["upper"] = float(limit_elem.get("upper"))
            
            # Build joint data
            joints[joint_name] = {
                "joint_type": joint_type,
                "joint_origin": origin,
                "joint_axis": axis,
                "joint_limit": joint_limit
            }
            
        if debug and joints:
            print(f"Extracted {len(joints)} joints from {urdf_path}")
            
    except Exception as e:
        if debug:
            print(f"Error parsing URDF {urdf_path}: {e}")
            traceback.print_exc()
    
    return joints

def find_best_prediction(prediction_dir, debug=False):
    """Find the best prediction in the iter/seed directory structure"""
    if not os.path.exists(prediction_dir):
        return None, None, None
    
    joint_actor_dir = os.path.join(prediction_dir, "joint_actor")
    if not os.path.exists(joint_actor_dir):
        if debug:
            print(f"No joint_actor directory in {prediction_dir}")
        return None, None, None
    
    # Try to find the latest iteration and seed with a valid mobility.urdf
    iter_dirs = sorted(
        [d for d in os.listdir(joint_actor_dir) if d.startswith("iter_")],
        key=lambda x: int(x.split("_")[1]), 
        reverse=True
    )
    
    if not iter_dirs:
        if debug:
            print(f"No iter directories in {joint_actor_dir}")
        return None, None, None
    
    for iter_dir in iter_dirs:
        iter_path = os.path.join(joint_actor_dir, iter_dir)
        if not os.path.isdir(iter_path):
            continue
            
        iter_num = int(iter_dir.split("_")[1])
        
        seed_dirs = sorted(
            [d for d in os.listdir(iter_path) if d.startswith("seed_")],
            key=lambda x: int(x.split("_")[1])
        )
        
        if not seed_dirs and debug:
            print(f"No seed directories in {iter_path}")
        
        for seed_dir in seed_dirs:
            seed_path = os.path.join(iter_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue
                
            seed_num = int(seed_dir.split("_")[1])
            
            urdf_path = os.path.join(seed_path, "mobility.urdf")
            if os.path.exists(urdf_path):
                if debug:
                    print(f"Found prediction at iter {iter_num}, seed {seed_num}")
                return urdf_path, iter_num, seed_num
    
    if debug:
        print(f"No valid mobility.urdf found in {prediction_dir}")
    return None, None, None

def get_available_objects(debug=False):
    """Get list of available objects with both ground truth and predictions"""
    available_objects = []
    
    # Check if processed directory exists
    if not os.path.exists(PROCESSED_DIR):
        print(f"Processed directory {PROCESSED_DIR} not found. Run preprocessing first.")
        return []
    
    # Walk through processed directory
    for category_path in Path(PROCESSED_DIR).iterdir():
        if not category_path.is_dir():
            continue
            
        category = category_path.name
        
        for object_dir in category_path.iterdir():
            if not object_dir.is_dir():
                continue
                
            for object_number_dir in object_dir.iterdir():
                if not object_number_dir.is_dir():
                    continue
                
                object_number = object_number_dir.name
                processed_dir = object_number_dir
                
                # Check if robot_joints.json exists
                joints_file = os.path.join(processed_dir, "robot_joints.json")
                if not os.path.exists(joints_file):
                    continue
                
                # Find corresponding predictions
                gen_pred_dir = os.path.join(GEN_RESULT_DIR, f"artvip_{object_number}_multi-view")
                baseline_pred_dir = os.path.join(BASELINE_RESULT_DIR, f"artvip_{object_number}_multi-view")
                
                # Check if at least one prediction exists
                if os.path.exists(gen_pred_dir) or os.path.exists(baseline_pred_dir):
                    available_objects.append({
                        "category": category, 
                        "object_name": object_dir.name,
                        "object_number": object_number, 
                        "gt_path": str(processed_dir), 
                        "gen_pred_dir": gen_pred_dir if os.path.exists(gen_pred_dir) else None,
                        "baseline_pred_dir": baseline_pred_dir if os.path.exists(baseline_pred_dir) else None
                    })
    
    if debug:
        print(f"Found {len(available_objects)} objects with ground truth and at least one prediction")
    
    return available_objects

def evaluate_part_segmentation(gt_parts, pred_parts, debug=False, scale_pred=1.0):
    """
    Evaluate part segmentation and reconstruction using Chamfer distance and Hungarian matching.
    
    Returns:
        dict: A dictionary with evaluation metrics.
    """
    if not gt_parts or not pred_parts:
        return {
            "part_count_gt": len(gt_parts) if gt_parts else 0,
            "part_count_pred": len(pred_parts) if pred_parts else 0,
            "part_count_error": abs(len(gt_parts if gt_parts else []) - len(pred_parts if pred_parts else [])),
            "part_count_correct": 1 if len(gt_parts if gt_parts else []) == len(pred_parts if pred_parts else []) else 0,
            "mean_chamfer_distance": float('inf'),
            "matched_parts": 0,
            "unmatched_gt": len(gt_parts) if gt_parts else 0,
            "unmatched_pred": len(pred_parts) if pred_parts else 0
        }

    if scale_pred != 1.0:
        if debug:
            print(f"    Scaling predicted parts by factor {scale_pred:.4f}")
        scaled_pred_parts = {}
        for name, vertices in pred_parts.items():
            scaled_pred_parts[name] = vertices * scale_pred
        pred_parts = scaled_pred_parts

    gt_part_names = list(gt_parts.keys())
    pred_part_names = list(pred_parts.keys())
    
    cost_matrix = np.full((len(gt_part_names), len(pred_part_names)), float('inf'))
    
    if debug:
        print(f"Computing part cost matrix: {len(gt_part_names)} GT parts x {len(pred_part_names)} pred parts")

    for i, gt_name in enumerate(gt_part_names):
        for j, pred_name in enumerate(pred_part_names):
            gt_vertices = gt_parts[gt_name]
            pred_vertices = pred_parts[pred_name]
            
            if gt_vertices is None or pred_vertices is None or len(gt_vertices) < 1 or len(pred_vertices) < 1:
                continue
            
            dist, _ = compute_chamfer_distance_with_alignment(pred_vertices, gt_vertices, alignment_method="z_axis_90")
            cost_matrix[i, j] = dist

    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    matched_distances = cost_matrix[gt_indices, pred_indices]
    
    valid_matches_mask = np.isfinite(matched_distances)
    matched_distances = matched_distances[valid_matches_mask]
    
    mean_cd = np.mean(matched_distances) if len(matched_distances) > 0 else float('inf')
    
    num_matched = len(matched_distances)
    num_unmatched_gt = len(gt_parts) - num_matched
    num_unmatched_pred = len(pred_parts) - num_matched
    
    if debug:
        print(f"  Matched {num_matched} parts with mean CD: {mean_cd:.4f}")
        print(f"  Unmatched GT: {num_unmatched_gt}, Unmatched Pred: {num_unmatched_pred}")

    return {
        "part_count_gt": len(gt_parts),
        "part_count_pred": len(pred_parts),
        "part_count_error": abs(len(gt_parts) - len(pred_parts)),
        "part_count_correct": 1 if len(gt_parts) == len(pred_parts) else 0,
        "mean_chamfer_distance": float(mean_cd),
        "matched_parts": num_matched,
        "unmatched_gt": num_unmatched_gt,
        "unmatched_pred": num_unmatched_pred
    }

def evaluate_joint_prediction(gt_joints, pred_joints, category, method_name="baseline", debug=False):
    """Evaluate joint prediction accuracy using adaptive scoring."""
    if not gt_joints or not pred_joints:
        if debug:
            print("Empty gt_joints or pred_joints, skipping evaluation")
        return {}
    
    # Compute similarity matrix
    gt_joint_names = list(gt_joints.keys())
    pred_joint_names = list(pred_joints.keys())
    
    if debug:
        print(f"GT joints: {gt_joint_names}")
        print(f"Pred joints: {pred_joint_names}")
    
    # Initialize similarity matrix
    similarity_matrix = np.full((len(gt_joint_names), len(pred_joint_names)), 1000.0)
    
    for i, gt_joint_name in enumerate(gt_joint_names):
        gt_joint = gt_joints[gt_joint_name]
        
        # Extract GT origin position
        try:
            gt_origin = gt_joint.get("joint_origin", {}).get("xyz", "0 0 0")
            gt_pos = np.array([float(x) for x in str(gt_origin).split()])
        except Exception as e:
            if debug:
                print(f"Error parsing GT joint origin for {gt_joint_name}: {e}")
            continue
        
        for j, pred_joint_name in enumerate(pred_joint_names):
            pred_joint = pred_joints[pred_joint_name]
            
            # Extract predicted origin position
            try:
                pred_origin = pred_joint.get("joint_origin", {}).get("xyz", "0 0 0")
                pred_pos = np.array([float(x) for x in str(pred_origin).split()])
            except Exception as e:
                if debug:
                    print(f"Error parsing pred joint origin for {pred_joint_name}: {e}")
                continue
            
            # Compute distance between joint origins
            position_distance = np.linalg.norm(gt_pos - pred_pos)
            similarity_matrix[i, j] = position_distance
    
    # Apply Hungarian algorithm
    gt_indices, pred_indices = linear_sum_assignment(similarity_matrix)
    
    if debug:
        print("Hungarian matching results:")
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            match_cost = similarity_matrix[gt_idx, pred_idx]
            print(f"  {gt_joint_names[gt_idx]} -> {pred_joint_names[pred_idx]} (cost: {match_cost:.4f})")
    
    # Evaluate each matched pair
    joint_results = {}
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        gt_joint_name = gt_joint_names[gt_idx]
        pred_joint_name = pred_joint_names[pred_idx]
        matching_cost = similarity_matrix[gt_idx, pred_idx]
        
        # Skip matches with very high cost
        if matching_cost > 100:
            if debug:
                print(f"Skipping high-cost match: {gt_joint_name} -> {pred_joint_name} (cost: {matching_cost:.4f})")
            continue
        
        # Compute joint difference with continuous support
        try:
            joint_diff = compute_joint_diff_with_continuous_support(
                gt_joints, pred_joints, gt_joint_name, pred_joint_name, debug
            )
            
            if not joint_diff:
                if debug:
                    print(f"Empty joint diff for {gt_joint_name} -> {pred_joint_name}")
                continue
            
            # Compute score using adaptive thresholds
            # "sem" for generated method (more lenient), "original" for baseline (stricter)
            naming_convention = "sem" if method_name == "generated" else "original"
            score, failure_reason = compute_joint_diff_score_adaptive(
                joint_diff, naming_convention=naming_convention
            )
            
            # Store result
            joint_results[(gt_joint_name, pred_joint_name)] = {
                "joint_diff": joint_diff,
                "failure_reason": failure_reason,
                "score": score,
                "matched_joint_name": pred_joint_name,
                "matching_cost": matching_cost,
                "hungarian_matched": True,
                "obj_type": category,
                "joint_type": joint_diff.get("original_joint_type", "unknown")
            }
            
            if debug:
                print(f"Evaluated {gt_joint_name} -> {pred_joint_name} with '{naming_convention}' thresholds: score={score}, failure={failure_reason}")
        
        except Exception as e:
            if debug:
                print(f"Error evaluating joint match {gt_joint_name} -> {pred_joint_name}: {e}")
                traceback.print_exc()
    
    # Handle unmatched GT joints
    for gt_idx in range(len(gt_joint_names)):
        if gt_idx not in gt_indices:
            gt_joint_name = gt_joint_names[gt_idx]
            gt_joint = gt_joints[gt_joint_name]
            
            if debug:
                print(f"Unmatched GT joint: {gt_joint_name}")
            
            # Create a failure result for unmatched joint
            result = create_specific_failure_result("joint_missing")
            
            # Add category and joint type info
            result["obj_type"] = category
            result["joint_type"] = gt_joint.get("joint_type", "unknown")
            
            joint_results[(gt_joint_name, "unmatched")] = result
    
    return joint_results

def evaluate_chamfer_distance(gt_path, pred_path, use_surface_samples=False, debug=False, scale_pred=1.0):
    """Evaluate chamfer distance between ground truth and prediction"""
    if use_surface_samples:
        vertices_path = os.path.join(gt_path, "vertices_surface.json")
    else:
        vertices_path = os.path.join(gt_path, "vertices.json")
    
    if not os.path.exists(vertices_path) or not os.path.exists(pred_path):
        if debug:
            print(f"Missing path: GT vertices={vertices_path}, Pred={pred_path}")
        return float('inf')
    
    # Load GT vertices
    gt_vertices = load_vertices(vertices_path)
    
    # Extract vertices from URDF
    pred_vertices = extract_mesh_vertices_from_urdf(pred_path)
    
    if gt_vertices is None or pred_vertices is None or len(gt_vertices) == 0 or len(pred_vertices) == 0:
        if debug:
            print("Could not load vertices for Chamfer distance.")
        return float('inf')
    
    gt_vertices = np.array(gt_vertices)
    pred_vertices = np.array(pred_vertices)

    if scale_pred != 1.0:
        if debug:
            print(f"    Scaling predicted mesh by factor {scale_pred:.4f}")
        pred_vertices *= scale_pred
    
    try:
        # The alignment function is assumed to perform ICP before Chamfer calculation.
        dist, _ = compute_chamfer_distance_with_alignment(pred_vertices, gt_vertices, alignment_method="z_axis_90")
        return dist
    except Exception as e:
        if debug:
            print(f"Error computing chamfer distance: {e}")
            traceback.print_exc()
        return float('inf')

def evaluate_artvip_dataset(debug=False, use_surface_samples=False):
    """Main evaluation function for ArtVIP dataset"""
    # Check if processed directory exists
    if not os.path.exists(PROCESSED_DIR):
        print(f"Processed directory {PROCESSED_DIR} not found. Please run preprocessing first with:")
        print("blender --background --python preprocess_artvip_usd.py")
        return

    # Get available objects
    available_objects = get_available_objects(debug=debug)
    if not available_objects:
        print("No objects found with both ground truth and predictions. Exiting.")
        return
    
    print(f"Found {len(available_objects)} objects with ground truth and predictions")
    
    # Results storage
    gen_joint_results = {}
    baseline_joint_results = {}
    gen_chamfer_results = {}
    baseline_chamfer_results = {}
    gen_part_seg_results = {}
    baseline_part_seg_results = {}

    try:
        # Evaluate each object
        for i, obj in enumerate(available_objects):
            category = obj["category"]
            object_name = obj["object_name"]
            object_number = obj["object_number"]
            gt_path = obj["gt_path"]
            gen_pred_dir = obj["gen_pred_dir"]
            baseline_pred_dir = obj["baseline_pred_dir"]
            
            print(f"[{i+1}/{len(available_objects)}] Evaluating {category}/{object_number}")
            
            # Load ground truth joints
            gt_joints_path = os.path.join(gt_path, "robot_joints.json")
            gt_joints = load_robot_joints(gt_joints_path)
            
            # Load ground truth part meshes
            gt_parts = load_gt_part_meshes(gt_path, use_surface_samples=use_surface_samples, debug=debug)

            if not gt_joints and not gt_parts:
                print(f"  No ground truth joints or parts found in {gt_path}")
                continue
            
            # Evaluate generated method
            if gen_pred_dir:
                gen_urdf_path, gen_iter, gen_seed = find_best_prediction(gen_pred_dir, debug=debug)
                if gen_urdf_path:
                    print(f"  Evaluating generated method (iter {gen_iter}, seed {gen_seed})")
                    
                    # --- Scaling Logic for Generated Method ---
                    scaling_factor = 1.0
                    gt_full_vertices_path = os.path.join(gt_path, "vertices_surface.json" if use_surface_samples else "vertices.json")
                    gt_full_vertices = load_vertices(gt_full_vertices_path)
                    pred_full_vertices = extract_mesh_vertices_from_urdf(gen_urdf_path)
                    
                    if gt_full_vertices is not None and pred_full_vertices is not None:
                        scaling_factor = calculate_bounding_sphere_scale(pred_full_vertices, gt_full_vertices)
                        if debug:
                            print(f"    Applying uniform scale factor: {scaling_factor:.4f}")
                    # --- End Scaling Logic ---

                    # Evaluate joints
                    gen_pred_joints = extract_joints_from_urdf(gen_urdf_path, debug=debug)
                    if gen_pred_joints:
                        joint_eval = evaluate_joint_prediction(gt_joints, gen_pred_joints, category, "generated", debug)
                        for key, result in joint_eval.items():
                            full_key = (object_number, key[0], key[1])
                            gen_joint_results[full_key] = result
                    else:
                        print(f"  No joints found in generated URDF")
                    
                    # Evaluate chamfer distance
                    chamfer_dist = evaluate_chamfer_distance(
                        gt_path, gen_urdf_path, use_surface_samples=use_surface_samples, debug=debug, scale_pred=scaling_factor
                    )
                    
                    if chamfer_dist != float('inf'):
                        gen_chamfer_results[object_number] = (chamfer_dist, category, object_name)
                        print(f"    Generated chamfer distance (scaled): {chamfer_dist:.4f}")
                    else:
                        print(f"    Failed to compute chamfer distance for generated method")
                    
                    # Evaluate part segmentation
                    if gt_parts:
                        print(f"  Evaluating generated part segmentation...")
                        gen_pred_parts = extract_part_meshes_from_urdf(gen_urdf_path, debug=debug)
                        part_seg_eval = evaluate_part_segmentation(gt_parts, gen_pred_parts, debug=debug, scale_pred=scaling_factor)
                        gen_part_seg_results[object_number] = part_seg_eval
                        print(f"      Part count correct: {'Yes' if part_seg_eval['part_count_correct'] else 'No'}. "
                              f"Mean CD (scaled): {part_seg_eval['mean_chamfer_distance']:.4f}")
                else:
                    # --- Fallback Logic for Generated Method ---
                    print(f"  No URDF found for generated method. Attempting fallback from segmentation results.")
                    
                    fallback_view_dir = os.path.join("datasets/output_views", object_number)
                    fallback_glb_path = None
                    if os.path.isdir(fallback_view_dir):
                        glb_files = [f for f in os.listdir(fallback_view_dir) if f.endswith('.glb')]
                        if glb_files:
                            fallback_glb_path = os.path.join(fallback_view_dir, glb_files[0])

                    fallback_parts_dir = os.path.join("datasets/segmentation_masks", object_number, "output")

                    has_fallback_full_mesh = fallback_glb_path is not None
                    has_fallback_parts = os.path.isdir(fallback_parts_dir)

                    if not has_fallback_full_mesh and not has_fallback_parts:
                        print("    No fallback data found. Skipping generated method for this object.")
                    else:
                        # --- Scaling Logic for Fallback ---
                        scaling_factor = 1.0
                        gt_full_vertices_path = os.path.join(gt_path, "vertices_surface.json" if use_surface_samples else "vertices.json")
                        gt_full_vertices = load_vertices(gt_full_vertices_path)
                        
                        pred_full_vertices = None
                        if has_fallback_full_mesh:
                            pred_full_vertices = extract_vertices_from_glb(fallback_glb_path, debug=debug)

                        if gt_full_vertices is not None and pred_full_vertices is not None:
                            scaling_factor = calculate_bounding_sphere_scale(pred_full_vertices, gt_full_vertices)
                            if debug:
                                print(f"    Applying uniform scale factor from GLB: {scaling_factor:.4f}")
                        # --- End Scaling Logic ---

                        # Evaluate Chamfer on full mesh if available
                        if has_fallback_full_mesh and pred_full_vertices is not None and gt_full_vertices is not None:
                            scaled_pred_vertices = pred_full_vertices * scaling_factor
                            dist, _ = compute_chamfer_distance_with_alignment(scaled_pred_vertices, gt_full_vertices)
                            if dist != float('inf'):
                                gen_chamfer_results[object_number] = (dist, category, object_name)
                                print(f"    Generated chamfer distance (fallback GLB, scaled): {dist:.4f}")
                            else:
                                print(f"    Failed to compute chamfer distance for fallback GLB")
                        
                        # Evaluate part segmentation if available
                        if has_fallback_parts and gt_parts:
                            print(f"    Evaluating generated part segmentation (fallback OBJs)...")
                            fallback_pred_parts = load_parts_from_obj_dir(fallback_parts_dir, debug=debug)
                            if fallback_pred_parts:
                                part_seg_eval = evaluate_part_segmentation(gt_parts, fallback_pred_parts, debug=debug, scale_pred=scaling_factor)
                                gen_part_seg_results[object_number] = part_seg_eval
                                print(f"      Part count correct: {'Yes' if part_seg_eval['part_count_correct'] else 'No'}. "
                                      f"Mean CD (fallback, scaled): {part_seg_eval['mean_chamfer_distance']:.4f}")
                            else:
                                print("      Could not load any parts from fallback directory.")

                        print("    Skipping joint prediction (no URDF).")

            # Evaluate baseline method
            if baseline_pred_dir:
                baseline_urdf_path, baseline_iter, baseline_seed = find_best_prediction(baseline_pred_dir, debug=debug)
                
                if baseline_urdf_path:
                    print(f"  Evaluating baseline method (iter {baseline_iter}, seed {baseline_seed})")
                    
                    # Evaluate joints
                    baseline_pred_joints = extract_joints_from_urdf(baseline_urdf_path, debug=debug)
                    if baseline_pred_joints:
                        obj_joint_results = evaluate_joint_prediction(
                            gt_joints, baseline_pred_joints, category, method_name="baseline", debug=debug
                        )
                        
                        for key, result in obj_joint_results.items():
                            # Create a unique identifier for this result
                            full_key = (object_number, key[0], key[1])
                            baseline_joint_results[full_key] = result
                    else:
                        print(f"  No joints found in baseline URDF")
                    
                    # Evaluate chamfer distance
                    chamfer_dist = evaluate_chamfer_distance(
                        gt_path, baseline_urdf_path, use_surface_samples=use_surface_samples, debug=debug
                    )
                    
                    if chamfer_dist != float('inf'):
                        baseline_chamfer_results[object_number] = (chamfer_dist, category, object_name)
                        print(f"  Baseline chamfer distance: {chamfer_dist:.4f}")
                    else:
                        print(f"  Failed to compute chamfer distance for baseline method")
                    
                    # Evaluate part segmentation
                    if gt_parts:
                        print(f"  Evaluating baseline part segmentation...")
                        baseline_pred_parts = extract_part_meshes_from_urdf(baseline_urdf_path, debug=debug)
                        part_seg_eval = evaluate_part_segmentation(gt_parts, baseline_pred_parts, debug=debug)
                        baseline_part_seg_results[object_number] = part_seg_eval
                        print(f"      Part count correct: {'Yes' if part_seg_eval['part_count_correct'] else 'No'}. "
                              f"Mean CD: {part_seg_eval['mean_chamfer_distance']:.4f}")
                else:
                    print(f"  No valid URDF found for baseline method")
    finally:
        print("\nEvaluation finished or interrupted. Saving results and generating reports...")
        # Save results to files
        save_results(
            gen_joint_results, baseline_joint_results,
            gen_chamfer_results, baseline_chamfer_results,
            gen_part_seg_results, baseline_part_seg_results
        )
        
        # Analyze and print results
        print("\n=== Generated Method Results ===")
        analyze_results(gen_joint_results, gen_chamfer_results, gen_part_seg_results, available_objects, method_name="Generated")
        
        print("\n=== Baseline Method Results ===")
        analyze_results(baseline_joint_results, baseline_chamfer_results, baseline_part_seg_results, available_objects, method_name="Baseline")
        
        # Generate comparison
        compare_methods(
            gen_joint_results, gen_chamfer_results, gen_part_seg_results,
            baseline_joint_results, baseline_chamfer_results, baseline_part_seg_results,
            available_objects
        )

def save_results(gen_joint, baseline_joint, gen_chamfer, baseline_chamfer, gen_part_seg, baseline_part_seg):
    """Save all result dictionaries to JSON files."""
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "gen_joint_results.json"), "w") as f:
        json.dump({str(k): v for k, v in gen_joint.items()}, f, indent=2)
        
    with open(os.path.join(output_dir, "baseline_joint_results.json"), "w") as f:
        json.dump({str(k): v for k, v in baseline_joint.items()}, f, indent=2)
        
    with open(os.path.join(output_dir, "gen_chamfer_results.json"), "w") as f:
        json.dump({k: (float(v[0]), v[1], v[2]) for k, v in gen_chamfer.items()}, f, indent=2)
        
    with open(os.path.join(output_dir, "baseline_chamfer_results.json"), "w") as f:
        json.dump({k: (float(v[0]), v[1], v[2]) for k, v in baseline_chamfer.items()}, f, indent=2)
    
    with open(os.path.join(output_dir, "gen_part_seg_results.json"), "w") as f:
        json.dump(gen_part_seg, f, indent=2)
        
    with open(os.path.join(output_dir, "baseline_part_seg_results.json"), "w") as f:
        json.dump(baseline_part_seg, f, indent=2)

    print(f"Results saved to {output_dir}")

def compare_methods(gen_joint_results, gen_chamfer_results, gen_part_seg_results,
                    baseline_joint_results, baseline_chamfer_results, baseline_part_seg_results,
                    available_objects):
    """Compare generated and baseline methods"""
    # Check if we have results to compare
    if not gen_joint_results and not baseline_joint_results:
        print("No joint results to compare")
        return
    
    # Get common objects for joint evaluation
    gen_objects = set([key[0] for key in gen_joint_results.keys()])
    baseline_objects = set([key[0] for key in baseline_joint_results.keys()])
    common_objects = gen_objects & baseline_objects
    
    print(f"\n=== Comparison on {len(common_objects)} Common Objects ===")
    
    if common_objects:
        # Joint prediction success rates
        gen_joint_stats = analyze_joint_pred_by_type({k: v for k, v in gen_joint_results.items() if k[0] in common_objects})
        baseline_joint_stats = analyze_joint_pred_by_type({k: v for k, v in baseline_joint_results.items() if k[0] in common_objects})
        
        print("\nJoint Prediction Success Rates:")
        all_obj_types = sorted(set(gen_joint_stats.keys()) | set(baseline_joint_stats.keys()))

        for obj_type in all_obj_types:
            # Skip non-dict summary keys
            if not isinstance(gen_joint_stats.get(obj_type), dict) and not isinstance(baseline_joint_stats.get(obj_type), dict):
                continue
            print(f"\n{obj_type}:")
            
            # Get all joint types for this object type
            joint_types = set()
            if isinstance(gen_joint_stats.get(obj_type), dict):
                joint_types.update(gen_joint_stats[obj_type].keys())
            if isinstance(baseline_joint_stats.get(obj_type), dict):
                joint_types.update(baseline_joint_stats[obj_type].keys())
            
            for joint_type in sorted(joint_types):
                gen_stats = gen_joint_stats.get(obj_type, {}).get(joint_type, {})
                baseline_stats = baseline_joint_stats.get(obj_type, {}).get(joint_type, {})

                # Check if the stats are dictionaries before calling .get()
                if not isinstance(gen_stats, dict) or not isinstance(baseline_stats, dict):
                    continue

                gen_rate = gen_stats.get('success_rate', 0)
                baseline_rate = baseline_stats.get('success_rate', 0)
                
                gen_count = gen_stats.get('count', 0)
                baseline_count = baseline_stats.get('count', 0)
                
                print(f"  {joint_type}: Generated: {gen_rate:.2f} ({gen_count}), Baseline: {baseline_rate:.2f} ({baseline_count})")
                
                if gen_rate > 0 and baseline_rate > 0 and baseline_rate != 0:
                    improvement = (gen_rate - baseline_rate) / baseline_rate
                    print(f"    Improvement: {gen_rate - baseline_rate:.2f} ({improvement:.2%})")
    
    # Chamfer distance comparison
    common_chamfer = set(gen_chamfer_results.keys()) & set(baseline_chamfer_results.keys())
    
    if common_chamfer:
        print("\nChamfer Distance Comparison:")
        
        # Group by category
        by_category = defaultdict(list)
        for obj_id in common_chamfer:
            gen_dist, category, _ = gen_chamfer_results[obj_id]
            baseline_dist, _, _ = baseline_chamfer_results[obj_id]
            by_category[category].append((gen_dist, baseline_dist))
        
        # Print comparison
        for category, distances in sorted(by_category.items()):
            gen_dists = [d[0] for d in distances]
            baseline_dists = [d[1] for d in distances]
            
            print(f"{category} ({len(distances)} objects):")
            print(f"  Generated: {np.mean(gen_dists):.4f}")
            print(f"  Baseline: {np.mean(baseline_dists):.4f}")
            
            if np.mean(baseline_dists) > 0:
                improvement = (np.mean(baseline_dists) - np.mean(gen_dists))/np.mean(baseline_dists)
                print(f"  Improvement: {np.mean(baseline_dists) - np.mean(gen_dists):.4f} ({improvement:.2%})")
        
        # Overall
        all_gen = [gen_chamfer_results[obj_id][0] for obj_id in common_chamfer]
        all_baseline = [baseline_chamfer_results[obj_id][0] for obj_id in common_chamfer]
        
        print(f"\nOverall ({len(common_chamfer)} objects):")
        print(f"  Generated: {np.mean(all_gen):.4f}")
        print(f"  Baseline: {np.mean(all_baseline):.4f}")
        
        if np.mean(all_baseline) > 0:
            improvement = (np.mean(all_baseline) - np.mean(all_gen))/np.mean(all_baseline)
            print(f"  Improvement: {np.mean(all_baseline) - np.mean(all_gen):.4f} ({improvement:.2%})")

    # Part segmentation comparison
    if gen_part_seg_results or baseline_part_seg_results:
        print("\nPart Segmentation Comparison:")
        gen_part_count_correct = np.mean([res['part_count_correct'] for res in gen_part_seg_results.values()]) if gen_part_seg_results else 0
        base_part_count_correct = np.mean([res['part_count_correct'] for res in baseline_part_seg_results.values()]) if baseline_part_seg_results else 0
        
        gen_valid_cds = [res['mean_chamfer_distance'] for res in gen_part_seg_results.values() if np.isfinite(res['mean_chamfer_distance'])]
        base_valid_cds = [res['mean_chamfer_distance'] for res in baseline_part_seg_results.values() if np.isfinite(res['mean_chamfer_distance'])]
        gen_mean_cd = np.mean(gen_valid_cds) if gen_valid_cds else float('inf')
        base_mean_cd = np.mean(base_valid_cds) if base_valid_cds else float('inf')

        print(f"  Part Count Correct Rate: Generated: {gen_part_count_correct:.2%}, Baseline: {base_part_count_correct:.2%}")
        print(f"  Mean Part Chamfer Distance: Generated: {gen_mean_cd:.4f} (scaled), Baseline: {base_mean_cd:.4f}")

    # Generate reports
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get stats for plots
    gen_joint_stats = analyze_joint_pred_by_type(gen_joint_results) if gen_joint_results else {}
    baseline_joint_stats = analyze_joint_pred_by_type(baseline_joint_results) if baseline_joint_results else {}

    generate_joint_latex_table(
        gen_joint_stats, baseline_joint_stats,
        os.path.join(output_dir, "joint_prediction_results.tex")
    )

    # Generate Part Segmentation LaTeX table
    if gen_part_seg_results or baseline_part_seg_results:
        print("\nGenerating Part Segmentation LaTeX table...")
        generate_part_seg_latex_table(
            gen_part_seg_results, baseline_part_seg_results,
            available_objects,
            os.path.join(output_dir, "part_segmentation_results.tex")
        )

    # Generate plots
    generate_comparison_plots(
        gen_joint_stats, baseline_joint_stats, 
        gen_chamfer_results, baseline_chamfer_results, 
        gen_joint_results, baseline_joint_results, # Pass raw results for pie charts
        output_dir
    )

def analyze_results(joint_results, chamfer_results, part_seg_results, available_objects, method_name=""):
    """Analyze and print evaluation results for a single method."""
    # Skip if no results
    if not joint_results and not chamfer_results and not part_seg_results:
        print("No results to analyze")
        return

    # Analyze joint predictions by type
    if joint_results:
        joint_stats = analyze_joint_pred_by_type(joint_results)
        print_joint_prediction_results(joint_stats, title=f"Joint Prediction Results ({method_name})")
    
    # Analyze chamfer distances
    if chamfer_results:
        chamfer_stats_by_cat = defaultdict(list)
        for obj_id, (distance, category, object_name) in chamfer_results.items():
            chamfer_stats_by_cat[category].append(distance)
        
        # Create stats dict in the format expected by print_chamfer_results
        analyzed_chamfer = {}
        for category, distances in sorted(chamfer_stats_by_cat.items()):
            if distances:
                analyzed_chamfer[category] = {
                    'count': len(distances),
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'median': np.median(distances),
                    'min': np.min(distances),
                    'max': np.max(distances)
                }
        
        # Add overall average
        all_distances = [d for distances in chamfer_stats_by_cat.values() for d in distances]
        if all_distances:
            analyzed_chamfer['Average'] = {
                'count': len(all_distances),
                'mean': np.mean(all_distances),
                'std': np.std(all_distances),
                'median': np.median(all_distances),
                'min': np.min(all_distances),
                'max': np.max(all_distances)
            }
        
        print_chamfer_results(analyzed_chamfer, title=f"Chamfer Distance Results ({method_name})")

    # Analyze part segmentation
    if part_seg_results:
        print_part_segmentation_results(part_seg_results, available_objects, title=f"Part Segmentation Results ({method_name})")

def print_part_segmentation_results(part_seg_results, available_objects, title="Part Segmentation Results"):
    """Prints a formatted table for part segmentation results."""
    print(f"\n{title}")
    print("=" * len(title))
    
    if not part_seg_results:
        print("No part segmentation results to display.")
        return

    by_category = defaultdict(list)
    obj_to_cat = {obj['object_number']: obj['category'] for obj in available_objects}
    for obj_id, result in part_seg_results.items():
        category = obj_to_cat.get(obj_id, "Unknown")
        by_category[category].append(result)

    print(f"{'Category':<20} {'Obj Count':<10} {'Count Acc.':<12} {'Mean CD':<10} {'Recall':<10}")
    print("-" * 75)

    for category, results in sorted(by_category.items()):
        count = len(results)
        count_acc = np.mean([r['part_count_correct'] for r in results])
        cds = [r['mean_chamfer_distance'] for r in results if np.isfinite(r['mean_chamfer_distance'])]
        mean_cd = np.mean(cds) if cds else float('nan')
        gt_parts = sum(r['part_count_gt'] for r in results)
        matched = sum(r['matched_parts'] for r in results)
        recall = matched / gt_parts if gt_parts > 0 else 0
        print(f"{category:<20} {count:<10} {f'{count_acc:.1%}':<12} {f'{mean_cd:.4f}':<10} {f'{recall:.1%}':<10}")

    # Overall stats
    print("-" * 75)
    num_objects = len(part_seg_results)
    part_count_correct_rate = np.mean([res['part_count_correct'] for res in part_seg_results.values()])
    valid_cds = [res['mean_chamfer_distance'] for res in part_seg_results.values() if np.isfinite(res['mean_chamfer_distance'])]
    mean_cd = np.mean(valid_cds) if valid_cds else float('nan')
    total_gt_parts = sum(res['part_count_gt'] for res in part_seg_results.values())
    total_matched = sum(res['matched_parts'] for res in part_seg_results.values())
    part_recall = total_matched / total_gt_parts if total_gt_parts > 0 else 0
    print(f"{'Overall':<20} {num_objects:<10} {f'{part_count_correct_rate:.1%}':<12} {f'{mean_cd:.4f}':<10} {f'{part_recall:.1%}':<10}")

def print_joint_prediction_results(joint_results, title="Joint Prediction Results"):
    """
    Print a nicely formatted table of joint prediction results.
    This version shows overall success rate per category, without joint type breakdown.
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    regular_categories = {k: v for k, v in joint_results.items() if k.lower() not in ['average', 'total', 'overall']}
    aggregate_categories = {k: v for k, v in joint_results.items() if k.lower() in ['average', 'total', 'overall']}
    
    def get_sort_key(item):
        obj_type, joint_data = item
        total_stats = joint_data.get('total', {})
        if isinstance(total_stats, dict):
            return (-total_stats.get('success_rate', 0), -total_stats.get('count', 0))
        return (0, 0)
    
    sorted_regular = sorted(regular_categories.items(), key=get_sort_key)
    
    for obj_type, joint_data in sorted_regular:
        if not isinstance(joint_data, dict): continue
        print(f"\n{obj_type}:")
        print("-" * (len(obj_type) + 1))
        print(f"{'Joint Type':<12} {'Count':<8} {'Success Rate':<12}")
        print("-" * 35)
        
        joint_items = [(jt, stats) for jt, stats in joint_data.items() if isinstance(stats, dict) and 'success_rate' in stats]
        joint_items.sort(key=lambda x: (-x[1]['success_rate'], -x[1]['count']))
        
        for joint_type, stats in joint_items:
            success_rate = f"{stats['success_rate']:.1%}"
            print(f"{joint_type:<12} {stats['count']:<8} {success_rate:<12}")
    
    for obj_type, joint_data in aggregate_categories.items():
        if not isinstance(joint_data, dict): continue
        print(f"\n{obj_type}:")
        print("-" * (len(obj_type) + 1))
        print(f"{'Joint Type':<12} {'Count':<8} {'Success Rate':<12}")
        print("-" * 35)
        
        joint_items = [(jt, stats) for jt, stats in joint_data.items() if isinstance(stats, dict) and 'success_rate' in stats]
        joint_items.sort(key=lambda x: (-x[1]['success_rate'], -x[1]['count']))
        
        for joint_type, stats in joint_items:
            success_rate = f"{stats['success_rate']:.1%}"
            print(f"{joint_type:<12} {stats['count']:<8} {success_rate:<12}")

def print_chamfer_results(chamfer_stats, title="URDF Chamfer Distance Results"):
    """
    Print a nicely formatted table of chamfer distance results.
    (Adapted from metric.py)
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(f"{'Category':<15} {'Count':<8} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 85)
    
    regular_rows = {k: v for k, v in chamfer_stats.items() if k.lower() not in ['average', 'total', 'overall']}
    aggregate_rows = {k: v for k, v in chamfer_stats.items() if k.lower() in ['average', 'total', 'overall']}
    
    sorted_regular = sorted(regular_rows.items(), key=lambda x: x[1]['mean'])
    
    for obj_type, stats in sorted_regular:
        mean_val = f"{stats['mean']:.4f}"
        std_val = f"{stats['std']:.4f}"
        median_val = f"{stats['median']:.4f}"
        min_val = f"{stats['min']:.4f}"
        max_val = f"{stats['max']:.4f}"
        print(f"{obj_type:<15} {stats['count']:<8} {mean_val:<10} {std_val:<10} {median_val:<10} {min_val:<10} {max_val:<10}")
    
    if aggregate_rows:
        print("-" * 85)
        for obj_type, stats in aggregate_rows.items():
            mean_val = f"{stats['mean']:.4f}"
            std_val = f"{stats['std']:.4f}"
            median_val = f"{stats['median']:.4f}"
            min_val = f"{stats['min']:.4f}"
            max_val = f"{stats['max']:.4f}"
            print(f"{obj_type:<15} {stats['count']:<8} {mean_val:<10} {std_val:<10} {median_val:<10} {min_val:<10} {max_val:<10}")

def generate_joint_latex_table(gen_stats, baseline_stats, output_path):
    """Generates a LaTeX table for joint prediction results, without revolute/prismatic breakdown."""
    header = r"""
\begin{table*}[t]
\centering
\caption{Joint Prediction Success Rate Comparison}
\label{tab:joint_prediction_artvip}
\resizebox{0.8\textwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Category} & \textbf{Gen. Joints} & \textbf{Gen. Success} & \textbf{Base. Joints} & \textbf{Base. Success} \\
\midrule
"""
    footer = r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    body = ""
    all_gen_total = 0
    all_gen_success = 0
    all_base_total = 0
    all_base_success = 0

    # Get all categories from both stats dicts, excluding 'Average'
    all_categories = sorted(list(set(gen_stats.keys()) | set(baseline_stats.keys()) - {'Average'}))

    for category in all_categories:
        gen_data = gen_stats.get(category, {})
        base_data = baseline_stats.get(category, {})
        
        # Safely get total and successful counts, handling both dict and int values
        gen_total_val = gen_data.get('total_joints', 0)
        gen_succ_val = gen_data.get('successful_joints', 0)
        base_total_val = base_data.get('total_joints', 0)
        base_succ_val = base_data.get('successful_joints', 0)

        gen_total = sum(gen_total_val.values()) if isinstance(gen_total_val, dict) else gen_total_val
        gen_succ = sum(gen_succ_val.values()) if isinstance(gen_succ_val, dict) else gen_succ_val
        base_total = sum(base_total_val.values()) if isinstance(base_total_val, dict) else base_total_val
        base_succ = sum(base_succ_val.values()) if isinstance(base_succ_val, dict) else base_succ_val

        all_gen_total += gen_total
        all_gen_success += gen_succ
        all_base_total += base_total
        all_base_success += base_succ

        gen_rate = (gen_succ / gen_total * 100) if gen_total > 0 else 0
        base_rate = (base_succ / base_total * 100) if base_total > 0 else 0

        body += f"{category.replace('_', ' ').title()} & {gen_total} & {gen_rate:.1f}\\% & {base_total} & {base_rate:.1f}\\% \\\\\n"

    # Add overall average
    avg_gen_rate = (all_gen_success / all_gen_total * 100) if all_gen_total > 0 else 0
    avg_base_rate = (all_base_success / all_base_total * 100) if all_base_total > 0 else 0
    body += r"\midrule" + "\n"
    body += f"\\textbf{{Overall}} & \\textbf{{{all_gen_total}}} & \\textbf{{{avg_gen_rate:.1f}\\%}} & \\textbf{{{all_base_total}}} & \\textbf{{{avg_base_rate:.1f}\\%}} \\\\\n"

    with open(output_path, "w") as f:
        f.write(header + body + footer)
    print(f"Joint prediction LaTeX table saved to {output_path}")


def generate_part_seg_latex_table(gen_results, base_results, available_objects, output_path, group_by='category'):
    """Generates a LaTeX table for part segmentation results with counts and 2-decimal CD."""
    header = r"""
\begin{table*}[t]
\centering
\caption{Part Segmentation and Reconstruction Comparison}
\label{tab:part_segmentation_artvip}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{\textbf{Generated (Ours)}} & \multicolumn{3}{c}{\textbf{Baseline}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Category} & \textbf{Count Acc.} & \textbf{Mean CD (n)} & \textbf{Recall} & \textbf{Count Acc.} & \textbf{Mean CD (n)} & \textbf{Recall} \\
\midrule
"""
    footer = r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    body = ""

    if group_by == 'category':
        obj_to_group = {obj['object_number']: obj['category'] for obj in available_objects}
    else: # subcategory
        obj_to_group = {obj['object_number']: obj['object_name'] for obj in available_objects}

    all_groups = set(obj_to_group.values())
    
    def get_stats(results, group_name):
        group_results = [res for obj_id, res in results.items() if obj_to_group.get(obj_id) == group_name]
        if not group_results:
            return 0, float('nan'), 0, 0
        
        count = len(group_results)
        count_acc = np.mean([r['part_count_correct'] for r in group_results])
        cds = [r['mean_chamfer_distance'] for r in group_results if np.isfinite(r['mean_chamfer_distance'])]
        mean_cd = np.mean(cds) if cds else float('nan')
        gt_parts = sum(r['part_count_gt'] for r in group_results)
        matched = sum(r['matched_parts'] for r in group_results)
        recall = matched / gt_parts if gt_parts > 0 else 0
        return count_acc, mean_cd, recall, count

    for group in sorted(list(all_groups)):
        gen_acc, gen_cd, gen_recall, gen_count = get_stats(gen_results, group)
        base_acc, base_cd, base_recall, base_count = get_stats(base_results, group)
        
        gen_cd_str = f"{gen_cd:.2f} (n={gen_count})" if not np.isnan(gen_cd) else "N/A"
        base_cd_str = f"{base_cd:.2f} (n={base_count})" if not np.isnan(base_cd) else "N/A"

        body += (f"{group.replace('_', ' ').title()} & "
                 f"{(gen_acc*100):.1f}\\% & {gen_cd_str} & {(gen_recall*100):.1f}\\% & "
                 f"{(base_acc*100):.1f}\\% & {base_cd_str} & {(base_recall*100):.1f}\\% \\\\\n")

    # Overall stats
    def get_overall_stats(results):
        if not results:
            return 0, float('nan'), 0, 0
        
        count = len(results)
        count_acc = np.mean([res['part_count_correct'] for res in results.values()])
        valid_cds = [res['mean_chamfer_distance'] for res in results.values() if np.isfinite(res['mean_chamfer_distance'])]
        mean_cd = np.mean(valid_cds) if valid_cds else float('nan')
        total_gt_parts = sum(res['part_count_gt'] for res in results.values())
        total_matched = sum(res['matched_parts'] for res in results.values())
        recall = total_matched / total_gt_parts if total_gt_parts > 0 else 0
        return count_acc, mean_cd, recall, count

    gen_acc_ov, gen_cd_ov, gen_recall_ov, gen_count_ov = get_overall_stats(gen_results)
    base_acc_ov, base_cd_ov, base_recall_ov, base_count_ov = get_overall_stats(base_results)
    
    gen_cd_ov_str = f"\\textbf{{{gen_cd_ov:.2f}}} (n={gen_count_ov})" if not np.isnan(gen_cd_ov) else "N/A"
    base_cd_ov_str = f"\\textbf{{{base_cd_ov:.2f}}} (n={base_count_ov})" if not np.isnan(base_cd_ov) else "N/A"

    body += r"\midrule" + "\n"
    body += (f"\\textbf{{Overall}} & "
             f"\\textbf{{{(gen_acc_ov*100):.1f}\\%}} & {gen_cd_ov_str} & \\textbf{{{(gen_recall_ov*100):.1f}\\%}} & "
             f"\\textbf{{{(base_acc_ov*100):.1f}\\%}} & {base_cd_ov_str} & \\textbf{{{(base_recall_ov*100):.1f}\\%}} \\\\\n")

    with open(output_path, "w") as f:
        f.write(header + body + footer)
    print(f"Part segmentation LaTeX table saved to {output_path}")


def generate_chamfer_latex_table(gen_chamfer, baseline_chamfer, output_path):
    pass

def generate_comparison_plots(gen_joint_stats, baseline_joint_stats, gen_chamfer_results, baseline_chamfer_results, gen_joint_results, baseline_joint_results, output_dir):
    """Generates and saves comparison plots in the style of metric.py."""
    
    # Plot 1: Joint Success Rate by Category (Generated)
    if gen_joint_stats:
        plot_joint_prediction_results(
            gen_joint_stats,
            title="Generated Method: Joint Success Rate",
            save_path=os.path.join(output_dir, "gen_joint_success_by_category.png")
        )

    # Plot 2: Joint Success Rate by Category (Baseline)
    if baseline_joint_stats:
        plot_joint_prediction_results(
            baseline_joint_stats,
            title="Baseline Method: Joint Success Rate",
            save_path=os.path.join(output_dir, "baseline_joint_success_by_category.png")
        )

    # Plot 3: Failure Breakdown Pie Chart (Generated)
    if gen_joint_results:
        plot_failure_breakdown_pie(
            gen_joint_results,
            title="Failure Breakdown",
            method_name="Generated Method",
            save_path=os.path.join(output_dir, "gen_failure_pie_chart.png")
        )

    # Plot 4: Failure Breakdown Pie Chart (Baseline)
    if baseline_joint_results:
        plot_failure_breakdown_pie(
            baseline_joint_results,
            title="Failure Breakdown",
            method_name="Baseline Method",
            save_path=os.path.join(output_dir, "baseline_failure_pie_chart.png")
        )

    # Plot 5: Chamfer Distance by Category (Comparison)
    if gen_chamfer_results or baseline_chamfer_results:
        gen_by_cat = defaultdict(list)
        for _, (dist, cat, _) in gen_chamfer_results.items():
            gen_by_cat[cat].append(dist)
        base_by_cat = defaultdict(list)
        for _, (dist, cat, _) in baseline_chamfer_results.items():
            base_by_cat[cat].append(dist)
        
        categories = sorted(set(gen_by_cat.keys()) | set(base_by_cat.keys()))
        if not categories: return

        gen_means = [np.mean(gen_by_cat.get(cat, [np.nan])) for cat in categories]
        base_means = [np.mean(base_by_cat.get(cat, [np.nan])) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, gen_means, width, label='Generated')
        rects2 = ax.bar(x + width/2, base_means, width, label='Baseline')

        ax.set_ylabel('Mean Chamfer Distance')
        ax.set_title('Mean Chamfer Distance by Category (lower is better)')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend()
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "chamfer_distance_comparison.png"))
        plt.close(fig)
        print(f"Plot saved to {os.path.join(output_dir, 'chamfer_distance_comparison.png')}")

def plot_joint_prediction_results(joint_results, title="Joint Prediction Success Rate by Object Categories",
                                 figsize=(10, 12), save_path=None, show_average=True, joint_type_filter=None):
    """
    Create a horizontal bar plot of joint prediction success rates by category.
    (Adapted from metric.py)
    """
    regular_categories = {k: v for k, v in joint_results.items() if k.lower() not in ['average', 'total', 'overall']}
    aggregate_categories = {k: v for k, v in joint_results.items() if k.lower() in ['average', 'total', 'overall']}
    
    if joint_type_filter is None:
        joint_type_filter = 'total'
    
    category_data = []
    for obj_type, joint_data in regular_categories.items():
        if isinstance(joint_data, dict) and joint_type_filter in joint_data:
            stats = joint_data[joint_type_filter]
            if isinstance(stats, dict) and 'success_rate' in stats:
                count = stats['count']
                success_rate = stats['success_rate']
                category_data.append((obj_type, success_rate, count))
    
    if not category_data:
        print(f"No data to plot for joint type '{joint_type_filter}'.")
        return

    category_data.sort(key=lambda x: x[1])
    
    categories = [f"{item[0]} (n={item[2]})" for item in category_data]
    success_rates = [item[1] for item in category_data]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(categories, success_rates, color='lightblue', edgecolor='steelblue', linewidth=0.5)
    
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        text_pos = rate / 2 if rate > 0.1 else rate + 0.01
        ha = 'center' if rate > 0.1 else 'left'
        ax.text(text_pos, i, f'{rate:.1%}', va='center', ha=ha, fontsize=10, fontweight='bold', color='black')
    
    if show_average and 'Average' in aggregate_categories:
        avg_data = aggregate_categories['Average']
        if isinstance(avg_data, dict) and joint_type_filter in avg_data:
            avg_stats = avg_data[joint_type_filter]
            if isinstance(avg_stats, dict) and 'success_rate' in avg_stats:
                avg_rate = avg_stats['success_rate']
                ax.axvline(x=avg_rate, color='red', linestyle='--', linewidth=2, alpha=0.8)
                ax.text(avg_rate + 0.02, len(categories) * 0.95, f'Average\n{avg_rate:.1%}', color='red', fontweight='bold', ha='left', va='top')
    
    ax.set_xlabel('Success Rate', fontsize=12)
    joint_type_title = joint_type_filter.title() if joint_type_filter != 'total' else 'Overall'
    ax.set_title(f"{title} ({joint_type_title})", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close(fig)

def get_adaptive_thresholds(naming_convention="original"):
    """Returns the thresholds assumed to be used for adaptive scoring."""
    if naming_convention == "sem":  
        return {
            "joint_origin": 0.1,  # meters
            "joint_axis_angle": 15.0,  # degrees
            "joint_limit": 0.5,  # radians
        }
    else:  
        return {
            "joint_origin": 0.05,
            "joint_axis_angle": 5.0,
            "joint_limit": 0.1,
        }

def print_joint_comparison_details(gt_joints, pred_joints, method_name):
    """Performs matching and prints a detailed breakdown of joint differences."""
    print(f"\n{'='*20} Detailed Comparison for: {method_name} {'='*20}")
    if not gt_joints or not pred_joints:
        print("Missing ground truth or predicted joints.")
        return

    # --- 1. Pre-filter joints to ensure they have position data ---
    matchable_gt_items = [{'name': name, 'joint': joint} for name, joint in gt_joints.items() if joint.get('origin', {}).get('xyz') is not None]
    matchable_pred_items = [{'name': name, 'joint': joint} for name, joint in pred_joints.items() if joint.get('origin', {}).get('xyz') is not None]
    
    unmatchable_gt_names = [name for name in gt_joints if name not in {j['name'] for j in matchable_gt_items}]
    unmatchable_pred_names = [name for name in pred_joints if name not in {j['name'] for j in matchable_pred_items}]

    if not matchable_gt_items or not matchable_pred_items:
        print("Could not perform matching because one or both sets had no joints with valid position data.")
        if unmatchable_gt_names:
            print(f"  - GT joints missing position data: {unmatchable_gt_names}")
        if unmatchable_pred_names:
            print(f"  - Predicted joints missing position data: {unmatchable_pred_names}")
        return

    # --- 2. Build similarity matrix ONLY for matchable joints ---
    similarity_matrix = np.zeros((len(matchable_gt_items), len(matchable_pred_items)))
    for i, gt_item in enumerate(matchable_gt_items):
        gt_pos = np.array(gt_item['joint']['origin']['xyz'])
        for j, pred_item in enumerate(matchable_pred_items):
            pred_pos = np.array(pred_item['joint']['origin']['xyz'])
            similarity_matrix[i, j] = np.linalg.norm(gt_pos - pred_pos)

    # --- 3. Run Hungarian matching on the clean matrix ---
    gt_indices, pred_indices = linear_sum_assignment(similarity_matrix)

    MATCHING_DISTANCE_THRESHOLD = 0.5 # meters
    matched_gt_indices = set()
    matched_pred_indices = set()
    optimal_assignment_map_gt_to_pred = {gt_idx: pred_idx for gt_idx, pred_idx in zip(gt_indices, pred_indices)}

    print(f"\n--- Matched & Evaluated Joints (Position distance < {MATCHING_DISTANCE_THRESHOLD}m) ---")
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        matching_cost = similarity_matrix[gt_idx, pred_idx]
        if matching_cost > MATCHING_DISTANCE_THRESHOLD:
            continue

        matched_gt_indices.add(gt_idx)
        matched_pred_indices.add(pred_idx)
        
        gt_item = matchable_gt_items[gt_idx]
        pred_item = matchable_pred_items[pred_idx]
        gt_name, pred_name = gt_item['name'], pred_item['name']
        
        print(f"\n--- Match: (GT) {gt_name} <-> (Pred) {pred_name} (Positional Distance: {matching_cost:.4f}m) ---")
        
        naming_convention = "sem" if method_name.lower() == "generated" else "original"
        thresholds = get_adaptive_thresholds(naming_convention)
        
        try:
            joint_diff = compute_joint_diff_with_continuous_support(gt_item['joint'], pred_item['joint'])
            score, failure_reason = compute_joint_diff_score_adaptive(joint_diff, naming_convention)

            print(f"  [Thresholds ({naming_convention})]")
            for k, v in thresholds.items(): print(f"    - {k:<18}: {v}")
            
            print("  [Differences]")
            if joint_diff:
                for k, v in joint_diff.items(): print(f"    - {k:<18}: {f'{v:.4f}' if isinstance(v, float) else v}")
            else:
                print("    Could not compute joint differences.")

            print("  [Result]")
            print(f"    - Success: {failure_reason == 'success'}")
            print(f"    - Failure Reason: {failure_reason}")
        except Exception as e:
            print(f"  [ERROR] Could not evaluate joint pair: {e}")

    print("\n--- Analysis of Unmatched Joints ---")
    
    # Report joints that were filtered out due to missing data
    if unmatchable_gt_names:
        print(f"\n  [Unmatchable GT Joints (missing position data): {len(unmatchable_gt_names)}]")
        for name in unmatchable_gt_names: print(f"  - {name}")
    if unmatchable_pred_names:
        print(f"\n  [Unmatchable Predicted Joints (missing position data): {len(unmatchable_pred_names)}]")
        for name in unmatchable_pred_names: print(f"  - {name}")

    # Report joints that were part of the matching but were not matched
    unmatched_gt_indices = set(range(len(matchable_gt_items))) - matched_gt_indices
    if unmatched_gt_indices:
        print(f"\n  [Unmatched GT Joints (from matching pool): {len(unmatched_gt_indices)}]")
        for gt_idx in sorted(list(unmatched_gt_indices)):
            gt_name = matchable_gt_items[gt_idx]['name']
            print(f"\n  - GT Joint: '{gt_name}'")
            best_pred_idx = np.argmin(similarity_matrix[gt_idx, :])
            min_dist = similarity_matrix[gt_idx, best_pred_idx]
            best_pred_name = matchable_pred_items[best_pred_idx]['name']
            if gt_idx in optimal_assignment_map_gt_to_pred:
                assigned_pred_idx = optimal_assignment_map_gt_to_pred_map_gt_to_pred[gt_idx]
                assigned_dist = similarity_matrix[gt_idx, assigned_pred_idx]
                print(f"    Reason: Optimal assignment was to '{matchable_pred_items[assigned_pred_idx]['name']}', but the distance of {assigned_dist:.4f}m exceeded the {MATCHING_DISTANCE_THRESHOLD}m threshold.")
            else:
                print(f"    Reason: Not part of the optimal assignment (e.g., due to joint count mismatch).")
                print(f"    Closest Candidate: '{best_pred_name}' with a distance of {min_dist:.4f}m.")

    unmatched_pred_indices = set(range(len(matchable_pred_items))) - matched_pred_indices
    if unmatched_pred_indices:
        print(f"\n  [Unmatched Predicted Joints (from matching pool): {len(unmatched_pred_indices)}]")
        for pred_idx in sorted(list(unmatched_pred_indices)):
            pred_name = matchable_pred_items[pred_idx]['name']
            print(f"\n  - Pred Joint: '{pred_name}'")
            best_gt_idx = np.argmin(similarity_matrix[:, pred_idx])
            min_dist = similarity_matrix[best_gt_idx, pred_idx]
            best_gt_name = matchable_gt_items[best_gt_idx]['name']
            if pred_idx not in optimal_assignment_map_gt_to_pred.values():
                print(f"    Reason: Not part of the optimal assignment.")
                print(f"    Closest Candidate: '{best_gt_name}' with a distance of {min_dist:.4f}m.")

    print(f"\n{'='*60}\n")


def debug_single_object_joints(object_id, debug=False):
    """Loads a single object and prints detailed joint evaluation info."""
    print(f"--- Running Inspection for Object ID: {object_id} ---")

    # Use the existing function to find all available objects
    available_objects = get_available_objects(debug=debug)
    if not available_objects:
        print("Could not find any available objects with both GT and prediction data.")
        return

    # Find the specific object the user wants to inspect from the valid list
    obj_info = next((obj for obj in available_objects if obj['object_number'] == object_id), None)

    if not obj_info:
        print(f"\nError: Could not find object '{object_id}' among the {len(available_objects)} available objects.")
        print("Please ensure the object ID is correct and that it has corresponding data in both the")
        print(f"processed ('{PROCESSED_DIR}') and results ('{GEN_RESULT_DIR}' or '{BASELINE_RESULT_DIR}') directories.")
        
        # Print a sample of found object IDs to help debugging
        all_found_ids = sorted([obj['object_number'] for obj in available_objects])
        print("\nHere is a sample of object IDs that were found:")
        if len(all_found_ids) > 20:
            for i in range(10):
                print(f"  - {all_found_ids[i]}")
            print("  ...")
            for i in range(len(all_found_ids) - 10, len(all_found_ids)):
                print(f"  - {all_found_ids[i]}")
        else:
            for found_id in all_found_ids:
                print(f"  - {found_id}")
        
        # Suggest partial matches if any exist
        partial_matches = [found_id for found_id in all_found_ids if object_id in found_id or found_id in object_id]
        if partial_matches:
            print("\nDid you possibly mean one of these?")
            for match in partial_matches:
                print(f"  - {match}")
        return

    print(f"Found object '{object_id}' in category '{obj_info['category']}'.")

    # Load GT joints
    gt_joints_path = os.path.join(obj_info['gt_path'], "robot_joints.json")
    gt_joints = load_robot_joints(gt_joints_path)
    if not gt_joints:
        print(f"Could not load ground truth joints from {gt_joints_path}")
        return
    print(f"Loaded {len(gt_joints)} GT joints.")

    # --- Evaluate Generated Method ---
    gen_pred_dir = obj_info.get('gen_pred_dir')
    if gen_pred_dir and os.path.exists(gen_pred_dir):
        gen_urdf_path, _, _ = find_best_prediction(gen_pred_dir, debug=debug)
        if gen_urdf_path and os.path.exists(gen_urdf_path):
            print(f"Found Generated prediction: {gen_urdf_path}")
            gen_pred_joints = extract_joints_from_urdf(gen_urdf_path, debug=debug)
            print_joint_comparison_details(gt_joints, gen_pred_joints, "Generated")
        else:
            print("\nCould not find a valid URDF prediction for the Generated method.")
    else:
        print("\nNo prediction directory found for the Generated method.")


    # --- Evaluate Baseline Method ---
    baseline_pred_dir = obj_info.get('baseline_pred_dir')
    if baseline_pred_dir and os.path.exists(baseline_pred_dir):
        baseline_urdf_path = os.path.join(baseline_pred_dir, "mobility.urdf")
        if os.path.exists(baseline_urdf_path):
            print(f"Found Baseline prediction: {baseline_urdf_path}")
            baseline_pred_joints = extract_joints_from_urdf(baseline_urdf_path, debug=debug)
            print_joint_comparison_details(gt_joints, baseline_pred_joints, "Baseline")
        else:
            print("\nCould not find a URDF prediction for the Baseline method.")
    else:
        print("\nNo prediction directory found for the Baseline method.")



def plot_failure_breakdown_pie(joint_diffs, title="Failure Breakdown", 
                               figsize=(8, 8), save_path=None, method_name="",
                               exclude_failures=None):
    """
    Create a pie chart showing the breakdown of failure types.
    (Adapted from metric.py)
    """
    all_failure_counts = defaultdict(int)
    for result in joint_diffs.values():
        all_failure_counts[result['failure_reason']] += 1
    
    if not all_failure_counts:
        print(f"No failure data to plot for {method_name}.")
        return

    total_joints = sum(all_failure_counts.values())
    
    display_failure_counts = all_failure_counts.copy()
    if exclude_failures:
        for failure_type in exclude_failures:
            if failure_type in display_failure_counts:
                del display_failure_counts[failure_type]
    
    if exclude_failures:
        title = f"{title}\n(excluding {', '.join(exclude_failures)})"
    
    color_map = {
        'success': '#90EE90', 'joint_type': '#4682B4', 'joint_axis': '#87CEEB',
        'joint_origin': '#ADD8E6', 'joint_limit': '#FFE4E1', 'joint_missing': '#F0E68C',
        'no_predicted_joints': '#DDA0DD', 'no_valid_result': '#D3D3D3',
        'joint_name_mismatch': '#FFDAB9', 'joint_diff_computation_failed': '#FFA07A',
        'joint_analysis_error': '#A9A9A9',
    }
    
    labels, sizes, colors = [], [], []
    
    sorted_failures = sorted(display_failure_counts.items(), key=lambda x: x[1], reverse=True)
    if 'success' in display_failure_counts:
        sorted_failures = [item for item in sorted_failures if item[0] != 'success']
        sorted_failures.insert(0, ('success', display_failure_counts['success']))
    
    for failure_type, count in sorted_failures:
        percentage = (count / total_joints) * 100
        labels.append(failure_type.replace("_", " ").title())
        sizes.append(percentage)
        colors.append(color_map.get(failure_type, '#D3D3D3'))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, pctdistance=0.85, labeldistance=1.1,
                                      textprops={'fontsize': 12})

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(centre_circle)
    
    full_title = f"{method_name}\n{title}" if method_name else title
    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to {save_path}")
    
    plt.close(fig)
