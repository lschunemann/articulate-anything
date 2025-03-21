import numpy as np
import trimesh
import os
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import argparse
import json
from pathlib import Path
import glob
from scipy.spatial.transform import Rotation

def load_mesh(file_path, debug=True):
    """Load a mesh from .obj or .glb file with improved handling for both formats."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if debug:
        print(f"Loading mesh from {file_path}...")
    
    try:
        # Special handling for OBJ files
        if file_path.lower().endswith('.obj'):
            try:
                # Try loading with process=True and maintain_order=True
                mesh = trimesh.load_mesh(file_path, process=True, force='mesh', maintain_order=True)
                
                if debug:
                    print(f"Initial load: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
                # Check if the mesh has valid faces
                if len(mesh.faces) == 0:
                    print(f"Warning: No faces found in {file_path}, trying alternative loading method")
                    # Try alternative loading method
                    mesh = trimesh.exchange.load.load(file_path, file_type='obj')
                    
                    if debug:
                        print(f"Alternative load: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                        
            except Exception as e:
                print(f"Error loading OBJ file: {e}")
                # Try alternative loading method with explicit OBJ loader
                try:
                    mesh = trimesh.exchange.load.load(file_path, file_type='obj')
                    if debug:
                        print(f"Fallback load: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                except Exception as e2:
                    raise ValueError(f"Failed to load {file_path}: {e2}")
        
        # Handle GLB and other formats with multiple fallback methods
        else:
            try:
                # First try standard loading
                loaded = trimesh.load(file_path)
                
            except Exception as e1:
                print(f"Standard loading failed for {file_path}: {e1}")
                try:
                    # Try explicit GLB loader
                    loaded = trimesh.exchange.load.load(file_path, file_type='glb')
                except Exception as e2:
                    print(f"Explicit GLB loader failed: {e2}")
                    try:
                        # Try loading as GLTF
                        loaded = trimesh.exchange.load.load(file_path, file_type='gltf')
                    except Exception as e3:
                        # As a last resort, try loading with pygltflib
                        try:
                            import pygltflib
                            from pygltflib import GLTF2
                            
                            # Load with pygltflib and convert to trimesh
                            gltf = GLTF2().load(file_path)
                            # This is a simplified conversion - you might need to expand this
                            # based on your specific GLB structure
                            
                            # For now, just raise the original error
                            raise ValueError(f"Failed to load {file_path} with multiple methods: {e1}")
                        except ImportError:
                            raise ValueError(f"Failed to load {file_path} with multiple methods: {e1}")
            
            # Check if the loaded object is a Scene
            if isinstance(loaded, trimesh.Scene):
                # Extract all meshes from the scene
                meshes = []
                for geometry in loaded.geometry.values():
                    if isinstance(geometry, trimesh.Trimesh):
                        meshes.append(geometry)
                
                # If there are multiple meshes, combine them into one
                if len(meshes) > 1:
                    mesh = trimesh.util.concatenate(meshes)
                    if debug:
                        print(f"Combined {len(meshes)} meshes from scene")
                elif len(meshes) == 1:
                    mesh = meshes[0]
                else:
                    raise ValueError(f"No valid meshes found in {file_path}")
            else:
                # If it's already a Trimesh object
                mesh = loaded
        
        # Verify the mesh has vertices
        if len(mesh.vertices) == 0:
            raise ValueError(f"Loaded mesh from {file_path} has no vertices")
            
        if debug:
            print(f"Successfully loaded mesh from {file_path} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
            print(f"Mesh bounds: {mesh.bounds}")
        
        return mesh
        
    except Exception as e:
        print(f"Failed to load mesh {file_path} with error: {e}")

def save_mesh_image(source_mesh, target_meshes, aligned_meshes, output_path, method_names):
    """Create and save a visualization of the original and aligned meshes for multiple methods."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Determine number of methods
    num_methods = len(target_meshes)
    
    # Create a figure with subplots: 1 for source + num_methods for targets + num_methods for aligned
    fig = plt.figure(figsize=(15, 5 * (1 + num_methods)))
    
    # Plot source mesh
    ax_source = fig.add_subplot(1 + num_methods, 3, 1, projection='3d')
    ax_source.set_title('Ground Truth Mesh')
    source_points = source_mesh.vertices
    ax_source.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
                c='blue', s=1, alpha=0.5)
    ax_source.set_box_aspect([1, 1, 1])
    ax_source.set_xlabel('X')
    ax_source.set_ylabel('Y')
    ax_source.set_zlabel('Z')
    
    # Colors for different methods
    colors = ['red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    # Plot each method's target and aligned meshes
    for i, (target_mesh, aligned_mesh, method_name) in enumerate(zip(target_meshes, aligned_meshes, method_names)):
        # Plot target mesh
        ax_target = fig.add_subplot(1 + num_methods, 3, 3*i + 2, projection='3d')
        ax_target.set_title(f'{method_name} Mesh')
        target_points = target_mesh.vertices
        ax_target.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                    c=colors[i % len(colors)], s=1, alpha=0.5)
        
        # Plot aligned mesh with target
        ax_aligned = fig.add_subplot(1 + num_methods, 3, 3*i + 3, projection='3d')
        ax_aligned.set_title(f'{method_name} Aligned with GT')
        aligned_points = aligned_mesh.vertices
        ax_aligned.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2], 
                    c=colors[i % len(colors)], s=1, alpha=0.5, label=f'{method_name}')
        ax_aligned.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
                    c='blue', s=1, alpha=0.5, label='Ground Truth')
        ax_aligned.legend()
        
        # Set equal aspect ratio for all plots
        for ax in [ax_target, ax_aligned]:
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_path}")

def chamfer_distance(points1, points2):
    """Compute the Chamfer distance between two point clouds."""
    # Build KD-trees for efficient nearest neighbor queries
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Find the nearest neighbor in points2 for each point in points1
    distances1, _ = tree1.query(points2)
    # Find the nearest neighbor in points1 for each point in points2
    distances2, _ = tree2.query(points1)
    
    # Compute the Chamfer distance
    chamfer_dist = np.mean(distances1) + np.mean(distances2)
    return chamfer_dist

def transform_points(points, params):
    """Apply transformation parameters to points."""
    # Extract parameters
    tx, ty, tz, rx, ry, rz, scale = params
    
    # Create a copy of the points to transform
    transformed_points = points.copy()
    
    # Apply scaling
    transformed_points = transformed_points * scale
    
    # Apply rotation (Euler angles)
    # Create rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combine rotation matrices
    R = Rz @ Ry @ Rx
    
    # Apply rotation
    transformed_points = transformed_points @ R.T
    
    # Apply translation
    transformed_points += np.array([tx, ty, tz])
    
    return transformed_points

def objective_function(params, source_points, target_points):
    """Objective function for alignment optimization."""
    # Transform source points
    transformed_points = transform_points(source_points, params)
    
    # Compute Chamfer distance between transformed source and target
    distance = chamfer_distance(transformed_points, target_points)
    return distance

def icp_align(source_points, target_points, max_iterations=50, tolerance=1e-6, rejection_ratio=0.2):
    """
    Enhanced ICP algorithm with outlier rejection and adaptive parameters.
    
    Args:
        source_points: Source point cloud
        target_points: Target point cloud
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        rejection_ratio: Ratio of correspondences to reject as outliers
    """
    # Make copies of the point clouds
    src = source_points.copy()
    tgt = target_points.copy()
    
    # Center the point clouds
    src_centroid = np.mean(src, axis=0)
    tgt_centroid = np.mean(tgt, axis=0)
    
    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid
    
    # Initialize transformation
    R = np.eye(3)
    t = np.zeros(3)
    scale = 1.0
    
    prev_error = float('inf')
    
    for i in range(max_iterations):
        # Find nearest neighbors
        tree = cKDTree(tgt_centered)
        distances, indices = tree.query(src_centered @ R.T * scale)
        
        # Reject outlier correspondences
        if rejection_ratio > 0:
            threshold = np.percentile(distances, (1-rejection_ratio)*100)
            valid = distances <= threshold
            if np.sum(valid) < 10:  # Ensure we have enough points
                valid = np.ones_like(distances, dtype=bool)
            
            # Use only valid correspondences
            src_valid = src_centered[valid]
            corresponding_points = tgt_centered[indices[valid]]
            weights = 1.0 / (distances[valid] + 1e-8)  # Weight by inverse distance
            weights = weights / np.sum(weights)  # Normalize weights
        else:
            src_valid = src_centered
            corresponding_points = tgt_centered[indices]
            weights = np.ones(len(src_valid)) / len(src_valid)
        
        # Compute weighted centroids
        src_weighted_centroid = np.sum(src_valid * weights[:, np.newaxis], axis=0)
        tgt_weighted_centroid = np.sum(corresponding_points * weights[:, np.newaxis], axis=0)
        
        # Center the points
        src_centered_weighted = src_valid - src_weighted_centroid
        tgt_centered_weighted = corresponding_points - tgt_weighted_centroid
        
        # Compute weighted covariance matrix
        H = np.zeros((3, 3))
        for j in range(len(src_valid)):
            H += weights[j] * np.outer(src_centered_weighted[j], tgt_centered_weighted[j])
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation
        R_new = Vt.T @ U.T
        
        # Ensure it's a proper rotation matrix (det=1)
        if np.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T
        
        # Estimate scale (optional)
        if np.sum(src_centered_weighted**2) > 0:
            numerator = 0
            denominator = 0
            for j in range(len(src_valid)):
                numerator += weights[j] * np.dot(tgt_centered_weighted[j], R_new @ src_centered_weighted[j])
                denominator += weights[j] * np.dot(src_centered_weighted[j], src_centered_weighted[j])
            
            scale_new = numerator / denominator if denominator > 0 else scale
            scale = max(0.01, min(100, scale_new))  # Limit scale to reasonable range
        
        # Compute translation
        t_new = tgt_centroid - scale * (R_new @ src_centroid)
        
        # Update transformation
        R = R_new
        t = t_new
        
        # Compute error
        transformed = (src_centered @ R.T) * scale
        current_error = np.mean(np.sqrt(np.sum((transformed - tgt_centered[indices])**2, axis=1)))
        
        # Check for convergence
        if abs(prev_error - current_error) < tolerance:
            print(f"ICP converged after {i+1} iterations with error {current_error:.6f}")
            break
        
        prev_error = current_error
    
    # Compute final transformation matrix
    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = t
    
    return T, current_error

def align_meshes(source_mesh, target_mesh, method_name, sample_count=20000):
    """
    Align source mesh to target mesh using a robust multi-stage approach.
    """
    # Sample points for alignment
    if len(source_mesh.faces) > 0:
        print(f"Sampling {sample_count} points from source mesh surface")
        source_points = source_mesh.sample(sample_count)
    else:
        source_points = source_mesh.vertices
    
    if len(target_mesh.faces) > 0:
        print(f"Sampling {sample_count} points from target mesh surface")
        target_points = target_mesh.sample(sample_count)
    else:
        target_points = target_mesh.vertices
    
    # STAGE 1: Normalize both point clouds
    # Center both point clouds
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # Scale to unit cube
    source_scale = np.max([np.ptp(source_centered[:, 0]), 
                          np.ptp(source_centered[:, 1]), 
                          np.ptp(source_centered[:, 2])])
    target_scale = np.max([np.ptp(target_centered[:, 0]), 
                          np.ptp(target_centered[:, 1]), 
                          np.ptp(target_centered[:, 2])])
    
    source_normalized = source_centered / source_scale
    target_normalized = target_centered / target_scale
    
    # STAGE 2: Try multiple initial alignments and pick the best
    best_transform = np.eye(4)
    best_distance = float('inf')
    
    # Try different initial rotations
    rotations = []
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        # Rotation around X
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
        rotations.append(Rx)
        
        # Rotation around Y
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
        rotations.append(Ry)
        
        # Rotation around Z
        Rz = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        rotations.append(Rz)
    
    # Try each rotation as initial alignment
    for rot in rotations:
        # Create transformation matrix for normalization
        init_transform = np.eye(4)
        init_transform[:3, :3] = rot[:3, :3] * (target_scale / source_scale)
        init_transform[:3, 3] = target_centroid - source_centroid @ rot[:3, :3].T * (target_scale / source_scale)
        
        # Apply initial transformation
        init_transformed = transform_points_with_matrix(source_points, init_transform)
        
        # Run ICP from this starting point
        icp_transform, error = icp_align(init_transformed, target_points)
        
        # Combine transformations
        combined_transform = icp_transform @ init_transform
        
        # Evaluate alignment
        aligned_points = transform_points_with_matrix(source_points, combined_transform)
        distance = chamfer_distance(aligned_points, target_points)
        
        if distance < best_distance:
            best_distance = distance
            best_transform = combined_transform
    
    # STAGE 3: Try PCA-based alignment
    try:
        # Compute covariance matrices
        source_cov = np.cov(source_normalized.T)
        target_cov = np.cov(target_normalized.T)
        
        # Get eigenvectors (principal axes)
        source_evals, source_evecs = np.linalg.eigh(source_cov)
        target_evals, target_evecs = np.linalg.eigh(target_cov)
        
        # Sort by eigenvalues (largest first)
        source_idx = np.argsort(source_evals)[::-1]
        source_evecs = source_evecs[:, source_idx]
        
        target_idx = np.argsort(target_evals)[::-1]
        target_evecs = target_evecs[:, target_idx]
        
        # Try all possible axis alignments (8 possibilities due to sign flips)
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    # Create modified eigenvector matrix with sign flips
                    mod_source_evecs = source_evecs.copy()
                    mod_source_evecs[:, 0] *= sx
                    mod_source_evecs[:, 1] *= sy
                    mod_source_evecs[:, 2] *= sz
                    
                    # Compute rotation matrix from principal axes
                    R_pca = target_evecs @ mod_source_evecs.T
                    
                    # Create transformation matrix
                    pca_transform = np.eye(4)
                    pca_transform[:3, :3] = R_pca * (target_scale / source_scale)
                    pca_transform[:3, 3] = target_centroid - source_centroid @ R_pca.T * (target_scale / source_scale)
                    
                    # Apply PCA transformation
                    pca_transformed = transform_points_with_matrix(source_points, pca_transform)
                    
                    # Run ICP from this starting point
                    icp_transform, error = icp_align(pca_transformed, target_points)
                    
                    # Combine transformations
                    combined_transform = icp_transform @ pca_transform
                    
                    # Evaluate alignment
                    aligned_points = transform_points_with_matrix(source_points, combined_transform)
                    distance = chamfer_distance(aligned_points, target_points)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_transform = combined_transform
    except np.linalg.LinAlgError:
        print("PCA alignment failed, using best rotation from grid search")
    
    # STAGE 4: Final ICP refinement
    aligned_points = transform_points_with_matrix(source_points, best_transform)
    final_transform, _ = icp_align(aligned_points, target_points, max_iterations=100, tolerance=1e-7)
    
    # Combine transformations
    final_combined_transform = final_transform @ best_transform
    
    # Apply the final transformation to the source mesh
    aligned_mesh = source_mesh.copy()
    aligned_mesh.apply_transform(final_combined_transform)
    
    # Extract transformation parameters for reporting
    # Translation
    tx, ty, tz = final_combined_transform[:3, 3]
    
    # Scale (approximated from the transformation matrix)
    scale_matrix = final_combined_transform[:3, :3]
    scale = np.cbrt(np.abs(np.linalg.det(scale_matrix)))
    
    # Rotation (convert to Euler angles)
    rotation_matrix = scale_matrix / scale
    try:
        r = Rotation.from_matrix(rotation_matrix)
        rx, ry, rz = r.as_euler('xyz')
    except:
        rx, ry, rz = 0, 0, 0
        print("Warning: Could not extract rotation angles from transformation matrix")
    
    # Pack parameters for reporting
    transform_params = [tx, ty, tz, rx, ry, rz, scale]
    
    # Calculate final chamfer distance
    final_aligned_points = transform_points_with_matrix(source_points, final_combined_transform)
    final_distance = chamfer_distance(final_aligned_points, target_points)
    
    # Create point clouds for visualization
    source_point_cloud = trimesh.points.PointCloud(source_points)
    target_point_cloud = trimesh.points.PointCloud(target_points)
    aligned_point_cloud = trimesh.points.PointCloud(final_aligned_points)
    
    return aligned_mesh, transform_params, final_distance, target_mesh, source_point_cloud, target_point_cloud, aligned_point_cloud

def save_mesh_image(source_point_cloud, target_point_clouds, aligned_point_clouds, output_path, method_names):
    """Create and save a visualization of the original and aligned point clouds."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Determine number of methods
    num_methods = len(target_point_clouds)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 5 * (1 + num_methods)))
    
    # Plot source point cloud
    ax_source = fig.add_subplot(1 + num_methods, 3, 1, projection='3d')
    ax_source.set_title('Ground Truth Points')
    source_points = source_point_cloud.vertices
    ax_source.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
                c='blue', s=1, alpha=0.5)
    ax_source.set_box_aspect([1, 1, 1])
    ax_source.set_xlabel('X')
    ax_source.set_ylabel('Y')
    ax_source.set_zlabel('Z')
    
    # Colors for different methods
    colors = ['red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    # Plot each method's target and aligned point clouds
    for i, (target_pc, aligned_pc, method_name) in enumerate(zip(target_point_clouds, aligned_point_clouds, method_names)):
        # Plot target point cloud
        ax_target = fig.add_subplot(1 + num_methods, 3, 3*i + 2, projection='3d')
        ax_target.set_title(f'{method_name} Points')
        target_points = target_pc.vertices
        ax_target.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                    c=colors[i % len(colors)], s=1, alpha=0.5)
        
        # Plot aligned point cloud with target
        ax_aligned = fig.add_subplot(1 + num_methods, 3, 3*i + 3, projection='3d')
        ax_aligned.set_title(f'{method_name} Aligned with GT')
        aligned_points = aligned_pc.vertices
        ax_aligned.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2], 
                    c=colors[i % len(colors)], s=1, alpha=0.5, label=f'{method_name}')
        ax_aligned.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
                    c='blue', s=1, alpha=0.5, label='Ground Truth')
        ax_aligned.legend()
        
        # Set equal aspect ratio for all plots
        for ax in [ax_target, ax_aligned]:
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_path}")

def transform_points_with_matrix(points, matrix):
    """Apply a 4x4 transformation matrix to points."""
    # Convert to homogeneous coordinates
    homogeneous_points = np.ones((len(points), 4))
    homogeneous_points[:, :3] = points
    
    # Apply transformation
    transformed_points = homogeneous_points @ matrix.T
    
    # Convert back to 3D coordinates
    return transformed_points[:, :3]

def compare_meshes(gt_path, method_paths, method_names, output_dir, save_aligned=True, force_recompute=False):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing results
    json_path = os.path.join(output_dir, "chamfer_comparison.json")
    existing_results = {}
    if os.path.exists(json_path) and not force_recompute:
        try:
            with open(json_path, 'r') as f:
                existing_results = json.load(f)
            print(f"Found existing results at {json_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = {}
    
    # Initialize results with existing data
    results = {
        "ground_truth_file": gt_path,
        "methods": existing_results.get("methods", {})
    }
    
    # Check if we need to process any methods at all
    methods_to_process = []
    for method_path, method_name in zip(method_paths, method_names):
        if method_name not in results["methods"] or force_recompute:
            methods_to_process.append((method_path, method_name))
        else:
            print(f"\nSkipping {method_name} - results already exist. Use --force to recompute.")
    
    # If no methods need processing, just return the existing results
    if not methods_to_process and not force_recompute:
        print("All methods already processed. No new computation needed.")
        return results
    
    # Load ground truth mesh only if we have methods to process
    print("Loading ground truth mesh...")
    gt_mesh = load_mesh(gt_path)
    
    # Initialize lists for visualization
    aligned_meshes = []
    target_meshes = []
    processed_method_names = []
    
    # For point cloud visualization
    source_point_cloud = None
    target_point_clouds = []
    aligned_point_clouds = []
    
    # Process each method that needs processing
    for method_path, method_name in methods_to_process:
        print(f"\nProcessing {method_name} mesh...")
        
        # Load method mesh
        method_mesh = load_mesh(method_path)
        
        # Align meshes with ICP
        print(f"Aligning {method_name} mesh with ground truth...")
        
        # Call the updated align_meshes function
        aligned_mesh, transform_params, final_distance, target_mesh_transformed, src_pc, tgt_pc, aligned_pc = align_meshes(
            gt_mesh, method_mesh, method_name
        )
        
        # Store for visualization
        aligned_meshes.append(aligned_mesh)
        target_meshes.append(target_mesh_transformed)
        processed_method_names.append(method_name)
        
        # Store point clouds for visualization
        if source_point_cloud is None:  # Only need one source point cloud
            source_point_cloud = src_pc
        target_point_clouds.append(tgt_pc)
        aligned_point_clouds.append(aligned_pc)
        
        # Output results
        print(f"\n{method_name} Alignment Results:")
        print(f"Translation: [{transform_params[0]:.4f}, {transform_params[1]:.4f}, {transform_params[2]:.4f}]")
        print(f"Rotation: [{transform_params[3]:.4f}, {transform_params[4]:.4f}, {transform_params[5]:.4f}]")
        print(f"Scale: {transform_params[6]:.4f}")
        print(f"Chamfer Distance: {final_distance:.6f}")
        
        # Add to results dictionary
        results["methods"][method_name] = {
            "file": method_path,
            "transformation": {
                "translation": {
                    "x": float(transform_params[0]),
                    "y": float(transform_params[1]),
                    "z": float(transform_params[2])
                },
                "rotation": {
                    "x": float(transform_params[3]),
                    "y": float(transform_params[4]),
                    "z": float(transform_params[5])
                },
                "scale": float(transform_params[6])
            },
            "chamfer_distance": float(final_distance)
        }
        
        # Save aligned mesh if requested
        if save_aligned:
            aligned_path = os.path.join(output_dir, f"{method_name}_aligned.obj")
            aligned_mesh.export(aligned_path)
            print(f"Aligned {method_name} mesh saved to {aligned_path}")
    
    # Save results to JSON file
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_path}")
    
    # Save visualization if we have processed methods
    if processed_method_names and source_point_cloud is not None:
        viz_path = os.path.join(output_dir, "mesh_alignment_comparison.png")
        save_mesh_image(source_point_cloud, target_point_clouds, aligned_point_clouds, viz_path, processed_method_names)
    
    return results

def process_all_meshes(gt_dir, method_dirs, output_base_dir, save_aligned=True, force_recompute=False):
    """
    Process all meshes found in the ground truth directory and compare with corresponding
    meshes in method directories.
    
    Args:
        gt_dir: Directory containing ground truth meshes (.obj files)
        method_dirs: List of directories containing method-generated meshes (.glb files)
        output_base_dir: Base directory to save results
        save_aligned: Whether to save aligned meshes
        force_recompute: Whether to force recomputation even if results exist
    """
    # Get all ground truth mesh files
    gt_files = glob.glob(os.path.join(gt_dir, "*.obj"))
    
    if not gt_files:
        print(f"No .obj files found in ground truth directory: {gt_dir}")
        return
    
    # Dictionary to store all results for summary
    all_results = {}
    
    # Process each ground truth mesh
    for gt_file in gt_files:
        mesh_name = os.path.splitext(os.path.basename(gt_file))[0]
        print(f"\n{'='*50}")
        print(f"Processing mesh: {mesh_name}")
        print(f"{'='*50}")
        
        # Find corresponding method meshes
        method_files = []
        method_names = []
        
        for method_dir in method_dirs:
            method_name = os.path.basename(method_dir)
            method_file = os.path.join(method_dir, f"{mesh_name}.glb")
            
            if os.path.exists(method_file):
                method_files.append(method_file)
                method_names.append(method_name)
                print(f"Found {method_name} mesh: {method_file}")
            else:
                print(f"Warning: No corresponding mesh found for {method_name} at {method_file}")
        
        if not method_files:
            print(f"No method meshes found for {mesh_name}, skipping...")
            continue
        
        # Create output directory for this mesh
        output_dir = os.path.join(output_base_dir, mesh_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Compare meshes
        results = compare_meshes(
            gt_file,
            method_files,
            method_names,
            output_dir,
            save_aligned=save_aligned,
            force_recompute=force_recompute
        )
        
        # Store results for summary
        all_results[mesh_name] = results
    
    # Generate summary
    generate_summary(all_results, output_base_dir)
    
    return all_results

def generate_summary(all_results, output_dir):
    """
    Generate a summary of all results in various formats including LaTeX table.
    
    Args:
        all_results: Dictionary of results for all meshes
        output_dir: Directory to save summary
    """
    # Extract all method names and mesh names
    method_names = set()
    mesh_names = list(all_results.keys())
    
    for mesh_results in all_results.values():
        if "methods" in mesh_results:
            for method_name in mesh_results["methods"].keys():
                method_names.add(method_name)
    
    method_names = sorted(list(method_names))
    
    # Create a CSV summary
    csv_path = os.path.join(output_dir, "chamfer_summary.csv")
    with open(csv_path, 'w') as f:
        # Write header
        f.write("Mesh," + ",".join(method_names) + "\n")
        
        # Write data
        for mesh_name in mesh_names:
            row = [mesh_name]
            for method_name in method_names:
                if (mesh_name in all_results and 
                    "methods" in all_results[mesh_name] and 
                    method_name in all_results[mesh_name]["methods"]):
                    distance = all_results[mesh_name]["methods"][method_name]["chamfer_distance"]
                    row.append(f"{distance:.6f}")
                else:
                    row.append("N/A")
            f.write(",".join(row) + "\n")
    
    print(f"CSV summary saved to {csv_path}")
    
    # Create a LaTeX table
    latex_path = os.path.join(output_dir, "chamfer_summary.tex")
    with open(latex_path, 'w') as f:
        # Write LaTeX table header
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Chamfer Distance Comparison}\n")
        f.write("\\label{tab:chamfer_comparison}\n")
        f.write("\\begin{tabular}{l" + "r" * len(method_names) + "}\n")
        f.write("\\toprule\n")
        f.write("Mesh & " + " & ".join(method_names) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Write data rows
        for mesh_name in mesh_names:
            mesh_name_latex = '\_'.join(mesh_name.split('_'))
            row = [mesh_name_latex]
            for method_name in method_names:
                if (mesh_name in all_results and 
                    "methods" in all_results[mesh_name] and 
                    method_name in all_results[mesh_name]["methods"]):
                    distance = all_results[mesh_name]["methods"][method_name]["chamfer_distance"]
                    # Format with scientific notation for LaTeX
                    row.append(f"{distance:.4f}")
                else:
                    row.append("--")
            f.write(" & ".join(row) + " \\\\\n")
        
        # Write LaTeX table footer
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_path}")
    
    # Also create a summary with average chamfer distance per method
    method_averages = {method: [] for method in method_names}
    
    for mesh_name in mesh_names:
        if mesh_name in all_results and "methods" in all_results[mesh_name]:
            for method_name in method_names:
                if method_name in all_results[mesh_name]["methods"]:
                    distance = all_results[mesh_name]["methods"][method_name]["chamfer_distance"]
                    method_averages[method_name].append(distance)
    
    # Calculate averages
    summary = {"method_averages": {}}
    for method_name, distances in method_averages.items():
        if distances:
            avg = sum(distances) / len(distances)
            summary["method_averages"][method_name] = avg
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, "chamfer_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Summary JSON saved to {summary_path}")
    
    # Print average results to console
    print("\nAverage Chamfer Distance per Method:")
    for method_name, avg in summary["method_averages"].items():
        print(f"{method_name}: {avg:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Compare multiple mesh methods to ground truth using Chamfer distance")
    parser.add_argument('--gt_dir', required=True, help="Directory containing ground truth meshes (.obj files)")
    parser.add_argument('--method_dirs', nargs='+', required=True, help="Directories containing method-generated meshes (.glb files)")
    parser.add_argument('--output_dir', default='results', help="Directory to save results")
    parser.add_argument('--save_aligned', action='store_true', help="Save aligned meshes")
    parser.add_argument('--force', action='store_true', help="Force recomputation of existing results")
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.isdir(args.gt_dir):
        print(f"Error: Ground truth directory not found: {args.gt_dir}")
        return
    
    valid_method_dirs = []
    for method_dir in args.method_dirs:
        if os.path.isdir(method_dir):
            valid_method_dirs.append(method_dir)
        else:
            print(f"Warning: Method directory not found: {method_dir}")
    
    if not valid_method_dirs:
        print("Error: No valid method directories provided")
        return
    
    # Process all meshes
    process_all_meshes(
        args.gt_dir,
        valid_method_dirs,
        args.output_dir,
        save_aligned=args.save_aligned,
        force_recompute=args.force
    )
    
    print("\nAll meshes processed successfully!")

if __name__ == "__main__":
    main()