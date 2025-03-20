import numpy as np
import trimesh
import os
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import argparse
import json
from pathlib import Path
import glob

def load_mesh(file_path, debug=True):
    """Load a mesh from .obj or .glb file with improved OBJ handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if debug:
        print(f"Loading mesh from {file_path}...")
    
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
    else:
        # Handle GLB and other formats
        try:
            loaded = trimesh.load(file_path)
            
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
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")
    
    # Verify the mesh has vertices
    if len(mesh.vertices) == 0:
        raise ValueError(f"Loaded mesh from {file_path} has no vertices")
        
    if debug:
        print(f"Successfully loaded mesh from {file_path} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        print(f"Mesh bounds: {mesh.bounds}")
    
    return mesh

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

def align_meshes(source_mesh, target_mesh, method_name):
    """
    Align source mesh to target mesh using optimization.
    
    Args:
        source_mesh: Ground truth mesh
        target_mesh: Method-generated mesh
        method_name: Name of the method (used to determine preprocessing)
    """
    # Use all vertices directly without sampling
    source_points = source_mesh.vertices
    target_points = target_mesh.vertices
    
    # Create copies for transformation
    source_mesh_transformed = source_mesh.copy()
    target_mesh_transformed = target_mesh.copy()
    
    # Apply method-specific preprocessing
    # Standard preprocessing for source (ground truth) mesh
    matrix = np.eye(4)
    rz = np.pi  # 180 degrees in radians
    rotation_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
    matrix = np.dot(matrix, rotation_z)
    source_mesh_transformed.apply_transform(matrix)
    
    # Method-specific preprocessing for target mesh
    if method_name.lower() == 'paris':
        # Paris-specific transformation
        matrix = np.eye(4)
        rx = np.pi  # 180 degrees in radians
        rotation_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
        matrix = np.dot(matrix, rotation_x)
        target_mesh_transformed.apply_transform(matrix)
    elif method_name.lower() == 'ours':
        # Ours-specific transformation
        matrix = np.eye(4)
        rx = np.pi  # 180 degrees in radians
        rotation_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
        matrix = np.dot(matrix, rotation_x)
        target_mesh_transformed.apply_transform(matrix)
    # Add more method-specific transformations as needed
    # elif method_name.lower() == 'another_method':
    #     # Another method's specific transformation
    
    # Update points after transformation
    source_points = source_mesh_transformed.vertices
    target_points = target_mesh_transformed.vertices
    
    # Initial parameters: [tx, ty, tz, rx, ry, rz, scale]
    initial_params = [0, 0, 0, 0, 0, 0, 1.0]
    
    # Bounds for parameters
    bounds = [
        (-100, 100),     # tx
        (-100, 100),     # ty
        (-100, 100),     # tz
        (-np.pi, np.pi), # rx
        (-np.pi, np.pi), # ry
        (-np.pi, np.pi), # rz
        (0.01, 100)      # scale
    ]
    
    # Run optimization
    result = minimize(
        objective_function,
        initial_params,
        args=(source_points, target_points),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    # Get the optimal parameters
    optimal_params = result.x
    
    # Create a transformed copy of the source mesh
    aligned_mesh = source_mesh_transformed.copy()
    
    # Apply the transformation to the mesh
    matrix = np.eye(4)
    
    # Apply scaling
    matrix[:3, :3] *= optimal_params[6]
    
    # Apply rotation
    rx, ry, rz = optimal_params[3:6]
    rotation_matrix = trimesh.transformations.euler_matrix(rx, ry, rz, 'rxyz')
    matrix = np.dot(matrix, rotation_matrix)
    
    # Apply translation
    matrix[:3, 3] = optimal_params[:3]
    
    # Apply the transformation
    aligned_mesh.apply_transform(matrix)
    
    # Calculate final chamfer distance
    final_distance = chamfer_distance(aligned_mesh.vertices, target_mesh_transformed.vertices)
    
    return aligned_mesh, optimal_params, final_distance, target_mesh_transformed

def compare_meshes(gt_path, method_paths, method_names, output_dir, save_aligned=True):
    """
    Compare multiple method meshes to the ground truth.
    
    Args:
        gt_path: Path to the ground truth mesh
        method_paths: List of paths to method-generated meshes
        method_names: List of method names corresponding to method_paths
        output_dir: Directory to save results
        save_aligned: Whether to save aligned meshes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth mesh
    print("Loading ground truth mesh...")
    gt_mesh = load_mesh(gt_path)
    
    # Prepare to store results
    results = {
        "ground_truth_file": gt_path,
        "methods": {}
    }
    
    aligned_meshes = []
    transformed_target_meshes = []
    
    # Process each method
    for method_path, method_name in zip(method_paths, method_names):
        print(f"\nProcessing {method_name} mesh...")
        
        # Load method mesh
        method_mesh = load_mesh(method_path)
        
        # Align meshes with method-specific preprocessing
        print(f"Aligning {method_name} mesh with ground truth...")
        aligned_mesh, transform_params, final_distance, transformed_target = align_meshes(
            gt_mesh, method_mesh, method_name
        )
        
        # Store aligned mesh and transformed target for visualization
        aligned_meshes.append(aligned_mesh)
        transformed_target_meshes.append(transformed_target)
        
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
    json_path = os.path.join(output_dir, "chamfer_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_path}")
    
    # Save visualization
    viz_path = os.path.join(output_dir, "mesh_alignment_comparison.png")
    save_mesh_image(gt_mesh, transformed_target_meshes, aligned_meshes, viz_path, method_names)
    
    return results

def process_all_meshes(gt_dir, method_dirs, output_base_dir, save_aligned=True):
    """
    Process all meshes found in the ground truth directory and compare with corresponding
    meshes in method directories.
    
    Args:
        gt_dir: Directory containing ground truth meshes (.obj files)
        method_dirs: List of directories containing method-generated meshes (.glb files)
        output_base_dir: Base directory to save results
        save_aligned: Whether to save aligned meshes
    """
    # Get all ground truth mesh files
    gt_files = glob.glob(os.path.join(gt_dir, "*.obj"))
    
    if not gt_files:
        print(f"No .obj files found in ground truth directory: {gt_dir}")
        return
    
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
        compare_meshes(
            gt_file,
            method_files,
            method_names,
            output_dir,
            save_aligned=save_aligned
        )

def main():
    parser = argparse.ArgumentParser(description="Compare multiple mesh methods to ground truth using Chamfer distance")
    parser.add_argument('--gt_dir', required=True, help="Directory containing ground truth meshes (.obj files)")
    parser.add_argument('--method_dirs', nargs='+', required=True, help="Directories containing method-generated meshes (.glb files)")
    parser.add_argument('--output_dir', default='results', help="Directory to save results")
    parser.add_argument('--save_aligned', action='store_true', help="Save aligned meshes")
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
        save_aligned=args.save_aligned
    )
    
    print("\nAll meshes processed successfully!")

if __name__ == "__main__":
    main()