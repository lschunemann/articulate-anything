import trimesh
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree
from pathlib import Path
import glob
import json
import argparse
import open3d as o3d


# --- Utilities ---
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

def sample_points(mesh, sample_count=20000):
    """Sample points uniformly from the surface of a mesh."""
    if hasattr(mesh, 'sample'):
        return mesh.sample(sample_count)
    else:
        return mesh.vertices

def scale_to_bounding_sphere(points):
    """Scale points such that the furthest distance from the centroid becomes 1."""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_distance = np.max(np.linalg.norm(points_centered, axis=1))
    scaled_points = points_centered / max_distance
    return scaled_points, max_distance, centroid

def center_mesh_to_source(source_points, target_points):
    """
    Translate the target mesh such that its centroid aligns with the source mesh centroid.

    Args:
        source_points (np.ndarray): Nx3 array of source point cloud coordinates.
        target_points (np.ndarray): Nx3 array of target point cloud coordinates.

    Returns:
        centered_target_points (np.ndarray): Nx3 array of translated target coordinates.
        translation_vector (np.ndarray): 1x3 array representing the translation applied to the target mesh.
    """
    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Compute translation vector
    translation_vector = source_centroid - target_centroid

    # Translate target points
    centered_target_points = target_points + translation_vector

    return centered_target_points, translation_vector

def generate_rotation_matrix(axis, angle):
    """Generate a rotation matrix for a given axis ('x', 'y', 'z') and angle in radians."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")

def transform_points(points, rotation_matrix):
    """Apply a rotation matrix to points."""
    return points @ rotation_matrix

def chamfer_distance(points1, points2):
    """Compute the Chamfer distance between two point clouds."""
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    distances1, _ = tree1.query(points2)  # From points2 to points1
    distances2, _ = tree2.query(points1)  # From points1 to points2
    return np.mean(distances1) + np.mean(distances2)


# --- Brute Force Alignment with Scaling ---
def brute_force_rotation_align_with_scaling(source_points, target_points):
    """
    Align source and target point clouds using scaling and brute-force rotation.
    """
    # Scale target points to match the source's bounding radius
    # scaled_target_points, scale_factor = scale_target_to_source(source_points, target_points)
    # print(f"Scale Factor Applied: {scale_factor:.6f}")
    scaled_target_points = target_points.copy()
    
    # Test all combinations of 90-degree rotations
    axes = ['x', 'y', 'z']
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    best_transform = None
    best_distance = float('inf')
    best_rotation_combination = None
    
    for rx in angles:
        for ry in angles:
            for rz in angles:
                # Generate rotation matrices
                R_x = generate_rotation_matrix('x', rx)
                R_y = generate_rotation_matrix('y', ry)
                R_z = generate_rotation_matrix('z', rz)
                rotation_matrix = R_z @ R_y @ R_x
                
                # Transform target points
                transformed_points = transform_points(scaled_target_points, rotation_matrix)
                distance = chamfer_distance(source_points, transformed_points)
                
                # Track the best alignment
                if distance < best_distance:
                    best_distance = distance
                    best_transform = rotation_matrix
                    best_rotation_combination = (rx, ry, rz)
    
    print(f"Best rotation: X={np.degrees(best_rotation_combination[0])}°, Y={np.degrees(best_rotation_combination[1])}°, Z={np.degrees(best_rotation_combination[2])}°")
    print(f"Minimum Chamfer Distance: {best_distance:.6f}")
    
    # Apply the best transform
    aligned_points = transform_points(target_points, best_transform)
    return aligned_points, best_transform, best_distance

def numpy_to_open3d_point_cloud(points):
    """
    Convert a NumPy array to an Open3D PointCloud object.

    Args:
        points (np.ndarray): Nx3 array of coordinates.
    Returns:
        point_cloud (open3d.geometry.PointCloud): Open3D PointCloud object.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))  # Ensure dtype is float64
    return point_cloud

# --- Visualization ---
def visualize_points(source_points, target_points, aligned_points, output_path=None):
    """Visualize original, target, and aligned point clouds."""
    fig = plt.figure(figsize=(15, 5))
    
    # Source points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='blue', s=1, alpha=0.5, label='GT Points')
    ax1.set_title("Source Points")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_box_aspect([1, 1, 1])
    
    # Target points
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='green', s=1, alpha=0.5, label='Scaled Target Points')
    ax2.set_title("Target Points")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_box_aspect([1, 1, 1])
    
    # Aligned points
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2], c='green', s=1, alpha=0.5, label='Aligned Target Points')
    ax3.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='blue', s=1, alpha=0.5, label='GT Points')
    ax3.set_title("Aligned Points")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_box_aspect([1, 1, 1])
    ax3.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

def scale_target_to_source(source_points, target_points):
    """
    Scale the target point cloud to match the bounding diameter of the source point cloud.

    Args:
        source_points (np.ndarray): Nx3 array of source point cloud coordinates.
        target_points (np.ndarray): Nx3 array of target point cloud coordinates.
    Returns:
        scaled_target_points (np.ndarray): Nx3 array of scaled target coordinates.
        scale_factor (float): Factor applied to scale target points.
    """
    # Compute bounding diameter for each point cloud
    source_diameter = np.max(np.linalg.norm(source_points[:, None, :] - source_points[None, :, :], axis=-1))
    target_diameter = np.max(np.linalg.norm(target_points[:, None, :] - target_points[None, :, :], axis=-1))
    scale_factor = source_diameter / target_diameter
    scaled_target_points = target_points * scale_factor
    return scaled_target_points, scale_factor

def simple_icp(source_cloud, target_cloud, max_correspondence_dist=50.0, max_iterations=100):
    """
    Perform point-to-point ICP alignment using Open3D's prebuilt function.

    Args:
        source_cloud (open3d.geometry.PointCloud): Source (moving) point cloud.
        target_cloud (open3d.geometry.PointCloud): Target (reference) point cloud.
        max_correspondence_dist (float): Maximum distance threshold for identifying correspondences.
        max_iterations (int): Maximum number of ICP iterations.
    Returns:
        aligned_cloud (open3d.geometry.PointCloud): Aligned source cloud.
        transformation_matrix (np.ndarray): 4x4 matrix (rotation + translation).
    """
    # Ensure normals are computed for both clouds
    source_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    target_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )

    # Perform ICP registration with point-to-plane method
    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_correspondence_dist,
        np.identity(4),  # Initial guess (identity matrix)
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    # Apply transformation matrix to align source cloud
    transformation_matrix = result.transformation
    aligned_cloud = o3d.geometry.PointCloud()
    aligned_cloud.points = o3d.utility.Vector3dVector(np.asarray(source_cloud.points))
    aligned_cloud.transform(transformation_matrix)
    return aligned_cloud, transformation_matrix

# --- Process All Meshes ---
def process_all_meshes(gt_dir, method_dirs, output_base_dir):
    """
    Process all meshes in the ground truth directory and compare with corresponding meshes across methods.
    """
    gt_files = glob.glob(os.path.join(gt_dir, "*.obj"))
    if not gt_files:
        print(f"No .obj files found in ground truth directory: {gt_dir}")
        return
    
    all_results = {}
    for gt_path in gt_files:
        gt_name = Path(gt_path).stem
        results = {"ground_truth": gt_name}
        
        # Create output directory for this mesh
        mesh_output_dir = os.path.join(output_base_dir, gt_name)
        os.makedirs(mesh_output_dir, exist_ok=True)
        
        # Process corresponding meshes in each method directory
        method_results = {}
        for method_dir in method_dirs:
            method_name = Path(method_dir).name
            method_mesh_path = os.path.join(method_dir, f"{gt_name}.glb")
            
            if os.path.exists(method_mesh_path):
                # Load meshes
                print(f"Processing {method_name} corresponding to {gt_name}...")
                gt_mesh = load_mesh(gt_path)
                method_mesh = load_mesh(method_mesh_path)
                
                # Sample points from meshes
                gt_points = sample_points(gt_mesh)
                method_points = sample_points(method_mesh)
                
                # Scale target points to match source points
                scaled_method_points, scale_factor = scale_target_to_source(gt_points, method_points)
                print(f"Scale Factor Applied: {scale_factor:.6f}")

                # Center target cloud to align its centroid with source cloud
                scaled_method_points, translation_vector = center_mesh_to_source(gt_points, scaled_method_points)
                print(f"Translation Vector Applied: {translation_vector}")                
                
                # if method_name.lower() == "trellis":
                #     # Use brute-force alignment as initial guess for trellis method
                #     scaled_method_points, best_transform, chamfer_dist = brute_force_rotation_align_with_scaling(
                #         gt_points, scaled_method_points
                #     )

                # Convert point clouds for ICP
                source_cloud = numpy_to_open3d_point_cloud(gt_points)
                target_cloud = numpy_to_open3d_point_cloud(scaled_method_points)
                
                # Perform ICP alignment
                aligned_cloud, transform_matrix = simple_icp(target_cloud, source_cloud)
                aligned_points = np.asarray(aligned_cloud.points)
                
                # Compute Chamfer distance
                chamfer_dist = chamfer_distance(gt_points, aligned_points)
                
                method_results[method_name] = {"chamfer_distance": chamfer_dist, "rotation_matrix": transform_matrix.tolist() if method_name == "trellis" else transform_matrix.tolist()}
                
                # Save visualization
                visualize_path = os.path.join(mesh_output_dir, f"{method_name}_visualization.png")
                visualize_points(gt_points, scaled_method_points, aligned_points, output_path=visualize_path)
        
        results["methods"] = method_results
        all_results[gt_name] = results
    
    # Save results to JSON
    results_path = os.path.join(output_base_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    return all_results


# --- Generate Summary ---
def generate_summary(all_results, output_dir):
    """
    Generate a summary of all results, including a LaTeX table sorted alphabetically by mesh name.
    
    Args:
        all_results: Dictionary containing Chamfer distances and methods for all meshes.
        output_dir: Directory to save outputs (e.g., LaTeX table).
    """
    # Extract all method names
    method_names = set()
    mesh_names = list(all_results.keys())
    
    for mesh_results in all_results.values():
        if "methods" in mesh_results:
            method_names.update(mesh_results["methods"].keys())
    
    # Sort method names and mesh names alphabetically
    method_names = sorted(method_names)
    mesh_names = sorted(mesh_names)
    
    # Create a LaTeX table
    latex_path = os.path.join(output_dir, "chamfer_summary.tex")
    with open(latex_path, 'w') as f:
        # Begin the LaTeX table
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Chamfer Distance Comparison Across Methods}\n")
        f.write("\\label{tab:chamfer_comparison}\n")
        f.write("\\begin{tabular}{l" + "r" * len(method_names) + "}\n")
        f.write("\\toprule\n")
        
        # Write column headers
        f.write("Mesh & " + " & ".join(method_names) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Write rows for each mesh
        for mesh_name in mesh_names:
            row = [mesh_name]
            for method_name in method_names:
                if mesh_name in all_results and method_name in all_results[mesh_name].get("methods", {}):
                    chamfer_dist = all_results[mesh_name]["methods"][method_name]["chamfer_distance"]
                    row.append(f"{chamfer_dist:.6f}")
                else:
                    row.append("--")  # Missing entry
            f.write(" & ".join(row) + " \\\\\n")
        
        # Finish the LaTeX table
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_path}")


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Align meshes using brute force rotation and scaling.")
    parser.add_argument('--gt', required=True, help="Path to the ground truth mesh.")
    parser.add_argument('--methods', nargs='+', required=True, help="Paths to the method-generated meshes.")
    parser.add_argument('--output_dir', required=True, help="Output directory for results.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compare meshes
    results = process_all_meshes(args.gt, args.methods, args.output_dir)

    generate_summary(results, args.output_dir)


if __name__ == "__main__":
    main()