import numpy as np
import trimesh
import os
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import argparse
import json

def load_mesh(file_path):
    """Load a mesh from .obj or .glb file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
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
        elif len(meshes) == 1:
            mesh = meshes[0]
        else:
            raise ValueError(f"No valid meshes found in {file_path}")
    else:
        # If it's already a Trimesh object
        mesh = loaded
    
    return mesh

def save_mesh_image(source_mesh, target_mesh, aligned_mesh, output_path):
    """Create and save a visualization of the original and aligned meshes."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot source mesh
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Source Mesh')
    source_points = source_mesh.vertices
    ax1.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
                c='blue', s=1, alpha=0.5)
    
    # Plot target mesh
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Target Mesh')
    target_points = target_mesh.vertices
    ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                c='red', s=1, alpha=0.5)
    
    # Plot aligned source and target mesh
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Aligned Meshes')
    aligned_points = aligned_mesh.vertices
    ax3.scatter(aligned_points[:, 0], aligned_points[:, 1], aligned_points[:, 2], 
                c='blue', s=1, alpha=0.5, label='Aligned Source')
    ax3.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                c='red', s=1, alpha=0.5, label='Target')
    ax3.legend()
    
    # Set equal aspect ratio for all plots
    for ax in [ax1, ax2, ax3]:
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

def align_meshes(source_mesh, target_mesh, sample_points=10000):
    """Align source mesh to target mesh using optimization."""
    # Sample points from meshes for faster computation
    #source_points = source_mesh.sample(sample_points)
    source_points = source_mesh.vertices
    #target_points = target_mesh.sample(sample_points)
    target_points = target_mesh.vertices
    
    # Initial parameters: [tx, ty, tz, rx, ry, rz, scale]
    initial_params = [0, 0, 0, 0, 0, 0, 1.0]
    
    # Bounds for parameters
    bounds = [
        (-100, 100),    # tx
        (-100, 100),    # ty
        (-100, 100),    # tz
        (-np.pi, np.pi),  # rx
        (-np.pi, np.pi),  # ry
        (-np.pi, np.pi),  # rz
        (0.01, 100)     # scale
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
    aligned_mesh = source_mesh.copy()
    
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

    # TODO: for now manually fit meshes
    # 1. Apply 180-degree rotation around x-axis (π radians)
    #rx = np.pi  # 180 degrees in radians
    #rotation_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
    #matrix = np.dot(matrix, rotation_x)
    
    # 2. Apply 90-degree rotation around z-axis (π/2 radians)
    #rz = np.pi/2  # 90 degrees in radians
    #rotation_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
    #matrix = np.dot(matrix, rotation_z)
    
    # Apply the transformation
    aligned_mesh.apply_transform(matrix)
    
    return aligned_mesh, optimal_params, result.fun

def main(gt_file, generated_file):
    # Load meshes
    print("Loading meshes...")
    source_mesh = load_mesh(gt_file)
    target_mesh = load_mesh(generated_file)
    
    # Normalize meshes to have similar scale
    source_mesh.vertices -= source_mesh.center_mass
    target_mesh.vertices -= target_mesh.center_mass
    
    # Align meshes
    print("Aligning meshes...")
    aligned_mesh, transform_params, min_distance = align_meshes(source_mesh, target_mesh)
    
    # Compute final Chamfer distance
    print("Computing final Chamfer distance...")
    final_distance = chamfer_distance(aligned_mesh.vertices, target_mesh.vertices)
    
    # Output results
    print("\nAlignment Results:")
    print(f"Translation: [{transform_params[0]:.4f}, {transform_params[1]:.4f}, {transform_params[2]:.4f}]")
    print(f"Rotation: [{transform_params[3]:.4f}, {transform_params[4]:.4f}, {transform_params[5]:.4f}]")
    print(f"Scale: {transform_params[6]:.4f}")
    print(f"Chamfer Distance: {final_distance:.6f}")

    # Create results dictionary
    results = {
        "source_file": f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.mesh}/{args.gt}_RLBench.obj",
        "target_file": f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.mesh}/{args.mesh}.glb",
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
    
    # Save results to JSON file
    json_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/results/{args.mesh}/"

    if not os.path.exists(json_path):
        os.makedirs(json_path)

    json_path = os.path.join(json_path, "chamfer.json")

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_path}")
    
    # Optionally save the aligned mesh
    if args.save.lower() == 'y':
        output_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/results/{args.mesh}/{args.mesh}_aligned.obj"
        aligned_mesh.export(output_path)
        print(f"Aligned mesh saved to {output_path}")

    # Save visualization
    viz_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/results/{args.mesh}/mesh_alignment.png"
    save_mesh_image(source_mesh, target_mesh, aligned_mesh, viz_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', default='fridge')
    parser.add_argument('mesh', default='fridge_rodin')
    parser.add_argument('save', default='y')
    args = parser.parse_args()
    gt_mesh = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.mesh}/{args.gt}_RLBench.obj"
    generated_mesh = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.mesh}/{args.mesh}.glb"
    main(gt_mesh, generated_mesh)