import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
import trimesh
from scipy.spatial import cKDTree
import json
import pycocotools.mask as mask_util
import argparse
import open3d as o3d
import copy

# Function that worked correctly for camera visualization
def visualize_camera_orientations(camera_files, axis_length=1.0):
    """
    Visualize camera orientations using the explicit orientation vectors.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different cameras
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    
    for i, camera_file in enumerate(camera_files):
        # Load camera parameters with orientation data
        data = np.load(camera_file, allow_pickle=True)
        t = data['t']
        R = data['R']
        
        print(f"Camera {i} position: {t}")
        
        # Plot camera position
        ax.scatter(t[0], t[1], t[2], c=colors[i % len(colors)], marker='^', s=100, label=f'Camera {i}')
        
        # In Blender, the camera looks along the -Z axis with Y up
        # Extract camera axes from the rotation matrix
        # R is a world-to-camera rotation matrix
        # To get camera axes in world space, we transpose R
        R_cam_to_world = R.T
        
        # Plot camera axes with clear labels
        # X axis (right) - red
        x_axis = R_cam_to_world[:, 0] * axis_length
        ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], 
                 color='r', length=axis_length, arrow_length_ratio=0.2)
        ax.text(t[0] + x_axis[0], t[1] + x_axis[1], t[2] + x_axis[2], "X", color='r')
        
        # Y axis (up in camera space) - green
        y_axis = R_cam_to_world[:, 1] * axis_length
        ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], 
                 color='g', length=axis_length, arrow_length_ratio=0.2)
        ax.text(t[0] + y_axis[0], t[1] + y_axis[1], t[2] + y_axis[2], "Y", color='g')
        
        # Z axis (viewing direction) - blue
        z_axis = -R_cam_to_world[:, 2] * axis_length  # Negate because camera looks along -Z
        ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], 
                 color='b', length=axis_length, arrow_length_ratio=0.2)
        ax.text(t[0] + z_axis[0], t[1] + z_axis[1], t[2] + z_axis[2], "Z", color='b')
        
        # Draw a line from camera to origin
        ax.plot([t[0], 0], [t[1], 0], [t[2], 0], c=colors[i % len(colors)], linestyle='--', alpha=0.5)
    
    # Plot origin
    ax.scatter(0, 0, 0, c='k', marker='*', s=200, label='Origin')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Coordinate Axes')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) * 0.5
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) * 0.5
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# The corrected back-projection function that worked correctly
def backproject_depth_blender_fixed(depth_map, K, R, t, max_depth=100.0):
    """
    Correctly backproject depth maps from Blender with proper coordinate transformations.
    
    Args:
        depth_map: Depth map from Blender
        K: Intrinsic camera matrix
        R: Rotation matrix (world to camera)
        t: Camera position in world coordinates
        max_depth: Maximum depth threshold
    """
    # Handle 3-channel depth maps
    if len(depth_map.shape) == 3:
        # If it's a 3-channel depth map, take the first channel
        depth_map = depth_map[:,:,0]
    
    # Get image dimensions
    height, width = depth_map.shape
    
    # Create pixel coordinates grid
    y, x = np.indices((height, width))
    
    # Flatten all arrays
    x = x.flatten()
    y = y.flatten()
    z = depth_map.flatten()
    
    # Filter out invalid, zero, or extreme depth values
    valid = (z > 0) & (z < max_depth) & np.isfinite(z)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    
    print(f"Valid depth points: {np.sum(valid)} out of {len(valid)}")
    
    if np.sum(valid) == 0:
        print("WARNING: No valid depth points found!")
        return np.array([])
    
    # Convert pixel coordinates to camera coordinates
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Calculate 3D coordinates in the camera space
    # In Blender, the camera looks along -Z, with Y up
    # So we map: x_pixel -> X, y_pixel -> Y, depth -> -Z
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    z_cam = -z  # Negate Z because Blender camera looks down -Z axis
    
    # Create points in camera space
    points_cam = np.vstack((x_cam, y_cam, z_cam)).T
    
    # Transform from camera space to world space
    # In Blender, R is world-to-camera rotation
    # We need its transpose for camera-to-world
    R_cam_to_world = R.T
    
    # Transform points from camera space to world space
    points_world = np.zeros_like(points_cam)
    for i, pt in enumerate(points_cam):
        points_world[i] = R_cam_to_world.dot(pt) + t
    
    return points_world

def backproject_masked_depth(depth_map, mask, K, R, t, max_depth=100.0, label=None, flip_z=True):
    """
    Backproject depth values only for masked regions using the same approach as the working function.
    
    Args:
        depth_map: Depth map from Blender
        mask: Binary mask of regions to include
        K: Intrinsic camera matrix
        R: Rotation matrix (world to camera)
        t: Camera position in world coordinates
        max_depth: Maximum depth threshold
        label: Optional label for the points
        flip_z: Whether to flip the Z-coordinate (usually not needed if using the correct transformation)
    """
    # Handle 3-channel depth maps
    if len(depth_map.shape) == 3:
        depth_map = depth_map[:,:,0]
    
    # Ensure mask has same shape as depth map
    if mask.shape != depth_map.shape:
        if len(mask.shape) == 3:
            mask = np.any(mask > 0, axis=2)
        else:
            mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
    
    # Find coordinates where mask is true
    y_coords, x_coords = np.where(mask > 0)
    
    if len(y_coords) == 0:
        print(f"No points in mask for {label if label else 'unknown'}")
        return np.array([])
    
    # Get depth values at these coordinates
    depth_values = depth_map[y_coords, x_coords]
    
    # Print statistics about depth values
    print(f"Depth values in mask for {label}:")
    print(f"  Min: {depth_values.min():.6f}")
    print(f"  Max: {depth_values.max():.6f}")
    print(f"  Mean: {depth_values.mean():.6f}")
    
    # Filter out invalid depth values
    valid = (depth_values > 0) & (depth_values < max_depth) & np.isfinite(depth_values)
    x_coords = x_coords[valid]
    y_coords = y_coords[valid]
    depth_values = depth_values[valid]
    
    print(f"  After filtering: {len(depth_values)} valid points (range: {depth_values.min():.6f} to {depth_values.max():.6f})")
    
    if len(depth_values) == 0:
        print(f"No valid depth values for {label if label else 'unknown'}")
        return np.array([])
    
    # Convert pixel coordinates to camera coordinates
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Calculate 3D coordinates in camera space
    # In Blender, the camera looks along -Z, with Y up
    x_cam = (x_coords - cx) * depth_values / fx
    y_cam = -(y_coords - cy) * depth_values / fy
    z_cam = -depth_values  # Negate Z because Blender camera looks down -Z axis
    
    # Create points in camera space
    points_cam = np.vstack((x_cam, y_cam, z_cam)).T
    
    # Transform from camera space to world space
    # In Blender, R is world-to-camera rotation
    # We need its transpose for camera-to-world
    R_cam_to_world = R.T
    
    # Transform points from camera space to world space
    points_world = np.zeros_like(points_cam)
    for i, pt in enumerate(points_cam):
        points_world[i] = R_cam_to_world.dot(pt) + t

        
    return points_world


# Function to visualize point clouds with camera positions
def visualize_point_cloud_with_cameras(points, labels, camera_params, title, output_path=None):
    """
    Visualize 3D point cloud with camera positions.
    
    Args:
        points: Nx3 array of 3D points
        labels: Array of string labels for each point
        camera_params: List of camera parameter dictionaries with 'R' and 't' keys
        title: Title for the plot
        output_path: Optional path to save the visualization
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap for the unique labels
    unique_labels = np.unique(labels)
    label_to_color = {}
    colors = plt.cm.tab10.colors
    
    for i, label in enumerate(unique_labels):
        label_to_color[label] = colors[i % len(colors)]
    
    # Plot the point cloud
    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            ax.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                c=[label_to_color[label]], 
                marker='.', s=1, alpha=0.5, label=label
            )
    
    # Add cameras
    camera_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    
    for i, params in enumerate(camera_params):
        # Extract camera parameters
        R = params['R']
        t = params['t']
        
        # Plot camera position
        ax.scatter(t[0], t[1], t[2], c=camera_colors[i % len(camera_colors)], 
                  marker='^', s=100, alpha=1, label=f'Camera {i}')
        
        # Plot camera axes
        R_cam_to_world = R.T
        
        # Set length for the camera axes
        axis_length = 0.5
        
        # X axis (right) - red
        x_axis = R_cam_to_world[:, 0] * axis_length
        ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], 
                 color='r', length=axis_length, arrow_length_ratio=0.2)
        
        # Y axis (up) - green
        y_axis = -R_cam_to_world[:, 1] * axis_length
        ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], 
                 color='g', length=axis_length, arrow_length_ratio=0.2)
        
        # Z axis (view direction) - blue
        z_axis = -R_cam_to_world[:, 2] * axis_length  # Negate for camera direction
        ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], 
                 color='b', length=axis_length, arrow_length_ratio=0.2)
        
        # Draw line to origin
        ax.plot([t[0], 0], [t[1], 0], [t[2], 0], 
                c=camera_colors[i % len(camera_colors)], linestyle='--', alpha=0.3)
    
    # Add origin
    ax.scatter(0, 0, 0, c='k', marker='*', s=200, label='Origin')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add legend with unique entries
    handles, labels_text = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_text, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax .get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) * 0.5
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) * 0.5
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    # plt.show()
    
    return fig, ax

# Main function to lift 2D masks to 3D
def lift_2d_masks_to_3d(rgb_images, results_paths, depth_maps, camera_params, OBJECT, output_dir, flip_z=True):
    """
    Lift 2D segmentation masks to 3D points using depth maps and camera parameters.
    Modified to preserve instance information by creating unique instance IDs.
    """
    all_3d_points = []
    all_labels = []
    all_instance_ids = []  # New array to track instance IDs
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a global instance counter for each label
    label_instance_counters = {}
    
    # Process each view
    for view_idx in range(len(rgb_images)):
        print(f"\nProcessing view {view_idx + 1}/{len(rgb_images)}")
        
        # Load masks for this view
        masks, label_names = load_masks_from_results(results_paths[view_idx], OBJECT)
        
        # Get camera parameters
        K = camera_params[view_idx]['K']
        R = camera_params[view_idx]['R']
        t = camera_params[view_idx]['t']
        
        # Get depth map
        depth_map = depth_maps[view_idx]
        
        # Print depth map stats
        if len(depth_map.shape) == 3:
            print(f"Depth map has shape {depth_map.shape} (3 channels), will use first channel")
        else:
            print(f"Depth map has shape {depth_map.shape}")
            
        valid_mask = depth_map > 0
        if np.any(valid_mask):
            print(f"Depth range: {np.min(depth_map[valid_mask])} to {np.max(depth_map[valid_mask])}")
        else:
            print("Warning: No valid depth values found")
        
        view_points = []
        view_labels = []
        view_instance_ids = []  # Track instance IDs for this view
        
        # Process each mask (each mask is a separate instance)
        for mask_idx, (mask, label_name) in enumerate(zip(masks, label_names)):
            # Create a unique instance ID for this mask
            if label_name not in label_instance_counters:
                label_instance_counters[label_name] = 0
            
            instance_id = f"{label_name}_{label_instance_counters[label_name]}"
            label_instance_counters[label_name] += 1
            
            # Backproject points for this mask
            points_3d = backproject_masked_depth(
                depth_map, mask, K, R, t, max_depth=100.0, label=label_name, flip_z=flip_z
            )
            
            if len(points_3d) == 0:
                print(f"No valid 3D points for {label_name} (instance {instance_id})")
                continue
                
            print(f"Generated {len(points_3d)} 3D points for {label_name} (instance {instance_id})")
            view_points.append(points_3d)
            view_labels.extend([label_name] * len(points_3d))
            view_instance_ids.extend([instance_id] * len(points_3d))  # Add instance IDs

        # Visualize points for this view if any were generated
        if view_points:
            view_points = np.vstack(view_points)
            vis_path = os.path.join(output_dir, f'view_{view_idx}_points.png')
            
            # Create camera parameters for visualization
            cam_params_dict = {'R': R, 't': t}
            
            # Also flip z-coordinate of camera position for visualization if we're flipping points
            if flip_z:
                t_vis = t.copy()
                t_vis[2] = -t_vis[2]
                cam_params_vis = {'R': R, 't': t_vis}
            else:
                cam_params_vis = cam_params_dict
                
            visualize_point_cloud_with_cameras(
                view_points, np.array(view_instance_ids),  # Use instance IDs instead of labels
                [cam_params_vis], 
                f'View {view_idx} Point Cloud', 
                vis_path
            )
            
            # Add to global collection
            all_3d_points.append(view_points)
            all_labels.extend(view_labels)
            all_instance_ids.extend(view_instance_ids)  # Add instance IDs
            
            print(f"View {view_idx} processed with {len(view_points)} points")
        else:
            print(f"No valid points found in view {view_idx}")

    # Process all views together
    if not all_3d_points:
        print("No valid 3D points found in any view")
        return np.zeros((0, 3)), np.array([]), np.array([])
        
    points_3d = np.vstack(all_3d_points)
    labels = np.array(all_labels)
    instance_ids = np.array(all_instance_ids)  # Convert to numpy array
    
    print(f"\nFinal point cloud has {len(points_3d)} points with {len(np.unique(labels))} unique labels")
    print(f"Final point cloud has {len(np.unique(instance_ids))} unique instances")
    
    # Visualize combined point cloud
    vis_path = os.path.join(output_dir, 'combined_point_cloud.png')
    
    # Prepare camera parameters for visualization
    all_camera_params = []
    for params in camera_params:
        if flip_z:
            t_vis = params['t'].copy()
            t_vis[2] = -t_vis[2]
            all_camera_params.append({'R': params['R'], 't': t_vis})
        else:
            all_camera_params.append({'R': params['R'], 't': params['t']})
            
    visualize_point_cloud_with_cameras(
        points_3d, instance_ids, all_camera_params,  # Use instance IDs for visualization
        'Combined Point Cloud', 
        vis_path
    )
    
    return points_3d, labels, instance_ids  # Return instance IDs as well

def load_masks_from_results(results_path, OBJECT):
    """Load masks from the JSON results file"""
    print(f"Loading masks from {results_path}")
    
    with open(os.path.join(f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{OBJECT}", results_path), 'r') as f:
        results = json.load(f)
    
    masks = []
    labels = []
    
    for annotation in results['annotations']:
        rle = annotation['segmentation']
        rle['counts'] = rle['counts'].encode('utf-8')
        mask = mask_util.decode(rle)
        
        masks.append(mask)
        labels.append(annotation['class_name'])
    
    print(f"Loaded {len(masks)} masks with labels: {labels}")
    return np.stack(masks), labels

def visualize_assignment_results(mesh, segmented_points, point_labels, vertex_labels, unique_labels, 
                           distances, output_dir, OBJECT, views=4):
    """
    Save images of point cloud and mesh assignments for headless servers.
    Modified to handle string labels.
    
    Args:
        mesh: Trimesh mesh object
        segmented_points: Nx3 array of 3D points
        point_labels: Array of labels for each point (can be strings)
        vertex_labels: Array of labels assigned to mesh vertices (can be strings)
        unique_labels: Array of unique label values (can be strings)
        distances: Array of distances from vertices to nearest points
        output_dir: Directory to save images
        OBJECT: Object name for image filenames
        views: Number of different viewpoints to generate (1-4)
    """
    print("\nGenerating visualization images for assignments...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Count assignments per label
    print("Vertex assignments per label:")
    for label in unique_labels:
        count = np.sum(vertex_labels == label)
        percentage = count / len(vertex_labels) * 100
        print(f"  Label {label}: {count} vertices ({percentage:.2f}%)")
    
    # Count unassigned vertices
    unassigned = np.sum(vertex_labels == "unassigned")
    print(f"  Unassigned: {unassigned} vertices ({unassigned/len(vertex_labels)*100:.2f}%)")
    
    # Distance distribution
    dist_percentiles = np.percentile(distances, [10, 25, 50, 75, 90, 95, 99])
    print(f"Distance distribution percentiles (10, 25, 50, 75, 90, 95, 99):")
    print(f"  {dist_percentiles}")
    
    # Create a colormap for labels
    num_labels = len(unique_labels)
    colormap = plt.cm.get_cmap('tab10', max(num_labels+1, 10))  # +1 for unassigned
    
    # Create a mapping from string labels to numeric indices for coloring
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    label_to_idx["unassigned"] = len(unique_labels)  # Add unassigned at the end
    
    # Create color dictionary
    label_colors = {label: colormap(label_to_idx[label] % 10)[:3] for label in unique_labels}
    label_colors["unassigned"] = (0.7, 0.7, 0.7)  # Gray for unassigned
    
    # Compute bounding box for consistent view
    all_points = np.vstack([segmented_points, mesh.vertices])
    min_bounds = np.min(all_points, axis=0)
    max_bounds = np.max(all_points, axis=0)
    center = (min_bounds + max_bounds) / 2
    max_range = np.max(max_bounds - min_bounds)
    
    # Create a visualization showing both point clouds and mesh vertices
    view_angles = [
        (30, 30),   # View 1
        (30, 120),  # View 2
        (30, 210),  # View 3
        (30, 300),  # View 4
    ]

    # Limit to requested number of views
    view_angles = view_angles[:views]

    # First, create visualizations for each view angle
    for view_idx, (elev, azim) in enumerate(view_angles):
        # Now create separate visualizations for each label
        for label in list(unique_labels) + ["unassigned"]:  # Include unassigned
            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot point cloud points for this label only
            mask = np.array(point_labels) == label
            if np.any(mask):
                points = segmented_points[mask]
                # Only plot a subset if there are too many points
                if len(points) > 1000:
                    indices = np.random.choice(len(points), 1000, replace=False)
                    points = points[indices]
                
                ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    color=label_colors[label],
                    marker='o',
                    s=25,
                    alpha=0.8,
                    label=f'Points (Label {label})'
                )
            
            # Plot a subset of mesh vertices for this label only
            sample_rate = max(1, len(mesh.vertices) // 5000)
            sampled_indices = np.arange(0, len(mesh.vertices), sample_rate)
            
            mesh_mask = np.array(vertex_labels)[sampled_indices] == label
            if np.any(mesh_mask):
                vertices = mesh.vertices[sampled_indices[mesh_mask]]
                ax.scatter(
                    vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    color=label_colors[label],
                    marker='x',
                    s=15,
                    alpha=0.6,
                    label=f'Mesh (Label {label})'
                )
            
            # Set consistent view limits
            ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
            ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
            ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])
            
            # Set view angle
            ax.view_init(elev=elev, azim=azim)
            
            # Add title
            ax.set_title(f'Label {label} - View {view_idx+1}')
            
            # Add legend
            ax.legend()
            
            # Save figure with label in the filename
            plt.savefig(os.path.join(output_dir, f'{OBJECT}_label_{label}_view_{view_idx+1}.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create mesh visualization with vertex colors for each label
    for label in unique_labels:
        # Create a colored mesh with this label highlighted
        colors = np.zeros((len(mesh.vertices), 4))  # RGBA
        
        # Set alpha based on label
        mask_label = np.array(vertex_labels) == label
        mask_other = (np.array(vertex_labels) != label) & (np.array(vertex_labels) != "unassigned")  # Other labels except unassigned
        mask_unassigned = np.array(vertex_labels) == "unassigned"
        
        # Highlighted label (fully opaque)
        colors[mask_label, :3] = label_colors[label]
        colors[mask_label, 3] = 1.0
        
        # Other labels (semi-transparent)
        for other_label in unique_labels:
            if other_label == label:
                continue
            mask = np.array(vertex_labels) == other_label
            colors[mask, :3] = label_colors[other_label]
        colors[mask_other, 3] = 0.2
        
        # Unassigned (less visible)
        colors[mask_unassigned, :3] = label_colors["unassigned"]
        colors[mask_unassigned, 3] = 0.1
        
        # Create visualization using trimesh
        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=colors
        )
        
        # Save as PNG
        scene = trimesh.Scene(colored_mesh)
        for view_idx, (elev, azim) in enumerate(view_angles):
            # Convert to radians
            rot_z = azim * np.pi / 180
            rot_x = elev * np.pi / 180
            
            # Apply rotation
            matrix = np.eye(4)
            matrix[:3, :3] = trimesh.transformations.euler_matrix(rot_x, 0, rot_z, 'sxyz')[:3, :3]
            scene.camera_transform = matrix
            
            try:
                png = scene.save_image(resolution=[1200, 1000], visible=True)
                # TODO: fix visualization
                # with open(os.path.join(output_dir, f'{OBJECT}_label_{label}_view_{view_idx+1}.png'), 'wb') as f:
                #     f.write(png)
            except Exception as e:
                print(f"Error saving mesh image: {e}")
    
    print(f"Saved visualization images to: {output_dir}")
    print("Generated images:")
    for label in unique_labels:
        for view_idx in range(len(view_angles)):
            print(f"  {OBJECT}_label_{label}_view_{view_idx+1}.png")
    print(f"  {OBJECT}_distance_histogram.png")
    print(f"  {OBJECT}_color_legend.png")
    for view_idx in range(len(view_angles)):
        print(f"  {OBJECT}_view_{view_idx+1}.png")


def segment_and_save_parts(mesh, segmented_points, point_labels, unique_labels, output_dir, OBJECT, flip_mesh_z=False):
    """
    Segment a mesh into parts based on point labels and save each part.
    Modified to handle string instance IDs.
    
    Args:
        mesh: Trimesh mesh object
        segmented_points: Nx3 array of 3D points
        point_labels: Array of labels for each point (can be strings)
        unique_labels: Array of unique label values
        output_dir: Output directory for part meshes
        OBJECT: Object name
    """
    print("\nSegmenting mesh into parts...")

    # Flip mesh if needed
    mesh_vertices = mesh.vertices
    if flip_mesh_z:
        mesh_vertices[:, 2] = -mesh_vertices[:, 2]
        mesh_vertices[:, 1] = -mesh_vertices[:, 1]
    mesh.vertices = mesh_vertices
    
    # Build KD-tree for nearest neighbor search
    kdtree = cKDTree(segmented_points)
    
    # Query nearest neighbors for each mesh vertex
    distances, indices = kdtree.query(mesh.vertices)
    
    # Set distance threshold
    max_distance = 0.05
    print(f"Using max distance threshold: {max_distance}")
    
    # Mask for vertices that are close enough to a point
    close_enough = distances < max_distance
    
    # Create a mapping from string labels to numeric IDs for internal use
    # This allows us to use string labels externally but numeric IDs internally
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    # Convert string point labels to numeric IDs
    numeric_point_labels = np.array([label_to_id[label] for label in point_labels])
    
    # Initialize vertex labels with -1 (unassigned)
    vertex_labels = np.full(len(mesh.vertices), -1)  # -1 for unassigned
    
    # Assign numeric IDs to vertices
    vertex_labels[close_enough] = numeric_point_labels[indices[close_enough]]

    print(f"vertex labels: {vertex_labels}")

    # For visualization, convert back to original labels
    vertex_label_strings = np.array([id_to_label.get(id, "unassigned") if id != -1 else "unassigned" 
                                    for id in vertex_labels])
    
    visualize_assignment_results(
            mesh=mesh,
            segmented_points=segmented_points,
            point_labels=point_labels,
            vertex_labels=vertex_label_strings,  # Use string labels for visualization
            unique_labels=unique_labels,
            distances=distances,
            output_dir=output_dir,
            OBJECT=OBJECT
        )
    
    print(f"Found {len(np.unique(vertex_labels))} unique parts")

    # Debug label distribution
    print("Label distribution in point cloud:")
    for label in unique_labels:
        count = np.sum(point_labels == label)
        print(f"  {label}: {count} points")
        
    print("Label distribution in mesh vertices after KNN:")
    for i, label in enumerate(unique_labels):
        count = np.sum(vertex_labels == i)  # Use numeric ID
        print(f"  {label}: {count} vertices")
    
    # Create separate mesh for each label
    for i, label in enumerate(unique_labels):
        print(f"\nProcessing part: {label}")
        
        # Get vertices for this label using numeric ID
        vertex_mask = vertex_labels == i
        if not np.any(vertex_mask):
            print(f"No vertices found for part {label}, skipping...")
            continue
            
        # Get faces that have majority of vertices with this label
        face_has_label = vertex_mask[mesh.faces]
        face_mask = face_has_label.sum(axis=1) >= 2  # At least 2 of 3 vertices have the label
        
        if not np.any(face_mask):
            print(f"No faces found for part {label}, skipping...")
            continue
        
        # Create new mesh for this part
        part_vertices = mesh.vertices[vertex_mask]
        
        # Create new face indices
        old_to_new = np.cumsum(vertex_mask) - 1
        part_faces = old_to_new[mesh.faces[face_mask]]
        
        part_mesh = trimesh.Trimesh(
            vertices=part_vertices,
            faces=part_faces
        )
        
        # Save part mesh with instance ID in filename
        output_path = os.path.join(output_dir, f'{OBJECT}_part_{label}.glb')
        part_mesh.export(output_path)
        print(f"Saved part mesh to: {output_path}")
        print(f"Part statistics:")
        print(f"  Vertices: {len(part_vertices)}")
        print(f"  Faces: {len(part_faces)}")


# The function for loading EXR depth maps
def load_blender_depth_exr(filepath):
    """Load depth from Blender EXR file with better error handling"""
    try:
        # Try using OpenEXR if available
        import OpenEXR
        import Imath
        import array
        
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Determine which channel has the depth information
        # Blender typically stores depth in Z channel, but could be R/G/B
        channels = header['channels']
        channel_names = list(channels.keys())
        
        print(f"Available channels in EXR: {channel_names}")
        
        # Try Z first, then R, G, B
        depth_channel = None
        for ch in ['Z', 'R', 'G', 'B']:
            if ch in channel_names:
                depth_channel = ch
                break
                
        if depth_channel is None:
            depth_channel = channel_names[0]  # Use first available channel
            
        print(f"Using channel '{depth_channel}' for depth")
        
        # Get pixel type
        pixel_type = channels[depth_channel].type
        
        # Read the data
        str_data = exr_file.channel(depth_channel, pixel_type)
        
        # Convert to numpy array
        if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
            depth = np.frombuffer(str_data, dtype=np.float32)
        elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
            depth = np.frombuffer(str_data, dtype=np.float16)
        else:
            print(f"Unknown pixel type {pixel_type}, trying float32")
            depth = np.frombuffer(str_data, dtype=np.float32)
            
        # Reshape to 2D
        depth = depth.reshape(height, width)
        
    except (ImportError, ModuleNotFoundError):
        print("OpenEXR not available, trying alternative method...")
        
        # Try using imageio
        try:
            import imageio.v3 as iio
            depth = iio.imread(filepath)
            
            # If multi-channel, take first channel
            if len(depth.shape) == 3:
                depth = depth[:,:,0]
                
        except (ImportError, ModuleNotFoundError):
            print("imageio not available, trying OpenCV...")
            
            # Try using OpenCV
            try:
                import cv2
                # IMREAD_UNCHANGED preserves the full precision
                depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                
                # If multi-channel, take first channel
                if len(depth.shape) == 3:
                    depth = depth[:,:,0]
                    
            except Exception as e:
                print(f"Failed to load depth with OpenCV: {e}")
                # If all else fails, create a dummy depth map
                print("WARNING: Creating placeholder depth map")
                depth = np.ones((512, 512), dtype=np.float32)
    
    # Print statistics
    print(f"Loaded depth with shape {depth.shape}, dtype {depth.dtype}")
    valid_mask = ~np.isnan(depth) & ~np.isinf(depth) & (depth > 0)
    if np.any(valid_mask):
        print(f"Valid depth range: {np.min(depth[valid_mask]):.6f} to {np.max(depth[valid_mask]):.6f}")
        print(f"Mean valid depth: {np.mean(depth[valid_mask]):.6f}")
    else:
        print("WARNING: No valid depth values found!")
    
    return depth

def visualize_point_cloud_with_mesh(points, labels, mesh, output_path, flip_mesh_z=True):
    """
    Visualize point cloud and mesh to check alignment, with fallback for headless environments.
    Make mesh more prominent and point clouds more transparent.
    
    Args:
        points: Nx3 array of 3D points
        labels: Array of labels for each point
        mesh: Trimesh mesh object
        output_path: Path to save the visualization
        flip_mesh_z: Whether to flip the Z coordinate of the mesh to match points
    """
    import numpy as np
    import os
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # First try using matplotlib (more widely compatible)
    try:
        print("Visualizing with matplotlib...")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a color map for the unique labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10.colors
        
        # Plot mesh first so it's in the background
        mesh_vertices = np.array(mesh.vertices)
        if flip_mesh_z:
            mesh_vertices[:, 2] = -mesh_vertices[:, 2]
            mesh_vertices[:, 1] = -mesh_vertices[:, 1]

        # Apply Z offset
        # mesh_vertices[:, 2] -= 1.0
            
        # Plot mesh vertices with higher visibility
        ax.scatter(
            mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2],
            c='black', marker='.', s=2, alpha=0.5, label='Mesh Vertices'
        )
        
        # Plot mesh wireframe for better visibility
        # This is computationally expensive for large meshes, so limit the faces
        mesh_faces = np.array(mesh.faces)
        max_faces = min(5000, len(mesh_faces))
        if len(mesh_faces) > max_faces:
            step = len(mesh_faces) // max_faces
            mesh_faces = mesh_faces[::step]
        
        for face in mesh_faces:
            vertices = mesh_vertices[face]
            # Draw each edge of the triangle
            for i in range(3):
                ax.plot3D(
                    [vertices[i, 0], vertices[(i+1)%3, 0]],
                    [vertices[i, 1], vertices[(i+1)%3, 1]],
                    [vertices[i, 2], vertices[(i+1)%3, 2]],
                    color='gray', linewidth=0.5, alpha=0.3
                )
        
        # Plot the point cloud with colors by label (more transparent)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if np.any(mask):
                ax.scatter(
                    points[mask, 0], points[mask, 1], points[mask, 2],
                    c=[colors[i % len(colors)]],
                    marker='.', s=1, alpha=0.3, label=label
                )
        
        # Add coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X axis')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y axis')
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z axis')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud and Mesh Alignment')
        
        # Add legend
        handles, labels_text = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_text, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Set equal aspect ratio
        max_range = np.array([
            ax.get_xlim()[1] - ax.get_xlim()[0],
            ax.get_ylim()[1] - ax.get_ylim()[0],
            ax.get_zlim()[1] - ax.get_zlim()[0]
        ]).max() / 2.0
        
        mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) * 0.5
        mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) * 0.5
        mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matplotlib visualization saved to: {output_path}")
        return
        
    except Exception as e:
        print(f"Matplotlib visualization failed: {e}")
        print("Falling back to Open3D...")
    
    # Try Open3D as a fallback, with better error handling
    try:
        import open3d as o3d
        
        # Create colored point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Create a simple color map for labels
        unique_labels = np.unique(labels)
        label_to_color = {}
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]
        
        for i, label in enumerate(unique_labels):
            label_to_color[label] = colors[i % len(colors)]
        
        # Assign colors to points
        point_colors = np.zeros((len(points), 3))
        for i, label in enumerate(labels):
            point_colors[i] = label_to_color[label]
        
        # Add alpha/transparency to point cloud colors (not supported directly in Open3D)
        # We'll make the colors less saturated as an approximation
        point_colors = point_colors * 0.7 + 0.3  # Reduce saturation
        
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # Create mesh with better visibility
        o3d_mesh = o3d.geometry.TriangleMesh()
        mesh_vertices = np.array(mesh.vertices)
        
        # Flip mesh Z coordinate if needed
        # if flip_mesh_z:
        #     mesh_vertices[:, 2] = -mesh_vertices[:, 2]
        #     mesh_vertices[:, 1] = -mesh_vertices[:, 1]

        # Rotation matrix for 180 degrees around Z
        # rot_z_180 = np.array([
        #     [-1, 0, 0],
        #     [0, -1, 0],
        #     [0, 0, 1]
        # ])
        # mesh_vertices = np.dot(mesh_vertices, rot_z_180.T)
            
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
        o3d_mesh.compute_vertex_normals()
        
        # Make mesh more prominent with darker gray
        o3d_mesh.paint_uniform_color([0.3, 0.3, 0.3])  # Darker gray for better contrast
        
        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        
        # Try non-interactive rendering (for headless systems)
        print("Attempting to render with Open3D offscreen renderer...")
        vis = o3d.visualization.Visualizer()
        try:
            vis.create_window(visible=False)
            vis.add_geometry(o3d_mesh)  # Add mesh first (background)
            vis.add_geometry(pcd)       # Then points (foreground)
            vis.add_geometry(coord_frame)
            
            # Set render options
            render_option = vis.get_render_option()
            render_option.mesh_show_wireframe = True
            render_option.line_width = 2.0  # Thicker wireframe
            render_option.point_size = 2.0
            render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Flat
            
            # Set camera parameters for a good view
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            # Update and capture
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(output_path)
            vis.destroy_window()
            
            print(f"Open3D non-interactive visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Open3D non-interactive visualization failed: {e}")
            vis.destroy_window()
            
            # If non-interactive fails, try to save the geometries directly
            print("Trying to save geometries directly...")
            o3d.io.write_point_cloud(output_path.replace(".png", "_points.ply"), pcd)
            o3d.io.write_triangle_mesh(output_path.replace(".png", "_mesh.ply"), o3d_mesh)
            
            print(f"Saved point cloud to: {output_path.replace('.png', '_points.ply')}")
            print(f"Saved mesh to: {output_path.replace('.png', '_mesh.ply')}")
            
    except Exception as e:
        print(f"All visualization methods failed: {e}")
        
        # As a last resort, save the raw data
        np.savez(output_path.replace(".png", "_data.npz"), 
                points=points, labels=labels, 
                mesh_vertices=mesh.vertices, mesh_faces=mesh.faces)
        
        print(f"Saved raw point cloud and mesh data to: {output_path.replace('.png', '_data.npz')}")


def merge_point_clouds_by_label(point_clouds, point_labels, unique_labels, align_clouds=True):
    """
    Merge multiple point clouds into one per label, with optional alignment.
    
    Args:
        point_clouds: List of Nx3 arrays containing point clouds, or a single Nx3 array
        point_labels: List of arrays containing labels for each point cloud, 
                     or a single array of labels for all points
        unique_labels: Array of all unique label values
        align_clouds: Whether to align point clouds before merging
        
    Returns:
        merged_points_by_label: Dictionary mapping label to merged point cloud
        merged_count_by_label: Dictionary mapping label to point count
    """
    print("Merging point clouds by label...")
    
    # Handle different input formats
    if not isinstance(point_clouds, list):
        # If point_clouds is a single array (not a list of arrays)
        if isinstance(point_labels, (np.ndarray, list)):
            # If point_labels is an array matching point_clouds
            if len(point_clouds) == len(point_labels):
                # Convert to list of arrays format (one cloud, with matching labels)
                point_clouds = [point_clouds]
                point_labels = [point_labels]
            else:
                raise ValueError(f"Point cloud size ({len(point_clouds)}) doesn't match labels size ({len(point_labels)})")
        else:
            # If point_labels is a single value
            raise ValueError("If point_clouds is a single array, point_labels must be a matching array")
    
    # Initialize dictionaries
    clouds_by_label = {label: [] for label in unique_labels}
    merged_points_by_label = {label: None for label in unique_labels}
    merged_count_by_label = {label: 0 for label in unique_labels}
    
    # Group point clouds by label
    for cloud_idx, (cloud, labels) in enumerate(zip(point_clouds, point_labels)):
        # Check if labels is a single value instead of an array
        if isinstance(labels, (int, np.integer)):
            # If labels is a single value, create an array of that value
            label = labels
            if label in unique_labels:
                clouds_by_label[label].append(cloud)
                merged_count_by_label[label] += len(cloud)
            continue
        
        # Normal case: labels is an array matching the cloud points
        if len(cloud) != len(labels):
            print(f"Warning: Point cloud {cloud_idx} size ({len(cloud)}) doesn't match labels size ({len(labels)})")
            continue
            
        # Add points to appropriate label bucket
        for label in unique_labels:
            mask = labels == label
            points_for_label = cloud[mask]
            if len(points_for_label) > 0:
                clouds_by_label[label].append(points_for_label)
                merged_count_by_label[label] += len(points_for_label)
    
    # Process and merge clouds for each label
    for label in unique_labels:
        label_clouds = clouds_by_label[label]
        
        if not label_clouds:
            print(f"  Label {label}: No points found")
            merged_points_by_label[label] = np.zeros((0, 3))
            continue
            
        # If we have multiple clouds for this label and alignment is requested
        if align_clouds and len(label_clouds) > 1:
            print(f"  Label {label}: Aligning {len(label_clouds)} point clouds")
            
            # Use the largest cloud as the reference target
            cloud_sizes = [len(cloud) for cloud in label_clouds]
            target_idx = np.argmax(cloud_sizes)
            target_cloud = label_clouds[target_idx]
            
            # Align and collect all aligned clouds
            aligned_clouds = [target_cloud]  # Start with the target cloud
            
            for i, source_cloud in enumerate(label_clouds):
                if i == target_idx:  # Skip the target cloud
                    continue
                    
                if len(source_cloud) < 10:  # Skip very small clouds
                    print(f"    Skipping cloud with only {len(source_cloud)} points")
                    continue
                
                try:
                    print(f"    Aligning cloud {i+1}/{len(label_clouds)} with {len(source_cloud)} points")
                    aligned_cloud = register_point_cloud(
                        source_cloud, target_cloud, 
                        voxel_size=0.01,  # Adjust based on your data scale
                        visualize=False   # Set to True for debugging
                    )
                    aligned_clouds.append(aligned_cloud)
                except Exception as e:
                    print(f"    Error aligning cloud {i}: {str(e)}")
                    # Fall back to using the original cloud
                    aligned_clouds.append(source_cloud)
            
            # Merge the aligned clouds
            merged_points_by_label[label] = np.vstack(aligned_clouds)
            
        else:
            # Simple concatenation if no alignment needed or only one cloud
            merged_points_by_label[label] = np.vstack(label_clouds)
        
        print(f"  Label {label}: {merged_count_by_label[label]} points merged from {len(label_clouds)} clouds")
        print(f"    Final cloud has {len(merged_points_by_label[label])} points")
    
    return merged_points_by_label, merged_count_by_label

def remove_duplicate_points(points, tolerance=0.001):
    """
    Remove duplicate points that are very close to each other.
    
    Args:
        points: Nx3 array of points
        tolerance: Distance tolerance for considering points as duplicates
        
    Returns:
        Nx3 array with duplicates removed
    """
    if len(points) == 0:
        return points
        
    # Build KD-tree for finding neighbors
    tree = cKDTree(points)
    
    # Find duplicate groups
    duplicate_groups = tree.query_ball_point(points, tolerance)
    
    # Keep track of which points to keep
    to_keep = np.ones(len(points), dtype=bool)
    
    # Process each point
    for i, group in enumerate(duplicate_groups):
        # If this point has already been marked for removal, skip
        if not to_keep[i]:
            continue
        
        # Mark all other points in this group for removal
        for j in group:
            if i != j:  # Don't remove the reference point
                to_keep[j] = False
    
    # Return only the points we want to keep
    return points[to_keep]

def downsample_point_cloud(points, target_count=None, voxel_size=None):
    """
    Downsample a point cloud to reduce density.
    
    Args:
        points: Nx3 array of points
        target_count: Target number of points (if None, use voxel_size)
        voxel_size: Size of voxel for downsampling (if None and target_count is None, 
                   will be calculated to reduce points by ~50%)
    
    Returns:
        Downsampled point cloud
    """
    if len(points) == 0:
        return points
        
    # If no target count or voxel size is specified, aim to reduce by half
    if target_count is None and voxel_size is None:
        target_count = len(points) // 2
    
    if voxel_size is None:
        # Estimate appropriate voxel size based on point density and target count
        # Get bounding box of points
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        volume = np.prod(max_bound - min_bound)
        
        # Calculate approximate voxel size
        points_per_volume = len(points) / volume
        target_points_per_volume = target_count / volume
        volume_ratio = points_per_volume / target_points_per_volume
        voxel_size = volume_ratio ** (1/3)  # Cube root to get linear dimension
    
    # Create a voxel grid
    voxel_grid = {}
    
    # Assign each point to a voxel and keep track of average point in each voxel
    for i, point in enumerate(points):
        # Calculate voxel coordinates
        voxel_key = tuple((point // voxel_size).astype(int))
        
        if voxel_key in voxel_grid:
            # Update existing voxel
            voxel_grid[voxel_key][0] += point
            voxel_grid[voxel_key][1] += 1
        else:
            # Create new voxel
            voxel_grid[voxel_key] = [point, 1]
    
    # Calculate average point for each voxel
    downsampled_points = np.array([summed_point / count for summed_point, count in voxel_grid.values()])
    
    print(f"  Downsampled from {len(points)} to {len(downsampled_points)} points")
    return downsampled_points

def process_point_clouds_for_segmentation(point_clouds, point_labels, unique_labels, max_points_per_label=10000):
    """
    Process and prepare point clouds for mesh segmentation.
    
    Args:
        point_clouds: List of Nx3 arrays containing point clouds
        point_labels: List of arrays containing labels for each point cloud
        unique_labels: Array of all unique label values
        max_points_per_label: Maximum number of points to keep per label
        
    Returns:
        Dictionary mapping label to processed point cloud
    """
    print("\nProcessing point clouds for segmentation...")
    
    # Step 1: Merge point clouds by label
    merged_points, merged_counts = merge_point_clouds_by_label(point_clouds, point_labels, unique_labels)
    
    # Step 2: Process each merged point cloud
    processed_points = {}
    
    for label in unique_labels:
        points = merged_points[label]
        if len(points) == 0:
            processed_points[label] = points
            continue
            
        print(f"\nProcessing label {label} with {len(points)} points:")
        
        # Remove duplicates
        points = remove_duplicate_points(points, tolerance=0.001)
        print(f"  After removing duplicates: {len(points)} points")
        
        # Downsample if necessary
        # if len(points) > max_points_per_label:
        #     points = downsample_point_cloud(points, target_count=max_points_per_label)
        
        processed_points[label] = points
    
    return processed_points


def segment_model_by_labels(mesh, point_clouds, point_labels, instance_ids, output_dir, OBJECT):
    """
    Main function to segment a 3D model based on labeled point clouds with instance information.
    
    Args:
        mesh: Trimesh mesh object
        point_clouds: List of point clouds (each a Nx3 array)
        point_labels: List of labels for each point cloud
        instance_ids: List of instance IDs for each point
        output_dir: Output directory for part meshes
        OBJECT: Object name
    """
    # Get unique instance IDs
    unique_instances = np.unique(instance_ids)
    print(f"Found {len(unique_instances)} unique instances to segment")
    
    # Process each instance separately
    for instance_id in unique_instances:
        # Get points for this instance
        instance_mask = instance_ids == instance_id
        instance_points = point_clouds[instance_mask]
        
        if len(instance_points) == 0:
            print(f"No points for instance {instance_id}, skipping...")
            continue
            
        # Get the label for this instance (should be the same for all points)
        instance_label = point_labels[instance_mask][0]
        
        print(f"\nSegmenting mesh for instance {instance_id} (label {instance_label}) with {len(instance_points)} points...")
        
        # Process this instance's points
        processed_points = remove_duplicate_points(instance_points, tolerance=0.001)
        print(f"  After removing duplicates: {len(processed_points)} points")
        
        # Create label array for these points (all same instance)
        labels = np.full(len(processed_points), instance_id)
        
        # Segment and save this part
        segment_and_save_parts(
            mesh=mesh,
            segmented_points=processed_points,
            point_labels=labels,
            unique_labels=[instance_id],  # Only process this single instance
            output_dir=output_dir,
            OBJECT=f"{OBJECT}_{instance_id}",  # Include instance ID in output filename
            flip_mesh_z=False
        )

def register_point_cloud(source_points, target_points, voxel_size=0.05, max_iterations=100, 
                          method="point_to_plane", visualize=False):
    """
    Align source point cloud to target using ICP via Open3D.
    
    Args:
        source_points: Nx3 numpy array of source points
        target_points: Mx3 numpy array of target points
        voxel_size: Size of voxels for downsampling (smaller = more accurate but slower)
        max_iterations: Maximum ICP iterations
        method: Registration method ("point_to_point" or "point_to_plane")
        visualize: Whether to visualize the registration result
        
    Returns:
        tuple: (aligned_points, transformation_matrix, fitness, rmse)
    """
    # Convert numpy arrays to Open3D point clouds
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    
    # Estimate normals if using point-to-plane ICP
    if method == "point_to_plane":
        print("Computing normals for target point cloud...")
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
        )
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
        )
    
    # Downsample both point clouds for faster registration
    print(f"Downsampling point clouds with voxel size {voxel_size}...")
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # Compute FPFH features for global registration initialization
    print("Computing FPFH features...")
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30)
    )
    
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    
    # Perform global registration using RANSAC to get initial alignment
    print("Performing global registration...")
    distance_threshold = voxel_size * 1.5
    
    result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    # Fine-tune alignment with either point-to-point or point-to-plane ICP
    print(f"Performing {method} ICP...")
    
    if method == "point_to_point":
        icp_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_global.transformation,
            icp_estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
    else:  # point_to_plane
        icp_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_global.transformation,
            icp_estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
    
    # Apply the transformation to the source point cloud
    aligned_source = copy.deepcopy(source)
    aligned_source.transform(result_icp.transformation)
    
    print(f"Registration result:")
    print(f"  Fitness: {result_icp.fitness}")
    print(f"  RMSE: {result_icp.inlier_rmse}")
    
    # Convert back to numpy array
    aligned_points = np.asarray(aligned_source.points)
    
    # Visualize if requested
    if visualize:
        # Color the point clouds
        source.paint_uniform_color([1, 0, 0])  # Red for source
        target.paint_uniform_color([0, 1, 0])  # Green for target
        aligned_source.paint_uniform_color([0, 0, 1])  # Blue for aligned source
        
        # Visualize point clouds
        print("Visualizing registration result:")
        print("  Red: Source point cloud")
        print("  Green: Target point cloud")
        print("  Blue: Aligned source point cloud")
        o3d.visualization.draw_geometries([source, target, aligned_source])
    
    return aligned_points, result_icp.transformation, result_icp.fitness, result_icp.inlier_rmse


def align_mesh_to_point_cloud(mesh, points, visualize=True, output_path=None):
    """
    Align mesh to point cloud using a fixed manual transformation.
    
    Args:
        mesh: Trimesh mesh object
        points: Nx3 array of point cloud points
        visualize: Whether to visualize the alignment result
        output_path: Path to save visualization
        
    Returns:
        Aligned mesh
    """
    print("Aligning mesh to point cloud using fixed transformation...")
    
    # Create a copy of the mesh to transform
    aligned_mesh = mesh.copy()
    
    # Apply a fixed transformation that works for your specific case
    # This is a combination of flips, rotations, and possibly scaling/translation
    
    # Example transformation - adjust these values based on your specific needs
    # 1. Flip Z axis
    flip_z = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    aligned_mesh.apply_transform(flip_z)
    
    # 2. Rotate 180 degrees around Y axis
    theta = np.radians(180)
    rot_y_180 = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    aligned_mesh.apply_transform(rot_y_180)
    
    # 3. Rotate 90 degrees around X axis
    theta = np.radians(90)  # 90 degrees in radians
    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    aligned_mesh.apply_transform(rotation_x)
    
    # Visualize the alignment if requested
    if visualize:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot a subset of the point cloud
        if len(points) > 5000:
            idx = np.random.choice(len(points), 5000, replace=False)
            pc_subset = points[idx]
        else:
            pc_subset = points
            
        ax.scatter(pc_subset[:, 0], pc_subset[:, 1], pc_subset[:, 2], 
                  c='blue', marker='.', s=1, alpha=0.5, label='Point Cloud')
        
        # Plot a subset of the aligned mesh vertices
        mesh_vertices = np.array(aligned_mesh.vertices)
        if len(mesh_vertices) > 5000:
            idx = np.random.choice(len(mesh_vertices), 5000, replace=False)
            mesh_subset = mesh_vertices[idx]
        else:
            mesh_subset = mesh_vertices
            
        ax.scatter(mesh_subset[:, 0], mesh_subset[:, 1], mesh_subset[:, 2],
                  c='red', marker='.', s=1, alpha=0.5, label='Aligned Mesh')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mesh and Point Cloud Alignment')
        ax.legend()
        
        # Save or show the visualization
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Alignment visualization saved to: {output_path}")
        else:
            plt.show()
    
    return aligned_mesh


def main(OBJECT, flip_z=True):
    # Define paths
    data_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{OBJECT}"
    render_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{OBJECT}"
    output_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{OBJECT}"
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load RGB images
    rgb_images = sorted([os.path.join(render_dir, f) for f in os.listdir(render_dir) 
                        if f.startswith(f"render_{OBJECT}") and f.endswith(".png")])
    
    # Load camera parameters
    camera_param_files = sorted([os.path.join(render_dir, f) for f in os.listdir(render_dir) 
                               if f.startswith(f"camera_params_{OBJECT}") and f.endswith(".npz")])
    
    # Load camera parameters
    camera_params = []
    for f in camera_param_files:
        data = np.load(f)
        camera_params.append({
            'K': data['K'],
            'R': data['R'],
            't': data['t']
        })
    
    # Debug camera positions
    print("\nCamera positions:")
    for i, params in enumerate(camera_params):
        print(f"Camera {i}: {params['t']}")
    
    # Load depth maps
    depth_files = sorted([os.path.join(render_dir, f) for f in os.listdir(render_dir) 
                         if f.startswith(f"depth_{OBJECT}") and f.endswith(".exr")])
    
    # Load depth maps
    depth_maps = []
    for f in depth_files:
        depth_map = load_blender_depth_exr(f)
        depth_maps.append(depth_map)
    
    # Load segmentation results
    results_paths = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                           if f.startswith(f"render_{OBJECT}") and f.endswith("_results.json")])
    
    # Check that we have the same number of items for each type
    print(f"\nFound {len(rgb_images)} RGB images")
    print(f"Found {len(camera_param_files)} camera parameter files")
    print(f"Found {len(depth_files)} depth files")
    print(f"Found {len(results_paths)} segmentation result files")
    
    if not (len(rgb_images) == len(camera_param_files) == len(depth_files) == len(results_paths)):
        print("Warning: Mismatched number of files. The script may not work correctly.")

    # update output_dir to new folder
    output_dir = os.path.join(output_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize camera positions
    visualize_camera_orientations(camera_param_files)
    
    # Lift 2D masks to 3D
    points_3d, labels, instance_ids = lift_2d_masks_to_3d(
        rgb_images, results_paths, depth_maps, camera_params, 
        OBJECT, output_dir, flip_z=flip_z
    )
    
    if len(points_3d) == 0:
        print("No valid 3D points found, exiting.")
        return
    
    # Load object mesh
    mesh_path = os.path.join(data_dir, f"{OBJECT}.glb")
    if not os.path.exists(mesh_path):
        print(f"Mesh file not found: {mesh_path}")
        return
        
    scene = trimesh.load(mesh_path)
    # mesh = scene.geometry['model']
    mesh = list(scene.geometry.values())[0]

    print(f"\nLoaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

    # Align mesh to point cloud before segmentation
    aligned_mesh = align_mesh_to_point_cloud(
        mesh, 
        points_3d, 
        visualize=True, 
        output_path=os.path.join(output_dir, "mesh_alignment.png")
    )
    
    # Visualize point cloud with mesh
    visualization_path = os.path.join(output_dir, "point_cloud_with_mesh.png")
    print("\nVisualizing point cloud alignment with mesh...")
    visualize_point_cloud_with_mesh(points_3d, labels, aligned_mesh, visualization_path)
    
    # Segment mesh into parts
    unique_labels = np.unique(labels)
    # segment_and_save_parts(mesh, points_3d, np.array([list(unique_labels).index(l) for l in labels]), 
    #                       unique_labels, output_dir, OBJECT)
    segment_model_by_labels(aligned_mesh, points_3d, labels,
                            instance_ids, output_dir, OBJECT)
    
    print("\nProcessing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('object', default='drawer_rodin')
    args = parser.parse_args()
    main(args.object, flip_z=False)