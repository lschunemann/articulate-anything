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
    # plt.show()
    
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
    Modified to handle empty arrays gracefully.
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
    
    # Check if we have any valid depth values after filtering
    if len(depth_values) == 0:
        print(f"  After filtering: No valid depth values (all values were out of range or invalid)")
        return np.array([])
    
    # If we have valid values, print their range
    print(f"  After filtering: {len(depth_values)} valid points (range: {depth_values.min():.6f} to {depth_values.max():.6f})")
    
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

def sanitize_filename(filename):
    """
    Sanitize a filename by replacing invalid characters with underscores.
    """
    # Replace characters that are invalid in filenames
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# Main function to lift 2D masks to 3D
def lift_2d_masks_to_3d(rgb_images, results_paths, depth_maps, camera_params, OBJECT, output_dir, flip_z=True):
    """
    Lift 2D segmentation masks to 3D points using depth maps and camera parameters.
    Modified to handle missing segmentation results.
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
        
        # Check if we have segmentation results for this view
        if view_idx >= len(results_paths):
            print(f"No segmentation results found for view {view_idx + 1}, skipping...")
            continue
            
        try:
            # Load masks for this view
            masks, label_names = load_masks_from_results(results_paths[view_idx], OBJECT)

            label_names = [sanitize_filename(label) for label in label_names]
            
            # Skip if no masks were found
            if len(masks) == 0:
                print(f"No masks found in results for view {view_idx + 1}, skipping...")
                continue
                
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

                sanitized_label = sanitize_filename(label_name)
                instance_id = f"{sanitized_label}_{label_instance_counters[label_name]}"
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
                
        except Exception as e:
            print(f"Error processing view {view_idx}: {str(e)}")
            print("Skipping this view and continuing...")
            continue

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
    """Load masks from the JSON results file with better error handling"""
    print(f"Loading masks from {results_path}")
    
    try:
        full_path = os.path.join(f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{OBJECT}", results_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"Results file not found: {full_path}")
            return np.array([]), []
        
        with open(full_path, 'r') as f:
            results = json.load(f)
        
        # Check if results contain annotations
        if 'annotations' not in results or not results['annotations']:
            print(f"No annotations found in results file")
            return np.array([]), []
        
        masks = []
        labels = []
        
        for annotation in results['annotations']:
            try:
                rle = annotation['segmentation']
                rle['counts'] = rle['counts'].encode('utf-8')
                mask = mask_util.decode(rle)
                
                masks.append(mask)
                labels.append(annotation['class_name'])
            except Exception as e:
                print(f"Error processing annotation: {str(e)}")
                continue
        
        if not masks:
            print(f"No valid masks could be decoded from results")
            return np.array([]), []
            
        print(f"Loaded {len(masks)} masks with labels: {labels}")
        return np.stack(masks), labels
        
    except Exception as e:
        print(f"Error loading masks from {results_path}: {str(e)}")
        return np.array([]), []

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

            # label = sanitize_filename(label)
            
            # Save figure with label in the filename
            plt.savefig(os.path.join(output_dir, f'{OBJECT}_points_label_{label}_view_{view_idx+1}.png'), 
        dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create mesh visualization with vertex colors for each label
    for label in unique_labels:
        # Create a colored mesh with this label highlighted
        # Get masks for different label categories
        mask_label = np.array(vertex_labels) == label
        mask_other = (np.array(vertex_labels) != label) & (np.array(vertex_labels) != "unassigned")  # Other labels except unassigned
        mask_unassigned = np.array(vertex_labels) == "unassigned"
        
        # For each view angle
        for view_idx, (elev, azim) in enumerate(view_angles):
            # Skip the trimesh visualization attempt and go straight to matplotlib
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot mesh vertices with colors
            vertices = mesh.vertices
            
            # Create a colormap for visualization
            vertex_colors = np.zeros((len(vertices), 3))
            vertex_colors[mask_label] = label_colors[label]
            vertex_colors[mask_other] = [0.8, 0.8, 0.8]  # Light gray for other labels
            vertex_colors[mask_unassigned] = [0.9, 0.9, 0.9]  # Very light gray for unassigned
            
            # Plot a subset of vertices
            sample_rate = max(1, len(vertices) // 10000)
            sampled_indices = np.arange(0, len(vertices), sample_rate)
            
            # Plot highlighted label with higher visibility
            if np.any(mask_label):
                highlighted_vertices = vertices[mask_label]
                # Subsample if there are too many points
                if len(highlighted_vertices) > 5000:
                    indices = np.random.choice(len(highlighted_vertices), 5000, replace=False)
                    highlighted_vertices = highlighted_vertices[indices]
                
                ax.scatter(
                    highlighted_vertices[:, 0],
                    highlighted_vertices[:, 1],
                    highlighted_vertices[:, 2],
                    c=[label_colors[label]],
                    marker='.',
                    s=15,  # Larger size for highlighted label
                    alpha=1.0,  # Full opacity
                    label=f'Label {label}'
                )
            
            # Plot other labels with less visibility
            if np.any(mask_other):
                # Sample other vertices
                other_indices = sampled_indices[mask_other[sampled_indices]]
                if len(other_indices) > 3000:
                    other_indices = np.random.choice(other_indices, 3000, replace=False)
                
                ax.scatter(
                    vertices[other_indices, 0],
                    vertices[other_indices, 1],
                    vertices[other_indices, 2],
                    c=[0.8, 0.8, 0.8],  # Light gray
                    marker='.',
                    s=5,  # Smaller size
                    alpha=0.3,  # More transparent
                    label='Other labels'
                )
            
            # Plot unassigned with least visibility
            if np.any(mask_unassigned):
                # Sample unassigned vertices
                unassigned_indices = sampled_indices[mask_unassigned[sampled_indices]]
                if len(unassigned_indices) > 2000:
                    unassigned_indices = np.random.choice(unassigned_indices, 2000, replace=False)
                
                ax.scatter(
                    vertices[unassigned_indices, 0],
                    vertices[unassigned_indices, 1],
                    vertices[unassigned_indices, 2],
                    c=[0.9, 0.9, 0.9],  # Very light gray
                    marker='.',
                    s=3,  # Smallest size
                    alpha=0.2,  # Most transparent
                    label='Unassigned'
                )
            
            # Set view angle
            ax.view_init(elev=elev, azim=azim)
            
            # Set consistent view limits
            ax.set_xlim([center[0] - max_range/2, center[0] + max_range/2])
            ax.set_ylim([center[1] - max_range/2, center[1] + max_range/2])
            ax.set_zlim([center[2] - max_range/2, center[2] + max_range/2])
            
            # Add title and labels
            ax.set_title(f'Mesh Label {label} - View {view_idx+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Add legend
            if np.any(mask_label) or np.any(mask_other) or np.any(mask_unassigned):
                ax.legend()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f'{OBJECT}_mesh_label_{label}_view_{view_idx+1}.png'), 
        dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved visualization for label {label}, view {view_idx+1}")
    
    print(f"Saved visualization images to: {output_dir}")
    print("Generated images:")
    for label in unique_labels:
        for view_idx in range(len(view_angles)):
            print(f"  {OBJECT}_label_{label}_view_{view_idx+1}.png")
    print(f"  {OBJECT}_distance_histogram.png")
    print(f"  {OBJECT}_color_legend.png")
    for view_idx in range(len(view_angles)):
        print(f"  {OBJECT}_view_{view_idx+1}.png")

def merge_instances_with_two_pass_approach(point_clouds_by_instance, instance_ids, 
                                          distance_threshold=0.05, consistency_threshold=0.3, # box: 0.3 consistency works well
                                          partial_overlap_threshold=0.7):
    # First pass: geometric consistency
    instance_mapping, clusters = merge_instances_with_geometric_consistency(
        point_clouds_by_instance, instance_ids, distance_threshold, consistency_threshold)
    
    # Second pass: partial containment check
    # updated_mapping, updated_clusters = merge_with_partial_observations(
    #     point_clouds_by_instance, instance_ids, clusters, 
    #     distance_threshold, partial_overlap_threshold)
    
    # return updated_mapping, updated_clusters
    return instance_mapping, clusters


def merge_with_partial_observations(point_clouds_by_instance, instance_mapping, clusters, 
                                   distance_threshold=0.05, partial_overlap_threshold=0.7):
    """
    Second pass to merge partial observations with more complete ones.
    
    Args:
        point_clouds_by_instance: Dictionary mapping instance IDs to point clouds
        instance_mapping: Current mapping from instance ID to representative ID
        clusters: Existing clusters from first pass
        distance_threshold: Distance threshold for point matching
        partial_overlap_threshold: Threshold for considering a smaller point cloud as contained
        
    Returns:
        Updated instance_mapping and updated clusters
    """
    from scipy.spatial import cKDTree
    import numpy as np
    import copy
    
    # Ensure instance_mapping is a dictionary
    if not isinstance(instance_mapping, dict):
        # Convert to dictionary if it's not already
        instance_mapping_dict = {}
        for i, instance_id in enumerate(instance_mapping):
            if instance_id is not None:  # Skip None values
                instance_mapping_dict[instance_id] = instance_mapping[instance_id]
        instance_mapping = instance_mapping_dict
    
    # Create a deep copy of the instance mapping and clusters to modify
    updated_mapping = copy.deepcopy(instance_mapping)
    updated_clusters = copy.deepcopy(clusters)
    
    # Function to check if a smaller point cloud is mostly contained within a larger one
    def check_partial_containment(smaller_pc, larger_pc, distance_threshold):
        if len(smaller_pc) == 0 or len(larger_pc) == 0:
            return 0.0
            
        # Build KD-tree for the larger point cloud
        tree_larger = cKDTree(larger_pc)
        
        # Find nearest neighbors from smaller to larger
        dist, _ = tree_larger.query(smaller_pc, distance_upper_bound=distance_threshold)
        
        # Count valid matches (within threshold)
        valid_matches = np.sum(np.isfinite(dist))
        
        # Calculate containment ratio - how much of the smaller is contained in the larger
        containment_ratio = valid_matches / len(smaller_pc) if len(smaller_pc) > 0 else 0
        
        return containment_ratio
    
    # Get singleton clusters (ones with only one instance)
    singleton_clusters = [i for i, cluster in enumerate(updated_clusters) if len(cluster) == 1]
    
    # For each singleton cluster, check if it's a partial observation of a larger cluster
    merged_count = 0
    clusters_to_remove = []
    
    for cluster_idx in singleton_clusters:
        if cluster_idx >= len(updated_clusters) or not updated_clusters[cluster_idx]:
            continue  # Skip if cluster is already removed or empty
            
        singleton_id = updated_clusters[cluster_idx][0]
        singleton_pc = point_clouds_by_instance[singleton_id]
        
        # Skip if no points
        if len(singleton_pc) == 0:
            continue
        
        best_score = 0.0
        best_cluster_idx = None
        
        # Compare with all other non-singleton clusters
        for other_idx, other_cluster in enumerate(updated_clusters):
            if other_idx == cluster_idx or len(other_cluster) <= 1:
                continue
            
            # Combine all point clouds in this cluster
            try:
                combined_pc = np.vstack([point_clouds_by_instance[instance_id] 
                                        for instance_id in other_cluster
                                        if instance_id in point_clouds_by_instance and 
                                        len(point_clouds_by_instance[instance_id]) > 0])
            except:
                # Skip if there's an issue combining the point clouds
                continue
            
            # If singleton is smaller, check containment
            if len(singleton_pc) < len(combined_pc):
                containment_score = check_partial_containment(singleton_pc, combined_pc, distance_threshold)
                
                if containment_score > partial_overlap_threshold and containment_score > best_score:
                    best_score = containment_score
                    best_cluster_idx = other_idx
        
        # If a good match was found, merge the singleton into that cluster
        if best_cluster_idx is not None:
            target_rep = updated_clusters[best_cluster_idx][0]
            print(f"Second pass: Instance {singleton_id} merged into cluster {best_cluster_idx} (containment={best_score:.4f})")
            
            # Update instance mapping
            updated_mapping[singleton_id] = target_rep
            
            # Move instance from singleton cluster to target cluster
            updated_clusters[best_cluster_idx].append(singleton_id)
            updated_clusters[cluster_idx] = []  # Empty this cluster (will be removed later)
            
            # Mark this cluster for removal
            clusters_to_remove.append(cluster_idx)
            
            merged_count += 1
    
    # Remove empty clusters
    final_clusters = [cluster for cluster in updated_clusters if cluster]
    
    print(f"Second pass merged {merged_count} partial instances, resulting in {len(final_clusters)} final clusters")
    
    # Make sure all instances in each cluster have the correct representative
    for cluster in final_clusters:
        if cluster:  # Only process non-empty clusters
            rep_id = cluster[0]  # First instance is the representative
            for instance_id in cluster:
                updated_mapping[instance_id] = rep_id
    
    return updated_mapping, final_clusters

    
def merge_instances_with_geometric_consistency(point_clouds_by_instance, instance_ids, distance_threshold=0.05, consistency_threshold=0.8): # TODO: find best threshold (for drawer 0.8 looked good, for microwave lower???) box:0.5
    """
    Merge instances using geometric consistency checking.
    
    Args:
        point_clouds_by_instance: Dictionary mapping instance IDs to point clouds
        instance_ids: List of all instance IDs
        distance_threshold: Distance threshold for point matching
        consistency_threshold: Ratio of consistent matches required
        
    Returns:
        Dictionary mapping original instance IDs to merged instance IDs
    """
    from scipy.spatial import cKDTree
    import numpy as np
    
    # Function to check geometric consistency between two point clouds
    def check_geometric_consistency(pc1, pc2, distance_threshold):
        # Build KD-trees
        tree1 = cKDTree(pc1)
        tree2 = cKDTree(pc2)
        
        # Find nearest neighbors in both directions
        dist1, idx1 = tree1.query(pc2, distance_upper_bound=distance_threshold)
        dist2, idx2 = tree2.query(pc1, distance_upper_bound=distance_threshold)
        
        # Count valid matches (within threshold)
        valid_matches1 = np.sum(np.isfinite(dist1))
        valid_matches2 = np.sum(np.isfinite(dist2))
        
        # Check consistency ratio
        consistency_ratio1 = valid_matches1 / len(pc2) if len(pc2) > 0 else 0
        consistency_ratio2 = valid_matches2 / len(pc1) if len(pc1) > 0 else 0
        
        # Use the minimum ratio as the consistency score
        consistency_score = min(consistency_ratio1, consistency_ratio2)
        
        return consistency_score
    
    # Initialize clusters
    clusters = []
    instance_to_cluster = {}
    
    # Process each instance
    for i, instance_id in enumerate(instance_ids):
        pc_i = point_clouds_by_instance[instance_id]
        
        # Skip if no points
        if len(pc_i) == 0:
            instance_to_cluster[instance_id] = None
            continue
        
        # Check if this instance is similar to any existing cluster
        assigned = False
        for cluster_idx, cluster_instances in enumerate(clusters):
            # Compare with all instances in the cluster
            consistency_scores = []
            
            for cluster_instance_id in cluster_instances:
                pc_j = point_clouds_by_instance[cluster_instance_id]
                score = check_geometric_consistency(pc_i, pc_j, distance_threshold)
                consistency_scores.append(score)
            
            # Use average consistency score
            avg_score = np.mean(consistency_scores)
            
            if avg_score > consistency_threshold:
                # Add to existing cluster
                clusters[cluster_idx].append(instance_id)
                instance_to_cluster[instance_id] = cluster_idx
                assigned = True
                print(f"Instance {instance_id} merged into cluster {cluster_idx} (score={avg_score:.4f})")
                break
        
        if not assigned:
            # Create new cluster
            cluster_idx = len(clusters)
            clusters.append([instance_id])
            instance_to_cluster[instance_id] = cluster_idx
            print(f"Instance {instance_id} creates new cluster {cluster_idx}")
    
    print(f"Merged {len(instance_ids)} instances into {len(clusters)} unique parts")
    
    # Create mapping from original instance IDs to representative instance IDs
    instance_mapping = {}
    for cluster_idx, cluster_instances in enumerate(clusters):
        representative_id = cluster_instances[0]
        for instance_id in cluster_instances:
            instance_mapping[instance_id] = representative_id
    
    return instance_mapping, clusters


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
    print("Calculating scaling based on bounding circle")
    # Compute bounding diameter for each point cloud
    source_diameter = np.max(np.linalg.norm(source_points[:, None, :] - source_points[None, :, :], axis=-1))
    target_diameter = np.max(np.linalg.norm(target_points[:, None, :] - target_points[None, :, :], axis=-1))
    scale_factor = source_diameter / target_diameter
    scaled_target_points = target_points * scale_factor
    return scaled_target_points, scale_factor

def simple_icp(source_cloud, target_cloud, max_correspondence_dist=100.0, max_iterations=500):
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
    print("performing icp point-to-plane alignment")
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
    print("Aligning mesh to point cloud")
    
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

    # Refine alignment with scaling and icp
    # use 10000 sampled points for scaling
    if len(points > 10000):
        indices = np.random.choice(len(points), 10000, replace=False)
        sampled_source_points = points[indices]
    else:
        sampled_source_points = points

    if len(aligned_mesh.vertices) > 10000:
        aligned_mesh_points = aligned_mesh.sample(10000)
    else:
        aligned_mesh_points = aligned_mesh.vertices

    # Scale target (mesh vertices) to match source (point cloud) diameter
    _, scale_factor = scale_target_to_source(aligned_mesh_points, sampled_source_points)

    # Create scaling matrix
    scaling_matrix = np.eye(4)
    scaling_matrix[0, 0] = scale_factor
    scaling_matrix[1, 1] = scale_factor
    scaling_matrix[2, 2] = scale_factor

    if len(points > 10000):
        # use a larger amount of points for icp alignment
        # indices = np.random.choice(len(points), 50000, replace=False)
        # sampled_source_points = points[indices]
        # scaled_mesh_vertices = sampled_source_points * scale_factor
        sampled_source_points = points * scale_factor ### no sampling
    else:
        sampled_source_points *= scale_factor
    
    # Convert NumPy arrays to Open3D point clouds for ICP
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(sampled_source_points)
    
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(aligned_mesh_points)
    
    # Perform ICP alignment
    aligned_cloud, transformation_matrix = simple_icp(source_cloud, target_cloud)

    # combine rototranslation and scaling
    transformation_matrix = np.matmul(transformation_matrix, scaling_matrix)
    
    # Apply the resulting transformation to the mesh
    # aligned_mesh.vertices = scaled_mesh_vertices  # First apply scaling
    # aligned_mesh.apply_transform(transformation_matrix)  # Then apply ICP transformation

    homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
    # Apply transformation
    transformed_homogeneous_points = homogeneous_points @ transformation_matrix.T
    # Convert back to 3D coordinates (divide by w if needed, which is the 4th column)
    transformed_points = transformed_homogeneous_points[:, :3]

    # Update the points
    points = transformed_points
    
    # # Compute vertex normals for rendering
    # # aligned_mesh.compute_vertex_normals()

    # # Let trimesh handle the normals calculation safely
    # try:
    #     # Use trimesh's built-in mechanisms that handle mixed face types
    #     aligned_mesh.fix_normals()
    # except Exception as e:
    #     print(f"Warning: Could not fix normals automatically: {str(e)}")
    #     # Fall back to a simpler approach
    #     try:
    #         # First ensure the mesh is triangulated
    #         if not aligned_mesh.is_watertight:
    #             print("Mesh is not watertight, attempting to process anyway")
            
    #         # For meshes with mixed face types, get triangulated faces
    #         triangles = aligned_mesh.triangles
    #         if triangles is not None and len(triangles) > 0:
    #             # Compute normals only if triangles are available
    #             normals = trimesh.triangles.normals(triangles)
    #             if len(normals) == len(aligned_mesh.faces):
    #                 aligned_mesh.face_normals = normals
    #     except Exception as e:
    #         print(f"Warning: Could not compute normals: {str(e)}")
    #         print("Visualization may have incorrect lighting")
    
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
    
    return aligned_mesh, points


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
    mesh = list(scene.geometry.values())[0]
    print(f"\nLoaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    # Align mesh to point cloud
    aligned_mesh, points_3d = align_mesh_to_point_cloud(
        mesh, 
        points_3d, 
        visualize=True, 
        output_path=os.path.join(output_dir, "mesh_alignment.png")
    )
    
    # Visualize point cloud with aligned mesh
    visualization_path = os.path.join(output_dir, "point_cloud_with_mesh.png")
    print("\nVisualizing point cloud alignment with mesh...")
    visualize_point_cloud_with_mesh(points_3d, labels, aligned_mesh, visualization_path, flip_mesh_z=False)
    
    # Organize point clouds by instance ID
    print("\nOrganizing point clouds by instance ID...")
    point_clouds_by_instance = {}
    for instance_id in np.unique(instance_ids):
        mask = instance_ids == instance_id
        point_clouds_by_instance[instance_id] = points_3d[mask]
        print(f"  Instance {instance_id}: {len(points_3d[mask])} points")
    
    # Merge similar instances using geometric consistency
    # print("\nMerging similar instances across views...")
    # instance_mapping, clusters = merge_instances_with_geometric_consistency(
    #     point_clouds_by_instance, 
    #     np.unique(instance_ids),
    #     distance_threshold=0.05,  # Adjust based on your data scale
    #     consistency_threshold=0.7  # Adjust based on desired strictness
    # )
    print("\nMerging similar instances across views with two-pass approach")
    instance_mapping, clusters = merge_instances_with_two_pass_approach(point_clouds_by_instance, np.unique(instance_ids))
    
    # Create merged point clouds
    print("\nCreating merged point clouds for unique parts...")
    merged_points_all = []
    merged_labels_all = []
    merged_instance_ids_all = []
    
    for cluster_idx, cluster_instances in enumerate(clusters):
        representative_id = cluster_instances[0]
        
        # Combine all points from this cluster
        cluster_points = []
        cluster_labels = []
        
        for instance_id in cluster_instances:
            mask = instance_ids == instance_id
            cluster_points.append(points_3d[mask])
            # Get the label for this instance (should be the same for all points in the instance)
            if np.any(mask):
                instance_label = labels[np.where(mask)[0][0]]
                cluster_labels.extend([instance_label] * np.sum(mask))
        
        # Stack all points
        if cluster_points:
            combined_points = np.vstack(cluster_points)
            
            # # Optionally downsample if there are too many points
            # if len(combined_points) > 10000:
            #     # Keep track of indices before downsampling
            #     indices = np.random.choice(len(combined_points), 10000, replace=False)
            #     combined_points = combined_points[indices]
            #     cluster_labels = [cluster_labels[i] for i in indices]
            
            # Add to merged collections
            merged_points_all.append(combined_points)
            merged_labels_all.extend(cluster_labels)
            merged_instance_ids_all.extend([representative_id] * len(combined_points))
            
            print(f"  Cluster {cluster_idx} (representative: {representative_id}): {len(combined_points)} points")
    
    # Stack all merged points
    if merged_points_all:
        merged_points = np.vstack(merged_points_all)
        merged_labels = np.array(merged_labels_all)
        merged_instance_ids = np.array(merged_instance_ids_all)
        
        print(f"\nFinal merged point cloud has {len(merged_points)} points with {len(np.unique(merged_instance_ids))} unique parts")
        
        # Visualize merged point cloud
        vis_path = os.path.join(output_dir, 'merged_point_cloud.png')
        
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
            merged_points, merged_instance_ids, all_camera_params,
            'Merged Point Cloud (Unique Parts)', 
            vis_path
        )
        
        # Segment mesh into parts using merged instances
        print("\nSegmenting mesh into unique parts...")
        
        # Create a mapping from string labels to numeric IDs for internal use
        unique_instances = np.unique(merged_instance_ids)
        label_to_id = {label: i for i, label in enumerate(unique_instances)}
        
        # Convert string instance IDs to numeric IDs
        numeric_instance_ids = np.array([label_to_id[label] for label in merged_instance_ids])
        
        # Segment and save parts
        segment_and_save_parts(
            mesh=aligned_mesh,
            segmented_points=merged_points,
            point_labels=numeric_instance_ids,
            unique_labels=list(range(len(unique_instances))),
            output_dir=output_dir,
            OBJECT=OBJECT,
            flip_mesh_z=False  # Already aligned
        )
        
        # Also save individual part meshes with meaningful names
        for i, instance_id in enumerate(unique_instances):
            # Get the original label for this instance
            instance_mask = merged_instance_ids == instance_id
            if np.any(instance_mask):
                instance_label = merged_labels[np.where(instance_mask)[0][0]]
                
                # Get points for this instance
                instance_points = merged_points[instance_mask]
                
                # Create label array for these points (all same instance)
                instance_numeric_id = label_to_id[instance_id]
                instance_point_labels = np.full(len(instance_points), instance_numeric_id)
                
                # Segment and save this part
                part_output_dir = os.path.join(output_dir, 'parts')
                os.makedirs(part_output_dir, exist_ok=True)
                
                segment_and_save_parts(
                    mesh=aligned_mesh,
                    segmented_points=instance_points,
                    point_labels=instance_point_labels,
                    unique_labels=[instance_numeric_id],
                    output_dir=part_output_dir,
                    OBJECT=f"{OBJECT}_{instance_label}_{i}",
                    flip_mesh_z=False  # Already aligned
                )
    else:
        print("No valid merged point clouds, skipping mesh segmentation.")
    
    print("\nProcessing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('object', default='drawer_rodin')
    args = parser.parse_args()
    main(args.object, flip_z=False)