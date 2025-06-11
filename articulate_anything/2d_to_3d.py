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
        
        right = data['right']
        up = data['up']
        view_dir = data['view_dir']
        
        # Ensure these are numpy arrays
        right = np.array(right)
        up = np.array(up)
        view_dir = np.array(view_dir)
        
        print(f"Camera {i} position: {t}")
        print(f"  Right: {right}")
        print(f"  Up: {up}")
        print(f"  View Dir: {view_dir}")
        
        # Plot camera position
        ax.scatter(t[0], t[1], t[2], c=colors[i % len(colors)], marker='^', s=100, label=f'Camera {i}')
        
        # Plot camera axes with clear labels
        # X axis (right) - red
        ax.quiver(t[0], t[1], t[2], right[0], right[1], right[2], 
                    color='r', length=axis_length, arrow_length_ratio=0.2)
        ax.text(t[0] + right[0]*axis_length, t[1] + right[1]*axis_length, t[2] + right[2]*axis_length, "X", color='r')
        
        # Y axis (up) - green
        ax.quiver(t[0], t[1], t[2], up[0], up[1], up[2], 
                    color='g', length=axis_length, arrow_length_ratio=0.2)
        ax.text(t[0] + up[0]*axis_length, t[1] + up[1]*axis_length, t[2] + up[2]*axis_length, "Y", color='g')
        
        # Z axis (viewing direction) - blue
        ax.quiver(t[0], t[1], t[2], -view_dir[0], -view_dir[1], -view_dir[2], 
                    color='b', length=axis_length, arrow_length_ratio=0.2)
        ax.text(t[0] - view_dir[0]*axis_length, t[1] - view_dir[1]*axis_length, t[2] - view_dir[2]*axis_length, "Z", color='b')
        
        # Draw a line from camera to origin (0,0,0.1) as target point
        ax.plot([t[0], 0], [t[1], 0], [t[2], 0.1], c=colors[i % len(colors)], linestyle='--', alpha=0.5)
    
    # Plot origin
    ax.scatter(0, 0, 0, c='k', marker='*', s=200, label='Origin')
    
    # Set labels and title
    ax.set_xlabel('X_world')
    ax.set_ylabel('Y_world')
    ax.set_zlabel('Z_world')
    ax.set_title('Camera Positions and Coordinate Axes (Blender World)')
    
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

# # The corrected back-projection function that worked correctly
# def backproject_depth_blender_fixed(depth_map, K, R, t, max_depth=100.0):
#     """
#     Correctly backproject depth maps from Blender with proper coordinate transformations.
    
#     Args:
#         depth_map: Depth map from Blender
#         K: Intrinsic camera matrix
#         R: Rotation matrix (world to camera)
#         t: Camera position in world coordinates
#         max_depth: Maximum depth threshold
#     """
#     # Handle 3-channel depth maps
#     if len(depth_map.shape) == 3:
#         # If it's a 3-channel depth map, take the first channel
#         depth_map = depth_map[:,:,0]
    
#     # Get image dimensions
#     height, width = depth_map.shape
    
#     # Create pixel coordinates grid
#     y, x = np.indices((height, width))
    
#     # Flatten all arrays
#     x = x.flatten()
#     y = y.flatten()
#     z = depth_map.flatten()
    
#     # Filter out invalid, zero, or extreme depth values
#     valid = (z > 0) & (z < max_depth) & np.isfinite(z)
#     x = x[valid]
#     y = y[valid]
#     z = z[valid]
    
#     print(f"Valid depth points: {np.sum(valid)} out of {len(valid)}")
    
#     if np.sum(valid) == 0:
#         print("WARNING: No valid depth points found!")
#         return np.array([])
    
#     # Convert pixel coordinates to camera coordinates
#     fx = K[0, 0]
#     fy = K[1, 1]
#     cx = K[0, 2]
#     cy = K[1, 2]
    
#     # Calculate 3D coordinates in the camera space
#     # In Blender, the camera looks along -Z, with Y up
#     # So we map: x_pixel -> X, y_pixel -> Y, depth -> -Z
#     x_cam = (x - cx) * z / fx
#     y_cam = (y - cy) * z / fy
#     z_cam = -z  # Negate Z because Blender camera looks down -Z axis
    
#     # Create points in camera space
#     points_cam = np.vstack((x_cam, y_cam, z_cam)).T
    
#     # Transform from camera space to world space
#     # In Blender, R is world-to-camera rotation
#     # We need its transpose for camera-to-world
#     R_cam_to_world = R.T
    
#     # Transform points from camera space to world space
#     points_world = np.zeros_like(points_cam)
#     for i, pt in enumerate(points_cam):
#         points_world[i] = R_cam_to_world.dot(pt) + t

#     # flip z axis since point clouds were mirrored
#     points_world[:,2] = -points_world[:,2] # TODO: check if correct
    
#     return points_world

def backproject_masked_depth(depth_map, mask, K, R, t, view_dir, right, up, max_depth=100.0, label=None, flip_z=True):
    """
    Backproject depth values only for masked regions using the same approach as the working function.
    Modified to handle empty arrays gracefully.
    """
    # Handle 3-channel depth maps
    if len(depth_map.shape) == 3:
        depth_map = depth_map[:,:,0]

    # cv2.imwrite('/tmp/depth.png', depth_map * 50)
    
    # Ensure mask has same shape as depth map
    if mask.shape != depth_map.shape:
        if len(mask.shape) == 3:
            mask = np.any(mask > 0, axis=2)
        else:
            mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
            mask = mask > 0
    
    # Find coordinates where mask is true
    y_coords, x_coords = np.where(mask == 1)
    
    if len(y_coords) == 0:
        print(f"No points in mask for {label if label else 'unknown'}")
        return np.array([])
    
    # # Erode mask to remove unsmooth edges
    mask_uint8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
    # mask_ = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    y_coords, x_coords = np.where(eroded_mask > 0)
    
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

    # Calculate normalized device coordinates
    x_ndc = (x_coords - cx) / fx
    y_ndc = (y_coords - cy) / fy

    z_cam = -depth_values  
    x_cam = x_ndc * depth_values
    y_cam = -y_ndc * depth_values
    
    # Create points in camera space
    points_cam = np.vstack((x_cam, y_cam, z_cam)).T
    
    # Transform from camera space to world space
    # In Blender, R is world-to-camera rotation
    # We need its transpose for camera-to-world
    R_cam_to_world = R.T

    points_world = np.zeros_like(points_cam)
        
    for i, pt in enumerate(points_cam):
        # Transform using camera's local coordinate system directly
        # pt[0] is along right vector, pt[1] is along up vector, pt[2] is along -view_dir
        points_world[i] = t + (right * pt[0]) + (up * pt[1]) + (-view_dir * pt[2])
    
    ## Transform points from camera space to world space
    ## points_world = np.zeros_like(points_cam)
    ## for i, pt in enumerate(points_cam):
    ##     points_world[i] = R_cam_to_world @ pt + t
    # points_world = (R.T @ points_cam.T).T + t
    
    ### points_world[:, 0] = -points_world[:, 0]  # Flip X-axis
        
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
        y_axis = R_cam_to_world[:, 1] * axis_length
        ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], 
                 color='g', length=axis_length, arrow_length_ratio=0.2)
        
        # Z axis (view direction) - blue
        z_axis = R_cam_to_world[:, 2] * axis_length  
        ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], 
                 color='b', length=axis_length, arrow_length_ratio=0.2)
        
        # Draw line to target point (0,0,0.1)
        ax.plot([t[0], 0], [t[1], 0], [t[2], 0.1], 
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
    Modified to handle missing segmentation results and ensure files are properly aligned.
    """
    all_3d_points = []
    all_labels = []
    all_instance_ids = []  # New array to track instance IDs
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a global instance counter for each label
    label_instance_counters = {}
    
    # Extract indices from segmentation result filenames (assuming `render_{OBJECT}_{index}_results.json`)
    view_indices = []
    for path in results_paths:
        try:
            # Example: render_myobject_123_results.json -> 123
            filename = os.path.basename(path)
            prefix = f"render_{OBJECT}_"
            suffix = "_results.json"
            if filename.startswith(prefix) and filename.endswith(suffix):
                index_str = filename[len(prefix):-len(suffix)].split('_')[0]
                view_indices.append(int(index_str))
            else:
                print(f"Warning: Filename format not recognized: {filename}, skipping index extraction.")
        except Exception as e:
            print(f"Error extracting index from {path}: {str(e)}")
            continue
    
    print(f"Extracted view indices from segmentation files: {view_indices}")
    
    # Process each view that has segmentation results
    for i, view_idx in enumerate(sorted(list(set(view_indices)))): # Ensure unique and sorted processing
        # Find the actual result path for this view_idx
        results_path_for_view = next((path for path in results_paths if f"_{view_idx}_" in path), None)
        if not results_path_for_view:
            print(f"No segmentation results found for view {view_idx}, skipping.")
            continue

        # Ensure we have corresponding camera params and depth map
        if view_idx >= len(camera_params) or camera_params[view_idx] is None:
            print(f"Missing camera parameters for view {view_idx}, skipping...")
            continue
        if view_idx >= len(depth_maps) or depth_maps[view_idx] is None:
            print(f"Missing depth map for view {view_idx}, skipping...")
            continue
            
        print(f"\nProcessing view {view_idx} (index {i+1}/{len(view_indices)} overall)")
        
        try:
            # Load masks for this view from the corresponding results file
            masks, label_names = load_masks_from_results(results_path_for_view, OBJECT, view_idx)

            label_names = [sanitize_filename(label) for label in label_names]
            
            # Skip if no masks were found
            if len(masks) == 0:
                print(f"No masks found in results for view {view_idx}, skipping...")
                continue
                
            # Get camera parameters
            K = camera_params[view_idx]['K']
            R = camera_params[view_idx]['R']
            t = camera_params[view_idx]['t']
            view_dir = camera_params[view_idx]['view_dir']
            right = camera_params[view_idx]['right']
            up = camera_params[view_idx]['up']
            
            # Get depth map
            depth_map = depth_maps[view_idx]
            
            # Print depth map stats
            if len(depth_map.shape) == 3:
                print(f"Depth map has shape {depth_map.shape} (3 channels), will use first channel")
            else:
                print(f"Depth map has shape {depth_map.shape}")
                
            valid_mask = (depth_map > 0) & (depth_map < 100000)
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
                points_3d_per_mask = backproject_masked_depth(
                    depth_map, mask, K, R, t, view_dir, right, up, max_depth=100.0, label=label_name, flip_z=flip_z
                )

                if len(points_3d_per_mask) == 0:
                    print(f"No valid 3D points for {label_name} (instance {instance_id})")
                    continue
                    
                print(f"Generated {len(points_3d_per_mask)} 3D points for {label_name} (instance {instance_id})")
                view_points.append(points_3d_per_mask)
                view_labels.extend([label_name] * len(points_3d_per_mask))
                view_instance_ids.extend([instance_id] * len(points_3d_per_mask))  # Add instance IDs

            # Visualize points for this view if any were generated
            if view_points:
                view_points = np.vstack(view_points)
                vis_path = os.path.join(output_dir, f'view_{view_idx}_points.png')
                
                # Create camera parameters for visualization (already in Blender World)
                cam_params_dict = {'R': R, 't': t}
                
                visualize_point_cloud_with_cameras(
                    view_points, np.array(view_instance_ids),  # Use instance IDs for coloring
                    [cam_params_dict], # Pass camera in Blender World
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
            import traceback
            traceback.print_exc()
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
    
    # Prepare camera parameters for visualization - only include cameras for views we processed
    all_camera_params_vis = []
    for idx in sorted(list(set(view_indices))): # Use sorted unique indices
        if idx < len(camera_params) and camera_params[idx] is not None:
            all_camera_params_vis.append({'R': camera_params[idx]['R'], 't': camera_params[idx]['t']})
            
    visualize_point_cloud_with_cameras(
        points_3d, instance_ids, all_camera_params_vis,  # Use instance IDs for visualization
        'Combined Point Cloud', 
        vis_path
    )
    
    return points_3d, labels, instance_ids  # Return instance IDs as well

# Add this new function after your existing functions
def make_masks_mutually_exclusive(masks, labels):
    """
    Make masks mutually exclusive by removing overlapping areas from larger masks.
    Smaller masks get priority over larger masks.
    
    Args:
        masks: List of binary masks (each a 2D numpy array)
        labels: List of corresponding label names
    
    Returns:
        List of modified masks that are mutually exclusive
    """
    if len(masks) <= 1:
        return masks, labels
    
    print(f"    Making {len(masks)} masks mutually exclusive...")
    
    # Convert all masks to boolean arrays to ensure consistent data types
    bool_masks = []
    for i, mask in enumerate(masks):
        if mask.dtype != bool:
            # Convert to boolean - handle both 0/1 and 0/255 cases
            bool_mask = mask > 0
        else:
            bool_mask = mask.copy()
        bool_masks.append(bool_mask)
    
    # Calculate mask sizes (number of True pixels)
    mask_sizes = [np.sum(mask) for mask in bool_masks]
    
    # Print original mask sizes
    for i, (label, size) in enumerate(zip(labels, mask_sizes)):
        print(f"      Original mask '{label}': {size} pixels")
    
    # Sort by size (smallest first gets highest priority)
    sorted_indices = np.argsort(mask_sizes)
    
    # Create copies of masks to modify
    exclusive_masks = [mask.copy() for mask in bool_masks]
    
    # Keep track of pixels that have been claimed by higher priority masks
    claimed_pixels = np.zeros_like(bool_masks[0], dtype=bool)
    
    # Process masks in order of priority (smallest first)
    for idx in sorted_indices:
        current_mask = exclusive_masks[idx]
        original_size = np.sum(current_mask)
        
        # Remove already claimed pixels from current mask
        exclusive_masks[idx] = current_mask & ~claimed_pixels
        final_size = np.sum(exclusive_masks[idx])
        
        # Add current mask pixels to claimed pixels
        claimed_pixels |= exclusive_masks[idx]
        
        removed_pixels = original_size - final_size
        if removed_pixels > 0:
            print(f"      Mask '{labels[idx]}': kept {final_size} pixels, removed {removed_pixels} overlapping pixels")
        else:
            print(f"      Mask '{labels[idx]}': kept all {final_size} pixels (no overlaps)")
    
    # Filter out masks that became empty
    filtered_masks = []
    filtered_labels = []
    
    for i, mask in enumerate(exclusive_masks):
        if np.sum(mask) > 0:  # Keep only non-empty masks
            # Convert back to the original mask format (0/1 as uint8)
            filtered_masks.append(mask.astype(np.uint8))
            filtered_labels.append(labels[i])
        else:
            print(f"      Removed empty mask '{labels[i]}' after overlap resolution")
    
    print(f"    Result: {len(filtered_masks)} non-empty mutually exclusive masks")
    return filtered_masks, filtered_labels

def load_masks_from_results(results_path, OBJECT, view):
    """Load masks from the JSON results file with better error handling and overlap resolution"""
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
        
        for i, annotation in enumerate(results['annotations']):
            try:
                rle = annotation['segmentation']
                rle['counts'] = rle['counts'].encode('utf-8')
                mask = mask_util.decode(rle)

                # Print mask statistics
                mask_sum = np.sum(mask)
                mask_shape = mask.shape
                print(f"  Decoded mask: shape={mask_shape}, nonzero={mask_sum} pixels")

                mask_vis_path = f"{'/'.join(results_path.split('/')[:-1])}/mask_{annotation['class_name']}_view_{view}.png"
                cv2.imwrite(mask_vis_path, mask * 255)
                print(f"  Saved mask visualization to {mask_vis_path}")
                
                masks.append(mask)
                labels.append(annotation['class_name'])
            except Exception as e:
                print(f"Error processing annotation: {str(e)}")
                continue
        
        if not masks:
            print(f"No valid masks could be decoded from results")
            return np.array([]), []
        
        print(f"Loaded {len(masks)} masks with labels: {labels}")
        
        # Make masks mutually exclusive (remove overlaps)
        if len(masks) > 1:
            exclusive_masks, exclusive_labels = make_masks_mutually_exclusive(masks, labels)
            
            # Save visualizations of the exclusive masks for debugging
            for i, (mask, label) in enumerate(zip(exclusive_masks, exclusive_labels)):
                exclusive_mask_vis_path = f"{'/'.join(results_path.split('/')[:-1])}/exclusive_mask_{label}_view_{view}.png"
                cv2.imwrite(exclusive_mask_vis_path, mask * 255)
                print(f"  Saved exclusive mask visualization to {exclusive_mask_vis_path}")
            
            return np.stack(exclusive_masks), exclusive_labels
        else:
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

def get_detailed_merge_scores(part1, part2, mesh):
    """
    Calculate detailed merge scores for each criterion.
    
    Returns:
        tuple: (boundary_score, continuity_score, contact_score, texture_score)
    """
    # Check if parts are too small for reliable analysis
    if len(part1) < 10 or len(part2) < 10:
        return 0.0, 0.0, 0.0, 0.5
    
    # Criterion 1: Check boundary smoothness
    try:
        boundary_score = evaluate_boundary_smoothness(part1, part2, mesh)
    except Exception as e:
        print(f"    Error in boundary score: {e}")
        boundary_score = 0.0
    
    # Criterion 2: Check for spatial continuity
    try:
        continuity_score = evaluate_spatial_continuity(part1, part2)
    except Exception as e:
        print(f"    Error in continuity score: {e}")
        continuity_score = 0.0
    
    # Criterion 3: Check for multiple contact points
    try:
        contact_score = evaluate_contact_points(part1, part2)
    except Exception as e:
        print(f"    Error in contact score: {e}")
        contact_score = 0.0
    
    # Criterion 4: Check for material/texture continuity
    try:
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            texture_score = evaluate_texture_continuity(part1, part2, mesh)
        else:
            texture_score = 0.5  # Neutral if no texture
    except Exception as e:
        print(f"    Error in texture score: {e}")
        texture_score = 0.5
    
    return boundary_score, continuity_score, contact_score, texture_score

def merge_instances_with_two_pass_approach(mesh, point_clouds_by_instance, instance_ids, point_labels,
                                          distance_threshold=0.05, consistency_threshold=0.3,
                                          merge_threshold=0.45, min_clusters=2):
    """
    Merge instances using a two-pass approach, ensuring merges only happen within same label.
    Fixed to handle empty point clouds and prevent infinite loops.
    """
    # First pass: geometric consistency
    instance_mapping, clusters = merge_instances_with_geometric_consistency(
        point_clouds_by_instance, instance_ids, distance_threshold, consistency_threshold)
    
    print(f"First pass created {len(clusters)} clusters from {len(instance_ids)} instances")
    
    # Remove any empty clusters before second pass
    clusters = [cluster for cluster in clusters if cluster]
    
    # Skip second pass if we already have min_clusters or fewer clusters
    if len(clusters) <= min_clusters or True:
        print(f"Second pass skipped - already have only {len(clusters)} clusters (minimum: {min_clusters})")
        return instance_mapping, clusters
    
    # Group clusters by label (use the label of the first instance in each cluster)
    clusters_by_label = {}
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
            
        # Get the label of this cluster (use first instance)
        rep_id = cluster[0]
        label = point_labels.get(rep_id, "unknown")
        
        if label not in clusters_by_label:
            clusters_by_label[label] = []
        clusters_by_label[label].append((i, cluster))
    
    print(f"Grouped {len(clusters)} clusters into {len(clusters_by_label)} label groups")
    
    # Second pass: Apply multi-criteria checks for merging within each label group
    print("Second pass: Applying multi-criteria checks for merging within labels...")
    
    # Track which merges have been attempted to avoid infinite loops
    attempted_merges = set()
    
    merges_happened = True
    iteration = 0
    max_iterations = 5  # Limit iterations to avoid loops
    
    while merges_happened and iteration < max_iterations:
        iteration += 1
        merges_happened = False
        
        for label, label_clusters in clusters_by_label.items():
            # Skip if only one cluster in this label
            if len(label_clusters) <= 1:
                continue
                
            print(f"\nProcessing label: {label} with {len(label_clusters)} clusters")
            
            # Get valid clusters for this label (non-empty)
            valid_label_clusters = [(idx, cluster) for idx, cluster in label_clusters 
                                   if cluster]  # Check if cluster is non-empty
            
            # Skip if only one valid cluster
            if len(valid_label_clusters) <= 1:
                continue
            
            # Compare each pair of valid clusters within this label
            i = 0
            while i < len(valid_label_clusters):
                cluster_idx_i, cluster_i = valid_label_clusters[i]
                
                j = i + 1
                while j < len(valid_label_clusters):
                    cluster_idx_j, cluster_j = valid_label_clusters[j]
                    
                    # Skip if we've already attempted this merge
                    merge_key = (cluster_idx_i, cluster_idx_j)
                    if merge_key in attempted_merges:
                        j += 1
                        continue
                        
                    # Mark this merge as attempted
                    attempted_merges.add(merge_key)
                    
                    # Get points from each cluster
                    try:
                        points1 = []
                        for id1 in cluster_i:
                            if id1 in point_clouds_by_instance and len(point_clouds_by_instance[id1]) > 0:
                                points1.append(point_clouds_by_instance[id1])
                        
                        points2 = []
                        for id2 in cluster_j:
                            if id2 in point_clouds_by_instance and len(point_clouds_by_instance[id2]) > 0:
                                points2.append(point_clouds_by_instance[id2])
                        
                        # Skip if either collection is empty
                        if not points1 or not points2:
                            print(f"  Skipping - empty point clouds: {len(points1)} and {len(points2)} arrays")
                            j += 1
                            continue
                            
                        # Now safely concatenate non-empty arrays
                        points1_combined = np.vstack(points1)
                        points2_combined = np.vstack(points2)
                        
                        # Skip if either concatenated cloud is empty
                        if len(points1_combined) == 0 or len(points2_combined) == 0:
                            print(f"  Skipping - empty combined point clouds: {len(points1_combined)} and {len(points2_combined)} points")
                            j += 1
                            continue
                            
                    except Exception as e:
                        print(f"  Error combining clusters {cluster_idx_i} and {cluster_idx_j}: {e}")
                        j += 1
                        continue
                    
                    # Check if these clusters should be merged using multi-criteria
                    try:
                        print(f"\n  Evaluating potential merge of clusters {cluster_idx_i} ({len(cluster_i)} instances) " +
                              f"and {cluster_idx_j} ({len(cluster_j)} instances)")
                        
                        # Get detailed scores and calculate merge score
                        boundary_score, continuity_score, contact_score, texture_score = get_detailed_merge_scores(
                            points1_combined, points2_combined, mesh)
                            
                        merge_score = (0.3 * boundary_score + 
                                      0.3 * continuity_score + 
                                      0.3 * contact_score + 
                                      0.1 * texture_score)
                        
                        # Print detailed scores
                        print(f"    Boundary: {boundary_score:.4f}, Continuity: {continuity_score:.4f}, " + 
                              f"Contact: {contact_score:.4f}, Texture: {texture_score:.4f}")
                        print(f"    Final merge score: {merge_score:.4f} (threshold: {merge_threshold:.2f})")
                        
                        # Check if merging would reduce us below min_clusters
                        total_valid_clusters = sum(len([c for c in cl if c[1]]) 
                                                  for cl in clusters_by_label.values())
                        would_violate_min = (total_valid_clusters - 1) < min_clusters
                        
                        # Use threshold for merging decision
                        should_merge = merge_score > merge_threshold and not would_violate_min
                        print(f"    Decision: {'MERGE' if should_merge else 'DO NOT MERGE'}")
                        
                        if should_merge:
                            print(f"  Multi-criteria merge: clusters {cluster_idx_i} and {cluster_idx_j}")
                            
                            # Merge clusters: add j's instances to i
                            clusters[cluster_idx_i].extend(clusters[cluster_idx_j])
                            clusters[cluster_idx_j] = []  # Empty cluster j
                            
                            # Also update the local valid_label_clusters list
                            # Find and update cluster_j in valid_label_clusters
                            for k in range(len(valid_label_clusters)):
                                if valid_label_clusters[k][0] == cluster_idx_j:
                                    valid_label_clusters[k] = (cluster_idx_j, [])
                                    break
                                    
                            # Update the local cluster_i reference
                            cluster_i = clusters[cluster_idx_i]
                            valid_label_clusters[i] = (cluster_idx_i, cluster_i)
                            
                            # Update instance mappings
                            rep_id = clusters[cluster_idx_i][0]  # Representative ID for the merged cluster
                            for instance_id in clusters[cluster_idx_i]:
                                instance_mapping[instance_id] = rep_id
                            
                            merges_happened = True
                            
                            # Don't increment j, just recheck with updated clusters
                        else:
                            # Only increment j if no merge happened
                            j += 1
                    except Exception as e:
                        print(f"  Error evaluating merge for clusters {cluster_idx_i} and {cluster_idx_j}: {e}")
                        import traceback
                        traceback.print_exc()
                        j += 1
                        continue
                
                # Move to the next i
                i += 1
        
        # Print status at the end of each iteration
        valid_cluster_count = len([c for c in clusters if c])
        print(f"  Iteration {iteration}: {valid_cluster_count} clusters remain")
        
        # Stop if we didn't make any merges this iteration
        if not merges_happened:
            print("  No more merges detected, stopping iterations")
            break
    
    # Final cleanup: remove empty clusters
    final_clusters = [cluster for cluster in clusters if cluster]
    
    print(f"Two-pass merging complete: {len(final_clusters)} clusters from {len(instance_ids)} instances")
    
    return instance_mapping, final_clusters


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

def compute_mesh_curvature(mesh):
    """Compute approximate curvature for mesh vertices."""
    import numpy as np
    
    # Initialize curvature array
    curvature = np.zeros(len(mesh.vertices))
    
    # Build vertex-to-face mapping
    vertex_faces = [[] for _ in range(len(mesh.vertices))]
    for i, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_faces[vertex].append(i)
    
    # Compute curvature for each vertex
    for i in range(len(mesh.vertices)):
        # Get adjacent faces
        adj_faces = vertex_faces[i]
        
        if adj_faces:
            # Get normals of adjacent faces
            normals = mesh.face_normals[adj_faces]
            
            # Calculate variance of normals as a measure of curvature
            if len(normals) > 1:
                mean_normal = np.mean(normals, axis=0)
                mean_normal = mean_normal / np.linalg.norm(mean_normal)
                
                # Calculate angular deviations
                deviations = np.array([1.0 - np.dot(n, mean_normal) for n in normals])
                curvature[i] = np.mean(deviations)
            
    return curvature

def detect_structural_boundaries(mesh, curvature, threshold=0.3):
    """Detect structural boundaries in the mesh based on curvature and normals."""
    import numpy as np
    
    # Initialize boundary flags
    boundaries = np.zeros(len(mesh.vertices), dtype=bool)
    
    # Mark high curvature points as boundaries
    boundaries = curvature > threshold
    
    # Also look for normal discontinuities
    for face in mesh.faces:
        v1, v2, v3 = face
        n1, n2, n3 = mesh.vertex_normals[face]
        
        # Check if any pair of normals has a large angle between them
        angle12 = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
        angle23 = np.arccos(np.clip(np.dot(n2, n3), -1.0, 1.0))
        angle31 = np.arccos(np.clip(np.dot(n3, n1), -1.0, 1.0))
        
        if angle12 > 0.5 or angle23 > 0.5 or angle31 > 0.5:  # ~30 degrees
            boundaries[v1] = True
            boundaries[v2] = True
            boundaries[v3] = True
    
    return boundaries

def watershed_from_boundaries(mesh, vertex_labels, unlabeled, boundaries, graph):
    """
    Use watershed algorithm to propagate labels from labeled regions,
    respecting structural boundaries.
    """
    import numpy as np
    import heapq
    
    # Initialize result
    result = {}
    
    # Create a priority queue for watershed propagation
    queue = []
    
    # Find boundary between labeled and unlabeled regions
    for u in range(len(vertex_labels)):
        if vertex_labels[u] is not None and any(
            vertex_labels[v] is None for v in graph.neighbors(u)):
            
            # Calculate priority based on structural boundary strength
            # (lower priority for boundary vertices)
            priority = 1.0 if boundaries[u] else 0.0
            
            # Add to queue: (priority, vertex_id, label)
            heapq.heappush(queue, (priority, u, vertex_labels[u]))
    
    # Process queue until empty
    visited = set()
    
    while queue:
        _, u, label = heapq.heappop(queue)
        
        if u in visited:
            continue
            
        visited.add(u)
        
        # Process neighbors
        for v in graph.neighbors(u):
            if v in unlabeled and v not in visited:
                # Propagate label
                result[v] = label
                
                # Calculate priority for propagation
                priority = 1.0 if boundaries[v] else 0.0
                
                # Add neighbor to queue
                heapq.heappush(queue, (priority, v, label))
    
    return result

def smooth_label_assignments(mesh, vertex_labels, graph, confidence, iterations=3):
    """
    Smooth label assignments while respecting structural boundaries.
    """
    import numpy as np
    from collections import Counter
    
    for _ in range(iterations):
        new_labels = vertex_labels.copy()
        
        for i in range(len(vertex_labels)):
            # Only smooth vertices with low confidence
            if confidence[i] > 0.8:
                continue
                
            # Get neighbors
            neighbors = list(graph.neighbors(i))
            
            if neighbors:
                # Calculate weighted vote for each label
                label_votes = Counter()
                
                for neighbor in neighbors:
                    # Get structural similarity weight
                    weight = graph[i][neighbor]['weight']
                    
                    # Add weighted vote
                    label = vertex_labels[neighbor]
                    if label is not None:
                        label_votes[label] += weight
                
                if label_votes:
                    # Get label with highest weighted votes
                    best_label = label_votes.most_common(1)[0][0]
                    
                    # Only change if different and has strong support
                    if best_label != vertex_labels[i] and label_votes[best_label] > 0.6 * sum(label_votes.values()):
                        new_labels[i] = best_label
        
        vertex_labels = new_labels
    
    return vertex_labels

def analyze_part_structure(segmented_points, point_labels, unique_labels):
    """
    Analyze the structural properties of each part.
    Returns a dictionary with structural information for each part.
    """
    part_structures = {}
    
    for label in unique_labels:
        # Get points for this part
        part_points = segmented_points[point_labels == label]
        
        if len(part_points) < 10:
            # Not enough points for analysis
            part_structures[label] = {
                'planarity': 0.5,  # Neutral values
                'linearity': 0.5,
                'sphericity': 0.5,
                'is_planar': False,
                'is_cylindrical': False
            }
            continue
        
        # PCA to analyze the structure
        centered = part_points - np.mean(part_points, axis=0)
        cov = np.cov(centered, rowvar=False)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Normalize eigenvalues to sum to 1
            eigenvalues = eigenvalues / np.sum(eigenvalues)
            
            # Calculate shape descriptors
            # - If first eigenvalue dominates: points lie on a line
            # - If first two dominate: points lie on a plane
            # - If all three similar: points form a 3D cloud
            planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
            linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
            sphericity = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
            
            # Classify the structure
            is_planar = planarity > 0.5 and eigenvalues[2] < 0.05
            is_cylindrical = linearity > 0.3 and planarity > 0.3
            
            # Store results
            part_structures[label] = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'planarity': planarity,
                'linearity': linearity,
                'sphericity': sphericity,
                'is_planar': is_planar,
                'is_cylindrical': is_cylindrical
            }
            
            print(f"Part {label} structure: " + 
                 f"planarity={planarity:.2f}, linearity={linearity:.2f}, sphericity={sphericity:.2f}")
            
        except np.linalg.LinAlgError:
            # Fallback for numerical issues
            part_structures[label] = {
                'planarity': 0.5,
                'linearity': 0.5,
                'sphericity': 0.5,
                'is_planar': False,
                'is_cylindrical': False
            }
    
    return part_structures

def enforce_structural_constraints(mesh, vertex_labels, part_structures):
    """
    Enforce structural constraints based on part analysis.
    Fixed to work with pre-computed vertex neighbors.
    """
    import numpy as np
    
    print("Enforcing structural constraints...")
    
    # Create vertex neighborhood list if not already done
    if not hasattr(mesh, 'vertex_neighbors_list'):
        print("  Building vertex neighborhood list...")
        mesh.vertex_neighbors_list = [set() for _ in range(len(mesh.vertices))]
        for face in mesh.faces:
            mesh.vertex_neighbors_list[face[0]].add(face[1])
            mesh.vertex_neighbors_list[face[0]].add(face[2])
            mesh.vertex_neighbors_list[face[1]].add(face[0])
            mesh.vertex_neighbors_list[face[1]].add(face[2])
            mesh.vertex_neighbors_list[face[2]].add(face[0])
            mesh.vertex_neighbors_list[face[2]].add(face[1])
    
    # Iterate through each part
    for label, structure in part_structures.items():
        # Get vertices assigned to this part
        part_vertices = np.where(vertex_labels == label)[0]
        
        if len(part_vertices) == 0:
            continue
            
        # For planar parts: check if vertices deviate too much from the plane
        if structure.get('is_planar', False) and len(part_vertices) > 10:
            # Get the plane normal (direction of smallest variance)
            plane_normal = structure['eigenvectors'][:, 2]
            
            # Get centroid of the part
            part_positions = mesh.vertices[part_vertices]
            centroid = np.mean(part_positions, axis=0)
            
            # Calculate distance from each vertex to the plane
            distances = np.abs(np.dot(part_positions - centroid, plane_normal))
            
            # Find outliers that deviate significantly from the plane
            threshold = np.percentile(distances, 95) * 2  # 2x the 95th percentile
            outliers = part_vertices[distances > threshold]
            
            # Mark these outliers for reconsideration
            for idx in outliers:
                # Get neighboring labels (excluding this part)
                neighbor_labels = []
                for n in mesh.vertex_neighbors_list[idx]:
                    if vertex_labels[n] != label:
                        neighbor_labels.append(vertex_labels[n])
                
                if neighbor_labels:
                    # Assign to most common neighboring label
                    from collections import Counter
                    vertex_labels[idx] = Counter(neighbor_labels).most_common(1)[0][0]
        
        # For cylindrical parts: similar constraint based on distance to axis
        if structure.get('is_cylindrical', False) and len(part_vertices) > 20:
            # Get the cylinder axis (direction of largest variance)
            axis = structure['eigenvectors'][:, 0]
            
            # Get centroid of the part
            part_positions = mesh.vertices[part_vertices]
            centroid = np.mean(part_positions, axis=0)
            
            # Calculate distance from each point to the axis
            # d = |cross(point-centroid, axis)|/|axis|
            v = part_positions - centroid
            cross_products = np.cross(v, axis)
            distances = np.linalg.norm(cross_products, axis=1) / np.linalg.norm(axis)
            
            # Find outliers that deviate significantly from the cylinder
            threshold = np.percentile(distances, 95) * 2  # 2x the 95th percentile
            outliers = part_vertices[distances > threshold]
            
            # Mark these outliers for reconsideration
            for idx in outliers:
                # Get neighboring labels (excluding this part)
                neighbor_labels = []
                for n in mesh.vertex_neighbors_list[idx]:
                    if vertex_labels[n] != label:
                        neighbor_labels.append(vertex_labels[n])
                
                if neighbor_labels:
                    # Assign to most common neighboring label
                    from collections import Counter
                    vertex_labels[idx] = Counter(neighbor_labels).most_common(1)[0][0]
    
    return vertex_labels

def segment_mesh_structure_aware(mesh, segmented_points, point_labels, unique_labels, max_distance=0.05):
    """
    Optimized version of structure-aware mesh segmentation.
    Fixed to work with pre-computed vertex neighbors.
    """
    import numpy as np
    import networkx as nx
    from scipy.spatial import cKDTree
    from collections import Counter
    
    print("Performing optimized structure-aware mesh segmentation...")
    
    # Step 1: Calculate simplified mesh features (just normals and basic curvature)
    print("  Computing basic geometric features...")
    
    # Ensure we have vertex normals
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.vertex_normals = np.zeros((len(mesh.vertices), 3))
        # Approximate normals by averaging face normals
        vertex_faces = [[] for _ in range(len(mesh.vertices))]
        for i, face in enumerate(mesh.faces):
            for v in face:
                vertex_faces[v].append(i)
        
        for i in range(len(mesh.vertices)):
            if vertex_faces[i]:
                mesh.vertex_normals[i] = np.mean(mesh.face_normals[vertex_faces[i]], axis=0)
                if np.linalg.norm(mesh.vertex_normals[i]) > 0:
                    mesh.vertex_normals[i] /= np.linalg.norm(mesh.vertex_normals[i])
    
    # Step 2: Build mesh connectivity graph early (for faster access)
    print("  Building mesh connectivity...")
    if not hasattr(mesh, 'vertex_neighbors_list'):
        mesh.vertex_neighbors_list = [set() for _ in range(len(mesh.vertices))]
        for face in mesh.faces:
            mesh.vertex_neighbors_list[face[0]].add(face[1])
            mesh.vertex_neighbors_list[face[0]].add(face[2])
            mesh.vertex_neighbors_list[face[1]].add(face[0])
            mesh.vertex_neighbors_list[face[1]].add(face[2])
            mesh.vertex_neighbors_list[face[2]].add(face[0])
            mesh.vertex_neighbors_list[face[2]].add(face[1])
    
    # Step 3: Initial label assignment based on proximity
    print("  Performing initial proximity-based assignment...")
    kdtree = cKDTree(segmented_points)
    distances, indices = kdtree.query(mesh.vertices, k=1)
    
    # Get initial assignments for vertices close enough to point cloud
    vertex_labels = np.full(len(mesh.vertices), "", dtype=object)
    confidence = np.zeros(len(mesh.vertices))
    
    close_enough = distances < max_distance
    vertex_labels[close_enough] = [point_labels[idx] for idx in indices[close_enough]]
    confidence[close_enough] = 1.0 - (distances[close_enough] / max_distance)
    
    print(f"    Initial assignment: {np.sum(close_enough)} vertices assigned directly")
    
    # Step 4: Fast label propagation with structural guidance
    print("  Propagating labels...")
    unlabeled = np.where(vertex_labels == "")[0]
    iterations = 0
    max_iterations = 20  # Limit iterations
    
    while len(unlabeled) > 0 and iterations < max_iterations:
        print(f"    Iteration {iterations+1}: {len(unlabeled)} unlabeled vertices")
        newly_labeled = []
        
        for vertex_idx in unlabeled:
            # Get labels of neighbors
            neighbor_labels = [
                vertex_labels[n] for n in mesh.vertex_neighbors_list[vertex_idx]
                if vertex_labels[n] != ""
            ]
            
            if neighbor_labels:
                # Use most common neighbor label
                label_counts = Counter(neighbor_labels)
                most_common = label_counts.most_common(1)[0]
                
                # Only assign if there's reasonable consensus
                if most_common[1] >= len(neighbor_labels) * 0.4:  # At least 40% agreement
                    vertex_labels[vertex_idx] = most_common[0]
                    confidence[vertex_idx] = 0.7  # Moderate confidence for propagated labels
                    newly_labeled.append(vertex_idx)
        
        if not newly_labeled:
            # No progress made in this iteration
            break
            
        unlabeled = np.setdiff1d(unlabeled, newly_labeled)
        iterations += 1
    
    # Step 5: Assign any remaining unlabeled vertices
    still_unlabeled = np.where(vertex_labels == "")[0]
    if len(still_unlabeled) > 0:
        print(f"  Assigning {len(still_unlabeled)} remaining vertices...")
        
        # Find closest labeled vertex for each unlabeled one
        for vertex_idx in still_unlabeled:
            # Use BFS to find closest labeled vertex
            queue = list(mesh.vertex_neighbors_list[vertex_idx])
            visited = set([vertex_idx])
            found_label = None
            
            while queue and found_label is None:
                neighbor = queue.pop(0)
                if vertex_labels[neighbor] != "":
                    found_label = vertex_labels[neighbor]
                    break
                
                visited.add(neighbor)
                for next_neighbor in mesh.vertex_neighbors_list[neighbor]:
                    if next_neighbor not in visited and next_neighbor not in queue:
                        queue.append(next_neighbor)
            
            if found_label is not None:
                vertex_labels[vertex_idx] = found_label
                confidence[vertex_idx] = 0.5  # Lower confidence for distant assignments
            else:
                # If BFS fails (disconnected component), use nearest labeled vertex in 3D space
                labeled_vertices = np.where(vertex_labels != "")[0]
                tree = cKDTree(mesh.vertices[labeled_vertices])
                _, nn_idx = tree.query(mesh.vertices[vertex_idx].reshape(1, -1))
                vertex_labels[vertex_idx] = vertex_labels[labeled_vertices[nn_idx[0]]]
                confidence[vertex_idx] = 0.3  # Even lower confidence
    
    # Step 6: Simple smoothing pass
    print("  Smoothing assignments...")
    for _ in range(3):  # 3 iterations of smoothing
        for i in range(len(mesh.vertices)):
            # Only smooth low-confidence vertices
            if confidence[i] > 0.8:
                continue
                
            # Get neighbor labels
            neighbor_labels = [
                vertex_labels[n] for n in mesh.vertex_neighbors_list[i]
            ]
            
            if neighbor_labels:
                # Get most common label
                label_counts = Counter(neighbor_labels)
                most_common = label_counts.most_common(1)[0][0]
                
                # If most neighbors have a different label, change this one
                if most_common != vertex_labels[i] and label_counts[most_common] > len(neighbor_labels)/2:
                    vertex_labels[i] = most_common
    
    print("Segmentation complete!")
    return vertex_labels

def process_mesh_segmentation(mesh, vertex_labels, segmented_points, combined_identifiers):
    """
    Process mesh segmentation to ensure quality results for any object type.
    This function:
    1. Enforces that each part has connected components
    2. Ensures all vertices are assigned
    3. Maintains clear boundaries between parts
    
    Args:
        mesh: Trimesh mesh object
        vertex_labels: Array of initial vertex labels
        segmented_points: Nx3 array of original point cloud points
        combined_identifiers: Labels for each point in segmented_points
        
    Returns:
        Updated vertex_labels with improved segmentation
    """
    import numpy as np
    from collections import deque
    from scipy.spatial import cKDTree
    
    print("Processing mesh segmentation for optimal results...")
    
    # Get unique labels
    unique_labels = np.unique(vertex_labels)
    unique_labels = [label for label in unique_labels if label != ""]
    
    if len(unique_labels) == 0:
        print("  No valid labels found")
        return vertex_labels
    
    # Step 1: Identify connected components for each label
    print("  Identifying connected components...")
    label_components = {}
    
    for label in unique_labels:
        vertices_with_label = np.where(vertex_labels == label)[0]
        
        if len(vertices_with_label) == 0:
            continue
        
        # Find connected components
        components = []
        unvisited = set(vertices_with_label)
        
        while unvisited:
            start_vertex = next(iter(unvisited))
            component = set()
            queue = deque([start_vertex])
            
            while queue:
                vertex = queue.popleft()
                if vertex not in unvisited:
                    continue
                
                component.add(vertex)
                unvisited.remove(vertex)
                
                for neighbor in mesh.vertex_neighbors_list[vertex]:
                    if neighbor in unvisited:
                        queue.append(neighbor)
            
            components.append(component)
        
        label_components[label] = components
        
        print(f"    Label {label}: {len(vertices_with_label)} vertices in {len(components)} components")
    
    # Step 2: Evaluate component quality using point cloud
    print("  Evaluating component quality...")
    point_cloud_tree = cKDTree(segmented_points)
    
    component_quality = {}
    
    for label, components in label_components.items():
        component_quality[label] = []
        
        for comp_idx, component in enumerate(components):
            # Skip very small components
            if len(component) < 10:
                component_quality[label].append({
                    'index': comp_idx,
                    'quality': 0.0,
                    'size': len(component)
                })
                continue
            
            # Get vertices in this component
            component_vertices = list(component)
            component_positions = mesh.vertices[component_vertices]
            
            # Find nearest points in the point cloud
            distances, indices = point_cloud_tree.query(component_positions, k=1)
            
            # Calculate match quality
            match_mask = combined_identifiers[indices] == label
            match_count = np.sum(match_mask)
            match_quality = match_count / len(component)
            
            component_quality[label].append({
                'index': comp_idx,
                'quality': match_quality,
                'size': len(component),
                'match_count': match_count
            })
    
    # Step 3: Merge components for each label
    print("  Merging components...")
    new_vertex_labels = np.full(len(mesh.vertices), "", dtype=object)
    
    for label in unique_labels:
        if label not in component_quality or len(component_quality[label]) == 0:
            continue
            
        # Get all components with reasonable quality for this label
        good_components = [c for c in component_quality[label] if c['quality'] >= 0.3]
        
        if good_components:
            # Sort by size for stable results (largest first)
            good_components.sort(key=lambda x: x['size'], reverse=True)
            
            print(f"    Label {label}: Keeping {len(good_components)} quality components")
            
            # Assign all vertices in good components to this label
            for comp in good_components:
                comp_idx = comp['index']
                component = label_components[label][comp_idx]
                
                for vertex in component:
                    new_vertex_labels[vertex] = label
        else:
            # If no good components, fall back to largest component
            components = label_components[label]
            if components:
                largest_comp = max(components, key=len)
                print(f"    Label {label}: No good components, using largest ({len(largest_comp)} vertices)")
                
                for vertex in largest_comp:
                    new_vertex_labels[vertex] = label
    
    # Step 4: Check for unlabeled vertices and assign them to nearest labeled vertex
    unlabeled = np.where(new_vertex_labels == "")[0]
    if len(unlabeled) > 0:
        print(f"  Assigning {len(unlabeled)} unlabeled vertices...")
        
        # Get labeled vertices
        labeled = np.where(new_vertex_labels != "")[0]
        
        if len(labeled) > 0:
            # Build graph and flood-fill from nearest labeled vertex
            for vertex in unlabeled:
                # First check direct neighbors
                neighbors = mesh.vertex_neighbors_list[vertex]
                neighbor_labels = {}
                
                for n in neighbors:
                    nl = new_vertex_labels[n]
                    if nl != "":
                        if nl not in neighbor_labels:
                            neighbor_labels[nl] = 0
                        neighbor_labels[nl] += 1
                
                if neighbor_labels:
                    # Assign to most common neighbor label
                    new_vertex_labels[vertex] = max(neighbor_labels.items(), key=lambda x: x[1])[0]
                else:
                    # No labeled neighbors, use BFS to find closest labeled vertex
                    visited = set([vertex])
                    queue = deque(neighbors)
                    found_label = None
                    
                    while queue and found_label is None:
                        next_vertex = queue.popleft()
                        if next_vertex in visited:
                            continue
                            
                        visited.add(next_vertex)
                        
                        if new_vertex_labels[next_vertex] != "":
                            found_label = new_vertex_labels[next_vertex]
                            break
                        
                        for n in mesh.vertex_neighbors_list[next_vertex]:
                            if n not in visited and n not in queue:
                                queue.append(n)
                    
                    if found_label is not None:
                        new_vertex_labels[vertex] = found_label
                    else:
                        # BFS failed - use closest labeled vertex in 3D space
                        kdtree = cKDTree(mesh.vertices[labeled])
                        _, nn_idx = kdtree.query(mesh.vertices[vertex].reshape(1, -1))
                        new_vertex_labels[vertex] = new_vertex_labels[labeled[nn_idx[0]]]
    
    # Step 5: Smooth boundaries between parts
    boundary_vertices = set()
    for vertex in range(len(mesh.vertices)):
        label = new_vertex_labels[vertex]
        for neighbor in mesh.vertex_neighbors_list[vertex]:
            if new_vertex_labels[neighbor] != label:
                boundary_vertices.add(vertex)
                break
    
    if boundary_vertices:
        print(f"  Smoothing {len(boundary_vertices)} boundary vertices...")
        
        # Simple smoothing - assign each boundary vertex to the most common label among neighbors
        for _ in range(2):  # 2 iterations of smoothing
            updates = {}
            
            for vertex in boundary_vertices:
                neighbor_labels = {}
                for n in mesh.vertex_neighbors_list[vertex]:
                    nl = new_vertex_labels[n]
                    if nl not in neighbor_labels:
                        neighbor_labels[nl] = 0
                    neighbor_labels[nl] += 1
                
                if neighbor_labels:
                    most_common = max(neighbor_labels.items(), key=lambda x: x[1])[0]
                    if most_common != new_vertex_labels[vertex]:
                        updates[vertex] = most_common
            
            # Apply updates
            for vertex, new_label in updates.items():
                new_vertex_labels[vertex] = new_label
            
            if not updates:
                break
    
    # Check final result
    final_unlabeled = np.sum(new_vertex_labels == "")
    if final_unlabeled > 0:
        print(f"  WARNING: {final_unlabeled} vertices remain unlabeled")
    else:
        print("  All vertices successfully labeled")
    
    return new_vertex_labels

def calculate_merge_score(part1, part2, mesh):
    """
    Calculate the merge score between two parts (without the debugging output).
    """
    # Check if parts are too small for reliable analysis
    if len(part1) < 10 or len(part2) < 10:
        return 0.0
    
    # Criterion 1: Check boundary smoothness
    try:
        boundary_score = evaluate_boundary_smoothness(part1, part2, mesh)
    except Exception:
        boundary_score = 0.0
    
    # Criterion 2: Check for spatial continuity
    try:
        continuity_score = evaluate_spatial_continuity(part1, part2)
    except Exception:
        continuity_score = 0.0
    
    # Criterion 3: Check for multiple contact points
    try:
        contact_score = evaluate_contact_points(part1, part2)
    except Exception:
        contact_score = 0.0
    
    # Criterion 4: Check for material/texture continuity
    try:
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            texture_score = evaluate_texture_continuity(part1, part2, mesh)
        else:
            texture_score = 0.5  # Neutral if no texture
    except Exception:
        texture_score = 0.5
    
    # Weighted decision based on all criteria
    merge_score = (
        0.3 * boundary_score + 
        0.3 * continuity_score + 
        0.3 * contact_score + 
        0.1 * texture_score
    )
    
    return merge_score

def should_merge_parts(part1, part2, mesh, debug=True):
    """
    Determine if two parts should be merged based on multiple criteria.
    
    Args:
        part1, part2: Point clouds for the two parts
        mesh: The full mesh
        debug: Whether to print detailed diagnostics
        
    Returns:
        bool: True if parts should be merged
    """
    # Check if parts are too small for reliable analysis
    if len(part1) < 10 or len(part2) < 10:
        if debug:
            print(f"  [Merge Check] Skipping - parts too small: {len(part1)} and {len(part2)} points")
        return False
    
    # For logging, get maximum distances within each part
    # This helps understand the scale of the parts
    from scipy.spatial import cKDTree
    tree1 = cKDTree(part1)
    dist1, _ = tree1.query(part1, k=2)  # k=2 to skip self
    max_dist1 = np.max(dist1[:, 1]) if len(dist1) > 0 else 0
    
    tree2 = cKDTree(part2)
    dist2, _ = tree2.query(part2, k=2)
    max_dist2 = np.max(dist2[:, 1]) if len(dist2) > 0 else 0
    
    if debug:
        print(f"\n[Merge Check] Evaluating merge: {len(part1)} points & {len(part2)} points")
        print(f"  Part 1 max internal dist: {max_dist1:.4f}, Part 2: {max_dist2:.4f}")
    
    merge_score = calculate_merge_score(part1, part2, mesh)
    
    # Lower the threshold for testing
    threshold = 0.45  # More permissive threshold for debugging
    
    should_merge = merge_score > threshold
    
    if debug:
        print(f"  Final merge score: {merge_score:.4f} (threshold: {threshold:.2f})")
        print(f"  Decision: {'MERGE' if should_merge else 'DO NOT MERGE'}")
    
    return should_merge

def compute_boundary_irregularity(part1, part2, distance_threshold=0.05):
    """
    Compute how irregular the boundary between two parts is.
    
    Args:
        part1: First point cloud (Nx3 array)
        part2: Second point cloud (Mx3 array)
        distance_threshold: Distance threshold for boundary detection
    
    Returns:
        float: 0-1 score, where higher means more irregular (likely same part)
    """
    import numpy as np
    from scipy.spatial import cKDTree
    
    # Check for empty point clouds
    if len(part1) < 10 or len(part2) < 10:
        return 0.0
    
    # Build KD-trees for both parts
    tree1 = cKDTree(part1)
    tree2 = cKDTree(part2)
    
    # Find points in part1 that are close to part2
    dist1, idx1 = tree1.query(part2, k=1)
    boundary_mask2 = dist1 < distance_threshold
    boundary_points2 = part2[boundary_mask2]
    
    # Find points in part2 that are close to part1
    dist2, idx2 = tree2.query(part1, k=1)
    boundary_mask1 = dist2 < distance_threshold
    boundary_points1 = part1[boundary_mask1]
    
    # Combine boundary points
    if len(boundary_points1) < 10 or len(boundary_points2) < 10:
        return 0.0  # Not enough boundary points
    
    boundary_points = np.vstack([boundary_points1, boundary_points2])
    
    # Calculate boundary shape characteristics
    
    # 1. Fit a plane to the boundary points using PCA
    centroid = np.mean(boundary_points, axis=0)
    centered = boundary_points - centroid
    
    # Compute covariance matrix and its eigendecomposition
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 2. Calculate metrics for the boundary shape
    
    # Planarity: If boundary is perfectly planar, smallest eigenvalue will be close to 0
    # Higher value means less planar (more irregular boundary)
    if np.sum(eigenvalues) > 0:
        planarity = eigenvalues[2] / np.sum(eigenvalues)  # Ratio of smallest eigenvalue
    else:
        planarity = 0.0
    
    # 3. Measure the variation of points from the fitted plane
    # Project points onto the plane defined by first two eigenvectors
    projection_matrix = eigenvectors[:, :2]
    projected = np.dot(centered, projection_matrix)
    
    # Convert back to 3D by adding third component as 0
    projected_3d = np.dot(projected, projection_matrix.T)
    
    # Calculate residuals (distance from original points to plane projection)
    residuals = np.linalg.norm(centered - projected_3d, axis=1)
    
    # Calculate standard deviation of residuals
    # Higher std dev means more irregular boundary
    residual_std = np.std(residuals)
    
    # 4. Calculate boundary curvature by looking at local neighborhoods
    curvature_values = []
    
    # Sample boundary points if there are too many
    sample_size = min(100, len(boundary_points))
    sample_indices = np.random.choice(len(boundary_points), sample_size, replace=False)
    sample_points = boundary_points[sample_indices]
    
    # For each sample point, calculate local curvature
    for point in sample_points:
        # Find nearest neighbors
        local_tree = cKDTree(boundary_points)
        dists, _ = local_tree.query(point, k=min(10, len(boundary_points)))
        
        # Local curvature based on variation in distances
        if len(dists) > 3:  # Need at least a few points
            local_curve = np.std(dists[1:]) / np.mean(dists[1:])  # Skip first point (self)
            curvature_values.append(local_curve)
    
    # Calculate average curvature
    avg_curvature = np.mean(curvature_values) if curvature_values else 0.0
    
    # 5. Combine metrics to get final irregularity score
    # Scale each metric to roughly 0-1 range
    planarity_score = min(1.0, planarity * 20)  # Scale up for sensitivity
    residual_score = min(1.0, residual_std / distance_threshold)
    curvature_score = min(1.0, avg_curvature * 5)  # Scale up for sensitivity
    
    # Weighted combination of metrics
    irregularity = (0.4 * planarity_score + 
                    0.4 * residual_score + 
                    0.2 * curvature_score)
    
    return irregularity

def evaluate_spatial_continuity(part1, part2, distance_threshold=0.05):
    """
    Evaluate if two parts form a continuous surface based on proximity.
    
    Args:
        part1: First point cloud (Nx3 array)
        part2: Second point cloud (Mx3 array)
        distance_threshold: Distance threshold for proximity evaluation
        
    Returns:
        float: 0-1 score, higher means more continuous
    """
    import numpy as np
    from scipy.spatial import cKDTree
    
    # Check for empty point clouds
    if len(part1) < 10 or len(part2) < 10:
        return 0.0
    
    # Build KD-trees for both parts
    tree1 = cKDTree(part1)
    tree2 = cKDTree(part2)
    
    # Find closest points between parts
    dist1, _ = tree1.query(part2, k=1)
    dist2, _ = tree2.query(part1, k=1)
    
    # Calculate percentage of points that are close to the other part
    close_ratio1 = np.mean(dist1 < distance_threshold)
    close_ratio2 = np.mean(dist2 < distance_threshold)
    
    # Take the average of both ratios as the continuity score
    continuity = (close_ratio1 + close_ratio2) / 2.0
    
    return float(continuity)

def evaluate_contact_points(part1, part2, distance_threshold=0.05):
    """
    Evaluate if parts have multiple separated contact regions.
    Multiple contact points suggest parts of the same object.
    
    Returns:
        float: 0-1 score, higher means more likely to be same part
    """
    # Find all points in part1 close to part2
    tree2 = cKDTree(part2)
    dist, _ = tree2.query(part1, distance_upper_bound=distance_threshold)
    contact_points = part1[np.isfinite(dist)]
    
    if len(contact_points) < 10:
        return 0.0  # Not enough contact
    
    # Cluster contact points to find separate contact regions
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=distance_threshold*2, min_samples=5).fit(contact_points)
    
    # Count number of significant clusters
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # More contact regions suggests parts should be merged
    score = min(1.0, n_clusters / 2.0)  # Normalize (2+ contacts -> high score)
    
    return score

def evaluate_texture_continuity(part1, part2, mesh, sampling_rate=0.1, distance_threshold=0.05):
    """
    Evaluate if textures are continuous across boundaries between two parts.
    
    Args:
        part1: First point cloud (Nx3 array)
        part2: Second point cloud (Mx3 array)
        mesh: The mesh object with texture information
        sampling_rate: Rate to sample points for analysis
        distance_threshold: Distance threshold for boundary detection
    
    Returns:
        float: 0-1 score, higher means more continuous texture
    """
    import numpy as np
    from scipy.spatial import cKDTree
    
    # Check if mesh has texture information
    has_texture = (hasattr(mesh, 'visual') and 
                  hasattr(mesh.visual, 'uv') and 
                  mesh.visual.uv is not None)
    
    if not has_texture:
        return 0.5  # Neutral score if no texture
    
    # Check for empty point clouds
    if len(part1) < 10 or len(part2) < 10:
        return 0.5
    
    try:
        # Find boundary points between parts
        tree1 = cKDTree(part1)
        tree2 = cKDTree(part2)
        
        # Find points in part1 close to part2
        dist1, idx1 = tree1.query(part2, k=1)
        boundary_mask2 = dist1 < distance_threshold
        boundary_points2 = part2[boundary_mask2]
        boundary_src2 = part1[idx1[boundary_mask2]]  # Corresponding points in part1
        
        # Find points in part2 close to part1
        dist2, idx2 = tree2.query(part1, k=1)
        boundary_mask1 = dist2 < distance_threshold
        boundary_points1 = part1[boundary_mask1]
        boundary_src1 = part2[idx2[boundary_mask1]]  # Corresponding points in part2
        
        # Sample boundary points if there are too many
        max_samples = 100
        
        if len(boundary_points1) > max_samples:
            indices = np.random.choice(len(boundary_points1), max_samples, replace=False)
            boundary_points1 = boundary_points1[indices]
            boundary_src1 = boundary_src1[indices]
            
        if len(boundary_points2) > max_samples:
            indices = np.random.choice(len(boundary_points2), max_samples, replace=False)
            boundary_points2 = boundary_points2[indices]
            boundary_src2 = boundary_src2[indices]
        
        # Combine all boundary point pairs
        p1_points = np.vstack([boundary_points1, boundary_src2])
        p2_points = np.vstack([boundary_src1, boundary_points2])
        
        # For each boundary point pair, find texture differences
        texture_diffs = []
        
        # Find the nearest mesh vertices for each boundary point
        mesh_tree = cKDTree(mesh.vertices)
        p1_dists, p1_vertex_idx = mesh_tree.query(p1_points, k=1)
        p2_dists, p2_vertex_idx = mesh_tree.query(p2_points, k=1)
        
        # Get UV coordinates for these vertices
        uv_diffs = []
        
        # Process vertex pairs only if they are close enough to the mesh
        valid_mask = (p1_dists < distance_threshold) & (p2_dists < distance_threshold)
        p1_vertex_idx = p1_vertex_idx[valid_mask]
        p2_vertex_idx = p2_vertex_idx[valid_mask]
        
        # Check if we have enough valid points
        if len(p1_vertex_idx) < 10:
            return 0.5  # Not enough valid points
        
        # Try to get UV coordinates - handling different UV formats
        try:
            # For vertex-based UVs
            if len(mesh.visual.uv) == len(mesh.vertices):
                p1_uvs = mesh.visual.uv[p1_vertex_idx]
                p2_uvs = mesh.visual.uv[p2_vertex_idx]
                
                # Calculate UV differences
                uv_diffs = np.linalg.norm(p1_uvs - p2_uvs, axis=1)
            
            # For face-based UVs (per corner)
            elif len(mesh.visual.uv) == len(mesh.faces) * 3:
                # This is more complex - for each vertex, find all faces it's part of
                # and average the UVs
                p1_uvs = []
                p2_uvs = []
                
                uv_by_vertex = [[] for _ in range(len(mesh.vertices))]
                
                # Reshape UVs to faces x 3 vertices x 2 UV coords
                uvs_reshaped = mesh.visual.uv.reshape((-1, 3, 2))
                
                # Map UVs to vertices through faces
                for face_idx, face in enumerate(mesh.faces):
                    for corner_idx, vertex_idx in enumerate(face):
                        uv_by_vertex[vertex_idx].append(uvs_reshaped[face_idx, corner_idx])
                
                # Get average UV for each vertex
                for i, idx in enumerate(p1_vertex_idx):
                    if uv_by_vertex[idx]:
                        p1_uvs.append(np.mean(uv_by_vertex[idx], axis=0))
                    else:
                        p1_uvs.append(np.array([0, 0]))  # Default
                
                for i, idx in enumerate(p2_vertex_idx):
                    if uv_by_vertex[idx]:
                        p2_uvs.append(np.mean(uv_by_vertex[idx], axis=0))
                    else:
                        p2_uvs.append(np.array([0, 0]))  # Default
                
                p1_uvs = np.array(p1_uvs)
                p2_uvs = np.array(p2_uvs)
                
                # Calculate UV differences
                uv_diffs = np.linalg.norm(p1_uvs - p2_uvs, axis=1)
            
            else:
                # Can't determine UV mapping
                return 0.5
        
        except Exception as e:
            print(f"Error processing UVs: {e}")
            return 0.5
        
        # Calculate texture continuity score
        # Lower UV differences mean more continuous texture
        # Normalize and invert (0 diff = 1.0 score, high diff = 0.0 score)
        
        # UV space typically is 0-1, so differences shouldn't exceed sqrt(2)
        # But might be larger if texture is repeated
        max_possible_diff = 1.0
        
        # Calculate normalized scores - higher means more continuous
        continuity_scores = np.maximum(0, 1.0 - (uv_diffs / max_possible_diff))
        
        # Return average continuity score
        return float(np.mean(continuity_scores))
    
    except Exception as e:
        print(f"Error in texture continuity evaluation: {e}")
        return 0.5  # Return neutral score on error

def evaluate_boundary_smoothness(part1, part2, mesh):
    """
    Evaluate how smooth the boundary would be if parts were merged.
    Fixed version to handle zero-length vectors properly.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    
    # Check for empty point clouds
    if len(part1) < 10 or len(part2) < 10:
        return 0.0
    
    # Build KD-trees for both parts
    tree1 = cKDTree(part1)
    tree2 = cKDTree(part2)
    
    # Step 1: Find boundary points (points close to the other part)
    distance_threshold = 0.05  # Adjust based on your data scale
    
    # Find points in part1 that are close to part2
    dist1, idx1 = tree1.query(part2, k=1)
    boundary_mask2 = dist1 < distance_threshold
    boundary_points2 = part2[boundary_mask2]
    boundary_idx2 = idx1[boundary_mask2]  # Corresponding indices in part1
    
    # Find points in part2 that are close to part1
    dist2, idx2 = tree2.query(part1, k=1)
    boundary_mask1 = dist2 < distance_threshold
    boundary_points1 = part1[boundary_mask1]
    boundary_idx1 = idx2[boundary_mask1]  # Corresponding indices in part2
    
    # Skip if not enough boundary points
    if len(boundary_points1) < 10 or len(boundary_points2) < 10:
        return 0.0
    
    # Step 2: Analyze the boundary shape
    
    # Combine boundary points
    boundary_points = np.vstack([boundary_points1, boundary_points2])
    
    # 2a: Try to fit the boundary points to a straight line using PCA
    centroid = np.mean(boundary_points, axis=0)
    centered = boundary_points - centroid
    
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # If boundary is a straight line, first eigenvalue will dominate
    # If boundary is planar, first two eigenvalues will dominate
    # If boundary is complex/jagged, all eigenvalues will be significant
    
    # Calculate linearity: high for straight boundaries (separate parts)
    if np.sum(eigenvalues) > 0:
        linearity = eigenvalues[0] / np.sum(eigenvalues)
    else:
        linearity = 0.0
    
    # Calculate planarity: high for planar boundaries
    if np.sum(eigenvalues) > 0:
        planarity = (eigenvalues[0] + eigenvalues[1]) / np.sum(eigenvalues) - linearity
    else:
        planarity = 0.0
    
    # 2b: Measure boundary curvature and irregularity
    
    # Project points onto the first two principal components to get a 2D representation
    projection_matrix = eigenvectors[:, :2]
    projected_2d = np.dot(centered, projection_matrix)
    
    # Sort projected points to form a path along the boundary
    # This is an approximation - for complex boundaries a more sophisticated approach would be needed
    try:
        # Start with a random point
        path = [0]
        remaining = set(range(1, len(projected_2d)))
        
        # Greedy nearest neighbor path
        for _ in range(min(100, len(projected_2d)-1)):  # Limit iterations
            if not remaining:
                break
                
            last = path[-1]
            # Find closest remaining point
            min_dist = float('inf')
            next_point = None
            
            for i in remaining:
                dist = np.sum((projected_2d[last] - projected_2d[i])**2)
                if dist < min_dist:
                    min_dist = dist
                    next_point = i
            
            if next_point is not None:
                path.append(next_point)
                remaining.remove(next_point)
        
        # Calculate curvature along the path
        if len(path) >= 3:
            # Use a sliding window to estimate curvature
            curvature_values = []
            window_size = 3
            
            for i in range(len(path) - window_size + 1):
                window = path[i:i+window_size]
                points = projected_2d[window]
                
                # Fit a circle to 3 consecutive points
                # Higher curvature = smaller circle = more irregular
                try:
                    # Calculate curvature using the angle between consecutive segments
                    v1 = points[1] - points[0]
                    v2 = points[2] - points[1]
                    
                    # Check for zero-length vectors
                    v1_length = np.linalg.norm(v1)
                    v2_length = np.linalg.norm(v2)
                    
                    if v1_length > 1e-10 and v2_length > 1e-10:  # Only normalize non-zero vectors
                        # Normalize vectors
                        v1_norm = v1 / v1_length
                        v2_norm = v2 / v2_length
                        
                        # Calculate angle using dot product
                        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                        angle = np.arccos(dot_product)
                        
                        # Higher angle = higher curvature
                        curvature_values.append(angle)
                except:
                    continue
            
            # Calculate statistics of curvature
            if curvature_values:
                mean_curvature = np.mean(curvature_values)
                std_curvature = np.std(curvature_values)
                max_curvature = np.max(curvature_values)
            else:
                mean_curvature = 0.0
                std_curvature = 0.0
                max_curvature = 0.0
        else:
            mean_curvature = 0.0
            std_curvature = 0.0
            max_curvature = 0.0
    except Exception as e:
        # Fall back if path creation fails
        print(f"    Error in boundary analysis: {e}")
        mean_curvature = 0.0
        std_curvature = 0.0
        max_curvature = 0.0
    
    # 2c: Check if boundary forms a closed loop (suggesting separate parts)
    # This is simplified - just check if endpoint is close to startpoint
    closed_loop = False
    if len(path) > 3:
        start_point = projected_2d[path[0]]
        end_point = projected_2d[path[-1]]
        dist = np.linalg.norm(end_point - start_point)
        
        # Check if path forms a closed loop
        if len(projected_2d) > 0:
            mean_size = np.mean(np.max(projected_2d, axis=0) - np.min(projected_2d, axis=0))
            if mean_size > 0:
                closed_loop = dist < mean_size * 0.1
    
    # Step 3: Calculate final boundary smoothness score
    
    # Invert linearity - high linearity means smooth boundary (separate parts)
    # Smooth boundaries score low, irregular boundaries score high
    linearity_score = 1.0 - linearity
    
    # Curvature - higher means more irregular
    curvature_score = min(1.0, mean_curvature / (np.pi/3))  # Normalize to 0-1
    
    # Variation in curvature - higher means more irregular
    variation_score = min(1.0, std_curvature / (np.pi/4))  # Normalize to 0-1
    
    # Closed loop factor - closed loops suggest separate parts (lower score)
    closed_loop_factor = 0.7 if closed_loop else 1.0
    
    # Combined score: higher means more irregular (likely same part)
    combined_score = (0.3 * linearity_score + 
                     0.3 * curvature_score + 
                     0.3 * variation_score) * closed_loop_factor
    
    # Print detailed info for debugging
    print(f"    Boundary analysis: linearity={linearity:.3f}, curvature={mean_curvature:.3f}, " +
         f"variation={std_curvature:.3f}, closed_loop={closed_loop}, score={combined_score:.3f}")
    
    return combined_score

def refine_part_boundaries(mesh, vertex_labels):
    """
    A simplified approach to:
    1. Remove disconnected components
    2. Create smooth boundaries
    """
    import numpy as np
    from collections import deque, Counter
    
    print("Refining part boundaries (simplified approach)...")
    
    # Get unique labels
    unique_labels = np.unique(vertex_labels)
    unique_labels = [label for label in unique_labels if label != ""]
    
    if len(unique_labels) < 2:
        print("  Not enough parts to refine boundaries")
        return vertex_labels
        
    # STEP 1: First remove small disconnected components
    print("  Step 1: Removing disconnected components...")
    
    # Process each label separately
    for label in unique_labels:
        # Get vertices with this label
        part_vertices = np.where(vertex_labels == label)[0]
        
        # Find connected components using BFS
        components = []
        unvisited = set(part_vertices)
        
        while unvisited:
            start_vertex = next(iter(unvisited))
            component = []
            queue = deque([start_vertex])
            
            while queue:
                vertex = queue.popleft()
                if vertex not in unvisited:
                    continue
                    
                component.append(vertex)
                unvisited.remove(vertex)
                
                for neighbor in mesh.vertex_neighbors_list[vertex]:
                    if neighbor in unvisited:
                        queue.append(neighbor)
            
            components.append(component)
        
        if len(components) > 1:
            # Sort components by size (largest first)
            components.sort(key=len, reverse=True)
            
            # Keep only the largest component
            largest_component = set(components[0])
            other_vertices = set()
            
            for comp in components[1:]:
                other_vertices.update(comp)
            
            print(f"    Label {label}: Keeping main component ({len(largest_component)} vertices), "
                 f"removing {len(components)-1} smaller components ({len(other_vertices)} vertices)")
            
            # Mark small component vertices for reassignment
            for vertex in other_vertices:
                vertex_labels[vertex] = f"TO_REASSIGN:{label}"
    
    # STEP 2: Find areas with triangle-shaped jagged edges
    print("  Step 2: Identifying jagged boundary areas...")
    
    # Find boundary vertices
    boundary_vertices = []
    
    for vertex in range(len(mesh.vertices)):
        if vertex_labels[vertex] == "" or vertex_labels[vertex].startswith("TO_REASSIGN:"):
            continue
            
        label = vertex_labels[vertex]
        neighbors = mesh.vertex_neighbors_list[vertex]
        
        different_labels = False
        for neighbor in neighbors:
            if (vertex_labels[neighbor] != "" and 
                not vertex_labels[neighbor].startswith("TO_REASSIGN:") and
                vertex_labels[neighbor] != label):
                different_labels = True
                break
        
        if different_labels:
            boundary_vertices.append(vertex)
    
    print(f"    Found {len(boundary_vertices)} boundary vertices")
    
    # Detect jagged edges by looking at neighbor label patterns
    jagged_edges = set()
    
    for vertex in boundary_vertices:
        # Get this vertex's label
        label = vertex_labels[vertex]
        
        # Get neighbor labels
        neighbors = mesh.vertex_neighbors_list[vertex]
        neighbor_labels = [vertex_labels[n] for n in neighbors 
                          if vertex_labels[n] != "" and not vertex_labels[n].startswith("TO_REASSIGN:")]
        
        # Count occurrences of each label
        label_counts = Counter(neighbor_labels)
        
        # If this vertex has more neighbors from a different label, it might be jagged
        for other_label, count in label_counts.items():
            if other_label != label and count > 1:
                # Count same-label neighbors
                same_label_count = label_counts.get(label, 0)
                
                # If more different-label neighbors, mark as jagged
                if count > same_label_count:
                    jagged_edges.add(vertex)
                    break
    
    print(f"    Identified {len(jagged_edges)} vertices in jagged areas")
    
    # STEP 3: Reassign any problem vertices
    print("  Step 3: Reassigning problematic vertices...")
    
    # First reassign disconnected components
    to_reassign = [v for v in range(len(vertex_labels)) 
                  if isinstance(vertex_labels[v], str) and vertex_labels[v].startswith("TO_REASSIGN:")]
    
    for vertex in to_reassign:
        # Get neighbors with labels
        neighbors = mesh.vertex_neighbors_list[vertex]
        neighbor_labels = Counter()
        
        for neighbor in neighbors:
            if (vertex_labels[neighbor] != "" and 
                not vertex_labels[neighbor].startswith("TO_REASSIGN:")):
                neighbor_labels[vertex_labels[neighbor]] += 1
        
        if neighbor_labels:
            # Assign to most common neighbor label
            vertex_labels[vertex] = neighbor_labels.most_common(1)[0][0]
        else:
            # No labeled neighbors - keep original label but remove "TO_REASSIGN:" prefix
            original_label = vertex_labels[vertex][12:]  # Remove "TO_REASSIGN:" prefix
            vertex_labels[vertex] = original_label
    
    print(f"    Reassigned {len(to_reassign)} vertices from disconnected components")
    
    # Then smooth jagged edges
    smooth_iterations = 3
    
    for iteration in range(smooth_iterations):
        updates = {}
        
        for vertex in jagged_edges:
            # Get neighbor labels
            neighbors = mesh.vertex_neighbors_list[vertex]
            neighbor_labels = Counter()
            
            for neighbor in neighbors:
                if vertex_labels[neighbor] != "":
                    neighbor_labels[vertex_labels[neighbor]] += 1
            
            if neighbor_labels:
                # Get most common label
                most_common = neighbor_labels.most_common(1)[0]
                
                # If different from current and has majority, update
                if most_common[0] != vertex_labels[vertex] and most_common[1] > len(neighbors) / 2:
                    updates[vertex] = most_common[0]
        
        # Apply updates
        for vertex, new_label in updates.items():
            vertex_labels[vertex] = new_label
        
        print(f"    Smoothing iteration {iteration+1}: Updated {len(updates)} vertices")
        
        if not updates:
            break
    
    # STEP 4: Final boundary smoothing
    print("  Step 4: Final boundary smoothing...")
    
    # Re-identify boundary vertices
    boundary_vertices = []
    
    for vertex in range(len(mesh.vertices)):
        if vertex_labels[vertex] == "":
            continue
            
        label = vertex_labels[vertex]
        neighbors = mesh.vertex_neighbors_list[vertex]
        
        different_labels = False
        for neighbor in neighbors:
            if vertex_labels[neighbor] != "" and vertex_labels[neighbor] != label:
                different_labels = True
                break
        
        if different_labels:
            boundary_vertices.append(vertex)
    
    # Apply multiple smoothing passes
    for iteration in range(3):
        updates = {}
        
        for vertex in boundary_vertices:
            # Get neighbor labels
            neighbors = mesh.vertex_neighbors_list[vertex]
            neighbor_labels = Counter()
            
            for neighbor in neighbors:
                if vertex_labels[neighbor] != "":
                    neighbor_labels[vertex_labels[neighbor]] += 1
            
            if neighbor_labels:
                # Get most common label
                most_common = neighbor_labels.most_common(1)[0]
                
                # If different from current and has majority, update
                if most_common[0] != vertex_labels[vertex] and most_common[1] > len(neighbors) / 2:
                    updates[vertex] = most_common[0]
        
        # Apply updates
        for vertex, new_label in updates.items():
            vertex_labels[vertex] = new_label
        
        print(f"    Final smoothing iteration {iteration+1}: Updated {len(updates)} vertices")
        
        if not updates:
            break
    
    return vertex_labels

def segment_and_save_parts(mesh, segmented_points, point_labels, instance_ids, output_dir, OBJECT, flip_mesh_z=False, inverse_transform=None):
    """
    Segment a mesh into parts based on point labels and instance IDs using structure-aware segmentation.
    Combines structure-aware segmentation with proper label handling.
    """
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    import os
    import tempfile
    
    print("\nSegmenting mesh with structure-aware approach...")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="mesh_segment_")
    print(f"Created temporary directory: {temp_dir}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract mesh from scene if needed
    if isinstance(mesh, trimesh.Scene):
        print("Input is a scene, extracting mesh...")
        original_scene = mesh
        mesh = next(iter(mesh.geometry.values()))
    else:
        mesh = mesh
        original_scene = trimesh.Scene(mesh)
    
    print(f"Working with mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Create a combined label+instance identifier for each point
    combined_identifiers = []
    for i in range(len(point_labels)):
        # Extract the base label without any trailing numbers
        base_label = point_labels[i]
        instance = instance_ids[i]
        
        # Create a combined identifier: label_instanceid
        combined_id = f"{base_label}_{instance}"
        combined_identifiers.append(combined_id)
    
    combined_identifiers = np.array(combined_identifiers)
    unique_combined_ids = np.unique(combined_identifiers)
    
    print(f"Created {len(unique_combined_ids)} unique part identifiers:")
    for i, part_id in enumerate(unique_combined_ids[:10]):  # Show first 10
        count = np.sum(combined_identifiers == part_id)
        print(f"  {part_id}: {count} points")
    if len(unique_combined_ids) > 10:
        print(f"  ... and {len(unique_combined_ids) - 10} more")
        
    # First analyze part structural properties
    print("\nAnalyzing structural properties of parts...")
    part_structures = analyze_part_structure(segmented_points, combined_identifiers, unique_combined_ids)
    
    # Then do the structure-aware segmentation
    print("\nPerforming structure-aware segmentation...")
    vertex_labels = segment_mesh_structure_aware(mesh, segmented_points, combined_identifiers, unique_combined_ids)
    
    # enforce structural constraints
    print("\nEnforcing structural constraints...")
    vertex_labels = enforce_structural_constraints(mesh, vertex_labels, part_structures)

    # # merge components
    # vertex_labels = process_mesh_segmentation(mesh, vertex_labels, segmented_points, combined_identifiers)

    # # refine boundaries between parts
    # vertex_labels = refine_part_boundaries(mesh, vertex_labels)
    
    # Count assigned vertices
    labeled_vertices = np.sum(vertex_labels != "")
    print(f"Assigned labels to {labeled_vertices} vertices (out of {len(mesh.vertices)})")
    
    # Process each unique part
    for part_id in unique_combined_ids:
        if part_id == "":
            continue  # Skip empty identifier
            
        # Split part_id into base_label and instance components
        parts = part_id.split('_')
        if len(parts) >= 2:
            base_label = '_'.join(parts[:-1])
            instance_num = parts[-1]  # Take the last part as instance number
        else:
            base_label = part_id
            instance_num = "0"
        
        print(f"\nProcessing part: {part_id} (label={base_label}, instance={instance_num})")
        
        # Get vertices with this part identifier
        vertex_mask = vertex_labels == part_id
        
        print(f"Found {np.sum(vertex_mask)} vertices with part identifier '{part_id}'")
        
        if not np.any(vertex_mask):
            print(f"No vertices found for part {part_id}, skipping...")
            continue
            
        # Get faces that have all vertices with this part label
        face_has_label = vertex_mask[mesh.faces]
        face_mask = np.all(face_has_label, axis=1)
        
        # Additionally, include faces that have majority (2 of 3) vertices with this label
        majority_mask = np.sum(face_has_label, axis=1) >= 2 # Do NOT delete
        face_mask = face_mask | majority_mask
        
        if not np.any(face_mask):
            print(f"No faces found for part {part_id}, skipping...")
            continue
        
        # Get the selected faces
        selected_faces = mesh.faces[face_mask]
        
        # Create a set of unique vertices used by these faces
        unique_vertices = np.unique(selected_faces)
        
        # Create a mapping from original vertex indices to new indices
        old_to_new = np.full(len(mesh.vertices), -1)
        old_to_new[unique_vertices] = np.arange(len(unique_vertices))
        
        # Create new vertices array with exact same 3D coordinates
        new_vertices = mesh.vertices[unique_vertices].copy()
        
        # Create new faces array with remapped indices
        new_faces = old_to_new[selected_faces]
        
        # Create the part mesh with explicit vertices and faces
        part_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False  # Don't process the mesh to preserve exact geometry
        )
        
        print(f"Part mesh has {len(part_mesh.vertices)} vertices and {len(part_mesh.faces)} faces")
        print(f"Part bounding box: {part_mesh.bounds}")
        
        # Transfer texture information if available
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            # Create a TextureVisuals object
            part_mesh.visual = trimesh.visual.texture.TextureVisuals()
            
            # Transfer the UV coordinates
            if len(mesh.visual.uv) == len(mesh.vertices):
                # Vertex-based UV mapping
                part_mesh.visual.uv = mesh.visual.uv[unique_vertices]
                print(f"Transferred vertex-based UVs: {part_mesh.visual.uv.shape}")
            elif len(mesh.visual.uv) == len(mesh.faces) * 3:
                # Face-based UV mapping (per corner)
                original_uvs = mesh.visual.uv.reshape((-1, 3, 2))
                new_uvs = original_uvs[face_mask]
                part_mesh.visual.uv = new_uvs.reshape((-1, 2))
                print(f"Transferred face-based UVs: {new_uvs.shape}")
            
            # Transfer the material and texture
            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                part_mesh.visual.material = mesh.visual.material.copy()
                print("Transferred material")
            
            if hasattr(mesh.visual, 'texture') and mesh.visual.texture is not None:
                part_mesh.visual.texture = mesh.visual.texture
                print("Transferred texture")

        # Apply inverse transformation if provided (to undo ICP alignment)
        inverse_transform = None
        if inverse_transform is not None:
            part_mesh.apply_transform(inverse_transform)
            print("Applied inverse transformation to restore original orientation")
        else:
            theta = np.radians(90)  # -90 box, +90 laptop & drawer
            rot_z_neg90 = np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            part_mesh.apply_transform(rot_z_neg90)
            print(f'Applied {theta} degree rotation around z axis manually')
        
        # Sanitize filename
        safe_label = base_label.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace(' ', '_')
        
        # Create filename with label and instance number
        filename = f"{safe_label}_{instance_num}"
        
        # Save as GLB
        glb_output_path = os.path.join(output_dir, f'{OBJECT}_{filename}.glb')
        try:
            # Create a scene with the part mesh to preserve materials
            part_scene = trimesh.Scene(part_mesh)
            part_scene.export(glb_output_path)
            print(f"Saved part mesh to: {glb_output_path}")
            
            # Verify the exported file
            try:
                test_load = trimesh.load(glb_output_path)
                if isinstance(test_load, trimesh.Scene):
                    test_mesh = next(iter(test_load.geometry.values()))
                else:
                    test_mesh = test_load
                
                print(f"Verified exported mesh: {len(test_mesh.vertices)} vertices, {len(test_mesh.faces)} faces")
                print(f"Exported bounding box: {test_mesh.bounds}")
                
                # Check if the mesh is flat
                bounds = test_mesh.bounds
                dimensions = bounds[1] - bounds[0]
                if any(dim < 0.001 for dim in dimensions):
                    print("WARNING: Exported mesh appears to be flat!")
            except Exception as e:
                print(f"Warning: Could not verify exported file: {e}")
        except Exception as e:
            print(f"Error saving GLB: {e}")
            
        # Try saving as OBJ
        obj_output_path = os.path.join(output_dir, f'{OBJECT}_{filename}.obj')
        try:
            part_mesh.export(obj_output_path, include_texture=True)
            print(f"Saved part mesh to: {obj_output_path}")
        except Exception as e2:
            print(f"Error saving OBJ: {e2}")
    
    # Clean up temporary directory
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except:
        print(f"Could not remove temporary directory: {temp_dir}")



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
        channels = header['channels']
        channel_names = list(channels.keys())
        
        # Try Z first, then R, G, B
        depth_channel = None
        for ch in ['Z', 'R', 'G', 'B']:
            if ch in channel_names:
                depth_channel = ch
                break
                
        if depth_channel is None:
            depth_channel = channel_names[0]  # Use first available channel
            
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
        print(f"Loaded depth with OpenEXR from {filepath}, using channel '{depth_channel}'")
        
    except (ImportError, ModuleNotFoundError):
        print("OpenEXR not available, trying alternative method...")
        
        # Try using imageio
        try:
            import imageio.v3 as iio
            depth = iio.imread(filepath)
            
            # If multi-channel, take first channel
            if len(depth.shape) == 3:
                depth = depth[:,:,0]
            print(f"Loaded depth with imageio from {filepath}")
                
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
                print(f"Loaded depth with OpenCV from {filepath}")
                    
            except Exception as e:
                print(f"Failed to load depth with OpenCV: {e}")
                # If all else fails, create a dummy depth map
                print("WARNING: Creating placeholder depth map (failed to load from file)")
                depth = np.ones((512, 512), dtype=np.float32) # Fallback if all fails
    
    # Print statistics
    print(f"Loaded depth with shape {depth.shape}, dtype {depth.dtype}")
    valid_mask = ~np.isnan(depth) & ~np.isinf(depth) & (depth > 0) & (depth < 100000)
    if np.any(valid_mask):
        print(f"Valid depth range: {np.min(depth[valid_mask]):.6f} to {np.max(depth[valid_mask]):.6f}")
        print(f"Mean valid depth: {np.mean(depth[valid_mask]):.6f}")
    else:
        print("WARNING: No valid depth values found!")
    
    return depth

def visualize_point_cloud_with_mesh(points, labels, mesh, output_path, flip_mesh_z=True):
    """
    Visualize point cloud and mesh to check alignment.
    Make mesh more prominent and point clouds more transparent.
    
    Args:
        points: Nx3 array of 3D points
        labels: Array of labels for each point
        mesh: Trimesh mesh object
        output_path: Path to save the visualization
        flip_mesh_z: (UNUSED in final version, kept for compatibility) Whether to flip the Z coordinate of the mesh to match points
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Visualizing with matplotlib...")
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color map for the unique labels
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10.colors
    
    # Plot mesh first so it's in the background
    mesh_vertices = np.array(mesh.vertices)
    
    # This flip_mesh_z part is from your original code and should be False in main() call.
    # If it was active, it would apply an additional (and likely incorrect) flip.
    # if flip_mesh_z:
    # mesh_vertices[:, 2] = mesh_vertices[:, 2] + 0.5
    #     mesh_vertices[:, 1] = -mesh_vertices[:, 1]

    # Plot mesh vertices with higher visibility
    ax.scatter(
        mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2],
        c='black', marker='.', s=2, alpha=0.5, label='Mesh Vertices'
    )
    
    # Plot mesh wireframe for better visibility
    # This is computationally expensive for large meshes, so limit the faces
    mesh_faces = np.array(mesh.faces)
    max_faces_for_wireframe = min(5000, len(mesh_faces))
    if len(mesh_faces) > max_faces_for_wireframe:
        step = len(mesh_faces) // max_faces_for_wireframe
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
            # Fixed: Remove the list wrapper around the color
            color = colors[i % len(colors)]
            ax.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                color=color,  # Changed from c=[colors[i % len(colors)]]
                marker='.', s=1, alpha=0.3, label=f'Label {label}'
            )
    
    # Add coordinate axes (Blender World reference)
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.2, arrow_length_ratio=0.3, label='X axis')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.2, arrow_length_ratio=0.3, label='Y axis')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.2, arrow_length_ratio=0.3, label='Z axis')
    
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

def _align_mesh_to_point_cloud_icp_only(mesh_to_align, points_to_align_to, max_correspondence_dist=0.05):
    """
    Internal helper: Align a mesh to a point cloud using ICP and scaling.
    Assumes rough initial alignment.
    Returns the transformed mesh (target), transformed points (source), and the combined transformation matrix applied to points.
    """
    
    # Sample points from mesh and points_3d for ICP to speed up
    num_samples = min(10000, len(points_to_align_to), len(mesh_to_align.vertices))
    
    if len(points_to_align_to) > num_samples:
        indices = np.random.choice(len(points_to_align_to), num_samples, replace=False)
        sampled_source_points = points_to_align_to[indices]
    else:
        sampled_source_points = points_to_align_to

    if len(mesh_to_align.vertices) > num_samples:
        aligned_mesh_points = mesh_to_align.sample(num_samples)
    else:
        aligned_mesh_points = mesh_to_align.vertices

    print(f"  ICP: Using {len(sampled_source_points)} points from source and {len(aligned_mesh_points)} from target mesh for alignment.")

    # Scale source (points_to_align_to) to match target (mesh) diameter
    # `scale_target_to_source` scales the *second* argument to match the *first*.
    # So to scale `sampled_source_points` to `aligned_mesh_points`:
    scaled_source_points, scale_factor_points = scale_target_to_source(aligned_mesh_points, sampled_source_points)
    print(f"  Scaled point cloud by factor: {scale_factor_points:.4f}")
    
    # Convert to Open3D point clouds for ICP
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(scaled_source_points)
    
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(aligned_mesh_points)
    
    # Estimate normals for point-to-plane ICP (required for good results)
    source_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    target_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Perform ICP registration
    result_icp = o3d.pipelines.registration.registration_icp(
        source_o3d,
        target_o3d,
        max_correspondence_dist,
        np.identity(4), # Initial guess (assuming rough prior alignment)
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
    )
    
    icp_transformation_matrix = result_icp.transformation
    print(f"  ICP Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")

    # Apply combined transformation to the original full point cloud
    # 1. Scale the points_to_align_to
    # 2. Apply the ICP transformation matrix to the scaled points
    transformed_points = points_to_align_to * scale_factor_points
    transformed_points_homogeneous = np.hstack([transformed_points, np.ones((len(transformed_points), 1))])
    transformed_points = (transformed_points_homogeneous @ icp_transformation_matrix.T)[:, :3]

    # The mesh (`mesh_to_align`) itself is the target, so it is not transformed by this ICP call.
    # We return it as is, and the transformed points.
    return mesh_to_align, transformed_points, np.eye(4)

def align_mesh_to_point_cloud(mesh, points, visualize=True, output_path=None, mirror_y=False):
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

    # Apply Y-axis mirroring if requested
    if mirror_y:
        # Create a transformation matrix that flips the Y coordinate
        mirror_matrix = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Negate Y coordinate
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        aligned_mesh.apply_transform(mirror_matrix)
        print("Applied Y-axis mirroring to mesh")
    
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

    # Combine the initial transformations for later inversion
    if mirror_y:
        initial_transform = np.matmul(rotation_x, np.matmul(rot_y_180, np.matmul(flip_z, mirror_matrix)))
    else:
        initial_transform = np.matmul(rotation_x, np.matmul(rot_y_180, flip_z))

    # Refine alignment with scaling and icp
    # use 10000 sampled points for scaling
    if len(points) > 10000:
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

    if len(points) > 10000:
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

    # Calculate the inverse transformation
    complete_transform = np.matmul(transformation_matrix, initial_transform)
    inverse_transform = np.linalg.inv(complete_transform)
    
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
    
    return aligned_mesh, points, inverse_transform

def make_3d_point_clouds_mutually_exclusive(point_clouds_by_instance, instance_ids, point_labels, distance_threshold=0.02):
    """
    Make 3D point clouds mutually exclusive by removing overlapping points from larger clouds.
    Smaller point clouds get priority over larger ones.
    
    Args:
        point_clouds_by_instance: Dictionary mapping instance IDs to point clouds
        instance_ids: Array of instance IDs  
        point_labels: Dictionary mapping instance IDs to their base labels
        distance_threshold: Distance threshold for considering points as overlapping
        
    Returns:
        Updated point_clouds_by_instance with overlaps removed
    """
    from scipy.spatial import cKDTree
    import numpy as np
    
    print("Making 3D point clouds mutually exclusive...")
    
    # Group instances by their base label first
    instances_by_label = {}
    for instance_id in point_clouds_by_instance.keys():
        if instance_id in point_labels:
            base_label = point_labels[instance_id]
            if base_label not in instances_by_label:
                instances_by_label[base_label] = []
            instances_by_label[base_label].append(instance_id)
    
    # Create a copy of the point clouds to modify
    updated_point_clouds = {k: v.copy() for k, v in point_clouds_by_instance.items()}
    
    # First pass: resolve conflicts within each label group
    for base_label, label_instances in instances_by_label.items():
        if len(label_instances) <= 1:
            continue  # No conflicts within this label
            
        print(f"  Processing label '{base_label}' with {len(label_instances)} instances")
        
        # Calculate sizes and sort by size (smallest first)
        instance_sizes = []
        for instance_id in label_instances:
            if instance_id in updated_point_clouds:
                size = len(updated_point_clouds[instance_id])
                instance_sizes.append((instance_id, size))
        
        # Sort by size (smallest first gets highest priority)
        instance_sizes.sort(key=lambda x: x[1])
        
        print(f"    Instance sizes: {[(iid, size) for iid, size in instance_sizes]}")
        
        # Track points that have been claimed by higher priority instances
        claimed_trees = []
        
        # Process instances in order of priority (smallest first)
        for i, (instance_id, original_size) in enumerate(instance_sizes):
            current_points = updated_point_clouds[instance_id]
            
            if len(current_points) == 0:
                continue
                
            # Check against all previously processed (higher priority) instances
            points_to_remove = set()
            
            for claimed_tree in claimed_trees:
                # Find points in current cloud that are too close to claimed points
                distances, _ = claimed_tree.query(current_points, distance_upper_bound=distance_threshold)
                
                # Mark points for removal if they're within threshold
                close_indices = np.where(np.isfinite(distances))[0]
                points_to_remove.update(close_indices)
            
            # Remove overlapping points
            if points_to_remove:
                keep_indices = np.setdiff1d(np.arange(len(current_points)), list(points_to_remove))
                updated_point_clouds[instance_id] = current_points[keep_indices]
                
                removed_count = len(points_to_remove)
                final_size = len(keep_indices)
                print(f"    Instance {instance_id}: kept {final_size} points, removed {removed_count} overlapping points")
            else:
                print(f"    Instance {instance_id}: kept all {len(current_points)} points (no overlaps)")
            
            # Add current points to claimed points for future comparisons
            final_points = updated_point_clouds[instance_id]
            if len(final_points) > 0:
                claimed_trees.append(cKDTree(final_points))
    
    # Second pass: handle conflicts between different labels (prioritize smaller instances globally)
    print("  Checking for conflicts between different labels...")
    
    # Get all instances sorted by size globally
    all_instances = []
    for instance_id, points in updated_point_clouds.items():
        if len(points) > 0:
            base_label = point_labels.get(instance_id, "unknown")
            all_instances.append((instance_id, len(points), points, base_label))
    
    # Sort by size (smallest first)
    all_instances.sort(key=lambda x: x[1])
    
    # Track globally claimed points
    global_claimed_trees = []
    
    for i, (instance_id, size, points, base_label) in enumerate(all_instances):
        if len(points) == 0:
            continue
            
        # Check against all previously processed instances (regardless of label)
        points_to_remove = set()
        
        for claimed_tree in global_claimed_trees:
            distances, _ = claimed_tree.query(points, distance_upper_bound=distance_threshold)
            close_indices = np.where(np.isfinite(distances))[0]
            points_to_remove.update(close_indices)
        
        # Remove overlapping points
        if points_to_remove:
            keep_indices = np.setdiff1d(np.arange(len(points)), list(points_to_remove))
            updated_point_clouds[instance_id] = points[keep_indices]
            
            removed_count = len(points_to_remove)
            final_size = len(keep_indices)
            if removed_count > 0:
                print(f"    Cross-label: Instance {instance_id} ({base_label}): kept {final_size} points, removed {removed_count} overlapping points")
        
        # Add to globally claimed points
        final_points = updated_point_clouds[instance_id]
        if len(final_points) > 0:
            global_claimed_trees.append(cKDTree(final_points))
    
    # Filter out instances that became empty
    original_count = len(updated_point_clouds)
    updated_point_clouds = {k: v for k, v in updated_point_clouds.items() if len(v) > 0}
    final_count = len(updated_point_clouds)
    
    if final_count < original_count:
        print(f"  Removed {original_count - final_count} instances that became empty after overlap removal")
    
    print(f"  Result: {final_count} non-empty mutually exclusive 3D point clouds")
    
    return updated_point_clouds


def main(OBJECT, flip_z=False):
    # Define paths
    data_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{OBJECT}"
    render_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{OBJECT}"
    output_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{OBJECT}"

    overwrite = True  # Set to True to force reprocessing

    # If already ran, skip repeated execution
    import glob
    matching_files = glob.glob(f"{output_dir}/output/{OBJECT}_*.obj")
    if (not overwrite) and matching_files:
        print(f"Segmented meshes already exist for object {args.object}, skipping...")
        return None
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # if 'partnet' in OBJECT:
    #     OBJECT = '_'.join(OBJECT.split('_')[1:-1])
    
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
            't': data['t'],
            'view_dir': data['view_dir'],
            'right': data['right'],
            'up': data['up']
        })
    
    # Debug camera positions
    print("\nCamera positions:")
    for i, params in enumerate(camera_params):
        print(f"Camera {i}: {params['t']}")

    # # Debug camera intrinsics
    # print("\nCamera intrinsics:")
    # for i, params in enumerate(camera_params):
    #     print(f"Camera {i}: {params['K']}")
    
    # Load depth maps
    depth_files = sorted([os.path.join(render_dir, f) for f in os.listdir(render_dir) 
                         if f.startswith("depth_") and f.endswith(".exr")])
    # if 'multi_view' not in OBJECT:
    #     depth_files = sorted([os.path.join(render_dir, f) for f in os.listdir(render_dir) 
    #                      if f.startswith(f"depth_{OBJECT}_multi-view") and f.endswith(".exr")])
        
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
    
    # Extract indices from segmentation result filenames
    seg_indices = [int(os.path.basename(path).split(f'{OBJECT}_')[1].split('_')[0]) for path in results_paths]
    # if "multi_view" in results_paths[0]:
    #     seg_indices = [int(os.path.basename(path).split('view_')[1].split('_')[0]) for path in results_paths]
    # elif "real" in results_paths[0]:
    #     seg_indices = [int(os.path.basename(path).split('real_')[1].split('_')[0]) for path in results_paths]

    if not (len(rgb_images) == len(camera_param_files) == len(depth_files) == len(results_paths)):
        # print("Warning: Mismatched number of files. The script may not work correctly.")

        # Filter other lists to only include files with matching indices
        rgb_images = [img for i, img in enumerate(rgb_images) if i in seg_indices]
        camera_param_files = [params for i, params in enumerate(camera_param_files) if i in seg_indices]
        camera_params = [p for i,p in enumerate(camera_params) if i in seg_indices]
        depth_maps = [depth for i, depth in enumerate(depth_maps) if i in seg_indices]

        # Print results
        print(f"\nAfter filtering:")
        print(f"Available indices: {seg_indices}")
        print(f"Found {len(rgb_images)} images with segmentation results")

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
    # aligned_mesh, points_3d, inverse_transform = align_mesh_to_point_cloud(
    #     mesh, 
    #     points_3d, 
    #     visualize=True, 
    #     output_path=os.path.join(output_dir, "mesh_alignment.png"),
    #     mirror_y=False
    # )
    # aligned_mesh, points_3d, inverse_transform = _align_mesh_to_point_cloud_icp_only(
    #     mesh, 
    #     points_3d
    # )
    aligned_mesh = mesh.copy()
    z_up = np.array([[ 1,  0,  0,  0 ],
                    [ 0,  0, -1,  0 ],
                    [ 0,  1,  0,  0 ], # 0,1,0,0.4
                    [ 0,  0,  0,  1 ]])
    aligned_mesh.apply_transform(z_up)
    inverse_transform = np.eye(4)
    
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
    
    # Create a mapping from instance IDs to their labels
    print("Creating instance-to-label mapping...")
    instance_to_label = {}

    # First, create arrays with unique instance IDs and their first occurrences
    unique_instance_ids, unique_indices = np.unique(instance_ids, return_index=True)

    # Use these indices to efficiently get the corresponding labels
    for i, idx in enumerate(unique_indices):
        instance_id = instance_ids[idx]
        instance_label = labels[idx]
        instance_to_label[instance_id] = instance_label
    

    # First merge similar instances (same semantic meaning across views)
    print("\nMerging similar instances across views with two-pass approach")
    instance_mapping, clusters = merge_instances_with_two_pass_approach(
        aligned_mesh, point_clouds_by_instance, np.unique(instance_ids), 
        point_labels=instance_to_label, min_clusters=len(np.unique(labels))
    )
    
    # Create merged point clouds from clusters
    print("\nCreating merged point clouds from clusters...")
    merged_point_clouds_by_cluster = {}
    cluster_to_label = {}
    
    for cluster_idx, cluster_instances in enumerate(clusters):
        if not cluster_instances:  # Skip empty clusters
            continue
            
        representative_id = cluster_instances[0]
        
        # Combine all points from this cluster
        cluster_points = []
        for instance_id in cluster_instances:
            if instance_id in point_clouds_by_instance:
                cluster_points.append(point_clouds_by_instance[instance_id])
        
        if cluster_points:
            combined_points = np.vstack(cluster_points)
            merged_point_clouds_by_cluster[representative_id] = combined_points
            cluster_to_label[representative_id] = instance_to_label[representative_id]
            print(f"  Cluster {cluster_idx} (rep: {representative_id}): {len(combined_points)} points")
    
    # Make 3D point clouds mutually exclusive
    print("\nRemoving overlapping points between merged semantic parts...")
    merged_point_clouds_by_cluster = make_3d_point_clouds_mutually_exclusive(
        merged_point_clouds_by_cluster, 
        list(merged_point_clouds_by_cluster.keys()), 
        cluster_to_label,
        distance_threshold=0.02  # Adjust based on your data scale
    )
    
    # Update the main point arrays to reflect the cleaned point clouds
    merged_points_all = []
    merged_labels_all = []
    merged_instance_ids_all = []
    
    for representative_id, points in merged_point_clouds_by_cluster.items():
        if len(points) > 0:
            merged_points_all.append(points)
            cluster_label = cluster_to_label[representative_id]
            merged_labels_all.extend([cluster_label] * len(points))
            merged_instance_ids_all.extend([representative_id] * len(points))
    
    # Stack all merged points
    if merged_points_all:
        merged_points = np.vstack(merged_points_all)
        merged_labels = np.array(merged_labels_all)
        merged_instance_ids = np.array(merged_instance_ids_all)
        
        print(f"\nFinal merged point cloud has {len(merged_points)} points with {len(np.unique(merged_instance_ids))} unique parts")
        
        # Visualize merged point cloud
        vis_path = os.path.join(output_dir, 'merged_point_cloud.png')
        
        all_camera_params = []
        for idx in seg_indices:
            if idx < len(camera_params):
                params = camera_params[idx]
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
            point_labels=merged_labels,  # Pass the semantic labels
            instance_ids=merged_instance_ids,  # Pass the instance IDs
            # unique_labels=np.unique(merged_instance_ids),
            output_dir=output_dir,
            OBJECT=OBJECT,
            flip_mesh_z=False,  # Already aligned
            inverse_transform=inverse_transform
        )
        
        # # Also save individual part meshes with meaningful names
        # for i, instance_id in enumerate(unique_instances):
        #     # Get the original label for this instance
        #     instance_mask = merged_instance_ids == instance_id
        #     if np.any(instance_mask):
        #         instance_label = merged_labels[np.where(instance_mask)[0][0]]
                
        #         # Get points for this instance
        #         instance_points = merged_points[instance_mask]
                
        #         # Create label array for these points (all same instance)
        #         instance_numeric_id = label_to_id[instance_id]
        #         instance_point_labels = np.full(len(instance_points), instance_numeric_id)
                
        #         # Segment and save this part
        #         part_output_dir = os.path.join(output_dir, 'parts')
        #         os.makedirs(part_output_dir, exist_ok=True)
                
        #         segment_and_save_parts(
        #             mesh=aligned_mesh,
        #             segmented_points=instance_points,
        #             point_labels=instance_point_labels,
        #             unique_labels=[instance_numeric_id],
        #             output_dir=part_output_dir,
        #             OBJECT=f"{OBJECT}_{instance_label}_{i}",
        #             flip_mesh_z=False  # Already aligned
        #         )
    else:
        print("No valid merged point clouds, skipping mesh segmentation.")
    
    print("\nProcessing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('object', default='drawer_rodin')
    args = parser.parse_args()
    main(args.object, flip_z=False)