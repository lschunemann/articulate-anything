import numpy as np
import trimesh
import cv2
import json
import os
from scipy.spatial import cKDTree
import pycocotools.mask as mask_util
import supervision as sv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from blender_depth import transform_blender_points, align_point_cloud_to_mesh, visualize_blender_depth

def label_to_colors(labels, class_names):
    """Convert labels to the same colors used in the segmentation visualization"""
    colors = get_default_colors(len(class_names))
    
    num_vertices = len(labels)
    vertex_colors = np.zeros((num_vertices, 4))  # RGBA colors
    
    for i, class_name in enumerate(class_names):
        # Get the same color that was used for visualization
        rgb_color = colors[i]  # Already in range [0-255]
        rgba_color = np.append(rgb_color / 255.0, 1.0)  # Convert to [0-1] range and add alpha
        vertex_colors[labels == i] = rgba_color
    
    return vertex_colors


def get_default_colors(num_classes):
    """Generate distinct colors for visualization"""
    colors = []
    for i in range(num_classes):
        # Generate colors using HSV color space for better distinction
        hue = i / num_classes
        # Convert HSV to RGB (using full saturation and value)
        rgb = cv2.cvtColor(np.uint8([[[hue * 180, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
        colors.append(rgb)
    return colors

def visualize_point_cloud_with_cameras(points, labels, camera_params, title, output_path=None):
    """
    Visualize point cloud with camera positions and orientations.
    
    Args:
        points: Nx3 array of 3D points
        labels: Array of N labels
        camera_params: List of dictionaries with 'R' and 't' keys
        title: Title for the plot
        output_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap for the unique labels
    unique_labels = np.unique(labels)
    label_to_color = {}
    
    # Define a list of colors for visualization
    colors = plt.cm.tab10.colors
    for i, label in enumerate(unique_labels):
        label_to_color[label] = colors[i % len(colors)]
    
    # Plot points with colors based on labels
    for label in unique_labels:
        mask = labels == label
        if np.any(mask):
            ax.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                c=[label_to_color[label]],
                marker='.', s=1, alpha=0.6, label=label
            )
    
    # Plot cameras
    camera_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    
    for i, params in enumerate(camera_params):
        # Extract camera parameters
        R = params['R']
        t = params['t']
        
        # Plot camera position
        color = camera_colors[i % len(camera_colors)]
        ax.scatter(t[0], t[1], t[2], c=color, marker='^', s=100, alpha=1, label=f'Camera {i}')
        
        # Plot camera orientation
        axis_length = 0.5
        
        # Calculate camera coordinate axes in world space
        # R is world-to-camera, so R.T is camera-to-world
        R_cam_to_world = R.T
        
        # Right direction (X) - red
        x_axis = R_cam_to_world[:, 0] * axis_length
        ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], 
                 color='r', length=axis_length, arrow_length_ratio=0.2)
        
        # Up direction (Y) - green
        y_axis = R_cam_to_world[:, 1] * axis_length
        ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], 
                 color='g', length=axis_length, arrow_length_ratio=0.2)
        
        # View direction (Z) - blue (negative because camera looks along -Z)
        z_axis = -R_cam_to_world[:, 2] * axis_length
        ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], 
                 color='b', length=axis_length, arrow_length_ratio=0.2)
        
        # Draw a line from camera to origin
        ax.plot([t[0], 0], [t[1], 0], [t[2], 0], c=color, linestyle='--', alpha=0.3)
    
    # Plot origin
    ax.scatter(0, 0, 0, c='k', marker='*', s=200, label='Origin')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add legend for unique part labels
    handles, labels = ax.get_legend_handles_labels()
    # Get unique labels for legend (avoid duplicates)
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Set equal aspect ratio for axes
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
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    return fig, ax


# And in visualize_masks:
def visualize_masks(masks, labels, image, save_path):
    """Visualize segmentation masks overlaid on image"""
    colors = get_default_colors(len(labels))
    overlay = image.copy()
    
    # Create colored overlay for each mask
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = colors[i]
        overlay[mask > 0] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Add labels
    for i, label in enumerate(labels):
        cv2.putText(result, label, (10, 30 + i*30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i].tolist(), 2)
    
    cv2.imwrite(save_path, result)

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

def process_depth_map(depth_map, scale_factor=1.0):  # Adjust scale_factor as needed
    """Process depth map with proper scaling"""
    if len(depth_map.shape) == 3:
        depth_map = depth_map[:,:,0]
    
    depth_map = depth_map.astype(float)
    
    # Scale to reasonable world units (e.g., meters)
    depth_map = depth_map / scale_factor
    
    return depth_map

def transform_points(pixels, depths, K, R, t):
    """Transform 2D image points to 3D world coordinates"""
    # 1. Convert pixel coordinates to camera coordinates
    pixels_homogeneous = np.stack([pixels[:, 0], pixels[:, 1], np.ones_like(pixels[:, 0])], axis=0)
    rays = np.linalg.inv(K) @ pixels_homogeneous
    
    # 2. Scale by depth to get points in camera coordinate system
    depths = process_depth_map(depths).reshape(-1)
    camera_points = rays * depths[None, :]  # Shape: (3, N)
    
    # 3. Transform from camera to world coordinates
    world_points = R @ camera_points + t.reshape(3, 1)
    
    return world_points.T  # Return as (N, 3)

def lift_2d_masks_to_3d(rgb_images, results_paths, depth_maps, camera_params, OBJECT, output_dir):
    all_3d_points = []
    all_labels = []
    
    # Check if the output directory exists, create if not
    os.makedirs(output_dir, exist_ok=True)
    
    for view_idx in range(len(rgb_images)):
        print(f"\nProcessing view {view_idx + 1}/{len(rgb_images)}")
        
        # Load masks for this view
        masks, label_names = load_masks_from_results(results_paths[view_idx], OBJECT)
        
        # Get camera parameters
        K = camera_params[view_idx]['K']
        R = camera_params[view_idx]['R']
        t = camera_params[view_idx]['t']
        
        # Get current depth map
        depth_map = depth_maps[view_idx]
        
        # Check if depth map needs transposing
        if depth_map.shape[0] == 1920 and depth_map.shape[1] == 1080:
            print(f"Transposing depth map for view {view_idx}")
            depth_map = depth_map.T
        
        # Log some statistics about the depth map
        depth_stats = {
            "shape": depth_map.shape,
            "min": np.min(depth_map[depth_map > 0]) if np.any(depth_map > 0) else None,
            "max": np.max(depth_map[depth_map > 0]) if np.any(depth_map > 0) else None,
            "mean": np.mean(depth_map[depth_map > 0]) if np.any(depth_map > 0) else None,
            "nan_count": np.sum(np.isnan(depth_map)),
            "inf_count": np.sum(np.isinf(depth_map)),
            "zero_count": np.sum(depth_map == 0)
        }
        print(f"Depth map stats: {depth_stats}")
        
        view_points = []
        view_labels = []
        
        for mask_idx, (mask, label_name) in enumerate(zip(masks, label_names)):  # Fixed this line (removed space)
            # Ensure mask has the same dimensions as depth map
            if mask.shape != depth_map.shape:
                print(f"Resizing mask for {label_name} from {mask.shape} to {depth_map.shape}")
                mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
            
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0:
                print(f"No mask pixels for {label_name}")
                continue
                
            # Stack pixel coordinates
            pixels = np.stack([x_coords, y_coords], axis=1)
            
            # Transform to 3D world coordinates
            points_3d = transform_blender_points(pixels, depth_map, K, R, t, max_depth=100.0)
            
            if len(points_3d) == 0:
                print(f"No valid 3D points for {label_name}")
                continue
                
            print(f"Generated {len(points_3d)} points for {label_name}")
            view_points.append(points_3d)
            view_labels.extend([label_name] * len(points_3d))

        if view_points:
            view_points = np.vstack(view_points)
            vis_path = os.path.join(output_dir, f'view_{view_idx}_points.png')
            
            # Create camera parameters for visualization
            cam_params_dict = {'R': R, 't': t}
            visualize_point_cloud_with_cameras(view_points, np.array(view_labels), 
                                            [cam_params_dict], 
                                            f'View {view_idx} Point Cloud', vis_path)
            all_3d_points.append(view_points)
            all_labels.extend(view_labels)
            
            print(f"View {view_idx} processed with {len(view_points)} total points")
        else:
            print(f"No valid points found in view {view_idx}")

    # At the end, for the combined visualization
    if not all_3d_points:
        print("No valid 3D points found in any view")
        return np.zeros((0, 3)), np.array([])
        
    points_3d = np.vstack(all_3d_points)
    labels = np.array(all_labels)

    print(f"\nFinal point cloud has {len(points_3d)} points with {len(np.unique(labels))} unique labels")
    
    # Visualize final combined point cloud with all cameras
    vis_path = os.path.join(output_dir, 'combined_point_cloud.png')
    all_camera_params = [{'R': params['R'], 't': params['t']} for params in camera_params]
    visualize_point_cloud_with_cameras(points_3d, labels, all_camera_params, 
                                    'Combined Point Cloud', vis_path)
    
    return points_3d, labels

def segment_and_save_parts(mesh, segmented_points, point_labels, unique_labels, output_dir, OBJECT):
    """Segment mesh and save separate .glb files for each part"""
    print("\nSegmenting mesh into parts...")
    
    # Build KD-tree for nearest neighbor search
    kdtree = cKDTree(segmented_points)
    
    # Query nearest neighbors for each mesh vertex
    distances, indices = kdtree.query(mesh.vertices)
    
    # Only assign labels to vertices that are close enough to a point
    max_distance = np.median(distances) * 2  # Adjust this threshold as needed
    print(f"Using max distance threshold: {max_distance}")
    
    # Mask for vertices that are close enough to a point
    close_enough = distances < max_distance
    vertex_labels = np.full(len(mesh.vertices), -1)  # -1 for unassigned
    vertex_labels[close_enough] = point_labels[indices[close_enough]]
    
    print(f"Found {len(np.unique(vertex_labels))} unique parts")

    # Debug label distribution
    print("Label distribution in point cloud:")
    for label in unique_labels:
        count = np.sum(point_labels == label)
        print(f"  {label}: {count} points")
        
    print("Label distribution in mesh vertices after KNN:")
    for label in unique_labels:
        count = np.sum(vertex_labels == label)
        print(f"  {label}: {count} vertices")
    
    # Create separate mesh for each label
    for label in unique_labels:
        print(f"\nProcessing part: {label}")
        
        # Get vertices for this label
        vertex_mask = vertex_labels == label
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
        
        # Save part mesh
        output_path = os.path.join(output_dir, f'{OBJECT}_part_{label}.glb')
        part_mesh.export(output_path)
        print(f"Saved part mesh to: {output_path}")
        print(f"Part statistics:")
        print(f"  Vertices: {len(part_vertices)}")
        print(f"  Faces: {len(part_faces)}")

def visualize_point_cloud_with_mesh(points, labels, mesh, output_path):
    """
    Visualize point cloud and mesh to check alignment.
    
    Args:
        points: Nx3 array of 3D points
        labels: Array of N labels
        mesh: Trimesh mesh object
        output_path: Path to save the visualization
    """
    import open3d as o3d
    import numpy as np
    import os
    
    # First, check if we have any points
    if len(points) == 0:
        print("WARNING: No points to visualize!")
        return
    
    # Create colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create a colormap for the unique labels
    unique_labels = np.unique(labels)
    
    # Use a simple color mapping
    color_map = {}
    default_colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
        [0.5, 0.5, 0],  # Olive
        [0.5, 0, 0.5],  # Purple
        [0, 0.5, 0.5],  # Teal
        [0.7, 0.3, 0.3],  # Brick
    ]
    
    # Assign colors to unique labels
    for i, label in enumerate(unique_labels):
        color_map[label] = default_colors[i % len(default_colors)]
    
    # Create color array
    colors = np.zeros((len(points), 3))
    
    # Assign colors based on labels
    for i, label in enumerate(labels):
        colors[i] = color_map[label]
    
    # Make sure colors are float32 and in the correct range [0, 1]
    colors = colors.astype(np.float32)
    
    # Check color range
    if np.max(colors) > 1.0 or np.min(colors) < 0.0:
        print("Warning: Colors out of range [0,1], normalizing...")
        colors = np.clip(colors, 0.0, 1.0)
    
    try:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception as e:
        print(f"Error setting colors: {e}")
        print("Trying with a uniform color instead...")
        pcd.paint_uniform_color([1, 0, 0])  # Red as fallback
    
    # Convert trimesh to open3d mesh
    try:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
        o3d_mesh.compute_vertex_normals()
        
        # Make mesh semi-transparent white
        o3d_mesh.paint_uniform_color([0.9, 0.9, 0.9])
    except Exception as e:
        print(f"Error creating mesh: {e}")
        print("Continuing without mesh...")
        o3d_mesh = None
    
    # Create coordinate frame visualization
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    
    # Create visualization
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1024, height=768)
        vis.add_geometry(pcd)
        
        if o3d_mesh is not None:
            vis.add_geometry(o3d_mesh)
        
        vis.add_geometry(coordinate_frame)
        
        # Set rendering options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        
        if o3d_mesh is not None:
            render_option.mesh_show_wireframe = True
            
        render_option.light_on = True
        render_option.background_color = np.array([0.8, 0.8, 0.8])
        
        # Set viewpoint
        vis.poll_events()
        vis.update_renderer()
        
        # Capture screenshot
        vis.capture_screen_image(output_path)
        print(f"Visualization saved to: {output_path}")
        
        # Interactive visualization
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"Error in visualization: {e}")

def debug_camera_positions(camera_params):
    """
    Debug function to print and visualize camera positions.
    
    Args:
        camera_params: List of dictionaries with 'R' and 't' keys for each camera
                      or a single dictionary for one camera
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Check if camera_params is a list or a single dictionary
    if isinstance(camera_params, dict):
        # It's a single camera, convert to list
        camera_params = [camera_params]
    
    # Make sure we have a list
    if not isinstance(camera_params, list):
        print("Error: camera_params must be a list of dictionaries or a single dictionary")
        return
    
    # Extract camera positions safely
    positions = []
    for i, params in enumerate(camera_params):
        if 't' not in params:
            print(f"Warning: Camera {i} has no 't' key, skipping")
            continue
        
        t = params['t']
        # Make sure t is a proper array with 3 elements
        if hasattr(t, '__len__') and len(t) == 3:
            positions.append([t[0], t[1], t[2]])
        else:
            print(f"Warning: Camera {i} has invalid position: {t}")
    
    # Convert to numpy array
    if not positions:
        print("Error: No valid camera positions found")
        return
    
    positions = np.array(positions)
    
    # Now positions should be a proper 2D array with shape (num_cameras, 3)
    print(f"Camera positions array shape: {positions.shape}")
    
    # Calculate distance from origin for each camera
    distances = np.sqrt(np.sum(positions**2, axis=1))
    
    print("\nCamera positions:")
    for i, pos in enumerate(positions):
        print(f"Camera {i}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print("\nCamera distances from origin:")
    for i, dist in enumerate(distances):
        print(f"Camera {i}: {dist:.2f} units")
    print(f"Mean distance: {np.mean(distances):.2f} (should be close to 5.0)")
    
    # Calculate heights
    heights = positions[:, 2]
    print("\nCamera heights:")
    for i, height in enumerate(heights):
        print(f"Camera {i}: {height:.2f} units")
    print(f"Mean height: {np.mean(heights):.2f} (should be close to 1.0)")
    
    # Plot camera positions
    plt.figure(figsize=(10, 8))
    
    # Plot camera positions in XY plane
    plt.subplot(1, 2, 1)
    xy_positions = positions[:, :2]  # Just x,y coordinates
    
    for i, pos in enumerate(xy_positions):
        plt.scatter(pos[0], pos[1], c=f'C{i}', marker='^', s=100, label=f'Camera {i}')
    
    # Plot origin
    plt.scatter(0, 0, c='k', marker='*', s=200, label='Origin')
    
    # Draw lines from cameras to origin
    for pos in xy_positions:
        plt.plot([pos[0], 0], [pos[1], 0], 'k--', alpha=0.3)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Camera Positions (Top View)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot camera positions in 3D
    ax = plt.subplot(1, 2, 2, projection='3d')
    
    for i, pos in enumerate(positions):
        ax.scatter(pos[0], pos[1], pos[2], c=f'C{i}', marker='^', s=100, label=f'Camera {i}')
    
    # Plot origin
    ax.scatter(0, 0, 0, c='k', marker='*', s=200, label='Origin')
    
    # Draw lines from cameras to origin
    for pos in positions:
        ax.plot([pos[0], 0], [pos[1], 0], [pos[2], 0], 'k--', alpha=0.3)
    
    ax.set_title('Camera Positions (3D View)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def main(OBJECT):
    output_dir = os.path.join(f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{OBJECT}", "visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing object: {OBJECT}")
    
    rgb_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{OBJECT}"
    masks_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{OBJECT}"
    
    # Dictionary to store files by view index
    view_data = {}
    
    # Process RGB path
    print("\nOrganizing files by view...")
    for filename in sorted(os.listdir(rgb_path)):
        # Extract view index from filename
        # Format examples:
        # camera_params_laptop_real_0.npz
        # render_laptop_real_0.png
        # depth_laptop_real_0_30.png
        
        if not any(ext in filename for ext in ['png', 'npz']):
            continue
            
        parts = filename.split('_')
        view_idx = int(parts[-1].split('.')[0]) if 'params' in filename else int(parts[-2])
        
        if view_idx not in view_data:
            view_data[view_idx] = {'rgb': None, 'depth': None, 'camera': None, 'mask': None}
        
        if filename.startswith('render') and not filename.endswith('results.json'):
            print(f"Found RGB image for view {view_idx}: {filename}")
            view_data[view_idx]['rgb'] = os.path.join(rgb_path, filename)
        elif filename.startswith('depth'):
            print(f"Found depth map for view {view_idx}: {filename}")
            view_data[view_idx]['depth'] = os.path.join(rgb_path, filename)
        elif filename.startswith('camera_params'):
            print(f"Found camera parameters for view {view_idx}: {filename}")
            view_data[view_idx]['camera'] = os.path.join(rgb_path, filename)
    
    # Process masks path
    for filename in sorted(os.listdir(masks_path)):
        if filename.endswith('results.json'):
            # Format: render_laptop_real_0_results.json
            parts = filename.split('_')
            view_idx = int(parts[-2])
            print(f"Found mask file for view {view_idx}: {filename}")
            view_data[view_idx]['mask'] = filename
    
    # Load data in order
    rgb_images = []
    depth_maps = []
    camera_params = []
    masks = []
    
    for view_idx in sorted(view_data.keys()):
        data = view_data[view_idx]
        
        # Verify all required files exist for this view
        if None in data.values():
            missing = [k for k, v in data.items() if v is None]
            print(f"Warning: Missing {missing} for view {view_idx}, skipping...")
            continue
            
        print(f"\nLoading view {view_idx}:")
        # Load RGB
        rgb_images.append(cv2.imread(data['rgb']))
        print(f"Loaded RGB image: {os.path.basename(data['rgb'])}")
        
        # Load depth
        depth_maps.append(cv2.imread(data['depth']))
        print(f"Loaded depth map: {os.path.basename(data['depth'])}")
        
        # Load camera parameters
        camera_params.append(np.load(data['camera']))
        print(f"Loaded camera parameters: {os.path.basename(data['camera'])}")
        
        # Add mask filename
        masks.append(data['mask'])
        print(f"Added mask file: {data['mask']}")
    
    print(f"\nLoaded {len(rgb_images)} complete views")

    # Load mesh
    mesh_path = f'/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{OBJECT}/{OBJECT}.glb'
    print(f"\nLoading mesh from: {mesh_path}")
    scene = trimesh.load(mesh_path)
    mesh = scene.geometry['model']
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    # Process all labels at once
    points_3d, labels = lift_2d_masks_to_3d(
        rgb_images, masks, depth_maps, camera_params, OBJECT, output_dir
    )
    
    # Get unique labels
    unique_labels = np.unique(labels)
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[label] for label in labels])

    # ADD THIS: Visualize point cloud with mesh to check alignment
    visualization_path = os.path.join(output_dir, f'{OBJECT}_point_cloud_with_mesh.png')

    # visualize_blender_depth(depth_maps, visualization_path) # show depth maps
    # debug_camera_positions(camera_params) # debug camera positions

    print(f"\nVisualizing point cloud alignment with mesh...")
    visualize_point_cloud_with_mesh(points_3d, labels, mesh, visualization_path)
    print(f"Saved visualization to: {visualization_path}")
    
    # You might want to add alignment code here based on what you observe
    # For example:
    # aligned_points = align_point_cloud_to_mesh(points_3d, mesh.vertices)
    # return
    # Segment and save individual parts
    output_dir = f'/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{OBJECT}/parts'
    os.makedirs(output_dir, exist_ok=True)
    segment_and_save_parts(mesh, points_3d, numeric_labels, unique_labels, output_dir, OBJECT)
    # segment_and_save_parts(mesh, aligned_points, numeric_labels, unique_labels, output_dir, OBJECT)
if __name__ == "__main__":
    OBJECT = 'laptop_real'
    main(OBJECT)