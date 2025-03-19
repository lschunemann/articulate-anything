import numpy as np

def load_blender_depth_exr(filepath):
    """Load depth from Blender EXR file with better error handling and proper orientation"""
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
            
        # Reshape to 2D with correct dimensions
        # OpenEXR apparently loads in row-major order, so width should be the second dimension
        depth = depth.reshape(height, width)
        
        # Verify the dimensions match the expected resolution
        print(f"Expected dimensions: {width}x{height}, Got: {depth.shape}")
        
        # Transpose if necessary to ensure width x height matches the expected dimensions
        if depth.shape[1] != width or depth.shape[0] != height:
            print("WARNING: Dimensions mismatch, transposing depth map")
            depth = depth.T
            
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

def visualize_camera_orientations(camera_files, axis_length=1.0):
    """
    Visualize camera orientations using the explicit orientation vectors.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different cameras
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    
    for i, camera_file in enumerate(camera_files):
        # Load camera parameters with orientation data
        data = np.load(camera_file, allow_pickle=True)
        t = data['t']
        
        # Try to get explicit orientation data if available
        if 'right' in data and 'up' in data and 'view_dir' in data:
            right = data['right']
            up = data['up']
            view_dir = data['view_dir']
            
            print(f"Camera {i} explicit orientation vectors:")
            print(f"  Right: {right}")
            print(f"  Up: {up}")
            print(f"  View: {view_dir}")
            
            # Plot camera position
            ax.scatter(t[0], t[1], t[2], c=colors[i % len(colors)], marker='^', s=100, label=f'Camera {i}')
            
            # Plot camera axes
            # Right direction (X) - red
            ax.quiver(t[0], t[1], t[2], 
                     right[0], right[1], right[2], 
                     color='r', length=axis_length, arrow_length_ratio=0.2)
            ax.text(t[0] + right[0], t[1] + right[1], t[2] + right[2], "X", color='r')
            
            # Up direction (Y) - green
            ax.quiver(t[0], t[1], t[2], 
                     up[0], up[1], up[2], 
                     color='g', length=axis_length, arrow_length_ratio=0.2)
            ax.text(t[0] + up[0], t[1] + up[1], t[2] + up[2], "Y", color='g')
            
            # View direction (Z) - blue (negative direction because camera looks along -Z)
            ax.quiver(t[0], t[1], t[2], 
                     -view_dir[0], -view_dir[1], -view_dir[2], 
                     color='b', length=axis_length, arrow_length_ratio=0.2)
            ax.text(t[0] - view_dir[0], t[1] - view_dir[1], t[2] - view_dir[2], "Z", color='b')
        else:
            # Fall back to old method if explicit data not available
            R = data['R']
            
            # Print warning
            print(f"WARNING: Camera {i} doesn't have explicit orientation data. Using rotation matrix instead.")
            
            # Plot camera position
            ax.scatter(t[0], t[1], t[2], c=colors[i % len(colors)], marker='^', s=100)
            
            # Use rotation matrix to extract camera axes
            R_cam_to_world = R.T
            
            # Plot camera axes
            for j, color, label in zip(range(3), ['r', 'g', 'b'], ['X', 'Y', 'Z']):
                axis = R_cam_to_world[:, j] * axis_length
                ax.quiver(t[0], t[1], t[2], axis[0], axis[1], axis[2], 
                         color=color, length=axis_length, arrow_length_ratio=0.2)
                ax.text(t[0] + axis[0], t[1] + axis[1], t[2] + axis[2], label, color=color)
        
        # Draw a line from camera to origin
        ax.plot([t[0], 0], [t[1], 0], [t[2], 0], c=colors[i % len(colors)], linestyle='--', alpha=0.5)
    
    # Plot origin (fix the typo here - remove the space)
    ax.scatter(0, 0, 0, c='k', marker='*', s=200, label='Origin')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Explicit Orientation Vectors')
    
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
    plt.savefig('camera_axes.png', dpi=200)
    plt.show()
    
    return fig, ax

def backproject_depth_with_explicit_orientation(depth_map, K, t, right, up, view_dir, max_depth=100.0):
    """
    Backproject using explicit camera orientation vectors.
    
    Args:
        depth_map: Depth map
        K: Intrinsic matrix
        t: Camera position
        right: Right direction vector (X axis in camera space)
        up: Up direction vector (Y axis in camera space)
        view_dir: View direction vector (negative Z axis in camera space)
        max_depth: Maximum valid depth
    """
    import numpy as np
    
    # Ensure depth map has the expected orientation
    height, width = depth_map.shape
    print(f"Processing depth map with shape {depth_map.shape}")
    
    # Create pixel coordinates grid
    v, u = np.indices((height, width))
    
    # Flatten all arrays
    u = u.flatten()
    v = v.flatten()
    z = depth_map.flatten()
    
    # Filter out invalid, zero, or extreme depth values
    valid = (z > 0) & (z < max_depth) & np.isfinite(z)
    u = u[valid]
    v = v[valid]
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
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = -z  # Negate Z because Blender camera looks along -Z axis
    
    # Transform points from camera space to world space using explicit vectors
    right = np.array(right)
    up = np.array(up)
    view_dir = np.array(view_dir)
    
    # Create points in world space
    points_world = []
    for i in range(len(x_cam)):
        # Transform point using camera orientation vectors
        # point = camera_position + x*right + y*up - z*view_dir
        point = t + x_cam[i] * right + y_cam[i] * up - z_cam[i] * view_dir
        points_world.append(point)
    
    return np.array(points_world)

def backproject_depth_fixed(depth_map, K, R, t, max_depth=100.0):
    """
    Correctly backproject depth maps from Blender with proper coordinate transformations.
    
    Args:
        depth_map: Depth map from Blender
        K: Intrinsic camera matrix
        R: Rotation matrix (world to camera)
        t: Camera position in world coordinates
        max_depth: Maximum depth threshold
    """
    import numpy as np
    
    # Ensure depth map has the expected orientation
    height, width = depth_map.shape
    print(f"Processing depth map with shape {depth_map.shape}")
    
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

def visualize_point_clouds_with_explicit_orientation(depth_files, camera_param_files, max_points=5000, max_depth=100.0):
    """
    Visualize point clouds using explicit camera orientations.
    
    Args:
        depth_files: List of depth map file paths
        camera_param_files: List of camera parameter files (.npz)
        max_points: Maximum number of points to plot per camera
        max_depth: Maximum valid depth value
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import random
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different camera viewpoints
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    
    # Store all points for calculating global bounds
    all_points = []
    
    for i, (depth_file, camera_file) in enumerate(zip(depth_files, camera_param_files)):
        print(f"\nProcessing camera {i}:")
        
        # Load depth map
        depth_map = load_blender_depth_exr(depth_file)  # Your working function
        
        # Load camera parameters
        data = np.load(camera_file, allow_pickle=True)
        K = data['K']
        t = data['t']
        
        # Check if we have explicit orientation data
        has_explicit_data = 'right' in data and 'up' in data and 'view_dir' in data
        
        if has_explicit_data:
            right = data['right']
            up = data['up']
            view_dir = data['view_dir']
            
            print(f"Using explicit orientation vectors for camera {i}")
            
            # Back-project using explicit orientation
            points_3d = backproject_depth_with_explicit_orientation(
                depth_map, K, t, right, up, view_dir, max_depth)
        else:
            # Fall back to rotation matrix
            R = data['R']
            print(f"Using rotation matrix for camera {i}")
            
            # Back-project using rotation matrix
            points_3d = backproject_depth_fixed(depth_map, K, R, t, max_depth)
        
        if len(points_3d) == 0:
            print(f"WARNING: No valid points for camera {i}")
            continue
        
        all_points.append(points_3d)
        
        # Check range of points
        print(f"Camera {i} point cloud range:")
        print(f"  X: {np.min(points_3d[:,0]):.2f} to {np.max(points_3d[:,0]):.2f}")
        print(f"  Y: {np.min(points_3d[:,1]):.2f} to {np.max(points_3d[:,1]):.2f}")
        print(f"  Z: {np.min(points_3d[:,2]):.2f} to {np.max(points_3d[:,2]):.2f}")
        
        # Subsample points for visualization
        if len(points_3d) > max_points:
            indices = random.sample(range(len(points_3d)), max_points)
            points_3d = points_3d[indices]
        
        # Plot the points
        color = colors[i % len(colors)]
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c=color, marker='.', s=1, alpha=0.5, label=f'Camera {i}')
        
        # Plot the camera position
        ax.scatter(t[0], t[1], t[2], c=color, marker='^', s=100, alpha=1)
        
        # Plot camera viewing direction
        if has_explicit_data:
            ax.quiver(t[0], t[1], t[2], 
                     -view_dir[0], -view_dir[1], -view_dir[2], 
                     color=color, length=1.0, arrow_length_ratio=0.2)
        
        # Draw a line from camera to origin
        ax.plot([t[0], 0], [t[1], 0], [t[2], 0], c=color, linestyle='--', alpha=0.5)
    
    # Calculate global bounds of all point clouds
    if all_points:
        all_points_combined = np.vstack(all_points)
        print("\nGlobal point cloud range:")
        print(f"  X: {np.min(all_points_combined[:,0]):.2f} to {np.max(all_points_combined[:,0]):.2f}")
        print(f"  Y: {np.min(all_points_combined[:,1]):.2f} to {np.max(all_points_combined[:,1]):.2f}")
        print(f"  Z: {np.min(all_points_combined[:,2]):.2f} to {np.max(all_points_combined[:,2]):.2f}")
    
    # Set plot limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multi-view Point Cloud Reconstruction (Explicit Orientation)')
    
    # Plot origin
    ax.scatter(0, 0, 0, c='k', marker='*', s=200, label='Origin')
    
    # Add a legend
    ax.legend()
    
    # Set equal aspect ratio for all axes
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
    plt.savefig('point_cloud_alignment.png', dpi=200)
    plt.show()
    
    return fig, ax

import glob
# Get the depth and camera parameter files
base_path = "/home/link/DreMa/third_party/articulate-anything/datasets/output_views/laptop_real/"
object_name = "laptop_real"
depth_files = sorted(glob.glob(f"{base_path}depth_{object_name}_*_0030.exr"))
camera_files = sorted(glob.glob(f"{base_path}camera_params_{object_name}_*.npz"))

# Print debug information
print(f"Found {len(depth_files)} depth files and {len(camera_files)} camera parameter files")

visualize_camera_orientations(camera_files)#, output="camera_axes.png")

# Visualize the point clouds
visualize_point_clouds_with_explicit_orientation(depth_files, camera_files)#, output="point_cloud_alignment.png")
