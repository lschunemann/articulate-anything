import numpy as np
# import OpenEXR
# import Imath
import array
import os
import cv2
import matplotlib.pyplot as plt

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

def process_depth_for_blender(depth_map, near_clip=0.1, far_clip=20.0):
    """Process depth map for Blender with proper handling of special values"""
    # Make a copy to avoid modifying the original
    depth = np.copy(depth_map)
    
    # Handle NaN and Inf values
    depth = np.nan_to_num(depth, nan=far_clip, posinf=far_clip, neginf=near_clip)
    
    # Clip values outside the expected range
    depth = np.clip(depth, near_clip, far_clip)
    
    # Optionally smooth the depth map a bit to reduce noise
    # (Uncomment if needed)
    # try:
    #     import cv2
    #     depth = cv2.medianBlur(depth.astype(np.float32), 3)
    # except:
    #     pass
    
    return depth

# Corrected code for the visualize_blender_depth function

def visualize_blender_depth(depth_map, output_path=None, colorbar_width=30):
    """
    Visualizes a Blender depth map with a colorbar.
    
    Args:
        depth_map: The depth map from Blender (numpy array or list)
        output_path: Optional path to save the visualization
        colorbar_width: Width of the colorbar in pixels
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy array if it's a list
    if isinstance(depth_map, list):
        depth_map = np.array(depth_map)
    
    # Ensure depth_map is a 2D array
    if len(depth_map.shape) > 2:
        # If it's a 3D array (like RGB), take the first channel
        depth_map = depth_map[:, :, 0]
    
    # Normalize the depth for visualization (ignoring zero values)
    valid_mask = depth_map > 0
    if valid_mask.any():
        min_val = np.min(depth_map[valid_mask])
        max_val = np.max(depth_map[valid_mask])
        norm_depth = np.zeros_like(depth_map, dtype=np.float32)
        norm_depth[valid_mask] = (depth_map[valid_mask] - min_val) / (max_val - min_val)
    else:
        norm_depth = np.zeros_like(depth_map, dtype=np.float32)
    
    # Create RGB visualization using viridis colormap
    cmap = plt.cm.viridis
    depth_colored = cmap(norm_depth)[:, :, :3]  # Drop alpha channel
    
    # Create colorbar
    height = depth_map.shape[0]
    colorbar = np.zeros((height, colorbar_width, 3))
    
    for i in range(height):
        # Calculate normalized position
        normalized_pos = 1.0 - (i / height)
        # Get color from colormap (returns RGBA, take only RGB)
        color = cmap(normalized_pos)[:3]
        # Apply to entire row of the colorbar
        colorbar[i, :, :] = color
    
    # Combine depth visualization and colorbar
    visualization = np.hstack([depth_colored, colorbar])
    
    # Convert to uint8 for display/saving
    visualization = (visualization * 255).astype(np.uint8)
    
    if output_path:
        plt.imsave(output_path, visualization)
    
    return visualization

# def transform_blender_points(pixels, depth_map, K, R, t, near_clip=0.1, far_clip=5.0, scale_factor=1.3):
#     """
#     Transform 2D image points to 3D world coordinates with adjusted depth scaling
    
#     Args:
#         pixels: Nx2 array of pixel coordinates (x, y)
#         depth_map: Depth map from Blender saved as PNG
#         K: 3x3 camera intrinsic matrix
#         R: 3x3 camera rotation matrix
#         t: 3x1 camera translation vector
#         near_clip: Near clipping plane value
#         far_clip: Far clipping plane value 
#         scale_factor: Additional scaling factor to adjust depth values
#     """
#     # Handle depth map format - make sure it's 2D
#     if len(depth_map.shape) == 3:
#         # Multi-channel - use first channel
#         depth_map = depth_map[:, :, 0]
    
#     # Get pixel depth values
#     h, w = depth_map.shape
#     y_coords = np.clip(pixels[:, 1].astype(int), 0, h-1)
#     x_coords = np.clip(pixels[:, 0].astype(int), 0, w-1)
#     normalized_depths = depth_map[y_coords, x_coords].astype(float)
    
#     # Normalize depths to 0-1 range based on data type
#     if depth_map.dtype == np.uint16:
#         normalized_depths /= 65535.0
#     else:
#         normalized_depths /= 255.0
    
#     # Convert to real-world depths with adjusted scaling
#     # This is the key change - apply scale_factor to get better depth scaling
#     real_depths = near_clip + (normalized_depths * (far_clip - near_clip) * scale_factor)
    
#     # Skip invalid depths
#     valid_mask = real_depths > 0.001  # Small threshold to avoid very close points
#     if not np.any(valid_mask):
#         print("No valid depth values found!")
#         return np.zeros((0, 3))
    
#     valid_pixels = pixels[valid_mask]
#     valid_depths = real_depths[valid_mask]
    
#     print(f"Using {len(valid_pixels)} valid depth values")
#     print(f"Depth range: {np.min(valid_depths):.4f} to {np.max(valid_depths):.4f}, mean: {np.mean(valid_depths):.4f}")
    
#     # Create homogeneous pixel coordinates
#     pixel_homogeneous = np.ones((3, len(valid_pixels)))
#     pixel_homogeneous[0, :] = valid_pixels[:, 0]
#     pixel_homogeneous[1, :] = valid_pixels[:, 1]
    
#     # Convert to camera coordinates
#     K_inv = np.linalg.inv(K)
#     rays = K_inv @ pixel_homogeneous  # shape: (3, N)
    
#     # Normalize rays
#     ray_lengths = np.linalg.norm(rays, axis=0)
#     ray_directions = rays / ray_lengths
    
#     # Scale ray directions by adjusted depth to get points in camera space
#     # Also flip Z direction to account for Blender's camera orientation
#     camera_points = ray_directions * valid_depths
#     camera_points[2, :] = -camera_points[2, :]  # Flip Z axis

#     # Option 2: Swap Y and Z, then negate Z (try this if Option 1 doesn't work)
#     # temp_y = camera_points[1, :].copy()
#     # camera_points[1, :] = -camera_points[2, :]
#     # camera_points[2, :] = temp_y
    
#     # Transform to world coordinates
#     world_points = (R @ camera_points) + t.reshape(3, 1)
    
#     return world_points.T


def align_point_cloud_to_mesh(points, mesh_vertices):
    """Align point cloud to mesh by scaling and centering"""
    if len(points) == 0:
        print("Empty point cloud, cannot align")
        return points
        
    # Get bounding boxes
    pc_min = np.min(points, axis=0)
    pc_max = np.max(points, axis=0) 
    pc_center = (pc_min + pc_max) / 2
    pc_scale = np.max(pc_max - pc_min)
    
    mesh_min = np.min(mesh_vertices, axis=0)
    mesh_max = np.max(mesh_vertices, axis=0)
    mesh_center = (mesh_min + mesh_max) / 2
    mesh_scale = np.max(mesh_max - mesh_min)
    
    # Calculate transformation
    scale_factor = mesh_scale / pc_scale if pc_scale > 0 else 1.0
    
    # Apply transformation: scale and center
    transformed_points = (points - pc_center) * scale_factor + mesh_center
    
    print(f"Point cloud alignment:")
    print(f"  Original bounds: {pc_min} to {pc_max}")
    print(f"  Mesh bounds: {mesh_min} to {mesh_max}")
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  Transformed bounds: {np.min(transformed_points, axis=0)} to {np.max(transformed_points, axis=0)}")
    
    return transformed_points

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

def transform_blender_points(pixels, depth_map, K, R, t, max_depth=100.0):
    """
    Transform 2D pixels with depth values to 3D points in world coordinates,
    with consistent camera orientation handling.
    
    Args:
        pixels: Nx2 array of [x, y] pixel coordinates
        depth_map: Depth map as a 2D numpy array or 3D with channels
        K: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix (world to camera from Blender)
        t: 3x1 translation vector (camera position in world coordinates)
        max_depth: Maximum valid depth value
        
    Returns:
        points_3d: Nx3 array of 3D points in world coordinates
    """
    import numpy as np
    
    # Check depth map dimensions and convert to single channel if needed
    if len(depth_map.shape) == 3:
        print(f"Depth map has 3 channels, converting to single channel")
        # If it's a 3-channel depth map, take the first channel
        # Or average the channels if they contain actual depth info
        if np.array_equal(depth_map[:,:,0], depth_map[:,:,1]) and np.array_equal(depth_map[:,:,0], depth_map[:,:,2]):
            # All channels are identical, just take the first one
            depth_map = depth_map[:,:,0]
        else:
            # Channels are different, use the average
            depth_map = np.mean(depth_map, axis=2)
    
    # Check depth map shape and orientation
    if depth_map.shape[0] == 1080 and depth_map.shape[1] == 1920:
        # This is likely correct (1080×1920)
        pass
    elif depth_map.shape[0] == 1920 and depth_map.shape[1] == 1080:
        # Transpose to correct orientation
        depth_map = depth_map.T
        print("Transposed depth map to correct orientation")
    
    # Print some debug info
    print(f"Depth map shape after processing: {depth_map.shape}")
    print(f"Pixels array shape: {pixels.shape}")
    
    # Extract x and y coordinates
    x_coords = pixels[:, 0]
    y_coords = pixels[:, 1]
    
    # Create valid mask for pixel coordinates
    valid_idx = (x_coords >= 0) & (x_coords < depth_map.shape[1]) & \
                (y_coords >= 0) & (y_coords < depth_map.shape[0])
    
    if not np.any(valid_idx):
        print("No valid pixel coordinates found")
        return np.array([])
    
    # Extract valid coordinates
    x_valid = x_coords[valid_idx]
    y_valid = y_coords[valid_idx]
    
    # Get depth values at these coordinates
    # Convert to integers for indexing
    x_int = np.round(x_valid).astype(int)
    y_int = np.round(y_valid).astype(int)
    
    # Get depth values
    depth_values = np.array([depth_map[y, x] for y, x in zip(y_int, x_int)])
    
    # Filter out invalid or extreme depth values
    depth_valid_mask = (depth_values > 0) & (depth_values < max_depth) & np.isfinite(depth_values)
    
    if not np.any(depth_valid_mask):
        print("No valid depth values found")
        return np.array([])
    
    # Apply further filtering to coordinates and depth
    x_final = x_valid[depth_valid_mask]
    y_final = y_valid[depth_valid_mask]
    z_final = depth_values[depth_valid_mask]
    
    print(f"Final valid points: {len(x_final)}")
    
    # Convert pixel coordinates to camera coordinates
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Calculate 3D coordinates in the camera space
    # In Blender, camera looks along negative Z with Y up
    x_cam = (x_final - cx) * z_final / fx
    y_cam = (y_final - cy) * z_final / fy
    z_cam = -z_final  # Negate Z because Blender camera looks along -Z axis
    
    # Create points in camera space
    points_cam = np.vstack((x_cam, y_cam, z_cam)).T
    
    # Convert from camera to world coordinates
    # R is world-to-camera, so we need its transpose for camera-to-world
    R_cam_to_world = R.T
    
    # Transform points from camera space to world space
    points_world = np.zeros_like(points_cam)
    for i, pt in enumerate(points_cam):
        points_world[i] = R_cam_to_world.dot(pt) + t
    
    return points_world
