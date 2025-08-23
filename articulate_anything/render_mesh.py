import bpy
import math
import numpy as np
from mathutils import Vector, Matrix, Quaternion
import os
import sys

def setup_render_settings():
    """Configure render settings to work without OpenImageDenoiser"""
    # Set render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Disable denoising which requires OpenImageDenoiser
    if hasattr(bpy.context.scene.cycles, 'use_denoising'):
        bpy.context.scene.cycles.use_denoising = False
    
    # For Blender 2.9+ with view layer denoising
    for view_layer in bpy.context.scene.view_layers:
        if hasattr(view_layer, 'cycles'):
            # Disable denoising for view layers
            if hasattr(view_layer.cycles, 'use_denoising'):
                view_layer.cycles.use_denoising = False
            if hasattr(view_layer.cycles, 'denoising_store_passes'):
                view_layer.cycles.denoising_store_passes = False
    
    # Set samples to a reasonable value for speed
    if hasattr(bpy.context.scene.cycles, 'samples'):
        bpy.context.scene.cycles.samples = 32
    
    # Disable caustics for faster rendering
    if hasattr(bpy.context.scene.cycles, 'caustics_reflective'):
        bpy.context.scene.cycles.caustics_reflective = False
    if hasattr(bpy.context.scene.cycles, 'caustics_refractive'):
        bpy.context.scene.cycles.caustics_refractive = False
    
    # Try to use GPU if available
    if hasattr(bpy.context.preferences.addons, 'cycles'):
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        if hasattr(cycles_prefs, 'compute_device_type'):
            try:
                # Prioritize CUDA, then OptiX, then OpenCL, fallback to CPU
                if 'CUDA' in cycles_prefs.get_devices_type('CUDA'):
                    cycles_prefs.compute_device_type = 'CUDA'
                    bpy.context.scene.cycles.device = 'GPU'
                    print("Set Cycles compute device to CUDA (GPU)")
                elif 'OPTIX' in cycles_prefs.get_devices_type('OPTIX'):
                    cycles_prefs.compute_device_type = 'OPTIX'
                    bpy.context.scene.cycles.device = 'GPU'
                    print("Set Cycles compute device to OPTIX (GPU)")
                elif 'OPENCL' in cycles_prefs.get_devices_type('OPENCL'):
                    cycles_prefs.compute_device_type = 'OPENCL'
                    bpy.context.scene.cycles.device = 'GPU'
                    print("Set Cycles compute device to OPENCL (GPU)")
                else:
                    cycles_prefs.compute_device_type = 'NONE' # Fallback to CPU
                    bpy.context.scene.cycles.device = 'CPU'
                    print("No GPU found for Cycles, using CPU.")
            except Exception as e:
                print(f"Error setting Cycles compute device: {e}. Falling back to CPU.")
                bpy.context.scene.cycles.device = 'CPU'
        else:
            print("Cycles preferences compute_device_type not found. Using default CPU.")
            bpy.context.scene.cycles.device = 'CPU'
    
    print("Render settings configured to work without OpenImageDenoiser")

def setup_world_lighting():
    # Access the world settings
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    # Enable nodes
    world.use_nodes = True
    node_tree = world.node_tree
    
    # Clear existing nodes
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)
    
    # Create new nodes
    background = node_tree.nodes.new(type='ShaderNodeBackground')
    background.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # Light gray
    background.inputs['Strength'].default_value = 1.5  # Adjust as needed
    
    output = node_tree.nodes.new(type='ShaderNodeOutputWorld')
    
    # Link nodes
    node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])
    
    print("World lighting setup complete")

def create_camera_with_explicit_orientation(location, name):
    """
    Create a camera that points at the origin with consistent up direction.
    This function explicitly sets the orientation.
    """
    # Create camera
    cam_data = bpy.data.cameras.new(name=name)
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    # Position camera
    cam_obj.location = location
    
    # Explicitly calculate the orientation
    target_point = Vector((0, 0, 0.1))  # Slightly above origin
    
    # Calculate the view direction (from camera to target)
    view_dir = target_point - location
    view_dir.normalize()
    
    # Define world up vector
    world_up = Vector((0, 0, 1))  # Z is up in Blender
    
    # Calculate camera orientation vectors
    forward = -view_dir  # Camera looks along negative Z
    right = view_dir.cross(world_up)  # Right is cross product of view and up
    right.normalize()
    up = right.cross(view_dir)  # Recalculate up to ensure orthogonality
    up.normalize()
    
    # Create rotation matrix from these vectors
    rot_matrix = Matrix((
        (right.x, up.x, -view_dir.x, 0),
        (right.y, up.y, -view_dir.y, 0),
        (right.z, up.z, -view_dir.z, 0),
        (0, 0, 0, 1)
    ))
    
    # Set camera rotation from matrix
    cam_obj.matrix_world = rot_matrix.to_4x4()
    cam_obj.matrix_world.translation = location
    
    # Set camera settings for better depth
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100
    cam_data.lens = 35
    
    print(f"Created camera {name} at location {location}")
    return cam_obj

def get_camera_parameters_explicit(camera):
    """
    Get camera parameters and explicitly calculate rotation matrix.
    """
    # Get camera intrinsics
    render = bpy.context.scene.render
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height
    resolution_x = render.resolution_x
    resolution_y = render.resolution_y
    
    # Calculate intrinsic matrix
    fx = focal_length * resolution_x / sensor_width
    fy = focal_length * resolution_y / sensor_height
    cx = resolution_x / 2
    cy = resolution_y / 2
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # Get camera location
    location = camera.location
    t_world = np.array([location.x, location.y, location.z])
    
    # Calculate normalized view direction (camera to target)
    target_point = Vector((0, 0, 0.1)) # Consistent with camera creation
    view_dir = target_point - location
    view_dir.normalize()
    
    # Define world up vector
    world_up = Vector((0, 0, 1))  # Z is up in Blender
    
    # Calculate camera orientation vectors
    forward = -view_dir  # Camera looks along negative Z
    right = view_dir.cross(world_up)  # Right is cross product of view and up
    right.normalize()
    up = right.cross(view_dir)  # Recalculate up to ensure orthogonality
    up.normalize()
    
    # Create rotation matrix from these vectors (transposed because we need world to camera)
    R = np.array([
        [right.x, right.y, right.z],
        [up.x, up.y, up.z],
        [-view_dir.x, -view_dir.y, -view_dir.z]
    ])
    
    t = t_world # Return camera's world position as t for clarity with pose (R, T_world)
    
    # Extra data for debugging
    extra_data = {
        'view_dir': np.array([view_dir.x, view_dir.y, view_dir.z]),
        'right': np.array([right.x, right.y, right.z]),
        'up': np.array([up.x, up.y, up.z]) 
    }
    
    print(f"Camera parameters calculated for {camera.name}")
    return K, R, t, extra_data

def add_lighting():
    """Add basic lighting to illuminate the scene properly"""
    # Clear existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True) # Use do_unlink for safe removal
    
    # Create a main key light (sun)
    key_light = bpy.data.lights.new(name="Key_Light", type='SUN')
    key_light.energy = 5.0  # Brighter light
    key_obj = bpy.data.objects.new(name="Key_Light", object_data=key_light)
    bpy.context.collection.objects.link(key_obj)
    key_obj.rotation_euler = (0.6, 0.3, 0.9)  # Angled to illuminate front of objects
    
    # Add a fill light for shadows
    fill_light = bpy.data.lights.new(name="Fill_Light", type='POINT')
    fill_light.energy = 2000.0  # Strong point light
    fill_obj = bpy.data.objects.new(name="Fill_Light", object_data=fill_light)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.location = (-5, -5, 5)  # Position to the front-left-top
    
    print("Added lighting to the scene")

def setup_depth_nodes(output_folder, object_name, index):
    """Set up depth nodes to output raw depth values in meters"""
    # switch on nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = 0, 0

    # Check available outputs
    available_outputs = {}
    for output in render_layers.outputs:
        available_outputs[output.name] = output
    
    print(f"Available render layer outputs: {list(available_outputs.keys())}")
    
    # Find depth output
    depth_output = None
    # Prioritize 'Depth' (Blender 4.0+) then 'Z' (older Blenders)
    for name in ['Depth', 'Z']:
        if name in available_outputs:
            depth_output = available_outputs[name]
            print(f"Using depth output: {name}")
            break
    
    if depth_output is None:
        print("ERROR: No depth output found! Make sure 'use_pass_z' is enabled on the view layer.")
        return False

    # RGB output for render
    rgb_output = tree.nodes.new('CompositorNodeOutputFile')
    rgb_output.location = 300, 300
    rgb_output.base_path = output_folder
    rgb_output.file_slots[0].path = f"render_{object_name}_{index}_"
    rgb_output.format.file_format = 'PNG'
    rgb_output.format.color_mode = 'RGB'
    rgb_output.format.color_depth = '8'
    
    # Connect image output directly
    if 'Image' in available_outputs:
        links.new(available_outputs['Image'], rgb_output.inputs[0])
    else:
        print("WARNING: 'Image' output not found in render layers for RGB render.")
    
    # ----- RAW DEPTH OUTPUT (Direct to EXR) -----
    
    # Create EXR output for raw depth data - no processing
    depth_exr_output = tree.nodes.new('CompositorNodeOutputFile')
    depth_exr_output.location = 300, 0
    depth_exr_output.base_path = output_folder
    depth_exr_output.file_slots[0].path = f"depth_raw_{object_name}_{index}_"
    depth_exr_output.file_slots[0].use_node_format = False
    depth_exr_output.file_slots[0].format.file_format = 'OPEN_EXR'
    depth_exr_output.file_slots[0].format.color_depth = '32'
    depth_exr_output.file_slots[0].format.color_mode = 'BW' # Single channel for depth
    
    # Connect raw depth directly - no math nodes or processing
    links.new(depth_output, depth_exr_output.inputs[0])
    
    # ----- VISUALIZATION ONLY (doesn't affect raw depth) -----
    
    # Create a separate branch for visualization that doesn't affect the raw output
    # Use a reroute node to split the path
    reroute = tree.nodes.new('NodeReroute')
    reroute.location = 150, -100
    links.new(depth_output, reroute.inputs[0])
    
    # Normalize for visualization
    normalize = tree.nodes.new('CompositorNodeNormalize')
    normalize.location = 300, -100
    links.new(reroute.outputs[0], normalize.inputs[0])
    
    # Invert so closer objects are brighter
    invert = tree.nodes.new('CompositorNodeInvert')
    invert.location = 450, -100
    links.new(normalize.outputs[0], invert.inputs[1])
    
    # Color ramp for visualization
    color_ramp = tree.nodes.new('CompositorNodeValToRGB')
    color_ramp.location = 600, -100
    
    # Configure color ramp
    color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[1]) # Remove default second
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Black (far)
    
    elem1 = color_ramp.color_ramp.elements.new(0.2)
    elem1.color = (0.0, 0.0, 1.0, 1.0)  # Blue
    
    elem2 = color_ramp.color_ramp.elements.new(0.4)
    elem2.color = (0.0, 1.0, 0.5, 1.0)  # Teal
    
    elem3 = color_ramp.color_ramp.elements.new(0.6)
    elem3.color = (1.0, 1.0, 0.0, 1.0)  # Yellow
    
    elem4 = color_ramp.color_ramp.elements.new(0.8)
    elem4.color = (1.0, 0.0, 0.0, 1.0)  # Red (near)
    
    elem5 = color_ramp.color_ramp.elements.new(1.0)
    elem5.color = (1.0, 1.0, 1.0, 1.0)  # White (near)
    
    links.new(invert.outputs[0], color_ramp.inputs[0])
    
    # Output for visualization
    depth_viz_output = tree.nodes.new('CompositorNodeOutputFile')
    depth_viz_output.location = 800, -100
    depth_viz_output.base_path = output_folder
    depth_viz_output.file_slots[0].path = f"depth_viz_{object_name}_{index}_"
    depth_viz_output.format.file_format = 'PNG'
    depth_viz_output.format.color_mode = 'RGB'
    depth_viz_output.format.color_depth = '8'
    
    links.new(color_ramp.outputs[0], depth_viz_output.inputs[0])
    
    # Update the node tree
    tree.update_tag()
    
    print(f"Depth nodes setup complete for {object_name} view {index}")
    return True

def setup_render_paths(output_folder, object_name, index):
    """Configure render paths and output settings"""
    # Set the output path for the main render
    bpy.context.scene.render.filepath = os.path.join(output_folder, f"render_{object_name}_{index}")
    
    # Set up EXR format for depth (this sets scene-wide default, but CompositorNodeOutputFile overrides)
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.color_depth = '32'
    
    # Enable depth pass explicitly
    for view_layer in bpy.context.scene.view_layers:
        view_layer.use_pass_z = True
    
    print(f"Configured render paths for {object_name} view {index}")

def fix_scene_objects():
    """Inspect and fix scene objects to ensure they render properly"""
    print("Fixing scene objects...")
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("No mesh objects found to fix.")
        return
    
    print(f"Found {len(mesh_objects)} mesh objects for potential fixing.")
    
    for obj in mesh_objects:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.select_set(True) 
        
        # For mesh objects, make sure they have a material that will render
        if not obj.data.materials:
            # Create a basic material
            mat_name = f"Material_{obj.name}"
            if mat_name not in bpy.data.materials:
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                # Set a default principled BSDF material
                if mat.node_tree.nodes.get('Principled BSDF'):
                    principled = mat.node_tree.nodes['Principled BSDF']
                    principled.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0) # Light gray
                    principled.inputs['Roughness'].default_value = 0.6
                    principled.inputs['Metallic'].default_value = 0.0
                print(f"Created and assigned default material to {obj.name}")
            else:
                obj.data.materials.append(bpy.data.materials[mat_name])
                print(f"Assigned existing material '{mat_name}' to {obj.name}")
    
    print("Scene objects fixed.")

def import_mesh(filepath):
    """Import a mesh file using the appropriate operator"""
    print(f"Attempting to import mesh: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    try:
        # Deselect all before import to easily identify new objects
        bpy.ops.object.select_all(action='DESELECT')

        if file_ext == '.obj':
            # In Blender 4.0.2, OBJ import is wm.obj_import
            if hasattr(bpy.ops.wm, 'obj_import'):
                bpy.ops.wm.obj_import(filepath=filepath)
            else:
                bpy.ops.import_scene.obj(filepath=filepath)
        elif file_ext in ['.glb', '.gltf']:
            # In Blender 4.0.2, GLTF import is import_scene.gltf
            try:
                bpy.ops.import_scene.gltf(filepath=filepath)
            except: # Fallback for older/different Blender versions
                bpy.ops.wm.gltf_import(filepath=filepath) 
        else:
            print(f"Unsupported file format: {file_ext}")
            return False
        
        print(f"Successfully imported: {filepath}")
        # Ensure scene is updated after import
        bpy.context.view_layer.update()
        return True
    except Exception as e:
        print(f"Error importing mesh: {e}")
        print("You may need to enable the glTF addon: bpy.ops.preferences.addon_enable(module='io_scene_gltf2')")
        return False
    
def remove_default_cube_and_cleanup():
    """Remove default cube and ensure only geometry objects remain visible"""
    # Check for and remove default cube
    if 'Cube' in bpy.data.objects:
        cube_obj = bpy.data.objects['Cube']
        bpy.data.objects.remove(cube_obj, do_unlink=True)
        print("Removed default cube 'Cube'")
    
    print("Cleanup of default objects completed.")

def get_overall_bounding_box_of_meshes():
    """
    Calculates the combined world-space bounding box of all visible mesh objects in the scene.
    Returns (min_coords, max_coords, dimensions, center_coords) or None if no meshes found.
    """
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    
    has_meshes = False
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and not obj.hide_render: # Only consider renderable meshes
            has_meshes = True
            # Get the world-space bounding box corners
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                min_x = min(min_x, world_corner.x)
                max_x = max(max_x, world_corner.x)
                min_y = min(min_y, world_corner.y)
                max_y = max(max_y, world_corner.y)
                min_z = min(min_z, world_corner.z)
                max_z = max(max_z, world_corner.z)
    
    if not has_meshes:
        return None # Indicates no renderable mesh found
            
    bbox_min = Vector((min_x, min_y, min_z))
    bbox_max = Vector((max_x, max_y, max_z))
    
    dims = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2.0
    
    return bbox_min, bbox_max, dims, center

def main():
    # Set render settings (before clearing scene, global settings)
    bpy.context.scene.render.image_settings.file_format = 'PNG' # For default render output
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1280
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.use_compositing = True # Enable compositing for depth nodes
    bpy.context.scene.use_nodes = True # Ensure nodes are enabled

    # Configure render engine specific settings
    setup_render_settings()
    
    # --- Clear existing scene ---
    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    # Delete selected objects
    bpy.ops.object.delete(use_global=False)
    # Ensure all data blocks are also cleared for a truly clean slate (optional, but good for multiple runs)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    
    # Set up world lighting (should be done after clearing scene, before adding objects)
    setup_world_lighting()
    
    # --- Parse command line arguments ---
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    
    if len(argv) < 1:
        print("Usage: blender --background --python script.py -- <mesh_name>")
        print("Example: blender --background --python script.py -- object_001")
        sys.exit(1) # Exit if essential argument is missing
    
    mesh_base_name = argv[0] 
    num_cameras = 8 # Fixed to 8 cameras
    
    print(f"Processing object: {mesh_base_name}")
    print(f"Fixed number of cameras: {num_cameras}")
    
    # Set output folder structure based on mesh_base_name
    output_base_dir = "/home/link/DreMa/third_party/articulate-anything/datasets/output_views/"
    output_folder = os.path.join(output_base_dir, mesh_base_name) + os.sep # Ensure trailing slash
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Construct the full path to the mesh file (assuming .glb extension)
    full_mesh_path = os.path.join(output_folder, mesh_base_name + ".glb") 
    
    # Import the mesh
    print(f"Importing mesh from: {full_mesh_path}")
    if not import_mesh(full_mesh_path):
        print(f"Failed to import mesh from {full_mesh_path}. Exiting.")
        sys.exit(1) # Exit if import fails
    
    # Ensure scene is fully updated after import before bounding box calculation
    bpy.context.view_layer.update()
    
    # Perform cleanup and assign materials
    remove_default_cube_and_cleanup()
    fix_scene_objects() 
    
    # Add scene lighting
    add_lighting() 
    
    # --- Dynamic Camera Parameter Calculation ---
    bbox_result = get_overall_bounding_box_of_meshes()

    if bbox_result is None: # No renderable mesh found
        print("No renderable mesh objects found after import. Cannot determine dynamic camera parameters. Using fallback values.")
        dynamic_radius = 3.0
        dynamic_height = 0.5
    else:
        bbox_min_world, bbox_max_world, dims_world, center_world = bbox_result
        
        print(f"Object world bounding box (min): {bbox_min_world}")
        print(f"Object world bounding box (max): {bbox_max_world}")
        print(f"Object world dimensions: {dims_world}")
        print(f"Object world center: {center_world}")

        # Calculate dynamic height: 0.5 meters above the highest point of the mesh
        dynamic_height = bbox_max_world.z + 0.1
        # Ensure a minimum height to prevent camera from being too low, e.g., 0.5m above world origin
        dynamic_height = max(dynamic_height, 0.5) 
        
        # Calculate dynamic radius: based on the maximum of X and Y dimensions + 1.0 meter padding
        max_xy_dimension = max(dims_world.x, dims_world.y, dims_world.z)
        dynamic_radius = max_xy_dimension + 1.5
        # Ensure a minimum radius to avoid cameras being too close for tiny objects
        dynamic_radius = max(dynamic_radius, 1.5) # Minimum radius of 1.5 meters
        
        print(f"Dynamically calculated camera parameters: radius={dynamic_radius:.2f}, height={dynamic_height:.2f}")

    # Set the calculated radius and height for camera creation loop
    camera_orbit_radius = dynamic_radius
    camera_orbit_height = dynamic_height

    # Create cameras and render from each viewpoint
    for i in range(num_cameras):
        # Calculate camera position in circle around origin
        angle = (2 * math.pi * i) / num_cameras
        x = camera_orbit_radius * math.cos(angle)
        y = camera_orbit_radius * math.sin(angle)
        z = camera_orbit_height
        
        camera = create_camera_with_explicit_orientation(Vector((x, y, z)), f"Camera_{i}")
        
        # Set active camera 
        bpy.context.scene.camera = camera

        # Setup render paths and compositor nodes for RGB, raw depth, and depth visualization
        setup_render_paths(output_folder, mesh_base_name, i)
        setup_depth_nodes(output_folder, mesh_base_name, i)
        
        # Render the current view
        bpy.ops.render.render(write_still=True)
        
        # Get and save camera parameters with explicit calculation
        K, R, t, extra_data = get_camera_parameters_explicit(camera)
        
        # Save parameters with extra orientation data for debugging
        np.savez(f"{output_folder}camera_params_{mesh_base_name}_{i}.npz", 
                 K=K, R=R, t=t, 
                 view_dir=extra_data['view_dir'],
                 right=extra_data['right'], 
                 up=extra_data['up'])
        
        print(f"Completed rendering view {i+1}/{num_cameras}")
        
        # Clean up camera object for next iteration. Blender will manage the data block.
        bpy.data.objects.remove(camera, do_unlink=True)
        
    print("All renders completed.")

if __name__ == "__main__":
    # Print system info for debugging
    print(f"Python: {sys.version}")
    print(f"Blender: {bpy.app.version_string}")
    print(f"OS: {os.name}")
    
    try:
        main()
        print("Script completed successfully")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()