import bpy
import bmesh
import numpy as np
import sys
import os
from mathutils import Vector

def remove_below_plane(obj, plane_normal=(0, 0, 1), plane_point=(0, 0, 0)):
    """
    Remove all vertices and faces from a mesh that are below a specified plane
    
    Args:
        obj: The Blender mesh object to modify
        plane_normal: Normal vector of the plane (default: upward Z axis)
        plane_point: Point on the plane (default: origin)
    
    Returns:
        The number of vertices removed
    """
    # Ensure we're in object mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Select and make the object active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Create BMesh for easier manipulation
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    
    # Convert the normal and point to vectors
    normal = Vector(plane_normal).normalized()
    point = Vector(plane_point)
    
    # Identify vertices below the plane
    vertices_to_remove = []
    for v in bm.verts:
        # Calculate signed distance from the vertex to the plane
        # Positive means above the plane, negative means below
        signed_distance = normal.dot(v.co - point)
        
        if signed_distance < 0:  # Vertex is below the plane
            vertices_to_remove.append(v)
    
    # Count vertices to be removed
    num_to_remove = len(vertices_to_remove)
    
    # Remove the vertices (and connected faces/edges)
    if vertices_to_remove:
        bmesh.ops.delete(bm, geom=vertices_to_remove, context='VERTS')
    
    # Update the mesh
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    
    return num_to_remove

def process_mesh(input_path, output_path, plane_normal=(0, 0, 1), plane_point=(0, 0, 0)):
    """
    Load a mesh, remove parts below a plane, and save the result
    
    Args:
        input_path: Path to the input mesh file
        output_path: Path to save the processed mesh
        plane_normal: Normal vector of the plane
        plane_point: Point on the plane
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Import the mesh
    if input_path.lower().endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=input_path)
    elif input_path.lower().endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=input_path)
    else:
        print(f"Unsupported file format: {input_path}")
        return False
    
    # Get the imported object
    if not bpy.context.selected_objects:
        print(f"No objects were imported from {input_path}")
        return False
    
    # Process each selected mesh object
    total_removed = 0
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            removed = remove_below_plane(obj, plane_normal, plane_point)
            total_removed += removed
            print(f"Removed {removed} vertices from {obj.name}")
    
    print(f"Total vertices removed: {total_removed}")
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export the result
    # if output_path.lower().endswith('.obj'):
    bpy.ops.wm.obj_export(filepath=output_path[:-3]+'obj')
    # elif output_path.lower().endswith('.glb'):
        # bpy.ops.export_scene.gltf(filepath=output_path)
    # else:
        # print(f"Unsupported output format: {output_path}")
        # return False
    
    print(f"Successfully exported processed mesh to: {output_path}")
    return True

# Get command line arguments after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

# Check for enough arguments
if len(argv) < 2:
    print("Usage: blender --background --python cut_mesh.py -- input.obj output.obj [nx ny nz px py pz]")
    print("  input.obj: Path to the input mesh file")
    print("  output.obj: Path to save the processed mesh")
    print("  nx ny nz: Normal vector of the cutting plane (default: 0 0 1, i.e., upward Z)")
    print("  px py pz: Point on the cutting plane (default: 0 0 0, i.e., origin)")
    sys.exit(1)

input_path = argv[0]
output_path = argv[1]

# Default plane: XY plane at origin (cuts everything below z=0)
plane_normal = (0, 0, 1)
plane_point = (0, 0, 0)

# Override defaults if provided
if len(argv) >= 5:
    plane_normal = (float(argv[2]), float(argv[3]), float(argv[4]))

if len(argv) >= 8:
    plane_point = (float(argv[5]), float(argv[6]), float(argv[7]))

print(f"Input path: {input_path}")
print(f"Output path: {output_path}")
print(f"Plane normal: {plane_normal}")
print(f"Plane point: {plane_point}")

# Process the mesh
process_mesh(input_path, output_path, plane_normal, plane_point)