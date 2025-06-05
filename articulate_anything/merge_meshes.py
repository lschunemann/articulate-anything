import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

base_path = "/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/washing_mashine_multi-view/output/"

# Import meshes
# bpy.ops.wm.obj_import(filepath=base_path + "drawer_open_multi-view_leg_leg_0.obj")
# bpy.ops.wm.obj_import(filepath=base_path + "drawer_open_multi-view_leg_leg_1.obj")
# bpy.ops.wm.obj_import(filepath=base_path + "drawer_open_multi-view_leg_leg_10.obj")
# bpy.ops.wm.obj_import(filepath=base_path + "drawer_open_multi-view_leg_leg_11.obj")
# bpy.ops.wm.obj_import(filepath=base_path + "drawer_open_multi-view_top_surface_top_surface_0.obj")
# bpy.ops.wm.obj_import(filepath=base_path + "drawer_open_multi-view_top_surface_top_surface_1.obj")
# bpy.ops.wm.obj_import(filepath=base_path + "drawer_open_multi-view_top_surface_top_surface_2.obj")

bpy.ops.wm.obj_import(filepath=base_path + "washing_mashine_multi-view_door_0.obj")
bpy.ops.wm.obj_import(filepath=base_path + "washing_mashine_multi-view_metal_0.obj")
# bpy.ops.wm.obj_import(filepath=base_path + "laptop_multi-view_wooden_stand_wooden_stand_4.obj")

# bpy.ops.wm.obj_import(filepath="/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/laptop_multi-view/output/laptop_multi-view_wooden_stand_wooden_stand_4.obj")

# Select all objects
bpy.ops.object.select_all(action='SELECT')

# Set the active object (required for join operation)
if len(bpy.context.selected_objects) > 0:
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    
    # Join selected objects
    bpy.ops.object.join()
    
    # Export the result
    bpy.ops.wm.obj_export(filepath=base_path + "base_0.obj")