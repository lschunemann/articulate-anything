import os
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict, Any
from actor_critic import (
    actor_critic_loop,
    error_handler,
)
from articulate_anything.utils.utils import (
    create_task_config,
    join_path,
    Steps,
)
from articulate_anything.agent.actor.link_placement.link_placement import LinkPlacementActor
from articulate_anything.agent.critic.link_placement.link_critic import LinkCritic
from articulate_anything.preprocess.preprocess_partnet import (
    render_partnet_obj,
    preprocess_partnet_object,
    render_object
)
from articulate_anything.agent.actor.mesh_retrieval.partnet_mesh_retrieval import (
    add_mesh_to_out_folder,
    write_box_layout_to_link_summary,
)
from articulate_anything.agent.actor.mesh_retrieval.obj_selector import (
    make_obj_selector,
)


def preprocess(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> Dict[str, Any]:
    """Prepare the environment and configuration based on the modality."""
    modality_processors = {
        "partnet": process_partnet,
        "image": process_visual,
        "video": process_visual,
        "text": process_text,
        "generated": process_generated
    }
    processor = modality_processors.get(cfg.modality)
    if not processor:
        raise ValueError(f"Preprocess failed. Unsupported modality: {cfg.modality}")

    cfg = processor(prompt, steps, gpu_id, cfg)

    steps.add_step("Link actor", [])  # list, one per iteration
    steps.add_step("Link critic", [])  # list, one per iteration
    return {"cfg": cfg, "prompt": prompt}


def process_visual(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    obj_selector = steps["Mesh Retrieval"]["Object Selection"]
    # we will articulate the object that was selected by template match
    selected_obj_id = obj_selector.load_prediction()["obj_id"]
    render_partnet_obj(selected_obj_id, gpu_id, cfg, "stationary")
    cfg.dataset_dir = join_path(cfg.dataset_dir, selected_obj_id)
    cfg = OmegaConf.create(cfg)  # copy for link_placement
    cfg.prompt = selected_obj_id
    temp_cfg = OmegaConf.create(cfg)
    temp_cfg.dataset_dir = os.path.dirname(cfg.dataset_dir)  # because
    # preprocess_partnet_object will change link_cfg.dataset_dir
    preprocess_partnet_object(selected_obj_id, gpu_id, temp_cfg)
    return cfg

def process_generated(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    """Process generated modality with custom mesh files."""
    obj_selector = steps["Mesh Retrieval"]["Object Selection"]
    selection_info = obj_selector.load_prediction()
    
    # Use the generated object ID
    selected_obj_id = selection_info["obj_id"]
    # segmented_mesh_dir = selection_info.get("segmented_mesh_dir")
    segmented_mesh_dir = cfg.segmented_mesh_dir
    
    # Create the necessary directory structure
    import os, shutil
    import logging
    import re

    segmented_mesh_dir = os.path.join(segmented_mesh_dir, "output")#/parts")
    
    # Create absolute paths to avoid nesting issues
    base_dataset_dir = os.path.dirname(cfg.dataset_dir) if cfg.dataset_dir.endswith(selected_obj_id) else cfg.dataset_dir
    base_output_dir = os.path.dirname(cfg.out_dir) if cfg.out_dir.endswith(selected_obj_id) else cfg.out_dir
    
    # Set up the dataset and output directories
    obj_dataset_dir = join_path(base_dataset_dir, cfg.task)
    obj_output_dir = join_path(base_output_dir, selected_obj_id)
    
    logging.info(f"Creating directories: {obj_dataset_dir} and {obj_output_dir}")
    os.makedirs(obj_dataset_dir, exist_ok=True)
    os.makedirs(obj_output_dir, exist_ok=True)
    
    # Get mesh files
    mesh_files = [f for f in os.listdir(segmented_mesh_dir) if f.endswith(('.obj'))]#, '.glb', '.stl'))]
    mesh_files_glb = [f for f in os.listdir(segmented_mesh_dir) if f.endswith(('.glb'))]#, '.glb', '.stl'))]
    
    if not mesh_files:
        raise ValueError(f"No mesh files found in {segmented_mesh_dir}")
    
    # Use VLM to identify which part should be the base
    base_part_name = identify_base_part(mesh_files_glb, segmented_mesh_dir, gpu_id, cfg)
    
    # Clean up link names to avoid spaces and special characters
    link_names = []
    base_index = None

    # logging.info(f"Link names before cleaning: {mesh_files}")
    
    for i, mesh_file in enumerate(mesh_files):
        original_name = os.path.splitext(mesh_file)[0]
        # # Replace spaces and special characters with underscores
        # clean_name = re.sub(r'[^a-zA-Z0-9]', '_', original_name)
        
        # Check if this is the base part
        if original_name == base_part_name:
            link_names.append("base")
            base_index = i
        else:
            link_names.append(original_name)

    # deduplicate
    # logging.info(f"Link names: {link_names}")
    # link_names = list(set(link_names))
    # logging.info(f"after deduplication: {link_names}")
    
    # If no base was found, use the first part as base
    if base_index is None:
        logging.warning("No base part identified by VLM, using first part as base")
        link_names[0] = "base"
        base_index = 0

    # Copy over material if it exists
    try:
        shutil.copy(join_path(segmented_mesh_dir, 'material_0.png'), join_path(obj_dataset_dir, 'material_0.png'))
    except Exception:
        logging.info("Failed to copy material")
    
    # Copy mesh files to the dataset directory with new names
    for i, mesh_file in enumerate(mesh_files):
        src_path = join_path(segmented_mesh_dir, mesh_file)
        new_filename = f"{link_names[i]}{os.path.splitext(mesh_file)[1]}"
        dst_path = join_path(obj_dataset_dir, new_filename)
        
        # Copy the OBJ file
        shutil.copy(src_path, dst_path)
        
        # Get the original MTL filename from the OBJ file
        original_mtl_filename = None
        with open(src_path, 'r') as f:
            for line in f:
                if line.startswith('mtllib '):
                    original_mtl_filename = line.strip().split(' ', 1)[1]
                    break
        
        if original_mtl_filename:
            # Copy and rename the MTL file to match the new OBJ filename
            original_mtl_path = join_path(segmented_mesh_dir, original_mtl_filename)
            new_mtl_filename = f"{link_names[i]}.mtl"
            new_mtl_path = join_path(obj_dataset_dir, new_mtl_filename)
            
            if os.path.exists(original_mtl_path):
                shutil.copy(original_mtl_path, new_mtl_path)
                
                # Update the MTL reference in the copied OBJ file
                with open(dst_path, 'r') as f:
                    obj_content = f.read()
                
                updated_obj_content = obj_content.replace(f'mtllib {original_mtl_filename}', f'mtllib {new_mtl_filename}')
                
                with open(dst_path, 'w') as f:
                    f.write(updated_obj_content)
                
                logging.info(f"Updated MTL reference in {new_filename} from {original_mtl_filename} to {new_mtl_filename}")
            else:
                logging.warning(f"MTL file {original_mtl_filename} referenced in {mesh_file} not found")
    
    # Create a URDF file for the object
    urdf_content = f"""<?xml version="1.0" ?>
<robot name="{selected_obj_id}">
  <link name="base">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="base{os.path.splitext(mesh_files[base_index])[1]}" scale="1 1 1"/>
      </geometry>
      <material name="material_base">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="base{os.path.splitext(mesh_files[base_index])[1]}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
"""
    
    # Add additional links for each mesh file (except the base)
    for i, mesh_file in enumerate(mesh_files):
        if i == base_index:  # Skip base, already added
            continue
            
        part_name = link_names[i]
        urdf_content += f"""
  <link name="{part_name}">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{part_name}{os.path.splitext(mesh_file)[1]}" scale="1 1 1"/>
      </geometry>
      <material name="material_{part_name}">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{part_name}{os.path.splitext(mesh_file)[1]}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
  <joint name="base_to_{part_name}" type="fixed">
    <parent link="base"/>
    <child link="{part_name}"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
"""
    
    urdf_content += "</robot>"
    
    # Save the URDF file
    urdf_file = join_path(obj_dataset_dir, "mobility.urdf")
    with open(urdf_file, "w") as f:
        f.write(urdf_content)
    
    logging.info(f"Created URDF file at {urdf_file}")
    
    # Create a link summary file - using cleaned part names
    link_summary = f"object_id: {selected_obj_id}\n    Robot Link Summary:\n    - base\n"
    
    # Add other parts to the link summary
    for i, part_name in enumerate(link_names):
        if part_name != "base":  # Skip the base
            link_summary += f"    - {part_name}\n"
    
    # Save the link summary
    link_summary_path = join_path(obj_dataset_dir, "link_summary.txt")
    with open(link_summary_path, "w") as f:
        f.write(link_summary)
    
    logging.info(f"Created link summary at {link_summary_path}")
    
    # Create semantics.txt file
    # Format: link_name joint_type semantic_label
    semantics_content = ""
    for part_name in link_names:
        # Use "fixed" as the joint type for all parts
        joint_type = "fixed"
        # Use the part name as the semantic category
        semantic_label = part_name
        semantics_content += f"{part_name} {joint_type} {semantic_label}\n"
    
    semantics_path = join_path(obj_dataset_dir, "semantics.txt")
    with open(semantics_path, "w") as f:
        f.write(semantics_content)
    
    logging.info(f"Created semantics file at {semantics_path}")
    
    # Render frontview using Open3D
    frontview_path = join_path(obj_dataset_dir, "robot_frontview.png")
    # render_frontview_with_open3d(segmented_mesh_dir, frontview_path)
    render_frontview_with_matplotlib(segmented_mesh_dir, frontview_path, cfg)
    render_object(join_path(obj_dataset_dir, "mobility.urdf"), gpu_id, OmegaConf.load("/home/link/DreMa/third_party/articulate-anything/conf/simulator/default.yaml"), "stationary")

    # Copy the frontview image to the link placement output directory
    link_placement_dir = os.path.dirname(cfg.out_dir)
    output_frontview_path = join_path(link_placement_dir, "robot_frontview.png")
    os.makedirs(os.path.dirname(output_frontview_path), exist_ok=True)
    shutil.copy(frontview_path, output_frontview_path)

    # Also copy to the specific iteration directory
    iter_seed_dir = cfg.out_dir
    iter_frontview_path = join_path(iter_seed_dir, "robot_frontview.png")
    shutil.copy(frontview_path, iter_frontview_path)

    logging.info(f"Copied frontview image to {output_frontview_path} and {iter_frontview_path}")
    
    # Set up the configuration for link placement
    cfg.dataset_dir = obj_dataset_dir
    cfg.out_dir = obj_output_dir
    cfg.prompt = selected_obj_id # TODO: maybe don't overwrite? Error in critic
    
    return cfg

def render_frontview_with_matplotlib(mesh_dir, output_path, cfg):
    """Render a front view of the entire object (all meshes combined) as one unified entity."""
    import os
    import trimesh
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import logging
    from scipy.spatial.transform import Rotation as R
    
    logging.info(f"Rendering front view of the entire object in {mesh_dir} using matplotlib...")

    # Get all mesh files
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
    
    if not mesh_files:
        raise ValueError(f"No mesh files found in {mesh_dir}")
    
    # Combine all vertices and faces across meshes
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0  # To handle global indexing for faces
    
    for mesh_file in mesh_files:
        try:
            # Load mesh
            mesh_path = os.path.join(mesh_dir, mesh_file)
            mesh_or_scene = trimesh.load(mesh_path)

            if isinstance(mesh_or_scene, trimesh.Scene):
                # Extract the first geometry from the scene
                if mesh_or_scene.geometry:
                    mesh = list(mesh_or_scene.geometry.values())[0]
                else:
                    raise RuntimeError(f"No geometries found in scene for {mesh_file}.")
            elif isinstance(mesh_or_scene, trimesh.Trimesh):
                mesh = mesh_or_scene
            else:
                raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")
            
            # Accumulate vertices and faces
            combined_vertices.extend(mesh.vertices)
            combined_faces.extend(mesh.faces + vertex_offset)
            vertex_offset += len(mesh.vertices)  # Update face indices based on vertex offset
            
        except Exception as e:
            logging.warning(f"Could not process mesh {mesh_file}: {str(e)}")
    
    # Ensure we have loaded vertices
    if not combined_vertices:
        raise ValueError("Failed to load any meshes for rendering.")
    
    # Convert vertices and faces to numpy arrays
    combined_vertices = np.array(combined_vertices)
    combined_faces = np.array(combined_faces)

    # Apply global rotation to all vertices
    if False: #if cfg.urdf.rotation_pose is not None: # TODO: use VLM to decide rotation
        rotation = R.from_euler('z', 90, degrees=True).as_matrix()  # Rotate 90 degrees around the Y-axis
        rotated_vertices = combined_vertices @ rotation.T  # Rotate vertices
    else: # 90 for drawer & laptop, -90 for microwave & box
        rotation = R.from_euler('z', -90, degrees=True).as_matrix()  # Rotate 90 degrees around the Y-axis
        rotated_vertices = combined_vertices @ rotation.T  # Rotate vertices
    # else:
    #     rotated_vertices = combined_vertices
    
    # Plot the rotated object
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2], 
                    triangles=combined_faces, alpha=0.7, 
                    edgecolor='gray', linewidth=0.2)

    # Set the view to front
    ax.view_init(elev=0, azim=0)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Remove axis labels and ticks
    ax.set_axis_off()
    
    # Set tight layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Rendered front view saved to {output_path}")


def identify_base_part(mesh_files, segmented_mesh_dir, gpu_id, cfg):
    """Use OpenAI API to identify which part should be the base using trimesh + matplotlib."""
    import os
    import logging
    import base64
    import json
    import openai
    import trimesh
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image
    
    logging.info("Identifying base part using OpenAI API with trimesh + matplotlib...")
    
    # Render individual parts
    part_images = {}
    
    for mesh_file in mesh_files:
        part_name = os.path.splitext(mesh_file)[0]
        mesh_path = os.path.join(segmented_mesh_dir, mesh_file)
        
        try:
            # Load mesh with trimesh
            mesh_or_scene = trimesh.load(mesh_path)
            
            # Check whether we have a Trimesh or a Scene object
            if isinstance(mesh_or_scene, trimesh.Scene):
                # Extract the first geometry from the scene
                if mesh_or_scene.geometry:
                    mesh = list(mesh_or_scene.geometry.values())[0]
                else:
                    raise RuntimeError(f"No geometries found in scene for {part_name}.")
            elif isinstance(mesh_or_scene, trimesh.Trimesh):
                mesh = mesh_or_scene  # It's already a Trimesh object
            else:
                raise TypeError(f"Unsupported mesh type for {part_name}: {type(mesh_or_scene)}")
            
            # Create a figure with just the front view
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get mesh vertices and faces
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Plot the mesh manually
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                            triangles=faces, color='lightgray', alpha=0.8, 
                            edgecolor='gray', linewidth=0.2)
            
            # Set the view to front
            ax.view_init(elev=0, azim=0)
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Add part name as figure title
            ax.set_title(f"Part: {part_name}", fontsize=14)
            
            # Add mesh properties as text
            props_text = f"Volume: {getattr(mesh, 'volume', 0):.2f}\n"
            props_text += f"Surface Area: {getattr(mesh, 'area', 0):.2f}\n"
            props_text += f"Dimensions: {mesh.extents[0]:.2f} x {mesh.extents[1]:.2f} x {mesh.extents[2]:.2f}"
            props_text += f"Center of Mass: {mesh.center_mass[0]:.2f}, {mesh.center_mass[1]:.2f}, {mesh.center_mass[2]:.2f}\n"
            props_text += f"Is the part flat? {np.min(mesh.extents) / np.max(mesh.extents) < 0.3}\n"
            props_text += f"Bounding box volume ratio: {mesh.volume / (mesh.extents[0] * mesh.extents[1] * mesh.extents[2]):.2f}\n"
            props_text += f"Lowest point: {np.min(vertices[:, 2]):.2f}"
            plt.figtext(0.5, 0.01, props_text, ha='center', fontsize=12)
            
            # Save the figure
            temp_img_path = os.path.join(segmented_mesh_dir, f"{part_name}_render.png")
            plt.tight_layout()
            plt.savefig(temp_img_path, dpi=100)
            plt.close(fig)
            
            # Add to part_images
            part_images[part_name] = temp_img_path
            logging.info(f"Rendered part: {part_name}")
            
        except Exception as e:
            logging.warning(f"Could not render part {part_name}: {str(e)}")
    
    if not part_images:
        raise ValueError("Failed to render any parts for analysis")
        
    # Prepare images for API call
    image_messages = []

    # Add rendered image of whole object
    iter_seed_dir = cfg.out_dir
    iter_frontview_path = join_path(iter_seed_dir, "robot_frontview.png")
    whole_object_path = os.path.join(iter_frontview_path, "whole_object.png")  # Path to your whole object rendering
    if os.path.exists(whole_object_path):
        # Add whole object context
        image_messages.append({
            "type": "text",
            "text": "REFERENCE: This is the complete assembled object. Use this as context for identifying the base part:"
        })
        
        # Convert whole object image to base64
        with open(whole_object_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        # Add to messages
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}",
                "detail": "low"
            }
        })
        
        # Add separator
        image_messages.append({
            "type": "text",
            "text": "Now examine each individual part:"
        })

    for part_name, img_path in part_images.items():
        # Add part name
        image_messages.append({
            "type": "text",
            "text": f"Part: {part_name}"
        })
        
        # Convert image to base64
        with open(img_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            
        # Add to messages
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}",
                "detail": "low"
            }
        })
    
    # Create the prompt
    prompt = """
I have parts of an articulated object (like furniture, appliances, etc.). You MUST identify which part should be the BASE or main body of the object.

CRITICAL CONSTRAINT: Articulated parts like lids, doors, drawers, knobs, handles, or other moving components should NEVER be identified as the base part. These are parts that are meant to move relative to the main body.

THE BASE PART:
1. Is the MAIN BODY or STRUCTURAL FRAMEWORK of the object to which other parts attach
2. Is likely the LARGEST and HEAVIEST component by volume and mass
3. Is typically positioned at the BOTTOM of the object when in normal use
4. SUPPORTS the object's weight and provides stability
5. Often has a FLAT BOTTOM surface that would rest on the floor/table
6. Is the STATIC part that doesn't move when the object is being used
7. HOUSES or CONTAINS other components (like drawers go INTO the base)
8. Is NOT typically thin, flat components on top or sides (like lids, doors)

NEVER SELECT AS BASE:
- Lids (even if flat and large)
- Doors or panels that swing open
- Drawers that slide in/out
- Handles, knobs, or hardware
- Small accessories or attachments
- Any component that appears to be movable relative to the main structure

ANALYSIS STEPS:
1. First identify any articulated/movable parts and ELIMINATE them from consideration
2. Among remaining parts, identify the largest, most substantial component
3. Check if this component would logically be at the bottom of the object
4. Verify that this component would provide stability for the whole object
5. Confirm that other parts would attach to or be supported by this component

Even if uncertain, you MUST select exactly ONE part as the base.
This is critical for the object assembly process to work correctly.

IMPORTANT: Your response must be in JSON format with a single field "base_part" containing the name of the part you've selected as the base.
Example response: {"base_part": "part_name_here"}
    """
    
    # Prepare the API call
    messages = [
        {"role": "system", "content": "You are a specialized assistant that identifies the base part of articulated objects like furniture and appliances. You understand which parts should be static (base) versus movable (articulated components)."},
        {"role": "user", "content": image_messages + [{"type": "text", "text": prompt}]}
    ]
    
    # Make the API call - single attempt, no retries
    try:
        client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'), base_url="https://ai-gateway.mytkhgroup.com/")
        response = client.chat.completions.create(
            model="claude-3-5-sonnet-latest",
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        
        # Extract the response
        response_text = response.choices[0].message.content
        
        # Try to parse JSON from the response
        try:
            # Find JSON in the response
            import re
            json_match = re.search(r'({.*?})', response_text.replace('\n', ''))
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                
                base_part = result.get("base_part")
                if base_part and base_part in part_images:
                    logging.info(f"API identified base part: {base_part}")
                    
                    # Clean up temporary images
                    for part_name, img_path in part_images.items():
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    
                    return base_part
            
            # If we couldn't parse JSON or the base_part wasn't valid
            logging.warning(f"Could not extract valid base part from API response: {response_text}")
            raise ValueError(f"Could not extract valid base part from API response: {response_text}")
            
        except json.JSONDecodeError:
            logging.warning(f"Could not parse JSON from API response: {response_text}")
            raise ValueError(f"Could not parse JSON from API response: {response_text}")
            
    except Exception as e:
        logging.warning(f"API call failed: {str(e)}")
        raise ValueError(f"API call failed: {str(e)}")

# def render_frontview_with_open3d(mesh_dir, output_path):
#     """Render a frontview of the object using Open3D."""
#     import open3d as o3d
#     import numpy as np
#     import os
#     import logging
    
#     # Create a visualization object
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=800, height=600, visible=False)
    
#     # Load all mesh files and add them to the visualizer
#     mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(('.obj'))]#, '.glb', '.stl'))]
    
#     if not mesh_files:
#         raise ValueError(f"No mesh files found in {mesh_dir}")
    
#     # Load and combine all meshes
#     combined_mesh = o3d.geometry.TriangleMesh()
    
#     for mesh_file in mesh_files:
#         mesh_path = os.path.join(mesh_dir, mesh_file)
#         mesh = o3d.io.read_triangle_mesh(mesh_path)
#         # Ensure the mesh has vertex colors
#         if not mesh.has_vertex_colors():
#             mesh.paint_uniform_color([0.8, 0.8, 0.8])
#         combined_mesh += mesh
    
#     # Add the combined mesh to the visualizer
#     vis.add_geometry(combined_mesh)
    
#     # Get the bounding box of the combined mesh
#     bbox = combined_mesh.get_axis_aligned_bounding_box()
#     bbox_center = bbox.get_center()
    
#     # Set up the camera view
#     ctr = vis.get_view_control()
    
#     # Position the camera for a frontview
#     front_view_params = {
#         "field_of_view": 60.0,
#         "zoom": 0.7,
#         "front": [0, -1, 0],  # Looking from front
#         "lookat": bbox_center,
#         "up": [0, 0, 1]
#     }
    
#     # Apply the view parameters
#     ctr.set_front(front_view_params["front"])
#     ctr.set_lookat(front_view_params["lookat"])
#     ctr.set_up(front_view_params["up"])
#     ctr.set_zoom(front_view_params["zoom"])
    
#     # Update the renderer
#     vis.poll_events()
#     vis.update_renderer()
    
#     # Capture the image
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     vis.capture_screen_image(output_path, do_render=True)
    
#     # Close the visualizer
#     vis.destroy_window()
    
#     logging.info(f"Successfully rendered frontview image to {output_path}")


def process_partnet(obj_id: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    """Process partnet modality."""
    render_partnet_obj(obj_id, gpu_id, cfg, "stationary")
    preprocess_partnet_object(obj_id, gpu_id, cfg)
    cfg.dataset_dir = join_path(cfg.dataset_dir, obj_id)
    cfg.out_dir = join_path(cfg.out_dir, obj_id)
    return cfg


def process_text(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> DictConfig:
    """Process text modality."""
    cfg.link_actor.mode = "text"
    layout_planner = steps["Mesh Retrieval"]["Box Layout"]
    box_layout = layout_planner.load_prediction()

    mesh_searcher = steps["Mesh Retrieval"]["Mesh Retrieval"]
    mesh_info = mesh_searcher.load_prediction()
    meshes = {k: v["mesh_file"] for k, v in mesh_info.items()}

    out_link_actor_dir = join_path("link_placement", "iter_0", "seed_0")
    # IMPORTANT: move meshes to the link_placement directory
    new_box_layout = add_mesh_to_out_folder(out_link_actor_dir, box_layout,
                                            meshes,
                                            cfg.out_dir)
    link_summary_path = join_path(
        cfg.out_dir, out_link_actor_dir, "link_summary.txt")
    write_box_layout_to_link_summary(new_box_layout, link_summary_path)

    cfg.link_actor.link_summary_path = link_summary_path
    cfg.link_actor.new_box_layout = new_box_layout
    return cfg


def actor_function(iteration: int, seed: int, cfg: DictConfig, prompt: str, gpu_id: str, retry_kwargs: dict) -> Dict[str, Any]:
    """Execute the actor part of the pipeline."""
    link_placement_actor = LinkPlacementActor(
        create_task_config(cfg, join_path(
            "link_placement", f"iter_{iteration}", f"seed_{seed}"))
    )
    link_placement_actor.generate_prediction(
        **cfg.gen_config, **retry_kwargs)

    result = {
        "pred_image_path": link_placement_actor.load_predicted_rendering(),
        "link_pred_path": join_path(link_placement_actor.cfg.out_dir, link_placement_actor.OUT_RESULT_PATH),
    }

    link_placement_actor.render_prediction(gpu_id)
    if (cfg.modality != "text"):# and (cfg.modality != 'generated'):
        gt_link_diff = link_placement_actor.compute_gt_diff()
        logging.info(f"GT link diff is {gt_link_diff}")
        result["gt_link_diff"] = gt_link_diff

    return result




def is_actor_only(cfg):
    # cfg.actor_critic.actor_only is either a boolean or a string "auto"
    # if "auto" then we should critic AND actor only if cfg.modality == "image" or "video".
    # i.e., actor_only is when cfg.modality == "text"
    return cfg.actor_critic.actor_only if isinstance(cfg.actor_critic.actor_only, bool) else cfg.modality == "text"




def critic_function(iteration: int, seed: int, cfg: DictConfig, prompt: str, actor_result: Dict[str, Any]) -> Dict[str, Any]:
    if is_actor_only(cfg):# or cfg.modality == "generated":
        return {
            "feedback_score": 10,
        }

    """Execute the critic part of the pipeline."""
    if cfg.modality == "text":
        return {"feedback_score": 10, "feedback_path": None}

    link_critic = LinkCritic(create_task_config(cfg, join_path(
        "link_critic", f"iter_{iteration}", f"seed_{seed}")))
    link_critic.generate_prediction(
        gt_image_path=join_path(
            cfg.dataset_dir, f"robot_{cfg.cam_view}.png"),
        **actor_result,
        **cfg.gen_config,
    )
    feedback = link_critic.load_prediction()

    return {
        "feedback_score": int(feedback['realism_rating']),
        "feedback_path": join_path(link_critic.cfg.out_dir, link_critic.OUT_RESULT_PATH),
        "link_summary_path": cfg.link_actor.link_summary_path,  # will be automatically
        # populated by the preprocess function
    }


def post_process_iter(best_result: Dict[str, Any], cfg: DictConfig, steps) -> Dict[str, Any]:
    iteration = best_result["iteration"]
    seed = best_result["seed"]
    link_critic = LinkCritic(create_task_config(cfg, join_path(
        "link_critic", f"iter_{iteration}", f"seed_{seed}")))

    link_actor = LinkPlacementActor(create_task_config(cfg, join_path(
        "link_placement", f"iter_{iteration}", f"seed_{seed}")))

    steps["Link critic"].append(link_critic)
    steps["Link actor"].append(link_actor)
    return steps


def articulate_link(prompt: str, steps: Steps, gpu_id: str, cfg: DictConfig) -> Dict[str, Any]:
    """Main function to articulate links based on the given prompt and configuration."""
    # Pre-processing
    preprocess_result = preprocess(prompt, steps, gpu_id, cfg)
    cfg = preprocess_result["cfg"]

    # Actor-Critic loop or single actor run
    if cfg.modality == "text":
        retry_kwargs = {
            "link_summary_path": cfg.link_actor.link_summary_path,  # will be automatically
            # populated by the preprocess function
        }
        actor_result = actor_function(iteration=0, seed=0,
                                      cfg=cfg, prompt=prompt,
                                      gpu_id=gpu_id,
                                      retry_kwargs=retry_kwargs)
        best_result = {**actor_result, "iteration": 0,
                       "seed": 0, "feedback_score": 10}
        post_process_iter(best_result, cfg, steps)
    else:
        best_result = actor_critic_loop(
            cfg,
            lambda i, s, r: actor_function(i, s, cfg, prompt, gpu_id, r),
            lambda i, s, a: critic_function(i, s, cfg, prompt, a),
            steps=steps,
            error_handler=lambda e, i, s: error_handler(
                e, "link_error", i, s, cfg),
            post_process_iter=post_process_iter,
        )

    return steps
