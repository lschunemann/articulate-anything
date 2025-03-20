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

    segmented_mesh_dir = os.path.join(segmented_mesh_dir, "output")
    
    # Create absolute paths to avoid nesting issues
    base_dataset_dir = os.path.dirname(cfg.dataset_dir) if cfg.dataset_dir.endswith(selected_obj_id) else cfg.dataset_dir
    base_output_dir = os.path.dirname(cfg.out_dir) if cfg.out_dir.endswith(selected_obj_id) else cfg.out_dir
    
    # Set up the dataset and output directories
    obj_dataset_dir = join_path(base_dataset_dir, selected_obj_id)
    obj_output_dir = join_path(base_output_dir, selected_obj_id)
    
    logging.info(f"Creating directories: {obj_dataset_dir} and {obj_output_dir}")
    os.makedirs(obj_dataset_dir, exist_ok=True)
    os.makedirs(obj_output_dir, exist_ok=True)
    
    # Get mesh files
    mesh_files = [f for f in os.listdir(segmented_mesh_dir) if f.endswith(('.obj', '.glb', '.stl'))]
    
    if not mesh_files:
        raise ValueError(f"No mesh files found in {segmented_mesh_dir}")
    
    # Use VLM to identify which part should be the base
    base_part_name = identify_base_part(mesh_files, segmented_mesh_dir, gpu_id, cfg)
    
    # Clean up link names to avoid spaces and special characters
    link_names = []
    base_index = None
    
    for i, mesh_file in enumerate(mesh_files):
        original_name = os.path.splitext(mesh_file)[0]
        # Replace spaces and special characters with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', original_name)
        
        # Check if this is the base part
        if original_name == base_part_name:
            link_names.append("base")
            base_index = i
        else:
            link_names.append(clean_name)
    
    # If no base was found, use the first part as base
    if base_index is None:
        logging.warning("No base part identified by VLM, using first part as base")
        link_names[0] = "base"
        base_index = 0
    
    # Copy mesh files to the dataset directory with new names
    for i, mesh_file in enumerate(mesh_files):
        src_path = join_path(segmented_mesh_dir, mesh_file)
        new_filename = f"{link_names[i]}{os.path.splitext(mesh_file)[1]}"
        dst_path = join_path(obj_dataset_dir, new_filename)
        shutil.copy(src_path, dst_path)
    
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
    render_frontview_with_open3d(segmented_mesh_dir, frontview_path)
    
    # Copy the frontview image to the output directory as well
    output_frontview_path = join_path(obj_output_dir, "robot_frontview.png")
    os.makedirs(os.path.dirname(output_frontview_path), exist_ok=True)
    shutil.copy(frontview_path, output_frontview_path)
    
    # Set up the configuration for link placement
    cfg.dataset_dir = obj_dataset_dir
    cfg.out_dir = obj_output_dir
    cfg.prompt = selected_obj_id
    
    return cfg

def identify_base_part(mesh_files, segmented_mesh_dir, gpu_id, cfg):
    """Use OpenAI API to identify which part should be the base."""
    import trimesh
    import numpy as np
    from PIL import Image
    import os
    import logging
    import base64
    import io
    import openai
    import json
    
    logging.info("Identifying base part using OpenAI API...")
    
    # Render individual parts using trimesh (works in headless environments)
    part_images = {}
    for mesh_file in mesh_files:
        part_name = os.path.splitext(mesh_file)[0]
        mesh_path = os.path.join(segmented_mesh_dir, mesh_file)
        
        try:
            # Load mesh with trimesh
            mesh = trimesh.load(mesh_path)
            
            # Create a scene with the mesh
            scene = trimesh.Scene(mesh)
            
            # Get a camera view
            camera_angles = [(0, 0, 0), (np.pi/4, 0, 0), (0, np.pi/4, 0)]
            
            for i, angles in enumerate(camera_angles):
                # Create a camera transform matrix
                camera_transform = trimesh.transformations.rotation_matrix(
                    angles[0], [1, 0, 0], scene.centroid)
                camera_transform = trimesh.transformations.rotation_matrix(
                    angles[1], [0, 1, 0], scene.centroid, camera_transform)
                camera_transform = trimesh.transformations.rotation_matrix(
                    angles[2], [0, 0, 1], scene.centroid, camera_transform)
                
                # Move the camera away from the object
                camera_transform[0:3, 3] = camera_transform[0:3, 3] + np.array([0, 0, 2.0]) * mesh.scale
                
                # Render the mesh
                try:
                    rendered = scene.save_image(resolution=[400, 400], 
                                               transform=camera_transform,
                                               visible=True)
                    
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(rendered))
                    
                    # Save the image
                    temp_img_path = os.path.join(segmented_mesh_dir, f"{part_name}_view_{i}.png")
                    img.save(temp_img_path)
                    
                    # Add to part_images
                    if part_name not in part_images:
                        part_images[part_name] = []
                    part_images[part_name].append(temp_img_path)
                    
                    logging.info(f"Rendered part: {part_name}, view {i}")
                except Exception as e:
                    logging.warning(f"Could not render view {i} of part {part_name}: {str(e)}")
        
        except Exception as e:
            logging.warning(f"Could not load part {part_name}: {str(e)}")
    
    if not part_images:
        raise ValueError("Failed to render any parts for analysis")
    
    # Prepare images for API call
    image_messages = []
    for part_name, img_paths in part_images.items():
        # Add part name
        image_messages.append({
            "type": "text",
            "text": f"Part: {part_name}"
        })
        
        # Add images for this part
        for img_path in img_paths:
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
    I have parts of an object. You MUST identify which part is the base or main body of the object.
    The base is typically:
    1. The largest or most substantial part
    2. The part that supports other components
    3. The part that would naturally be at the bottom when the object is in use
    4. The part that other components would be attached to
    
    Look at each part image and tell me which one is DEFINITELY the base.
    You MUST choose exactly one part as the base, even if you're uncertain.
    This is critical for the object assembly process to work correctly.
    
    IMPORTANT: Your response must be in JSON format with a single field "base_part" containing the name of the part you've selected as the base.
    Example response: {"base_part": "part_name_here"}
    """
    
    # Prepare the API call
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes 3D object parts and identifies which part should be the base."},
        {"role": "user", "content": image_messages + [{"type": "text", "text": prompt}]}
    ]
    
    # Make the API call
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="claude-3-5-sonnet-latest",
                messages=messages,
                max_tokens=300,
                temperature=0.2,
                api_key=os.environ.get("API_KEY"),
                base_url="https://ai-gateway.mytkhgroup.com/"
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
                        for part_name, img_paths in part_images.items():
                            for img_path in img_paths:
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                        
                        return base_part
                
                # If we couldn't parse JSON or the base_part wasn't valid
                logging.warning(f"Could not extract valid base part from API response: {response_text}")
            except json.JSONDecodeError:
                logging.warning(f"Could not parse JSON from API response: {response_text}")
            
            # If we get here, try again with a more forceful prompt
            prompt = f"""
            CRITICAL: You MUST select exactly one part as the base from the following options: {', '.join(part_images.keys())}
            
            The base is the main supporting structure of the object. It's usually:
            - The largest part
            - The part at the bottom
            - The part that other components attach to
            
            This is a forced choice situation. You MUST select ONE part as the base.
            Your response MUST be in valid JSON format: {{"base_part": "part_name_here"}}
            """
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes 3D object parts and identifies which part should be the base."},
                {"role": "user", "content": image_messages + [{"type": "text", "text": prompt}]}
            ]
        except Exception as e:
            logging.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {str(e)}")
    
    # If we've exhausted all retries and still don't have a base part, raise an error
    raise ValueError(f"Failed to identify a base part after {max_retries} attempts. Cannot proceed without a definitive base selection.")

def render_frontview_with_open3d(mesh_dir, output_path):
    """Render a frontview of the object using Open3D."""
    import open3d as o3d
    import numpy as np
    import os
    import logging
    
    # Create a visualization object
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, visible=False)
    
    # Load all mesh files and add them to the visualizer
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(('.obj', '.glb', '.stl'))]
    
    if not mesh_files:
        raise ValueError(f"No mesh files found in {mesh_dir}")
    
    # Load and combine all meshes
    combined_mesh = o3d.geometry.TriangleMesh()
    
    for mesh_file in mesh_files:
        mesh_path = os.path.join(mesh_dir, mesh_file)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        # Ensure the mesh has vertex colors
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
        combined_mesh += mesh
    
    # Add the combined mesh to the visualizer
    vis.add_geometry(combined_mesh)
    
    # Get the bounding box of the combined mesh
    bbox = combined_mesh.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()
    
    # Set up the camera view
    ctr = vis.get_view_control()
    
    # Position the camera for a frontview
    front_view_params = {
        "field_of_view": 60.0,
        "zoom": 0.7,
        "front": [0, -1, 0],  # Looking from front
        "lookat": bbox_center,
        "up": [0, 0, 1]
    }
    
    # Apply the view parameters
    ctr.set_front(front_view_params["front"])
    ctr.set_lookat(front_view_params["lookat"])
    ctr.set_up(front_view_params["up"])
    ctr.set_zoom(front_view_params["zoom"])
    
    # Update the renderer
    vis.poll_events()
    vis.update_renderer()
    
    # Capture the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vis.capture_screen_image(output_path, do_render=True)
    
    # Close the visualizer
    vis.destroy_window()
    
    logging.info(f"Successfully rendered frontview image to {output_path}")


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
    if cfg.modality != "text":
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
    if is_actor_only(cfg):
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
