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
    
    # Copy mesh files to the dataset directory first
    for mesh_file in mesh_files:
        src_path = join_path(segmented_mesh_dir, mesh_file)
        dst_path = join_path(obj_dataset_dir, mesh_file)
        shutil.copy(src_path, dst_path)
    
    # Use the first mesh as the root link
    root_mesh = mesh_files[0]
    root_link_name = os.path.splitext(root_mesh)[0]
    
    # Create a URDF file for the object
    urdf_content = f"""<?xml version="1.0" ?>
<robot name="{selected_obj_id}">
  <link name="{root_link_name}">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{root_mesh}" scale="1 1 1"/>
      </geometry>
      <material name="material_{root_link_name}">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{root_mesh}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
"""
    
    # Add additional links for each mesh file (except the root)
    for i, mesh_file in enumerate(mesh_files):
        if i == 0:  # Skip root, already added
            continue
            
        part_name = os.path.splitext(mesh_file)[0]
        urdf_content += f"""
  <link name="{part_name}">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{mesh_file}" scale="1 1 1"/>
      </geometry>
      <material name="material_{part_name}">
        <color rgba="0.8 0.8 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{mesh_file}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
  <joint name="{root_link_name}_to_{part_name}" type="fixed">
    <parent link="{root_link_name}"/>
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
    
    # Create a link summary file - using actual part names
    link_summary = f"object_id: {selected_obj_id}\n    Robot Link Summary:\n    - {root_link_name}\n"
    
    # Add other parts to the link summary
    for i, mesh_file in enumerate(mesh_files):
        if i > 0:  # Skip the root
            part_name = os.path.splitext(mesh_file)[0]
            link_summary += f"    - {part_name}\n"
    
    # Save the link summary
    link_summary_path = join_path(obj_dataset_dir, "link_summary.txt")
    with open(link_summary_path, "w") as f:
        f.write(link_summary)
    
    logging.info(f"Created link summary at {link_summary_path}")

    # Create semantics.txt file
    # This file maps link names to joint types and semantic categories
    # Format: link_name joint_type semantic_label
    semantics_content = ""
    for i, mesh_file in enumerate(mesh_files):
        part_name = os.path.splitext(mesh_file)[0]
        # Replace any spaces in part names with underscores
        part_name = part_name.replace(" ", "_")
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
    # frontview_path = "/home/link/DreMa/third_party/articulate-anything/datasets/output_views/drawer_multi-view/render_drawer_multi-view_6_0031.png"
    
    # Copy the frontview image to the output directory as well
    output_frontview_path = join_path(obj_output_dir, "robot_frontview.png")
    os.makedirs(os.path.dirname(output_frontview_path), exist_ok=True)
    shutil.copy(frontview_path, output_frontview_path)
    
    # Set up the configuration for link placement
    cfg.dataset_dir = obj_dataset_dir
    cfg.out_dir = obj_output_dir
    cfg.prompt = selected_obj_id
    
    return cfg

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
