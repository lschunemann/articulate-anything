from articulate import articulate
from omegaconf import OmegaConf
from dotenv import load_dotenv
from articulate_anything.utils.viz import (
    show_video, 
    display_code, 
    show_videos, 
    display_codes,
    show_images,
)
from articulate_anything.utils.utils import load_config, join_path
from articulate_anything.utils import global_config
from articulate_anything.utils.cotracker_utils import make_cotracker
from PIL import Image
import json
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Name of the folder (e.g., partnet_12345_single-view)')
    args = parser.parse_args()
    
    load_dotenv()
    API_KEY = "03e734219b6c4ba8b2724e0d788c8a8e"

    # API_KEY = os.environ.get('API_KEY')

    task = args.task

    # Extract task number from task name (e.g., "partnet_1034_single-view" -> "1034")
    task_number = task.split('_')[1]
    
    # Construct the segmented_mesh_dir path like in the original
    cfg = load_config()
    # cfg.segmented_mesh_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{task}"
    
    # # Video directory
    # video_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/partnet-mobility-v0/dataset/{task_number}/"
    
    # # Find the video file that doesn't start with 'aug_'
    # if not os.path.exists(video_dir):
    #     raise ValueError(f"Video directory {video_dir} does not exist")
        
    # video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4') and not f.startswith('aug_')]
    
    # if not video_files:
    #     raise ValueError(f"No valid video files found in {video_dir}")
    
    # video_file = video_files[0]  # Take the first one
    # video_path = os.path.join(video_dir, video_file)
    
    # print(f"Using video file: {video_path}")

    if task.split('_')[0] == "partnet":
        image_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/partnet-mobility-v0/dataset/{task_number}/robot_frontview.png"
    elif task.split('_')[0] == "artvip":
        image_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/partnet-mobility-v0/dataset/{task}/robot_frontview.png"
    cfg.image_path = image_dir

    modality = "image"
    prompt = image_dir# video_path

    cfg.prompt = prompt
    cfg.modality = modality
    cfg.out_dir = join_path("/home/link/DreMa/third_party/articulate-anything/", "results", modality, task)

    cfg.dataset_dir = "/home/link/DreMa/third_party/articulate-anything/datasets/partnet-mobility-v0/dataset"

    # cfg.video_path = video_path

    use_cotracker = True  # Enable cotracker for video mode
    mode = "image"#"video"  # Changed from image to video
    actor_prompting_type = "incontext"
    critic_prompting_type = "incontext"

    cfg.joint_actor.mode = mode
    cfg.joint_actor.use_cotracker = use_cotracker
    cfg.joint_actor.type = actor_prompting_type
    cfg.joint_actor.targetted_affordance = False

    cfg.joint_actor.examples_dir = "/home/link/DreMa/third_party/articulate-anything/datasets/multi_modal_incontext_examples/joint_actor/in_context_actor_examples_datasets"

    cfg.joint_critic.mode = mode
    cfg.joint_critic.use_cotracker = use_cotracker
    cfg.joint_critic.type = critic_prompting_type

    cfg.joint_critic.examples_dir = "/home/link/DreMa/third_party/articulate-anything/datasets/multi_modal_incontext_examples/joint_critic/in_context_examples_datasets"

    cfg.actor_critic.actor_only = False

    cfg.simulator.flip_video = False
    cfg.simulator.ray_tracing = True
    cfg.simulator.floor_texture = "plain"

    cfg.simulator.urdf.raise_distance_offset = 0.3

    cfg.category_selector.topk = 1
    cfg.obj_selector.frame_index = 0
    cfg.actor_critic.max_iter = 3
    cfg.actor_critic.num_seeds = 2

    cfg.in_context.num_examples = 'all'
    cfg.video_encoding.num_frames = 5

    cfg.model_name = "gemini-2.5-flash-latest"#preview-04-17"
    cfg.actor_critic.cutoff = 7
    cfg.api_key = API_KEY

    cfg.gen_config.overwrite = False
    cfg.task = task

    global_config.set_config(cfg)

    steps = articulate(cfg)
    