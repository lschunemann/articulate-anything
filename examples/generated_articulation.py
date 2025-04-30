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
    parser.add_argument('task', default='laptop_real')
    args = parser.parse_args()
    
    
    load_dotenv()
    API_KEY = os.environ.get('API_KEY')

    # task = "drawer_RL_Bench"
    # task = "laptop_RLBench"
    task = args.task


    # video_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/in-the-wild-dataset/videos/{task}.mp4"
    video_path = f"/home/link/DreMa/third_party/articulate-anything/datasets/RLBench/videos/{task}.mp4"

    modality = "generated"
    prompt = video_path

    cfg = load_config()
    cfg.prompt = prompt
    cfg.modality = modality
    cfg.out_dir = join_path("/home/link/DreMa/third_party/articulate-anything/", "results", modality, task)

    cfg.dataset_dir = "/home/link/DreMa/third_party/articulate-anything/datasets/partnet-mobility-v0/dataset"

    glb = '_'.join([task.split('_')[0], 'multi-view'])
    # cfg.segmented_mesh_dir = "/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/drawer_multi-view"
    cfg.segmented_mesh_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{glb}"
    cfg.video_path = video_path

    use_cotracker = False #True # {True, False}
    mode = "video"
    # mode = "image"
    actor_prompting_type = "basic" # {basic, incontext}
    critic_prompting_type = "incontext" # {basic, incontext}

    cfg.joint_actor.mode = mode
    cfg.joint_actor.use_cotracker = use_cotracker
    cfg.joint_actor.type = actor_prompting_type
    cfg.joint_actor.targetted_affordance = False
    # cfg.joint_actor.targetted_affordance = False

    cfg.joint_actor.examples_dir = "/home/link/DreMa/third_party/articulate-anything/datasets/multi_modal_incontext_examples/joint_actor/in_context_actor_examples_datasets" ## Put your examples here


    cfg.joint_critic.mode = mode
    cfg.joint_critic.use_cotracker = use_cotracker
    cfg.joint_critic.type = critic_prompting_type

    cfg.joint_critic.examples_dir = "/home/link/DreMa/third_party/articulate-anything/datasets/multi_modal_incontext_examples/joint_critic/in_context_examples_datasets" ## Put your examples here

    cfg.actor_critic.actor_only = False

    ## important to set correctly for the joint_critic to works properly
    ## this should have the same direction as the ground-truth video
    cfg.simulator.flip_video = False ## flip time for suitcase
    cfg.simulator.ray_tracing=False
    cfg.simulator.floor_texture = "plain"


    # cfg.category_selector.topk = 3
    cfg.category_selector.topk = 1 ## how many top categories should we search for an object template match
    # this is because PartNet-Mobility dataset categories labels are sparse and sometimes not great


    cfg.obj_selector.frame_index = 0

    cfg.actor_critic.max_iter = 10
    cfg.actor_critic.num_seeds = 3 # standard=1

    cfg.model_name = "gpt-4o"#"claude-3-5-sonnet-latest"#gemini-1.5-flash-latest"

    cfg.actor_critic.cutoff = 9 ## what score is needed to stop actor-critic loop


    cfg.api_key = API_KEY ## loaded from .env file

    cfg.gen_config.overwrite = True
    cfg.task = task

    global_config.set_config(cfg)

    steps = articulate(cfg)