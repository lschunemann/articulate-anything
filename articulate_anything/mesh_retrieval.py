from omegaconf import DictConfig
import logging
from termcolor import colored
from articulate_anything.agent.actor.mesh_retrieval.text_task_specifier import TextTaskSpecifier
from articulate_anything.agent.actor.mesh_retrieval.text_layout_planner import TextLayoutPlanner
from articulate_anything.agent.actor.mesh_retrieval.partnet_mesh_retrieval import PartnetMeshRetrieval
from articulate_anything.agent.actor.mesh_retrieval.category_selector import (
    CategorySelector,
    DummyCategorySelector
)
from articulate_anything.agent.actor.mesh_retrieval.obj_selector import (
    make_obj_selector,
    get_candidate_objs,
    get_obj_image,
    save_obj_selector_viz,
    get_candidate_objs_from_categories,
    GeneratedObjectSelector
)
from articulate_anything.utils.utils import (
    create_task_config,
    Steps,
    join_path
)
        
def mesh_retrieval(cfg: DictConfig):
    visual_modalities = ["image", "video"]
    if cfg.modality == "text":
        return mesh_retrieval_text(cfg)
    elif cfg.modality in visual_modalities:
        return mesh_retrieval_visual(cfg)
    elif cfg.modality == "partnet":
        return
    elif cfg.modality == "generated":
        return mesh_retrieval_generated(cfg)
    else:
        logging.error(
            f"Modality {colored(cfg.modality, 'red')} not supported. "
            f"Available modalities are: text, {', '.join(visual_modalities)}, partnet, generated")




def mesh_retrieval_generated(cfg: DictConfig) -> Steps:
    """Load pre-generated and segmented meshes based on visual input."""
    steps = Steps()
    
    # 1. Similar to visual mesh retrieval, we'll use a category selector
    # This step can be simplified since we already know what object we're working with
    category_selector = DummyCategorySelector(create_task_config(cfg, "category_selector"))
    category_selector.generate_prediction(cfg.prompt,
                                         additional_prompt=cfg.additional_prompt,
                                         **cfg.gen_config,
                                         **cfg.video_encoding)
    
    # We can either use the actual categories or just set a placeholder
    obj_categories = ["generated_object"]  # Simplified since we're using our own meshes
    steps.add_step("Category Selection", category_selector)
    
    # 2. Create a custom object selector that points to our generated meshes
    obj_selector = GeneratedObjectSelector(create_task_config(cfg, "obj_selector"))
    
    # Get the frame from the video to use as reference
    gt_image = get_obj_image(cfg.prompt, frame_index=cfg.obj_selector.frame_index)
    
    # Instead of selecting from candidates, we're directly using our generated object
    obj_selector.generate_prediction(gt_image, 
                                    segmented_mesh_dir=cfg.segmented_mesh_dir,
                                    **cfg.gen_config)
    
    steps.add_step("Object Selection", obj_selector)
    return steps


# def segment_mesh():
#     import subprocess

#     subprocess.run(['conda', 'run', '-n', 'sam', 'python', 'mesh_segmentation.py'])


# def mesh_generation(cfg: DictConfig):
#     # TODO: add API mesh generation
#     steps = Steps()
#     # mesh_generator = generate_mesh()
#     # mesh_generator.generate_prediction()
#     # steps.add_step("Mesh Generation", mesh_generator)

#     mesh_load = mesh_loader()
#     steps.add_step("Mesh Generation", mesh_load)
#     segmented_mesh = segment_mesh(video_path)
#     steps.add_step("Mesh Segmentation", segmented_mesh)
#     return steps



def mesh_retrieval_text(cfg: DictConfig) -> Steps:
    steps = Steps()

    # 1. Task specifier: expands the text prompt to specify the parts
    task_specifier = TextTaskSpecifier(
        create_task_config(cfg, "task_specifier"))
    task_specifier.generate_prediction(cfg.prompt, **cfg.gen_config)
    steps.add_step("Task Specification", task_specifier)

    # 2. Layout planner: specifies the box dim of the object
    layout_planner = TextLayoutPlanner(create_task_config(cfg, "box_layout"))
    layout_planner.generate_prediction(task_specifier.load_prediction()[
                                       'output'], **cfg.gen_config)
    steps.add_step("Box Layout", layout_planner)

    # 3. Part Mesh retrieval: retrieve the mesh of each part in the box layout
    mesh_searcher = PartnetMeshRetrieval(
        create_task_config(cfg, "mesh_retrieval"))
    mesh_searcher.generate_prediction(
        layout_planner.load_prediction(),
        **cfg.gen_config,
        **cfg.partnet_mesh_retrieval,
    )
    steps.add_step("Mesh Retrieval", mesh_searcher)

    return steps

def mesh_retrieval_visual(cfg: DictConfig):
    steps = Steps()
    # 1. Match to an object category first
    # using CLIP
    category_selector = CategorySelector(
        create_task_config(cfg, "category_selector"))

    category_selector.generate_prediction(cfg.prompt,
                                          additional_prompt=cfg.additional_prompt,
                                          **cfg.gen_config,
                                          **cfg.video_encoding)

    obj_categories = category_selector.load_prediction()[
        "most_similar_objects"]
    steps.add_step("Category Selection", category_selector)

    # 2. Match to an object instance
    # by using VLM to compare the gt_image and candidate_imgs
    obj_selector = make_obj_selector(cfg)
    candidate_images, candidate_obj_ids = get_candidate_objs_from_categories(obj_categories,
                                                                             cam_view=cfg.cam_view,)
    print("candidate_obj_ids", candidate_obj_ids)
    print("CFG.prompt", cfg.prompt)
    gt_image = get_obj_image(cfg.prompt, frame_index=cfg.obj_selector.frame_index)
    obj_selector.generate_prediction(gt_image, candidate_images, candidate_obj_ids,
                                     **cfg.gen_config,
                                     **cfg.obj_selector)
    selected_obj = obj_selector.load_prediction()
    save_obj_selector_viz(selected_obj,
                          candidate_images,
                          gt_image,
                          candidate_obj_ids,
                          obj_selector.cfg.out_dir)
    steps.add_step("Object Selection", obj_selector)
    return steps