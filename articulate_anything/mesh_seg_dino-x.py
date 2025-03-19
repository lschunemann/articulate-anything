import os
import cv2
import json
import torch
import numpy as np
import argparse
import supervision as sv
from pathlib import Path
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import pycocotools.mask as mask_util

def load_models(device="cuda"):
    """Load SAM-2 model"""
    sam2_checkpoint = "/home/link/DreMa/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    return sam2_predictor

def get_dinox_boxes(image_path, text_prompt, client):
    """Use DINO-X API to get bounding boxes."""
    image_url = client.upload_file(image_path)
    task = DinoxTask(
        image_url=image_url,
        prompts=[TextPrompt(text=text_prompt)],
        bbox_threshold=0.4,
        targets=[DetectionTarget.BBox],
    )
    client.run_task(task)
    result = task.result

    # print(f"Dino-x result: {result}")
    
    input_boxes, confidences, labels = [], [], []
    for obj in result.objects:
        input_boxes.append(obj.bbox)
        confidences.append(obj.score)
        labels.append(obj.category.lower().strip())
    
    return np.array(input_boxes), np.array(confidences), labels

def segment_image(image_path, text_prompt, sam2_predictor, client, output_dir, device="cuda"):
    """Segment objects using DINO-X and SAM-2."""
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    input_boxes, confidences, labels = get_dinox_boxes(image_path, text_prompt, client)
    if len(input_boxes) == 0:
        print(f"No objects detected in {image_path}")
        return
    
    sam2_predictor.set_image(image)
    
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    vis_labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(labels, confidences)]
    
    # Visualization
    detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=np.arange(len(labels)))
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=vis_labels)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    output_base = os.path.basename(image_path).split('.')[0]
    vis_path = os.path.join(output_dir, f"{output_base}_vis.jpg")
    cv2.imwrite(vis_path, annotated_frame)

    # Save results as JSON
    mask_rles = [mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0] for mask in masks]
    for rle in mask_rles:
        rle["counts"] = rle["counts"].decode("utf-8")
    
    # Save results
    results = {
        "image_path": image_path,
        "annotations": [{
            "class_name": cls,
            "bbox": box.tolist(),
            "segmentation": mask_rle,#mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0],
            "score": score.item(),
        } for cls, box, mask_rle, score in zip(labels, input_boxes, mask_rles, scores)],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }


    
    with open(os.path.join(output_dir, f"{output_base}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

def process_multiple_views(input_dir, text_prompt, output_dir, api_token):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_predictor = load_models(device)
    client = Client(Config(api_token))
    
    for filename in os.listdir(input_dir):
        if filename.startswith('render'):
            input_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            segment_image(input_path, text_prompt, sam2_predictor, client, output_dir, device)
    
    print("Segmentation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('object', default='laptop_real')
    args = parser.parse_args()
    input_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.object}"
    output_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{args.object}"
    #text_prompt = "fridge main body. upper door. lower door."
    # text_prompt = "main body. upper slidable drawer. upmiddleper slidable drawer. lower slidable drawer."
    # text_prompt = "top surface.  leg.  frame.  handle.  drawer." # drawer
    text_prompt = "display window.  main body/housing.  control panel.  handle.  door."
    api_token = "1c9a49043bb8d909075526295bb20fc4"
    
    process_multiple_views(input_dir, text_prompt, output_dir, api_token)
