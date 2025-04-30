import os
import cv2
import json
import torch
import numpy as np
import argparse
import supervision as sv
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
# from detect_articulated_parts import detect_articulated_parts

GROUNDED_SAM2_PATH = "/home/link/DreMa/third_party/Grounded-SAM-2"  # Update this path
sys.path.append(GROUNDED_SAM2_PATH)
sys.path.append(os.path.join(GROUNDED_SAM2_PATH, "grounding_dino"))

# sys.path.append("/home/link/DreMa/third_party/articulate-anything")
# sys.path.append("/home/link/DreMa/third_party/articulate-anything/articulate_anything")

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import pycocotools.mask as mask_util

def load_models(device="cuda"):
    """Load Grounded-SAM-2 and GroundingDINO models"""
    # Build SAM2
    sam2_checkpoint = "/home/link/DreMa/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    # Build GroundingDINO
    grounding_model = load_model(
        model_config_path="/home/link/DreMa/third_party/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="/home/link/DreMa/third_party/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth",
        device=device
    )
    
    return sam2_predictor, grounding_model

def segment_image(image_path, text_prompt, sam2_predictor, grounding_model, output_dir, device="cuda"):
    # Load image
    image_source, image = load_image(image_path)
    
    # Set image for SAM2
    sam2_predictor.set_image(image_source)
    
    # Get predictions from GroundingDINO
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=0.5,
        text_threshold=0.45,
    )
    
    # If no boxes were detected, return early
    if len(boxes) == 0:
        print(f"No objects detected in {image_path}")
        return None, None, None
    
    # Process boxes for SAM2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # Process boxes one at a time
    all_masks = []
    all_scores = []
    all_logits = []
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        for box in input_boxes:
            box_expanded = box[None, :]  # Add batch dimension
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_expanded,
                multimask_output=False,
            )
            all_masks.append(masks[0])  # Remove batch dimension
            all_scores.append(scores[0])
            all_logits.append(logits[0])
    
    # Stack results
    masks = np.stack(all_masks, axis=0)
    scores = np.stack(all_scores, axis=0)
    logits = np.stack(all_logits, axis=0)
    
    # Process masks
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # Create visualization
    confidences = confidences.numpy().tolist()
    class_ids = np.array(list(range(len(labels))))
    
    vis_labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(labels, confidences)
    ]
    
    # Create output paths
    output_base = os.path.basename(image_path).split('.')[0]
    mask_path = os.path.join(output_dir, f"{output_base}_mask.png")
    vis_path = os.path.join(output_dir, f"{output_base}_vis.jpg")
    json_path = os.path.join(output_dir, f"{output_base}_results.json")
    
    # Save visualization
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    # Create annotated image
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_frame = img.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=vis_labels)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    cv2.imwrite(vis_path, annotated_frame)

    # Save individual masks
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for idx, mask in enumerate(masks, start=1):
        mask_bool = mask.astype(bool)
        final_mask[mask_bool] = idx * 50

    
    # Save results as JSON
    mask_rles = [mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0] for mask in masks]
    for rle in mask_rles:
        rle["counts"] = rle["counts"].decode("utf-8")
    
    results = {
        "image_path": image_path,
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box.tolist(),
                "segmentation": mask_rle,
                "score": score.item(),
            }
            for class_name, box, mask_rle, score in zip(labels, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    return final_mask, annotated_frame, results

def process_multiple_views(input_dir, text_prompt, output_dir):
    """Process all views in the input directory"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_predictor, grounding_model = load_models(device)
    
    # Process each view
    for filename in os.listdir(input_dir):
        # if filename.endswith('.png'):
        if filename.startswith('render'):
            input_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            
            segment_image(
                image_path=input_path,
                text_prompt=text_prompt,
                sam2_predictor=sam2_predictor,
                grounding_model=grounding_model,
                output_dir=output_dir,
                device=device
            )
    
    print("Segmentation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('object', default='laptop_real')
    args = parser.parse_args()
    input_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.object}"#drawer_rodin"
    output_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{args.object}"#drawer_rodin"
    # video_path = "/home/link/DreMa/third_party/articulate-anything/datasets/in-the-wild-dataset/videos/drawer_RL_Bench.mp4"
    # Make sure each part ends with a dot
    # text_prompt = "only the movable screen part of the laptop, excluding the base." ## this works really well!!
    #text_prompt = "screen. laptop base."
    #text_prompt = "frame. left sliding door. right sliding door."
    #text_prompt = "body. top drawer. center drawer. lower drawer."
    #text_prompt = "body. top drawer. center drawer. lower drawer."
    # text_prompt = "glass jar. lid."
    # text_prompt = "cabinet. sliding door." # cabinet
    text_prompt = "hinge joint.  mounting plate.  handle." # washing machine
    # text_prompt = "Top drawer. Middle drawer. Bottom drawer. Drawer base."
    # text_prompt = detect_articulated_parts()
    
    process_multiple_views(input_dir, text_prompt, output_dir)
