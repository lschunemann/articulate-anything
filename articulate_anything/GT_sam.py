import os
import json
import cv2
import numpy as np
import torch
import supervision as sv
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import pycocotools.mask as mask_util
import argparse

def load_models(device="cuda"):
    """Load SAM2 model."""
    sam2_checkpoint = "/home/link/DreMa/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    return sam2_predictor

def segment_image_with_points(image_path, points, point_labels, sam2_predictor, output_dir, device="cuda"):
    # Load image
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # Set image for SAM2
    sam2_predictor.set_image(image)
    
    # Get unique labels
    unique_labels = np.unique(point_labels)
    
    all_masks = []
    all_scores = []
    all_class_ids = []
    all_bboxes = []
    
    # Process each label separately
    for label_id in unique_labels:
        # Get points for this label
        label_points = [p for i, p in enumerate(points) if point_labels[i] == label_id]
        
        if not label_points:
            continue
            
        # Convert to numpy array
        label_points = np.array(label_points)
        
        # All points are foreground for SAM
        label_point_labels = np.ones(len(label_points))
        
        # Get prediction for this label's points
        masks, scores, logits = sam2_predictor.predict(
            point_coords=label_points,
            point_labels=label_point_labels,
            multimask_output=False,  # One mask per label
        )
        
        # Ensure masks are in correct format
        if masks.ndim == 3 and masks.shape[0] == 1:
            mask = masks[0]  # Take the first mask
        else:
            mask = masks
            
        # Convert to boolean
        mask_bool = mask.astype(bool)
        
        # Find bounding box
        nonzero = np.nonzero(mask_bool)
        if len(nonzero) == 2 and len(nonzero[0]) > 0:
            y_indices, x_indices = nonzero
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox = [x_min, y_min, x_max, y_max]
        else:
            # If mask is empty, use a small dummy box
            bbox = [0, 0, 10, 10]
            
        # Store results
        all_masks.append(mask_bool)
        all_scores.append(scores[0] if isinstance(scores, np.ndarray) and scores.size > 0 else 1.0)
        all_class_ids.append(label_id)
        all_bboxes.append(bbox)
    
    # Create output paths
    output_base = os.path.basename(image_path).split('.')[0]
    mask_path = os.path.join(output_dir, f"{output_base}_mask.png")
    vis_path = os.path.join(output_dir, f"{output_base}_vis.jpg")
    json_path = os.path.join(output_dir, f"{output_base}_results.json")
    
    # Stack masks for visualization
    if all_masks:
        stacked_masks = np.stack(all_masks)
        xyxy = np.array(all_bboxes)
        class_ids = np.array(all_class_ids)
        
        # Create detections object
        detections = sv.Detections(
            xyxy=xyxy,
            mask=stacked_masks,
            class_id=class_ids
        )
        
        # Create annotated image
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        # Create labels with class IDs
        labels = [f"Label {class_id}" for class_id in class_ids]
        label_annotator = sv.LabelAnnotator()
        
        annotated_frame = image.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        
        cv2.imwrite(vis_path, annotated_frame)
        
        # Save individual masks
        final_mask = np.zeros((h, w), dtype=np.uint8)
        for idx, mask in enumerate(all_masks):
            final_mask[mask] = all_class_ids[idx] + 1  # Use class ID + 1 as the mask value
        
        # cv2.imwrite(mask_path, final_mask)
        
        # Save results as JSON
        mask_rles = []
        for mask in all_masks:
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            mask_rles.append(mask_rle)
        
        results = {
            "image_path": image_path,
            "annotations": [
                {
                    "segmentation": mask_rle,
                    "score": float(score),
                    "label": int(label_id)
                }
                for mask_rle, score, label_id in zip(mask_rles, all_scores, all_class_ids)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(f"No valid masks generated for {image_path}")
    
    return final_mask if all_masks else None, annotated_frame if all_masks else image, results if all_masks else {}

def process_images_with_ground_truth(input_dir, annotations_dir, output_dir):
    """Process images using ground truth points."""
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_predictor = load_models(device)

    # Process each image
    for filename in os.listdir(input_dir):
        if filename.startswith('render') and filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            annotation_path = os.path.join(annotations_dir, f"{os.path.splitext(filename)[0]}_annotations.json")

            # Load ground truth points
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    annotation_data = json.load(f)
                    points = annotation_data['points']
                    point_labels = annotation_data['point_labels']
                
                print(f"Processing {filename} with ground truth points...")
                segment_image_with_points(
                    image_path=input_path,
                    points=points,
                    point_labels=point_labels,
                    sam2_predictor=sam2_predictor,
                    output_dir=output_dir,
                    device=device
                )
            else:
                print(f"No annotation found for {filename}")

    print("Segmentation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('object', default='fridge_rodin')
    args = parser.parse_args()
    input_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.object}"
    annotations_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/output_views/{args.object}/gt_annotated_parts"
    output_dir = f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/{args.object}/gt_segmentations"

    process_images_with_ground_truth(input_dir, annotations_dir, output_dir)
