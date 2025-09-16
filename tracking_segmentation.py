#!/usr/bin/env python3
"""
Tracking-based Segmentation Pipeline

This script implements a refined pipeline for object part segmentation using:
1. Point tracking (CoTracker)
2. Viewpoint estimation and GLB rendering
3. FoundationPose for object pose estimation
4. SAM2 segmentation with tracking points as prompts

Pipeline stages:
1. Extract frames from video
2. Track points using CoTracker
3. Estimate viewpoint and render GLB from same perspective
4. Use FoundationPose to get object pose in first frame
5. Render GLB in estimated pose with virtual camera
6. SAM2 segmentation using first frame tracking points as prompts

Usage:
    python tracking_segmentation.py --video video.mp4 --glb object.glb --output output_dir
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image
import trimesh
import pyrender
import torch

# Import CoTracker utilities
try:
    from articulate_anything.utils.cotracker_utils import make_cotracker
    from omegaconf import OmegaConf
    COTRACKER_AVAILABLE = True
except ImportError:
    COTRACKER_AVAILABLE = False
    print("Warning: CoTracker not available.")

# Import existing utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from articulate_anything.utils.viz import get_frames_from_video
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("Warning: articulate_anything.utils.viz not available.")

# Import FoundationPose
try:
    # FoundationPose is in third_party/FoundationPose
    foundation_pose_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FoundationPose'))
    sys.path.append(foundation_pose_path)

    from estimater import *
    from datareader import *
    import torch
    FOUNDATION_POSE_AVAILABLE = True
    print("FoundationPose loaded successfully.")
except ImportError as e:
    FOUNDATION_POSE_AVAILABLE = False
    print(f"Warning: FoundationPose not available: {e}")

# Import Grounded-SAM-2
try:
    # Grounded-SAM-2 is in third_party/Grounded-SAM-2
    grounded_sam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Grounded-SAM-2'))
    sys.path.append(grounded_sam2_path)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import supervision as sv
    SAM2_AVAILABLE = True
    print("Grounded-SAM-2 loaded successfully.")
except ImportError as e:
    SAM2_AVAILABLE = False
    print(f"Warning: Grounded-SAM-2 not available: {e}")


class TrackingSegmentationPipeline:
    """Refined pipeline for tracking-based segmentation with FoundationPose and SAM2."""

    def __init__(self, video_path: str, glb_path: str, output_dir: str,
                 camera_intrinsics: Optional[np.ndarray] = None,
                 camera_extrinsics: Optional[np.ndarray] = None,
                 use_cotracker: bool = True, cotracker_config: Optional[Dict] = None):

        self.video_path = "/home/link/DreMa/third_party/articulate-anything/datasets/RLBench/videos/" + video_path
        self.glb_path = "/home/link/DreMa/third_party/articulate-anything/datasets/output_views/" + glb_path + f"/{glb_path}.glb"
        self.output_dir = "/home/link/DreMa/third_party/articulate-anything/video_seg_results/" + output_dir

        # Camera parameters (if not provided, will be estimated)
        self.camera_intrinsics = camera_intrinsics
        self.camera_extrinsics = camera_extrinsics

        # Setup output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Initialize tracking configuration
        self.use_cotracker = use_cotracker and COTRACKER_AVAILABLE
        if self.use_cotracker:
            default_cotracker_config = {
                'mode': 'offline',
                'checkpoint_path': '../co-tracker/checkpoints/cotracker2.pth',
                'grid_size': 100,
                'render_all': False,
                'displacement_threshold': 0.1,
                'max_moving_points': 1000,
                'linewidth': 2,
                'overwrite': True
            }
            if cotracker_config:
                default_cotracker_config.update(cotracker_config)

            self.cotracker_cfg = OmegaConf.create({'cotracker': default_cotracker_config})

            try:
                self.cotracker = make_cotracker(self.cotracker_cfg)
                self.logger.info("CoTracker initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CoTracker: {e}")
                self.use_cotracker = False

        # Initialize components
        self.frames: List[Image.Image] = []
        self.point_tracks: List[Dict] = []
        self.first_frame_points: Dict[str, List[Tuple[float, float]]] = {}
        self.glb_mesh = None
        self.object_pose = None
        self.rendered_frame = None
        self.segmentation_results = None

        # Initialize FoundationPose wrapper
        if FOUNDATION_POSE_AVAILABLE:
            # Load GLB mesh first for FoundationPose initialization
            if os.path.exists(self.glb_path):
                mesh = trimesh.load(self.glb_path)
                if hasattr(mesh, 'geometry'):
                    mesh = list(mesh.geometry.values())[0]

                debug_dir = os.path.join(self.output_dir, "foundation_pose_debug")
                os.makedirs(debug_dir, exist_ok=True)

                self.foundation_pose_wrapper = self._create_foundation_pose_wrapper(mesh, debug_dir)
                self.logger.info("FoundationPose wrapper initialized")
            else:
                self.logger.warning(f"GLB file not found for FoundationPose: {self.glb_path}")
                self.foundation_pose_wrapper = None
        else:
            self.foundation_pose_wrapper = None

        # Initialize SAM2 predictor
        if SAM2_AVAILABLE:
            # SAM2 configuration for Grounded-SAM-2
            grounded_sam2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Grounded-SAM-2"))
            sam2_checkpoint = os.path.join(grounded_sam2_root, "checkpoints", "sam2.1_hiera_large.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                # Change to Grounded-SAM-2 directory temporarily for config resolution
                original_cwd = os.getcwd()
                os.chdir(grounded_sam2_root)

                sam2_model = build_sam2(model_cfg, sam2_checkpoint).to(device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)

                # Restore original working directory
                os.chdir(original_cwd)

                self.logger.info("Grounded-SAM-2 predictor initialized")
            except Exception as e:
                # Restore original working directory in case of error
                if 'original_cwd' in locals():
                    os.chdir(original_cwd)
                self.logger.error(f"Failed to initialize Grounded-SAM-2: {e}")
                raise RuntimeError(f"SAM2 initialization failed: {e}")
        else:
            raise RuntimeError("Grounded-SAM-2 not available but required for pipeline")

    def step1_extract_frames(self, num_frames: int = 8) -> List[Image.Image]:
        """Step 1: Extract frames from video."""
        self.logger.info(f"Step 1: Extracting {num_frames} frames from video: {self.video_path}")

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        if VIZ_AVAILABLE:
            self.frames = get_frames_from_video(
                self.video_path,
                num_frames=num_frames,
                video_encoding_strategy="individual",
                to_crop_white=False,
                flip_horizontal=False
            )
        else:
            self.frames = self._extract_frames_fallback(num_frames)

        # Save extracted frames
        frames_dir = os.path.join(self.output_dir, "extracted_frames")
        os.makedirs(frames_dir, exist_ok=True)

        for i, frame in enumerate(self.frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            frame.save(frame_path)

        self.logger.info(f"Extracted and saved {len(self.frames)} frames")
        return self.frames

    def step2_track_points(self) -> List[Dict]:
        """Step 2: Track points using CoTracker."""
        self.logger.info("Step 2: Tracking points using CoTracker")

        if not self.frames:
            raise ValueError("No frames available. Run step1_extract_frames first.")

        if self.use_cotracker:
            self.point_tracks = self._track_points_cotracker()
        else:
            self.logger.error("CoTracker not available and no fallback implemented")
            return []

        # Extract first frame points for each cluster
        self._extract_first_frame_points()

        # Save tracking results
        self._save_tracking_results()

        self.logger.info(f"Completed point tracking with {len(self.point_tracks)} tracks")
        return self.point_tracks

    def step3_estimate_viewpoint_and_render(self) -> np.ndarray:
        """Step 3: Estimate viewpoint and render GLB from same perspective."""
        self.logger.info("Step 3: Estimating viewpoint and rendering GLB")

        if not os.path.exists(self.glb_path):
            raise FileNotFoundError(f"GLB file not found: {self.glb_path}")

        # Load GLB mesh
        self.glb_mesh = trimesh.load(self.glb_path)
        if hasattr(self.glb_mesh, 'geometry'):
            self.glb_mesh = list(self.glb_mesh.geometry.values())[0]

        # Estimate camera parameters if not provided
        if self.camera_intrinsics is None or self.camera_extrinsics is None:
            self._estimate_camera_parameters()

        # Render GLB from estimated viewpoint
        rendered_frame = self._render_glb_from_viewpoint()

        # Save rendered frame
        render_path = os.path.join(self.output_dir, "reference_render.png")
        cv2.imwrite(render_path, rendered_frame)

        self.logger.info("Completed viewpoint estimation and GLB rendering")
        return rendered_frame

    def step4_foundation_pose(self) -> np.ndarray:
        """Step 4: Use FoundationPose to get object pose in first frame."""
        self.logger.info("Step 4: Estimating object pose using FoundationPose")

        if not FOUNDATION_POSE_AVAILABLE:
            raise RuntimeError("FoundationPose not available. Cannot proceed without pose estimation.")

        self.object_pose = self._estimate_pose_foundation_pose()

        # Save pose
        pose_path = os.path.join(self.output_dir, "object_pose.json")
        with open(pose_path, 'w') as f:
            json.dump({
                'pose_matrix': self.object_pose.tolist(),
                'method': 'FoundationPose'
            }, f, indent=2)

        self.logger.info("Completed object pose estimation")
        return self.object_pose

    def step5_render_glb_in_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Step 5: Render GLB in estimated pose with virtual camera."""
        self.logger.info("Step 5: Rendering GLB in estimated pose")

        if self.object_pose is None:
            raise ValueError("Object pose not available. Run step4_foundation_pose first.")

        # Render GLB in estimated pose
        self.rendered_frame = self._render_glb_with_pose(self.object_pose)

        # Render with white background for point filtering
        white_bg_render = self._render_glb_with_pose(self.object_pose, white_background=True)

        # Save rendered frames
        render_path = os.path.join(self.output_dir, "posed_render.png")
        white_bg_path = os.path.join(self.output_dir, "posed_render_white_bg.png")
        cv2.imwrite(render_path, self.rendered_frame)
        cv2.imwrite(white_bg_path, white_bg_render)

        self.logger.info("Completed GLB rendering in pose")
        return self.rendered_frame, white_bg_render

    def step6_sam2_segmentation(self) -> Dict[str, Any]:
        """Step 6: SAM2 segmentation using first frame tracking points as prompts."""
        self.logger.info("Step 6: SAM2 segmentation with tracking point prompts")

        if not self.first_frame_points:
            raise ValueError("No first frame points available. Run step2_track_points first.")

        if self.rendered_frame is None:
            raise ValueError("No rendered frame available. Run step5_render_glb_in_pose first.")

        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 not available. Cannot proceed without segmentation.")

        # Filter object vs background points using white background render
        object_points = self._filter_object_points()

        self.segmentation_results = self._run_sam2_segmentation(object_points)

        # Save segmentation results
        seg_path = os.path.join(self.output_dir, "segmentation_results.json")
        with open(seg_path, 'w') as f:
            json.dump(self._serialize_for_json(self.segmentation_results), f, indent=2)

        # Visualize results
        self._visualize_segmentation_results()

        self.logger.info("Completed SAM2 segmentation")
        return self.segmentation_results

    def _track_points_cotracker(self) -> List[Dict]:
        """Track points using CoTracker."""
        self.logger.info("Tracking points using CoTracker...")

        # Save frames as temporary video
        temp_video_path = os.path.join(self.output_dir, "temp_video.mp4")
        self._frames_to_video(temp_video_path)

        try:
            # Run CoTracker
            pred_tracks, pred_visibility, moving_mask, total_displacement = self.cotracker.forward(
                video_path=temp_video_path,
                grid_size=self.cotracker_cfg.cotracker.grid_size,
                render_all=self.cotracker_cfg.cotracker.render_all,
                displacement_threshold=self.cotracker_cfg.cotracker.displacement_threshold,
                max_moving_points=self.cotracker_cfg.cotracker.max_moving_points,
                linewidth=self.cotracker_cfg.cotracker.linewidth,
                overwrite=self.cotracker_cfg.cotracker.overwrite
            )

            # Convert to track format
            tracks = self._convert_cotracker_output(pred_tracks, pred_visibility, moving_mask, total_displacement)

        except Exception as e:
            self.logger.error(f"CoTracker tracking failed: {e}")
            return []
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        return tracks

    def _extract_first_frame_points(self):
        """Extract first frame points from tracking results."""
        first_frame_points = []

        for track in self.point_tracks:
            if len(track['points']) > 0 and 0 in track['frame_indices']:
                first_frame_idx = track['frame_indices'].index(0)
                point = track['points'][first_frame_idx]
                first_frame_points.append({
                    'track_id': track['id'],
                    'x': float(point[0]),
                    'y': float(point[1])
                })

        self.first_frame_points = {'all_points': first_frame_points}

        # Save first frame points
        points_path = os.path.join(self.output_dir, "first_frame_points.json")
        with open(points_path, 'w') as f:
            json.dump(self.first_frame_points, f, indent=2)

    def _estimate_camera_parameters(self):
        """Estimate camera parameters if not provided."""
        # Placeholder implementation
        # In practice, you would use camera calibration or SLAM

        width, height = self.frames[0].size
        focal_length = max(width, height) * 0.8

        self.camera_intrinsics = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ])

        # Simple extrinsics (camera at origin looking down -Z)
        self.camera_extrinsics = np.eye(4)
        self.camera_extrinsics[2, 3] = 2.0  # Move camera back

        self.logger.info("Using estimated camera parameters")


    def _create_foundation_pose_wrapper(self, mesh, debug_dir):
        """Create FoundationPose wrapper."""
        class PoseWrapper:
            def __init__(self, mesh, debug_dir, refiner_iter=5):
                scorer = ScorePredictor()
                refiner = PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()
                self.est = FoundationPose(
                    model_pts=mesh.vertices,
                    model_normals=mesh.vertex_normals,
                    mesh=mesh,
                    scorer=scorer,
                    refiner=refiner,
                    debug_dir=debug_dir,
                    debug=0,
                    glctx=glctx
                )
                self.refiner_iter = refiner_iter

            @torch.no_grad()
            def estimate_pose_with_mask(self, image, depth, mask, K):
                """Estimate pose for a single image using mask."""
                pose = self.est.register(K=K, rgb=image, depth=depth, ob_mask=mask, iteration=self.refiner_iter)
                torch.cuda.empty_cache()
                return pose.reshape(4, 4)

        return PoseWrapper(mesh, debug_dir)

    def _estimate_pose_foundation_pose(self) -> np.ndarray:
        """Estimate pose using FoundationPose."""
        if self.foundation_pose_wrapper is None:
            raise RuntimeError("FoundationPose wrapper not available. Check initialization.")

        # Convert first frame to required format
        first_frame_cv = cv2.cvtColor(np.array(self.frames[0]), cv2.COLOR_RGB2BGR)

        # Create a simple depth map (you need to provide actual depth)
        height, width = first_frame_cv.shape[:2]
        depth = np.ones((height, width), dtype=np.float32) * 1.0  # TODO: Replace with actual depth

        # Create a mask from all first frame points
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.first_frame_points and 'all_points' in self.first_frame_points:
            for point_data in self.first_frame_points['all_points']:
                x, y = int(point_data['x']), int(point_data['y'])
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(mask, (x, y), 5, 255, -1)

        if mask.sum() == 0:
            raise ValueError("No valid mask points for FoundationPose estimation")

        pose = self.foundation_pose_wrapper.estimate_pose_with_mask(
            first_frame_cv, depth, mask, self.camera_intrinsics
        )
        return pose

    def _render_glb_from_viewpoint(self) -> np.ndarray:
        """Render GLB from estimated viewpoint."""
        return self._render_glb_with_pose(np.eye(4))

    def _render_glb_with_pose(self, pose: np.ndarray, white_background: bool = False) -> np.ndarray:
        """Render GLB with given pose."""
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0])

        # Add mesh with pose
        mesh = pyrender.Mesh.from_trimesh(self.glb_mesh)
        scene.add(mesh, pose=pose)

        # Setup camera
        width, height = self.frames[0].size
        camera = pyrender.IntrinsicsCamera(
            fx=self.camera_intrinsics[0, 0],
            fy=self.camera_intrinsics[1, 1],
            cx=self.camera_intrinsics[0, 2],
            cy=self.camera_intrinsics[1, 2]
        )
        scene.add(camera, pose=self.camera_extrinsics)

        # Add lighting
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=self.camera_extrinsics)

        # Render
        renderer = pyrender.OffscreenRenderer(width, height)
        color, depth = renderer.render(scene)
        renderer.delete()

        return color

    def _filter_object_points(self) -> List[Tuple[float, float]]:
        """Filter points that are on the object vs background using white background render."""
        white_bg_path = os.path.join(self.output_dir, "posed_render_white_bg.png")
        white_bg_img = cv2.imread(white_bg_path)

        object_points = []

        for point_data in self.first_frame_points['all_points']:
            x, y = int(point_data['x']), int(point_data['y'])

            # Check if point is on object (not white background)
            if (0 <= x < white_bg_img.shape[1] and 0 <= y < white_bg_img.shape[0]):
                pixel = white_bg_img[y, x]
                # If pixel is not white (with some tolerance), it's on the object
                if np.sum(pixel) < 240 * 3:  # Not pure white
                    object_points.append((float(point_data['x']), float(point_data['y'])))

        self.logger.info(f"Filtered {len(object_points)} object points from {len(self.first_frame_points['all_points'])} total points")
        return object_points

    def _run_sam2_segmentation(self, object_points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Run SAM2 segmentation with object points as prompts."""
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 predictor not initialized")

        if not object_points:
            raise ValueError("No object points available for SAM2 segmentation")

        # Convert first frame to RGB array
        first_frame_rgb = np.array(self.frames[0])

        # Set image for SAM2 predictor
        self.sam2_predictor.set_image(first_frame_rgb)

        # Prepare points and labels (all positive prompts)
        points_np = np.array(object_points)
        labels_np = np.ones(len(object_points), dtype=np.int32)  # All positive prompts

        # Run SAM2 prediction
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True
        )

        if masks.size == 0:
            raise RuntimeError("SAM2 generated no masks")

        # Find the best mask based on scores
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]

        # Ensure we have a valid 2D mask
        if best_mask.ndim != 2 or best_mask.sum() == 0:
            raise RuntimeError("SAM2 generated invalid or empty mask")

        # Create supervision detection object
        detection = sv.Detections(
            xyxy=sv.mask_to_xyxy(np.array([best_mask])),
            mask=np.array([best_mask]),
            class_id=np.array([0])
        )

        return {
            'detections': detection,
            'masks': [{
                'mask': best_mask,
                'bbox': detection.xyxy[0].tolist(),
                'area': int(best_mask.sum()),
                'confidence': float(best_score)
            }],
            'points_used': object_points,
            'method': 'SAM2',
            'num_points_used': len(object_points)
        }

    def _visualize_segmentation_results(self):
        """Visualize segmentation results."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Original first frame with points
        first_frame = np.array(self.frames[0])
        axes[0].imshow(first_frame)

        if self.first_frame_points['all_points']:
            points = np.array([(p['x'], p['y']) for p in self.first_frame_points['all_points']])
            axes[0].scatter(points[:, 0], points[:, 1], c='red', s=10, alpha=0.7)

        axes[0].set_title('First Frame with Tracking Points')
        axes[0].axis('off')

        # Rendered frame
        if self.rendered_frame is not None:
            axes[1].imshow(self.rendered_frame)
            axes[1].set_title('Rendered GLB in Estimated Pose')
            axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'segmentation_overview.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_tracking_results(self):
        """Save tracking results."""
        tracking_dir = os.path.join(self.output_dir, "tracking")
        os.makedirs(tracking_dir, exist_ok=True)

        tracks_path = os.path.join(tracking_dir, "point_tracks.json")
        with open(tracks_path, 'w') as f:
            json.dump(self._serialize_for_json(self.point_tracks), f, indent=2)

    def _frames_to_video(self, output_path: str):
        """Convert frames to video for CoTracker."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = np.array(self.frames[0]).shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        for frame in self.frames:
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_cv)
        out.release()

    def _convert_cotracker_output(self, pred_tracks, pred_visibility, moving_mask, total_displacement):
        """Convert CoTracker output to track format."""
        tracks = []

        pred_tracks_np = pred_tracks.squeeze().cpu().numpy()
        pred_visibility_np = pred_visibility.squeeze().cpu().numpy()

        num_frames, num_points, _ = pred_tracks_np.shape

        for point_idx in range(num_points):
            if hasattr(moving_mask, '__len__') and not moving_mask[point_idx]:
                continue

            track = {
                'id': point_idx,
                'points': [],
                'frame_indices': [],
                'active': True,
                'displacement': total_displacement[point_idx] if hasattr(total_displacement, '__len__') else 0
            }

            for frame_idx in range(num_frames):
                if pred_visibility_np[frame_idx, point_idx] > 0.5:
                    x, y = pred_tracks_np[frame_idx, point_idx]
                    track['points'].append((float(x), float(y)))
                    track['frame_indices'].append(frame_idx)

            if len(track['points']) > 1:
                tracks.append(track)

        return tracks

    def _extract_frames_fallback(self, num_frames: int) -> List[Image.Image]:
        """Fallback frame extraction using OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)

        cap.release()
        return frames

    def _serialize_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._serialize_for_json(item) for item in obj]
        else:
            return obj

    def run_pipeline(self, num_frames: int = 8) -> Dict[str, Any]:
        """Run the complete tracking-segmentation pipeline."""
        self.logger.info("Starting tracking-segmentation pipeline")

        try:
            # Execute pipeline steps
            self.step1_extract_frames(num_frames)
            self.step2_track_points()
            self.step3_estimate_viewpoint_and_render()
            self.step4_foundation_pose()
            self.step5_render_glb_in_pose()
            results = self.step6_sam2_segmentation()

            self.logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Tracking-based Segmentation Pipeline")
    parser.add_argument("--video", required=True, help="Path to input MP4 video")
    parser.add_argument("--glb", required=True, help="Path to digital asset GLB file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to extract")

    # CoTracker options
    parser.add_argument("--no-cotracker", action="store_true", help="Disable CoTracker")
    parser.add_argument("--grid-size", type=int, default=100, help="CoTracker grid size")
    parser.add_argument("--displacement-threshold", type=float, default=0.1, help="Displacement threshold")
    parser.add_argument("--max-moving-points", type=int, default=1000, help="Max moving points")

    args = parser.parse_args()

    # Build CoTracker configuration
    cotracker_config = {
        'grid_size': args.grid_size,
        'displacement_threshold': args.displacement_threshold,
        'max_moving_points': args.max_moving_points,
    }

    # Initialize and run pipeline
    pipeline = TrackingSegmentationPipeline(
        video_path=args.video,
        glb_path=args.glb,
        output_dir=args.output,
        use_cotracker=not args.no_cotracker,
        cotracker_config=cotracker_config
    )

    results = pipeline.run_pipeline(num_frames=args.frames)

    print(f"Pipeline completed. Results saved to: {args.output}")
    print(f"\nTracking-segmentation results:")
    print(f"  - Extracted frames: {args.output}/extracted_frames/")
    print(f"  - Point tracking: {args.output}/tracking/")
    print(f"  - First frame points: {args.output}/first_frame_points.json")
    print(f"  - Object pose: {args.output}/object_pose.json")
    print(f"  - Rendered frames: {args.output}/posed_render.png")
    print(f"  - Segmentation results: {args.output}/segmentation_results.json")


if __name__ == "__main__":
    main()