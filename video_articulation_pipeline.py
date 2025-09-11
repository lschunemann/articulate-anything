#!/usr/bin/env python3
"""
Video-based Articulation Estimation Pipeline

This script implements a pipeline to estimate articulation from an MP4 video
of an object being manipulated, given a corresponding .glb digital asset.

Pipeline stages:
1. Extract frames from MP4 video
2. (Skip camera motion compensation - assume static camera)
3. Segment objects in video frames
4. Detect articulated parts
5. Estimate viewpoint and render GLB from matching view
6. Establish point correspondences and estimate articulation

Usage:
    python video_articulation_pipeline.py --video input_video.mp4 --glb digital_asset.glb --output output_dir
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
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial import cKDTree
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Clustering will use simplified method.")

# Import CoTracker utilities if available
try:
    from articulate_anything.utils.cotracker_utils import make_cotracker
    from omegaconf import OmegaConf
    COTRACKER_AVAILABLE = True
except ImportError:
    COTRACKER_AVAILABLE = False
    print("Warning: CoTracker not available. Using Lucas-Kanade optical flow as fallback.")

# Import alignment utilities if available
try:
    from phystwin_alignment.utils.align_util import (
        render_multi_images,
        render_image,
        as_mesh,
        project_2d_to_3d,
    )
    from phystwin_alignment.match_pairs import image_pair_matching
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False
    print("Warning: phystwin_alignment module not available. Using simplified implementations.")

# Import existing articulate-anything modules  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from articulate_anything.utils.viz import get_frames_from_video
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("Warning: articulate_anything.utils.viz not available. Using fallback frame extraction.")

import pycocotools.mask as mask_util


class VideoArticulationPipeline:
    """Main pipeline class for video-based articulation estimation."""
    
    def __init__(self, video_path: str, glb_path: str, output_dir: str, 
                 use_cotracker: bool = True, cotracker_config: Optional[Dict] = None):
        self.video_path = "/home/link/DreMa/third_party/articulate-anything/datasets/RLBench/videos/" + video_path 
        self.glb_path = "/home/link/DreMa/third_party/articulate-anything/datasets/output_views/" + glb_path + f"/{glb_path}.glb"
        self.output_dir = "/home/link/DreMa/third_party/articulate-anything/video_seg_results/" + output_dir
        
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking configuration
        self.use_cotracker = use_cotracker and COTRACKER_AVAILABLE
        if self.use_cotracker:
            # Set up CoTracker configuration
            default_cotracker_config = {
                'mode': 'online',
                'checkpoint_path': '../co-tracker/checkpoints/cotracker2.pth',
                'grid_size': 30,
                'render_all': False,
                'displacement_threshold': None,
                'max_moving_points': None,
                'linewidth': 5,
                'overwrite': True
            }
            if cotracker_config:
                default_cotracker_config.update(cotracker_config)
            
            # Create OmegaConf config for CoTracker
            self.cotracker_cfg = OmegaConf.create({'cotracker': default_cotracker_config})
            
            try:
                self.cotracker = make_cotracker(self.cotracker_cfg)
                self.logger.info("CoTracker initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CoTracker: {e}. Falling back to Lucas-Kanade.")
                self.use_cotracker = False
        
        if not self.use_cotracker:
            self.logger.info("Using Lucas-Kanade optical flow for point tracking")
        
        # Initialize components
        self.frames: List[Image.Image] = []
        self.segmentation_results: List[Dict] = []
        self.tracking_results: Dict = {}
        self.articulated_parts: str = ""
        self.glb_mesh = None
        self.camera_params = None
        self.point_tracks = []
        
    def step1_extract_frames(self, num_frames: int = 8) -> List[Image.Image]:
        """Step 1: Extract frames from MP4 video."""
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
            # Fallback frame extraction using OpenCV
            self.frames = self._extract_frames_fallback(num_frames)
        
        # Save extracted frames
        frames_dir = os.path.join(self.output_dir, "extracted_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(self.frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            frame.save(frame_path)
            
        self.logger.info(f"Extracted and saved {len(self.frames)} frames to {frames_dir}")
        return self.frames
    
    def step2_skip_camera_motion(self):
        """Step 2: Skip camera motion compensation (assume static camera)."""
        self.logger.info("Step 2: Skipping camera motion compensation (static camera assumed)")
        
    def step3_segment_objects(self) -> List[Dict]:
        """Step 3: Segment objects using point tracking and clustering."""
        self.logger.info("Step 3: Segmenting objects using point tracking and clustering")
        
        if not self.frames:
            raise ValueError("No frames available. Run step1_extract_frames first.")
        
        # Track points across all frames
        self.point_tracks = self._track_points_across_frames()
        
        # Cluster tracked points to identify parts
        clustered_tracks = self._cluster_point_tracks(self.point_tracks)
        
        # Convert to segmentation format
        self.segmentation_results = self._tracks_to_segmentation_format(clustered_tracks)
        
        # Save tracking results
        self._save_tracking_results(clustered_tracks)
        
        self.logger.info(f"Completed tracking-based segmentation with {len(clustered_tracks)} clusters")
        return self.segmentation_results
    
    
    def step4_detect_articulated_parts(self) -> str:
        """Step 4: Skip articulated parts detection (not needed for point tracking approach)."""
        self.logger.info("Step 4: Skipping articulated parts detection (using point tracking clustering instead)")
        return ""
    
    def step5_estimate_viewpoint_and_render(self) -> List[np.ndarray]:
        """Step 5: Estimate viewpoint and render GLB from matching view."""
        self.logger.info("Step 5: Estimating viewpoint and rendering GLB")
        
        if not os.path.exists(self.glb_path):
            raise FileNotFoundError(f"GLB file not found: {self.glb_path}")
        
        # Load GLB mesh
        self.glb_mesh = trimesh.load(self.glb_path)
        if hasattr(self.glb_mesh, 'geometry'):
            # If it's a scene, get the first mesh
            self.glb_mesh = list(self.glb_mesh.geometry.values())[0]
        
        # Save mesh as temporary file for alignment
        temp_mesh_path = os.path.join(self.output_dir, "temp_mesh.obj")
        self.glb_mesh.export(temp_mesh_path)
        
        # Estimate camera parameters and render frames
        rendered_frames = []
        render_dir = os.path.join(self.output_dir, "rendered_frames")
        os.makedirs(render_dir, exist_ok=True)
        
        viewpoint_dir = os.path.join(self.output_dir, "estimated_viewpoints")
        os.makedirs(viewpoint_dir, exist_ok=True)
        
        for i, frame in enumerate(self.frames):
            self.logger.info(f"Processing frame {i+1}/{len(self.frames)}")
            
            # Estimate viewpoint for this frame
            camera_params = self._estimate_viewpoint_from_frame(frame, i)
            
            # Render GLB from estimated viewpoint
            rendered_frame = self._render_glb_with_pose(camera_params)
            rendered_frames.append(rendered_frame)
            
            # Save rendered frame and camera parameters
            render_path = os.path.join(render_dir, f"render_{i:03d}.png")
            cv2.imwrite(render_path, rendered_frame)
            
            viewpoint_path = os.path.join(viewpoint_dir, f"viewpoint_{i:03d}.json")
            with open(viewpoint_path, 'w') as f:
                json.dump(self._serialize_camera_params(camera_params), f, indent=2)
        
        self.logger.info(f"Rendered {len(rendered_frames)} frames from GLB")
        return rendered_frames
    
    def step6_point_correspondence_and_articulation(self) -> Dict[str, Any]:
        """Step 6: Establish point correspondences and estimate articulation."""
        self.logger.info("Step 6: Establishing point correspondences and estimating articulation")
        
        if not self.segmentation_results:
            raise ValueError("No segmentation results available. Run step3_segment_objects first.")
        
        # Load rendered frames and viewpoint data
        render_dir = os.path.join(self.output_dir, "rendered_frames")
        viewpoint_dir = os.path.join(self.output_dir, "estimated_viewpoints")
        
        correspondences = self._establish_correspondences(render_dir, viewpoint_dir)
        
        # Track object parts across frames
        part_trajectories = self._track_part_motion(correspondences)
        
        # Estimate articulation parameters
        articulation_params = self._estimate_articulation_parameters(part_trajectories)
        
        # Save results
        results = {
            'correspondences': correspondences,
            'part_trajectories': part_trajectories,
            'articulation_params': articulation_params,
            'detected_parts': self.articulated_parts,
            'num_frames': len(self.frames)
        }
        
        results_path = os.path.join(self.output_dir, "articulation_results.json")
        with open(results_path, 'w') as f:
            json.dump(self._serialize_results(results), f, indent=2)
        
        self.logger.info("Point correspondence and articulation estimation completed")
        return articulation_params
    
    def _establish_correspondences(self, render_dir: str, viewpoint_dir: str) -> List[Dict[str, Any]]:
        """Establish point correspondences between video frames and rendered frames."""
        correspondences = []
        frames_dir = os.path.join(self.output_dir, "extracted_frames")
        
        for i in range(len(self.frames)):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            render_path = os.path.join(render_dir, f"render_{i:03d}.png")
            viewpoint_path = os.path.join(viewpoint_dir, f"viewpoint_{i:03d}.json")
            
            if not all(os.path.exists(p) for p in [frame_path, render_path, viewpoint_path]):
                self.logger.warning(f"Missing files for frame {i}, skipping")
                continue
            
            # Load images and viewpoint
            frame_img = cv2.imread(frame_path)
            render_img = cv2.imread(render_path)
            
            with open(viewpoint_path, 'r') as f:
                viewpoint = json.load(f)
            
            # Establish correspondences for this frame pair
            frame_correspondences = self._find_frame_correspondences(
                frame_img, render_img, viewpoint, i
            )
            
            correspondences.append({
                'frame_index': i,
                'correspondences': frame_correspondences,
                'viewpoint': viewpoint
            })
        
        return correspondences
    
    def _find_frame_correspondences(self, frame_img: np.ndarray, render_img: np.ndarray, 
                                   viewpoint: Dict, frame_idx: int) -> List[Dict[str, Any]]:
        """Find point correspondences between a video frame and rendered frame."""
        # Convert to grayscale for feature detection
        frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        render_gray = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
        
        # Use SIFT for robust feature detection
        sift = cv2.SIFT_create(nfeatures=500)
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(frame_gray, None)
        kp2, des2 = sift.detectAndCompute(render_gray, None)
        
        correspondences = []
        
        if des1 is not None and des2 is not None:
            # Match features using FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Convert matches to correspondences with 2D-3D mapping
            camera_matrix = np.array(viewpoint['camera_matrix'])
            camera_pose = np.array(viewpoint['camera_pose'])
            
            for match in good_matches:
                # 2D points
                pt_frame = kp1[match.queryIdx].pt
                pt_render = kp2[match.trainIdx].pt
                
                # Project render point to 3D (simplified)
                pt_3d = self._project_to_3d(pt_render, camera_matrix, camera_pose)
                
                correspondences.append({
                    'frame_2d': list(pt_frame),
                    'render_2d': list(pt_render), 
                    'world_3d': pt_3d.tolist() if pt_3d is not None else None,
                    'confidence': 1.0 / (1.0 + match.distance)
                })
        
        return correspondences
    
    def _project_to_3d(self, pt_2d: tuple, camera_matrix: np.ndarray, 
                       camera_pose: np.ndarray) -> Optional[np.ndarray]:
        """Project 2D point to 3D space (simplified version)."""
        # This is a simplified implementation
        # In practice, you'd need depth information or mesh intersection
        
        # For now, assume points are on the mesh surface
        # This would require proper ray-mesh intersection
        
        # Placeholder - project to a fixed depth
        x, y = pt_2d
        depth = 1.0  # Simplified assumption
        
        # Convert to normalized coordinates
        x_norm = (x - camera_matrix[0, 2]) / camera_matrix[0, 0]
        y_norm = (y - camera_matrix[1, 2]) / camera_matrix[1, 1]
        
        # Create ray in camera space
        ray_camera = np.array([x_norm * depth, y_norm * depth, depth, 1.0])
        
        # Transform to world space
        ray_world = camera_pose @ ray_camera
        
        return ray_world[:3]
    
    def _track_part_motion(self, correspondences: List[Dict]) -> Dict[str, List[Dict]]:
        """Track motion of different object parts across frames."""
        part_trajectories = {}
        
        # Use segmentation results to group correspondences by part
        for i, corr_data in enumerate(correspondences):
            if i >= len(self.segmentation_results):
                continue
                
            seg_result = self.segmentation_results[i]
            frame_correspondences = corr_data['correspondences']
            
            # Group correspondences by segmented parts
            for annotation in seg_result.get('annotations', []):
                part_name = annotation['class_name']
                
                if part_name not in part_trajectories:
                    part_trajectories[part_name] = []
                
                # Find correspondences within this part's mask
                part_points = self._filter_correspondences_by_mask(
                    frame_correspondences, annotation['segmentation'],
                    seg_result['img_width'], seg_result['img_height']
                )
                
                part_trajectories[part_name].append({
                    'frame_index': i,
                    'points': part_points,
                    'bbox': annotation['bbox'],
                    'score': annotation['score']
                })
        
        return part_trajectories
    
    def _filter_correspondences_by_mask(self, correspondences: List[Dict], 
                                       mask_rle: Dict, width: int, height: int) -> List[Dict]:
        """Filter correspondences to those within a segmentation mask."""
        # Decode mask from RLE format
        mask = mask_util.decode(mask_rle)
        
        filtered_correspondences = []
        for corr in correspondences:
            x, y = corr['frame_2d']
            x, y = int(x), int(y)
            
            if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
                filtered_correspondences.append(corr)
        
        return filtered_correspondences
    
    def _estimate_articulation_parameters(self, part_trajectories: Dict) -> Dict[str, Any]:
        """Estimate articulation parameters from part trajectories."""
        articulation_params = {
            'joints': [],
            'joint_types': [],
            'joint_axes': [],
            'joint_limits': [],
            'joint_origins': []
        }
        
        # Analyze each part's motion to determine joint type and parameters
        for part_name, trajectory in part_trajectories.items():
            if len(trajectory) < 2:
                continue
            
            joint_params = self._analyze_part_motion(part_name, trajectory)
            
            if joint_params:
                articulation_params['joints'].append(part_name)
                articulation_params['joint_types'].append(joint_params['type'])
                articulation_params['joint_axes'].append(joint_params['axis'])
                articulation_params['joint_limits'].append(joint_params['limits'])
                articulation_params['joint_origins'].append(joint_params['origin'])
        
        return articulation_params
    
    def _analyze_part_motion(self, part_name: str, trajectory: List[Dict]) -> Optional[Dict]:
        """Analyze motion pattern of a part to determine joint parameters."""
        if len(trajectory) < 2:
            return None
        
        # Extract 3D points across frames
        points_3d = []
        for frame_data in trajectory:
            for point in frame_data['points']:
                if point['world_3d'] is not None:
                    points_3d.append(np.array(point['world_3d']))
        
        if len(points_3d) < 3:
            return None
        
        points_3d = np.array(points_3d)
        
        # Analyze motion pattern
        motion_type, motion_params = self._classify_motion(points_3d)
        
        return {
            'type': motion_type,
            'axis': motion_params.get('axis', [0, 0, 1]),
            'origin': motion_params.get('origin', [0, 0, 0]),
            'limits': motion_params.get('limits', [-np.pi, np.pi])
        }
    
    def _classify_motion(self, points: np.ndarray) -> tuple:
        """Classify motion type from 3D point trajectories."""
        # Simplified motion classification
        # In practice, this would use more sophisticated analysis
        
        if len(points) < 3:
            return 'fixed', {}
        
        # Calculate displacement vectors
        displacements = np.diff(points, axis=0)
        
        # Check for linear motion (prismatic joint)
        displacement_norms = np.linalg.norm(displacements, axis=1)
        if np.std(displacement_norms) < 0.1:  # Nearly constant displacement magnitude
            # Likely prismatic joint
            avg_direction = np.mean(displacements, axis=0)
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
            
            return 'prismatic', {
                'axis': avg_direction.tolist(),
                'origin': points[0].tolist(),
                'limits': [0, np.linalg.norm(points[-1] - points[0])]
            }
        
        # Check for rotational motion (revolute joint)
        # Find potential rotation center
        center = np.mean(points, axis=0)
        radial_vectors = points - center
        radii = np.linalg.norm(radial_vectors, axis=1)
        
        if np.std(radii) < 0.2:  # Nearly constant radius
            # Likely revolute joint
            # Estimate rotation axis using cross products
            if len(radial_vectors) >= 2:
                axis = np.cross(radial_vectors[0], radial_vectors[-1])
                if np.linalg.norm(axis) > 0.01:
                    axis = axis / np.linalg.norm(axis)
                else:
                    axis = [0, 0, 1]  # Default axis
            else:
                axis = [0, 0, 1]
            
            return 'revolute', {
                'axis': axis.tolist(),
                'origin': center.tolist(),
                'limits': [-np.pi, np.pi]  # Full rotation assumed
            }
        
        return 'fixed', {}
    
    def _serialize_for_json(self, obj):
        """Convert numpy types and other non-JSON-serializable objects to JSON-serializable types."""
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
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Serialize results for JSON storage."""
        return self._serialize_for_json(results)
    
    def _estimate_viewpoint_from_frame(self, frame: Image.Image, frame_idx: int) -> Dict[str, Any]:
        """Estimate camera viewpoint from a single frame using feature matching."""
        width, height = frame.size
        
        if ALIGNMENT_AVAILABLE:
            try:
                # Use advanced alignment if available
                return self._estimate_pose_with_alignment(frame, frame_idx)
            except Exception as e:
                self.logger.warning(f"Advanced alignment failed: {e}. Using simplified method.")
        
        # Simplified estimation
        focal_length = max(width, height) * 0.8  # Rough estimate
        
        # Generate multiple candidate poses and select best one
        best_pose = self._select_best_pose_simple(frame, focal_length)
        
        return {
            'width': width,
            'height': height,
            'focal_length': focal_length,
            'camera_matrix': np.array([
                [focal_length, 0, width/2],
                [0, focal_length, height/2],
                [0, 0, 1]
            ]),
            'camera_pose': best_pose,
            'fov': 2 * np.arctan(height / (2 * focal_length))
        }
    
    def _estimate_pose_with_alignment(self, frame: Image.Image, frame_idx: int) -> Dict[str, Any]:
        """Advanced pose estimation using alignment utilities."""
        # Convert PIL to OpenCV format
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        width, height = frame.size
        fov = np.pi / 3  # 60 degrees default
        focal_length = height / (2 * np.tan(fov / 2))
        
        # Calculate suitable rendering radius
        bounding_box = self.glb_mesh.bounds
        max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
        radius = 2 * (max_dimension / 2) / np.tan(fov / 2)
        
        temp_mesh_path = os.path.join(self.output_dir, "temp_mesh.obj")
        
        # Render multiple candidate views
        colors, depths, camera_poses, camera_intrinsics = render_multi_images(
            temp_mesh_path,
            width, height, fov,
            radius=radius,
            num_samples=16,  # More samples for better matching
            num_ups=4,
            device="cpu"
        )
        
        # Find best matching pose using feature matching
        best_pose_idx = 0
        best_score = 0
        
        for i, rendered_color in enumerate(colors):
            try:
                # Convert to grayscale for feature matching
                frame_gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)
                render_gray = cv2.cvtColor(rendered_color, cv2.COLOR_BGR2GRAY)
                
                # Use SIFT or ORB for feature matching as fallback
                score = self._compute_matching_score(frame_gray, render_gray)
                
                if score > best_score:
                    best_score = score
                    best_pose_idx = i
                    
            except Exception as e:
                self.logger.debug(f"Feature matching failed for pose {i}: {e}")
                continue
        
        return {
            'width': width,
            'height': height,
            'focal_length': focal_length,
            'camera_matrix': camera_intrinsics[best_pose_idx],
            'camera_pose': camera_poses[best_pose_idx],
            'fov': fov,
            'matching_score': best_score
        }
    
    def _compute_matching_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute feature matching score between two images."""
        # Use ORB detector for feature matching
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Return normalized matching score
        if len(matches) == 0:
            return 0.0
        
        # Score based on number of good matches and their quality
        good_matches = [m for m in matches if m.distance < 50]
        score = len(good_matches) / max(len(kp1), len(kp2), 1)
        
        return score
    
    def _select_best_pose_simple(self, frame: Image.Image, focal_length: float) -> np.ndarray:
        """Simple pose selection using silhouette matching."""
        # Generate candidate poses around the object
        bounding_box = self.glb_mesh.bounds
        center = (bounding_box[0] + bounding_box[1]) / 2
        max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
        distance = max_dimension * 2
        
        best_pose = np.eye(4)
        best_pose[:3, 3] = center + np.array([0, 0, distance])
        
        return best_pose
    
    def _render_glb_with_pose(self, camera_params: Dict[str, Any]) -> np.ndarray:
        """Render GLB mesh with given camera parameters."""
        # Create pyrender scene
        scene = pyrender.Scene()
        
        # Add mesh to scene
        mesh = pyrender.Mesh.from_trimesh(self.glb_mesh)
        scene.add(mesh)
        
        # Setup camera
        if 'fov' in camera_params:
            camera = pyrender.PerspectiveCamera(
                yfov=camera_params['fov'],
                aspectRatio=camera_params['width'] / camera_params['height']
            )
        else:
            camera = pyrender.PerspectiveCamera(
                yfov=np.pi / 3.0,
                aspectRatio=camera_params['width'] / camera_params['height']
            )
        
        # Use estimated camera pose
        camera_pose = camera_params['camera_pose']
        scene.add(camera, pose=camera_pose)
        
        # Add lighting
        light = pyrender.SpotLight(
            color=np.ones(3), intensity=3.0,
            innerConeAngle=np.pi/16.0,
            outerConeAngle=np.pi/6.0
        )
        scene.add(light, pose=camera_pose)
        
        # Additional ambient light
        ambient_light = pyrender.DirectionalLight(color=np.ones(3), intensity=0.5)
        scene.add(ambient_light)
        
        # Render
        renderer = pyrender.OffscreenRenderer(
            camera_params['width'], 
            camera_params['height']
        )
        color, depth = renderer.render(scene)
        renderer.delete()
        
        return color
    
    def _serialize_camera_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize camera parameters for JSON storage."""
        serialized = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def run_pipeline(self, num_frames: int = 8, 
                     skip_final_articulation_estimation: bool = False) -> Dict[str, Any]:
        """Run the complete articulation estimation pipeline using point tracking."""
        self.logger.info("Starting video articulation estimation pipeline with point tracking")
        
        try:
            # Execute pipeline steps
            self.step1_extract_frames(num_frames)
            self.step2_skip_camera_motion()
            self.step3_segment_objects()
            self.step4_detect_articulated_parts()
            self.step5_estimate_viewpoint_and_render()
            
            if not skip_final_articulation_estimation:
                results = self.step6_point_correspondence_and_articulation()
            else:
                self.logger.info("Skipping final articulation estimation")
                results = {
                    'joints': [],
                    'joint_types': [],
                    'joint_axes': [],
                    'joint_limits': [],
                    'joint_origins': []
                }
            
            self.logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _track_points_cotracker(self) -> List[Dict[str, Any]]:
        """Track feature points using CoTracker."""
        self.logger.info("Tracking points using CoTracker...")
        
        # Save frames as a temporary video for CoTracker
        temp_video_path = os.path.join(self.output_dir, "temp_video.mp4")
        
        # Convert PIL frames to video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = np.array(self.frames[0]).shape[:2]
        out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
        
        for frame in self.frames:
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_cv)
        out.release()
        
        try:
            # Run CoTracker on the video
            pred_tracks, pred_visibility, moving_mask, total_displacement = self.cotracker.forward(
                video_path=temp_video_path,
                grid_size=self.cotracker_cfg.cotracker.grid_size,
                render_all=self.cotracker_cfg.cotracker.render_all,
                displacement_threshold=self.cotracker_cfg.cotracker.displacement_threshold,
                max_moving_points=self.cotracker_cfg.cotracker.max_moving_points,
                linewidth=self.cotracker_cfg.cotracker.linewidth,
                overwrite=self.cotracker_cfg.cotracker.overwrite
            )
            
            # Convert CoTracker output to our track format
            tracks = []
            
            # pred_tracks shape: (1, num_frames, num_points, 2)
            pred_tracks_np = pred_tracks.squeeze().cpu().numpy()  # (num_frames, num_points, 2)
            pred_visibility_np = pred_visibility.squeeze().cpu().numpy()  # (num_frames, num_points)
            
            num_frames, num_points, _ = pred_tracks_np.shape
            
            for point_idx in range(num_points):
                # Only include moving points if we have a moving mask
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
                    if pred_visibility_np[frame_idx, point_idx] > 0.5:  # Visible
                        x, y = pred_tracks_np[frame_idx, point_idx]
                        track['points'].append((float(x), float(y)))
                        track['frame_indices'].append(frame_idx)
                
                if len(track['points']) > 1:  # Only keep tracks with multiple points
                    tracks.append(track)
            
            self.logger.info(f"CoTracker found {len(tracks)} tracks")
            
        except Exception as e:
            self.logger.error(f"CoTracker tracking failed: {e}")
            self.logger.info("Falling back to Lucas-Kanade tracking")
            return self._track_points_lucas_kanade()
        finally:
            # Clean up temporary video
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        
        return tracks

    def _track_points_lucas_kanade(self) -> List[Dict[str, Any]]:
        """Track feature points using Lucas-Kanade optical flow (fallback method)."""
        self.logger.info("Tracking points using Lucas-Kanade optical flow...")
        
        # Convert frames to OpenCV format
        frames_cv = []
        for frame in self.frames:
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frames_cv.append(frame_cv)
        
        # Initialize tracking
        tracks = []
        track_id = 0
        
        # Detect features in first frame
        first_frame_gray = cv2.cvtColor(frames_cv[0], cv2.COLOR_BGR2GRAY)
        
        # Use goodFeaturesToTrack for initial feature detection
        corners = cv2.goodFeaturesToTrack(
            first_frame_gray,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )
        
        if corners is None:
            self.logger.warning("No initial corners detected")
            return []
        
        # Initialize tracks with first frame corners
        for corner in corners:
            x, y = corner.ravel()
            tracks.append({
                'id': track_id,
                'points': [(float(x), float(y))],
                'frame_indices': [0],
                'active': True
            })
            track_id += 1
        
        # Track points through remaining frames using Lucas-Kanade
        prev_gray = first_frame_gray
        prev_points = corners
        
        for frame_idx in range(1, len(frames_cv)):
            curr_gray = cv2.cvtColor(frames_cv[frame_idx], cv2.COLOR_BGR2GRAY)
            
            if len(prev_points) == 0:
                break
            
            # Track points using Lucas-Kanade optical flow
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_points, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Update tracks with new positions
            good_tracks = []
            good_points = []
            
            for i, (track, point, st, err) in enumerate(zip(tracks, next_points, status, error)):
                if not track['active']:
                    continue
                
                # Extract x, y from the nested array structure
                x, y = point[0] if len(point) > 0 else (0, 0)
                    
                if st[0] == 1 and err[0] < 30:  # Good track
                    track['points'].append((float(x), float(y)))
                    track['frame_indices'].append(frame_idx)
                    good_tracks.append(track)
                    good_points.append([[x, y]])
                else:
                    track['active'] = False
            
            tracks = good_tracks
            prev_points = np.array(good_points, dtype=np.float32) if good_points else np.array([])
            prev_gray = curr_gray
            
            # Add new features if we're losing too many tracks
            if len(tracks) < 100:
                mask = np.zeros_like(curr_gray)
                for track in tracks:
                    if track['active'] and track['frame_indices'][-1] == frame_idx:
                        x, y = track['points'][-1]
                        cv2.circle(mask, (int(x), int(y)), 10, 0, -1)
                
                new_corners = cv2.goodFeaturesToTrack(
                    curr_gray, 
                    maxCorners=200,
                    qualityLevel=0.01,
                    minDistance=10,
                    mask=255-mask
                )
                
                if new_corners is not None:
                    for corner in new_corners:
                        x, y = corner.ravel()
                        tracks.append({
                            'id': track_id,
                            'points': [(float(x), float(y))],
                            'frame_indices': [frame_idx],
                            'active': True
                        })
                        track_id += 1
                    
                    # Update prev_points to include new points
                    all_current_points = []
                    for track in tracks:
                        if track['active'] and track['frame_indices'][-1] == frame_idx:
                            x, y = track['points'][-1]
                            all_current_points.append([[x, y]])
                    prev_points = np.array(all_current_points, dtype=np.float32)
        
        # Filter tracks to only include those with sufficient length
        min_track_length = max(2, len(self.frames) // 4)
        long_tracks = [track for track in tracks if len(track['points']) >= min_track_length]
        
        self.logger.info(f"Tracked {len(long_tracks)} point tracks across {len(self.frames)} frames")
        return long_tracks
    
    def _track_points_across_frames(self) -> List[Dict[str, Any]]:
        """Track feature points across all video frames."""
        if self.use_cotracker:
            return self._track_points_cotracker()
        else:
            return self._track_points_lucas_kanade()
    
    def _cluster_point_tracks(self, tracks: List[Dict]) -> Dict[str, List[Dict]]:
        """Cluster point tracks to identify object parts."""
        self.logger.info("Clustering point tracks to identify parts...")
        
        if not tracks:
            return {}
        
        # Extract track motion features for clustering
        track_features = []
        valid_tracks = []
        
        for track in tracks:
            if len(track['points']) < 3:
                continue
                
            points = np.array(track['points'])
            
            # Compute motion features
            displacement = points[-1] - points[0]  # Overall displacement
            trajectory_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            
            # Velocity features
            velocities = np.diff(points, axis=0)
            avg_velocity = np.mean(velocities, axis=0)
            velocity_std = np.std(velocities, axis=0)
            
            # Position features
            center_pos = np.mean(points, axis=0)
            
            # Use CoTracker displacement if available
            cotracker_displacement = track.get('displacement', np.linalg.norm(displacement))
            
            # Create feature vector (enhanced with CoTracker data when available)
            features = np.concatenate([
                displacement,
                [trajectory_length],
                [cotracker_displacement],  # Add CoTracker displacement as feature
                avg_velocity,
                velocity_std,
                center_pos
            ])
            
            track_features.append(features)
            valid_tracks.append(track)
        
        if not track_features:
            return {}
        
        track_features = np.array(track_features)
        
        if SKLEARN_AVAILABLE:
            # Normalize features
            scaler = StandardScaler()
            track_features_normalized = scaler.fit_transform(track_features)
            
            # Cluster tracks using K-means
            n_clusters = min(8, max(2, len(valid_tracks) // 20))  # Adaptive number of clusters
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(track_features_normalized)
        else:
            # Simplified clustering without sklearn
            self.logger.warning("Using simplified clustering (sklearn not available)")
            track_features_normalized = track_features / (np.std(track_features, axis=0) + 1e-6)
            
            # Simple distance-based clustering
            n_clusters = min(4, max(2, len(valid_tracks) // 30))
            cluster_labels = self._simple_kmeans(track_features_normalized, n_clusters)
        
        # Group tracks by cluster
        clustered_tracks = {}
        for i, label in enumerate(cluster_labels):
            cluster_name = f"part_{label}"
            if cluster_name not in clustered_tracks:
                clustered_tracks[cluster_name] = []
            clustered_tracks[cluster_name].append(valid_tracks[i])
        
        # Filter out clusters with too few tracks
        min_tracks_per_cluster = 5
        filtered_clusters = {
            name: tracks for name, tracks in clustered_tracks.items() 
            if len(tracks) >= min_tracks_per_cluster
        }
        
        self.logger.info(f"Identified {len(filtered_clusters)} parts with sufficient tracks")
        return filtered_clusters
    
    def _tracks_to_segmentation_format(self, clustered_tracks: Dict[str, List[Dict]]) -> List[Dict]:
        """Convert clustered tracks to segmentation format compatible with existing pipeline."""
        segmentation_results = []
        
        for frame_idx in range(len(self.frames)):
            frame_width, frame_height = self.frames[frame_idx].size
            
            annotations = []
            for part_name, tracks in clustered_tracks.items():
                # Get all points for this part in this frame
                points_in_frame = []
                for track in tracks:
                    if frame_idx in track['frame_indices']:
                        track_frame_idx = track['frame_indices'].index(frame_idx)
                        point = track['points'][track_frame_idx]
                        points_in_frame.append(point)
                
                if not points_in_frame:
                    continue
                
                # Create bounding box from points
                points_array = np.array(points_in_frame)
                x_min, y_min = np.min(points_array, axis=0)
                x_max, y_max = np.max(points_array, axis=0)
                
                # Expand bbox slightly
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame_width, x_max + padding)
                y_max = min(frame_height, y_max + padding)
                
                # Create simple segmentation mask (just the bounding box for now)
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.rectangle(mask, (int(x_min), int(y_min)), (int(x_max), int(y_max)), 255, -1)
                
                # Convert to RLE format
                mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
                
                annotations.append({
                    'class_name': part_name,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'segmentation': mask_rle,
                    'score': 1.0,  # Confidence score
                    'points': points_in_frame
                })
            
            segmentation_results.append({
                'frame_index': frame_idx,
                'annotations': annotations,
                'box_format': 'xyxy',
                'img_width': frame_width,
                'img_height': frame_height
            })
        
        return segmentation_results
    
    def _save_tracking_results(self, clustered_tracks: Dict[str, List[Dict]]):
        """Save point tracking and clustering results."""
        tracking_dir = os.path.join(self.output_dir, "tracking")
        os.makedirs(tracking_dir, exist_ok=True)
        
        # Save raw tracks
        tracks_path = os.path.join(tracking_dir, "point_tracks.json")
        with open(tracks_path, 'w') as f:
            json.dump(self._serialize_for_json(self.point_tracks), f, indent=2)
        
        # Save clustered tracks
        clusters_path = os.path.join(tracking_dir, "clustered_tracks.json")
        with open(clusters_path, 'w') as f:
            json.dump(self._serialize_for_json(clustered_tracks), f, indent=2)
        
        # Save visualization
        self._visualize_tracking_results(clustered_tracks)
    
    def project_tracks_to_rendered_views(self) -> Dict[str, Any]:
        """Project tracked points from video frames to rendered GLB views."""
        self.logger.info("Projecting tracked points to rendered views...")
        
        if not self.point_tracks:
            self.logger.warning("No point tracks available for projection")
            return {}
        
        projection_results = []
        viewpoint_dir = os.path.join(self.output_dir, "estimated_viewpoints")
        
        for frame_idx in range(len(self.frames)):
            viewpoint_path = os.path.join(viewpoint_dir, f"viewpoint_{frame_idx:03d}.json")
            
            if not os.path.exists(viewpoint_path):
                continue
            
            # Load viewpoint data
            with open(viewpoint_path, 'r') as f:
                viewpoint = json.load(f)
            
            camera_matrix = np.array(viewpoint['camera_matrix'])
            camera_pose = np.array(viewpoint['camera_pose'])
            
            # Project points for this frame
            frame_projections = []
            for track in self.point_tracks:
                if frame_idx in track['frame_indices']:
                    track_frame_idx = track['frame_indices'].index(frame_idx)
                    video_point = track['points'][track_frame_idx]
                    
                    # Project to 3D using estimated depth
                    world_3d = self._project_to_3d(video_point, camera_matrix, camera_pose)
                    
                    if world_3d is not None:
                        # Project 3D point back to rendered view coordinate system
                        render_2d = self._project_3d_to_2d(world_3d, camera_matrix, camera_pose)
                        
                        frame_projections.append({
                            'track_id': track['id'],
                            'video_2d': list(video_point),
                            'world_3d': world_3d.tolist(),
                            'render_2d': render_2d.tolist() if render_2d is not None else None
                        })
            
            projection_results.append({
                'frame_index': frame_idx,
                'projections': frame_projections,
                'viewpoint': viewpoint
            })
        
        # Save projection results
        projection_path = os.path.join(self.output_dir, "point_projections.json")
        with open(projection_path, 'w') as f:
            json.dump(self._serialize_for_json(projection_results), f, indent=2)
        
        self._visualize_point_projections(projection_results)
        
        self.logger.info(f"Saved point projection results to {projection_path}")
        return {'projections': projection_results}
    
    def _visualize_tracking_results(self, clustered_tracks: Dict[str, List[Dict]]):
        """Create visualizations of tracking results."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        tracking_dir = os.path.join(self.output_dir, "tracking")
        
        # Create color map for parts
        colors = list(mcolors.TABLEAU_COLORS.values())
        part_colors = {name: colors[i % len(colors)] for i, name in enumerate(clustered_tracks.keys())}
        
        # Visualize tracks for each frame
        for frame_idx in range(len(self.frames)):
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Show frame
            frame_cv = cv2.cvtColor(np.array(self.frames[frame_idx]), cv2.COLOR_RGB2BGR)
            ax.imshow(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
            
            # Plot tracks
            for part_name, tracks in clustered_tracks.items():
                color = part_colors[part_name]
                points = []
                
                for track in tracks:
                    if frame_idx in track['frame_indices']:
                        track_frame_idx = track['frame_indices'].index(frame_idx)
                        point = track['points'][track_frame_idx]
                        points.append(point)
                
                if points:
                    points = np.array(points)
                    ax.scatter(points[:, 0], points[:, 1], c=color, label=part_name, s=20)
            
            ax.set_title(f'Frame {frame_idx}: Tracked Points by Part')
            ax.legend()
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(tracking_dir, f'tracking_frame_{frame_idx:03d}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Saved tracking visualizations to {tracking_dir}")
    
    def _simple_kmeans(self, data: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
        """Simple k-means implementation when sklearn is not available."""
        n_samples, n_features = data.shape
        
        # Initialize centroids randomly
        centroids = data[np.random.choice(n_samples, k, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to closest centroids
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return labels
    
    def _project_3d_to_2d(self, point_3d: np.ndarray, camera_matrix: np.ndarray, 
                          camera_pose: np.ndarray) -> Optional[np.ndarray]:
        """Project 3D world point to 2D image coordinates."""
        # Transform to camera coordinate system
        point_3d_homogeneous = np.append(point_3d, 1.0)
        camera_coords = np.linalg.inv(camera_pose) @ point_3d_homogeneous
        
        if camera_coords[2] <= 0:  # Point behind camera
            return None
        
        # Project to image plane
        image_coords = camera_matrix @ camera_coords[:3]
        
        if image_coords[2] != 0:
            pixel_coords = image_coords[:2] / image_coords[2]
            return pixel_coords
        
        return None
    
    def _visualize_point_projections(self, projection_results: List[Dict]):
        """Visualize point projections between video and rendered views."""
        import matplotlib.pyplot as plt
        
        projection_dir = os.path.join(self.output_dir, "point_projections")
        os.makedirs(projection_dir, exist_ok=True)
        
        render_dir = os.path.join(self.output_dir, "rendered_frames")
        frames_dir = os.path.join(self.output_dir, "extracted_frames")
        
        for result in projection_results:
            frame_idx = result['frame_index']
            projections = result['projections']
            
            if not projections:
                continue
            
            # Load video frame and rendered frame
            video_frame_path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
            render_frame_path = os.path.join(render_dir, f"render_{frame_idx:03d}.png")
            
            if not (os.path.exists(video_frame_path) and os.path.exists(render_frame_path)):
                continue
            
            video_img = cv2.imread(video_frame_path)
            render_img = cv2.imread(render_frame_path)
            
            # Create side-by-side visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Video frame with original points
            ax1.imshow(cv2.cvtColor(video_img, cv2.COLOR_BGR2RGB))
            video_points = np.array([p['video_2d'] for p in projections])
            ax1.scatter(video_points[:, 0], video_points[:, 1], c='red', s=30, alpha=0.7)
            ax1.set_title(f'Video Frame {frame_idx}: Original Points')
            ax1.axis('off')
            
            # Rendered frame with projected points
            ax2.imshow(cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB))
            render_points = np.array([p['render_2d'] for p in projections if p['render_2d'] is not None])
            if len(render_points) > 0:
                ax2.scatter(render_points[:, 0], render_points[:, 1], c='blue', s=30, alpha=0.7)
            ax2.set_title(f'Rendered Frame {frame_idx}: Projected Points')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(projection_dir, f'projection_{frame_idx:03d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Saved projection visualizations to {projection_dir}")
    
    def _extract_frames_fallback(self, num_frames: int) -> List[Image.Image]:
        """Fallback frame extraction using OpenCV when articulate_anything.utils.viz is not available."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate frame indices to extract uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and create PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
        
        cap.release()
        return frames


def main():
    parser = argparse.ArgumentParser(description="Video-based Articulation Estimation Pipeline using Point Tracking")
    parser.add_argument("--video", required=True, help="Path to input MP4 video")
    parser.add_argument("--glb", required=True, help="Path to digital asset GLB file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to extract")
    parser.add_argument("--enable-articulation", action="store_true",
                       help="Enable final articulation parameter estimation")
    
    # CoTracker options
    parser.add_argument("--no-cotracker", action="store_true",
                       help="Disable CoTracker and use Lucas-Kanade optical flow only")
    parser.add_argument("--cotracker-mode", choices=["online", "offline"], default="online",
                       help="CoTracker mode (online or offline)")
    parser.add_argument("--cotracker-checkpoint", 
                       help="Path to CoTracker checkpoint file")
    parser.add_argument("--grid-size", type=int, default=30,
                       help="Grid size for CoTracker point initialization")
    parser.add_argument("--displacement-threshold", type=float,
                       help="Minimum displacement threshold for moving points")
    parser.add_argument("--max-moving-points", type=int,
                       help="Maximum number of moving points to track")
    
    args = parser.parse_args()
    
    # Build CoTracker configuration
    cotracker_config = {
        'mode': args.cotracker_mode,
        'grid_size': args.grid_size,
    }
    
    if args.cotracker_checkpoint:
        cotracker_config['checkpoint_path'] = args.cotracker_checkpoint
    if args.displacement_threshold:
        cotracker_config['displacement_threshold'] = args.displacement_threshold
    if args.max_moving_points:
        cotracker_config['max_moving_points'] = args.max_moving_points
    
    # Initialize and run pipeline
    pipeline = VideoArticulationPipeline(
        video_path=args.video,
        glb_path=args.glb,
        output_dir=args.output,
        use_cotracker=not args.no_cotracker,
        cotracker_config=cotracker_config
    )
    
    results = pipeline.run_pipeline(
        num_frames=args.frames,
        skip_final_articulation_estimation=not args.enable_articulation
    )
    
    print(f"Pipeline completed. Results saved to: {args.output}")
    
    print(f"\nPoint tracking-based segmentation results:")
    print(f"  - Extracted frames: {args.output}/extracted_frames/")
    print(f"  - Point tracking results: {args.output}/tracking/")
    print(f"  - Tracking visualizations: {args.output}/tracking/tracking_frame_*.png")
    print(f"  - Rendered GLB views: {args.output}/rendered_frames/")
    print(f"  - Estimated viewpoints: {args.output}/estimated_viewpoints/")
    if args.enable_articulation:
        print(f"  - Articulation results: {args.output}/articulation_results.json")


if __name__ == "__main__":
    main()