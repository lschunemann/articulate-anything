#!/usr/bin/env python3
"""
Test script for the video articulation pipeline

This script demonstrates how to use the video_articulation_pipeline.py
with example parameters.
"""

import os
import sys
from video_articulation_pipeline import VideoArticulationPipeline


def test_pipeline():
    """Test the video articulation pipeline with example data."""
    
    # Example paths - you should replace these with your actual file paths
    video_path = "example_video.mp4"  # Path to your MP4 video
    glb_path = "example_object.glb"   # Path to your GLB digital asset
    output_dir = "pipeline_output"    # Output directory
    
    # Check if input files exist
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found at {video_path}")
        print("Please update the video_path variable with your actual video file")
        return
    
    if not os.path.exists(glb_path):
        print(f"Warning: GLB file not found at {glb_path}")
        print("Please update the glb_path variable with your actual GLB file")
        return
    
    # Initialize pipeline with point tracking segmentation
    print("Initializing video articulation pipeline with point tracking...")
    pipeline = VideoArticulationPipeline(
        video_path=video_path,
        glb_path=glb_path,
        output_dir=output_dir
    )
    
    try:
        # Run the pipeline with point tracking segmentation
        print("Running pipeline with point tracking segmentation...")
        results = pipeline.run_pipeline(
            num_frames=8,  # Extract 8 frames from the video
            skip_final_articulation_estimation=True  # Focus on segmentation
        )
        
        print("Pipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Display tracking results summary
        print("\nTracking-based Segmentation Results:")
        print(f"- Number of point tracks: {len(pipeline.point_tracks)}")
        print(f"- Number of segmented parts: {len(pipeline.segmentation_results[0]['annotations']) if pipeline.segmentation_results else 0}")
        
        # Show output structure
        print(f"\nGenerated outputs:")
        print(f"  - Extracted frames: {output_dir}/extracted_frames/")
        print(f"  - Point tracking results: {output_dir}/tracking/")
        print(f"  - Tracking visualizations: {output_dir}/tracking/tracking_frame_*.png")
        print(f"  - Rendered GLB views: {output_dir}/rendered_frames/")
        print(f"  - Estimated viewpoints: {output_dir}/estimated_viewpoints/")
        
        if pipeline.segmentation_results:
            parts = set()
            for frame_result in pipeline.segmentation_results:
                for ann in frame_result['annotations']:
                    parts.add(ann['class_name'])
            print(f"  - Identified parts: {', '.join(sorted(parts))}")
    
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        print("Please check your input files and ensure OpenCV, pyrender, and other dependencies are installed")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Video Articulation Pipeline")
    parser.add_argument("--video", help="Path to input MP4 video")
    parser.add_argument("--glb", help="Path to GLB digital asset")
    parser.add_argument("--output", default="test_output", help="Output directory")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to extract")
    parser.add_argument("--enable-articulation", action="store_true", help="Enable articulation estimation")
    parser.add_argument("--no-cotracker", action="store_true", help="Use Lucas-Kanade instead of CoTracker")
    
    args = parser.parse_args()
    
    if args.video and args.glb:
        # Use command line arguments
        pipeline = VideoArticulationPipeline(
            video_path=args.video,
            glb_path=args.glb,
            output_dir=args.output,
            use_cotracker=not args.no_cotracker
        )
        
        results = pipeline.run_pipeline(
            num_frames=args.frames,
            skip_final_articulation_estimation=not args.enable_articulation
        )
        
        print(f"Pipeline completed. Results in {args.output}")
        print(f"Segmentation method used: Point tracking")
    else:
        # Run test with example parameters
        print("Running test with example parameters...")
        print("Use --video and --glb arguments for real usage")
        print("Uses point tracking segmentation (no external dependencies needed)")
        test_pipeline()


if __name__ == "__main__":
    main()