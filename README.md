Main methods I added:

- articulate_anyhting/2d_to_3d.py

"""
usage: 'python articulate_anything/2d_to_3d.py drawer'
input: name of RLBench object

expects the data structure to be the following:

datasets/
   output_views/
      drawer/
         camera_params_...
         depth_drawer_...
         render_drawer_...
         drawer.glb # generated object mesh
   segmentation_masks/
      drawer/
         render_drawer_..._results.json
         render_drawer_..._vis.png


This method takes segmented views of the target object along with the mesh and reprojects the segmentation masks into 3D pointclouds.
It groups segmented part instances together if they overlap in 3D.
The output is visualizations of the part meshes and segmented point clouds as well as part meshes for each unique part.
"""


- articulate_anything/chamfer_distance.py
"""
usage: 'python articulate_anything/chamfer_distance.py drawer drawer y'
input: name of groundtruth object "name_RLBench.obj", name of generated mesh, whether to save visualization (y or n)

This method aligns the two meshes and calculates the chamfer distance between them
"""


- articulate_anything/detect_articulated_parts.py
"""
This method takes as input a video demonstration of the target object and makes a VLM call to detect the individual parts of the object.
The output is a formatted string of the detected parts in the input format of dino/SAM, such that it can directly be used for part segmentation.
"""


- articulate_anything/mesh_seg_dino-x.py
"""
This method takes as input the directory to rendered views of the target object and outputs segmentation masks, both as .png and as .json by calling the DINO-X API
"""


- examples/articulate_test.ipynb
"""
This notebook sets up articulate-anything such that it works with generated object meshes.
The helper functions are still WIP.
"""



This work is based on the code from the following paper:

# Articulate Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model
# ICLR 2025


[![arXiv](https://img.shields.io/badge/arXiv-2401.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2410.13882)
