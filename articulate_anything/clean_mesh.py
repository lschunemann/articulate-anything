import trimesh
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mesh")
parser.add_argument("--cc")

args = parser.parse_args()

mesh_path = args.mesh #f"/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/laptop_multi-view/output/laptop_multi-view_lid_lid_0.obj"


scene = trimesh.load(mesh_path)
mesh = list(scene.geometry.values())[0]
# mesh = scene

cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=args.cc) #100

mask = np.zeros(len(mesh.faces), dtype=bool)
mask[np.concatenate(cc)] = True
mesh.update_faces(mask)

mesh.export(mesh_path[:-4] + '_clean.' + mesh_path[-3:])
print(f"saved mesh to : {mesh_path[:-4] + '_clean.' + mesh_path[-3:]}")