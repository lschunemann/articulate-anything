#!/bin/bash

# input_dir="/home/link/DreMa/third_party/articulate-anything/datasets/partnet-mobility-v0/dataset/generated_object"
input_dir="/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/laptop_multi-view/output"
# input_dir="/home/link/DreMa/third_party/articulate-anything/datasets/segmentation_masks/microwave_multi-view/output/parts"

for file in "$input_dir"/*.glb; do
  base_name=$(basename "$file" .glb)
  assimp export "$file" "$input_dir/$base_name.obj"
done

echo "Conversion finished! Files saved in $input_dir."