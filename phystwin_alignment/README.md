# Requirements
Check https://github.com/Jianghanxiao/PhysTwin

# Download the weights for superglue and put them inside the models folder
```bash
mkdir models/weights
cd models/weights
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_indoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_outdoor.pth
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superpoint_v1.pth4
cd ../..
```

# Run the code align.py
If anything is wrong check the original behaviour of align_original.py

The align.py script needs the following data:
- Rgb image for the frame used for Trellis
- extrinsics and intrinsics for the rgb image
- Segmentation mask for the rgb image
- pcd for the rgb image
- PCD for the full objects

The view generation is done using the cpu by default (but it is slow), --use_gpu_for_rendering=True to use the gpu