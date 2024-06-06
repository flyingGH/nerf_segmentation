# Open-Set Semantic Segmentation using Gaussian Splatting
## Setup
```
git clone https://github.com/Luca-Wiehe/nerf_segmentation.git
cd nerf_segmentation
git submodule update --init
cd segment_3d_gaussians
conda env update --file environment.yml
git submodule update --init
```

Download [CLIP Model](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K/resolve/main/open_clip_pytorch_model.bin) and place it inside `nerf_segmentation/segment_3d_gaussians/clip_ckpt/` as `ViT-B-16-laion2b-s34b_b88k.bin`.

Download [SAM Checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place it inside `nerf_segmentation/segment_3d_gaussians/third_party/segment-anything/sam_ckpt/` as `sam_checkpoint.pth`.
