# OmniSyn

This repo contains code for training and evaluating OmniSyn.<br>
OmniSyn predicts depth from two panoramas and then renders meshes using PyTorch3D from intermediate positions.

## Getting Started

1. Install PyTorch and PyTorch3d
2. Download our [processed carla dataset](https://drive.google.com/drive/folders/1UXzWGIpEPlLVhf9t0LJ0VPiAq7wYpudC?usp=sharing).

Currently, the main scripts are:
* `train_depth.py` which trains the depth estimator only.
* `train_inpainting.py` to train the inpainting component.

## Requirements

* Install v0.2.5 of PyTorch3d \
`pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.2.5'`