# OmniSyn

This repo contains code for training and evaluating OmniSyn.<br>
OmniSyn predicts depth from two panoramas and then renders meshes using PyTorch3D from intermediate positions.

## Getting Started

1. Setup a conda environment
    ```shell
    conda create -n omnisyn python=3.7
    conda activate omnisyn
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt
    conda install habitat-sim=0.1.6 headless -c conda-forge -c aihabitat
    ```
2. Download [our processed CARLA dataset](https://drive.google.com/drive/folders/1UXzWGIpEPlLVhf9t0LJ0VPiAq7wYpudC?usp=sharing).

3. Download [our CARLA model weights](https://drive.google.com/drive/folders/1p5XrgGdqdc3TSB41GcacYyTpnDF5NhIV?usp=sharing) to `example/` and run our selected example to check everything is working.
    ```shell
    python train_inpainting.py -c example/config_example.txt
    ```

Currently, the main scripts are:
* `train_depth.py` which trains the depth estimator only.
* `train_inpainting.py` to train the inpainting component.