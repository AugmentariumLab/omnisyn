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

## Acknowledgements
Some code in this repo is borrowed from [facebookresearch/synsin](https://github.com/facebookresearch/synsin) and [nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2).

## Reference
If you use this in your research, please reference it as:
```bibtex
@INPROCEEDINGS{Li2022Omnisyn,
  author={Li, David and Zhang, Yinda and H\"{a}ne, Christian and Tang, Danhang and Varshney, Amitabh and Du, Ruofei},
  booktitle={2022 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW)}, 
  title={{OmniSyn}: Synthesizing 360 Videos with Wide-baseline Panoramas}, 
  year={2022},
  volume={},
  number={},
  pages={670-671},
  doi={10.1109/VRW55335.2022.00186}}
```
