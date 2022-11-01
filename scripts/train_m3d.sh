#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "${DIR}/.." || exit 1

# Do training
# python train_depth.py -c configs/m3d_depth_config.txt \
#  --script-mode=train_depth_pose \
#  --carla-path=../run_carla_2020_07_24 \
#  --train-depth-steps=90600
# python train_depth.py -c configs/m3d_depth_config.txt \
#  --script-mode=eval_depth_test \
#  --carla-path=../run_carla_2020_07_24 \
#  --m3d-dist=1.0


# python train_inpainting.py -c configs/m3d_inpainting_config.txt \
#  --script-mode=train_inpainting \
#  --carla-path=../run_carla_2020_07_24 \
#  --train-inpainting-steps=45200
python train_inpainting.py -c configs/m3d_inpainting_config_withdepth.txt \
 --script-mode=eval_inpainting_testset \
 --carla-path=../run_carla_2020_07_24 \
 --use-pred-depth=true \
 --checkpoints-dir=runs/run_m3d_depth