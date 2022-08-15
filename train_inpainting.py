# Lint as: python3
"""Train and eval the inpainting network.
"""

import os
import subprocess

import distro
import numpy as np
import torch
from progress.bar import Bar
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_readers.carla_reader import CarlaReader
from data_readers.gsv_reader import GSVReader
from data_readers.habitat_data import HabitatImageGenerator
from helpers import my_helpers
from helpers import my_torch_helpers
from helpers.torch_checkpoint_manager import CheckpointManager
from models import loss_lib
from models.inpainting_unet import InpaintNet
from models.metrics import WSPSNR
from models.pipeline3_model import FullPipeline
from models.pointcloud_model import PointcloudModel
from models.ssim import ssim
from options import parse_training_options
from renderer.SphereMeshGenerator import SphereMeshGenerator


class App:
  """Training app for the inpainting network."""

  def __init__(self):
    self.mesh_generator = SphereMeshGenerator()
    self.args = None
    self.full_width = 1024
    self.full_height = 1024

    # Model
    self.depth_model = None
    self.inpainting_model = None
    self.inpainting_depth_model = None

    # Training
    self.writer = None
    self.optimizer = None
    self.checkpoint_manager = None
    self.inpainting_checkpoint_manager = None

    # Attributes to hold training data.
    self.train_data = None
    self.train_data_loader = None

    # Attributes to hold validation data.
    self.val_data = None
    self.val_data_indices = None
    self.input_panos_val = None
    self.input_depths_val = None
    self.input_rots_val = None
    self.input_trans_val = None
    self.rng = np.random.default_rng()
    self.pointcloud_model = PointcloudModel()

    self.debug_cache = {}

  def parse_args(self):
    args = parse_training_options()
    args.device = my_torch_helpers.find_torch_device(args.device)
    self.args = args

    if args.train_pred_depth:
      print("Training end to end")

  def start(self):
    """Starts the training."""
    try:
      self.parse_args()
      args = self.args
      self.setup_depth_pose_model()
      self.setup_inpainting_model()
      self.setup_checkpoints()
      self.load_validation_data()

      if args.script_mode in ('train_inpainting',
                              'eval_inpainting',
                              'eval_inpainting_testset',
                              'train_monocular',):
        self.load_training_data()

      if args.script_mode in ('train_inpainting',):
        self.run_training_loop()
      elif args.script_mode == 'run_inpainting':
        self.run_inpainting()
      elif args.script_mode == 'run_inpainting_single':
        self.run_inpainting_single()
      elif args.script_mode in ('eval_inpainting',
                                'eval_inpainting_testset'):
        self.eval_inpainting()
      elif args.script_mode == 'train_monocular':
        self.train_monocular()
      elif args.script_mode == 'dump_examples':
        self.dump_examples()  
      elif args.script_mode == 'run_example':
        self.run_example()
      elif args.script_mode == 'visualize_point_cloud':
        self.visualize_point_cloud()
      else:
        raise ValueError("Unknown script mode" + str(args.script_mode))
    except KeyboardInterrupt:
      print("Ending")
      self.writer.close()

  def setup_depth_pose_model(self):
    args = self.args
    if args.checkpoints_dir == "":
      return

    model = FullPipeline(
      device=args.device,
      monodepth_model=args.model_name,
      width=args.width,
      height=args.height,
      layers=5,
      raster_resolution=args.width,
      point_radius=args.point_radius,
      depth_input_images=1,
      depth_output_channels=1,
      include_poseestimator=True,
      verbose=args.verbose,
      input_uv=args.depth_input_uv,
      interpolation_mode=args.interpolation_mode,
      cost_volume=args.cost_volume,
      use_v_input=args.model_use_v_input,
      depth_tye=args.depth_type).to(args.device)
    checkpoint_manager = CheckpointManager(
      args.checkpoints_dir,
      max_to_keep=args.checkpoint_count).freeze()

    latest_checkpoint = checkpoint_manager.load_latest_checkpoint()
    if latest_checkpoint is not None:
      model.load_state_dict(latest_checkpoint['model_state_dict'])
    elif args.use_pred_depth:
      print("Could not find depth network")
      exit(0)

    self.depth_model = model
    self.checkpoint_manager = checkpoint_manager

  def setup_inpainting_model(self):

    args = self.args
    input_channels = 8
    output_channels = 3 if not args.use_blending else 1
    if args.inpaint_depth:
      input_channels = input_channels + 2
      output_channels = output_channels + 1

    inpaint_model = InpaintNet(
      input_channels=input_channels,
      output_channels=output_channels,
      size=args.inpaint_model_size,
      layers=args.inpaint_model_layers,
      device=args.device,
      gate=False,
      use_residual=args.inpaint_use_residual,
      use_wrap_padding=args.inpaint_wrap_padding,
      use_one_conv=args.inpaint_one_conv,
      use_final_convblock=args.inpaint_final_convblock,
      model_version=args.inpaint_model_version).to(args.device)

    self.inpainting_model = inpaint_model
    self.perceptual_loss = loss_lib.VGGLoss()

  def setup_checkpoints(self):
    args = self.args

    inpainting_checkpoint_manager = CheckpointManager(
      args.inpaint_checkpoints_dir,
      max_to_keep=args.checkpoint_count)
    writer = SummaryWriter(
      log_dir=os.path.join(args.inpaint_checkpoints_dir, "logs"))
    if args.train_pred_depth:
      print("Training depth model")
      optimizer = torch.optim.Adam(list(self.inpainting_model.parameters()) +
                                   list(self.depth_model.parameters()),
                                   lr=args.learning_rate,
                                   betas=(args.opt_beta1, args.opt_beta2))
    else:
      optimizer = torch.optim.Adam(self.inpainting_model.parameters(),
                                   lr=args.learning_rate,
                                   betas=(args.opt_beta1, args.opt_beta2))
    latest_inpaint_checkpoint = \
      inpainting_checkpoint_manager.load_latest_checkpoint()
    if latest_inpaint_checkpoint is not None:
      self.inpainting_model.load_state_dict(
        latest_inpaint_checkpoint["model_state_dict"])
      optimizer.load_state_dict(
        latest_inpaint_checkpoint["optimizer_state_dict"])
      if "depth_state_dict" in latest_inpaint_checkpoint:
        self.depth_model.load_state_dict(
          latest_inpaint_checkpoint["depth_state_dict"])

    self.inpainting_checkpoint_manager = inpainting_checkpoint_manager
    self.writer = writer
    self.optimizer = optimizer

  def load_training_data(self):
    args = self.args

    seq_len = 3
    reference_idx = 1

    # Prepare dataset loaders for train and validation datasets.
    if args.dataset == "carla":
      train_data = CarlaReader(args.carla_path,
                               width=self.full_width,
                               height=self.full_height,
                               towns=["Town01", "Town02", "Town03", "Town04"],
                               min_dist=args.carla_min_dist,
                               max_dist=args.carla_max_dist,
                               seq_len=seq_len,
                               reference_idx=reference_idx,
                               use_meters_depth=True,
                               interpolation_mode=args.interpolation_mode,
                               sampling_method="dense")
      train_dataloader = DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=4, )
    elif args.dataset == "m3d":
      train_data = HabitatImageGenerator(
        "train",
        full_width=self.full_width,
        full_height=self.full_height,
        seq_len=seq_len,
        reference_idx=reference_idx,
        m3d_dist=args.m3d_dist
      )
      train_dataloader = DataLoader(
        dataset=train_data,
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
      )
    elif args.dataset == "gsv":
      train_data = None
      train_dataloader = None
    else:
      raise ValueError("Unknown dataset: %s" % args.dataset)

    self.train_data = train_data
    self.train_data_loader = train_dataloader

  def load_validation_data(self):
    """Loads validation data."""
    args = self.args

    seq_len = 3
    reference_idx = 1

    if args.script_mode in ['eval_inpainting_testset']:
      towns = ['Town06']
    else:
      towns = ['Town05']

    if args.dataset == "carla":
      val_data = CarlaReader(
        args.carla_path,
        width=self.full_width,
        height=self.full_height,
        towns=towns,
        min_dist=args.carla_min_dist,
        max_dist=args.carla_max_dist,
        seq_len=seq_len,
        reference_idx=reference_idx,
        use_meters_depth=True,
        interpolation_mode=args.interpolation_mode)
    elif args.dataset == "m3d":
      split = 'val'
      if args.script_mode in ['eval_inpainting_testset', 'dump_examples', 'visualize_point_cloud']:
        split = 'test'
      val_data = HabitatImageGenerator(
        split,
        full_width=self.full_width,
        full_height=self.full_height,
        seq_len=seq_len,
        reference_idx=reference_idx,
        m3d_dist=args.m3d_dist
      )
    elif args.dataset == "gsv":
      return
    else:
      raise ValueError("Unknown dataset: %d" % args.dataset)

    if args.dataset != "carla" or (args.dataset == "carla" and args.carla_path):
      # Load a single batch of validation data.
      # val_data_indices = [20, 40, 60, 80, 100]
      val_data_indices = [20, 40, 60, 80]
      if args.script_mode == 'train_monocular':
        val_data_indices = [20, 40, 100, 200]
      val_data_all = tuple(val_data[i] for i in val_data_indices)
      input_panos_val = np.stack(tuple(
        val_data["rgb_panos"] for val_data in val_data_all),
        axis=0)
      input_panos_val = torch.tensor(input_panos_val,
                                    dtype=torch.float32,
                                    device=args.device)
      input_rots_val = np.stack(
        tuple(val_data["rots"] for val_data in val_data_all),
        axis=0)
      input_rots_val = torch.tensor(input_rots_val,
                                    dtype=torch.float32,
                                    device=args.device)
      input_trans_val = np.stack(tuple(
        val_data["trans"] for val_data in val_data_all),
        axis=0)
      input_trans_val = torch.tensor(input_trans_val,
                                    dtype=torch.float32,
                                    device=args.device)
      input_depths_val = np.stack(tuple(
        val_data["depth_panos"] for val_data in val_data_all),
        axis=0)
      input_depths_val = torch.tensor(input_depths_val,
                                      dtype=torch.float32,
                                      device=args.device)

      self.val_data = val_data
      self.val_data_indices = val_data_indices
      self.input_panos_val = input_panos_val
      self.input_depths_val = input_depths_val
      self.input_rots_val = input_rots_val
      self.input_trans_val = input_trans_val

  def run_inpainting_mesh_carla(self, step, panos, rots, trans, depths):
    """Does a single run.

    Args:
      step: Current step.
      panos: Input panos at full resolution.
      rots: Input rotations.
      trans: Input translations.
      depths: input depths.

    Returns:
      run_outputs: A dictionary containing outputs of the run.
      run_outputs["panos_small"]: Resized input panos.
      run_outputs["l1_loss"]: Loss.


    """
    args = self.args
    mesh_generator = self.mesh_generator
    depth_model = self.depth_model
    inpainting_model = self.inpainting_model
    width = args.width
    height = args.height
    inpaint_resolution = args.inpaint_resolution if args.inpaint_resolution > 0 else width

    batch_size, seq_len, panos_height, panos_width = panos.shape[:4]
    downsample_size = int(64 + (width - 64) * np.clip(step / 2000, 0, 1))

    panos_small = panos.reshape(
      (batch_size * seq_len, panos_height, panos_width, 3))
    panos_small = my_torch_helpers.resize_torch_images(
      panos_small, (args.width, args.height), mode=args.interpolation_mode)
    panos_small = panos_small.reshape(batch_size, seq_len, height, width, 3)

    panos_small2 = panos.reshape(
      (batch_size * seq_len, panos_height, panos_width, 3))
    panos_small2 = my_torch_helpers.resize_torch_images(
      panos_small2, (inpaint_resolution, inpaint_resolution), mode=args.interpolation_mode)
    panos_small2 = panos_small2.reshape(batch_size, seq_len, inpaint_resolution, inpaint_resolution, 3)

    rotated_panos_small = []
    for i in range(panos.shape[1]):
      rotated_pano = my_torch_helpers.rotate_equirectangular_image(
        panos[:, i],
        rots[:, i])
      rotated_panos_small.append(
        my_torch_helpers.resize_torch_images(
          rotated_pano,
          (args.width, args.height),
          mode=args.interpolation_mode)
      )
    rotated_panos_small = torch.stack(rotated_panos_small, dim=1)

    rotated_depths_small = []
    for i in range(panos.shape[1]):
      rotated_pano = my_torch_helpers.rotate_equirectangular_image(
        depths[:, i, :, :, None],
        rots[:, i])
      rotated_depths_small.append(
        my_torch_helpers.resize_torch_images(
          rotated_pano,
          (args.width, args.height),
          mode=args.interpolation_mode)
      )
    rotated_depths_small = torch.stack(rotated_depths_small, dim=1)

    if args.use_pred_depth:
      # print('Using predicted depth')
      if args.cost_volume:
        outputs0 = depth_model.estimate_depth_using_cost_volume(
          panos_small[:, [2, 0], :, :, :],
          rots[:, [2, 0]],
          trans[:, [2, 0]],
          min_depth=args.min_depth,
          max_depth=args.max_depth)
        outputs2 = depth_model.estimate_depth_using_cost_volume(
          panos_small[:, [0, 2], :, :, :],
          rots[:, [0, 2]],
          trans[:, [0, 2]],
          min_depth=args.min_depth,
          max_depth=args.max_depth)
        depths_small = torch.stack(
          (outputs0["depth"], outputs2["depth"], outputs2["depth"]),
          dim=1)
      else:
        depths_pred_0 = depth_model.estimate_depth(panos_small[:, 0, :, :, :])
        depths_pred_2 = depth_model.estimate_depth(panos_small[:, 2, :, :, :])
        depths_small = torch.stack(
          (depths_pred_0, depths_pred_0, depths_pred_2),
          dim=1)
    else:
      depths_height, depths_width = depths.shape[2:4]
      depths_small = depths.reshape(
        (batch_size * seq_len, depths_height, depths_width, 1))
      depths_small = my_torch_helpers.resize_torch_images(
        depths_small, (args.width, args.height), mode=args.interpolation_mode)
      depths_small = depths_small.reshape(batch_size, seq_len, height, width, 1)

    if args.add_depth_noise:
      depths_small = depths_small + args.add_depth_noise * (
          2 * torch.rand(depths.shape, device=args.device,
                         dtype=depths.dtype) - 1)

    thresholded_gt_depth = depths_small
    if args.threshold_depth and args.clamp_depth_to > 65:
      thresholded_gt_depth = depths_small.clone()
      thresholded_gt_depth[thresholded_gt_depth > 65] = args.clamp_depth_to

    if args.representation == 'mesh':
      meshes_0 = mesh_generator.generate_mesh(
        thresholded_gt_depth[:, 0],
        panos[:, 0],
        apply_depth_mask=True,
        facing_inside=True,
        width_segments=args.mesh_width,
        height_segments=args.mesh_height,
        mesh_removal_threshold=args.mesh_removal_threshold)
      meshes_2 = mesh_generator.generate_mesh(
        thresholded_gt_depth[:, 2],
        panos[:, 2],
        apply_depth_mask=True,
        facing_inside=True,
        width_segments=args.mesh_width,
        height_segments=args.mesh_height,
        mesh_removal_threshold=args.mesh_removal_threshold)

      meshes_0 = mesh_generator.apply_rot_trans(meshes_0,
                                                rots[:, 0],
                                                trans[:, 0],
                                                inv_rot=True)
      meshes_2 = mesh_generator.apply_rot_trans(meshes_2,
                                                rots[:, 2],
                                                trans[:, 2],
                                                inv_rot=True)

      output_0, depth_image_0 = mesh_generator.render_mesh(meshes_0, image_size=inpaint_resolution)
      output_2, depth_image_2 = mesh_generator.render_mesh(meshes_2, image_size=inpaint_resolution)
    elif args.representation == 'pointcloud':
      pointcloud0 = self.pointcloud_model.make_point_cloud(
        depths=thresholded_gt_depth[:, 0],
        images=panos_small[:, 0],
        rots=rots[:, 0],
        trans=trans[:, 0],
        inv_rot_trans=True)
      pointcloud1 = self.pointcloud_model.make_point_cloud(
        depths=thresholded_gt_depth[:, 2],
        images=panos_small[:, 2],
        rots=rots[:, 2],
        trans=trans[:, 2],
        inv_rot_trans=True)

      output_0, depth_image_0 = self.pointcloud_model.render_point_cloud(
        pointcloud0, size=height, radius=args.point_radius)
      output_2, depth_image_2 = self.pointcloud_model.render_point_cloud(
        pointcloud1, size=height, radius=args.point_radius)
      depth_image_0 = depth_image_0[..., :1]
      depth_image_2 = depth_image_2[..., :1]

    if args.use_depth_mask:
      depth_mask_0 = torch.lt(depth_image_0, -0.01).float()
      depth_mask_2 = torch.lt(depth_image_2, -0.01).float()
    else:
      depth_mask_0 = depth_image_0 / 100.0
      depth_mask_2 = depth_image_2 / 100.0

    if args.inpaint_depth:
      inpaint_depth_scale = args.inpaint_depth_scale
      clamped_depth_image_0 = torch.clamp(depth_image_0, 1.0, 100.0)
      clamped_depth_image_2 = torch.clamp(depth_image_2, 1.0, 100.0)
      concatenated_output = torch.cat(
        (output_0, depth_mask_0,
         output_2, depth_mask_2,
         torch.log(clamped_depth_image_0) * inpaint_depth_scale,
         torch.log(clamped_depth_image_2) * inpaint_depth_scale), dim=3)
    else:
      concatenated_output = torch.cat(
        (output_0, depth_mask_0, output_2, depth_mask_2), dim=3)
    inpainted_image = inpainting_model(concatenated_output.detach())
    if isinstance(inpainted_image, dict):
      weight_map = None if "weight_map" not in inpainted_image else inpainted_image["weight_map"]
      inpainted_image = inpainted_image["output"]
    inpainted_depth = None
    if args.inpaint_depth:
      inpainted_depth = (1.0 / inpaint_depth_scale) * torch.exp(
        inpainted_image[..., -1:])
      inpainted_image = inpainted_image[..., :-1]

    if args.use_blending:
      inpainted_image = (1 - inpainted_image) * output_0 + \
                        inpainted_image * output_2

    l1_loss = 0.0
    perceptual_loss = 0.0
    if args.loss == "l1":
      l1_loss = loss_lib.compute_l1_loss(inpainted_image, panos_small2[:, 1])
    elif args.loss == "l2":
      l1_loss = loss_lib.compute_l2_loss(inpainted_image, panos_small2[:, 1])
    elif args.loss == "downsampled":
      l1_loss = loss_lib.compute_downsampled_loss(inpainted_image,
                                                  panos_small2[:, 1],
                                                  size=(downsample_size,
                                                        downsample_size))
    elif args.loss == "min_patch":
      l1_loss = loss_lib.compute_min_patch_loss(
        inpainted_image,
        panos_small2[:, 1],
        patch_size=args.patch_loss_patch_size,
        stride=args.patch_loss_stride,
        stride_dist=args.patch_loss_stride_dist)
    elif args.loss == "l1+perceptual":
      l1_loss = loss_lib.compute_l1_loss(inpainted_image, panos_small2[:, 1])
      perceptual_loss = self.perceptual_loss(
        inpainted_image.permute(0, 3, 1, 2),
        panos_small2[:, 1].permute(0, 3, 1, 2))
    else:
      raise Exception('Invalid Loss')

    inpaint_depth_loss = torch.tensor(0, dtype=torch.float32,
                                      device=args.device)
    final_loss = l1_loss + perceptual_loss
    if args.inpaint_depth:
      inpaint_depth_loss = loss_lib.l1_depth_ignore_sky(
        inpainted_depth,
        depths_small[:, 1],
        threshold=65)
      final_loss = final_loss + args.inpaint_depth_factor * inpaint_depth_loss

    return {
      "final_loss": final_loss,
      "l1_loss": l1_loss,
      "perceptual_loss": perceptual_loss,
      "inpaint_depth_loss": inpaint_depth_loss,
      "panos_small": panos_small,
      "panos_small2": panos_small2,
      "depths_small": depths_small,
      "output_0": output_0,
      "output_2": output_2,
      "depth_image_0": depth_image_0,
      "depth_image_2": depth_image_2,
      "inpainted_image": inpainted_image,
      "inpainted_depth": inpainted_depth
    }

  def log_training_to_tensorboard(self, step, run_outputs):
    args = self.args
    writer = self.writer

    l1_loss = run_outputs["l1_loss"]
    self.writer.add_scalar('train_l1_loss', l1_loss.item(), step)
    final_loss = run_outputs["final_loss"]
    self.writer.add_scalar('train_final_loss', final_loss.item(), step)

    if step == 1 or \
        args.train_tensorboard_interval == 0 or \
        step % args.train_tensorboard_interval == 0:
      panos_small = run_outputs["panos_small"]
      depths_small = run_outputs["depths_small"]
      depth_image_0 = run_outputs["depth_image_0"]
      depth_image_2 = run_outputs["depth_image_2"]
      inpainted_image = run_outputs["inpainted_image"]
      output_0 = run_outputs["output_0"]
      output_2 = run_outputs["output_2"]

      middle_pano_index = 1 if "target_pano_index" not in run_outputs else run_outputs["target_pano_index"]
      reference_imgs = torch.cat(
        (panos_small[:, 0], panos_small[:, middle_pano_index], panos_small[:, -1]), dim=2)
      output_cat = torch.cat((output_0, output_2), dim=2)
      depth_image_turbo_0 = my_torch_helpers.depth_to_turbo_colormap(
        depth_image_0[:, :, :, 0:1], min_depth=args.turbo_cmap_min)
      depth_image_turbo_2 = my_torch_helpers.depth_to_turbo_colormap(
        depth_image_2[:, :, :, 0:1], min_depth=args.turbo_cmap_min)
      depth_image_turbo_cat = torch.cat(
        (depth_image_turbo_0, depth_image_turbo_2), dim=2)

      writer.add_images("00_reference_imgs",
                        reference_imgs.clamp(0, 1),
                        step,
                        dataformats="NHWC")
      writer.add_images("05_rendered_mesh",
                        output_cat.clamp(0, 1),
                        step,
                        dataformats="NHWC")
      writer.add_images("06_rendered_depth",
                        depth_image_turbo_cat.clamp(0, 1),
                        step,
                        dataformats="NHWC")
      writer.add_images("10_inpainted_render",
                        inpainted_image.clamp(0, 1),
                        step,
                        dataformats="NHWC")
      if inpainted_image.shape[1] == panos_small.shape[2]:
        delta_img = torch.abs(inpainted_image - panos_small[:, 1, :, :, :])
        writer.add_images("15_inpainted_loss",
                          delta_img.clamp(0, 1),
                          step,
                          dataformats='NHWC')

      if args.inpaint_depth:
        inpainted_depth = run_outputs['inpainted_depth']
        inpainted_depth_t = my_torch_helpers.depth_to_turbo_colormap(
          inpainted_depth, min_depth=args.turbo_cmap_min)
        gt_depth_1_t = my_torch_helpers.depth_to_turbo_colormap(
          depths_small[:, 1], min_depth=args.turbo_cmap_min)
        writer.add_images('20_inpainted_depth',
                          inpainted_depth_t.clamp(0, 1),
                          step,
                          dataformats='NHWC')
        writer.add_images('21_gt_depth',
                          gt_depth_1_t.clamp(0, 1),
                          step,
                          dataformats='NHWC')

      if "weight_map" in run_outputs and run_outputs["weight_map"] is not None:
        writer.add_images("07_weight_map",
                          run_outputs["weight_map"].clamp(0, 1).expand(-1, -1, -1, 3),
                          step,
                          dataformats="NHWC")

  def do_validation_run(self, step):
    args = self.args

    if step == 1 or \
        args.validation_interval == 0 or \
        step % args.validation_interval == 0:
      # Calculate validation loss
      with torch.no_grad():

        panos = self.input_panos_val
        rots = self.input_rots_val
        trans = self.input_trans_val
        depths = self.input_depths_val

        run_outputs = self.run_inpainting_mesh_carla(step, panos, rots, trans,
                                                     depths)

        panos_small = run_outputs['panos_small2']
        output_0 = run_outputs['output_0']
        output_2 = run_outputs['output_2']
        inpainted_image = run_outputs['inpainted_image']
        depths_small = run_outputs['depths_small']

        self.writer.add_scalar('validation_l1_loss',
                               run_outputs['l1_loss'].item(), step)
        self.writer.add_scalar('validation_final_loss',
                               run_outputs['final_loss'].item(), step)
        self.writer.add_scalar('validation_inpaint_depth_loss',
                               run_outputs['inpaint_depth_loss'].item(), step)

        middle_pano_index = 1 if "target_pano_index" not in run_outputs else run_outputs["target_pano_index"]
        depths_turbo = my_torch_helpers.depth_to_turbo_colormap(
          depths_small[:, 0, :, :, :], min_depth=args.turbo_cmap_min)
        y_stacked_a = torch.cat(
          (panos_small[:, 0], panos_small[:, middle_pano_index], panos_small[:, -1]), dim=2)
        y_stacked_b = torch.cat((output_0, output_2, inpainted_image), dim=2)
        y_stacked = torch.cat((y_stacked_a, y_stacked_b), dim=1)
        for j in range(len(self.val_data_indices)):
          self.writer.add_image('80_val_image_%02d' % j,
                                y_stacked[j].clamp(0, 1),
                                step,
                                dataformats='HWC')
        if args.inpaint_depth:
          inpainted_depth = run_outputs['inpainted_depth']
          inpainted_depth_t = my_torch_helpers.depth_to_turbo_colormap(
            inpainted_depth, min_depth=args.turbo_cmap_min
          )
          for j in range(len(self.val_data_indices)):
            self.writer.add_image('85_val_inpainted_depth_%02d' % j,
                                  inpainted_depth_t[j].clamp(0, 1),
                                  step,
                                  dataformats='HWC')

  def save_checkpoint(self, step):
    args = self.args

    if args.checkpoint_interval == 0 or step % args.checkpoint_interval == 0:
      # Save a checkpoint
      if args.train_pred_depth:
        self.inpainting_checkpoint_manager.save_checkpoint({
          'depth_state_dict': self.depth_model.state_dict(),
          'model_state_dict': self.inpainting_model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict()
        })
      else:
        self.inpainting_checkpoint_manager.save_checkpoint({
          'model_state_dict': self.inpainting_model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict()
        })
      self.writer.flush()

  def run_training_loop(self):
    args = self.args

    for epoch in range(args.epochs):
      for i, data in enumerate(self.train_data_loader):
        self.optimizer.zero_grad()
        step = self.inpainting_checkpoint_manager.increment_step()

        if args.debug_mode and args.debug_one_batch:
          assert distro.linux_distribution()[0] == 'Ubuntu', 'Debug mode is on'  # For debugging only
          if "batch_data" not in self.debug_cache:
            panos = data['rgb_panos'].to(args.device)
            rots = data['rots'].to(args.device)
            trans = data['trans'].to(args.device)
            depths = data['depth_panos'].to(args.device)
            self.debug_cache["batch_data"] = (panos, rots, trans, depths)
          panos, rots, trans, depths = self.debug_cache["batch_data"]
        else:
          panos = data['rgb_panos'].to(args.device)
          rots = data['rots'].to(args.device)
          trans = data['trans'].to(args.device)
          depths = data['depth_panos'].to(args.device)

        run_outputs = self.run_inpainting_mesh_carla(
          step, panos, rots, trans, depths)
        self.log_training_to_tensorboard(step, run_outputs)

        final_loss = run_outputs['final_loss']
        final_loss.backward()
        self.optimizer.step()
        self.do_validation_run(step)
        self.save_checkpoint(step)
        print('Step: %d [%d:%d] Loss: %f' % (step, epoch, i, final_loss.item()))

  def run_inpainting(self):
    """Run the inpainting network to generate GIFs.

    Returns:

    """
    args = self.args
    train_dataloader = self.train_data_loader
    depth_model = self.depth_model
    inpainting_model = self.inpainting_model
    width = args.width
    height = args.height
    inpaint_resolution = args.inpaint_resolution if args.inpaint_resolution > 0 else width
    device = args.device

    depth_model.eval()
    inpainting_model.eval()

    mesh_generator = SphereMeshGenerator()
    frame_number = 0
    start_slice_idx = 300
    step = 100000

    runs_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir, 'runs')
    runs_temp_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir, 'runs_temp')
    input0_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir, 'input0')
    input1_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir, 'input1')
    output0_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir, 'output0')
    output1_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir, 'output1')

    val_data_indices = [75]
    input_panos_val = np.stack(tuple(
      self.val_data[i]['rgb_panos'] for i in val_data_indices),
      axis=0)
    input_panos_val = torch.tensor(input_panos_val,
                                   dtype=torch.float32,
                                   device=args.device)
    input_rots_val = np.stack(
      tuple(self.val_data[i]['rots'] for i in val_data_indices),
      axis=0)
    input_rots_val = torch.tensor(input_rots_val,
                                  dtype=torch.float32,
                                  device=args.device)
    input_trans_val = np.stack(tuple(
      self.val_data[i]['trans'] for i in val_data_indices),
      axis=0)
    input_trans_val = torch.tensor(input_trans_val,
                                   dtype=torch.float32,
                                   device=args.device)
    input_depths_val = np.stack(tuple(
      self.val_data[i]['depth_panos'] for i in val_data_indices),
      axis=0)
    input_depths_val = torch.tensor(input_depths_val,
                                    dtype=torch.float32,
                                    device=args.device)
    val_data = CarlaReader(
      args.carla_path,
      width=self.full_width,
      height=self.full_height,
      towns=['Town01'],
      min_dist=args.carla_min_dist,
      max_dist=args.carla_max_dist,
      seq_len=2,
      reference_idx=1,
      use_meters_depth=True,
      interpolation_mode=args.interpolation_mode)

    validation_dataloader = DataLoader(val_data,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=0 if args.dataset == 'm3d' else 4)
    for i, data in enumerate(validation_dataloader):

      # Do a run through training data.
      panos = data['rgb_panos'].to(device)
      rots = data['rots'].to(device)
      trans = data['trans'].to(device)
      depths = data['depth_panos'].to(device)

      if os.path.isfile(os.path.join(runs_dir, "%d.gif" % i)):
        continue

      # Do a run through GSV data.
      # panos = input_panos_gsv

      # Do a run through validation data.
      # panos = input_panos_val
      # rots = input_rots_val
      # trans = input_trans_val
      # depths = input_depths_val

      batch_size, seq_len, panos_height, panos_width = panos.shape[:4]
      downsample_size = int(64 + (width - 64) * np.clip(step / 2000, 0, 1))

      panos_small = panos.reshape(
        (batch_size * seq_len, panos_height, panos_width, 3))
      panos_small = my_torch_helpers.resize_torch_images(panos_small,
                                                         (args.width,
                                                          args.height))
      panos_small = panos_small.reshape(batch_size, seq_len, height, width, 3)

      panos_small2 = panos.reshape(
        (batch_size * seq_len, panos_height, panos_width, 3))
      panos_small2 = my_torch_helpers.resize_torch_images(
        panos_small2, (inpaint_resolution, inpaint_resolution), mode=args.interpolation_mode)
      panos_small2 = panos_small2.reshape(batch_size, seq_len, inpaint_resolution, inpaint_resolution, 3)

      depths_height, depths_width = depths.shape[2:4]
      depths_small = depths.reshape(
        (batch_size * seq_len, depths_height, depths_width, 1))
      depths_small = my_torch_helpers.resize_torch_images(
        depths_small, (args.width, args.height))
      depths_small = depths_small.reshape(batch_size, seq_len, height, width, 1)

      rot_pred_1 = torch.stack((rots[:, 0], rots[:, 1]), dim=1)
      trans_pred_1 = torch.stack((trans[:, 0], trans[:, 1]), dim=1)

      if args.threshold_depth and args.clamp_depth_to > 65:
        depths[depths > 65] = args.clamp_depth_to

      print('Predicting depth')

      if args.cost_volume:
        outputs_1 = depth_model.estimate_depth_using_cost_volume(
          panos_small[:, [1, 0], :, :, :],
          rots[:, [1, 0]],
          trans[:, [1, 0]],
          min_depth=args.min_depth,
          max_depth=args.max_depth
        )
        depths_pred_1 = outputs_1['depth']
        depths_pred_2 = depth_model.estimate_depth_using_cost_volume(
          panos_small[:, [0, 1], :, :, :],
          rots[:, [0, 1]],
          trans[:, [0, 1]],
          min_depth=args.min_depth,
          max_depth=args.max_depth)['depth']
        depths_pred = torch.stack((depths_pred_1, depths_pred_2), dim=1)
        depths_pred = torch.clamp_min(depths_pred, 0.0)
      else:
        depths_pred_0 = depth_model.estimate_depth(panos_small[:, 0, :, :, :])[
          'depth']
        depths_pred_1 = depth_model.estimate_depth(panos_small[:, 1, :, :, :])[
          'depth']
        depths_pred = torch.stack((depths_pred_0, depths_pred_1), dim=1)

      print('Generating mesh')

      meshes0 = mesh_generator.generate_mesh(
        depths_pred[0:1, 0],
        panos_small[0:1, 0],
        apply_depth_mask=True,
        facing_inside=True,
        width_segments=args.mesh_width,
        height_segments=args.mesh_height,
        mesh_removal_threshold=args.mesh_removal_threshold)
      meshes1 = mesh_generator.generate_mesh(
        depths_pred[0:1, 1],
        panos_small[0:1, 1],
        apply_depth_mask=True,
        facing_inside=True,
        width_segments=args.mesh_width,
        height_segments=args.mesh_height,
        mesh_removal_threshold=args.mesh_removal_threshold)

      meshes1 = mesh_generator.apply_rot_trans(meshes1,
                                               rot_pred_1[0:1, 0],
                                               -trans_pred_1[0:1, 0])

      for j in range(3):
        output_gt_path = os.path.join(args.inpaint_checkpoints_dir,
                                      'erp_%d.png' % j)
        # my_torch_helpers.save_torch_image(output_gt_path, panos_small[:, j])
      output_gt_d_path = os.path.join(args.inpaint_checkpoints_dir,
                                      'erp_depth_1_pred.png')
      depths_turbo = my_torch_helpers.depth_to_turbo_colormap(depths_pred[:, 0],
                                                              min_depth=args.turbo_cmap_min)
      # my_torch_helpers.save_torch_image(output_gt_d_path, depths_turbo)
      output_gt_d_path = os.path.join(args.inpaint_checkpoints_dir,
                                      'erp_depth_2_pred.png')
      depths_turbo = my_torch_helpers.depth_to_turbo_colormap(depths_pred[:, 1],
                                                              min_depth=args.turbo_cmap_min)
      # my_torch_helpers.save_torch_image(output_gt_d_path, depths_turbo)

      my_torch_helpers.save_torch_image(
        os.path.join(input0_dir, "%d.png" % i),
        my_torch_helpers.rotate_equirectangular_image(
          panos[:, 0], rot_mat=rots[:, 0], linearize_angle=0
        ))
      my_torch_helpers.save_torch_image(
        os.path.join(input1_dir, "%d.png" % i),
        panos[:, 1])

      num_frames = 60
      frame_number = frame_number + 1
      for k in tqdm(range(1, num_frames - 1), leave=True, desc="Frame"):
        # Ignore t=0 and t=1
        t = k / (num_frames - 1)
        meshes_shifted_0 = mesh_generator.apply_rot_trans(meshes0,
                                                          rots[0:1, 0],
                                                          t * (trans_pred_1[0:1, 0]),
                                                          inv_rot=True)
        meshes_shifted_1 = mesh_generator.apply_rot_trans(meshes1,
                                                          rots[0:1, 0],
                                                          t * (trans_pred_1[0:1, 0]),
                                                          inv_rot=True)

        output_1, depth_image_1 = mesh_generator.render_mesh(meshes_shifted_0,
                                                             image_size=inpaint_resolution)
        output_2, depth_image_2 = mesh_generator.render_mesh(meshes_shifted_1,
                                                             image_size=inpaint_resolution)

        if args.use_depth_mask:
          depth_mask_1 = torch.lt(depth_image_1, -0.01).type(torch.float32)
          depth_mask_2 = torch.lt(depth_image_2, -0.01).type(torch.float32)
        else:
          depth_mask_1 = depth_image_1 / 100.0
          depth_mask_2 = depth_image_2 / 100.0

        if args.inpaint_depth:
          clamped_depth_image_0 = torch.clamp(depth_image_1, 1.0, 100.0)
          clamped_depth_image_2 = torch.clamp(depth_image_2, 1.0, 100.0)
          concatenated_output = torch.cat(
            (output_1, depth_mask_1,
             output_2, depth_mask_2,
             torch.log(clamped_depth_image_0) / 2,
             torch.log(clamped_depth_image_2) / 2), dim=3)
        else:
          concatenated_output = torch.cat(
            (output_1, depth_mask_1, output_2, depth_mask_2), dim=3)
        inpainted_image = inpainting_model(concatenated_output.detach())
        if isinstance(inpainted_image, dict):
          weight_map = None if "weight_map" not in inpainted_image else inpainted_image["weight_map"]
          inpainted_image = inpainted_image["output"]
        inpainted_depth = None
        if args.inpaint_depth:
          inpainted_depth = torch.exp(inpainted_image[..., -1:])
          inpainted_image = inpainted_image[..., :-1]

        output_path = os.path.join(runs_temp_dir, "%d.png" % frame_number)
        my_torch_helpers.save_torch_image(output_path, inpainted_image)
        # my_torch_helpers.save_torch_image(
        #   os.path.join(output0_dir, "%d.png" % frame_number), output_1)
        # my_torch_helpers.save_torch_image(
        #   os.path.join(output1_dir, "%d.png" % frame_number), output_2)

        frame_number = frame_number + 1
      frame_number = 0
      make_gif_args = [
        "ffmpeg",
        "-i", os.path.join(runs_temp_dir, "%d.png"),
        os.path.join(runs_dir, "%d.gif" % i),
        "-y"
      ]
      subprocess.run(make_gif_args)

  def run_inpainting_single(self):
    """Run the inpainting network to generate GIFs.

    Returns:

    """
    args = self.args
    depth_model = self.depth_model
    inpainting_model = self.inpainting_model
    width = args.width
    height = args.height

    depth_model.eval()
    inpainting_model.eval()

    frame_number = 0
    start_slice_idx = 300
    step = 100000

    if args.dataset == "carla":

      run_starting_points = [
        {
          'town': 'Town06',
          'run_id': '14',
          'start_frame': 13466,
          'distance': 1
        },
        {
          'town': 'Town05',
          'run_id': '20',
          'start_frame': 191444,
          'distance': 1
        },
        {
          'town': 'Town06',
          'run_id': '39',
          'start_frame': 13333,
          'distance': 1
        },
        {
          'town': 'Town05',
          'run_id': '26',
          'start_frame': 197654,
          'distance': 1
        },
        {
          'town': 'Town01',
          'run_id': '26',
          'start_frame': 32754,
          'distance': 1
        },
        {
          'town': 'Town05',
          'run_id': '10',
          'start_frame': 181700,
          'distance': 1
        },
        {
          'town': 'Town06',
          'run_id': '32',
          'start_frame': 6105,
          'distance': 1
        },
        {
          'town': 'Town06',
          'run_id': '37',
          'start_frame': 11499,
          'distance': 1
        },
        {
          'town': 'Town03',
          'run_id': '2',
          'start_frame': 90587,
          'distance': 1
        },
      ]

      cubemap_side = 4

      dataset = CarlaReader(
        args.carla_path,
        width=self.full_width,
        height=self.full_height,
        towns=[],
        min_dist=args.carla_min_dist,
        max_dist=args.carla_max_dist,
        seq_len=6,
        reference_idx=5,
        use_meters_depth=True,
        interpolation_mode=args.interpolation_mode,
        sampling_method="custom",
        custom_params=run_starting_points[5],
        return_path=True)

      # Do a run through validation data.
      instance = dataset[0]
      panos = torch.tensor(instance['rgb_panos'], device=args.device,
                           dtype=torch.float32)[None]
      rotations = torch.tensor(instance['rots'], device=args.device,
                               dtype=torch.float32)[None]
      translations = torch.tensor(instance['trans'], device=args.device,
                                  dtype=torch.float32)[None]
      depths = torch.tensor(instance['depth_panos'], device=args.device,
                            dtype=torch.float32)[None]
      print("dataset", panos.shape, rotations[0, 5])

      assert args.use_pred_depth, "Not using predicted depth"
      output_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir,
                                            'single_path')
      for i in range(6):
        rotated_torch_image = my_torch_helpers.rotate_equirectangular_image(
          panos[:, i], rotations[:, i]
        )
        my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "gt_%d.png" % i),
          my_torch_helpers.resize_torch_images(rotated_torch_image,
                                               size=(256, 256)))
        gt_cb = my_torch_helpers.equirectangular_to_cubemap(
          rotated_torch_image, side=cubemap_side, flipx=True
        )
        my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "gt_cb_%d.png" % i),
          gt_cb)
        if i in [0, 5]:
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "%d.png" % i),
            my_torch_helpers.resize_torch_images(rotated_torch_image,
                                                 size=(256, 256)))

      for i in range(1, 5):
        indices = [0, i, 5]
        rotations, translations = dataset.calculate_rot_trans(
          instance['path'][np.newaxis], reference_idx=i)
        rotations = torch.tensor(rotations, device=args.device,
                                 dtype=torch.float32)
        translations = torch.tensor(translations, device=args.device,
                                    dtype=torch.float32)
        run_outputs = self.run_inpainting_mesh_carla(
          100000,
          panos[:, indices],
          rotations[:, indices],
          translations[:, indices],
          depths[:, indices]
        )
        inpainted_image = run_outputs['inpainted_image']
        my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "pred_%d.png" % i),
          inpainted_image)
        my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "%d.png" % i),
          inpainted_image)

        for cubemap_side in range(6):
          inpainted_image_cb = my_torch_helpers.equirectangular_to_cubemap(
            inpainted_image, side=cubemap_side, flipx=True
          )
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "pred_c%d_%d.png" % (cubemap_side, i)),
            inpainted_image_cb)
        my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "output_0_%d.png" % i),
          run_outputs['output_0'])
        my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "output_2_%d.png" % i),
          run_outputs['output_2'])
        my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "pred_depth_0_%d.png" % i),
          my_torch_helpers.depth_to_turbo_colormap(
            run_outputs['depths_small'][:, 0],
            min_depth=args.turbo_cmap_min
          ))

        my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "gt_depth_0_%d.png" % i),
          my_torch_helpers.depth_to_turbo_colormap(
            depths[:, 0],
            min_depth=args.turbo_cmap_min
          ))
    elif args.dataset == 'gsv':
      dataset = GSVReader(
        gsv_path=args.gsv_path,
        width=256,
        height=256,
        data_type="train",
        seq_len=2,
        reference_idx=-1
      )

      # Do a run through validation data.
      for val_idx in [1231]:
        instance = dataset[val_idx]
        panos = torch.tensor(instance['rgb_panos'], device=args.device,
                             dtype=torch.float32)[None]
        rotations = torch.tensor(instance['rots'], device=args.device,
                                 dtype=torch.float32)[None]
        translations = torch.tensor(instance['trans'], device=args.device,
                                    dtype=torch.float32)[None]
        depths = panos[:, :, :, 0]
        print("panos", panos.shape, rotations.shape, translations.shape)
        print("Translation norm",
              torch.norm(translations[:, 1] - translations[:, 0]))

        assert args.use_pred_depth, "Not using predicted depth"
        output_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir,
                                              'single_path_gsv')
        for i in range(2):
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "gt_%d_%d.png" % (val_idx, i)),
            my_torch_helpers.resize_torch_images(panos[:, i], size=(256, 256)))

        num_frames = 2
        for i in range(num_frames):
          t = (i + 1) / (num_frames + 1)
          # print("Distance", torch.norm(t * translations[:, 0]))

          apanos = torch.stack([
            panos[:, 0], panos[:, 0], panos[:, 1]
          ], dim=1
          )
          arotations = torch.stack([
            rotations[:, 0], rotations[:, 1], rotations[:, 1]
          ], dim=1)
          atranslation = torch.stack([
            t * translations[:, 0],
            translations[:, 1],
            translations[:, 1] + ((t - 1.0) * translations[:, 0])
          ], dim=1)
          adepths = torch.stack([
            depths[:, 0], depths[:, 0], depths[:, 0]
          ], dim=1)
          # print("atranslation", atranslation)
          run_outputs = self.run_inpainting_mesh_carla(
            100000,
            apanos,
            arotations,
            atranslation,
            adepths
          )
          inpainted_image = run_outputs['inpainted_image']
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "pred_%d_%d.png" % (val_idx, i)),
            inpainted_image)
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "output_0_%d_%d.png" % (val_idx, i)),
            run_outputs['output_0'])
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "output_2_%d_%d.png" % (val_idx, i)),
            run_outputs['output_2'])
    elif args.dataset == 'm3d':
      dataset = HabitatImageGenerator(
        "test",
        full_width=self.full_width,
        full_height=self.full_height,
        seq_len=3,
        reference_idx=1,
        m3d_dist=1 / 2
      )
      cubemap_side = 2

      # Do a run through validation data.
      for k in range(150):
        output_dir = my_helpers.join_and_make(args.inpaint_checkpoints_dir,
                                              'single_path_%d' % k)
        instance = dataset[k]
        if k not in [2, 79]:
          continue
        panos = torch.tensor(instance['rgb_panos'], device=args.device,
                             dtype=torch.float32)[None]
        rotations = torch.tensor(instance['rots'], device=args.device,
                                 dtype=torch.float32)[None]
        translations = torch.tensor(instance['trans'], device=args.device,
                                    dtype=torch.float32)[None]
        depths = torch.tensor(instance['depth_panos'], device=args.device,
                              dtype=torch.float32)[None]
        print("dataset", panos.shape)

        assert args.use_pred_depth, "Not using predicted depth"
        for i in range(panos.shape[1]):
          rotated_torch_image = my_torch_helpers.rotate_equirectangular_image(
            panos[:, i], rotations[:, i]
          )
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "gt_%d.png" % i),
            my_torch_helpers.resize_torch_images(rotated_torch_image,
                                                 size=(256, 256)))
          for side in range(6):
            gt_cb = my_torch_helpers.equirectangular_to_cubemap(
              rotated_torch_image, side=side, flipx=True
            )
            my_torch_helpers.save_torch_image(
              os.path.join(output_dir, "gt_cb_%d_%d.png" % (side, i)),
              gt_cb)
          if i in [0, 5]:
            my_torch_helpers.save_torch_image(
              os.path.join(output_dir, "%d.png" % i),
              my_torch_helpers.resize_torch_images(rotated_torch_image,
                                                   size=(256, 256)))
        translations_diff = translations[:, 0] - translations[:, 2]

        for i in range(1, 5):
          indices = [0, 1, 2]
          t = i / 5
          new_translations = torch.stack((
            translations[:, 0] + t * translations_diff - translations[:, 0],
            translations[:, 1] + t * translations_diff - translations[:, 0],
            translations[:, 2] + t * translations_diff - translations[:, 0]
          ), dim=1)
          with torch.no_grad():
            run_outputs = self.run_inpainting_mesh_carla(
              100000,
              panos[:, indices],
              rotations[:, indices],
              new_translations,
              depths[:, indices]
            )
          inpainted_image = run_outputs['inpainted_image']
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "pred_%d.png" % i),
            inpainted_image)
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "%d.png" % i),
            inpainted_image)

          for cubemap_side in range(6):
            inpainted_image_cb = my_torch_helpers.equirectangular_to_cubemap(
              inpainted_image, side=cubemap_side, flipx=True
            )
            my_torch_helpers.save_torch_image(
              os.path.join(output_dir, "pred_c%d_%d.png" % (cubemap_side, i)),
              inpainted_image_cb)
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "output_0_%d.png" % i),
            run_outputs['output_0'])
          my_torch_helpers.save_torch_image(
            os.path.join(output_dir, "output_2_%d.png" % i),
            run_outputs['output_2'])
    else:
      raise ValueError("Unknown dataset: %s" % args.dataset)

  def eval_inpainting(self):
    args = self.args

    self.depth_model.eval()
    self.inpainting_model.eval()

    validation_dataloader = DataLoader(self.val_data,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=0 if args.dataset == 'm3d' else 4)
    max_examples = len(self.val_data)
    if args.dataset == 'm3d':
      max_examples = min(len(self.val_data), 5000)
    if args.script_mode in ['eval_inpainting_testset']:
      bar = Bar('Eval on test data', max=max_examples)
      if args.dataset == 'm3d':
        results_file_path = os.path.join(
          args.inpaint_checkpoints_dir,
          "test_eval_results_m3d_%02f.txt" % (
            args.m3d_dist))
        results_file = open(
          results_file_path, "w"
        )
      else:
        results_file_path = os.path.join(
          args.inpaint_checkpoints_dir,
          "test_eval_results_carla_%02f_%02f.txt" % (
            args.carla_min_dist, args.carla_max_dist))
        results_file = open(
          results_file_path, "w"
        )
    else:
      bar = Bar('Eval on validation data', max=max_examples)
      if args.dataset == 'm3d':
        results_file_path = os.path.join(
          args.inpaint_checkpoints_dir,
          "validation_eval_results_m3d_%02f.txt" % (
            args.m3d_dist))
        results_file = open(results_file_path, "w")
      else:
        results_file_path = os.path.join(
          args.inpaint_checkpoints_dir,
          "validation_eval_results_carla_%02f_%02f.txt" % (
            args.carla_min_dist,
            args.carla_max_dist))
        results_file = open(results_file_path, "w")

    wspsnr_calculator = WSPSNR()
    assert args.use_pred_depth, "Not using predicted depth in inpainting eval"

    validation_l1_arr = []
    validation_l2_arr = []
    wspsnr_arr = []
    validation_ssim_arr = []
    with torch.no_grad():
      for i, data in enumerate(validation_dataloader):
        if i * args.batch_size > max_examples:
          break
        self.optimizer.zero_grad()
        step = self.inpainting_checkpoint_manager.increment_step()

        if args.debug_mode:
          assert distro.linux_distribution()[
                   0] == 'Ubuntu', 'Debug mode is on'
          panos = self.input_panos_val
          rots = self.input_rots_val
          trans = self.input_trans_val
          depths = self.input_depths_val
        else:
          panos = data['rgb_panos'].to(args.device)
          rots = data['rots'].to(args.device)
          trans = data['trans'].to(args.device)
          depths = data['depth_panos'].to(args.device)

        run_outputs = self.run_inpainting_mesh_carla(
          step, panos, rots, trans, depths)
        if args.script_mode in ['eval_inpainting', 'eval_inpainting_testset']:
          panos_small = run_outputs['panos_small2']
          inpainted_image = run_outputs['inpainted_image']
          inpainted_image = torch.clamp(inpainted_image, 0.0, 1.0)
          l1_loss = loss_lib.compute_l1_loss(
            255 * inpainted_image,
            255 * panos_small[:, 1],
            keep_batch=True)
          l2_loss = loss_lib.compute_l2_loss(255 * inpainted_image,
                                             255 * panos_small[:, 1],
                                             keep_batch=True)
          wspsnr_vals = wspsnr_calculator.ws_psnr(inpainted_image,
                                                  panos_small[:, 1])

          ssim_value = ssim(inpainted_image.permute((0, 3, 1, 2)),
                            panos_small[:, 1].permute((0, 3, 1, 2)),
                            size_average=False)

          validation_l1_arr.append(l1_loss.detach().cpu().numpy())
          validation_l2_arr.append(l2_loss.detach().cpu().numpy())
          wspsnr_arr.append(wspsnr_vals.detach().cpu().numpy())
          validation_ssim_arr.append(ssim_value.detach().cpu().numpy())
        else:
          rotated_panos_small = run_outputs['rotated_panos_small']
          inpainted_image = run_outputs['inpainted_image']
          inpainted_image = torch.clamp(inpainted_image, 0.0, 1.0)
          l1_loss = (loss_lib.compute_l1_sphere_loss(
            255 * inpainted_image[:, 1],
            255 * rotated_panos_small[:, 1],
            keep_batch=True) +
                     loss_lib.compute_l1_sphere_loss(
                       255 * inpainted_image[:, 2],
                       255 * rotated_panos_small[:, 2],
                       keep_batch=True)) / 2
          l2_loss = (loss_lib.compute_l2_sphere_loss(
            255 * inpainted_image[:, 1],
            255 * rotated_panos_small[:, 1],
            keep_batch=True) +
                     loss_lib.compute_l2_sphere_loss(
                       255 * inpainted_image[:, 2],
                       255 * rotated_panos_small[:, 2],
                       keep_batch=True)) / 2
          wspsnr_vals = wspsnr_calculator.ws_psnr(inpainted_image,
                                                  panos_small[:, 1])
          validation_l1_arr.append(l1_loss.detach().cpu().numpy())
          validation_l2_arr.append(l2_loss.detach().cpu().numpy())
          wspsnr_arr.append(wspsnr_vals.detach().cpu().numpy())
        bar.next(n=args.batch_size)

    wspsnr_arr = np.concatenate(wspsnr_arr)
    if args.dataset != "m3d":
      print("wspsnr_arr.shape", wspsnr_arr.shape, len(self.val_data))
      assert len(wspsnr_arr) == len(self.val_data), "wspsnr incorrect length"
    mean_validation_mae = np.mean(np.concatenate(validation_l1_arr))
    mean_validation_mse = np.mean(np.concatenate(validation_l2_arr))
    mean_validation_psnr = np.mean(wspsnr_arr)
    mean_validation_psnr_std = np.std(wspsnr_arr)
    mean_validation_ssim = np.mean(np.concatenate(validation_ssim_arr))

    bar.finish()

    print("WS-MAE: %f" % (mean_validation_mae,))
    print("WS-MSE: %f" % (mean_validation_mse,))
    print(
      "WS-PSNR: %f, std: %f" % (mean_validation_psnr, mean_validation_psnr_std))

    results_file.write(
      "%s \t %0.5f \n" % ('MAE', mean_validation_mae)
    )
    results_file.write(
      "%s \t %0.5f \n" % ('MSE', mean_validation_mse)
    )
    results_file.write(
      "%s \t %0.5f \n" % ('WS-PSNR', mean_validation_psnr)
    )
    results_file.write(
      "%s \t %0.5f \n" % ('WS-PSNR STD:', mean_validation_psnr_std)
    )
    results_file.write(
      "%s \t %0.5f \n" % ('SSIM', mean_validation_ssim)
    )
    results_file.close()

  def dump_examples(self):
    args = self.args

    self.depth_model.eval()
    self.inpainting_model.eval()

    validation_dataloader = DataLoader(self.val_data,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=0 if args.dataset == 'm3d' else 4)
    max_examples = len(self.val_data)
    if args.dataset == 'm3d':
      max_examples = min(len(self.val_data), 5000)
    bar = Bar('Dumping examples', max=max_examples)

    wspsnr_calculator = WSPSNR()
    assert args.use_pred_depth, "Not using predicted depth in inpainting eval"

    save_dir = my_helpers.join_and_make(
      args.inpaint_checkpoints_dir,
      "examples"
    )
    validation_l1_arr = []
    validation_l2_arr = []
    wspsnr_arr = []
    with torch.no_grad():
      for i, data in enumerate(validation_dataloader):
        if i * args.batch_size > max_examples:
          break
        self.optimizer.zero_grad()
        step = self.inpainting_checkpoint_manager.increment_step()

        if args.debug_mode:
          assert distro.linux_distribution()[
                   0] == 'Ubuntu', 'Debug mode is on'
          panos = self.input_panos_val
          rots = self.input_rots_val
          trans = self.input_trans_val
          depths = self.input_depths_val
        else:
          panos = data['rgb_panos'].to(args.device)
          rots = data['rots'].to(args.device)
          trans = data['trans'].to(args.device)
          depths = data['depth_panos'].to(args.device)

        run_outputs = self.run_inpainting_mesh_carla(
          step, panos, rots, trans, depths)
        if args.script_mode in {'eval_inpainting', 'eval_inpainting_testset',
                                'dump_examples'}:
          panos_small = run_outputs['panos_small']
          inpainted_image = run_outputs['inpainted_image']
          inpainted_image = torch.clamp(inpainted_image, 0.0, 1.0)
          my_torch_helpers.save_torch_image(
            os.path.join(save_dir, "%d_1input0.png" % i),
            run_outputs['panos_small'][:, 0]
          )
          my_torch_helpers.save_torch_image(
            os.path.join(save_dir, "%d_2gt_output.png" % i),
            run_outputs['panos_small'][:, 1]
          )
          my_torch_helpers.save_torch_image(
            os.path.join(save_dir, "%d_3gt_output.png" % i),
            run_outputs['panos_small'][:, 2]
          )
          my_torch_helpers.save_torch_image(
            os.path.join(save_dir, "%d_4render0.png" % i),
            run_outputs['output_0']
          )
          my_torch_helpers.save_torch_image(
            os.path.join(save_dir, "%d_5render2.png" % i),
            run_outputs['output_2']
          )
          my_torch_helpers.save_torch_image(
            os.path.join(save_dir, "%d_6pred.png" % i),
            inpainted_image
          )
        else:
          rotated_panos_small = run_outputs['rotated_panos_small']
          inpainted_image = run_outputs['inpainted_image']
          inpainted_image = torch.clamp(inpainted_image, 0.0, 1.0)
          l1_loss = (loss_lib.compute_l1_sphere_loss(
            255 * inpainted_image[:, 1],
            255 * rotated_panos_small[:, 1],
            keep_batch=True) +
                     loss_lib.compute_l1_sphere_loss(
                       255 * inpainted_image[:, 2],
                       255 * rotated_panos_small[:, 2],
                       keep_batch=True)) / 2
          l2_loss = (loss_lib.compute_l2_sphere_loss(
            255 * inpainted_image[:, 1],
            255 * rotated_panos_small[:, 1],
            keep_batch=True) +
                     loss_lib.compute_l2_sphere_loss(
                       255 * inpainted_image[:, 2],
                       255 * rotated_panos_small[:, 2],
                       keep_batch=True)) / 2
          wspsnr_vals = wspsnr_calculator.ws_psnr(inpainted_image,
                                                  panos_small[:, 1])
          validation_l1_arr.append(l1_loss.detach().cpu().numpy())
          validation_l2_arr.append(l2_loss.detach().cpu().numpy())
          wspsnr_arr.append(wspsnr_vals.detach().cpu().numpy())
        bar.next(n=args.batch_size)

  def run_example(self):
    args = self.args
    depth_model = self.depth_model
    inpainting_model = self.inpainting_model

    depth_model.eval()
    inpainting_model.eval()
    assert args.dataset == "carla"

    inputs_dir = "example/inputs"
    input_pano0 = plt.imread(os.path.join(inputs_dir, "0.png"))[:,:,:3]
    input_pano5 = plt.imread(os.path.join(inputs_dir, "5.png"))[:,:,:3]
    rotations_np = np.load(os.path.join(inputs_dir, "rotations.npy"))
    translations_np = np.load(os.path.join(inputs_dir, "translations.npy"))

    panos = torch.tensor(np.stack((
      input_pano0,
      input_pano0,
      input_pano0,
      input_pano0,
      input_pano0,
      input_pano5
    )), device=args.device, dtype=torch.float32)[None]
    rotations = torch.tensor(rotations_np, device=args.device,
                              dtype=torch.float32)[None]
    translations = torch.tensor(translations_np, device=args.device,
                                dtype=torch.float32)[None]
    depths = 0.0 * panos[:, :, :, :, 0]


    assert args.use_pred_depth, "Not using predicted depth"
    output_dir = my_helpers.join_and_make("example/outputs")

    dataset = CarlaReader("")

    for i in range(1, 5):
      indices = [0, i, 5]
      run_outputs = self.run_inpainting_mesh_carla(
        100000,
        panos[:, indices],
        rotations[:, indices],
        translations[:, indices] - translations[:, i, None],
        depths[:, indices]
      )
      inpainted_image = run_outputs['inpainted_image']
      my_torch_helpers.save_torch_image(
          os.path.join(output_dir, "pred_%d.png" % i),
          inpainted_image)

  def visualize_point_cloud(self):
    import open3d as o3d
    args = self.args
    device = args.device
    dtype = torch.float32
    val_dataloader = DataLoader(self.val_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0 if args.dataset == 'm3d' else 4)
    self.depth_model.eval()
    with torch.no_grad():
      it = iter(val_dataloader)
      data = it.next()
      panos = data["rgb_panos"].to(args.device)
      depths = data["depth_panos"].to(args.device)
      rots = data["rots"].to(args.device)
      trans = data["trans"].to(args.device)

      _, seq_len, panos_height, panos_width, _ = panos.shape
      panos_small = panos.reshape(
        (1 * seq_len, panos_height, panos_width, 3))
      panos_small = my_torch_helpers.resize_torch_images(
        panos_small, (args.width, args.height), mode=args.interpolation_mode)
      panos_small = panos_small.reshape(1, seq_len, args.height, args.width, 3)

      visualize_using_gt_depth = False
      if visualize_using_gt_depth:
        depths_small = depths.reshape(
          (1 * seq_len, panos_height, panos_width, 1))
        depths_small = my_torch_helpers.resize_torch_images(
          depths_small, (args.width, args.height), mode=args.interpolation_mode)
        depths_small = depths_small.reshape(1, seq_len, args.height, args.width)

        depth0 = depths_small[0,0]
        depth2 = depths_small[0,2]
      else:
        outputs0 = self.depth_model.estimate_depth_using_cost_volume(
          panos_small[:, [2, 0], :, :, :],
          rots[:, [2, 0]],
          trans[:, [2, 0]],
          min_depth=args.min_depth,
          max_depth=args.max_depth)
        
        outputs2 = self.depth_model.estimate_depth_using_cost_volume(
          panos_small[:, [0, 2], :, :, :],
          rots[:, [0, 2]],
          trans[:, [0, 2]],
          min_depth=args.min_depth,
          max_depth=args.max_depth)
        depth0 = outputs0["depth"][0, :, :, 0]
        depth2 = outputs2["depth"][0, :, :, 0]


      height, width = depth0.shape

      phi = torch.arange(0, height, device=device, dtype=dtype)
      phi = (phi + 0.5) * (np.pi / height)
      theta = torch.arange(0, width, device=device, dtype=dtype)
      theta = (theta + 0.5) * (2 * np.pi / width) + np.pi / 2
      phi, theta = torch.meshgrid(phi, theta)

      pano_0_positions = my_torch_helpers.spherical_to_cartesian(theta, phi, depth0)
      pano_0_positions = pano_0_positions.reshape(height * width, 3) - trans[0, 0][None]
      pano_0_positions = pano_0_positions.cpu().numpy()
      pano_0_colors = panos_small[0, 0].reshape(height * width, 3).cpu().numpy()
      point_cloud_0 = o3d.geometry.PointCloud()
      point_cloud_0.points = o3d.utility.Vector3dVector(pano_0_positions)
      point_cloud_0.colors = o3d.utility.Vector3dVector(pano_0_colors)

      pano_1_positions = my_torch_helpers.spherical_to_cartesian(theta, phi, depth2)
      pano_1_positions = pano_1_positions.reshape(height * width, 3) - trans[0, 2][None]
      pano_1_positions = pano_1_positions.cpu().numpy()
      pano_1_colors = panos_small[0, 2].reshape(height * width, 3).cpu().numpy()
      point_cloud_1 = o3d.geometry.PointCloud()
      point_cloud_1.points = o3d.utility.Vector3dVector(pano_1_positions)
      point_cloud_1.colors = o3d.utility.Vector3dVector(pano_1_colors)

      o3d.visualization.draw_geometries([point_cloud_0, point_cloud_1])

if __name__ == "__main__":
  app = App()
  app.start()
