# Lint as: python3
"""Train depth and pose on the carla dataset.
"""

import os

import distro
import numpy as np
# Pytorch Imports
import torch
from progress.bar import Bar
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_readers.carla_reader import CarlaReader
from data_readers.habitat_data import HabitatImageGenerator
from helpers import my_torch_helpers
from helpers.torch_checkpoint_manager import CheckpointManager
from models import loss_lib
from models.pipeline3_model import FullPipeline
from options import parse_training_options


class App:
  """Main app class"""

  def __init__(self):
    self.model = None
    self.optimizer = None
    self.checkpoint_manager = None
    self.args = None
    self.full_width = 512
    self.full_height = 256
    self.writer = None

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

  def parse_args(self):
    args = parse_training_options()
    args.device = my_torch_helpers.find_torch_device(args.device)
    print("args", args)
    self.args = args

  def start(self):
    """Starts the training."""

    try:
      self.parse_args()
      args = self.args
      self.load_training_data()
      self.load_validation_data()
      self.setup_model()
      self.setup_checkpoints()

      if args.script_mode == "train_depth_pose":
        self.run_training_loop()
      elif args.script_mode == "eval_depth_pose":
        total_params = self.model.get_total_params()
        print("Total parameters:", total_params)
        self.eval_on_training_data()
        self.eval_on_validation_data()
      elif args.script_mode == "eval_depth_test":
        total_params = self.model.get_total_params()
        print("Total parameters:", total_params)
        self.eval_on_validation_data()
      else:
        raise ValueError("Unknown script mode: " + str(args.script_mode))

    except KeyboardInterrupt:
      print("Terminating script")
      self.writer.close()

  def setup_model(self):
    """Sets up the model."""
    args = self.args
    model = FullPipeline(device=args.device,
                         monodepth_model=args.model_name,
                         width=args.width,
                         height=args.height,
                         layers=7,
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
                         depth_type=args.depth_type).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(args.opt_beta1, args.opt_beta2))
    self.model = model
    self.optimizer = optimizer

  def setup_checkpoints(self):
    """Sets up the checkpoint manager."""
    args = self.args
    model = self.model
    optimizer = self.optimizer

    checkpoint_manager = CheckpointManager(args.checkpoints_dir,
                                           max_to_keep=args.checkpoint_count)
    latest_checkpoint = checkpoint_manager.load_latest_checkpoint()
    if latest_checkpoint is not None:
      model.load_state_dict(latest_checkpoint['model_state_dict'])
      optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])

    writer = SummaryWriter(log_dir=os.path.join(args.checkpoints_dir, "logs"))

    self.checkpoint_manager = checkpoint_manager
    self.writer = writer

  def load_training_data(self):
    """Loads training data."""
    args = self.args

    # Prepare dataset loaders for train and validation datasets.
    if args.dataset == "carla":
      train_data = CarlaReader(
        args.carla_path,
        width=self.full_width,
        height=self.full_height,
        towns=["Town01", "Town02", "Town03", "Town04"],
        min_dist=args.carla_min_dist,
        max_dist=args.carla_max_dist,
        seq_len=2,
        reference_idx=1,
        use_meters_depth=True,
        interpolation_mode=args.interpolation_mode,
        sampling_method="dense")
      print("Size of training set: %d" % (len(train_data),))
      train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)
    elif args.dataset == "m3d":
      train_data = HabitatImageGenerator(
        "train",
        full_width=self.full_width,
        full_height=self.full_height,
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

    train_data.cache_depth_to_dist(args.height, args.width)

    self.train_data = train_data
    self.train_data_loader = train_dataloader

  def load_validation_data(self):
    """Loads validation data."""
    args = self.args

    if args.dataset == "carla":
      towns = ["Town05"]
      if args.script_mode == "eval_depth_test":
        towns = ["Town06"]
      val_data = CarlaReader(
        args.carla_path,
        width=self.full_width,
        height=self.full_height,
        towns=towns,
        min_dist=args.carla_min_dist,
        max_dist=args.carla_max_dist,
        seq_len=2,
        reference_idx=1,
        use_meters_depth=True,
        interpolation_mode=args.interpolation_mode)
    elif args.dataset == "m3d":
      if args.script_mode == "eval_depth_test":
        val_data = HabitatImageGenerator(
          "val",
          full_width=self.full_width,
          full_height=self.full_height,
          m3d_dist=args.m3d_dist)
      else:
        val_data = HabitatImageGenerator(
          "test",
          full_width=self.full_width,
          full_height=self.full_height,
          m3d_dist=args.m3d_dist)

    # Load a single batch of validation data.
    # val_data_indices = [20, 40, 60, 80, 100]
    val_data_indices = [0, 40, 80, 120]
    val_data_all = tuple(val_data[i] for i in val_data_indices)
    input_panos_val = np.stack(tuple(
      v_data["rgb_panos"] for v_data in val_data_all),
      axis=0)
    input_panos_val = torch.tensor(input_panos_val,
                                   dtype=torch.float32,
                                   device=args.device)
    input_depths_val = np.stack(tuple(
      v_data["depth_panos"] for v_data in val_data_all),
      axis=0)
    input_depths_val = torch.tensor(input_depths_val,
                                    dtype=torch.float32,
                                    device=args.device)
    input_rots_val = np.stack(
      tuple(v_data["rots"] for v_data in val_data_all),
      axis=0)
    input_rots_val = torch.tensor(input_rots_val,
                                  dtype=torch.float32,
                                  device=args.device)
    input_trans_val = np.stack(tuple(
      v_data["trans"] for v_data in val_data_all),
      axis=0)
    input_trans_val = torch.tensor(input_trans_val,
                                   dtype=torch.float32,
                                   device=args.device)

    self.val_data = val_data
    self.val_data_indices = val_data_indices
    self.input_panos_val = input_panos_val
    self.input_depths_val = input_depths_val
    self.input_rots_val = input_rots_val
    self.input_trans_val = input_trans_val

  def run_depth_pose_carla(self, step, panos, depths, rots, trans):
    """Does a single run and returns results.

    Args:
      step: Current step.
      panos: Input panoramas.
      depths: GT depths.
      rots: GT rotations.
      trans: GT translations.

    Returns:
      Dictionary containing all outputs.

    """

    args = self.args
    height = args.height
    width = args.width
    model = self.model
    train_data = self.train_data

    batch_size, seq_len = panos.shape[:2]
    downsample_size = int(64 + (width - 64) * np.clip(step / 2000, 0, 1))
    # downsample_size = 256
    # panos_channels_stacked = torch.cat((panos[:, 0], panos[:, 1], panos[:, 2]),
    #                                    dim=3)

    panos_small = panos.reshape(
      (batch_size * seq_len, self.full_height, self.full_width, 3))
    panos_small = my_torch_helpers.resize_torch_images(
      panos_small, (args.width, args.height), mode=args.interpolation_mode)
    panos_small = panos_small.reshape(batch_size, seq_len, height, width, 3)

    depths_height, depths_width = depths.shape[2:4]
    depths_small = depths.reshape(
      (batch_size * seq_len, depths_height, depths_width, 1))
    depths_small = my_torch_helpers.resize_torch_images(
      depths_small, (args.width, args.height), mode=args.interpolation_mode)
    depths_small = depths_small.reshape(batch_size, seq_len, height, width, 1)

    rots_pred, trans_pred = model.estimate_pose(panos_small[:, :2, :, :, :])
    if args.cost_volume:
      outputs = model.estimate_depth_using_cost_volume(panos_small, rots, trans,
                                                       min_depth=args.min_depth,
                                                       max_depth=args.max_depth)
      depths_pred = outputs["depth"]
    else:
      outputs = model.estimate_depth(panos_small[:, 1], loss=args.loss)
      depths_pred = outputs["depth"]
    if args.predict_zdepth:
      depths_pred = train_data.zdepth_to_distance_torch(depths_pred)
    depths_pred = depths_pred.reshape(
      (batch_size, 1, height, width, depths_pred.shape[3]))
    assert torch.isfinite(depths_pred).all(), "Nan in depths_pred"
    depth_smoothness_loss = loss_lib.compute_depth_smoothness_loss(
      depths_pred[:, 0])
    if args.normalize_depth:
      depth_range_penalty = loss_lib.depth_range_penalty(depths_pred[:, 0],
                                                         min=1,
                                                         max=100)
    else:
      depth_range_penalty = torch.tensor(0.0,
                                         device=args.device,
                                         dtype=depths_pred.dtype)

    disp_c1 = None
    depths_c1 = None
    zdepths_small_1 = None
    rect_gt_depth = None
    rect_gt_disp = None
    disp_pred_c1 = None
    if args.cost_volume == "v1" or \
        args.cost_volume == "v2" or \
        args.cost_volume == "v3":
      rect_gt_depth = my_torch_helpers.rotate_equirectangular_image(
        depths_small[:, 1], outputs["rect_rots"][:, 1])
      rect_gt_disp = model.erp_depth_to_disparity(
        rect_gt_depth.permute((0, 3, 1, 2)), outputs["trans_norm"])
      rect_gt_disp = rect_gt_disp.permute((0, 2, 3, 1))
      unrect_gt_disp = model.unrectify_image(rect_gt_disp,
                                             outputs["rect_rots"][:, 1])
      assert torch.isfinite(rect_gt_disp).all(), "Nan in rect_gt_disp"
      assert torch.isfinite(
        outputs["raw_image_features"]).all(), "Nan in raw image features"
    if args.loss == "l1_cost_volume_output":
      loss1 = torch.mean(
        torch.abs(rect_gt_disp - outputs["raw_image_features"]))
    elif args.loss == "l1_sphere_cost_volume_output":
      loss1 = loss_lib.compute_l1_sphere_loss(outputs["raw_image_features"],
                                              rect_gt_disp)
    elif args.loss == "l1_sphere_cost_volume_output_v3":
      loss1 = loss_lib.compute_l1_sphere_loss(outputs["raw_image_features"],
                                              rect_gt_disp)
      loss2 = loss_lib.compute_l1_sphere_loss(outputs["raw_image_features_d1"],
                                              rect_gt_disp)
      loss1 = loss1 + loss2
    elif args.loss == "l1_cylindrical_cost_volume_output":
      rect_gt_depth = my_torch_helpers.equirectangular_to_cylindrical(
        depths_small[:, 1],
        cylinder_length=10.0,
        width=args.width,
        height=args.height,
        rect_rots=outputs["rect_rots"][:, 1],
        depth=True
      )
      rect_gt_disp = my_torch_helpers.safe_divide(
        outputs["trans_norm"].view(batch_size, 1, 1, 1),
        rect_gt_depth
      )
      loss1 = loss_lib.compute_l1_sphere_loss(outputs["raw_image_features"],
                                              rect_gt_disp)
    elif args.loss == "l1_unrect_sphere_cost_volume_output":
      loss1 = loss_lib.compute_l1_sphere_loss(outputs["unrect_disp"],
                                              unrect_gt_disp)
    elif args.loss == "l1_cubemap_cost_volume_output":
      zdepths_small_1 = self.train_data.distance_to_zdepth_torch(
        depths_small[:, 1])
      depths_c1 = my_torch_helpers.equirectangular_to_cubemap(zdepths_small_1,
                                                              side=2)
      disp_c1 = torch.div(outputs["trans_norm"][:, None, None, None], depths_c1)
      loss1 = loss_lib.compute_l1_loss(outputs["raw_image_features"], disp_c1)
      assert torch.isfinite(
        outputs["raw_image_features"]).all(), "Nan in raw image features"
    elif args.loss == "l1":
      assert torch.isfinite(depths_small).all(), "Nan in depths_small"
      loss1 = loss_lib.compute_l1_loss(depths_pred[:, 0],
                                       depths_small[:, 1],
                                       normalize=args.normalize_depth)
    elif args.loss == "l1_cost_volume_erp":
      assert torch.isfinite(depths_small).all(), "Nan in depths_small"
      rect_gt_depth = my_torch_helpers.rotate_equirectangular_image(
        depths_small[:, 1], outputs["rect_rots"][:, 1])
      one_over_gt_depth = my_torch_helpers.safe_divide(1.0, rect_gt_depth)
      loss1 = loss_lib.compute_l1_sphere_loss(
        outputs['raw_image_features'],
        one_over_gt_depth,
        mask=torch.gt(rect_gt_depth, 0.1))
      loss1 = loss1 + 0.5 * loss_lib.compute_l1_sphere_loss(
        outputs['raw_image_features_d1'],
        one_over_gt_depth,
        mask=torch.gt(rect_gt_depth, 0.1))
    elif args.loss == "l1_cost_volume_carla":
      assert torch.isfinite(depths_small).all(), "Nan in depths_small"
      rect_gt_depth = my_torch_helpers.rotate_equirectangular_image(
        depths_small[:, 1], outputs["rect_rots"][:, 1])
      loss1 = loss_lib.compute_l1_sphere_loss(
        outputs['rectified_depth'],
        rect_gt_depth,
        mask=torch.gt(rect_gt_depth, 0.1))
      loss1 = loss1 + 0.5 * loss_lib.compute_l1_sphere_loss(
        outputs['rectified_depth_d1'],
        rect_gt_depth,
        mask=torch.gt(rect_gt_depth, 0.1))
    elif args.loss == "l1_sphere":
      assert torch.isfinite(depths_small).all(), "Nan in depths_small"
      loss1 = loss_lib.compute_l1_sphere_loss(depths_pred[:, 0],
                                              depths_small[:, 1])
    elif args.loss == "l1_one_over_depth":
      print("outputs shape", outputs["output"].shape, depths_small[:, 1].shape)
      loss1 = 100.0 * loss_lib.compute_l1_loss(outputs["output"],
                                               1.0 / depths_small[:, 1],
                                               normalize=args.normalize_depth)
    elif args.loss == "l1_log_depth":
      loss1 = loss_lib.compute_l1_loss(outputs["output"],
                                       torch.log(depths_small[:, 1]),
                                       normalize=args.normalize_depth)
    else:
      raise ValueError("Loss not found: %s" % (args.loss,))

    rot_loss = torch.mean(torch.abs(rots_pred - rots[:, 0]))
    trans_loss = torch.mean(torch.abs(trans_pred - trans[:, 0, :]))
    final_loss = loss1 + \
                 args.smoothness_loss_lambda * depth_smoothness_loss + \
                 args.rot_loss_lambda * rot_loss + \
                 args.trans_loss_lambda * trans_loss + \
                 args.depth_range_loss_lambda * depth_range_penalty

    depths_pred = torch.clamp(depths_pred, min=0.1)

    assert torch.isfinite(loss1).all(), "Nan in depth final_loss"
    assert torch.isfinite(args.rot_loss_lambda *
                          rot_loss).all(), "Nan in rot_loss"
    assert torch.isfinite(args.trans_loss_lambda *
                          trans_loss).all(), "Nan in trans_loss"
    assert torch.isfinite(args.smoothness_loss_lambda *
                          depth_smoothness_loss).all(), "Nan in smoothness final_loss"
    assert torch.isfinite(args.depth_range_loss_lambda *
                          depth_range_penalty).all(), "Nan in depth_range_penalty"
    assert torch.isfinite(final_loss).all(), "Nan in final_loss function"

    return {
      "loss1": loss1,
      "final_loss": final_loss,
      "rot_loss": rot_loss,
      "trans_loss": trans_loss,
      "depths_pred": depths_pred,
      "panos_small": panos_small,
      "depths_small": depths_small,
      "outputs": outputs,
      "rect_gt_depth": rect_gt_depth,
      "rect_gt_disp": rect_gt_disp,
      "depth_smoothness_loss": depth_smoothness_loss,
      "disp_c1": disp_c1,
      "disp_c1_pred": disp_pred_c1,
      "depths_c1": depths_c1,
      "zdepths_small_1": zdepths_small_1,
      "rots_pred": rots_pred,
      "trans_pred": trans_pred
    }

  def do_validation_run(self, step):
    """Does a validation run.

    Args:
      step: Current step.

    Returns:
      None.

    """

    args = self.args
    model = self.model
    writer = self.writer

    if step == 1 or \
        args.validation_interval == 0 or \
        step % args.validation_interval == 0:
      # Calculate validation final_loss.
      with torch.no_grad():
        panos = self.input_panos_val
        depths = self.input_depths_val
        rots = self.input_rots_val
        trans = self.input_trans_val
        batch_size, seq_len = panos.shape[:2]

        run_outputs = self.run_depth_pose_carla(step, panos, depths, rots,
                                                trans)
        final_loss = run_outputs["final_loss"]
        depths_pred = run_outputs["depths_pred"]
        depths_small = run_outputs["depths_small"]
        panos_small = run_outputs["panos_small"]

        writer.add_scalar("val_loss", final_loss.item(), step)
        writer.add_scalar("val_image_loss", run_outputs["loss1"].item(), step)
        writer.add_scalar("val_rot_loss", run_outputs["rot_loss"].item(), step)
        writer.add_scalar("val_trans_loss", run_outputs["trans_loss"].item(),
                          step)

        if args.cost_volume == "v2_cubemap":

          panos_c0 = my_torch_helpers.equirectangular_to_cubemap(panos[:, 0],
                                                                 side=2)
          panos_c1 = my_torch_helpers.equirectangular_to_cubemap(panos[:, 1],
                                                                 side=2)
          depths_c1_t = my_torch_helpers.depth_to_turbo_colormap(
            run_outputs["depths_c1"], min_depth=args.turbo_cmap_min)
          depths_pred_c1_t = my_torch_helpers.depth_to_turbo_colormap(
            run_outputs["depths_pred"][:, 0], min_depth=args.turbo_cmap_min)
          y_stacked = torch.cat(
            (panos_c0, panos_c1, depths_c1_t, depths_pred_c1_t), dim=2)

          for j in range(len(self.val_data_indices)):
            writer.add_image("80_val_image_%02d" % j,
                             y_stacked[j].clamp(0, 1),
                             step,
                             dataformats="HWC")
        else:
          depths_turbo = my_torch_helpers.depth_to_turbo_colormap(
            depths_small[:, 1], min_depth=args.turbo_cmap_min)
          normalized_depth_pred = depths_pred[:, 0]
          if args.normalize_depth:
            std, mean = torch.std_mean(depths_small[:, 1],
                                       dim=(1, 2),
                                       keepdim=True)
            normalized_depth_pred = loss_lib.normalize_depth(
              normalized_depth_pred, new_std=std, new_mean=mean)
          depths_pred_turbo = my_torch_helpers.depth_to_turbo_colormap(
            normalized_depth_pred, min_depth=args.turbo_cmap_min)

          # back_warped_1 = model.backwards_warping(panos[:, 0],
          #                                         depths_small[:, 1, :, :, 0],
          #                                         run_outputs["rots_pred"],
          #                                         run_outputs["trans_pred"],
          #                                         inv_rot=False)

          depth_abs_error_img = torch.abs(depths_small[:, 1] -
                                          normalized_depth_pred)
          depth_abs_error_img = depth_abs_error_img.expand(
            (batch_size, args.height, args.width, 3))
          depth_abs_error_img_stacked = torch.cat(
            (panos_small[:, 1], depths_turbo, depths_pred_turbo,
             depth_abs_error_img),
            dim=2)
          depth_mae = torch.mean(torch.abs(depths_small[:, 1] -
                                           normalized_depth_pred),
                                 dim=(1, 2, 3))
          depth_mse = torch.mean(torch.pow(
            depths_small[:, 1] - normalized_depth_pred, 2.0),
            dim=(1, 2, 3))

          y_stacked = torch.cat((panos_small[:, 0], panos_small[:, 1],
                                 depths_turbo, depths_pred_turbo),
                                dim=2)
          for j in range(len(self.val_data_indices)):
            writer.add_image("80_val_image_%02d" % j,
                             y_stacked[j].clamp(0, 1),
                             step,
                             dataformats="HWC")
            writer.add_image("82_val_depth_ae_%02d" % j,
                             depth_abs_error_img_stacked[j].clamp(0, 1),
                             step,
                             dataformats="HWC")
            writer.add_scalar("84_val_depth_mae_%02d" % j, depth_mae[j], step)
            writer.add_scalar("86_val_depth_mse_%02d" % j, depth_mse[j], step)

  def log_training_to_tensorboard(self, step, run_outputs):
    """Logs training to tensorboard.

    Args:
      step: Current step.
      run_outputs: Outputs of the training step.

    Returns:
      None.

    """
    args = self.args
    model = self.model
    writer = self.writer

    depths_pred = run_outputs["depths_pred"]
    depths_small = run_outputs["depths_small"]
    outputs = run_outputs["outputs"]
    rect_gt_depth = run_outputs["rect_gt_depth"]
    rect_gt_disp = run_outputs["rect_gt_disp"]
    panos_small = run_outputs["panos_small"]

    final_loss = run_outputs["final_loss"]
    loss_np = final_loss.detach().cpu().numpy()
    average_depth_np = torch.mean(depths_pred).detach().cpu().numpy()

    writer.add_scalar("train_loss", loss_np, step)
    writer.add_scalar("train_depth", average_depth_np, step)
    writer.add_scalar("train_image_loss", run_outputs["loss1"].item(), step)
    writer.add_scalar("train_depth_smoothness_loss",
                      run_outputs["depth_smoothness_loss"].item(), step)
    writer.add_scalar("train_rot_loss", run_outputs["rot_loss"].item(), step)
    writer.add_scalar("train_trans_loss", run_outputs["trans_loss"].item(),
                      step)

    if step == 1 or \
        args.train_tensorboard_interval == 0 or \
        step % args.train_tensorboard_interval == 0:
      with torch.no_grad():
        depths_small_turbo = my_torch_helpers.depth_to_turbo_colormap(
          depths_small[:, 1], min_depth=args.turbo_cmap_min)
        depths_scale_factor = 1
        normalized_depth_pred = depths_pred[:, 0]
        if args.normalize_depth:
          std, mean = torch.std_mean(depths_small[:, 1], dim=(1, 2),
                                     keepdim=True)
          normalized_depth_pred = loss_lib.normalize_depth(
            normalized_depth_pred,
            new_std=std,
            new_mean=mean)
        depths_pred_turbo = my_torch_helpers.depth_to_turbo_colormap(
          normalized_depth_pred, min_depth=args.turbo_cmap_min)
        stacked_input_panos = torch.cat((panos_small[:, 0], panos_small[:, 1]),
                                        dim=1)
        writer.add_images("00_train_inputs",
                          stacked_input_panos,
                          step,
                          dataformats="NHWC")
        writer.add_images("05_train_depths_gt",
                          depths_small_turbo,
                          step,
                          dataformats="NHWC")
        writer.add_images("10_train_pred_depths",
                          depths_pred_turbo,
                          step,
                          dataformats="NHWC")
        y_pred = depths_pred[:, 0]
        y_true = depths_small[:, 1]
        if args.normalize_depth:
          y_pred = loss_lib.normalize_depth(y_pred)
          y_true = loss_lib.normalize_depth(y_true)
        depth_loss_image = torch.abs(y_true - y_pred)
        writer.add_images("11_train_l1_loss_image",
                          depth_loss_image.clamp(0, 1),
                          step,
                          dataformats="NHWC")
        if args.cost_volume == "v2_cubemap":
          panos_c0 = run_outputs["outupts"]["panos_c0"]
          panos_c1 = run_outputs["outupts"]["panos_c1"]
          panos_c = torch.cat((panos_c0, panos_c1), dim=1)
          disp_c1_t = my_torch_helpers.depth_to_turbo_colormap(
            run_outputs["disp_c1"], min_depth=0.1)
          depths_c1_t = my_torch_helpers.depth_to_turbo_colormap(
            run_outputs["depths_c1"], min_depth=args.turbo_cmap_min)
          raw_image_features = my_torch_helpers.depth_to_turbo_colormap(
            outputs["raw_image_features"], min_depth=0.1)
          writer.add_images("12_train_cubemap_gt_depth",
                            depths_c1_t,
                            step,
                            dataformats="NHWC")
          writer.add_images("13_train_cubemap_gt_disp",
                            disp_c1_t,
                            step,
                            dataformats="NHWC")
          writer.add_images("15_train_cubemap_images",
                            panos_c,
                            step,
                            dataformats="NHWC")
          writer.add_images("25_train_raw_image_features",
                            raw_image_features,
                            step,
                            dataformats="NHWC")
          writer.add_images("26_train_depths_pred",
                            depths_pred_turbo,
                            step,
                            dataformats="NHWC")
        elif args.cost_volume == "v3_cylindrical":
          raw_image_features = my_torch_helpers.depth_to_turbo_colormap(
            outputs["raw_image_features"], min_depth=0.01)
          rectified_panos_cat = torch.cat(
            (outputs["rectified_panos"][:, 0],
             outputs["rectified_panos"][:, 1]),
            dim=1)

          rect_gt_depth_t = my_torch_helpers.depth_to_turbo_colormap(
            rect_gt_depth, min_depth=args.turbo_cmap_min)
          rect_gt_disp_t = my_torch_helpers.depth_to_turbo_colormap(
            rect_gt_disp, min_depth=0.01)

          writer.add_images("12_train_rect_gt_depth",
                            rect_gt_depth_t,
                            step,
                            dataformats="NHWC")
          writer.add_images("13_train_rect_gt_disp",
                            rect_gt_disp_t,
                            step,
                            dataformats="NHWC")
          writer.add_images("25_train_rect_disp",
                            raw_image_features,
                            step,
                            dataformats="NHWC")
          writer.add_images("26_train_rect_images",
                            rectified_panos_cat,
                            step,
                            dataformats="NHWC")
        elif args.cost_volume == 'v3_erp':
          pass
        elif args.cost_volume:
          raw_image_features = my_torch_helpers.depth_to_turbo_colormap(
            outputs["raw_image_features"], min_depth=0.01)
          rectified_panos_cat = torch.cat(
            (outputs["rectified_panos"][:, 0], outputs["rectified_panos"][:,
                                               1]),
            dim=1)

          rect_gt_depth_t = my_torch_helpers.depth_to_turbo_colormap(
            rect_gt_depth, min_depth=args.turbo_cmap_min)
          rect_gt_disp_t = my_torch_helpers.depth_to_turbo_colormap(
            rect_gt_disp, min_depth=0.01)
          rect_gt_depth_b = model.erp_disparity_to_depth(
            rect_gt_disp.permute((0, 3, 1, 2)), outputs["trans_norm"])
          rect_gt_depth_b_t = my_torch_helpers.depth_to_turbo_colormap(
            rect_gt_depth_b.permute((0, 2, 3, 1)),
            min_depth=args.turbo_cmap_min)

          unrect_gt_depth_b = model.unrectify_image(
            rect_gt_depth_b.permute((0, 2, 3, 1)), outputs["rect_rots"][:, 1])
          unrect_gt_depth_b_t = my_torch_helpers.depth_to_turbo_colormap(
            unrect_gt_depth_b, min_depth=args.turbo_cmap_min)

          writer.add_images("12_train_rect_gt_depth",
                            rect_gt_depth_t,
                            step,
                            dataformats="NHWC")
          writer.add_images("13_train_rect_gt_disp",
                            rect_gt_disp_t,
                            step,
                            dataformats="NHWC")
          writer.add_images("15_train_rect_gt_depth_b",
                            rect_gt_depth_b_t,
                            step,
                            dataformats="NHWC")
          writer.add_images("15_train_unrect_gt_depth_b",
                            unrect_gt_depth_b_t,
                            step,
                            dataformats="NHWC")
          writer.add_images("25_train_raw_image_features",
                            raw_image_features,
                            step,
                            dataformats="NHWC")
          writer.add_images("26_train_rect_images",
                            rectified_panos_cat,
                            step,
                            dataformats="NHWC")

  def save_checkpoint(self, step):
    """Saves a checkpoint.

    Args:
      step: Current step.

    Returns:
      None.

    """
    args = self.args

    if args.checkpoint_interval == 0 or step % args.checkpoint_interval == 0:
      # Save a checkpoint
      self.checkpoint_manager.save_checkpoint({
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict()
      })
      self.writer.flush()

  def run_training_loop(self):
    args = self.args
    train_dataloader = self.train_data_loader
    optimizer = self.optimizer
    checkpoint_manager = self.checkpoint_manager
    model = self.model

    for epoch in range(args.epochs):
      for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        step = checkpoint_manager.increment_step()

        if args.debug_mode:
          assert distro.linux_distribution()[0] == "Ubuntu", "Debug mode is on"
          panos = self.input_panos_val
          depths = self.input_depths_val
          rots = self.input_rots_val
          trans = self.input_trans_val
        else:
          assert distro is not None
          panos = data["rgb_panos"].to(args.device)
          depths = data["depth_panos"].to(args.device)
          rots = data["rots"].to(args.device)
          trans = data["trans"].to(args.device)

        # print("panos", panos.dtype, depths.dtype, rots.dtype, trans.dtype)
        # print("maxmin", torch.max(panos), torch.min(panos))

        run_outputs = self.run_depth_pose_carla(step, panos, depths,
                                                rots,
                                                trans)
        self.log_training_to_tensorboard(step, run_outputs)

        final_loss = run_outputs["final_loss"]
        depths_pred = run_outputs["depths_pred"]

        final_loss.backward()
        if args.clip_grad_value > 1e-10:
          # print("Clipping gradients to %f" % args.clip_grad_value)
          torch.nn.utils.clip_grad_value_(model.parameters(),
                                          args.clip_grad_value)

        optimizer.step()

        self.do_validation_run(step)
        self.save_checkpoint(step)

        loss_np = final_loss.detach().cpu().numpy()
        average_depth_np = torch.mean(depths_pred).detach().cpu().numpy()
        print("Step: %d [%d:%d] Loss: %f, average depth %f" %
              (step, epoch, i, loss_np, average_depth_np))

  def eval_on_training_data(self):
    """Performs evaluation on the whole evaluation dataset.

    Returns:
      None
    """
    args = self.args
    train_dataloader = DataLoader(self.train_data,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=4)

    self.model.eval()

    num_iterations = 10
    bar = Bar('Eval on training data',
              max=num_iterations)

    train_iterator = iter(train_dataloader)

    min_pred_depth = 9999.9
    max_pred_depth = 0.0
    min_gt_depth = 999.9
    max_gt_depth = 0.0
    l1_errors = []
    l2_errors = []
    wl1_errors = []
    wl2_errors = []
    with torch.no_grad():
      step = 100000
      weight = (torch.arange(0, args.height, device=args.device,
                             dtype=torch.float32) + 0.5) * np.pi / args.height
      weight = torch.sin(weight).view(1, args.height, 1, 1)
      for i in range(num_iterations):
        data = next(train_iterator)
        panos = data["rgb_panos"].to(args.device)
        depths = data["depth_panos"].to(args.device)
        rots = data["rots"].to(args.device)
        trans = data["trans"].to(args.device)

        run_outputs = self.run_depth_pose_carla(step, panos, depths,
                                                rots,
                                                trans)

        m_weight = weight.expand(
          panos.shape[0], args.height, args.width, 1)

        depths_small = run_outputs["depths_small"][:, 1]
        depths_pred = run_outputs["depths_pred"][:, 0]
        depths_pred = torch.clamp_min(depths_pred, 0.0)

        wl1_error = torch.abs(depths_small - depths_pred) * m_weight
        wl1_error = torch.sum(wl1_error, dim=(1, 2, 3)) / torch.sum(m_weight,
                                                                    dim=(
                                                                      1, 2, 3))
        wl1_errors.append(wl1_error.cpu().numpy())

        wl2_error = torch.pow(depths_small - depths_pred, 2.0) * m_weight
        wl2_error = torch.sum(wl2_error, dim=(1, 2, 3)) / torch.sum(m_weight,
                                                                    dim=(
                                                                      1, 2, 3))
        wl2_errors.append(wl2_error.cpu().numpy())

        l1_error = torch.mean(torch.abs(depths_small - depths_pred),
                              dim=(1, 2, 3))
        l1_errors.append(l1_error.cpu().numpy())

        l2_error = torch.mean(torch.pow(depths_small - depths_pred, 2.0),
                              dim=(1, 2, 3))
        l2_errors.append(l2_error.cpu().numpy())

        min_pred_depth = min(min_pred_depth, torch.min(depths_pred).item())
        max_pred_depth = max(max_pred_depth, torch.max(depths_pred).item())
        min_gt_depth = min(min_gt_depth, torch.min(depths_small).item())
        max_gt_depth = max(max_gt_depth, torch.max(depths_small).item())
        bar.next()
    total_l1_errors = np.mean(np.stack(l1_errors))
    total_l2_errors = np.mean(np.stack(l2_errors))
    total_wl1_errors = np.mean(np.stack(wl1_errors))
    total_wl2_errors = np.mean(np.stack(wl2_errors))

    bar.finish()

    print("Evaluation on training data:")
    print("Total l1 error:", total_l1_errors, "Weighted:", total_wl1_errors)
    print("Total l2 error:", total_l2_errors, "Weighted:", total_wl2_errors)
    print("True depth range", min_gt_depth, max_gt_depth)
    print("Pred depth range", min_pred_depth, max_pred_depth)

  def eval_on_validation_data(self):
    """Performs evaluation on the whole evaluation dataset.

    Returns:
      None
    """
    args = self.args

    self.model.eval()

    val_dataloader = DataLoader(self.val_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=0 if args.dataset == 'm3d' else 4)
    max_examples = len(self.val_data)
    if args.dataset == 'm3d':
      max_examples = min(len(self.val_data), 5000)
    if args.script_mode == 'eval_depth_test':
      bar = Bar('Eval on test data',
                max=max_examples)
      results_file = os.path.join(
        args.checkpoints_dir,
        "test_eval_results_%02f_%02f.txt" % (
          args.carla_min_dist, args.carla_max_dist))
      if args.dataset == 'm3d':
        results_file = os.path.join(
          args.checkpoints_dir,
          "test_eval_results_%02f.txt" % (
            args.m3d_dist))
      results_file = open(results_file, "w")
    else:
      bar = Bar('Eval on validation data',
                max=max_examples)
      results_file = os.path.join(
        args.checkpoints_dir,
        "val_eval_results_%02f_%02f.txt" % (
          args.carla_min_dist, args.carla_max_dist))
      if args.dataset == 'm3d':
        results_file = os.path.join(
          args.checkpoints_dir,
          "val_eval_results_%02f.txt" % (
            args.m3d_dist))
      results_file = open(results_file, "w")

    min_pred_depth = 9999.9
    max_pred_depth = 0.0
    min_gt_depth = 999.9
    max_gt_depth = 0.0
    all_errors = {}

    with torch.no_grad():
      step = 100000
      weight = (torch.arange(0, args.height, device=args.device,
                             dtype=torch.float32) + 0.5) * np.pi / args.height
      weight = torch.sin(weight).view(1, args.height, 1, 1)
      for i, data in enumerate(val_dataloader):
        if i * args.batch_size > max_examples:
          break
        panos = data["rgb_panos"].to(args.device)
        depths = data["depth_panos"].to(args.device)
        rots = data["rots"].to(args.device)
        trans = data["trans"].to(args.device)

        run_outputs = self.run_depth_pose_carla(step, panos, depths,
                                                rots,
                                                trans)

        m_weight = weight.expand(
          panos.shape[0], args.height, args.width, 1)

        depths_small = run_outputs["depths_small"][:, 1]
        depths_pred = run_outputs["depths_pred"][:, 0]
        depths_pred = torch.clamp_min(depths_pred, 0.0)

        erp_errors = self.compute_erp_depth_results(
          gt_depth=depths_small,
          pred_depth=depths_pred,
          m_weight=m_weight
        )

        cube_errors = self.compute_zdepth_results(
          gt_depth=depths[:, 1, :, :, None],
          pred_depth=depths_pred
        )

        for k, v in erp_errors.items():
          if k not in all_errors:
            all_errors[k] = []
          all_errors[k].append(v.detach().cpu().numpy())

        for k, v in cube_errors.items():
          if k not in all_errors:
            all_errors[k] = []
          all_errors[k].append(v.detach().cpu().numpy())

        # min_pred_depth = min(min_pred_depth, torch.min(depths_pred).item())
        # max_pred_depth = max(max_pred_depth, torch.max(depths_pred).item())
        # min_gt_depth = min(min_gt_depth, torch.min(depths_small).item())
        # max_gt_depth = max(max_gt_depth, torch.max(depths_small).item())

        bar.next(n=args.batch_size)

    all_errors_concatenated = {}
    for k, v in all_errors.items():
      all_errors_concatenated[k] = np.mean(np.concatenate(v))

    for k, v in all_errors_concatenated.items():
      results_file.write("%s: %0.5f\n" % (k, v))
    results_file.close()

    bar.finish()
    print("Evaluation done")

  def compute_erp_depth_results(self, gt_depth, pred_depth, m_weight):
    """Computes and returns results.

    Args:
      gt_depth: ERP GT depth.
      pred_depth: ERP predicted euclidean depth.

    Returns:
      Dictionary of torch tensors.

    """
    args = self.args
    valid_regions = torch.logical_and(torch.gt(gt_depth, 1.0),
                                      torch.lt(gt_depth, 50.0))
    valid_regions_sum = torch.sum(valid_regions, dim=(1, 2, 3))
    m_weight = m_weight * valid_regions
    m_weight_sum = torch.sum(m_weight, dim=(1, 2, 3))
    one_over_gt_depth = my_torch_helpers.safe_divide(1.0, gt_depth)
    one_over_pred_depth = my_torch_helpers.safe_divide(1.0, pred_depth)

    print("Min depth, max depth", torch.min(gt_depth), torch.max(gt_depth))
    print("Min pred depth, max pred depth", torch.min(pred_depth),
          torch.max(pred_depth))

    imae_error = torch.abs(
      one_over_gt_depth - one_over_pred_depth) * valid_regions
    if torch.any(imae_error > 100.0):
      print("max imae", torch.max(imae_error))
      big_error = (imae_error > 100.0).float()
      for i in range(pred_depth.shape[0]):
        my_torch_helpers.save_torch_image(
          os.path.join(args.checkpoints_dir, "pred_depth_%d.png" % i),
          my_torch_helpers.depth_to_turbo_colormap(
            pred_depth[i:(i + 1)],
            min_depth=args.turbo_cmap_min
          )
        )
        my_torch_helpers.save_torch_image(
          os.path.join(args.checkpoints_dir, "gt_depth_%d.png" % i),
          my_torch_helpers.depth_to_turbo_colormap(
            gt_depth[i:(i + 1)],
            min_depth=args.turbo_cmap_min
          )
        )
        my_torch_helpers.save_torch_image(
          os.path.join(args.checkpoints_dir, "error_depth_%d.png" % i),
          big_error[i:(i + 1)].expand((-1, -1, -1, 3))
        )
      raise ValueError("Error")

    imae_error = torch.abs(
      one_over_gt_depth - one_over_pred_depth) * valid_regions
    imae_error = torch.sum(imae_error, dim=(1, 2, 3)) / valid_regions_sum

    irmse_error = torch.pow(
      one_over_gt_depth - one_over_pred_depth, 2.0) * valid_regions
    irmse_error = torch.sum(irmse_error, dim=(1, 2, 3)) / valid_regions_sum
    irmse_error = torch.sqrt(irmse_error)

    l1_error = torch.abs(gt_depth - pred_depth) * valid_regions
    l1_error = torch.sum(l1_error, dim=(1, 2, 3)) / valid_regions_sum

    l2_error = torch.pow(gt_depth - pred_depth, 2.0)
    l2_error = torch.sum(l2_error, dim=(1, 2, 3)) / valid_regions_sum
    rmse_error = torch.sqrt(l2_error)

    wl1_error = torch.abs(gt_depth - pred_depth) * m_weight
    wl1_error = torch.sum(wl1_error, dim=(1, 2, 3)) / m_weight_sum

    wl2_error = torch.pow(gt_depth - pred_depth, 2.0) * m_weight
    wl2_error = torch.sum(wl2_error, dim=(1, 2, 3)) / m_weight_sum

    wrmse_error = torch.sqrt(wl2_error)

    relative_error = (torch.abs(
      gt_depth - pred_depth) / gt_depth) * valid_regions
    relative_105 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.05 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum
    relative_110 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.10 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum
    relative_125 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum
    relative_125_2 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 ** 2 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum
    relative_125_3 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 ** 3 - 1)).float(),
      dim=(1, 2, 3)) / valid_regions_sum

    return {
      'l1_error': l1_error,
      'l2_error': l2_error,
      'rmse_error': rmse_error,
      'wl1_error': wl1_error,
      'wl2_error': wl2_error,
      'wrmse_error': wrmse_error,
      'relative_105': relative_105,
      'relative_110': relative_110,
      'relative_125': relative_125,
      'relative_125_2': relative_125_2,
      'relative_125_3': relative_125_3,
      'imae_error': imae_error,
      'irmse_error': irmse_error
    }

  def compute_zdepth_results(self, gt_depth, pred_depth,
                             cubemap_sides=(2, 3, 4, 5)):
    """Computes z-depth results on the 4 cubemap sides.

    Args:
      gt_depth: ERP euclidean depth.
      pred_depth: Predicted depth.
      cubemap_sides: Which sides of the cubemap.

    Returns:
      Dictionary of torch tensors.

    """

    pred_depth_cube = []
    gt_depth_cube = []
    pred_zdepth = self.train_data.distance_to_zdepth_torch(pred_depth)
    gt_zdepth = self.train_data.distance_to_zdepth_torch(gt_depth)
    for side in cubemap_sides:
      pred_depth_cube.append(
        my_torch_helpers.equirectangular_to_cubemap(
          pred_zdepth, side=side))
      gt_depth_cube.append(
        my_torch_helpers.equirectangular_to_cubemap(
          gt_zdepth, side=side))
    pred_zdepth_cube = torch.stack(pred_depth_cube, dim=1)
    gt_zdepth_cube = torch.stack(gt_depth_cube, dim=1)

    # for i in range(pred_zdepth_cube.shape[1]):
    #   my_torch_helpers.save_torch_image(
    #     os.path.join(self.args.checkpoints_dir, "depth_%d.png" % i),
    #     my_torch_helpers.depth_to_turbo_colormap(
    #       gt_zdepth_cube[:, i], min_depth=self.args.turbo_cmap_min
    #     )
    #   )
    #   my_torch_helpers.save_torch_image(
    #     os.path.join(self.args.checkpoints_dir, "depth__%d.png" % i),
    #     my_torch_helpers.depth_to_turbo_colormap(
    #       pred_zdepth_cube[:, i], min_depth=self.args.turbo_cmap_min
    #     )
    #   )

    valid_regions = torch.logical_and(torch.gt(gt_zdepth_cube, 1.0),
                                      torch.lt(gt_zdepth_cube, 50.0))
    valid_regions_sum = torch.sum(valid_regions, dim=(1, 2, 3, 4))
    one_over_gt = my_torch_helpers.safe_divide(1.0, gt_zdepth_cube)
    one_over_pred = my_torch_helpers.safe_divide(1.0, pred_zdepth_cube)

    imae_error = torch.abs(one_over_gt - one_over_pred) * valid_regions
    imae_error = torch.sum(imae_error, dim=(1, 2, 3, 4)) / valid_regions_sum

    irmse_error = torch.pow(one_over_gt - one_over_pred, 2.0) * valid_regions
    irmse_error = torch.sum(irmse_error, dim=(1, 2, 3, 4)) / valid_regions_sum
    irmse_error = torch.sqrt(irmse_error)

    l1_error = torch.abs(gt_zdepth_cube - pred_zdepth_cube) * valid_regions
    l1_error = torch.sum(l1_error, dim=(1, 2, 3, 4)) / valid_regions_sum

    l2_error = torch.pow(gt_zdepth_cube - pred_zdepth_cube, 2.0)
    l2_error = torch.sum(l2_error, dim=(1, 2, 3, 4)) / valid_regions_sum
    rmse_error = torch.sqrt(l2_error)

    relative_error = (torch.abs(
      gt_zdepth_cube - pred_zdepth_cube) / gt_zdepth_cube) * valid_regions
    relative_105 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.05 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum
    relative_110 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.10 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum
    relative_125 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum
    relative_125_2 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 ** 2 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum
    relative_125_3 = torch.sum(
      torch.logical_and(valid_regions,
                        torch.lt(relative_error, 1.25 ** 3 - 1)).float(),
      dim=(1, 2, 3, 4)) / valid_regions_sum

    return {
      'cube_l1_error': l1_error,
      'cube_l2_error': l2_error,
      'cube_rmse_error': rmse_error,
      'cube_relative_105': relative_105,
      'cube_relative_110': relative_110,
      'cube_relative_125': relative_125,
      'cube_relative_125_2': relative_125_2,
      'cube_relative_125_3': relative_125_3,
      'cube_imae_error': imae_error,
      'cube_irmse_error': irmse_error
    }


if __name__ == "__main__":
  app = App()
  app.start()
