import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from helpers import my_torch_helpers
from models import loss_lib
from models.inpainting_unet import InpaintNet
from models.pipeline3_model import FullPipeline
from models.pointcloud_model import PointcloudModel
from renderer.SphereMeshGenerator import SphereMeshGenerator


class End2EndModel(nn.Module):
  """End to end model which combines the depth and inpainting modules together.

  """

  def __init__(self, args):
    super().__init__()
    self.mesh_generator = SphereMeshGenerator()
    self.args = args
    self.depth_model = None
    self.inpainting_model = None
    self.pointcloud_model = PointcloudModel()

    self.setup_depth_pose_model()
    self.setup_inpainting_model()

  def setup_depth_pose_model(self):
    args = self.args

    depth_model = FullPipeline(
      device=args.device,
      monodepth_model=args.model_name,
      width=args.width,
      height=args.height,
      layers=7,
      size=3,
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
      depth_type=args.depth_type,
      out_channels=2 if args.predict_meshcuts else 1
    ).to(args.device)
    self.depth_model = depth_model

  def setup_inpainting_model(self):

    args = self.args
    input_channels = 8
    output_channels = 3 if not args.use_blending else 1
    if args.script_mode in ('train_monocular', 'eval_monocular'):
      input_channels = 4
    if args.inpaint_depth:
      input_channels = input_channels + 2
      output_channels = output_channels + 1

    inpainting_model = InpaintNet(
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
      model_version=args.inpaint_model_version,
      upscale=args.inpaint_upscale
    ).to(args.device)
    self.inpainting_model = inpainting_model

  def compute_image_loss(self, inpainted_image, panos_small, panos=None):
    args = self.args
    reference_panos = panos_small
    if inpainted_image.shape != panos_small[:, 1].shape:
      reference_panos = panos if panos is not None else panos_small
      if panos is not None:
        batch_size, seq_len, panos_height, panos_width = panos.shape[:4]
        resized_panos = panos.reshape(
          (batch_size * seq_len, panos_height, panos_width, 3))
        resized_panos = my_torch_helpers.resize_torch_images(
          resized_panos, (inpainted_image.shape[2], inpainted_image.shape[1]), mode=args.interpolation_mode)
        resized_panos = resized_panos.reshape(
          batch_size, seq_len, inpainted_image.shape[1], inpainted_image.shape[2], 3)
        reference_panos = resized_panos
      else:
        raise ValueError("No panos supplied")

    if args.inpaint_loss == "l1":
      image_loss = loss_lib.compute_l1_loss(inpainted_image, reference_panos[:, 1])
    elif args.inpaint_loss == "l2":
      image_loss = loss_lib.compute_l2_loss(inpainted_image, reference_panos[:, 1])
    else:
      raise Exception('Invalid Loss')
    return image_loss

  def compute_depth_loss(self, depth_outputs0, depth_outputs1, real_depth):
    args = self.args
    if args.depth_loss == "l1_cost_volume_erp":
      assert torch.isfinite(real_depth[:, 2]).all(), "Nan in depths_small"
      rect_gt_depth = my_torch_helpers.rotate_equirectangular_image(
        real_depth[:, 2], depth_outputs1["rect_rots"][:, 1])
      one_over_gt_depth = my_torch_helpers.safe_divide(1.0, rect_gt_depth)
      depth_loss = loss_lib.compute_l1_sphere_loss(
        depth_outputs1['raw_image_features'][:, :, :, :1],
        one_over_gt_depth,
        mask=torch.gt(rect_gt_depth, 0.1))
      depth_loss = depth_loss + 0.5 * loss_lib.compute_l1_sphere_loss(
        depth_outputs1['raw_image_features_d1'],
        one_over_gt_depth,
        mask=torch.gt(rect_gt_depth, 0.1))
    else:
      raise ValueError("Unknown depth_loss")
    return depth_loss

  def forward(self, step, panos, rots, trans, depths):
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
      run_outputs["image_loss"]: Loss.


    """
    args = self.args
    mesh_generator = self.mesh_generator
    depth_model = self.depth_model
    inpainting_model = self.inpainting_model
    width = args.width
    height = args.height

    batch_size, seq_len, panos_height, panos_width = panos.shape[:4]

    panos_small = panos.reshape(
      (batch_size * seq_len, panos_height, panos_width, 3))
    panos_small = my_torch_helpers.resize_torch_images(
      panos_small, (args.width, args.height), mode=args.interpolation_mode)
    panos_small = panos_small.reshape(batch_size, seq_len, height, width, 3)

    depths_height, depths_width = depths.shape[2:4]
    depths_small = depths.reshape(
      (batch_size * seq_len, depths_height, depths_width, 1))
    depths_small = my_torch_helpers.resize_torch_images(
      depths_small, (args.width, args.height), mode=args.interpolation_mode)
    depths_small = depths_small.reshape(batch_size, seq_len, height, width, 1)

    # rotated_panos_small = []
    # for i in range(panos.shape[1]):
    #   rotated_pano = my_torch_helpers.rotate_equirectangular_image(
    #     panos[:, i],
    #     rots[:, i])
    #   rotated_panos_small.append(
    #     my_torch_helpers.resize_torch_images(
    #       rotated_pano,
    #       (args.width, args.height),
    #       mode=args.interpolation_mode)
    #   )
    # rotated_panos_small = torch.stack(rotated_panos_small, dim=1)
    #
    # rotated_depths_small = []
    # for i in range(panos.shape[1]):
    #   rotated_pano = my_torch_helpers.rotate_equirectangular_image(
    #     depths[:, i, :, :, None],
    #     rots[:, i])
    #   rotated_depths_small.append(
    #     my_torch_helpers.resize_torch_images(
    #       rotated_pano,
    #       (args.width, args.height),
    #       mode=args.interpolation_mode)
    #   )
    # rotated_depths_small = torch.stack(rotated_depths_small, dim=1)

    assert args.cost_volume, "Not using cost volume"
    if args.cost_volume:
      depthnet_outputs0 = depth_model.estimate_depth_using_cost_volume(
        panos_small[:, [2, 0], :, :, :],
        rots[:, [2, 0]],
        trans[:, [2, 0]],
        min_depth=args.min_depth,
        max_depth=args.max_depth)
      depthnet_outputs2 = depth_model.estimate_depth_using_cost_volume(
        panos_small[:, [0, 2], :, :, :],
        rots[:, [0, 2]],
        trans[:, [0, 2]],
        min_depth=args.min_depth,
        max_depth=args.max_depth)
      depths_pred = torch.stack(
        (depthnet_outputs0["depth"], depthnet_outputs2["depth"], depthnet_outputs2["depth"]),
        dim=1)
    else:
      depthnet_outputs0 = depth_model.estimate_depth(panos_small[:, 0, :, :, :])
      depthnet_outputs2 = depth_model.estimate_depth(panos_small[:, 2, :, :, :])
      depths_pred = torch.stack(
        (depthnet_outputs0["depth"], depthnet_outputs2["depth"], depthnet_outputs2["depth"]),
        dim=1)

    if args.threshold_depth and args.clamp_depth_to > 65:
      # depths_pred = depths_pred.clone()
      depths_pred[depths_pred > 65] = args.clamp_depth_to

    if args.representation == 'mesh':
      depth_mask0 = None
      depth_mask2 = None
      if args.predict_meshcuts:
        depth_mask0 = mesh_generator.logits_to_depth_mask(
          depthnet_outputs0["raw_image_features"][:, :, :, 1:2],
          width=args.mesh_width,
          height=args.mesh_height)
        depth_mask2 = mesh_generator.logits_to_depth_mask(
          depthnet_outputs2["raw_image_features"][:, :, :, 1:2],
          width=args.mesh_width,
          height=args.mesh_height)
      meshes_0 = mesh_generator.generate_mesh(
        depths_pred[:, 0],
        panos[:, 0],
        apply_depth_mask=True,
        facing_inside=True,
        width_segments=args.mesh_width,
        height_segments=args.mesh_height,
        mesh_removal_threshold=args.mesh_removal_threshold,
        depth_mask=depth_mask0)
      meshes_2 = mesh_generator.generate_mesh(
        depths_pred[:, 2],
        panos[:, 2],
        apply_depth_mask=True,
        facing_inside=True,
        width_segments=args.mesh_width,
        height_segments=args.mesh_height,
        mesh_removal_threshold=args.mesh_removal_threshold,
        depth_mask=depth_mask2)

      meshes_0 = mesh_generator.apply_rot_trans(meshes_0,
                                                rots[:, 0],
                                                trans[:, 0],
                                                inv_rot=True)
      meshes_2 = mesh_generator.apply_rot_trans(meshes_2,
                                                rots[:, 2],
                                                trans[:, 2],
                                                inv_rot=True)

      output_0, depth_image_0 = mesh_generator.render_mesh(meshes_0, image_size=width)
      output_2, depth_image_2 = mesh_generator.render_mesh(meshes_2, image_size=width)
    elif args.representation == 'pointcloud':
      pointcloud0 = self.pointcloud_model.make_point_cloud(
        depths=depths_pred[:, 0],
        images=panos_small[:, 0],
        rots=rots[:, 0],
        trans=trans[:, 0],
        inv_rot_trans=True)
      pointcloud1 = self.pointcloud_model.make_point_cloud(
        depths=depths_pred[:, 2],
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
    inpainted_image = inpainting_model(concatenated_output)
    inpainted_depth = None
    if args.inpaint_depth:
      inpainted_depth = (1.0 / inpaint_depth_scale) * torch.exp(
        inpainted_image[..., -1:])
      inpainted_image = inpainted_image[..., :-1]

    if args.use_blending:
      inpainted_image = (1 - inpainted_image) * output_0 + \
                        inpainted_image * output_2

    image_loss = self.compute_image_loss(inpainted_image, panos_small, panos=panos)
    depth_loss = self.compute_depth_loss(depthnet_outputs0, depthnet_outputs2, depths_small)
    final_loss = image_loss + depth_loss

    return {
      "final_loss": final_loss,
      "image_loss": image_loss,
      "depth_loss": depth_loss,
      "panos_small": panos_small,
      "depths_small": depths_small,
      "pred_depth_0": depthnet_outputs0["depth"],
      "pred_depth_2": depthnet_outputs2["depth"],
      "output_0": output_0,
      "output_2": output_2,
      "depth_image_0": depth_image_0,
      "depth_image_2": depth_image_2,
      "inpainted_image": inpainted_image,
      "inpainted_depth": inpainted_depth
    }
