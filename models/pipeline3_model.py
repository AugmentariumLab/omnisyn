# Lint as: python3
"""Encapsulates the full pipeline as a single torch module.
"""

import os
import sys

import numpy as np
import torch
from pytorch3d.renderer import (OpenGLPerspectiveCameras,
                                PointsRasterizationSettings, AlphaCompositor)
from pytorch3d.structures import Pointclouds
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F

from helpers import my_helpers
from helpers import my_torch_helpers
from models.common_blocks import (ConvBlock, Conv3DBlock, Conv3DBlockv2,
                                  ConvBlock2, UNet2)
from models.inpainting_unet import InpaintNet
from renderer.SpherePointsRasterizer import SpherePointsRasterizer
from renderer.SpherePointsRenderer import SpherePointsRenderer

# Monodepth2 Imports
sys.path.append('monodepth2')
from monodepth2 import networks
from monodepth2.layers import disp_to_depth, rot_from_axisangle


class FullPipeline(nn.Module):
  """Full pipeline 3.
  """

  def __init__(self,
               device=torch.device("cuda"),
               monodepth_model="mono+stereo_1024x320",
               raster_resolution=256,
               width=256,
               height=256,
               layers=7,
               min_depth=0.1,
               max_depth=100.0,
               verbose=True,
               point_radius=0.004,
               points_per_pixel=8,
               depth_input_images=1,
               depth_output_channels=1,
               include_inpaintnet=False,
               include_poseestimator=False,
               input_uv=False,
               interpolation_mode="bilinear",
               cost_volume=False,
               use_v_input=False,
               depth_type="one_over",
               size=4,
               out_channels=1,
               **kwargs):
    """Initializes the pipeline.

    Args:
      device: Cuda device.
      monodepth_model: Name of the monodepth2 model. None for custom UNet.
      raster_resolution: Resolution of point cloud rendering.
      width: Width for custom unet.
      height: Height for custom unet.
      layers: Layers for the custom unet.
      verbose: Print debug statements.
      min_depth: Minimum depth.
      max_depth: Maximum depth.
      point_radius: Radius for point cloud renderer.
      points_per_pixel: Pointer per pixel for point cloud renderer.
      depth_input_images: Number of input images for the depth.
      depth_output_channels: Number of output images for the depth.
      include_inpaintnet: Include inpainting network.
      include_poseestimator: Include pose estimation.
      input_uv: Input UVs.
      interpolation_mode: Either bilinear or nearest.
      cost_volume: Use cost volume for depth estimation (requires 2 input images).
      use_v_input: Input v of uv coordinates into each conv block.
      **kwargs:
    """
    super().__init__()

    self.device = device
    self.verbose = verbose
    self.monodepth_model = monodepth_model
    self.width = width
    self.height = height
    self.layers = layers
    self.input_uv = input_uv
    self.min_depth = min_depth
    self.max_depth = max_depth
    self.interpolation_mode = interpolation_mode
    self.align_corners = False if interpolation_mode == "bilinear" else None
    self.using_monodepth_model = monodepth_model == "mono+stereo_1024x320"
    self.omega = None
    self.use_v_input = use_v_input
    self.depth_type = depth_type

    if isinstance(cost_volume, str):
      cost_volume = cost_volume.lower()
    if cost_volume == "true":
      cost_volume = "v1"
    self.use_cost_volume = cost_volume

    if self.using_monodepth_model:
      # Load monodepth2 model.
      load_state = depth_input_images == 1 \
                   and depth_output_channels == 1 \
                   and not input_uv
      print("Load state", load_state)
      model_path = os.path.join("models", monodepth_model)
      if verbose:
        print("Loading Monodepth2 from", model_path)
      encoder_path = os.path.join(model_path, "encoder.pth")
      depth_decoder_path = os.path.join(model_path, "depth.pth")
      if input_uv:
        depth_input_images = depth_input_images + 1
      monodepth_encoder = networks.ResnetEncoder(
        18, not load_state, num_input_images=depth_input_images)
      loaded_dict_enc = torch.load(encoder_path, map_location=device)
      self.monodepth_feed_height = loaded_dict_enc['height']
      self.monodepth_feed_width = loaded_dict_enc['width']
      filtered_dict_enc = {
        k: v
        for k, v in loaded_dict_enc.items()
        if k in monodepth_encoder.state_dict()
      }
      if load_state:
        monodepth_encoder.load_state_dict(filtered_dict_enc)
      monodepth_encoder.to(device)
      # monodepth_encoder.eval()
      depth_decoder = networks.DepthDecoder(
        num_ch_enc=monodepth_encoder.num_ch_enc,
        scales=range(4),
        num_output_channels=depth_output_channels)
      loaded_dict = torch.load(depth_decoder_path, map_location=device)
      if load_state:
        depth_decoder.load_state_dict(loaded_dict)
      depth_decoder.to(device)
      # depth_decoder.eval()
      self.monodepth_encoder = monodepth_encoder
      self.monodepth_decoder = depth_decoder
    elif cost_volume == "v1" or cost_volume == "true":
      print("Using cost volume v1")
      encoders, decoders, final_conv, cv_layers = \
        self.initialize_cost_volume_network()
      self.encoders = encoders
      self.decoders = decoders
      self.final_conv = final_conv
      self.cv_layers = cv_layers
    elif cost_volume in ['v2', 'v2_cubemap']:
      print("Using cost volume v2")
      unet, cv_layers, decoders2 = self.initialize_cost_volume_network_v2(
        use_wrap_padding=(cost_volume == "v2"), use_v_input=use_v_input
      )
      self.unet = unet
      self.cv_layers = cv_layers
      self.decoders2 = decoders2
      if verbose:
        unet_params = my_torch_helpers.total_params(unet)
        cv_params = my_torch_helpers.total_params(cv_layers)
        decoders2_params = my_torch_helpers.total_params(decoders2)
        total_params = unet_params + cv_params + decoders2_params
        print("Total depth params: %d" % (total_params,))
    elif cost_volume in ['v3', 'v3_cylindrical', 'v3_erp']:
      print('Using cost volume v3')
      unet, unet3d, decoders1, decoders2 = self.initialize_cost_volume_network_v3(
        layers=layers, size=size,
        use_wrap_padding=True, use_v_input=use_v_input,
        out_channels=out_channels
      )
      self.unet = unet
      self.unet3d = unet3d
      self.decoders1 = decoders1
      self.decoders2 = decoders2
      if verbose:
        unet_params = my_torch_helpers.total_params(unet)
        cv_params = my_torch_helpers.total_params(unet3d)
        decoders2_params = my_torch_helpers.total_params(decoders2)
        total_params = unet_params + cv_params + decoders2_params
        print("Total depth params: %d" % (total_params,))
    else:
      print("Using UNet instead of monodepth2 depth model")
      input_channels = 3
      if input_uv:
        input_channels = input_channels + 3
      encoders, decoders, final_conv = self.initialize_unet(
        input_channels,
        1,
        layers=layers,
        use_batchnorm=True,
        use_wrap_padding=True,
        gate=False)
      self.encoders = encoders
      self.decoders = decoders
      self.final_conv = final_conv

    # Save rendering settings
    self.raster_resolution = raster_resolution
    self.point_radius = point_radius
    self.points_per_pixel = points_per_pixel

    self.include_inpaintnet = include_inpaintnet
    if include_inpaintnet:
      self.inpaint_net = InpaintNet(input_channels=3,
                                    output_channels=3,
                                    layers=7,
                                    width=256,
                                    height=256,
                                    device=device,
                                    gate=True)

    self.include_poseestimator = include_poseestimator
    if include_poseestimator:
      model_path = os.path.join("models", "mono+stereo_1024x320")
      pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
      pose_decoder_path = os.path.join(model_path, "pose.pth")

      self.pose_encoder = networks.ResnetEncoder(18, False, 2)
      self.pose_encoder.load_state_dict(torch.load(pose_encoder_path))
      self.pose_encoder.to(device)

      self.pose_decoder = networks.PoseDecoder(self.pose_encoder.num_ch_enc, 1,
                                               2)
      self.pose_decoder.load_state_dict(torch.load(pose_decoder_path))
      self.pose_decoder.to(device)

    if input_uv:
      uv_width = width
      uv_height = height
      if self.using_monodepth_model:
        uv_width = self.monodepth_feed_width
        uv_height = self.monodepth_feed_height
      u = torch.arange(0, uv_width, device=device, dtype=torch.float32)
      u = (u + 0.5) / uv_width
      v = torch.arange(0, uv_height, device=device, dtype=torch.float32)
      v = (v + 0.5) / uv_height
      u, v = torch.meshgrid(v, u)
      z = torch.tensor(0, device=device, dtype=torch.float32)
      z = z.expand_as(u)
      self.input_uv_cache = torch.stack((u, v, z), dim=0)[None, ...]
      # if self.verbose:
      #   print("Input uv cache shape", self.input_uv_cache.shape)

  def initialize_unet(self,
                      input_channels=3,
                      output_channels=1,
                      layers=7,
                      use_batchnorm=True,
                      use_wrap_padding=True,
                      gate=False):
    """Initializes a Unet.

    Args:
      input_channels: Input channels.
      output_channels: Output channels.
      use_batchnorm: Batch norm.
      use_wrap_padding:
      gate:

    Returns:

    """
    encoders = [
      ConvBlock(in_channels=input_channels,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                upscale=False,
                use_batch_norm=use_batchnorm,
                use_wrap_padding=use_wrap_padding,
                gate=gate)
    ]
    decoders = [
      ConvBlock(in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                use_batch_norm=use_batchnorm,
                use_wrap_padding=use_wrap_padding,
                upscale=True)
    ]

    for i in range(1, layers):
      channels = 2 ** (i + 3)
      encoders.append(
        ConvBlock(in_channels=channels,
                  out_channels=2 * channels,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  use_batch_norm=use_batchnorm,
                  use_wrap_padding=use_wrap_padding,
                  upscale=False,
                  gate=gate))
      decoders.append(
        ConvBlock(in_channels=4 * channels,
                  out_channels=channels,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  use_batch_norm=use_batchnorm,
                  use_wrap_padding=use_wrap_padding,
                  upscale=True))

    encoders = nn.ModuleList(encoders)
    decoders = nn.ModuleList(decoders)
    final_conv = ConvBlock(in_channels=16,
                           out_channels=output_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           use_batch_norm=False,
                           use_wrap_padding=use_wrap_padding,
                           use_activation=False,
                           upscale=False)
    return encoders, decoders, final_conv

  def initialize_cost_volume_network(self, layers=2, size=5):
    """Initilizes a cost volume network.

    Args:
      layers: Number of layers. Unused for now.
      size: Size of the network. Each number multiplies the size by 2.

    Returns:
      Encoders, Decoders, and a final convolution layer.

    """
    use_batchnorm = True
    use_wrap_padding = True
    gate = False
    encoders = [
      ConvBlock(in_channels=3,
                out_channels=2 ** size,
                kernel_size=4,
                stride=2,
                padding=1,
                upscale=False,
                use_batch_norm=use_batchnorm,
                use_wrap_padding=use_wrap_padding,
                gate=gate)
    ]
    decoders = [
      ConvBlock(in_channels=2 ** 5,
                out_channels=2 ** 4,
                kernel_size=3,
                stride=1,
                padding=1,
                use_batch_norm=use_batchnorm,
                use_wrap_padding=use_wrap_padding,
                upscale=True)
    ]

    for i in range(1, layers):
      channels = 2 ** (i + size - 1)
      encoders.append(
        ConvBlock(in_channels=channels,
                  out_channels=2 * channels,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  use_batch_norm=use_batchnorm,
                  use_wrap_padding=use_wrap_padding,
                  upscale=False,
                  gate=gate))
    for i in range(1, 2):
      dchannels = 2 ** (i + 4)
      decoders.append(
        ConvBlock(in_channels=2 * dchannels,
                  out_channels=dchannels,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  use_batch_norm=use_batchnorm,
                  use_wrap_padding=use_wrap_padding,
                  upscale=True))

    # The code below reduces the channel dimension to 1.
    cv_layers = []
    for i in range(size + 1):
      cv_layers.append(
        Conv3DBlock(in_channels=2 ** (size + 1 - i),
                    out_channels=2 ** (size - i),
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                    use_wrap_padding=use_wrap_padding))

    cv_layers = nn.ModuleList(cv_layers)
    encoders = nn.ModuleList(encoders)
    decoders = nn.ModuleList(decoders)
    final_conv = ConvBlock(in_channels=2 ** 4,
                           out_channels=1,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           use_batch_norm=False,
                           use_wrap_padding=use_wrap_padding,
                           use_activation=False,
                           upscale=False)
    return encoders, decoders, final_conv, cv_layers

  def initialize_cost_volume_network_v2(self, layers=5, size=4,
                                        use_wrap_padding=True,
                                        use_v_input=False):
    """Initilizes a cost volume network.

    This initializes v2 of our cost volume network.
    The primary difference is that everything is a UNet now.
    Also we use UNet conv blocks which are conv-lrelu-conv-lrelu-pool.

    Args:
      layers: Layers.
      size: Size of channels.
      use_wrap_padding: Use wrap padding.
      use_v_input: Use v input.

    Returns:
      None.

    """
    encoders = [
      ConvBlock2(in_channels=3,
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 upscale=False,
                 use_wrap_padding=use_wrap_padding,
                 use_residual=False,
                 use_v_input=use_v_input)
    ]
    decoders = [None]

    for i in range(1, layers):
      channels = 2 ** (i + size - 1)
      encoders.append(
        ConvBlock2(in_channels=channels,
                   out_channels=2 * channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False,
                   use_residual=False,
                   use_v_input=use_v_input))
      if i > 1:
        decoders.append(
          ConvBlock2(in_channels=4 * channels,
                     out_channels=channels,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     use_wrap_padding=use_wrap_padding,
                     pooling=nn.Identity(),
                     upscale=False,
                     use_residual=False,
                     use_v_input=use_v_input))
      else:
        decoders.append(None)

    encoders.append(
      ConvBlock2(in_channels=2 ** (layers + size - 1),
                 out_channels=2 ** (layers + size),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 use_residual=False,
                 pooling=nn.Identity(),
                 use_v_input=use_v_input))

    # The code below reduces the channel dimension to 1.
    cv_layers = []
    for i in range(0, size + 1):
      cv_layers.append(
        Conv3DBlock(in_channels=2 ** (size + 1 - i),
                    out_channels=2 ** (size - i),
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                    use_batch_norm=False,
                    use_wrap_padding=use_wrap_padding,
                    use_v_input=use_v_input))

    decoders2 = [
      ConvBlock2(in_channels=64 + 32,
                 out_channels=32,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=True,
                 upscale=True,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
      ConvBlock2(in_channels=32,
                 out_channels=16,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=True,
                 upscale=True,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
      ConvBlock2(in_channels=16,
                 out_channels=1,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=False,
                 upscale=False,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
    ]

    encoders = nn.ModuleList(encoders)
    decoders = nn.ModuleList(decoders)
    unet = UNet2(encoders, decoders)
    cv_layers = nn.ModuleList(cv_layers)
    decoders2 = nn.ModuleList(decoders2)
    return unet, cv_layers, decoders2

  def initialize_cost_volume_network_v3(self, layers=5, size=4,
                                        use_wrap_padding=True,
                                        use_v_input=False,
                                        out_channels=1):
    """Initilizes a cost volume network.

    This initializes v2 of our cost volume network.
    The primary difference is that everything is a UNet now.
    Also we use UNet conv blocks which are conv-lrelu-conv-lrelu-pool.

    Args:
      layers: Layers.
      size: Size of channels.
      use_wrap_padding: Use wrap padding.
      use_v_input: Use v input.

    Returns:
      None.

    """
    encoders = [
      ConvBlock2(in_channels=3,
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 upscale=False,
                 use_wrap_padding=use_wrap_padding,
                 use_residual=False,
                 use_v_input=use_v_input)
    ]
    decoders = [None]

    for i in range(1, layers):
      channels = 2 ** (i + size - 1)
      encoders.append(
        ConvBlock2(in_channels=channels,
                   out_channels=2 * channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False,
                   use_residual=False,
                   use_v_input=use_v_input))
      if i > 1:
        decoders.append(
          ConvBlock2(in_channels=4 * channels,
                     out_channels=channels,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     use_wrap_padding=use_wrap_padding,
                     pooling=nn.Identity(),
                     upscale=False,
                     use_residual=False,
                     use_v_input=use_v_input))
      else:
        decoders.append(None)

    encoders.append(
      ConvBlock2(in_channels=2 ** (layers + size - 1),
                 out_channels=2 ** (layers + size),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 use_residual=False,
                 pooling=nn.Identity(),
                 use_v_input=use_v_input))

    # The code below reduces the channel dimension to 1.
    cv_encoders = []
    cv_decoders = [
      Conv3DBlockv2(in_channels=2 ** (size + 3),
                    out_channels=1,
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                    use_batch_norm=False,
                    use_wrap_padding=use_wrap_padding,
                    pooling=nn.Identity(),
                    use_v_input=use_v_input)
    ]
    for i in range(0, 3):
      channels = 2 ** (i + size + 1)
      cv_encoders.append(
        Conv3DBlockv2(in_channels=channels,
                      out_channels=2 * channels,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      use_batch_norm=False,
                      use_wrap_padding=use_wrap_padding,
                      use_v_input=use_v_input))
      if i > 0:
        cv_decoders.append(
          Conv3DBlockv2(in_channels=4 * channels,
                        out_channels=channels,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        use_batch_norm=False,
                        use_wrap_padding=use_wrap_padding,
                        pooling=nn.Identity(),
                        use_v_input=use_v_input))

    cv_encoders.append(
      Conv3DBlockv2(in_channels=2 ** (3 + size + 1),
                    out_channels=2 ** (3 + size + 2),
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                    use_batch_norm=False,
                    pooling=nn.Identity(),
                    use_wrap_padding=use_wrap_padding,
                    use_v_input=use_v_input))

    decoders1 = ConvBlock(
      64,
      1,
      kernel_size=1,
      padding=0,
      stride=1,
      upscale=False,
      gate=False,
      use_wrap_padding=False,
      use_batch_norm=False,
      use_activation=False,
    )

    decoders2 = [
      ConvBlock2(in_channels=64 + 2 ** (size + 1),
                 out_channels=2 ** (size + 1),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=True,
                 upscale=True,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
      ConvBlock2(in_channels=2 ** (size + 1),
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=True,
                 upscale=True,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
      ConvBlock2(in_channels=2 ** size,
                 out_channels=out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=False,
                 upscale=False,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
    ]

    unet = UNet2(nn.ModuleList(encoders),
                 nn.ModuleList(decoders))
    unet3d = UNet2(nn.ModuleList(cv_encoders),
                   nn.ModuleList(cv_decoders),
                   interpolation="trilinear",
                   name="unet3d")
    decoders2 = nn.ModuleList(decoders2)
    return unet, unet3d, decoders1, decoders2

  def forward_thru_unet(self, input):
    """Forward pass through the UNet.

    Args:
      input: Input as channels first tensor.

    Returns:
      Output of the forward pass as a channels first tensor.

    """
    x = input
    x_all = []
    # Encode.
    for i in range(self.layers):
      x = self.encoders[i](x)
      x_all.append(x)

    # Decode with skip connections.
    for i in range(self.layers - 1, -1, -1):
      x = torch.cat((x, x_all[i]), dim=1)
      x = self.decoders[i](x)
    x = self.final_conv(x)
    return x

  def estimate_depth(self, input_images, reshape_depth=True, loss="l1"):
    """Estimates depth using monodepth2.

    Args:
      input_images: Input images channels last.
      reshape_depth: Whether to reshape the depth output.
      loss: Loss.

    Returns:
      Depth output as BxHxWxC tensor.

    """
    batch_size, original_height, original_width = input_images.shape[:3]
    # Change to channels first
    input_images = input_images.permute((0, 3, 1, 2))

    if self.using_monodepth_model:
      shrinked_input_images = torch.nn.functional.interpolate(
        input_images, (self.monodepth_feed_height, self.monodepth_feed_width),
        mode=self.interpolation_mode,
        align_corners=self.align_corners)

      if self.input_uv:
        input_uvs = self.input_uv_cache
        input_uvs = input_uvs.expand((batch_size, 3, self.monodepth_feed_height,
                                      self.monodepth_feed_width))
        shrinked_input_images = torch.cat((shrinked_input_images, input_uvs),
                                          dim=1)

      features = self.monodepth_encoder(shrinked_input_images)
      outputs = self.monodepth_decoder(features)

      disp = outputs[("disp", 0)]
      if reshape_depth:
        disp = torch.nn.functional.interpolate(
          disp, (original_height, original_width),
          mode=self.interpolation_mode,
          align_corners=self.align_corners)
      scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

      # Change to channels last
      depth = depth.permute((0, 2, 3, 1))

      return {"depth": depth}
    elif self.use_cost_volume:
      print("Using cost volume; exiting")
      sys.exit(0)
    else:
      shrinked_input_images = torch.nn.functional.interpolate(
        input_images, (self.height, self.width),
        mode=self.interpolation_mode,
        align_corners=self.align_corners)
      if self.input_uv:
        input_uvs = self.input_uv_cache
        input_uvs = input_uvs.expand((batch_size, 3, self.height, self.width))
        shrinked_input_images = torch.cat((shrinked_input_images, input_uvs),
                                          dim=1)

      output = self.forward_thru_unet(shrinked_input_images)
      output = output.permute((0, 2, 3, 1))
      # depth = torch.exp(output)
      disparity = torch.sigmoid(output)
      min_disp = 1 / 100
      max_disp = 1 / 0.1
      disparity = min_disp + (max_disp - min_disp) * disparity
      depth = 1 / disparity

      return {"depth": depth, "output": output}

  def estimate_depth_using_cost_volume(self, panos, rots, trans,
                                       min_depth=2, max_depth=100):
    """Estimates depth using cost volume either v1 or v2.

    Args:
      panos: Panos.
      rots: Trans.
      trans: Rots.
      min_depth: Minimum depth.
      max_depth: Maximum depth.

    Returns:
      Predicted depth.
    """
    if self.use_cost_volume == "v1":
      return self.estimate_depth_using_cost_volume_v1(panos, rots, trans)
    elif self.use_cost_volume == "v2":
      return self.estimate_depth_using_cost_volume_v2(panos, rots, trans)
    elif self.use_cost_volume == "v2_cubemap":
      return self.estimate_depth_using_cost_volume_v2_cubemap(
        panos, rots, trans)
    elif self.use_cost_volume == "v3_cylindrical":
      return self.estimate_depth_using_cost_volume_v3_cylindrical(
        panos, rots, trans)
    elif self.use_cost_volume == "v3":
      return self.estimate_depth_using_cost_volume_v3(panos, rots, trans)
    elif self.use_cost_volume == "v3_erp":
      return self.estimate_depth_using_cost_volume_v3_erp(panos, rots, trans,
                                                          min_depth=min_depth,
                                                          max_depth=max_depth)
    return None

  def estimate_depth_using_cost_volume_v1(self, panos, rots, trans):
    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """
    batch_size, _, height, width, _ = panos.shape
    rectified_panos, rect_rots = self.rectify_images(panos, rots, trans)

    assert torch.isfinite(panos).all(), "Nan in panos"
    assert torch.isfinite(rots).all(), "Nan in rots"
    assert torch.isfinite(trans).all(), "Nan in trans"

    # Change to channels first
    image_features_0 = rectified_panos[:, 0].permute((0, 3, 1, 2))
    image_features_1 = rectified_panos[:, 1].permute((0, 3, 1, 2))
    for i, block in enumerate(self.encoders):
      # print("Image features shape0", image_features_0.shape)
      image_features_0 = block(image_features_0)
      image_features_1 = block(image_features_1)
    # print("Image features shape0", image_features_0.shape)
    # Change to channels last
    image_features_0 = image_features_0.permute((0, 2, 3, 1))
    image_features_1 = image_features_1.permute((0, 2, 3, 1))
    image_features = torch.stack((image_features_0, image_features_1), dim=1)
    cost_volume = self.calculate_cost_volume(image_features,
                                             stride=0.5,
                                             slide_range=0.5,
                                             cost_type="abs_diff")

    assert torch.isfinite(cost_volume).all(), "Nan in cost volume"

    # Change to channels first
    cost_volume = cost_volume.permute((0, 4, 1, 2, 3))
    for i, block in enumerate(self.cv_layers):
      # print("Cost volume shape1", cost_volume.shape)
      cost_volume = block(cost_volume)
    # print("Cost volume shape2", cost_volume.shape)
    # Now the length dimension is now the channel dimension.
    # Image_features is channels-first
    image_features = cost_volume[:, 0, :, :, :]
    # image_features = cost_volume[:, :, 0, :, :]
    for i in range(len(self.decoders) - 1, -1, -1):
      # print("Image features shape1", image_features.shape)
      image_features = self.decoders[i](image_features)
    # print("Image features shape2", image_features.shape)
    image_features = self.final_conv(image_features)

    assert torch.isfinite(image_features).all(), "Nan in image features"

    trans_norm = torch.norm(trans[:, 0] + trans[:, 1], dim=1)
    depth = self.erp_disparity_to_depth(image_features, trans_norm)
    depth = depth.permute((0, 2, 3, 1))
    depth = self.unrectify_image(depth, rect_rots[:, 1])

    # Change to channels last.
    # print("Final depth shape", depth.shape)
    # sys.exit(0)
    return {
      "depth": depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "rectified_panos": rectified_panos,
      "rect_rots": rect_rots,
      "trans_norm": trans_norm
    }

  def estimate_depth_using_cost_volume_v2(self, panos, rots, trans):
    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """
    batch_size, channels, height, width, _ = panos.shape
    rectified_panos, rect_rots = self.rectify_images(panos, rots, trans)

    # Change to channels first.
    rectified_panos_cf = rectified_panos.permute((0, 1, 4, 2, 3))
    image_features_0_cf = rectified_panos_cf[:, 0]
    image_features_1_cf = rectified_panos_cf[:, 1]
    # print("Image features shape", image_features_0.shape)
    # Pass through encoding UNet.
    image_features_0_cf = self.unet(image_features_0_cf)
    image_features_1_cf = self.unet(image_features_1_cf)
    # Change to channels last.
    image_features_0 = image_features_0_cf.permute((0, 2, 3, 1))
    image_features_1 = image_features_1_cf.permute((0, 2, 3, 1))

    # Create cost volume.
    image_features = torch.stack((image_features_0, image_features_1), dim=1)
    cost_volume = self.calculate_cost_volume(image_features,
                                             stride=0.5,
                                             slide_range=0.5,
                                             cost_type="abs_diff",
                                             direction="up")

    # Pass cost volume through Conv3D layers.
    cost_volume = cost_volume.permute((0, 4, 1, 2, 3))
    for i, block in enumerate(self.cv_layers):
      # print("Cost volume shape1", cost_volume.shape)
      cost_volume = block(cost_volume)
    # print("Cost volume shape2", cost_volume.shape)

    # Now the length dimension is now the channel dimension.
    image_features = cost_volume[:, 0, :, :, :]
    image_features = torch.cat((image_features, image_features_1_cf), dim=1)
    for i in range(len(self.decoders2)):
      # print("Image features shape1", image_features.shape)
      image_features, _ = self.decoders2[i](image_features)
    # print("Image features shape2", image_features.shape)

    assert torch.isfinite(image_features).all(), "Nan in image features"

    trans_norm = torch.norm(trans[:, 0] + trans[:, 1], dim=1)
    depth = self.erp_disparity_to_depth(image_features, trans_norm)
    depth = depth.permute((0, 2, 3, 1))
    depth = self.unrectify_image(depth, rect_rots[:, 1])

    unrect_disp = self.unrectify_image(image_features.permute((0, 2, 3, 1)),
                                       rect_rots[:, 1])

    # Change to channels last.
    # print("Final depth shape", depth.shape)
    # sys.exit(0)
    return {
      "depth": depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "rectified_panos": rectified_panos,
      "unrect_disp": unrect_disp,
      "rect_rots": rect_rots,
      "trans_norm": trans_norm
    }

  def estimate_depth_using_cost_volume_v2_cubemap(self, panos, rots, trans):
    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """
    batch_size, _, height, width, _ = panos.shape
    panos_0_rot = my_torch_helpers.rotate_equirectangular_image(panos[:, 0],
                                                                rots[:, 0])
    panos_c0 = my_torch_helpers.equirectangular_to_cubemap(panos_0_rot, side=2)
    panos_c1 = my_torch_helpers.equirectangular_to_cubemap(panos[:, 1], side=2)
    rectified_panos = torch.stack((panos_c0, panos_c1), dim=1)

    # Change to channels first.
    rectified_panos_cf = rectified_panos.permute((0, 1, 4, 2, 3))
    image_features_0_cf = rectified_panos_cf[:, 0]
    image_features_1_cf = rectified_panos_cf[:, 1]
    # print("Image features shape", image_features_0.shape)
    # Pass through encoding UNet.
    image_features_0_cf = self.unet(image_features_0_cf)
    image_features_1_cf = self.unet(image_features_1_cf)
    # Change to channels last.
    image_features_0 = image_features_0_cf.permute((0, 2, 3, 1))
    image_features_1 = image_features_1_cf.permute((0, 2, 3, 1))

    # Create cost volume.
    image_features = torch.stack((image_features_0, image_features_1), dim=1)
    first_cost_volume = self.calculate_cost_volume(image_features,
                                                   stride=1.0,
                                                   slide_range=1.0,
                                                   cost_type="abs_diff",
                                                   direction="left")

    assert torch.isfinite(first_cost_volume).all(), "Nan in cost volume"

    # Pass cost volume through Conv3D layers.
    cost_volume = first_cost_volume.permute((0, 4, 1, 2, 3))
    for i, block in enumerate(self.cv_layers):
      # print("Cost volume shape1", cost_volume.shape)
      cost_volume = block(cost_volume)
    # print("Cost volume shape2", cost_volume.shape)

    # Now the length dimension is now the channel dimension.
    # image_features = cost_volume[:, 0, :, :, :]
    image_features = torch.cat(
      (cost_volume[:, 0, :, :, :], image_features_1_cf),
      dim=1)
    for i in range(len(self.decoders2)):
      # print("Image features shape1", image_features.shape)
      image_features, _ = self.decoders2[i](image_features)
    # print("Image features shape2", image_features.shape)

    assert torch.isfinite(image_features).all(), "Nan in image features"

    trans_norm = torch.norm(trans[:, 0] + trans[:, 1], dim=1)

    depth = my_torch_helpers.safe_divide(trans_norm.view(batch_size, 1, 1, 1),
                                         image_features)
    depth = depth.permute((0, 2, 3, 1))

    return {
      "depth": depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "rectified_panos": rectified_panos,
      "trans_norm": trans_norm,
      "first_cost_volume": first_cost_volume,
      "panos_c0": panos_c0,
      "panos_c1": panos_c1,
      "image_features_0": image_features_0,
      "image_features_1": image_features_1
    }

  def estimate_depth_using_cost_volume_v3(self, panos, rots, trans):
    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """
    batch_size, channels, height, width, _ = panos.shape
    rectified_panos, rect_rots = self.rectify_images(panos, rots, trans)

    # Change to channels first.
    rectified_panos_cf = rectified_panos.permute((0, 1, 4, 2, 3))
    image_features_0_cf = rectified_panos_cf[:, 0]
    image_features_1_cf = rectified_panos_cf[:, 1]
    # print("Image features shape", image_features_0.shape)
    # Pass through encoding UNet.
    image_features_0_cf = self.unet(image_features_0_cf)
    image_features_1_cf = self.unet(image_features_1_cf)
    # Change to channels last.
    image_features_0 = image_features_0_cf.permute((0, 2, 3, 1))
    image_features_1 = image_features_1_cf.permute((0, 2, 3, 1))

    # Create cost volume.
    image_features = torch.stack((image_features_0, image_features_1), dim=1)
    cost_volume = self.calculate_cost_volume(image_features,
                                             stride=0.25,
                                             slide_range=0.25,
                                             cost_type="abs_diff",
                                             direction="up")

    # Pass cost volume through Conv3D layers.
    cost_volume = cost_volume.permute((0, 4, 1, 2, 3))
    cost_volume = self.unet3d(cost_volume)

    # Now the length dimension is now the channel dimension.
    image_features = cost_volume[:, 0, :, :, :]

    raw_image_features_d1 = self.decoders1(image_features)
    raw_image_features_d1 = nn.functional.interpolate(
      raw_image_features_d1,
      scale_factor=4,
      mode="bilinear",
      align_corners=False
    ).permute((0, 2, 3, 1))

    image_features = torch.cat((image_features, image_features_1_cf), dim=1)
    for i in range(len(self.decoders2)):
      # print("Image features shape1", image_features.shape)
      image_features, _ = self.decoders2[i](image_features)
    # print("Image features shape2", image_features.shape)

    assert torch.isfinite(image_features).all(), "Nan in image features"

    trans_norm = torch.norm(trans[:, 0] + trans[:, 1], dim=1)
    rectified_depth = self.erp_disparity_to_depth(image_features[:, :1], trans_norm)
    rectified_depth = rectified_depth.permute((0, 2, 3, 1))
    depth = self.unrectify_image(rectified_depth, rect_rots[:, 1])

    unrect_disp = self.unrectify_image(image_features[:, :1].permute((0, 2, 3, 1)),
                                       rect_rots[:, 1])

    # Change to channels last.
    # print("Final depth shape", depth.shape)
    # sys.exit(0)
    return {
      "depth": depth,
      "rectified_depth": rectified_depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "raw_image_features_d1": raw_image_features_d1,
      "rectified_panos": rectified_panos,
      "unrect_disp": unrect_disp,
      "rect_rots": rect_rots,
      "trans_norm": trans_norm
    }

  def estimate_depth_using_cost_volume_v3_cylindrical(self, panos, rots, trans):
    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """
    batch_size, _, height, width, _ = panos.shape
    rect_erp_panos, rect_rots, rot_up, rot_fw = self.rectify_images(panos, rots,
                                                                    trans,
                                                                    include_rots=True)

    cylinder_length = 10
    rectified_panos_0 = my_torch_helpers.equirectangular_to_cylindrical(
      panos[:, 0],
      cylinder_length=cylinder_length,
      width=self.width,
      height=self.height,
      rect_rots=rect_rots[:, 0],
      depth=False)
    rectified_panos_1 = my_torch_helpers.equirectangular_to_cylindrical(
      panos[:, 1],
      cylinder_length=cylinder_length,
      width=self.width,
      height=self.height,
      rect_rots=rect_rots[:, 1],
      depth=False)

    rectified_panos = torch.stack((rectified_panos_0, rectified_panos_1), dim=1)

    # Change to channels first.
    rectified_panos_cf = rectified_panos.permute((0, 1, 4, 2, 3))
    image_features_0_cf = rectified_panos_cf[:, 0]
    image_features_1_cf = rectified_panos_cf[:, 1]
    # Pass through encoding UNet.
    image_features_0_cf = self.unet(image_features_0_cf)
    image_features_1_cf = self.unet(image_features_1_cf)
    # Change to channels last.
    image_features_0 = image_features_0_cf.permute((0, 2, 3, 1))
    image_features_1 = image_features_1_cf.permute((0, 2, 3, 1))

    # Create cost volume.
    image_features = torch.stack((image_features_0, image_features_1), dim=1)
    cost_volume = self.calculate_cost_volume(image_features,
                                             stride=1.0,
                                             slide_range=1.0,
                                             cost_type="abs_diff",
                                             direction="up")

    assert torch.isfinite(cost_volume).all(), "Nan in cost volume"

    # Pass cost volume through Conv3D layers.
    cost_volume = cost_volume.permute((0, 4, 1, 2, 3))
    cost_volume = self.unet3d(cost_volume)

    # Now the length dimension is now the channel dimension.
    image_features = cost_volume[:, 0, :, :, :]

    raw_image_features_d1 = self.decoders1(image_features)
    raw_image_features_d1 = nn.functional.interpolate(
      raw_image_features_d1,
      scale_factor=4,
      mode="bilinear",
      align_corners=False
    ).permute((0, 2, 3, 1))

    # Now the length dimension is now the channel dimension.
    # image_features = cost_volume[:, 0, :, :, :]
    image_features = torch.cat(
      (cost_volume[:, 0, :, :, :], image_features_1_cf),
      dim=1)

    for i in range(len(self.decoders2)):
      # print("Image features shape1", image_features.shape)
      image_features, _ = self.decoders2[i](image_features)
    # print("Image features shape2", image_features.shape)

    assert torch.isfinite(image_features).all(), "Nan in image features"

    trans_norm = torch.norm(trans[:, 0] + trans[:, 1], dim=1)

    depth = my_torch_helpers.safe_divide(trans_norm.view(batch_size, 1, 1, 1),
                                         image_features)
    depth = depth.permute((0, 2, 3, 1))
    depth = my_torch_helpers.cylindrical_to_equirectangular(
      depth,
      cylinder_length=cylinder_length,
      height=self.height,
      width=self.width,
      rect_rots=rect_rots[:, 1],
      depth=True
    )

    return {
      "depth": depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "raw_image_features_d1": raw_image_features_d1,
      "rectified_panos": rectified_panos,
      "trans_norm": trans_norm,
      "image_features_0": image_features_0,
      "image_features_1": image_features_1,
      "rect_rots": rect_rots
    }

  def estimate_depth_using_cost_volume_v3_erp(self, panos, rots, trans,
                                              min_depth=2.0, max_depth=100):

    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """
    batch_size, _, height, width, _ = panos.shape

    trans_norm = torch.norm(trans[:, 0] - trans[:, 1], dim=1)
    rectified_panos, rect_rots = self.rectify_images(panos, rots, trans,
                                                     rectify_vertical=False)
    # print("rectified_panos.shape", rectified_panos.shape)
    # my_torch_helpers.save_torch_image("rectified0.png", rectified_panos[0])
    # my_torch_helpers.save_torch_image("rectified1.png", rectified_panos[0,1:])
    # print("Trans", trans)
    # print("Trans norm", trans_norm)
    # cost_volume = self.calculate_cost_volume_erp(
    #   rectified_panos,
    #   depths=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, 64),
    #   trans_norm=trans_norm,
    #   cost_type='abs_diff')
    # for i in range(cost_volume.shape[1]):
    #   my_torch_helpers.save_torch_image("test/cv/%d.png" % i, cost_volume[:,i])
    # # exit(0)

    # Change to channels first.
    rectified_panos_cf = rectified_panos.permute((0, 1, 4, 2, 3))
    image_features_0_cf = rectified_panos_cf[:, 0]
    image_features_1_cf = rectified_panos_cf[:, 1]
    # Pass through encoding UNet.
    image_features_0_cf = self.unet(image_features_0_cf)
    image_features_1_cf = self.unet(image_features_1_cf)
    # Change to channels last.
    image_features_0 = image_features_0_cf.permute((0, 2, 3, 1))
    image_features_1 = image_features_1_cf.permute((0, 2, 3, 1))

    # Create cost volume.
    image_features = torch.stack((image_features_0, image_features_1), dim=1)
    cost_volume = self.calculate_cost_volume_erp(
      image_features,
      depths=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, 64),
      trans_norm=trans_norm,
      cost_type='abs_diff')

    assert torch.isfinite(cost_volume).all(), "Nan in cost volume"

    # Pass cost volume through Conv3D layers.
    cost_volume = cost_volume.permute((0, 4, 1, 2, 3))
    cost_volume = self.unet3d(cost_volume)

    # Now the length dimension is now the channel dimension.
    image_features = cost_volume[:, 0, :, :, :]

    raw_image_features_d1 = self.decoders1(image_features)
    raw_image_features_d1 = nn.functional.interpolate(
      raw_image_features_d1,
      scale_factor=4,
      mode="bilinear",
      align_corners=False
    ).permute((0, 2, 3, 1))

    # Here the cost volume already takes into account the translation.
    if self.depth_type == "one_over":
      rectified_depth_d1 = my_torch_helpers.safe_divide(
        1.0, raw_image_features_d1)
    elif self.depth_type == "disparity":
      min_disp = 1 / 100
      max_disp = 1 / 0.1
      rectified_depth_d1 = min_disp + (max_disp - min_disp) * torch.sigmoid(
        raw_image_features_d1)
      rectified_depth_d1 = torch.div(1.0, rectified_depth_d1)

    # Now the length dimension is now the channel dimension.
    # image_features = cost_volume[:, 0, :, :, :]
    image_features = torch.cat(
      (cost_volume[:, 0, :, :, :], image_features_1_cf),
      dim=1)

    for i in range(len(self.decoders2)):
      image_features, _ = self.decoders2[i](image_features)

    assert torch.isfinite(image_features).all(), "Nan in image features"

    # rectified_depth = my_torch_helpers.safe_divide(
    #   trans_norm.view(batch_size, 1, 1, 1),
    #   image_features)

    # Here the cost volume already takes into account the translation.
    if self.depth_type == "one_over":
      rectified_depth = my_torch_helpers.safe_divide(
        1.0, image_features[:, :1])
    elif self.depth_type == "disparity":
      min_disp = 1 / 100
      max_disp = 1 / 0.1
      rectified_depth = min_disp + (max_disp - min_disp) * torch.sigmoid(
        image_features[:, :1])
      rectified_depth = torch.div(1.0, rectified_depth)

    rectified_depth = rectified_depth.permute((0, 2, 3, 1))
    depth = self.unrectify_image(rectified_depth, rect_rots[:, 1])

    return {
      "depth": depth,
      "rectified_depth_d1": rectified_depth_d1,
      "rectified_depth": rectified_depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "raw_image_features_d1": raw_image_features_d1,
      "rectified_panos": rectified_panos,
      "rect_rots": rect_rots,
      "trans_norm": trans_norm
    }

  def erp_disparity_to_depth(self, disparity, trans_norm,
                             clamp_disparity=False):
    """Converts rectified disparity to depth.

    Uses law of sines. See https://ieeexplore.ieee.org/document/7738212.

    Args:
      disparity: Disparity Image as (B, C, H, W) tensor.
      trans_norm: Norm of the translation.
      clamp_disparity: Clamp the disparity to [0, pi-omega]

    Returns:
      Depth image as a (B, C, H, W) tensor.

    """
    batch_size, channels, height, width = disparity.shape

    if not torch.isfinite(disparity).all():
      raise ValueError("Nan in disparity")

    omega = self.initialize_omega(disparity.dtype, height, width)
    trans_norm = trans_norm.view(batch_size, 1, 1, 1)
    clamped_disparity = disparity
    if clamp_disparity:
      clamped_disparity = torch.clamp_min(clamped_disparity, 0.0)
      clamped_disparity = torch.min(clamped_disparity, np.pi - omega)
    depth = my_torch_helpers.safe_divide(
      trans_norm * torch.sin(omega + clamped_disparity),
      torch.sin(clamped_disparity))

    assert torch.isfinite(depth).all(), "Nan in depth"
    return depth

  def erp_depth_to_disparity(self, depth, trans_norm):
    """Converts rectified depth to disparity.

    Opposite of erp_disparity_to_depth.

    Args:
      depth: Depth as (B, C, H, W) tensor.
      trans_norm: Norm of the translation as (B) tensor.

    Returns:
      Disparity image.
    """
    dtype = depth.dtype
    batch_size, channels, height, width = depth.shape
    omega = self.initialize_omega(dtype, height, width)
    trans_norm = trans_norm.view(batch_size, 1, 1, 1)
    # Side of the third triangle using law of cosines.
    c_sq = depth * depth + trans_norm * trans_norm - \
           2 * depth * trans_norm * torch.cos(omega)
    c = torch.sqrt(c_sq)
    # Disparity using law of sines.
    sin_disp = my_torch_helpers.safe_divide(trans_norm * torch.sin(omega), c)
    sin_disp = torch.clamp(sin_disp, -1, 1)
    disp = torch.asin(sin_disp)
    return disp

  def initialize_omega(self, dtype, height, width):
    if self.omega is None:
      omega = torch.arange(0, height, device=self.device, dtype=dtype) + 0.5
      omega = omega * np.pi / height
      omega = omega.view(1, 1, height, 1).repeat(1, 1, 1, width)
      # print("Omega shape", omega.shape)
      self.omega = omega
    return self.omega

  def calculate_cost_volume(self,
                            images,
                            stride=1.0,
                            slide_range=1.0,
                            cost_type="abs_diff",
                            direction="up"):
    """Calculates a cost volume from the images.

    Args:
      images: Tensor of shape (B, 2, H, W, C).
        The target image should be in index 1 along dim 1.
      stride: Stride to move the tensor. Can be float.
      slide_range: How far to move the tensor. Can be float.
      cost_type: Type of the cost volume.
      direction: Direction of cost volume.

    Returns:
      Tensor of shape (B, L, H, W, C).
    """
    batch_size, image_ct, height, width, channels = images.shape
    other_images_cl = images[:, 0].permute((0, 3, 1, 2))
    reference_image = images[:, 1]
    cost_volume = []
    grid_x = torch.linspace(-1, 1, width, device=self.device)
    grix_y = torch.linspace(-1, 1, height, device=self.device)
    grid_y, grid_x = torch.meshgrid(grix_y, grid_x)
    for i in np.arange(0, slide_range * height, stride):
      if direction == "up":
        m_grid_x = grid_x
        m_grid_y = grid_y + i * 2 / height
      elif direction == "left":
        m_grid_x = grid_x + i * 2 / width
        m_grid_y = grid_y
      m_grid = torch.stack((m_grid_x, m_grid_y), dim=2).expand(
        (batch_size, height, width, 2))
      other_image = torch.nn.functional.grid_sample(other_images_cl,
                                                    m_grid,
                                                    mode="bilinear",
                                                    align_corners=False)
      other_image = other_image.permute((0, 2, 3, 1))
      if cost_type == "abs_diff":
        diff_image = torch.abs(reference_image - other_image)
      cost_volume.append(diff_image)
    cost_volume = torch.stack(cost_volume, dim=1)
    return cost_volume

  def calculate_cost_volume_erp(self,
                                images,
                                depths,
                                trans_norm,
                                cost_type="abs_diff",
                                direction="up"):
    """Calculates a cost volume for ERP images via backwards warping.

    Panos should be moving forward between images 0 and 1.

    Args:
      images: Tensor of shape (B, 2, H, W, C).
        The target image should be in index 1 along dim 1.
      depths: Tensor of depths to test.
      trans_norm: Norm of the translation.
      cost_type: Type of the cost volume.
      direction: Direction of cost volume.

    Returns:
      Tensor of shape (B, L, H, W, C).
    """
    batch_size, image_ct, height, width, channels = images.shape
    other_image = images[:, 0]
    other_image_cf = other_image.permute((0, 3, 1, 2))
    reference_image_cf = images[:, 1].permute((0, 3, 1, 2))
    phi = torch.arange(0,
                       height,
                       device=images.device,
                       dtype=images.dtype)
    phi = (phi + 0.5) * (np.pi / height)
    theta = torch.arange(0,
                         width,
                         device=images.device,
                         dtype=images.dtype)
    theta = (theta + 0.5) * (2 * np.pi / width) + np.pi / 2
    phi, theta = torch.meshgrid(phi, theta)
    translation = torch.stack(
      (torch.zeros_like(trans_norm), torch.zeros_like(trans_norm), trans_norm),
      dim=1)
    xyz = my_torch_helpers.spherical_to_cartesian(theta, phi, r=1)
    xyz = xyz[None, :, :, :].expand(batch_size, height, width, 3)

    cost_volume = []
    for i, depth in enumerate(depths):
      m_xyz = depth * xyz - translation[:, None, None, :]
      uv = my_torch_helpers.cartesian_to_spherical(m_xyz)
      u = torch.fmod(uv[..., 0] - np.pi / 2 + 4 * np.pi, 2 * np.pi) / np.pi - 1
      v = 2 * (uv[..., 1] / np.pi) - 1
      cv_image = torch.nn.functional.grid_sample(
        other_image_cf,
        torch.stack((u, v,), dim=-1),
        mode='bilinear',
        align_corners=True)
      if cost_type == 'abs_diff':
        cv_image = torch.abs(cv_image - reference_image_cf)
      elif cost_type != 'none':
        raise ValueError('Unknown cost type')
      cost_volume.append(cv_image)

    cost_volume = torch.stack(cost_volume, dim=1).permute((0, 1, 3, 4, 2))
    return cost_volume

  def backwards_warping_uv(self, target_depth, rot, trans, inv_rot=True):
    """Does backwards warping returning the UV values.

    Args:
      target_depth: target depth. Should be (B, H, W) tensor.
      rot: rotation. Should be a (B, 3, 3) tensor.
      trans: translation. Should be (B, 3) tensor.
      inv_rot: invert the rotation.

    Returns:
      UV coordinates of the source image in range (0, 2*pi) x (0, pi)

    """
    batch_size, height, width = target_depth.shape[:3]

    phi = torch.arange(0,
                       height,
                       device=target_depth.device,
                       dtype=target_depth.dtype)
    phi = (phi + 0.5) * (np.pi / height)
    theta = torch.arange(0,
                         width,
                         device=target_depth.device,
                         dtype=target_depth.dtype)
    theta = (theta + 0.5) * (2 * np.pi / width) + np.pi / 2
    phi, theta = torch.meshgrid(phi, theta)

    xyz = my_torch_helpers.spherical_to_cartesian(theta[None, :, :],
                                                  phi[None, :, :], target_depth)

    if inv_rot:
      rot_inv = torch.inverse(rot)
      xyz = torch.matmul(rot_inv[:, None, None, :, :], xyz[:, :, :, :,
                                                       None])[:, :, :, :, 0]
      xyz = xyz + trans[:, None, None, :]
    else:
      xyz = xyz + trans[:, None, None, :]
      xyz = torch.matmul(rot[:, None, None, :, :], xyz[:, :, :, :,
                                                   None])[:, :, :, :, 0]

    uvs = my_torch_helpers.cartesian_to_spherical(xyz)
    u = torch.fmod(uvs[:, :, :, 0] - np.pi / 2 + 4 * np.pi, 2 * np.pi)
    v = uvs[:, :, :, 1]
    uvs = torch.stack((u, v), dim=-1)
    return uvs

  def backwards_warping(self, images, target_depth, rot, trans, inv_rot=True):
    """Does backwards warping.

    Args:
      images: Panoramas
      target_depth: Target depth. Should be BxHxW.
      rot: Rotation. Should be Bx3x3.
      trans: Translation. Should be Bx3.
      inv_rot: Invert the rotation.

    Returns:
      RGB images as BxHxWxC tensor.

    """
    full_height, full_width = images.shape[1:3]
    batch_size, height, width = target_depth.shape[:3]
    uvs = self.backwards_warping_uv(target_depth, rot, trans, inv_rot=inv_rot)
    uvs = uvs * torch.tensor([full_width / (2 * np.pi), full_height / np.pi],
                             dtype=images.dtype,
                             device=images.device)
    uvs = uvs - 0.5
    uvs = uvs * torch.tensor([2 / full_width, 2 / full_height],
                             dtype=images.dtype,
                             device=images.device) - 1.0
    images = F.grid_sample(images.permute((0, 3, 1, 2)),
                           uvs,
                           mode="bilinear",
                           align_corners=False)
    return images.permute(0, 2, 3, 1)

  def render_point_cloud(self, pointcloud, rot=None, trans=None,
                         linearize_angle=np.deg2rad(10)):
    """Renders a point cloud.

    Args:
      pointcloud: point cloud.
      rot: rotation of points.
      trans: translation of points.

    Returns:
      Rendered images as a channels-last tensor.

    """
    # Prepares the renderer.
    if rot is None:
      rot = torch.eye(3, dtype=torch.float32)
    if trans is None:
      trans = torch.zeros(3, dtype=torch.float32)
    print("Rots shape", rot.shape)
    print("Trans shape", trans.shape)
    cameras = OpenGLPerspectiveCameras(device=self.device,
                                       R=rot,
                                       T=trans,
                                       fov=99999,
                                       znear=0.0000001,
                                       zfar=100.0)
    raster_settings = PointsRasterizationSettings(
      image_size=self.raster_resolution,
      radius=self.point_radius,
      points_per_pixel=self.points_per_pixel)
    rasterizer = SpherePointsRasterizer(cameras=cameras,
                                        raster_settings=raster_settings,
                                        linearize_angle=linearize_angle)
    renderer = SpherePointsRenderer(rasterizer=rasterizer,
                                    compositor=AlphaCompositor())
    return renderer(pointcloud)

  def make_point_cloud(self,
                       depths,
                       images,
                       rots=None,
                       trans=None,
                       inv_rot_trans=True,
                       upscale=False,
                       full_images=None,
                       linearize_angle=np.deg2rad(10)):
    """Creates a single point cloud from the rendered depths.

    Args:
      depths: Depths.
      images: Panos.
      rots: Rotations to the reference.
      trans: Translations to the reference.
      inv_rot_trans: Should we invert the rotation and translations

    Returns:
      Vertices and colors of the point cloud.

    """

    height, width = depths.shape[2:4]
    if upscale and full_images is not None:
      full_height, full_width = full_images.shape[2:4]
    theta, phi = np.meshgrid(
      (np.arange(width) + 0.5) * (2 * np.pi / width) + np.pi / 2,
      (np.arange(height) + 0.5) * (np.pi / height))
    theta = torch.tensor(theta, dtype=depths.dtype, device=depths.device)
    phi = torch.tensor(phi, dtype=depths.dtype, device=depths.device)
    if upscale and full_images is not None:
      theta_full, phi_full = np.meshgrid(
        (np.arange(full_width) + 0.5) * (2 * np.pi / full_width) + np.pi / 2,
        (np.arange(full_height) + 0.5) * (np.pi / full_height))
      theta_full = torch.tensor(theta_full,
                                dtype=depths.dtype,
                                device=depths.device)
      phi_full = torch.tensor(phi_full,
                              dtype=depths.dtype,
                              device=depths.device)
      depths_full = torch.nn.functional.interpolate(
        depths, (full_width, full_height),
        mode=self.interpolation_mode,
        align_corners=self.align_corners)
      depths_right = torch.cat((depths[:, :, :, 1:], depths[:, :, :, 0:1]),
                               dim=3)
      depths_left = torch.cat((depths[:, :, :, -1:], depths[:, :, :, 0:-1]),
                              dim=3)
      depths_grad = (depths_right - depths_left) / 2
    all_xyz_coords = []
    all_point_colors = []
    for batch in range(depths.shape[0]):
      xyz_coords = []
      point_colors = []
      for img_idx in range(depths.shape[1]):
        xyz = my_torch_helpers.spherical_to_cartesian(
          theta, phi, depths[batch, img_idx, :, :])
        colors = images[batch, img_idx, :, :, :].reshape((-1, 3))
        if rots is not None and trans is not None:
          if inv_rot_trans:
            m_rot_inv = rots[batch, img_idx, :, :].detach().cpu().numpy()
            m_rot_inv = Rotation.from_matrix(m_rot_inv).inv().as_matrix()
            m_rot_inv = torch.tensor(m_rot_inv,
                                     device=rots.device,
                                     dtype=rots.dtype)
            m_trans = trans[batch, img_idx, :]

            # Perform xyz = m_rot_inv @ xyz - m_trans
            xyz = torch.tensordot(m_rot_inv,
                                  xyz.reshape((-1, 3)),
                                  dims=([1], [1])).permute((1, 0))
            xyz = xyz - m_trans
          else:
            m_rot_inv = rots[batch, img_idx, :, :]
            m_trans = trans[batch, img_idx, :]
            # Perform xyz = m_rot_inv @ xyz - m_trans
            xyz = torch.tensordot(m_rot_inv,
                                  xyz.reshape((-1, 3)),
                                  dims=([1], [1])).permute((1, 0))
            xyz = xyz + m_trans

          if upscale and full_images is not None:
            xyz = xyz.reshape(height, width, 3)
            uv_pos = my_torch_helpers.cartesian_to_spherical(xyz,
                                                             linearize_angle=linearize_angle)
            u = torch.fmod(uv_pos[:, :, 0] + 2 * np.pi,
                           2 * np.pi) * (width / (2 * np.pi))
            v = uv_pos[:, :, 1] * (height / np.pi)
            u_diff = torch.gt(torch.abs(u[1:, :] - u[:-1, :]), 1.0)
            v_diff = torch.gt(torch.abs(v[1:, :] - v[:-1, :]), 1.0)
            depths_small = torch.lt(torch.abs(depths_grad[batch, img_idx]), 1.0)
            x_diff = (u_diff | v_diff)
            x_diff = torch.cat((x_diff[0:1, :], x_diff), dim=0)
            x_diff = x_diff & depths_small
            x_diff = x_diff.type(torch.float32)
            x_diff_up = torch.nn.functional.interpolate(
              x_diff[None, None, :, :], (full_width, full_height),
              mode=self.interpolation_mode,
              align_corners=self.align_corners)
            x_diff_up = torch.gt(x_diff_up[0, 0], 0.5)

            xyz = xyz.reshape((-1, 3))
            xyz_full = my_torch_helpers.spherical_to_cartesian(
              theta_full, phi_full, depths_full[batch, img_idx, :, :])
            xyz_full = xyz_full[x_diff_up]
            if rots is not None and trans is not None:
              if inv_rot_trans:
                m_rot_inv = rots[batch, img_idx, :, :].detach().cpu().numpy()
                m_rot_inv = Rotation.from_matrix(m_rot_inv).inv().as_matrix()
                m_rot_inv = torch.tensor(m_rot_inv,
                                         device=rots.device,
                                         dtype=rots.dtype)
                m_trans = trans[batch, img_idx, :]

                # Perform xyz = m_rot_inv @ xyz - m_trans
                xyz_full = torch.tensordot(
                  m_rot_inv, xyz_full.reshape(
                    (-1, 3)), dims=([1], [1])).permute((1, 0)) - m_trans
              else:
                m_rot_inv = rots[batch, img_idx, :, :]
                m_trans = trans[batch, img_idx, :]
                # Perform xyz = m_rot_inv @ xyz - m_trans
                xyz_full = torch.tensordot(
                  m_rot_inv, xyz_full.reshape(
                    (-1, 3)), dims=([1], [1])).permute((1, 0)) + m_trans

              colors_full = full_images[batch, img_idx][x_diff_up].reshape(
                (-1, 3))
              xyz = torch.cat((xyz, xyz_full), dim=0)
              colors = torch.cat((colors, colors_full), dim=0)

        xyz = xyz.reshape((-1, 3))
        xyz_coords.append(xyz)
        point_colors.append(colors)
      xyz_coords = torch.cat(xyz_coords, dim=0)
      point_colors = torch.cat(point_colors, dim=0)
      all_xyz_coords.append(xyz_coords)
      all_point_colors.append(point_colors)
    return Pointclouds(points=all_xyz_coords, features=all_point_colors)

  def forward(self, x_in):
    """Performs a forward pass and return the rendered point cloud.

    Args:
      x_in: Panos, rotations, translations as a tuple.

    Returns:
      Rendered point cloud.

    """
    panos, rots, trans = x_in

    batch_size, seq_len, height, width = panos.shape[:4]
    # print("batch size", batch_size, seq_len, height, width)
    depths = self.estimate_depth(
      panos.reshape((batch_size * seq_len, height, width, 3)))
    depths = depths.reshape((batch_size, seq_len, height, width))
    if self.verbose:
      print("Average depths", np.mean(depths.detach().cpu().numpy()))
    verts, colors = self.make_point_cloud(depths, panos, rots, 0.5 * trans)

    images = self.render_point_cloud(verts, colors)
    return images

  def predict_with_backwards_warping(self, panos, rots, trans, full_panos):
    """Performs prediction with backwards warping.

    Args:
      panos: Input panos, shrunken
      rots: Input rotations
      trans: Input translations
      full_panos: Full resolution panos

    Returns:
      Warped images

    """
    height = self.raster_resolution
    width = self.raster_resolution

    batch_size, seq_len, full_height, full_width = panos.shape[:4]

    depths = self.estimate_depth(panos[:, [0, 2], :, :, :].reshape(
      (batch_size * 2, height, width, 3)))
    depths = depths.reshape((batch_size, 2, height, width))
    outputs1 = self.backwards_warping(full_panos[:, 1, :, :, :],
                                      depths[:, 0, :, :], rots[:, 0, :, :],
                                      0.5 * trans[:, 0, :])
    outputs2 = self.backwards_warping(full_panos[:, 1, :, :, :],
                                      depths[:, 1, :, :], rots[:, 2, :, :],
                                      0.5 * trans[:, 2, :])
    return torch.stack((outputs1, outputs2), dim=1)

  def do_inpainting(self, input):
    """Runs the inpainting UNet.

    Args:
      input: input image. Channels last.

    Returns:
      Output image. Channels last.

    """
    return self.inpaint_net(input)

  def estimate_pose(self, images):
    """Estimates the pose using Monodepth2's pose estimator

    Args:
      images: Set of 2 images. Bx2xHxWxC.

    Returns:
      rotations: Bx3x3
      translation: Bx3

    """

    m_images = torch.cat((images[:, 0, :, :, :], images[:, 1, :, :, :]), dim=3)
    m_images = m_images.permute((0, 3, 1, 2))
    shrinked_input_images = torch.nn.functional.interpolate(m_images,
                                                            (320, 1024),
                                                            mode="bilinear",
                                                            align_corners=False)

    features = [self.pose_encoder(shrinked_input_images)]
    axisangle, translation = self.pose_decoder(features)

    rotation = rot_from_axisangle(axisangle[:, 0])
    rotation = rotation[:, :3, :3]

    translation = 25 * translation[:, 0, 0]

    return rotation, translation

  def correct_depths(self, depths, panos1, panos0, rot, trans, far_depth=100):
    """This sets the depths to infinity if the backward warping loss is smaller for the unwarped image.

    Args:
      depths: Depths of pano1. Should be BxHxWx1.
      panos1: Pano1. BxHxWxC.
      panos0: Pano0. BxHxWxC
      rot: Rotation from pano0 to pano1. Should be (B, 3, 3) tensor.
      trans: Translation from pano0 to pano1. Should be (B, 3) tensor.

    Returns:
      Depths of pano1 as (B, H, W, C) tensor.

    """
    print("Shapes", depths.shape, panos1.shape, panos0.shape, rot.shape,
          trans.shape)
    y_pred = self.backwards_warping(panos0, depths[:, :, :, 0], rot, trans)
    y_true = panos1
    y_pred_org = panos0
    loss_pred = torch.mean(torch.abs(y_pred - y_true), dim=3)
    loss_orig = torch.mean(torch.abs(y_pred_org - y_true), dim=3)
    far_depth_tensor = torch.tensor(far_depth,
                                    dtype=depths.dtype,
                                    device=depths.device).expand_as(depths)
    far_indices = torch.lt(loss_orig, loss_pred)[:, :, :, None]
    print("Shapes", far_indices.shape, depths.shape)
    new_depths = torch.where(far_indices, far_depth_tensor, depths)
    return new_depths

  def build_depth_mask(self,
                       depth_img,
                       threshold=1,
                       use_second_depth=True,
                       dtype=torch.float32):
    """Builds a depth mask from the image.

    Args:
      depth_img: Depth image as (N, H, W, K) tensor.
      threshold: Threshold.
      use_second_depth: Whether to consider the depth of the second point.
      dtype: Return dtype.

    Returns: Depth mask as (N, H, W, 1) tensor.

    """
    mask1 = torch.lt(depth_img[:, :, :, 0:1], threshold)
    final_mask = mask1
    if use_second_depth:
      mask2 = torch.lt(depth_img[:, :, :, 1:2], threshold)
      final_mask = mask1 | mask2
    return final_mask.type(dtype)

  def rectify_images(self, panos, rots, trans, include_rots=False,
                     rectify_vertical=True):
    """Rectifies images. Second pano should be a reference pano.

    Args:
      panos: Panos as (B, 2, H, W, C) tensor.
      rots: Rotations as (B, 2, 3, 3) tensor.
      trans: Translation as (B, 2, 3, 3) tensor.
      include_rots: Debugging flag.
      rectify_vertical: Rectify upwards.

    Returns:
      Rectified panos as (B, 2, H, W, C) tensor.
      Rotation matrices as (B, 2, 3, 3) tensor.

    """
    batch_size, _, height, width, channels = panos.shape
    # Calculate rotation matrices so that the movement is forward.
    rot_fw = []
    for i in range(batch_size):
      m_trans = (trans[i, 0] - trans[i, 1]).cpu().numpy()
      m_trans = m_trans / np.linalg.norm(m_trans)
      m_ang = np.arccos(
        -m_trans[2] / np.linalg.norm(m_trans[[0, 2]])) * (2 * (-m_trans[0] > 0) - 1)
      # print("Angle", m_trans, np.rad2deg(m_ang))
      m_rot_fw = my_helpers.rotate_around_axis(np.array([0, 1, 0]), m_ang)
      rot_fw.append(
        torch.tensor(m_rot_fw, device=panos.device, dtype=torch.float32))
    rot_fw = torch.stack(rot_fw, dim=0)
    # Rotation matrix to make forward face down.
    rot_up = my_helpers.rotate_around_axis(np.array([1, 0, 0]), np.pi / 2)
    rot_up = torch.tensor(rot_up, device=panos.device, dtype=torch.float32)
    rot_up = rot_up.view(1, 3, 3).repeat(batch_size, 1, 1)
    # Calculate rotation matrices for images 0 and 1.
    if rectify_vertical:
      rot_0 = rots[:, 0] @ rot_fw @ rot_up
      rot_1 = rots[:, 1] @ rot_fw @ rot_up
    else:
      rot_0 = rots[:, 0] @ rot_fw
      rot_1 = rots[:, 1] @ rot_fw

    rect_panos_0 = my_torch_helpers.rotate_equirectangular_image(
      panos[:, 0], rot_0, linearize_angle=np.deg2rad(5))
    rect_panos_1 = my_torch_helpers.rotate_equirectangular_image(
      panos[:, 1], rot_1, linearize_angle=np.deg2rad(5))
    rect_panos = torch.stack((rect_panos_0, rect_panos_1), dim=1)
    rect_rots = torch.stack((rot_0, rot_1), dim=1)
    if include_rots:
      return rect_panos, rect_rots, rot_up, rot_fw
    return rect_panos, rect_rots

  def unrectify_image(self, pano, rot_mat):
    """Undo a rectification.

    Args:
      pano: Panos to derotate as (B, H, W, C) tensor.
      rot_mat: Rotation matrix for rectification as (B, 3, 3) tensor.

    Returns:
      Unrotated pano as (B, H, W, C) tensor.
    """
    rot_inv = torch.inverse(rot_mat)
    unrot_pano = my_torch_helpers.rotate_equirectangular_image(pano, rot_inv)
    return unrot_pano

  def get_total_params(self):
    """Gets the total number of parameters.

    Returns:
      int total number of parameters.
    """
    if self.using_monodepth_model:
      encoder_params = my_torch_helpers.total_params(self.monodepth_encoder)
      decoder_params = my_torch_helpers.total_params(self.monodepth_decoder)
      depth_params = encoder_params + decoder_params
      pose_encoder_params = my_torch_helpers.total_params(self.pose_encoder)
      pose_decoder_params = my_torch_helpers.total_params(self.pose_decoder)
      pose_params = pose_encoder_params + pose_decoder_params
      total_params = depth_params + pose_params
    elif self.use_cost_volume == "v1" or self.use_cost_volume == "true":
      encoder_params = my_torch_helpers.total_params(self.encoders)
      decoder_params = my_torch_helpers.total_params(self.decoders)
      final_conv_params = my_torch_helpers.total_params(self.final_conv)
      cv_layers_params = my_torch_helpers.total_params(self.cv_layers)
      total_params = encoder_params + decoder_params + final_conv_params + cv_layers_params
    elif self.use_cost_volume == "v2":
      unet_params = my_torch_helpers.total_params(self.unet)
      cv_params = my_torch_helpers.total_params(self.cv_layers)
      decoders2_params = my_torch_helpers.total_params(self.decoders2)
      total_params = unet_params + cv_params + decoders2_params
    elif self.use_cost_volume in ["v3", "v3_cylindrical", "v3_erp"]:
      unet_params = my_torch_helpers.total_params(self.unet)
      unet3d_params = my_torch_helpers.total_params(self.unet3d)
      decoders1_params = my_torch_helpers.total_params(self.decoders1)
      decoders2_params = my_torch_helpers.total_params(self.decoders2)
      total_params = unet_params + unet3d_params + decoders1_params + decoders2_params
    else:
      encoder_params = my_torch_helpers.total_params(self.encoders)
      decoder_params = my_torch_helpers.total_params(self.decoders)
      final_conv_params = my_torch_helpers.total_params(self.final_conv)
      depth_params = encoder_params + decoder_params + final_conv_params
      pose_encoder_params = my_torch_helpers.total_params(self.pose_encoder)
      pose_decoder_params = my_torch_helpers.total_params(self.pose_decoder)
      pose_params = pose_encoder_params + pose_decoder_params
      total_params = depth_params + pose_params
    return total_params
