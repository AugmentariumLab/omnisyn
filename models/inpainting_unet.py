# Lint as: python3
"""This file contains the InpaintNet module.
"""

import torch
from torch import nn

from helpers import my_torch_helpers
from models.common_blocks import (ConvBlock2, ConvBlock, UNet2)


class InpaintNet(nn.Module):
  """A UNet for inpainting.
  """

  def __init__(self,
               input_channels=3,
               output_channels=3,
               size=4,
               layers=7,
               width=256,
               height=256,
               device="cuda",
               gate=False,
               use_residual=False,
               use_batchnorm=True,
               use_wrap_padding=False,
               use_one_conv=False,
               use_final_convblock=True,
               model_version="v1",
               upscale=False,
               **kwargs):
    """Create an InpaintNet.

    Args:
      input_channels: Number of input channels.
      output_channels: Number of output channels.
      layers: Number of layers.
      device: Cuda device.
      **kwargs:
    """
    super().__init__(**kwargs)
    self.width = width
    self.height = height
    self.device = device
    self.size = size
    self.gate = gate
    self.layers = layers
    self.use_residual = use_residual
    self.use_wrap_padding = use_wrap_padding
    self.use_one_conv = use_one_conv
    self.model_version = model_version
    self.interpolation = "bilinear"
    self.upscale = upscale

    if model_version == "v1":
      print("Using inpainting model v1")
      model = self.build_v1_model(input_channels=input_channels,
                                  output_channels=output_channels,
                                  size=size,
                                  use_batchnorm=use_batchnorm,
                                  use_wrap_padding=use_wrap_padding,
                                  use_one_conv=use_one_conv,
                                  gate=gate,
                                  layers=layers,
                                  use_final_convblock=use_final_convblock)
      self.encoders = model["encoders"]
      self.decoders = model["decoders"]
      self.final_conv = model["final_conv"]
    elif model_version == "v2":
      print("Using inpainting model v2")
      model = self.build_v2_model(input_channels=input_channels,
                                  output_channels=output_channels,
                                  size=size,
                                  layers=layers,
                                  use_wrap_padding=use_wrap_padding)
      self.unet = model["unet"]
      self.final_conv = model["final_conv"]
      self.upscale_layer = model["upscale_layer"]
      unet_params = my_torch_helpers.total_params(self.unet)
      final_conv_params = my_torch_helpers.total_params(self.final_conv)
      upscale_params = my_torch_helpers.total_params(self.upscale_layer)
      total_params = unet_params + final_conv_params + upscale_params
      print("Total params: %d" % (total_params,))
    elif model_version == "v3":
      """This model has two decoders for RGB+D."""
      print("Using inpainting model v3")
      model = self.build_v3_model(input_channels=input_channels,
                                  output_channels=output_channels,
                                  size=size,
                                  layers=layers,
                                  use_wrap_padding=use_wrap_padding)
      self.encoders = model["encoders"]
      self.decoders = model["decoders"]
      self.final_conv = model["final_conv"]
      self.decoders2 = model["decoders2"]
      self.final_conv2 = model["final_conv2"]
    elif model_version == "v4":
      """This model generates feature maps and blends them using learned weights."""
      print("Using inpainting model v4")
      self.build_v4_model(input_channels=input_channels,
                          output_channels=output_channels,
                          size=size,
                          layers=layers,
                          use_wrap_padding=use_wrap_padding)
    else:
      raise ValueError("Unknown inpainting model version: %s" %
                       (model_version,))

  def build_v1_model(self, input_channels, output_channels, size, layers,
                     use_batchnorm, use_wrap_padding, use_one_conv, gate,
                     use_final_convblock):
    """Builds the V1 model.

    The v1 model is a UNet based on Conv-BN-LRelu blocks.

    Args:
      input_channels: Input channels.
      output_channels: Output channels.
      size: Size
      layers: Number of layers.
      use_batchnorm: Batch norm.
      use_wrap_padding: Wrap padding.
      use_one_conv: Use 1x1 conv at the end.
      gate: Use gated conv.
      use_final_convblock: Use a conv block at the end.

    Returns:

    """

    encoders = [
      ConvBlock(in_channels=input_channels,
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
      ConvBlock(in_channels=2 ** (size + 1),
                out_channels=2 ** size,
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

    if use_final_convblock:
      if use_one_conv:
        final_conv = ConvBlock(in_channels=2 ** size,
                               out_channels=output_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               use_batch_norm=False,
                               use_wrap_padding=use_wrap_padding,
                               use_activation=False,
                               upscale=False)
      else:
        final_conv = ConvBlock(in_channels=2 ** size,
                               out_channels=output_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               use_batch_norm=False,
                               use_wrap_padding=use_wrap_padding,
                               use_activation=False,
                               upscale=False)
    else:
      # For legacy purposes only.
      if use_one_conv:
        final_conv = nn.Conv2d(2 ** size,
                               output_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
      else:
        final_conv = nn.Conv2d(2 ** size,
                               output_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
    return {
      "encoders": encoders,
      "decoders": decoders,
      "final_conv": final_conv
    }

  def build_v2_model(self, input_channels, output_channels, size, layers,
                     use_wrap_padding):
    """Builds the V2 model.

    The V2 model is a UNet.

    Args:
      input_channels: Input channels.
      output_channels: Output channels.
      size: Size
      layers: Number of layers.
      use_wrap_padding: Wrap padding.

    Returns:
      Model
    """

    encoders = [
      ConvBlock2(in_channels=input_channels,
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 upscale=False,
                 use_wrap_padding=use_wrap_padding)
    ]
    decoders = [
      ConvBlock2(in_channels=2 ** (size + 1),
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False)
    ]

    for i in range(1, layers):
      channels = 2 ** (i + size - 1)
      encoders.append(
        ConvBlock2(in_channels=channels,
                   out_channels=2 * channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False))
      decoders.append(
        ConvBlock2(in_channels=4 * channels,
                   out_channels=channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False,
                   pooling=False))

    encoders.append(
      ConvBlock2(in_channels=2 ** (layers + size - 1),
                 out_channels=2 ** (layers + size),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False))

    encoders = nn.ModuleList(encoders)
    decoders = nn.ModuleList(decoders)

    upscale_layer = torch.nn.Identity()
    if self.upscale:
      upscale_layer = ConvBlock2(in_channels=2 ** size,
                                 out_channels=2 ** size,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 use_wrap_padding=use_wrap_padding,
                                 upscale=True,
                                 pooling=False)

    final_conv = ConvBlock(in_channels=2 ** size,
                           out_channels=output_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           use_batch_norm=False,
                           use_wrap_padding=use_wrap_padding,
                           use_activation=False,
                           upscale=False)

    unet = UNet2(encoders=encoders, decoders=decoders)
    return {"unet": unet, "final_conv": final_conv, "upscale_layer": upscale_layer}

  def build_v3_model(self, input_channels, output_channels, size, layers,
                     use_wrap_padding):
    """Builds the v3 model.

    Args:
      input_channels: Input channels.
      output_channels: Output channels.
      size: Size
      layers: Number of layers.
      use_wrap_padding: Wrap padding.

    Returns:
      Model
    """

    encoders = [
      ConvBlock2(in_channels=input_channels,
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 upscale=False,
                 use_wrap_padding=use_wrap_padding)
    ]
    decoders = [
      ConvBlock2(in_channels=2 ** (size + 1),
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False)
    ]
    decoders2 = [
      ConvBlock2(in_channels=2 ** (size + 1),
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False)
    ]

    for i in range(1, layers):
      channels = 2 ** (i + size - 1)
      encoders.append(
        ConvBlock2(in_channels=channels,
                   out_channels=2 * channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False))
      decoders.append(
        ConvBlock2(in_channels=4 * channels,
                   out_channels=channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False,
                   pooling=False))
      decoders2.append(
        ConvBlock2(in_channels=4 * channels,
                   out_channels=channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False,
                   pooling=False))

    encoders.append(
      ConvBlock2(in_channels=2 ** (layers + size - 1),
                 out_channels=2 ** (layers + size),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False))

    encoders = nn.ModuleList(encoders)
    decoders = nn.ModuleList(decoders)
    decoders2 = nn.ModuleList(decoders2)

    final_conv = ConvBlock(in_channels=2 ** size,
                           out_channels=output_channels - 1,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           use_batch_norm=False,
                           use_wrap_padding=use_wrap_padding,
                           use_activation=False,
                           upscale=False)
    final_conv2 = ConvBlock(in_channels=2 ** size,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            use_batch_norm=False,
                            use_wrap_padding=use_wrap_padding,
                            use_activation=False,
                            upscale=False)

    return {
      "encoders": encoders,
      "decoders": decoders,
      "decoders2": decoders2,
      "final_conv": final_conv,
      "final_conv2": final_conv2
    }

  def build_v4_model(self, input_channels, output_channels, size, layers,
                     use_wrap_padding):
    """Builds the v4 model.

    Args:
      input_channels: Input channels.
      output_channels: Output channels.
      size: Size
      layers: Number of layers.
      use_wrap_padding: Wrap padding.

    Returns:
      Model
    """

    encoders = [
      ConvBlock2(in_channels=input_channels // 2,
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 upscale=False,
                 use_wrap_padding=use_wrap_padding)
    ]
    decoders = [
      ConvBlock2(in_channels=2 ** (size + 1),
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False)
    ]
    for i in range(1, layers):
      channels = 2 ** (i + size - 1)
      encoders.append(
        ConvBlock2(in_channels=channels,
                   out_channels=2 * channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False))
      decoders.append(
        ConvBlock2(in_channels=4 * channels,
                   out_channels=channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False,
                   pooling=False))
    encoders.append(
      ConvBlock2(in_channels=2 ** (layers + size - 1),
                 out_channels=2 ** (layers + size),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False))
    encoders = nn.ModuleList(encoders)
    decoders = nn.ModuleList(decoders)

    upscale_layer = torch.nn.Identity()
    if self.upscale:
      upscale_layer = ConvBlock2(in_channels=2 ** size,
                                 out_channels=2 ** size,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 use_wrap_padding=use_wrap_padding,
                                 upscale=True,
                                 pooling=False)

    final_conv = ConvBlock(in_channels=2 ** size,
                           out_channels=output_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           use_batch_norm=False,
                           use_wrap_padding=use_wrap_padding,
                           use_activation=False,
                           upscale=False)

    weight_map_encoders = [
      ConvBlock2(in_channels=input_channels,
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 upscale=False,
                 use_wrap_padding=use_wrap_padding)
    ]
    weight_map_decoders = [
      ConvBlock2(in_channels=2 ** (size + 1),
                 out_channels=1,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False,
                 use_activation=False)
    ]
    weight_map_layers = 2
    for i in range(1, weight_map_layers):
      channels = 2 ** (i + size - 1)
      weight_map_encoders.append(
        ConvBlock2(in_channels=channels,
                   out_channels=2 * channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False))
      weight_map_decoders.append(
        ConvBlock2(in_channels=4 * channels,
                   out_channels=channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False,
                   pooling=False))
    weight_map_encoders.append(
      ConvBlock2(in_channels=2 ** (weight_map_layers + size - 1),
                 out_channels=2 ** (weight_map_layers + size),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 pooling=False))
    weight_map_encoders = nn.ModuleList(weight_map_encoders)
    weight_map_decoders = nn.ModuleList(weight_map_decoders)
    self.weight_map_network = UNet2(
      encoders=weight_map_encoders,
      decoders=weight_map_decoders)
    self.encoders = encoders
    self.decoders = decoders
    self.final_conv = final_conv
    self.upscale_layer = upscale_layer
    return

  def forward_thru_v1_model(self, input):
    """Forward pass through the v1 model.

    Args:
      input: Inputs.

    Returns:
      Output of the network.

    """
    # Change to channels first.
    x = input.permute((0, 3, 1, 2))
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

    # Change back to channels last.
    x = x.permute((0, 2, 3, 1))
    if self.use_residual:
      return x + input
    return x

  def forward_thru_v2_model(self, input):
    """Forward pass through the v2 model.

    Args:
      input: Input image.

    Returns:
      Output of the forward pass.

    """
    x = input.permute((0, 3, 1, 2))
    x = self.unet(x)
    if self.upscale:
      x = self.upscale_layer(x)[0]
    x = self.final_conv(x)
    x = x.permute((0, 2, 3, 1))
    return x

  def forward_thru_v3_model(self, input):
    """Forward pass through the v1 model.

    Args:
      input: Inputs.

    Returns:
      Output of the network.

    """
    # Change to channels first.
    x = input.permute((0, 3, 1, 2))
    x_all = []

    for i in range(len(self.encoders)):
      x, x_unpooled = self.encoders[i](x)
      x_all.append(x_unpooled)

    x = torch.nn.functional.interpolate(x,
                                        scale_factor=2,
                                        mode=self.interpolation,
                                        align_corners=False)

    x1 = x
    x2 = x

    x1, _ = self.decoders[-1](x1)

    for i in range(len(self.decoders) - 2, -1, -1):
      if self.decoders[i] is not None:
        x1 = torch.nn.functional.interpolate(x1,
                                             scale_factor=2,
                                             mode=self.interpolation,
                                             align_corners=False)
        x1 = torch.cat((x1, x_all[i]), dim=1)
        x1, _ = self.decoders[i](x1)

    x1 = self.final_conv(x1)
    x1 = x1.permute((0, 2, 3, 1))

    x2, _ = self.decoders[-1](x2)

    for i in range(len(self.decoders) - 2, -1, -1):
      if self.decoders[i] is not None:
        x2 = torch.nn.functional.interpolate(x2,
                                             scale_factor=2,
                                             mode=self.interpolation,
                                             align_corners=False)
        x2 = torch.cat((x2, x_all[i]), dim=1)
        x2, _ = self.decoders2[i](x2)

    x2 = self.final_conv2(x2)
    x2 = x2.permute((0, 2, 3, 1))

    return torch.cat((x1, x2), dim=3)

  def forward_thru_v4_model(self, input):
    """Forward pass through the v1 model.

    Args:
      input: Inputs.

    Returns:
      Output of the network.

    """
    # Change to channels first.
    x = input.permute((0, 3, 1, 2))
    num_input_channels = x.shape[1]

    weight_map = torch.sigmoid(self.weight_map_network(x))

    x1 = x[:, :(num_input_channels // 2)]
    x2 = x[:, (num_input_channels // 2):]

    x_all = []
    for i in range(len(self.encoders)):
      x1, x1_unpooled = self.encoders[i](x1)
      x2, x2_unpooled = self.encoders[i](x2)
      resized_weight_map = torch.nn.functional.interpolate(
        weight_map,
        (x1_unpooled.shape[2], x1_unpooled.shape[3]),
        mode="bilinear",
        align_corners=False
      )
      fused_x = (1 - resized_weight_map) * x1_unpooled + resized_weight_map * x2_unpooled
      x_all.append(fused_x)
    resized_weight_map = torch.nn.functional.interpolate(
      weight_map,
      (x1.shape[2], x1.shape[3]),
      mode="bilinear",
      align_corners=False
    )
    x = (1 - resized_weight_map) * x1 + resized_weight_map * x2

    x = torch.nn.functional.interpolate(x,
                                        scale_factor=2,
                                        mode=self.interpolation,
                                        align_corners=False)
    x, _ = self.decoders[-1](x)

    for i in range(len(self.decoders) - 2, -1, -1):
      if self.decoders[i] is not None:
        x = torch.nn.functional.interpolate(
          x,
          scale_factor=2,
          mode=self.interpolation,
          align_corners=False)
        x = torch.cat((x, x_all[i]), dim=1)
        x, _ = self.decoders[i](x)

    x = self.final_conv(x)
    x = x.permute((0, 2, 3, 1))

    return {
      "output": x,
      "weight_map": weight_map.permute((0, 2, 3, 1))
    }

  def forward(self, input, **kwargs):
    """Forward pass through the inpainting network.

    Args:
      input: Input image as (B, H, W, C) tensor.
      **kwargs:

    Returns:
      Output of the forward pass. Ideally a full image.

    """

    if self.model_version == "v1":
      return self.forward_thru_v1_model(input)
    elif self.model_version == "v2":
      return self.forward_thru_v2_model(input)
    elif self.model_version == "v3":
      return self.forward_thru_v3_model(input)
    elif self.model_version == "v4":
      return self.forward_thru_v4_model(input)
    else:
      raise ValueError("Unknown inpainting model")
