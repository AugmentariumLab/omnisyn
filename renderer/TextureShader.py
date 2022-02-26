# Lint as: python3
"""A basic shader which returns textures without any lighting.
"""
import torch
import torch.nn as nn

from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.shading import flat_shading, gouraud_shading, phong_shading


class TextureShader(nn.Module):
  """
  Basic shader which just returns the texels.
  """

  def __init__(self,
               device="cpu",
               cameras=None,
               lights=None,
               materials=None,
               blend_params=None):
    super().__init__()
    self.lights = lights if lights is not None else PointLights(device=device)
    self.materials = (materials if materials is not None else Materials(
        device=device))
    self.cameras = cameras
    self.blend_params = blend_params if blend_params is not None else BlendParams(
    )

  def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
    # cameras = kwargs.get("cameras", self.cameras)
    # if cameras is None:
    #   msg = "Cameras must be specified either at initialization \
    #           or in the forward pass of TexturedSoftPhongShader"
    #
    #   raise ValueError(msg)
    texels = meshes.sample_textures(fragments)
    # lights = kwargs.get("lights", self.lights)
    # materials = kwargs.get("materials", self.materials)
    # blend_params = kwargs.get("blend_params", self.blend_params)
    # colors = phong_shading(
    #     meshes=meshes,
    #     fragments=fragments,
    #     texels=texels,
    #     lights=lights,
    #     cameras=cameras,
    #     materials=materials,
    # )
    # images = softmax_rgb_blend(colors, fragments, blend_params)
    # print("Texels shape", texels.shape)
    return texels[:, :, :, 0, :]
    # return images
