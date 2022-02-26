# Lint as: python3
"""SphereMeshRenderer holds our custom renderer which outputs the depth.
"""

import torch
from pytorch3d.renderer.mesh.renderer import MeshRenderer


class SphereMeshRenderer(MeshRenderer):

  def __init__(self, rasterizer, shader, left_rasterizer, right_rasterizer):
    super().__init__(rasterizer, shader)
    self.rasterizer = rasterizer
    self.shader = shader
    self.left_rasterizer = left_rasterizer
    self.right_rasterizer = right_rasterizer

  def forward(self, meshes_world, **kwargs):
    """
    Render a batch of images from a batch of meshes by rasterizing and then
    shading.
    NOTE: If the blur radius for rasterization is > 0.0, some pixels can
    have one or more barycentric coordinates lying outside the range [0, 1].
    For a pixel with out of bounds barycentric coordinates with respect to a
    face f, clipping is required before interpolating the texture uv
    coordinates and z buffer so that the colors and depths are limited to
    the range for the corresponding face.
    """
    if self.left_rasterizer is not None and self.right_rasterizer is not None:
      left_fragments = self.left_rasterizer(meshes_world, **kwargs)
      left_images = self.shader(left_fragments, meshes_world, **kwargs)
      right_fragments = self.right_rasterizer(meshes_world, **kwargs)
      right_images = self.shader(right_fragments, meshes_world, **kwargs)
      half_width = left_images.shape[2] // 2
      images = torch.cat(
        (left_images[:, :, :half_width], right_images[:, :, half_width:]), dim=2)
      zbuf = torch.cat(
        (left_fragments.zbuf[:, :, :half_width], right_fragments.zbuf[:, :, half_width:]),
        dim=2)
      # return left_images, left_fragments.zbuf
      # print("Returning right images", right_images.shape, right_fragments.zbuf.shape)
      # return right_images, right_fragments.zbuf
      return images, zbuf

    fragments = self.rasterizer(meshes_world, **kwargs)
    images = self.shader(fragments, meshes_world, **kwargs)

    return images, fragments.zbuf
