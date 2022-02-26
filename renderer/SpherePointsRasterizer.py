# Lint as: python3
"""A Points rasterizer to create 360 equirectangular images from point clouds.

Based on pytorch3d/renderer/rasterizer.py
https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/renderer/points/rasterizer.py
"""

import numpy as np
import torch
from pytorch3d.renderer.points.rasterizer import PointsRasterizer

from helpers import my_torch_helpers


class SpherePointsRasterizer(PointsRasterizer):
  """
  This class implements methods for rasterizing a batch of pointclouds to an
  equirectangular image.
  """

  def __init__(self, cameras=None, raster_settings=None,
               linearize_angle=np.deg2rad(10)):
    super().__init__(cameras, raster_settings)
    self.linearize_angle = linearize_angle

  def transform(self, point_clouds, **kwargs) -> torch.Tensor:
    """
    Args:
        point_clouds: a set of point clouds
    Returns:
        points_screen: the points with the vertex positions in screen
        space
    NOTE: keeping this as a separate function for readability but it could
    be moved into forward.
    """

    pts_world = point_clouds.points_padded()
    pts_world_packed = point_clouds.points_packed()
    pts_camera = torch.tensordot(
      self.cameras.R, pts_world.reshape((-1, 3)), dims=([0], [1])).permute(
      (1, 0)) - self.cameras.T
    pts_camera = pts_camera.reshape(pts_world.shape)

    pts_screen = my_torch_helpers.cartesian_to_spherical(
      pts_camera,
      linearize_angle=self.linearize_angle)
    u = 2 * (-torch.fmod(pts_screen[:, :, 0] + 3 * np.pi / 2 + 2 * np.pi,
                         2 * np.pi) / (2 * np.pi)) + 1
    v = 2 * (-pts_screen[:, :, 1] / np.pi) + 1
    pts_screen = torch.stack((u, v, pts_screen[:, :, 2]), dim=-1)

    # Offset points of input pointcloud to reuse cached padded/packed calculations.
    pad_to_packed_idx = point_clouds.padded_to_packed_idx()
    pts_screen_packed = pts_screen.view(-1, 3)[pad_to_packed_idx, :]
    pts_packed_offset = pts_screen_packed - pts_world_packed
    point_clouds = point_clouds.offset(pts_packed_offset)
    return point_clouds
