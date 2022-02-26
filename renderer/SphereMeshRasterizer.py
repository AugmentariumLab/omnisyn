# Lint as: python3
"""A Points rasterizer to create 360 equirectangular images from meshes.

Based on pytorch3d/renderer/rasterizer.py
https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/renderer/points/rasterizer.py
"""

import numpy as np
import torch
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.structures.meshes import Meshes

from helpers import my_torch_helpers


class SphereMeshRasterizer(MeshRasterizer):
  """
  This class implements methods for rasterizing meshes to an
  equirectangular image.
  """

  def __init__(self, cameras=None, raster_settings=None, side="none"):
    """
    Args:
        cameras: A cameras object which has a  `transform_points` method
            which returns the transformed points after applying the
            world-to-view and view-to-screen
            transformations.
        raster_settings: the parameters for rasterization. This should be a
            named tuple.

    All these initial settings can be overridden by passing keyword
    arguments to the forward function.
    """
    super().__init__()
    if raster_settings is None:
      raster_settings = RasterizationSettings()

    self.cameras = cameras
    self.raster_settings = raster_settings
    self.side = side

  def discard_wraparound_triangles(self, verts_padded, faces_padded):
    # print("verts padded shape", verts_padded.shape)
    # print("faces padded shape", faces_padded.shape)
    faces_padded = faces_padded.clone()
    batch_size = verts_padded.shape[0]
    invalid_faces = faces_padded < 0
    faces_padded[invalid_faces] = 0
    for i in range(batch_size):
      vert_positions = torch.stack((verts_padded[i][faces_padded[i, :, 0]],
                                    verts_padded[i][faces_padded[i, :, 1]],
                                    verts_padded[i][faces_padded[i, :, 2]]),
                                   dim=1)
      wraps_around_0 = torch.any(vert_positions[:, :, 0] < -1, dim=1) & \
                       torch.any(vert_positions[:, :, 0] > 0, dim=1)
      wraps_around_1 = torch.any(vert_positions[:, :, 0] < 0, dim=1) & \
                       torch.any(vert_positions[:, :, 0] > 1, dim=1)
      wraps_around = wraps_around_0 | wraps_around_1
      faces_padded[i, wraps_around, :] = 0
    faces_padded[invalid_faces] = -1
    return verts_padded, faces_padded

  def transform(self, meshes_world, **kwargs):
    """
    Args:
        meshes_world: a Meshes object representing a batch of meshes with
            vertex coordinates in world space.
    Returns:
        meshes_screen: a Meshes object with the vertex positions in screen
        space
    NOTE: keeping this as a separate function for readability but it could
    be moved into forward.
    """
    cameras = kwargs.get("cameras", self.cameras)
    if cameras is None:
      msg = "Cameras must be specified either at initialization \
              or in the forward pass of MeshRasterizer"

      raise ValueError(msg)
    verts_world = meshes_world.verts_padded()
    faces_padded = meshes_world.faces_padded()

    pts_screen = my_torch_helpers.cartesian_to_spherical(verts_world)
    if self.side == "left":
      u = torch.fmod(pts_screen[:, :, 0] + 4 * np.pi,
                     2 * np.pi) - np.pi / 2
    elif self.side == "right":
      u = torch.fmod(pts_screen[:, :, 0] + 4 * np.pi - np.pi,
                     2 * np.pi) + np.pi / 2
    else:
      u = torch.fmod(pts_screen[:, :, 0] + 4 * np.pi - np.pi / 2, 2 * np.pi)
    u = 2 * (-u / (2 * np.pi)) + 1
    v = 2 * (-pts_screen[:, :, 1] / np.pi) + 1
    z = pts_screen[:, :, 2]
    verts_screen = torch.stack((u, v, z), dim=-1)

    # verts_view = cameras.get_world_to_view_transform(
    #     **kwargs).transform_points(verts_world)
    # verts_screen = cameras.get_projection_transform(
    #     **kwargs).transform_points(verts_view)
    # verts_screen[..., 2] = verts_view[..., 2]
    verts_padded, faces_padded = self.discard_wraparound_triangles(
      verts_screen, faces_padded
    )
    # meshes_screen = meshes_world.update_padded(new_verts_padded=verts_screen)
    # return meshes_screen
    return Meshes(verts=verts_screen, faces=faces_padded,
                  textures=meshes_world.textures)
