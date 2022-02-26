# Lint as: python3
"""Generates sphere meshes using the UV sphere.
"""

import numpy as np
import torch
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer import (look_at_view_transform,
                                OpenGLPerspectiveCameras, RasterizationSettings)
from pytorch3d.structures.meshes import Meshes

from helpers import my_torch_helpers
from renderer.SphereMeshRasterizer import SphereMeshRasterizer
from renderer.SphereMeshRenderer import SphereMeshRenderer
from renderer.TextureShader import TextureShader


class SphereMeshGenerator:
  """This class is used for generating sphere meshes.
  Meshes are output using PyTorch3d's Mesh format.
  """

  def __init__(self):
    """Initializes a SphereMeshGenerator.

    """
    pass

  def generate_depth_mask(self, depth, threshold=999.0):
    """Generates a depth mask by filtering out areas based on the gradient.

    Args:
      depth: Depth image as (B, H, W, 1) tensor.
      threshold: Threshold for depth mask.

    Returns:
      Depth mask as a (B, H, W, 1) tensor of floats.

    """
    depths_grad_y = depth[:, 1:, :, :] - depth[:, :-1, :, :]
    depths_grad_y = torch.cat((depths_grad_y, depths_grad_y[:, 0:1]), dim=1)
    depths_grad_y = torch.lt(torch.abs(depths_grad_y), threshold)
    depths_grad_x = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    depths_grad_x = torch.cat((depths_grad_x, depths_grad_x[:, :, 0:1]), dim=2)
    depths_grad_x = torch.lt(torch.abs(depths_grad_x), threshold)
    depths_grad_mask = depths_grad_x & depths_grad_y
    if not torch.any(depths_grad_mask):
      print("Discarding all faces!!")
      exit(0)
    return depths_grad_mask

  def generate_mesh(self,
                    depth_image,
                    rgb_image,
                    width_segments=512,
                    height_segments=512,
                    apply_depth_mask=True,
                    facing_inside=True,
                    mesh_removal_threshold=2.0,
                    depth_mask=None,
                    device=None,
                    dtype=torch.float32):
    """Generates a mesh from the provided depth image.

    Args:
      depth_image: Depth image as a (B, H, W, 1) tensor.
      rgb_image: RGB Panorama as a (B, H, W, 3) tensor.
      width_segments: Width segments.
      height_segments: Height segments.
      apply_depth_mask: Delete triangles based on depth mask
      dtype: Dtype of the output mesh. Default is float32.
      device: Device of the output mesh. Default matches depth_image.

    Returns:
      Mesh with vertices set based on the depth image.

    """
    EPS = 1e-5
    if device is None:
      device = depth_image.device

    batch_size = depth_image.shape[0]
    depth = my_torch_helpers.resize_torch_images(
      depth_image, (width_segments, height_segments), mode="nearest")
    if depth_mask is None:
      depth_mask = self.generate_depth_mask(
        depth,
        threshold=mesh_removal_threshold)

    r = depth[:, :, :, 0]
    r = torch.cat((r, r[:, 0:1]), dim=1)
    r = torch.cat((r, r[:, :, 0:1]), dim=2)

    theta = torch.arange(0, width_segments + 1, device=device,
                         dtype=dtype) * (2 * np.pi / width_segments)
    theta = torch.clamp(theta, EPS, 2 * np.pi - EPS) + np.pi / 2
    phi = torch.arange(0, height_segments + 1, device=device,
                       dtype=dtype) * (np.pi / height_segments)
    phi = torch.clamp(phi, EPS, np.pi - EPS)
    phi, theta = torch.meshgrid(phi, theta)
    phi = phi[None, :, :].expand(
      (batch_size, height_segments + 1, width_segments + 1))
    theta = theta[None, :, :].expand(
      (batch_size, height_segments + 1, width_segments + 1))

    verts = my_torch_helpers.spherical_to_cartesian(theta, phi, r=r)
    verts = verts.reshape((batch_size, -1, 3))
    verts = list(torch.unbind(verts, dim=0))

    padded_rgb_image = torch.cat(
      (rgb_image[:, :, -1:, :], rgb_image, rgb_image[:, :, 0:1, :]),
      dim=2)
    padded_rgb_width = padded_rgb_image.shape[2]
    theta = torch.linspace(0.0 + 1 / padded_rgb_width,
                           1.0 - 1 / padded_rgb_width,
                           width_segments + 1, device=device, dtype=dtype)
    phi = torch.linspace(0, 1, height_segments + 1, device=device, dtype=dtype)
    phi, theta = torch.meshgrid(phi, theta)
    verts_uvs = torch.stack((theta, 1 - phi), dim=2)
    verts_uvs = verts_uvs.reshape((1, -1, 2)).expand((batch_size, -1, -1))
    verts_uvs = list(torch.unbind(verts_uvs, dim=0))

    vertex_idx = torch.arange(0, (height_segments + 1) * (width_segments + 1),
                              device=device,
                              dtype=torch.long)

    vertex_idx = vertex_idx.reshape((height_segments + 1, width_segments + 1))
    face_idx_top_left = vertex_idx[:-1, :-1]
    face_idx_top_right = vertex_idx[:-1, 1:]
    face_idx_bottom_left = vertex_idx[1:, :-1]
    face_idx_bottom_right = vertex_idx[1:, 1:]
    if facing_inside:
      face_idx_a = torch.stack(
        (face_idx_top_right, face_idx_top_left, face_idx_bottom_left), dim=2)
      face_idx_b = torch.stack(
        (face_idx_bottom_right, face_idx_top_right, face_idx_bottom_left),
        dim=2)
    else:
      face_idx_a = torch.stack(
        (face_idx_bottom_left, face_idx_top_left, face_idx_top_right), dim=2)
      face_idx_b = torch.stack(
        (face_idx_bottom_left, face_idx_top_right, face_idx_bottom_right),
        dim=2)

    faces = []
    faces_uvs = []
    for i in range(batch_size):
      if apply_depth_mask:
        face_idx_a_keep = face_idx_a[depth_mask[i, :, :, 0]]
        face_idx_b_keep = face_idx_b[depth_mask[i, :, :, 0]]
      else:
        face_idx_a_keep = face_idx_a.reshape(-1, 3)
        face_idx_b_keep = face_idx_b.reshape(-1, 3)
      face_idx = torch.cat((face_idx_a_keep, face_idx_b_keep), dim=0)
      faces.append(face_idx)
      faces_uvs.append(face_idx)

    # rgb_image_list = list(torch.unbind(rgb_image, dim=0))
    # faces_uvs = torch.stack(faces_uvs, dim=0)
    # verts_uvs = torch.stack(verts_uvs, dim=0)
    # faces = torch.stack(faces, dim=0)
    # verts = torch.stack(verts, dim=0)
    # print("shapes", faces_uvs.shape, verts_uvs.shape, faces.shape, verts.shape,
    #       rgb_image.shape)
    textures = TexturesUV(
      maps=padded_rgb_image,
      faces_uvs=faces_uvs,
      verts_uvs=verts_uvs)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return meshes

  def apply_rot_trans(self, mesh, rot, trans, inv_rot=False, inv_trans=False):
    """Applies a rotation and translation to a mesh.

    Args:
      mesh: Mesh on which to apply the rotation.
      rot: Rotation to apply as a (B, 3, 3) tensor.
      trans: Translation to apply as a (B, 3) tensor.
      inv_rot: Invert the rotation. (Unimplemented)
      inv_trans: Invert the translation. (Unimplemented)

    Returns:
      New mesh (shallow copied) with the new rotation and translation.
    """
    if mesh.isempty():
      print("Mesh is empty!")
      exit(0)
    verts = mesh.verts_padded()

    m_rot = rot
    if inv_rot:
      m_rot = torch.inverse(rot)

    verts_shifted = torch.matmul(m_rot[:, None, :, :], verts[:, :, :, None])
    verts_shifted = verts_shifted[:, :, :, 0]
    verts_shifted = verts_shifted - trans[:, None, :]
    meshes_shifted = mesh.update_padded(new_verts_padded=verts_shifted)
    return meshes_shifted

  def render_mesh(self, mesh, image_size=256, device="cuda"):
    """Render a mesh.

    Args:
      mesh: Mesh.
      image_size: Image size.
      device: Device.

    Returns:
      Rendered image.
      Rendered depth.

    """
    raster_settings = RasterizationSettings(image_size=image_size,
                                            blur_radius=0.0,
                                            faces_per_pixel=1,
                                            cull_backfaces=True)

    R, T = look_at_view_transform(eye=((1, 10, 150),), at=((0, 10, 0),))
    cameras = OpenGLPerspectiveCameras(device=device,
                                       R=R,
                                       T=T,
                                       fov=90,
                                       znear=0.01,
                                       zfar=500.0)
    left_rasterizer = SphereMeshRasterizer(cameras=cameras,
                                           raster_settings=raster_settings,
                                           side="left")
    right_rasterizer = SphereMeshRasterizer(cameras=cameras,
                                            raster_settings=raster_settings,
                                            side="right")
    shader = TextureShader(device=device, cameras=cameras)
    renderer = SphereMeshRenderer(rasterizer=left_rasterizer, shader=shader,
                                  left_rasterizer=left_rasterizer,
                                  right_rasterizer=right_rasterizer)
    return renderer(mesh)

  def logits_to_depth_mask(self, logits, width, height):
    logits = my_torch_helpers.resize_torch_images(logits, (width, height))
    probabilities = torch.sigmoid(logits)
    random_values = torch.rand(
      probabilities.shape,
      dtype=probabilities.dtype,
      device=probabilities.device)
    return torch.lt(probabilities, random_values)
