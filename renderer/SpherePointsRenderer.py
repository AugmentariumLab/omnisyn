# Lint as: python3
"""
A sphere points renderer based on Pytorch3d which also outputs z-depth images.
"""

import torch
from pytorch3d.renderer.points.renderer import PointsRenderer

class SpherePointsRenderer(PointsRenderer):
  """
  SpherePointsRenderer class renders points into RGB and depth images.
  """

  def forward(self, point_clouds, **kwargs):
    """Render a point cloud.

    Args:
      point_clouds:Input point cloud.
      **kwargs:

    Returns:
      images: RGB panos as (N, H, W, C) tensor.
      depth_images: Depth images as (N, H, W, K) tensor.

    """
    fragments = self.rasterizer(point_clouds, **kwargs)

    r = self.rasterizer.raster_settings.radius

    dists2 = fragments.dists.permute(0, 3, 1, 2)
    weights = 1 - dists2 / (r * r)
    images = self.compositor(
      fragments.idx.long().permute(0, 3, 1, 2),
      weights,
      point_clouds.features_packed().permute(1, 0),
      **kwargs
    )

    # Permute to channels last.
    images = images.permute(0, 2, 3, 1)
    depth_images = fragments.zbuf

    return images, depth_images
