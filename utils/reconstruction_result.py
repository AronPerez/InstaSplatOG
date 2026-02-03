from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class ReconstructionResult:
    """Model-agnostic interface for geometry initialization output.

    Any model (MASt3R, DUSt3R, custom) should produce this format.
    """
    # Required: flattened point cloud
    points: np.ndarray              # (N, 3) world coordinates
    camera_poses_c2w: np.ndarray    # (M, 4, 4) camera-to-world transforms
    intrinsics: np.ndarray          # (M, 3, 3) K matrices

    # Optional
    colors: Optional[np.ndarray] = None           # (N, 3) RGB [0,1]
    images: Optional[np.ndarray] = None           # (M, H, W, 3) RGB [0,1]
    confidence: Optional[np.ndarray] = None       # (N,) confidence scores
    depth_maps: Optional[np.ndarray] = None       # (M, H, W)
    image_names: Optional[List[str]] = None

    @classmethod
    def from_scene(cls, scene, imgs: np.ndarray,
                   pts3d: np.ndarray,
                   confs: Optional[np.ndarray] = None,
                   depthmaps: Optional[np.ndarray] = None,
                   image_files: Optional[list] = None,
                   conf_threshold: float = 1.0) -> "ReconstructionResult":
        """Extract from InstantSplat scene object after global alignment.

        Args:
            scene: Scene from global_aligner (any model).
                   Used only to get camera poses and intrinsics.
            imgs: (M, H, W, 3) numpy array [0,1] -- from np.array(scene.imgs)
            pts3d: (M, H, W, 3) numpy array -- from to_numpy(scene.get_pts3d())
            confs: (M, H, W) confidence maps -- from scene.im_conf
            depthmaps: (M, H, W) depth maps -- from scene.im_depthmaps
            image_files: list of image file paths
            conf_threshold: minimum confidence to include a point
        """
        from dust3r.utils.device import to_numpy

        # c2w poses directly from scene (before inversion to w2c)
        camera_poses_c2w = to_numpy(scene.get_im_poses())  # (M, 4, 4)
        intrinsics_K = to_numpy(scene.get_intrinsics())     # (M, 3, 3)

        # Flatten points and colors, filter by confidence
        n_views = pts3d.shape[0]
        all_points, all_colors, all_conf = [], [], []
        for i in range(n_views):
            pts_flat = pts3d[i].reshape(-1, 3)
            col_flat = imgs[i].reshape(-1, 3)
            if confs is not None:
                conf_flat = confs[i].reshape(-1)
                mask = conf_flat > conf_threshold
                pts_flat = pts_flat[mask]
                col_flat = col_flat[mask]
                all_conf.append(conf_flat[mask])
            all_points.append(pts_flat)
            all_colors.append(col_flat)

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        confidence = np.concatenate(all_conf, axis=0) if all_conf else None

        names = None
        if image_files:
            from pathlib import Path
            names = [Path(f).name for f in image_files]

        return cls(
            points=points,
            camera_poses_c2w=camera_poses_c2w,
            intrinsics=intrinsics_K,
            colors=colors,
            images=imgs,
            confidence=confidence,
            depth_maps=depthmaps,
            image_names=names,
        )
