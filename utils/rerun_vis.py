"""Rerun visualization utilities for InstantSplat.

All functions gracefully no-op when rerun-sdk is not installed.
"""
import numpy as np
from typing import Optional

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


def is_rerun_available() -> bool:
    return RERUN_AVAILABLE


def init_rerun(recording_name: str, spawn: bool = True) -> bool:
    """Initialize Rerun. Returns True if successful."""
    if not RERUN_AVAILABLE:
        print("[Rerun] rerun-sdk not installed. Install with: pip install rerun-sdk")
        return False
    rr.init(recording_name, spawn=spawn)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    return True


def log_point_cloud(entity_path: str, points: np.ndarray,
                    colors: Optional[np.ndarray] = None,
                    radii: Optional[float] = None,
                    static: bool = False):
    """Log Points3D. Auto-downsamples above 1M points."""
    if not RERUN_AVAILABLE:
        return
    # Downsample if needed
    if len(points) > 1_000_000:
        idx = np.random.choice(len(points), 1_000_000, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
        print(f"[Rerun] Downsampled to 1M points")

    # Convert [0,1] float colors to uint8
    if colors is not None and colors.dtype != np.uint8:
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    rr.log(entity_path, rr.Points3D(positions=points, colors=colors,
                                      radii=radii), static=static)


def log_camera(entity_path: str, pose_c2w: np.ndarray, K: np.ndarray,
               image: Optional[np.ndarray] = None, static: bool = False):
    """Log camera transform + pinhole + optional image."""
    if not RERUN_AVAILABLE:
        return
    rr.log(entity_path,
           rr.Transform3D(translation=pose_c2w[:3, 3],
                          mat3x3=pose_c2w[:3, :3]),
           static=static)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if image is not None:
        h, w = image.shape[:2]
    else:
        w, h = int(cx * 2), int(cy * 2)

    rr.log(entity_path,
           rr.Pinhole(focal_length=[fx, fy],
                      principal_point=[cx, cy],
                      width=w, height=h),
           static=static)

    if image is not None:
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) \
            if image.dtype != np.uint8 else image
        rr.log(f"{entity_path}/image", rr.Image(img_uint8), static=static)


def log_depth_map(entity_path: str, depth: np.ndarray, static: bool = False):
    """Log a depth map."""
    if not RERUN_AVAILABLE:
        return
    rr.log(entity_path, rr.DepthImage(depth), static=static)


def log_reconstruction(result, entity_prefix: str = "world",
                       log_depths: bool = False, static: bool = True):
    """Log a full ReconstructionResult."""
    if not RERUN_AVAILABLE:
        return

    # Point cloud
    if result.points is not None:
        log_point_cloud(f"{entity_prefix}/points", result.points,
                        colors=result.colors, radii=0.005, static=static)
        print(f"[Rerun] Logged {len(result.points)} points")

    # Cameras
    n = len(result.camera_poses_c2w)
    for i in range(n):
        name = result.image_names[i] if result.image_names else f"cam_{i:03d}"
        img = result.images[i] if result.images is not None else None
        log_camera(f"{entity_prefix}/cameras/{name}",
                   result.camera_poses_c2w[i], result.intrinsics[i],
                   image=img, static=static)
        if log_depths and result.depth_maps is not None:
            log_depth_map(f"{entity_prefix}/cameras/{name}/depth",
                          result.depth_maps[i], static=static)
    print(f"[Rerun] Logged {n} cameras")


def log_gaussians(entity_path: str, gaussians, max_points: int = 100_000):
    """Log Gaussian splat centers as a point cloud on the current timeline.

    Args:
        entity_path: Rerun entity path
        gaussians: GaussianModel instance -- uses .get_xyz (property),
                   ._features_dc, .get_opacity (property), .get_scaling (property)
        max_points: max points to visualize
    """
    if not RERUN_AVAILABLE:
        return
    from utils.sh_utils import SH2RGB

    # All accessed as @property (no parens)
    xyz = gaussians.get_xyz.detach().cpu().numpy()              # (N, 3)
    features_dc = gaussians._features_dc.detach().cpu()         # (N, 1, 3)
    opacity = gaussians.get_opacity.detach().cpu().numpy()      # (N, 1)
    scaling = gaussians.get_scaling.detach().cpu().numpy()       # (N, 3)

    # SH DC -> RGB: SH2RGB(sh) = sh * C0 + 0.5 (utils/sh_utils.py)
    sh_dc = features_dc[:, 0, :]                                # (N, 3) torch tensor
    colors = SH2RGB(sh_dc).clamp(0, 1).numpy()                 # (N, 3)

    # Downsample
    n = len(xyz)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        xyz, colors, opacity, scaling = xyz[idx], colors[idx], opacity[idx], scaling[idx]

    radii = scaling.mean(axis=1) * 0.3
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    rr.log(entity_path, rr.Points3D(positions=xyz, colors=colors_uint8, radii=radii))


def log_scalar(entity_path: str, value: float):
    """Log a scalar value (e.g. training loss)."""
    if not RERUN_AVAILABLE:
        return
    rr.log(entity_path, rr.Scalar(value))
