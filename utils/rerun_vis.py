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


def init_rerun(recording_name: str, spawn: bool = True, web: bool = False) -> bool:
    """Initialize Rerun. Returns True if successful."""
    if not RERUN_AVAILABLE:
        print("[Rerun] rerun-sdk not installed. Install with: pip install rerun-sdk")
        return False
    rr.init(recording_name)
    if web:
        server_uri = rr.serve_grpc()
        rr.serve_web_viewer(open_browser=False, connect_to=server_uri)
        print(f"[Rerun] Web viewer at http://localhost:9090 (gRPC at {server_uri})")
    elif spawn:
        rr.spawn()
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


# --- Segmentation visualization ---

# Per-object color palette (distinguishable colors)
_OBJECT_COLORS = [
    [230, 25, 75],    # red
    [60, 180, 75],    # green
    [0, 130, 200],    # blue
    [255, 225, 25],   # yellow
    [245, 130, 48],   # orange
    [145, 30, 180],   # purple
    [70, 240, 240],   # cyan
    [240, 50, 230],   # magenta
    [210, 245, 60],   # lime
    [250, 190, 212],  # pink
    [0, 128, 128],    # teal
    [170, 110, 40],   # brown
]


def log_segmented_scene(
    rec,
    segmentation,
    pts3d_flat: np.ndarray,
    colors_flat: Optional[np.ndarray] = None,
):
    """Log segmented scene to Rerun with separate entities per object.

    Args:
        rec: Rerun recording handle (from rr.new_recording)
        segmentation: SceneSegmentation instance
        pts3d_flat: (N, 3) flattened 3D points
        colors_flat: (N, 3) optional RGB colors [0,1] or uint8
    """
    if not RERUN_AVAILABLE:
        return

    # Adaptive downsample limits based on object count to cap total at ~1M points
    n_objects = len(segmentation.objects)
    max_bg_pts = min(500_000, 1_000_000 // max(n_objects + 1, 1))
    max_obj_pts = min(200_000, 1_000_000 // max(n_objects, 1))
    print(f"[Rerun] Logging {n_objects} objects (max {max_obj_pts} pts/obj, {max_bg_pts} bg pts)...")

    # Prepare colors
    if colors_flat is not None and colors_flat.dtype != np.uint8:
        colors_u8 = (np.clip(colors_flat, 0, 1) * 255).astype(np.uint8)
    else:
        colors_u8 = colors_flat

    # Log background points
    if segmentation.background_indices is not None and len(segmentation.background_indices) > 0:
        bg_idx = segmentation.background_indices
        bg_pts = pts3d_flat[bg_idx]
        bg_colors = colors_u8[bg_idx] if colors_u8 is not None else None

        # Downsample background if large
        if len(bg_pts) > max_bg_pts:
            sample = np.random.choice(len(bg_pts), max_bg_pts, replace=False)
            bg_pts = bg_pts[sample]
            bg_colors = bg_colors[sample] if bg_colors is not None else None

        rec.log("world/background/points", rr.Points3D(
            positions=bg_pts, colors=bg_colors, radii=0.003,
        ), static=True)

    # Log each object
    for i, obj in enumerate(segmentation.objects):
        color = _OBJECT_COLORS[i % len(_OBJECT_COLORS)]
        safe_label = obj.label.replace(" ", "_")
        entity = f"world/objects/{safe_label}_{obj.object_id}"

        obj_pts = pts3d_flat[obj.point_indices]

        # Downsample if large
        if len(obj_pts) > max_obj_pts:
            sample = np.random.choice(len(obj_pts), max_obj_pts, replace=False)
            obj_pts = obj_pts[sample]

        # Use object-specific color for distinction
        obj_colors = np.tile(np.array(color, dtype=np.uint8), (len(obj_pts), 1))

        rec.log(f"{entity}/points", rr.Points3D(
            positions=obj_pts, colors=obj_colors, radii=0.005,
        ), static=True)

        # Log bounding box
        bbox_min, bbox_max = obj.bbox_3d
        center = (bbox_min + bbox_max) / 2
        half_size = (bbox_max - bbox_min) / 2
        rec.log(f"{entity}/bbox", rr.Boxes3D(
            centers=[center],
            half_sizes=[half_size],
            colors=[color],
            labels=[f"{obj.label} ({obj.confidence:.2f})"],
        ), static=True)

    print(f"[Rerun] Segmented scene logged.")


def update_object_transform(rec, object_id: int, label: str, transform):
    """Re-log only the Transform3D for an object entity (fast update).

    Args:
        rec: Rerun recording handle
        object_id: object ID
        label: object label string
        transform: ObjectTransform instance with translation, rotation_euler, scale
    """
    if not RERUN_AVAILABLE:
        return

    from scipy.spatial.transform import Rotation as R

    safe_label = label.replace(" ", "_")
    entity = f"world/objects/{safe_label}_{object_id}"

    # Build 3x3 rotation matrix from euler angles
    rot_mat = R.from_euler("xyz", transform.rotation_euler, degrees=True).as_matrix()

    # Build scale matrix
    scale_mat = np.diag(transform.scale)

    # Combined transform: scale then rotate
    mat3x3 = rot_mat @ scale_mat

    rec.log(entity, rr.Transform3D(
        translation=transform.translation,
        mat3x3=mat3x3,
    ))
