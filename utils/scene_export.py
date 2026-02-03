"""Multi-format scene export for edited Gaussian scenes.

Supports:
  - Gaussian PLY: same format as GaussianModel.save_ply()
  - PLY + JSON: standard colored point cloud + per-object metadata sidecar
  - glTF: trimesh-based export with separate mesh nodes per object
"""

import json
import os
import numpy as np
from typing import Optional


def export_gaussian_ply(
    editor,
    output_path: str,
    sh_degree: int = 3,
) -> str:
    """Export modified scene as a Gaussian PLY file.

    Uses the same format as GaussianModel.save_ply() so it can be
    loaded back with GaussianModel.load_ply().

    Args:
        editor: SceneEditor instance
        output_path: path for the output .ply file
        sh_degree: SH degree used (default 3, for attribute naming)

    Returns:
        The output file path
    """
    from plyfile import PlyData, PlyElement

    data = editor.get_full_scene_gaussians()

    xyz = data["xyz"]                   # (N, 3)
    normals = np.zeros_like(xyz)        # (N, 3)
    rotation = data["rotation"]         # (N, 4)
    scaling = data["scaling"]           # (N, 3)
    opacity = data["opacity"]           # (N, 1)
    features_dc = data["features_dc"]   # (N, 1, 3)
    features_rest = data["features_rest"]  # (N, K, 3)

    # Flatten SH features to match GaussianModel format
    # features_dc: (N, 1, 3) -> transpose to (N, 3, 1) -> flatten to (N, 3)
    f_dc = features_dc.transpose(0, 2, 1).reshape(xyz.shape[0], -1)
    # features_rest: (N, K, 3) -> transpose to (N, 3, K) -> flatten to (N, 3*K)
    f_rest = features_rest.transpose(0, 2, 1).reshape(xyz.shape[0], -1)

    # Build attribute names
    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]):
        attr_names.append(f'f_dc_{i}')
    for i in range(f_rest.shape[1]):
        attr_names.append(f'f_rest_{i}')
    attr_names.append('opacity')
    for i in range(scaling.shape[1]):
        attr_names.append(f'scale_{i}')
    for i in range(rotation.shape[1]):
        attr_names.append(f'rot_{i}')

    dtype_full = [(attr, 'f4') for attr in attr_names]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    attributes = np.concatenate([
        xyz, normals, f_dc, f_rest, opacity, scaling, rotation
    ], axis=1)
    elements[:] = list(map(tuple, attributes))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)

    print(f"[Export] Gaussian PLY saved to {output_path} ({xyz.shape[0]} Gaussians)")
    return output_path


def export_ply_with_json(
    editor,
    segmentation,
    output_dir: str,
) -> str:
    """Export scene as a colored PLY point cloud + JSON metadata sidecar.

    The PLY contains RGB-colored points (from SH DC coefficients).
    The JSON contains per-object metadata including transforms and bounding boxes.

    Args:
        editor: SceneEditor instance
        segmentation: SceneSegmentation instance
        output_dir: directory for output files

    Returns:
        Path to the output directory
    """
    from plyfile import PlyData, PlyElement

    os.makedirs(output_dir, exist_ok=True)

    data = editor.get_full_scene_gaussians()
    xyz = data["xyz"]
    features_dc = data["features_dc"]  # (N, 1, 3)

    # SH DC -> RGB
    C0 = 0.28209479177387814
    rgb = features_dc[:, 0, :] * C0 + 0.5
    rgb = np.clip(rgb, 0, 1)
    rgb_u8 = (rgb * 255).astype(np.uint8)

    # Write PLY
    ply_path = os.path.join(output_dir, "scene.ply")
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ]
    vertices = np.empty(len(xyz), dtype=dtype)
    vertices['x'] = xyz[:, 0]
    vertices['y'] = xyz[:, 1]
    vertices['z'] = xyz[:, 2]
    vertices['red'] = rgb_u8[:, 0]
    vertices['green'] = rgb_u8[:, 1]
    vertices['blue'] = rgb_u8[:, 2]

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(ply_path)

    # Write JSON sidecar
    json_path = os.path.join(output_dir, "scene_metadata.json")
    metadata = {
        "total_points": int(len(xyz)),
        "objects": [],
    }
    for obj in segmentation.objects:
        transform = editor.get_transform(obj.object_id)
        obj_meta = obj.to_dict()
        obj_meta["transform"] = transform.to_dict()
        # Don't include point_indices in JSON (too large); keep bbox/centroid
        del obj_meta["point_indices"]
        metadata["objects"].append(obj_meta)

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Export] PLY+JSON saved to {output_dir}")
    return output_dir


def export_gltf(
    editor,
    segmentation,
    output_path: str,
) -> str:
    """Export scene as glTF with separate mesh nodes per object.

    Uses trimesh to create a Scene with each object as a separate
    point cloud mesh node for easy manipulation in 3D tools.

    Args:
        editor: SceneEditor instance
        segmentation: SceneSegmentation instance
        output_path: path for the output .glb/.gltf file

    Returns:
        The output file path
    """
    import trimesh

    scene = trimesh.Scene()

    data = editor.get_full_scene_gaussians()
    xyz = data["xyz"]
    features_dc = data["features_dc"]  # (N, 1, 3)

    # SH DC -> RGB
    C0 = 0.28209479177387814
    rgb = features_dc[:, 0, :] * C0 + 0.5
    rgb = np.clip(rgb, 0, 1)
    rgba = np.column_stack([rgb, np.ones(len(rgb))])
    rgba_u8 = (rgba * 255).astype(np.uint8)

    # Add background as a point cloud
    if segmentation.background_indices is not None and len(segmentation.background_indices) > 0:
        bg_idx = segmentation.background_indices
        # Limit to manageable size for glTF
        if len(bg_idx) > 100_000:
            bg_idx = np.random.choice(bg_idx, 100_000, replace=False)
        bg_cloud = trimesh.PointCloud(
            vertices=xyz[bg_idx],
            colors=rgba_u8[bg_idx],
        )
        scene.add_geometry(bg_cloud, node_name="background")

    # Add each object
    for obj in segmentation.objects:
        obj_indices = editor._get_gaussian_indices(obj.object_id)
        if len(obj_indices) == 0:
            continue

        obj_pts = xyz[obj_indices]
        obj_colors = rgba_u8[obj_indices]

        # Limit per-object for glTF
        if len(obj_pts) > 50_000:
            sample = np.random.choice(len(obj_pts), 50_000, replace=False)
            obj_pts = obj_pts[sample]
            obj_colors = obj_colors[sample]

        cloud = trimesh.PointCloud(
            vertices=obj_pts,
            colors=obj_colors,
        )
        node_name = f"{obj.label}_{obj.object_id}"
        scene.add_geometry(cloud, node_name=node_name)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scene.export(output_path)

    print(f"[Export] glTF saved to {output_path}")
    return output_path
