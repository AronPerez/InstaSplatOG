"""Scene editor for transforming segmented furniture objects.

Provides ObjectTransform and SceneEditor for manipulating individual
Gaussian splat objects (translate, rotate, scale) while preserving
correct parameter spaces (log-space scaling, quaternion rotation).
"""

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.segmentation import SceneSegmentation, SegmentedObject, project_mask_to_gaussians


@dataclass
class ObjectTransform:
    """Transform applied to a single object."""
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_euler: np.ndarray = field(default_factory=lambda: np.zeros(3))  # degrees XYZ
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))

    def is_identity(self) -> bool:
        return (
            np.allclose(self.translation, 0)
            and np.allclose(self.rotation_euler, 0)
            and np.allclose(self.scale, 1)
        )

    def to_dict(self) -> dict:
        return {
            "translation": self.translation.tolist(),
            "rotation_euler": self.rotation_euler.tolist(),
            "scale": self.scale.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "ObjectTransform":
        return ObjectTransform(
            translation=np.array(d["translation"]),
            rotation_euler=np.array(d["rotation_euler"]),
            scale=np.array(d["scale"]),
        )


def _euler_to_rotation_matrix(euler_degrees: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles (degrees) to 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation as R
    return R.from_euler("xyz", euler_degrees, degrees=True).as_matrix()


def _rotation_matrix_to_quaternion(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    from scipy.spatial.transform import Rotation as R
    q = R.from_matrix(mat).as_quat()  # returns (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])  # convert to (w, x, y, z)


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


class SceneEditor:
    """Manages transforms on segmented objects and produces modified Gaussian data.

    Works with raw numpy arrays extracted from GaussianModel parameters.
    The gaussian_data dict contains:
        - xyz: (N, 3) positions
        - rotation: (N, 4) quaternions (w, x, y, z) — raw _rotation values
        - scaling: (N, 3) log-space scales — raw _scaling values
        - opacity: (N, 1) inverse-sigmoid opacity — raw _opacity values
        - features_dc: (N, 1, 3) SH DC coefficients
        - features_rest: (N, K, 3) SH higher-order coefficients
    """

    def __init__(
        self,
        segmentation: SceneSegmentation,
        gaussian_data: Dict[str, np.ndarray],
    ):
        self.segmentation = segmentation
        # Store original (unmodified) Gaussian data
        self._original = {k: v.copy() for k, v in gaussian_data.items()}
        self._gaussian_data = {k: v.copy() for k, v in gaussian_data.items()}
        self._transforms: Dict[int, ObjectTransform] = {}

        # Map object_id -> gaussian indices (computed lazily)
        self._object_gaussian_indices: Dict[int, np.ndarray] = {}

    def _get_gaussian_indices(self, object_id: int) -> np.ndarray:
        """Get Gaussian indices for an object, computing via KD-tree if needed."""
        if object_id not in self._object_gaussian_indices:
            obj = self.segmentation.get_object(object_id)
            if obj is None:
                return np.array([], dtype=np.int64)

            # If point_indices are indices into pts3d_flat, we need to
            # map them to Gaussian indices. If the Gaussians were created
            # directly from pts3d, the indices may coincide. Otherwise,
            # use KD-tree projection.
            # For now, assume gaussian indices = point indices (pre-training)
            # or use KD-tree lookup (post-training).
            n_gaussians = len(self._original["xyz"])

            if obj.point_indices.max() < n_gaussians:
                # Direct mapping — indices are within Gaussian range
                valid = obj.point_indices[obj.point_indices < n_gaussians]
                self._object_gaussian_indices[object_id] = valid
            else:
                # Need KD-tree lookup
                from utils.segmentation import project_mask_to_gaussians
                pts3d_flat_subset = np.zeros((len(obj.point_indices), 3))
                # We don't have pts3d here, so fall back to using centroid/bbox
                # The caller should pre-compute this via set_object_gaussian_indices
                self._object_gaussian_indices[object_id] = np.array([], dtype=np.int64)

        return self._object_gaussian_indices[object_id]

    def set_object_gaussian_indices(self, object_id: int, indices: np.ndarray) -> None:
        """Explicitly set which Gaussians belong to an object."""
        self._object_gaussian_indices[object_id] = indices.astype(np.int64)

    def get_transform(self, object_id: int) -> ObjectTransform:
        if object_id not in self._transforms:
            self._transforms[object_id] = ObjectTransform()
        return self._transforms[object_id]

    def set_object_transform(self, object_id: int, transform: ObjectTransform) -> None:
        """Apply a transform to an object's Gaussians."""
        self._transforms[object_id] = transform
        self._apply_transform(object_id)

    def _apply_transform(self, object_id: int) -> None:
        """Recompute Gaussian parameters for a single object from original data."""
        transform = self._transforms.get(object_id)
        if transform is None or transform.is_identity():
            # Reset to original
            indices = self._get_gaussian_indices(object_id)
            if len(indices) == 0:
                return
            for key in self._original:
                self._gaussian_data[key][indices] = self._original[key][indices]
            return

        indices = self._get_gaussian_indices(object_id)
        if len(indices) == 0:
            return

        obj = self.segmentation.get_object(object_id)
        if obj is None:
            return

        centroid = obj.centroid

        # Start from original values
        orig_xyz = self._original["xyz"][indices].copy()  # (K, 3)
        orig_rot = self._original["rotation"][indices].copy()  # (K, 4) wxyz
        orig_scale = self._original["scaling"][indices].copy()  # (K, 3) log-space

        # 1. Scale: scale positional offsets from centroid, add log(factor) to scaling
        offsets = orig_xyz - centroid
        scale_factors = transform.scale
        offsets = offsets * scale_factors
        log_scale_delta = np.log(np.clip(scale_factors, 1e-6, None))
        new_scale = orig_scale + log_scale_delta  # log-space addition

        # 2. Rotation: rotate offsets about centroid, compose quaternion
        if not np.allclose(transform.rotation_euler, 0):
            rot_mat = _euler_to_rotation_matrix(transform.rotation_euler)
            offsets = (rot_mat @ offsets.T).T

            rot_quat = _rotation_matrix_to_quaternion(rot_mat)
            new_rot = np.array([
                _quaternion_multiply(rot_quat, orig_rot[i])
                for i in range(len(orig_rot))
            ])
            # Normalize quaternions
            norms = np.linalg.norm(new_rot, axis=1, keepdims=True)
            new_rot = new_rot / np.clip(norms, 1e-8, None)
        else:
            new_rot = orig_rot

        # 3. Translation: add delta to positions
        new_xyz = centroid + offsets + transform.translation

        # Write back
        self._gaussian_data["xyz"][indices] = new_xyz
        self._gaussian_data["rotation"][indices] = new_rot
        self._gaussian_data["scaling"][indices] = new_scale

        # Opacity and SH features unchanged by rigid transforms + uniform scale

    def delete_object(self, object_id: int) -> None:
        """Remove an object's Gaussians from the scene."""
        indices = self._get_gaussian_indices(object_id)
        if len(indices) == 0:
            return

        # Create mask of points to keep
        n = len(self._gaussian_data["xyz"])
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[indices] = False

        # Filter all arrays
        for key in self._gaussian_data:
            self._gaussian_data[key] = self._gaussian_data[key][keep_mask]
            self._original[key] = self._original[key][keep_mask]

        # Remove from segmentation
        self.segmentation.remove_object(object_id)
        if object_id in self._transforms:
            del self._transforms[object_id]
        if object_id in self._object_gaussian_indices:
            del self._object_gaussian_indices[object_id]

        # Reindex remaining objects' gaussian indices
        # Build index remapping
        old_to_new = np.full(n, -1, dtype=np.int64)
        new_indices = np.arange(keep_mask.sum())
        old_indices = np.where(keep_mask)[0]
        old_to_new[old_indices] = new_indices

        for oid in list(self._object_gaussian_indices.keys()):
            old_gi = self._object_gaussian_indices[oid]
            new_gi = old_to_new[old_gi]
            self._object_gaussian_indices[oid] = new_gi[new_gi >= 0]

    def duplicate_object(self, object_id: int) -> int:
        """Duplicate an object's Gaussians and return the new object_id."""
        indices = self._get_gaussian_indices(object_id)
        if len(indices) == 0:
            return -1

        obj = self.segmentation.get_object(object_id)
        if obj is None:
            return -1

        # Append duplicated Gaussians
        n_existing = len(self._gaussian_data["xyz"])
        for key in self._gaussian_data:
            duped = self._gaussian_data[key][indices].copy()
            self._gaussian_data[key] = np.concatenate(
                [self._gaussian_data[key], duped], axis=0
            )
            duped_orig = self._original[key][indices].copy()
            self._original[key] = np.concatenate(
                [self._original[key], duped_orig], axis=0
            )

        new_indices = np.arange(n_existing, n_existing + len(indices))
        new_id = self.segmentation.allocate_id()

        new_obj = SegmentedObject(
            object_id=new_id,
            label=obj.label,
            point_indices=obj.point_indices.copy(),
            centroid=obj.centroid.copy(),
            bbox_3d=obj.bbox_3d.copy(),
            per_view_masks={k: v.copy() for k, v in obj.per_view_masks.items()},
            confidence=obj.confidence,
        )
        self.segmentation.add_object(new_obj)
        self._object_gaussian_indices[new_id] = new_indices

        return new_id

    def get_full_scene_gaussians(self) -> Dict[str, np.ndarray]:
        """Return current (transformed) Gaussian data as a dict."""
        return {k: v.copy() for k, v in self._gaussian_data.items()}

    def get_object_points(self, object_id: int) -> Optional[np.ndarray]:
        """Return current xyz positions for an object's Gaussians."""
        indices = self._get_gaussian_indices(object_id)
        if len(indices) == 0:
            return None
        return self._gaussian_data["xyz"][indices]

    def get_object_colors(self, object_id: int) -> Optional[np.ndarray]:
        """Return RGB colors for an object's Gaussians (from SH DC)."""
        indices = self._get_gaussian_indices(object_id)
        if len(indices) == 0:
            return None
        features_dc = self._gaussian_data["features_dc"][indices]  # (K, 1, 3)
        # SH DC -> RGB: c = sh * C0 + 0.5 where C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        rgb = features_dc[:, 0, :] * C0 + 0.5
        return np.clip(rgb, 0, 1)
