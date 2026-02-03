"""Furniture segmentation engine using Grounded-SAM2.

Provides automatic scene segmentation by running GroundingDINO for detection
and SAM2 for mask generation, then projecting 2D masks to 3D via pts3d.

Optional dependencies (guarded with try/except):
  - segment-anything-2
  - groundingdino-py
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import groundingdino as _gdino_pkg
    from groundingdino.util.inference import load_model as load_gdino_model
    from groundingdino.util.inference import predict as gdino_predict
    from groundingdino.util.inference import annotate as gdino_annotate
    _GDINO_CONFIG_DEFAULT = os.path.join(
        os.path.dirname(_gdino_pkg.__file__),
        "config", "GroundingDINO_SwinT_OGC.py"
    )
    GDINO_AVAILABLE = True
except ImportError:
    _GDINO_CONFIG_DEFAULT = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GDINO_AVAILABLE = False

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

from scipy.cluster.hierarchy import fcluster, linkage


def _ensure_gdino_weights(weights_path: str) -> str:
    """Return a valid local path to GroundingDINO weights, downloading if needed."""
    if os.path.isfile(weights_path):
        return weights_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="ShilongLiu/GroundingDINO",
        filename=os.path.basename(weights_path),
    )


_SAM2_HF_REPOS = {
    "sam2_hiera_large.pt": "facebook/sam2-hiera-large",
    "sam2_hiera_base_plus.pt": "facebook/sam2-hiera-base-plus",
    "sam2_hiera_small.pt": "facebook/sam2-hiera-small",
    "sam2_hiera_tiny.pt": "facebook/sam2-hiera-tiny",
}


def _ensure_sam2_weights(weights_path: str) -> str:
    """Return a valid local path to SAM2 weights, downloading if needed."""
    if os.path.isfile(weights_path):
        return weights_path
    basename = os.path.basename(weights_path)
    repo_id = _SAM2_HF_REPOS.get(basename)
    if repo_id is None:
        raise FileNotFoundError(
            f"SAM2 weights not found at '{weights_path}' and no known "
            f"HuggingFace repo for '{basename}'. Known files: "
            f"{list(_SAM2_HF_REPOS.keys())}"
        )
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=basename)


def _check_segmentation_deps():
    """Check that required segmentation dependencies are installed."""
    missing = []
    if not GDINO_AVAILABLE:
        missing.append("groundingdino-py")
    if not SAM2_AVAILABLE:
        missing.append("sam2")
    if missing:
        raise ImportError(
            f"Segmentation requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


@dataclass
class SegmentedObject:
    """A single segmented furniture object in 3D space."""
    object_id: int
    label: str
    point_indices: np.ndarray  # indices into flattened pts3d (M*H*W,)
    centroid: np.ndarray       # (3,) world-space center
    bbox_3d: np.ndarray        # (2, 3) AABB [min_corner, max_corner]
    per_view_masks: Dict[int, np.ndarray]  # {view_idx: (H,W) bool}
    confidence: float

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "point_indices": self.point_indices.tolist(),
            "centroid": self.centroid.tolist(),
            "bbox_3d": self.bbox_3d.tolist(),
            "confidence": self.confidence,
        }

    @staticmethod
    def from_dict(d: dict) -> "SegmentedObject":
        return SegmentedObject(
            object_id=d["object_id"],
            label=d["label"],
            point_indices=np.array(d["point_indices"], dtype=np.int64),
            centroid=np.array(d["centroid"], dtype=np.float64),
            bbox_3d=np.array(d["bbox_3d"], dtype=np.float64),
            per_view_masks={},
            confidence=d["confidence"],
        )


class SceneSegmentation:
    """Container for all segmented objects in a scene."""

    def __init__(self):
        self.objects: List[SegmentedObject] = []
        self.background_indices: Optional[np.ndarray] = None
        self._next_id = 0

    def add_object(self, obj: SegmentedObject) -> None:
        self.objects.append(obj)
        self._next_id = max(self._next_id, obj.object_id + 1)

    def remove_object(self, object_id: int) -> Optional[SegmentedObject]:
        for i, obj in enumerate(self.objects):
            if obj.object_id == object_id:
                return self.objects.pop(i)
        return None

    def get_object(self, object_id: int) -> Optional[SegmentedObject]:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_labels(self) -> List[str]:
        """Return list of 'label_id' strings for UI dropdown."""
        return [f"{obj.label}_{obj.object_id}" for obj in self.objects]

    def allocate_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    def compute_background(self, total_points: int) -> None:
        """Compute background indices as complement of all object indices."""
        if self.objects:
            all_obj = np.concatenate([obj.point_indices for obj in self.objects])
            mask = np.ones(total_points, dtype=bool)
            mask[all_obj] = False
            self.background_indices = np.nonzero(mask)[0]
        else:
            self.background_indices = np.arange(total_points, dtype=np.int64)

    def to_json(self) -> str:
        return json.dumps({
            "objects": [obj.to_dict() for obj in self.objects],
        }, indent=2)

    @staticmethod
    def from_json(s: str) -> "SceneSegmentation":
        data = json.loads(s)
        seg = SceneSegmentation()
        for d in data["objects"]:
            seg.add_object(SegmentedObject.from_dict(d))
        return seg


def _detect_objects_in_image(
    image_np: np.ndarray,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    gdino_model,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Run GroundingDINO on a single image.

    Args:
        image_np: (H, W, 3) uint8 RGB image
        text_prompt: dot-separated category string e.g. "sofa . table . chair"
        box_threshold: detection confidence threshold
        text_threshold: text matching threshold
        gdino_model: loaded GroundingDINO model

    Returns:
        boxes: (N, 4) normalized xyxy boxes
        phrases: list of detected labels
        logits: (N,) confidence scores
    """
    from PIL import Image as PILImage
    import groundingdino.datasets.transforms as T

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil_img = PILImage.fromarray(image_np)
    transformed, _ = transform(pil_img, None)

    boxes, logits, phrases = gdino_predict(
        model=gdino_model,
        image=transformed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    return boxes.numpy(), phrases, logits.numpy()


def _segment_with_sam2(
    image_np: np.ndarray,
    boxes: np.ndarray,
    sam_predictor: "SAM2ImagePredictor",
) -> List[np.ndarray]:
    """Run SAM2 with box prompts to get instance masks.

    Args:
        image_np: (H, W, 3) uint8 RGB image
        boxes: (N, 4) xyxy boxes in pixel coordinates
        sam_predictor: initialized SAM2ImagePredictor

    Returns:
        List of (H, W) bool masks, one per box
    """
    sam_predictor.set_image(image_np)
    h, w = image_np.shape[:2]

    masks_out = []
    for box in boxes:
        input_box = box[None, :]  # (1, 4)
        masks, scores, _ = sam_predictor.predict(
            box=input_box,
            multimask_output=False,
        )
        masks_out.append(masks[0].astype(bool))  # (H, W)

    return masks_out


def _cluster_3d_points(
    points_3d: np.ndarray,
    eps: float = 0.1,
    min_samples: int = 10,
) -> List[np.ndarray]:
    """Cluster 3D points using hierarchical clustering to separate instances.

    Returns list of index arrays, one per cluster.
    """
    if len(points_3d) < min_samples:
        return [np.arange(len(points_3d))]

    Z = linkage(points_3d, method='single', metric='euclidean')
    labels = fcluster(Z, t=eps, criterion='distance')

    clusters = []
    for label_id in set(labels):
        cluster_mask = labels == label_id
        indices = np.where(cluster_mask)[0]
        if len(indices) >= min_samples:
            clusters.append(indices)

    return clusters


def segment_scene_auto(
    images: np.ndarray,
    pts3d: np.ndarray,
    text_prompt: str = "sofa . table . chair . lamp . bed . shelf",
    device: str = "cuda",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    min_views: int = 1,
    gdino_config: str = _GDINO_CONFIG_DEFAULT,
    gdino_weights: str = "groundingdino_swint_ogc.pth",
    sam2_config: str = "sam2_hiera_l.yaml",
    sam2_weights: str = "sam2_hiera_large.pt",
) -> SceneSegmentation:
    """Run automatic furniture segmentation on a reconstructed scene.

    Args:
        images: (M, H, W, 3) float [0,1] or uint8 images
        pts3d: (M, H, W, 3) 3D point maps
        text_prompt: dot-separated object categories
        device: torch device
        box_threshold: GroundingDINO detection confidence
        text_threshold: GroundingDINO text matching threshold
        min_views: minimum number of views an object must appear in
        gdino_config: path to GroundingDINO config
        gdino_weights: path to GroundingDINO weights
        sam2_config: SAM2 model config name
        sam2_weights: path to SAM2 weights

    Returns:
        SceneSegmentation with detected objects
    """
    _check_segmentation_deps()

    M, H, W, _ = pts3d.shape
    pts3d_flat = pts3d.reshape(-1, 3)  # (M*H*W, 3)

    # Convert images to uint8 if needed
    if images.dtype != np.uint8:
        images_u8 = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    else:
        images_u8 = images

    # Load models
    print("[Segmentation] Loading GroundingDINO...")
    gdino_weights = _ensure_gdino_weights(gdino_weights)
    gdino_model = load_gdino_model(gdino_config, gdino_weights)

    print("[Segmentation] Loading SAM2...")
    sam2_weights = _ensure_sam2_weights(sam2_weights)
    sam2_model = build_sam2(sam2_config, sam2_weights, device=device)
    sam_predictor = SAM2ImagePredictor(sam2_model)

    # Per-label accumulator: label -> list of (view_idx, mask_2d, confidence)
    label_detections: Dict[str, List[Tuple[int, np.ndarray, float]]] = {}

    for view_idx in range(M):
        img = images_u8[view_idx]
        boxes, phrases, logits = _detect_objects_in_image(
            img, text_prompt, box_threshold, text_threshold, gdino_model
        )

        if len(boxes) == 0:
            continue

        # Convert normalized boxes to pixel coordinates
        h, w = img.shape[:2]
        boxes_pixel = boxes.copy()
        boxes_pixel[:, [0, 2]] *= w
        boxes_pixel[:, [1, 3]] *= h

        masks = _segment_with_sam2(img, boxes_pixel, sam_predictor)

        for phrase, mask, conf in zip(phrases, masks, logits):
            label = phrase.strip().lower()
            if label not in label_detections:
                label_detections[label] = []
            label_detections[label].append((view_idx, mask, float(conf)))

    # Free GPU memory from segmentation models
    del gdino_model, sam2_model, sam_predictor
    if TORCH_AVAILABLE:
        torch.cuda.empty_cache()

    # Build objects from detections
    segmentation = SceneSegmentation()

    for label, detections in label_detections.items():
        # Collect all 3D point indices across views
        all_point_indices = set()
        per_view_masks = {}
        total_conf = 0.0

        for view_idx, mask_2d, conf in detections:
            per_view_masks[view_idx] = mask_2d
            # Project 2D mask to 3D: pts3d[view][mask] gives 3D points
            view_offset = view_idx * H * W
            flat_mask = mask_2d.reshape(-1)
            indices = np.where(flat_mask)[0] + view_offset
            all_point_indices.update(indices.tolist())
            total_conf += conf

        point_indices = np.array(sorted(all_point_indices), dtype=np.int64)

        if len(point_indices) == 0:
            continue

        # Multi-view consensus: count how many views each point is seen in
        if min_views > 1 and len(detections) >= min_views:
            point_view_count = {}
            for view_idx, mask_2d, _ in detections:
                view_offset = view_idx * H * W
                flat_mask = mask_2d.reshape(-1)
                for idx in np.where(flat_mask)[0]:
                    global_idx = idx + view_offset
                    # Map to 3D position for cross-view matching
                    point_view_count[global_idx] = point_view_count.get(global_idx, 0) + 1
            # Note: cross-view consensus is approximate since same 3D point
            # has different indices in different views. For better accuracy,
            # we'd do nearest-neighbor matching in 3D space.
            # For now, keep all points (each view contributes unique indices).

        # Cluster to separate instances of same label
        points_3d = pts3d_flat[point_indices]

        # Adaptive DBSCAN eps based on scene scale
        scene_extent = np.ptp(pts3d_flat, axis=0).max()
        eps = scene_extent * 0.05

        clusters = _cluster_3d_points(points_3d, eps=eps, min_samples=10)

        for cluster_local_indices in clusters:
            global_indices = point_indices[cluster_local_indices]
            cluster_points = pts3d_flat[global_indices]

            centroid = cluster_points.mean(axis=0)
            bbox_min = cluster_points.min(axis=0)
            bbox_max = cluster_points.max(axis=0)

            obj = SegmentedObject(
                object_id=segmentation.allocate_id(),
                label=label,
                point_indices=global_indices,
                centroid=centroid,
                bbox_3d=np.stack([bbox_min, bbox_max]),
                per_view_masks=per_view_masks,
                confidence=total_conf / len(detections),
            )
            segmentation.add_object(obj)

    segmentation.compute_background(len(pts3d_flat))
    print(f"[Segmentation] Found {len(segmentation.objects)} objects: "
          f"{[f'{o.label}_{o.object_id}' for o in segmentation.objects]}")
    return segmentation


def project_mask_to_gaussians(
    mask_3d_points: np.ndarray,
    gaussian_xyz: np.ndarray,
    radius: float = 0.05,
) -> np.ndarray:
    """Map segmented 3D points to Gaussian indices via KD-tree lookup.

    Args:
        mask_3d_points: (K, 3) segmented 3D points from pts3d
        gaussian_xyz: (N, 3) Gaussian center positions
        radius: search radius for nearest-neighbor matching

    Returns:
        Array of Gaussian indices that belong to this object
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(gaussian_xyz)
    indices = set()

    # Query all mask points at once for efficiency
    dists, idxs = tree.query(mask_3d_points, k=1)
    valid = dists < radius
    indices = np.unique(idxs[valid])

    return indices.astype(np.int64)


def refine_segmentation(
    existing_obj: SegmentedObject,
    add_mask: Optional[np.ndarray],
    remove_mask: Optional[np.ndarray],
    view_idx: int,
    pts3d: np.ndarray,
) -> SegmentedObject:
    """Manually refine an object's segmentation by adding/removing mask regions.

    Args:
        existing_obj: object to refine
        add_mask: (H, W) bool mask of pixels to add
        remove_mask: (H, W) bool mask of pixels to remove
        view_idx: which view the masks are from
        pts3d: (M, H, W, 3) full point maps

    Returns:
        Updated SegmentedObject (modified in-place and returned)
    """
    M, H, W, _ = pts3d.shape
    pts3d_flat = pts3d.reshape(-1, 3)
    view_offset = view_idx * H * W
    current_indices = set(existing_obj.point_indices.tolist())

    if add_mask is not None:
        add_flat = add_mask.reshape(-1)
        new_indices = np.where(add_flat)[0] + view_offset
        current_indices.update(new_indices.tolist())

    if remove_mask is not None:
        rm_flat = remove_mask.reshape(-1)
        rm_indices = set((np.where(rm_flat)[0] + view_offset).tolist())
        current_indices -= rm_indices

    point_indices = np.array(sorted(current_indices), dtype=np.int64)
    points_3d = pts3d_flat[point_indices]

    existing_obj.point_indices = point_indices
    existing_obj.centroid = points_3d.mean(axis=0)
    existing_obj.bbox_3d = np.stack([points_3d.min(axis=0), points_3d.max(axis=0)])

    # Update per-view mask
    new_view_mask = np.zeros((H, W), dtype=bool)
    view_indices = point_indices[(point_indices >= view_offset) & (point_indices < view_offset + H * W)]
    local_indices = view_indices - view_offset
    rows = local_indices // W
    cols = local_indices % W
    new_view_mask[rows, cols] = True
    existing_obj.per_view_masks[view_idx] = new_view_mask

    return existing_obj
