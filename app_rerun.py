"""Gradio app with embedded Rerun 3D viewer for InstantSplat.

Includes furniture segmentation, object manipulation, and multi-format export.
"""
import os
import tempfile

import numpy as np
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose

import gradio as gr

try:
    from gradio_rerun import Rerun
    GRADIO_RERUN_AVAILABLE = True
except ImportError:
    GRADIO_RERUN_AVAILABLE = False

from utils.rerun_vis import (
    is_rerun_available, log_reconstruction,
    log_segmented_scene, update_object_transform,
)
from utils.reconstruction_result import ReconstructionResult


def preview_uploads(input_files):
    """Process uploaded images: resize, save to temp dir, return gallery preview."""
    if not input_files:
        return None, ""
    img_list = []
    temp_dir = tempfile.mkdtemp()
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir()
    for i, f in enumerate(input_files):
        img = exif_transpose(Image.open(f)).convert("RGB")
        max_dim = max(img.size)
        if max_dim > 720:
            scale = 720 / max_dim
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        img.save(images_dir / f"img_{i}.jpg")
        img_list.append(np.array(img))
    return img_list, temp_dir


def run_init_geo_with_rerun(source_path, uploaded_dir, n_views, conf_threshold):
    """Run geometry initialization and stream results to Rerun viewer.

    Returns: (viewer_bytes, pts3d, images, rec, stream)
    The pts3d and images are preserved for later segmentation.
    """
    import rerun as rr

    scene_dir = uploaded_dir if uploaded_dir else source_path

    rec = rr.new_recording(
        application_id="InstantSplat",
        recording_id=f"init_{Path(scene_dir).name}",
    )
    stream = rec.binary_stream()

    rec.log("world", rr.ViewCoordinates.RDF, static=True)
    yield stream.read(), None, None, None, None

    # --- Run the actual geometry pipeline ---
    import torch
    from mast3r.model import AsymmetricMASt3R
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from utils.sfm_utils import get_sorted_image_files, split_train_test, load_images, stack_images, stack_maps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    model = AsymmetricMASt3R.from_pretrained(ckpt).to(device)
    image_dir = Path(scene_dir) / "images"
    image_files, _ = get_sorted_image_files(image_dir)
    image_files = image_files[:int(n_views)]
    images, _ = load_images(image_files, size=512)

    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

    imgs = stack_images(scene.imgs)
    pts3d = np.array(to_numpy(scene.get_pts3d()))
    confs = stack_maps([p.detach().cpu().numpy() for p in scene.im_conf])

    result = ReconstructionResult.from_scene(
        scene, imgs, pts3d, confs=confs,
        image_files=image_files,
        conf_threshold=float(conf_threshold),
    )

    # Free MASt3R from GPU
    del model, output, scene
    torch.cuda.empty_cache()

    # Log point cloud
    if result.colors is not None and result.colors.dtype != np.uint8:
        colors_uint8 = (np.clip(result.colors, 0, 1) * 255).astype(np.uint8)
    else:
        colors_uint8 = result.colors
    rec.log("world/points", rr.Points3D(
        positions=result.points, colors=colors_uint8, radii=0.005
    ), static=True)
    yield stream.read(), None, None, None, None

    # Log cameras
    for i in range(len(result.camera_poses_c2w)):
        name = result.image_names[i] if result.image_names else f"cam_{i:03d}"
        pose = result.camera_poses_c2w[i]
        K = result.intrinsics[i]
        rec.log(f"world/cameras/{name}",
                rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
                static=True)
        img = result.images[i] if result.images is not None else None
        if img is not None:
            h, w = img.shape[:2]
        else:
            w, h = int(K[0, 2] * 2), int(K[1, 2] * 2)
        rec.log(f"world/cameras/{name}",
                rr.Pinhole(focal_length=[K[0, 0], K[1, 1]],
                           principal_point=[K[0, 2], K[1, 2]],
                           width=w, height=h),
                static=True)
        if img is not None:
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            rec.log(f"world/cameras/{name}/image", rr.Image(img_u8), static=True)

    # Yield final data: viewer bytes + preserved state for segmentation
    yield stream.read(), pts3d, imgs, rec, stream


def run_segmentation(
    pts3d_state, imgs_state, rec_state, stream_state,
    text_prompt, box_threshold,
):
    """Run auto-segmentation and update Rerun viewer with segmented entities.

    Yields: (viewer_bytes, segmentation, editor, object_dropdown_choices)
    Generator: first yield updates the viewer immediately, second adds the editor.
    """
    import rerun as rr
    from utils.segmentation import segment_scene_auto, SceneSegmentation
    from utils.scene_editor import SceneEditor

    if pts3d_state is None:
        gr.Warning("Run geometry initialization first.")
        yield None, None, None, gr.update(choices=[], value=None)
        return

    pts3d = pts3d_state
    imgs = imgs_state
    rec = rec_state
    stream = stream_state

    # Run segmentation
    print("[Segmentation] Running segment_scene_auto...")
    try:
        segmentation = segment_scene_auto(
            images=imgs,
            pts3d=pts3d,
            text_prompt=text_prompt,
            box_threshold=float(box_threshold),
        )
    except ImportError as e:
        gr.Warning(str(e))
        yield stream.read() if stream else None, None, None, gr.update(choices=[], value=None)
        return

    print(f"[Segmentation] Found {len(segmentation.objects)} objects.")

    if len(segmentation.objects) == 0:
        gr.Warning("No objects detected. Try lowering the confidence threshold or changing the text prompt.")
        yield stream.read() if stream else None, None, None, gr.update(choices=[], value=None)
        return

    # Build colors for visualization
    M, H, W, _ = pts3d.shape
    pts3d_flat = pts3d.reshape(-1, 3)
    if imgs.dtype != np.uint8:
        colors_flat = (np.clip(imgs, 0, 1) * 255).astype(np.uint8).reshape(-1, 3)
    else:
        colors_flat = imgs.reshape(-1, 3)

    # Clear existing point cloud and log segmented scene
    print("[Segmentation] Logging segmented scene to Rerun...")
    rec.log("world/points", rr.Clear(recursive=True))
    log_segmented_scene(rec, segmentation, pts3d_flat, colors_flat)
    print("[Segmentation] Rerun logging complete.")

    # Yield viewer update before building editor (so UI refreshes immediately)
    print("[Segmentation] Reading Rerun stream...")
    viewer_bytes = stream.read()
    print(f"[Segmentation] Stream read complete ({len(viewer_bytes)} bytes).")

    labels = segmentation.get_labels()

    # First yield: update viewer immediately, editor not ready yet
    yield (
        viewer_bytes,
        segmentation,
        None,
        gr.update(choices=labels, value=labels[0] if labels else None),
    )

    # Create editor with point-cloud-based Gaussian data
    # (pre-training: pts3d points serve as proto-Gaussians)
    print("[Segmentation] Building SceneEditor...")
    gaussian_data = {
        "xyz": pts3d_flat,
        "rotation": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(pts3d_flat), 1)),
        "scaling": np.zeros((len(pts3d_flat), 3), dtype=np.float32),
        "opacity": np.zeros((len(pts3d_flat), 1), dtype=np.float32),
        "features_dc": np.zeros((len(pts3d_flat), 1, 3), dtype=np.float32),
        "features_rest": np.zeros((len(pts3d_flat), 0, 3), dtype=np.float32),
    }
    editor = SceneEditor(segmentation, gaussian_data)

    # Set up gaussian indices for each object (direct mapping pre-training)
    for obj in segmentation.objects:
        editor.set_object_gaussian_indices(obj.object_id, obj.point_indices)
    print("[Segmentation] SceneEditor ready.")

    # Second yield: add editor
    yield (
        viewer_bytes,
        segmentation,
        editor,
        gr.update(choices=labels, value=labels[0] if labels else None),
    )


def on_object_selected(selection, editor_state):
    """When user selects an object from dropdown, return its current transform values."""
    if not selection or editor_state is None:
        return 0, 0, 0, 0, 0, 0, 1.0

    # Parse "label_id" format
    parts = selection.rsplit("_", 1)
    if len(parts) != 2:
        return 0, 0, 0, 0, 0, 0, 1.0

    try:
        object_id = int(parts[1])
    except ValueError:
        return 0, 0, 0, 0, 0, 0, 1.0

    transform = editor_state.get_transform(object_id)
    return (
        float(transform.translation[0]),
        float(transform.translation[1]),
        float(transform.translation[2]),
        float(transform.rotation_euler[0]),
        float(transform.rotation_euler[1]),
        float(transform.rotation_euler[2]),
        float(transform.scale[0]),  # uniform scale for UI
    )


def on_transform_changed(
    selection, editor_state, rec_state, stream_state,
    tx, ty, tz, rx, ry, rz, scale_val,
):
    """Apply slider values as object transform and update Rerun."""
    from utils.scene_editor import ObjectTransform

    if not selection or editor_state is None or rec_state is None:
        return None

    parts = selection.rsplit("_", 1)
    if len(parts) != 2:
        return None

    try:
        object_id = int(parts[1])
    except ValueError:
        return None

    label = parts[0]

    transform = ObjectTransform(
        translation=np.array([tx, ty, tz], dtype=np.float64),
        rotation_euler=np.array([rx, ry, rz], dtype=np.float64),
        scale=np.array([scale_val, scale_val, scale_val], dtype=np.float64),
    )

    editor_state.set_object_transform(object_id, transform)
    update_object_transform(rec_state, object_id, label, transform)

    return stream_state.read() if stream_state else None


def on_reset_transform(selection, editor_state, rec_state, stream_state):
    """Reset the selected object's transform to identity."""
    from utils.scene_editor import ObjectTransform

    if not selection or editor_state is None:
        return None, 0, 0, 0, 0, 0, 0, 1.0

    parts = selection.rsplit("_", 1)
    if len(parts) != 2:
        return None, 0, 0, 0, 0, 0, 0, 1.0

    try:
        object_id = int(parts[1])
    except ValueError:
        return None, 0, 0, 0, 0, 0, 0, 1.0

    label = parts[0]
    identity = ObjectTransform()
    editor_state.set_object_transform(object_id, identity)
    update_object_transform(rec_state, object_id, label, identity)

    viewer_bytes = stream_state.read() if stream_state else None
    return viewer_bytes, 0, 0, 0, 0, 0, 0, 1.0


def on_delete_object(selection, segmentation_state, editor_state, rec_state, stream_state):
    """Delete the selected object from the scene."""
    import rerun as rr

    if not selection or editor_state is None:
        return None, gr.update(), segmentation_state, editor_state

    parts = selection.rsplit("_", 1)
    if len(parts) != 2:
        return None, gr.update(), segmentation_state, editor_state

    try:
        object_id = int(parts[1])
    except ValueError:
        return None, gr.update(), segmentation_state, editor_state

    label = parts[0]

    # Clear the entity in Rerun
    if rec_state is not None:
        entity = f"world/objects/{label}_{object_id}"
        rec_state.log(f"{entity}/points", rr.Clear(recursive=True))
        rec_state.log(f"{entity}/bbox", rr.Clear(recursive=True))

    editor_state.delete_object(object_id)

    labels = segmentation_state.get_labels()
    viewer_bytes = stream_state.read() if stream_state else None
    return (
        viewer_bytes,
        gr.update(choices=labels, value=labels[0] if labels else None),
        segmentation_state,
        editor_state,
    )


def on_duplicate_object(selection, segmentation_state, editor_state, rec_state, stream_state):
    """Duplicate the selected object."""
    if not selection or editor_state is None:
        return None, gr.update(), segmentation_state, editor_state

    parts = selection.rsplit("_", 1)
    if len(parts) != 2:
        return None, gr.update(), segmentation_state, editor_state

    try:
        object_id = int(parts[1])
    except ValueError:
        return None, gr.update(), segmentation_state, editor_state

    new_id = editor_state.duplicate_object(object_id)
    if new_id < 0:
        return None, gr.update(), segmentation_state, editor_state

    labels = segmentation_state.get_labels()
    new_label = f"{parts[0]}_{new_id}"
    viewer_bytes = stream_state.read() if stream_state else None
    return (
        viewer_bytes,
        gr.update(choices=labels, value=new_label),
        segmentation_state,
        editor_state,
    )


def on_export_scene(export_format, editor_state, segmentation_state):
    """Export the scene in the selected format."""
    if editor_state is None:
        gr.Warning("No scene to export. Run segmentation first.")
        return None

    export_dir = tempfile.mkdtemp()

    if export_format == "Gaussian PLY":
        from utils.scene_export import export_gaussian_ply
        path = os.path.join(export_dir, "scene_gaussians.ply")
        export_gaussian_ply(editor_state, path)
        return path

    elif export_format == "PLY + JSON":
        from utils.scene_export import export_ply_with_json
        out_dir = os.path.join(export_dir, "ply_json")
        export_ply_with_json(editor_state, segmentation_state, out_dir)
        # Zip the directory for download
        import shutil
        zip_path = os.path.join(export_dir, "scene_ply_json")
        shutil.make_archive(zip_path, "zip", out_dir)
        return zip_path + ".zip"

    elif export_format == "glTF":
        from utils.scene_export import export_gltf
        path = os.path.join(export_dir, "scene.glb")
        export_gltf(editor_state, segmentation_state, path)
        return path

    return None


def build_app():
    with gr.Blocks(title="InstantSplat + Rerun") as demo:
        gr.Markdown("# InstantSplat 3D Viewer")
        gr.Markdown("Run geometry initialization, segment furniture, and manipulate objects in 3D.")

        # --- State variables ---
        segmentation_state = gr.State(None)
        editor_state = gr.State(None)
        raw_pts3d = gr.State(None)
        raw_imgs = gr.State(None)
        rec_state = gr.State(None)
        stream_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                # --- Reconstruction controls ---
                with gr.Tabs():
                    with gr.TabItem("Upload Images"):
                        input_files = gr.File(
                            file_count="multiple",
                            label="Drop images here",
                        )
                        gallery = gr.Gallery(label="Preview")
                    with gr.TabItem("Server Path"):
                        source_path = gr.Textbox(
                            label="Scene Path",
                            value="assets/sora/Art",
                            info="Path to scene directory (must contain images/ subfolder)",
                        )
                processed_folder = gr.State(value="")
                n_views = gr.Slider(
                    minimum=2, maximum=24, value=3, step=1,
                    label="Number of Views",
                )
                conf_threshold = gr.Slider(
                    minimum=0.0, maximum=10.0, value=1.0, step=0.1,
                    label="Confidence Threshold",
                    info="Filter points below this confidence",
                )
                run_btn = gr.Button("Run Geometry Init", variant="primary")

                # --- Segmentation controls ---
                with gr.Accordion("Segmentation", open=False):
                    seg_prompt = gr.Textbox(
                        label="Object Categories",
                        value="sofa . table . chair . lamp . bed . shelf",
                        info="Dot-separated list of object types to detect",
                    )
                    seg_confidence = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                        label="Detection Confidence",
                    )
                    seg_btn = gr.Button("Run Auto-Segmentation", variant="secondary")
                    object_dropdown = gr.Dropdown(
                        label="Detected Objects",
                        choices=[],
                        interactive=True,
                    )
                    with gr.Row():
                        delete_btn = gr.Button("Delete Object", variant="stop", size="sm")
                        duplicate_btn = gr.Button("Duplicate Object", size="sm")

                # --- Transform controls ---
                with gr.Accordion("Transform", open=False):
                    tx = gr.Slider(-5, 5, 0, step=0.01, label="Translate X")
                    ty = gr.Slider(-5, 5, 0, step=0.01, label="Translate Y")
                    tz = gr.Slider(-5, 5, 0, step=0.01, label="Translate Z")
                    rx = gr.Slider(-180, 180, 0, step=1, label="Rotate X (deg)")
                    ry = gr.Slider(-180, 180, 0, step=1, label="Rotate Y (deg)")
                    rz = gr.Slider(-180, 180, 0, step=1, label="Rotate Z (deg)")
                    scale_slider = gr.Slider(0.1, 3.0, 1.0, step=0.01, label="Scale")
                    reset_btn = gr.Button("Reset Transform", size="sm")

                # --- Export controls ---
                with gr.Accordion("Export", open=False):
                    export_format = gr.Radio(
                        choices=["Gaussian PLY", "PLY + JSON", "glTF"],
                        value="Gaussian PLY",
                        label="Export Format",
                    )
                    export_btn = gr.Button("Export Scene", variant="secondary")
                    export_file = gr.File(label="Download", interactive=False)

            with gr.Column(scale=3):
                viewer = Rerun(
                    streaming=True,
                    height=600,
                    panel_states={
                        "time": "collapsed",
                        "blueprint": "hidden",
                        "selection": "collapsed",
                    },
                )

        # --- Event wiring ---

        input_files.change(
            fn=preview_uploads,
            inputs=[input_files],
            outputs=[gallery, processed_folder],
        )

        run_btn.click(
            run_init_geo_with_rerun,
            inputs=[source_path, processed_folder, n_views, conf_threshold],
            outputs=[viewer, raw_pts3d, raw_imgs, rec_state, stream_state],
        )

        seg_btn.click(
            run_segmentation,
            inputs=[raw_pts3d, raw_imgs, rec_state, stream_state, seg_prompt, seg_confidence],
            outputs=[viewer, segmentation_state, editor_state, object_dropdown],
        )

        object_dropdown.change(
            on_object_selected,
            inputs=[object_dropdown, editor_state],
            outputs=[tx, ty, tz, rx, ry, rz, scale_slider],
        )

        # Wire all transform sliders to update on change
        transform_sliders = [tx, ty, tz, rx, ry, rz, scale_slider]
        transform_inputs = [object_dropdown, editor_state, rec_state, stream_state] + transform_sliders

        for slider in transform_sliders:
            slider.release(
                on_transform_changed,
                inputs=transform_inputs,
                outputs=[viewer],
            )

        reset_btn.click(
            on_reset_transform,
            inputs=[object_dropdown, editor_state, rec_state, stream_state],
            outputs=[viewer, tx, ty, tz, rx, ry, rz, scale_slider],
        )

        delete_btn.click(
            on_delete_object,
            inputs=[object_dropdown, segmentation_state, editor_state, rec_state, stream_state],
            outputs=[viewer, object_dropdown, segmentation_state, editor_state],
        )

        duplicate_btn.click(
            on_duplicate_object,
            inputs=[object_dropdown, segmentation_state, editor_state, rec_state, stream_state],
            outputs=[viewer, object_dropdown, segmentation_state, editor_state],
        )

        export_btn.click(
            on_export_scene,
            inputs=[export_format, editor_state, segmentation_state],
            outputs=[export_file],
        )

    return demo


if __name__ == "__main__":
    if not GRADIO_RERUN_AVAILABLE:
        print("Install gradio_rerun: pip install gradio_rerun")
        exit(1)
    demo = build_app()
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
