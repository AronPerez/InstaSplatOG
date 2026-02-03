"""Gradio app with embedded Rerun 3D viewer for InstantSplat."""
import numpy as np
from pathlib import Path

import gradio as gr

try:
    from gradio_rerun import Rerun
    GRADIO_RERUN_AVAILABLE = True
except ImportError:
    GRADIO_RERUN_AVAILABLE = False

from utils.rerun_vis import is_rerun_available, log_reconstruction
from utils.reconstruction_result import ReconstructionResult


def run_init_geo_with_rerun(source_path, n_views, conf_threshold):
    """Run geometry initialization and stream results to Rerun viewer."""
    import rerun as rr

    rec = rr.RecordingStream(
        application_id="InstantSplat",
        recording_id=f"init_{Path(source_path).name}",
    )
    stream = rec.binary_stream()

    rec.log("world", rr.ViewCoordinates.RDF, static=True)
    yield stream.read()

    # --- Run the actual geometry pipeline ---
    import torch
    from mast3r.model import AsymmetricMASt3R
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from utils.sfm_utils import get_sorted_image_files, split_train_test, load_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    model = AsymmetricMASt3R.from_pretrained(ckpt).to(device)
    image_dir = Path(source_path) / "images"
    image_files, _ = get_sorted_image_files(image_dir)
    image_files = image_files[:int(n_views)]
    images, _ = load_images(image_files, size=512)

    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

    imgs = np.array(scene.imgs)
    pts3d = np.array(to_numpy(scene.get_pts3d()))
    confs = np.array([p.detach().cpu().numpy() for p in scene.im_conf])

    result = ReconstructionResult.from_scene(
        scene, imgs, pts3d, confs=confs,
        image_files=image_files,
        conf_threshold=float(conf_threshold),
    )

    # Log point cloud
    if result.colors is not None and result.colors.dtype != np.uint8:
        colors_uint8 = (np.clip(result.colors, 0, 1) * 255).astype(np.uint8)
    else:
        colors_uint8 = result.colors
    rec.log("world/points", rr.Points3D(
        positions=result.points, colors=colors_uint8, radii=0.005
    ), static=True)
    yield stream.read()

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
    yield stream.read()


def build_app():
    with gr.Blocks(title="InstantSplat + Rerun") as demo:
        gr.Markdown("# InstantSplat 3D Viewer")
        gr.Markdown("Run geometry initialization and visualize point clouds + cameras in 3D.")

        with gr.Row():
            with gr.Column(scale=1):
                source_path = gr.Textbox(
                    label="Scene Path",
                    value="assets/sora/Art",
                    info="Path to scene directory (must contain images/ subfolder)",
                )
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

            with gr.Column(scale=3):
                viewer = Rerun(
                    streaming=True,
                    height=600,
                    panel_states={
                        "time": "collapsed",
                        "blueprint": "hidden",
                        "selection": "hidden",
                    },
                )

        run_btn.click(
            run_init_geo_with_rerun,
            inputs=[source_path, n_views, conf_threshold],
            outputs=[viewer],
        )

    return demo


if __name__ == "__main__":
    if not GRADIO_RERUN_AVAILABLE:
        print("Install gradio_rerun: pip install gradio_rerun")
        exit(1)
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
