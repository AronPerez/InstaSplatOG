# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InstantSplat — sparse-view 3D Gaussian Splatting reconstruction from images. Three-stage pipeline: geometry initialization (MASt3R) → Gaussian optimization (train) → novel-view rendering.

## Essential Commands

### Full inference pipeline (no evaluation)
```bash
# Edit DATA_ROOT_DIR in script first, then:
bash scripts/run_infer.sh
```

### Full evaluation pipeline (with GT comparison)
```bash
bash scripts/run_eval.sh
```

### Quick run with Rerun visualization
```bash
./scripts/run_with_rerun.sh <DATA_ROOT> <SCENE> [N_VIEWS] [ITERATIONS]
# Example:
./scripts/run_with_rerun.sh ./assets/sora Art 3 1000
```

### Individual pipeline steps
```bash
# 1. Geometry initialization
python init_geo.py -s <source_path> -m <model_path> --n_views 3 --focal_avg --co_vis_dsp --conf_aware_ranking

# 2. Training (joint pose + Gaussian optimization)
python train.py -s <source_path> -m <model_path> --n_views 3 --iterations 1000 --pp_optimizer --optim_pose

# 3. Rendering
python render.py -s <source_path> -m <model_path> --n_views 3 --iterations 1000 --infer_video

# 4. Metrics (PSNR, SSIM, LPIPS)
python metrics.py -s <source_path> -m <model_path> --n_views 3
```

### Gradio web UI
```bash
python app_rerun.py
```

### Build CUDA submodules (if not already built)
```bash
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/fused-ssim
cd croco/models/curope/ && python setup.py build_ext --inplace
```

## Conda Environment Setup

```bash
# 1. Install Miniconda (if not installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda init bash

# 2. Create environment
conda create -n instantsplat python=3.10 cmake -y
conda activate instantsplat

# 3. Install PyTorch via pip (NOT conda — conda-installed PyTorch causes
#    "undefined symbol: iJIT_NotifyEvent" errors from Intel MKL/JIT conflicts)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install pip dependencies
pip install -r requirements.txt

# 5. Build CUDA submodules (--no-build-isolation required so setup.py can import torch)
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/fused-ssim

# 6. Build CroCo RoPE kernels
cd croco/models/curope/ && python setup.py build_ext --inplace && cd ../../..

# 7. (Optional) Rerun visualization
pip install rerun-sdk==0.27.3 gradio_rerun==0.27.3
```

## Architecture

### Pipeline Flow
```
Images → init_geo.py (MASt3R → DUSt3R global alignment → COLMAP format)
      → train.py (GaussianModel optimization with optional pose refinement)
      → render.py (novel view synthesis + evaluation)
```

### Key Modules
- **`dust3r/`, `mast3r/`, `croco/`** — Local modules (NOT pip packages). Must run all scripts from repo root for imports to resolve.
- **`scene/gaussian_model.py`** — Core GaussianModel class: stores per-Gaussian parameters (xyz, SH features, scaling, rotation, opacity) and optimizer setup.
- **`scene/__init__.py`** — Scene class: loads COLMAP data, manages camera lists, initializes Gaussians from point cloud.
- **`gaussian_renderer/__init__.py`** — `render()` function: transforms Gaussians to camera frame, calls CUDA rasterizer, returns rendered image.
- **`arguments/__init__.py`** — Three param groups: `ModelParams`, `PipelineParams`, `OptimizationParams`. Config can be loaded from `<model_path>/cfg_args`.
- **`utils/sfm_utils.py`** — Image loading, COLMAP I/O, co-visibility mask computation.
- **`utils/pose_utils.py`** — Quaternion/rotation conversions, camera pose optimization.

### Segmentation & Editing (current branch)
- **`utils/segmentation.py`** — GroundingDINO detection + SAM2 instance segmentation → 3D projection via pts3d.
- **`utils/scene_editor.py`** — SceneEditor: applies translate/rotate/scale transforms to segmented object Gaussians.
- **`utils/rerun_vis.py`** — Rerun 3D visualization (gracefully no-ops if rerun-sdk not installed).
- **`app_rerun.py`** — Gradio web UI combining geometry init, segmentation, object editing, and export.

### CUDA Submodules (`submodules/`)
- `simple-knn` — KNN distance computation for Gaussian initialization
- `diff-gaussian-rasterization` — Differentiable Gaussian rasterizer (forward + backward)
- `fused-ssim` — Fused SSIM loss computation

### Data Format
Input: directory of images at `<source_path>/images/`. Geometry init outputs COLMAP-format files under `<model_path>/sparse_<n_views>/` (cameras, images, points3D). Training saves checkpoints and `cfg_args` to `<model_path>/`.

## Runtime Notes

- All scripts must be run from the repo root (`/root/InstaSplatOG/`) — `dust3r` and `mast3r` are imported as local modules via CWD.
- MASt3R checkpoint must exist at `mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`.
- No test suite exists. Verify changes by running the pipeline on sample data in `assets/sora/`.
- No linter or formatter is configured.
- Loss function: `(1 - λ) * L1 + λ * (1 - SSIM)` where λ = `lambda_dssim` (default 0.2).
- `rerun-sdk==0.27.3` and `gradio_rerun==0.27.3` are required — version 0.28+ removed the `rr.new_recording()` API used by `app_rerun.py`.
