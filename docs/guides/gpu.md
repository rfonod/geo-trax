# GPU acceleration

Object **detection** already runs on CUDA whenever a compatible GPU and PyTorch build are present (the [`ultralytics.device`](../reference/config/ultralytics.md#ultralytics.device) key, auto by default). The **stabilization** (`--stab-gpu`) and **georeferencing** (`--geo-gpu`) image-matching steps can *optionally* be CUDA-accelerated too, through [Stabilo](https://github.com/rfonod/stabilo). This needs a CUDA-enabled OpenCV build and is Linux and Windows only. Stabilo accelerates the **ORB** detector only, so `--geo-gpu` also requires `--geo-detector orb`. There is no CPU fallback: requesting the GPU without a working CUDA device raises an error.

!!! note
    This page covers the **OpenCV CUDA** path, which applies to the classical detectors. The learning-based detectors use a separate torch device (`--stab-device` / `--geo-device`) and are covered in [Feature detectors](feature-detectors.md#acceleration); the two paths cannot be combined.

Throughout this page, dotted names like `ultralytics.device` or `tracker.botsort.gmc_method` are **keys in the pipeline config**. Change one for a single run with `--set KEY=VALUE`, or permanently in a copied config; see [Configuration](configuration.md).

## 1. Build OpenCV with CUDA

The PyPI OpenCV wheels are CPU-only, so you must build `opencv-contrib-python` from source with CUDA. Follow Stabilo's [CUDA build guide](https://rfonod.github.io/stabilo/cuda/), then confirm it works:

```bash
python -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"   # expect: <version> 1
```

!!! warning "Add `video` to the `BUILD_LIST`"
    geo-trax needs one OpenCV module beyond the minimal stabilo build. The minimal list in stabilo's guide omits `video`, but the `sparseOptFlow` global-motion-compensation (GMC) method uses `cv2.calcOpticalFlowPyrLK` from that module. It is the default for the **BoT-SORT** (the active default) and **TrackTrack** trackers, and an option for **DeepOCSORT** (`tracker.<name>.gmc_method`). GMC runs during **track association on the original, pre-stabilization frames**, so Stabilo does not make it redundant. Without `video` you get a `cv2 has no attribute 'calcOpticalFlowPyrLK'` warning and GMC silently falls back to identity.

    If you would rather not rebuild, set `gmc_method: none` for your active tracker ([step 3](#3-enable-gpu-per-stage)). That disables camera-motion compensation in tracking; for near-nadir BEV or quasi-stationary drone footage that is often acceptable, but check your tracking quality before relying on it.

## 2. Install geo-trax without clobbering your CUDA OpenCV

`ultralytics[extra]` transitively requires `opencv-python` and `opencv-python-headless`, both CPU wheels. A plain install drops them on top of your compiled `cv2/` and disables CUDA. A transitive dependency cannot be excluded in `pyproject.toml`, so use one of these:

### A. Install, then restore your wheel (recommended)

Let pip install everything, then overwrite the CPU OpenCV by reinstalling the CUDA `opencv_contrib_python-*.whl` you built in step 1:

```bash
python -m pip install geo-trax   # or, from a source checkout: python -m pip install -e .
                                 # installs torch/cuda/ultralytics/..., plus (temporarily) CPU opencv
# reinstall the wheel produced by your CUDA OpenCV build in step 1 (path is wherever you built it):
python -m pip install --force-reinstall --no-deps /path/to/opencv_contrib_python-*.whl
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"   # expect: 1
```

If the CUDA check above reports `0` devices (or a `cv2`/CUDA error such as `module 'cv2' has no attribute 'cuda'`) immediately after the reinstall, re-activate the CUDA venv (`source .venv-cuda/bin/activate`, matching stabilo's [CUDA build guide](https://rfonod.github.io/stabilo/cuda/)) and run it again; if it persists, confirm `cv2.__file__` points into that venv's `site-packages` (see the [troubleshooting section](https://rfonod.github.io/stabilo/cuda/#9-troubleshooting) of stabilo's CUDA build guide). It should then report `1`.

!!! danger
    Afterwards `opencv-python` / `opencv-python-headless` stay registered, but their recorded files now belong to your build. **Never** `pip uninstall opencv-python` / `opencv-python-headless`, or you will delete the shared `cv2/`.

### B. Stub the CPU wheels first

No download and cleaner metadata: install two metadata-only packages so pip treats the requirements as already satisfied and never fetches a CPU wheel.

```bash
CVVER=$(python -c "import cv2; print(cv2.__version__)")
for pkg in opencv-python opencv-python-headless; do
  d=$(mktemp -d)
  cat > "$d/pyproject.toml" <<EOF
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[project]
name = "$pkg"
version = "$CVVER"
[tool.setuptools]
py-modules = []
EOF
  python -m pip install --no-deps "$d" && rm -rf "$d"
done
python -m pip install geo-trax   # or python -m pip install -e . from a source checkout;
                                 # opencv is satisfied by the stubs, so your CUDA build is untouched
```

## 3. Enable GPU per stage

| Stage | How | Notes |
|-------|-----|-------|
| Detection | automatic | `ultralytics.device` (auto = CUDA when available) |
| Stabilization | `--stab-gpu` | works out of the box (default detector is ORB) |
| Georeferencing | `--geo-gpu --geo-detector orb` | ORB is required; RootSIFT is not CUDA-accelerated |

Only ORB is CUDA-accelerated, so georeferencing must be switched to it, which `--geo-detector orb` does on the command line. If you did **not** add `video` to your OpenCV `BUILD_LIST` (step 1), also turn off the tracker's GMC with `--set tracker.botsort.gmc_method=none` (for another active tracker, use its block, e.g. `tracker.tracktrack.gmc_method`), and verify the tracking quality.

These are the GPU-accelerated analogues of the two reproduce commands in [`data/README.md`](https://github.com/rfonod/geo-trax/blob/main/data/README.md). Their outputs are *equivalent, not identical* to the committed `data/results-pixel/` and `data/results-full/`: the CUDA path georeferences with ORB (RootSIFT is not GPU-accelerated), GPU feature extraction is not bit-identical to CPU, and with `gmc_method: none` the tracker runs without motion compensation.

```bash
# Pixel-coordinate results: GPU stabilization, no georeferencing
geotrax batch data/U_video_cut.mp4 --no-geo --show-class-names --show-conf --stab-gpu

# Full pipeline: GPU stabilization + georeferencing (needs the orthophoto/segmentation/master-frame data; see data/README.md)
geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -osf data/segmentations -mf data/master_frames \
  --show-lanes --plot-segmentations -vm 0 3 --stab-gpu --geo-gpu --geo-detector orb
```

Append `--set tracker.botsort.gmc_method=none` to both if you needed the GMC workaround above.

## 4. Benchmark: detection-only CUDA vs fully CUDA

To isolate the GPU effect, disable the CPU-bound visualization and plotting (`--no-save --no-show --no-plot-save --no-plot-show`) and wipe a scratch output folder before each run (a clean, non-skipped run that leaves your real results untouched). Do **not** pass `--recompute`: the master → orthophoto homography is a one-time, expensive registration normally cached in the master-frames folder, so leaving it cached keeps the benchmark on the per-video work (detection, stabilization, reference → master registration).

Three configurations are compared:

- **geo-trax default (CPU)**: the shipped defaults with RootSIFT georeferencing; what a CPU-only user actually runs.
- **ORB (CPU)**: the GPU-tuned config on CPU; a controlled, same-config baseline for the GPU run.
- **fully CUDA**: the same ORB config with `--stab-gpu --geo-gpu`.

The default run uses the shipped defaults (RootSIFT georeferencing, CPU for both stabilization and georeferencing). The ORB runs switch the georeferencing detector with `--geo-detector orb`. If you needed the GMC workaround, append `--set tracker.botsort.gmc_method=none` to `COMMON` (or to every `bench` line), so that all runs do the same work.

With [hyperfine](https://github.com/sharkdp/hyperfine):

```bash
OUT=/tmp/geotrax_bench
COMMON="geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -osf data/segmentations -mf data/master_frames --no-save --no-show --no-plot-save --no-plot-show -of $OUT"
hyperfine --warmup 1 --runs 5 --prepare "rm -rf $OUT" \
  -n "geo-trax default (CPU, RootSIFT)" "$COMMON" \
  -n "ORB (CPU)"                        "$COMMON --geo-detector orb" \
  -n "fully CUDA (ORB)"                 "$COMMON --geo-detector orb --stab-gpu --geo-gpu"
```

Or without extra tools:

```bash
bench () { local label="$1"; shift; rm -rf /tmp/geotrax_bench
  local t0=$(date +%s.%N)
  geotrax batch data/U_video_cut.mp4 -orf data/orthophotos -osf data/segmentations -mf data/master_frames \
    --no-save --no-show --no-plot-save --no-plot-show -y -o -of /tmp/geotrax_bench "$@" >/tmp/bench.log 2>&1 \
    && awk -v a="$t0" -v b="$(date +%s.%N)" -v l="$label" 'BEGIN{printf "%-28s %.1f s\n", l, b-a}' \
    || echo "$label FAILED (see /tmp/bench.log)"; }
bench "geo-trax default (CPU)"
bench "ORB (CPU)"        --geo-detector orb
bench "fully CUDA (ORB)" --geo-detector orb --stab-gpu --geo-gpu
```

Example on an **NVIDIA RTX 4090** (5-second sample clip, 150 frames; hyperfine mean ± σ over 5 runs):

??? info "Benchmark system"
    - **OS**: Ubuntu 24.04.4 LTS (Linux 6.8, x86_64)
    - **CPU**: 13th Gen Intel Core i9-13900KF (24 cores / 32 threads)
    - **RAM**: 62 GiB
    - **GPU**: NVIDIA GeForce RTX 4090 (24 GB, driver 580.159.03)
    - **Software**: Python 3.11.5, torch 2.12.1 (CUDA 13.0), OpenCV 4.13.0, stabilo 1.3.0

| Pipeline | Wall time | vs default |
|----------|-----------|------------|
| CPU stabilization and georeferencing (geo-trax defaults) | 273.7 ± 0.9 s | 1× |
| CPU stabilization and georeferencing (GPU-matched config) | 348.4 ± 0.4 s | 0.8× |
| Fully CUDA (GPU-matched config) | 16.4 ± 2.0 s | **16.7×** |

Fully CUDA is **16.7× faster than the geo-trax default** and **21.3× faster than the same ORB config on CPU** (row 2). Row 1 is the shipped defaults (RootSIFT georeferencing); rows 2 and 3 use the GPU-matched config, which switches georeferencing to ORB so the CPU and GPU runs do identical work (only ORB is CUDA-accelerated). Row 2 is slower than row 1 because ORB at a 250k feature ceiling with brute-force matching is costlier on CPU than RootSIFT.

Treat these as a **relative** comparison, not absolute throughput: geo-trax's defaults favor maximum accuracy and reliability, with detection at **1920×1920**, stabilization at only a **0.5 downscale** (roughly 2K per frame on this 4K clip) with a high `max_features` ceiling, and RootSIFT georeferencing with a very high `max_features` and conservative MAGSAC++ matcher and projection settings, all against an **8000×8000** orthophoto. Lighter settings would cut absolute times across the board; the point is the CPU → GPU ratio.
