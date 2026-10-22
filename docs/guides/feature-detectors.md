# Feature detectors

Both image-matching steps rely on feature detection: **stabilization** registers every video frame to a reference frame, and **georeferencing** registers the reference frame (or a master frame) to an orthophoto. The detector is chosen per step:

| Step | Default | Flag | Config key |
|---|---|---|---|
| Stabilization | `orb` | `--stab-detector` | [`stabilo.detector_name`](../reference/config/stabilo.md#stabilo.detector_name) |
| Georeferencing | `rsift` | `--geo-detector` | [`georef.matching.detector_name`](../reference/config/georef.md#georef.matching.detector_name) |

The stabilization default is `orb`, tuned for high-altitude BEV footage with [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) and used for every published result. Georeferencing defaults to RootSIFT, since matching a drone frame against an orthophoto is a cross-domain match, often across years, seasons and lighting, and much harder than matching consecutive video frames.

## Classical detectors

| Detector | Description |
|---|---|
| `orb` | fast binary descriptor; the only CUDA-accelerated option |
| `sift` | classic scale-invariant descriptor |
| `rsift` | RootSIFT, a normalized SIFT variant with better matching accuracy |
| `brisk` | binary, scale- and rotation-invariant |
| `kaze` | nonlinear scale space, slower but accurate |
| `akaze` | accelerated KAZE, a good speed/accuracy compromise |

All six are OpenCV detectors, rotation invariant, and need no extra downloads. Each keeps at most `max_features` keypoints per frame, the strongest by response, counted in the downsampled image. ORB and SIFT do this natively; BRISK, KAZE and AKAZE first detect every keypoint above their threshold ([`brisk_threshold`](../reference/config/stabilo.md#stabilo.brisk_threshold), `kaze_threshold`, `akaze_threshold`, OpenCV's defaults) and then keep the strongest. The cap is a maximum, not a target: if a frame yields fewer keypoints than `max_features`, lower the threshold.

```bash
geotrax extract video.mp4 --stab-detector akaze
geotrax georeference video.mp4 -orf path/to/orthophotos --geo-detector sift
```

Masking, RANSAC and downsampling parameters live under [`stabilo`](../reference/config/stabilo.md) and [`georef.matching`](../reference/config/georef.md#georefmatching). To skip stabilization entirely, set `extraction.stabilize: false`.

## Learning-based detectors

Stabilo 1.4.0 also exposes five [kornia](https://kornia.readthedocs.io/) models: `xfeat`, `disk`, `dedode`, `keynet`, and the detector-free `loftr`, plus the learned `lightglue` matcher (which pairs only with `disk`, `dedode` and `keynet`). They are available for experimentation on both steps, but they are **not** used by any geo-trax default and were not used to produce any published result. On this workload the classical detectors are faster, more predictable, and easier to reproduce.

Two properties make them awkward here.

### Rotation

`xfeat`, `disk`, `dedode` and `loftr` are *upright* models: matching collapses once the two images differ by more than roughly 30°. That is tolerable for stabilization, where consecutive frames are nearly aligned, but a poor fit for georeferencing, where a drone frame is registered against a north-up orthophoto and arbitrary rotation is the norm. Only `keynet` estimates keypoint orientation.

### Memory

They are far hungrier than classical detectors, and their cost grows with the *processed* frame size. **This can exhaust system memory and hang the machine.** Pick `downsample_ratio` so that `input_pixels × downsample_ratio²` stays under the model's guideline:

| Detector | Guideline | Max ratio at 4K (8.3 MP) | Max ratio at 8000 × 8000 ortho (64 MP) |
|---|---|---|---|
| `xfeat` | 2.0 MP | `0.49` | `0.17` |
| `disk`, `dedode`, `keynet` | 0.5 MP | `0.24` | `0.08` |
| `loftr` | 0.3 MP (cost is **quadratic** in pixels) | `0.19` | `0.06` |

Stabilo warns when the processed frame exceeds its guideline but does not cap it, so heed the warning. geo-trax does cap `max_features` on the georeferencing path, since these models consume it as `top_k` and the RootSIFT-sized default of 250000 would be fatal. An out-of-memory error stops the run instead of being reported as a failed frame.

### Acceleration

These models run on PyTorch, so they are GPU-accelerated straight from the CLI with `--stab-device` / `--geo-device` (`auto` picks cuda > mps > cpu, and is the default). Nothing extra is needed: unlike the OpenCV CUDA path in [GPU acceleration](gpu.md), which requires a source-built OpenCV and covers only ORB, this works on stock wheels, accelerates every learned detector and the `lightglue` matcher, and supports `mps` on Apple silicon. A GPU with dedicated VRAM is also the safer choice memory-wise, since it does not compete with system RAM. The two paths are mutually exclusive; Stabilo rejects `--stab-gpu`/`--geo-gpu` combined with a learned detector.

Weights download on first use and are cached by torch (`~/.cache/torch/hub/checkpoints`; set `TORCH_HOME` to relocate), then reused offline. Set the ratio well below the maximum before the first run:

```bash
geotrax extract video.mp4 --stab-detector xfeat --stab-device cuda --set stabilo.downsample_ratio=0.25
```

The model weights (`loftr_weights`, `disk_weights`, `dedode_detector_weights`, `dedode_descriptor_weights`) and `loftr_confidence` are config-only keys in [`stabilo`](../reference/config/stabilo.md) and [`georef.matching`](../reference/config/georef.md#georefmatching).

!!! note "Georeferencing cache"
    Every key in `georef.matching` feeds the hash that validates the cached master → orthophoto homography, so changing the detector or its settings recomputes that cache on the next run. Switching only `device`, `gpu` or `gpu_device_id` does not; see [Master frames](georeference.md#master-frames).
