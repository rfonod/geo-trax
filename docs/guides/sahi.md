# SAHI inference

[SAHI](https://github.com/obss/sahi) (Slicing Aided Hyper Inference) runs the detector on overlapping slices of each frame instead of on the whole frame, then merges the detections before tracking. It raises recall for vehicles near the detection floor, such as footage from higher altitudes or sub-4K sensors, at roughly 5× the per-frame detection cost with the default slices. The overview is in the README under [Small-Object Detection with SAHI](https://github.com/rfonod/geo-trax#small-object-detection-with-sahi); this page covers setup and tuning.

## Setup

SAHI is an optional extra:

```bash
python -m pip install 'geo-trax[sahi]'      # from PyPI
python -m pip install -e '.[sahi]'          # from a source checkout
```

Then enable it for one run, on `extract` or `batch`:

```bash
geotrax extract video.mp4 --sahi
geotrax batch path/to/PROCESSED/ --sahi --no-geo
```

or permanently with [`extraction.sahi.enable: true`](../reference/config/extraction.md#extraction.sahi.enable) in a copied config. If the extra is missing, the run stops at once with the install command for your setup and the flag that turns SAHI off. The tracks file has the same format either way.

## Why slicing helps

The detector resizes its input to `ultralytics.imgsz` (1920 by default). A full 4K frame is therefore shrunk by half before detection, while a 1920 × 1080 slice (the default) is processed at its native resolution. Each vehicle is seen at twice the size, which is what recovers the small ones. In SAHI mode `imgsz` applies to each slice, so slices larger than `imgsz` are shrunk again and lose part of the benefit.

## Settings

All slicing and merging settings are config-only keys in [`extraction.sahi`](../reference/config/extraction.md#extractionsahi); set them in a copied config or for one run with `--set`, e.g. `--set sahi.slice_width=1280`.

| Key | Effect |
|---|---|
| `slice_height`, `slice_width` | Slice size in pixels. Smaller slices magnify more and cost more passes per frame. |
| `overlap_height_ratio`, `overlap_width_ratio` | Overlap between neighboring slices, so a vehicle cut by one slice border appears whole in the next slice. |
| `perform_standard_pred` | Also runs one full-frame pass, which keeps large vehicles (buses, trucks) that no single slice contains. |
| `postprocess_type`, `postprocess_match_metric`, `postprocess_match_threshold` | How overlapping detections from different slices are merged into one. |
| `class_agnostic` | Merge overlapping detections across classes, the SAHI counterpart of `ultralytics.agnostic_nms`. |

Of the [`ultralytics`](../reference/config/ultralytics.md) keys, SAHI mode honors `conf`, `device`, `imgsz` (per slice) and `classes`, as well as the per-class thresholds in [`extraction.class_conf`](../reference/config/extraction.md#extraction.class_conf). The NMS-related keys (`iou`, `max_det`, `agnostic_nms` and the like) are replaced by the merge settings above.

## Limitations

- YOLO models only; RT-DETR models are not supported.
- In SAHI mode the detections are fed to a tracker that geo-trax creates itself rather than to a live Ultralytics predictor. The `tracktrack` tracker and the ReID setting `model: auto` both need that predictor, so they are rejected at startup.
- The cost grows with the number of slices: halving both slice dimensions roughly quadruples the passes per frame.
