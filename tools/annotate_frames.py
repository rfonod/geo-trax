#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
annotate_frames.py - Generate YOLO-format annotations for vehicle detection in images.

This script runs vehicle detection on images using a YOLOv8 model and produces YOLO-format
annotation files with normalized bounding box coordinates. Optionally saves visualization
images with overlaid bounding boxes or masked images with blacked-out vehicle regions.

Usage:
    python tools/annotate_frames.py <source> [options]

Arguments:
    source                              : Path to the directory containing images to annotate.

Options:
    -h, --help                          : Show this help message and exit.
    -a, --annotations <path>            : Directory to save the annotation .txt files. Defaults to
                                          '<source>/../pre-labels'.
    -c, --cfg <path>                    : Pipeline config (default: the bundled geotrax/cfg/default.yaml)
                                          or a flat Ultralytics YAML. The annotator uses the config's
                                          'ultralytics:' detection settings; a bundled preset name
                                          (default, confident, lenient, stable) also works.
    -m, --model <str>                   : Override the config model — a local file path OR an
                                          'hf://<org>/<name>/<file>.pt' Hugging Face reference
                                          (auto-downloaded and cached).
    -cn, --class-names <ID=NAME|FILE>   : Override class-id -> name labels on saved visualizations with a
                                          .yaml/.json mapping file or inline ID=NAME pairs (e.g. -cn 0=car 1=bus).
    -v, --save-viz                      : Save images with colored bounding boxes overlaid.
    -z, --viz-dir <path>                : Directory to save visualization images. Defaults to
                                          '<annotations>/visualizations' when --save-viz is set.
    -mk, --save-masked                  : Save images with blacked-out vehicle regions.
    -g, --margin <float>                : Margin factor to enlarge bounding boxes for masked images
                                          only (default: 0.0).
    -f, --conf <float>                  : Override the confidence threshold from the config file.
    -i, --iou <float>                   : Override the IoU threshold for NMS from the config file.
    -sz, --imgsz <int>                  : Override the inference image size [px]; higher detects
                                          smaller/farther vehicles at the cost of speed.
    -ag, --augment / --no-augment       : Enable/disable test-time augmentation; can improve recall.
    -md, --max-det <int>                : Override the maximum number of detections per image.
    -an, --agnostic-nms / --no-...      : Enable/disable class-agnostic NMS.
    -k, --classes <ID> [<ID> ...]       : Restrict inference to the specified class IDs.
    -t, --class-conf <ID=THRESH> [...]  : Per-class confidence thresholds as CLASS_ID=THRESHOLD
                                          pairs (e.g. -t 0=0.3 1=0.5). Classes not listed fall
                                          back to the base --conf threshold. Applied as a
                                          post-inference filter affecting both annotations and
                                          visualizations.
    -s, --save-conf                     : Append the detection confidence score to each annotation
                                          line: class_id x_center y_center width height confidence.
    -o, --overwrite                     : Regenerate and overwrite existing annotation files. By
                                          default, images that already have an annotation file are
                                          skipped (and reported) so prior/edited labels are preserved.
    -hc, --hide-conf                    : Hide confidence scores on visualizations.
    -hl, --hide-labels                  : Hide class labels on visualizations.
    -w, --line-width <int>              : Line thickness for bounding boxes in visualizations
                                          (default: auto-scaled).
    -lp, --log-path <str>               : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
    -q, --quiet                         : Reduce console verbosity to important messages only (default: show INFO-level detail).

Examples:
  1. Generate annotations for images in a directory:
     python tools/annotate_frames.py path/to/images/

  2. Generate annotations and save visualization images:
     python tools/annotate_frames.py path/to/images/ --save-viz

  3. Save visualizations to a custom directory, without confidence scores and with thick lines:
     python tools/annotate_frames.py path/to/images/ -v -z path/to/viz/ -hc -w 3

  4. Override confidence, IoU, resolution, and enable test-time augmentation:
     python tools/annotate_frames.py path/to/images/ --conf 0.2 --iou 0.5 --imgsz 2560 --augment

  5. Restrict inference to class IDs 0 and 2 with per-class confidence thresholds:
     python tools/annotate_frames.py path/to/images/ -k 0 2 -t 0=0.3 2=0.6

  6. Generate masked images with enlarged bounding boxes:
     python tools/annotate_frames.py path/to/images/ --save-masked --margin 0.2

     # Re-run and overwrite previously generated annotations (default skips existing ones):
     python tools/annotate_frames.py path/to/images/ --overwrite

  7. Annotate with a locally copied, edited config (e.g. higher resolution for small vehicles):
     geotrax config copy                          # writes default_copy.yaml to the current directory
     # edit the 'ultralytics:' section (imgsz, conf, iou, augment, classes) ...
     python tools/annotate_frames.py path/to/images/ -c default_copy.yaml

Tuning:
  - The detection settings come from the config's 'ultralytics:' section. The bundled default
    baseline is conf=0.25, iou=0.7, imgsz=1920, classes=[0,1,2,3] (the four geo-trax vehicle
    classes). This is a starting point — for annotation you will often want to experiment:
      * imgsz   : higher (e.g. 2560) detects smaller/farther vehicles at the cost of speed.
      * conf    : lower (e.g. 0.2) increases recall (more candidate boxes to review/correct).
      * iou     : NMS overlap threshold for merging duplicate boxes.
      * augment : test-time augmentation; can help recall on hard frames.
      * classes : restrict or widen the detected class set.
  - Quick one-off overrides: -f/--conf, -i/--iou, -sz/--imgsz, -ag/--augment, -md/--max-det,
    -an/--agnostic-nms, -k/--classes. For persistent changes, run 'geotrax config copy', edit the
    'ultralytics:' section of the copy, and pass it via -c.

Notes:
  - Generates YOLO-format .txt files: class_id x_center y_center width height (normalized to [0,1]).
  - Images with no detections still get an output: an empty .txt (a valid YOLO "background" label)
    and, when requested, a plain visualization / masked image. With --overwrite this also clears any
    stale boxes a previous run may have written for that image.
  - Existing annotations are preserved: images whose .txt file already exists in the output
    directory are skipped (and the count is reported). Pass --overwrite to regenerate them.
  - The --margin parameter only affects masked images, not annotation coordinates or visualizations.
  - Per-class thresholds (--class-conf) are applied as a post-inference filter on top of the base
    --conf threshold. Both annotations and visualizations reflect only the surviving detections.
  - When --save-conf is set, annotation lines use the extended format:
    class_id x_center y_center width height confidence
  - Visualizations use the ultralytics color scheme with distinct colors per object class.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO
from ultralytics.utils.checks import check_yolo

from geotrax.utils.cli_utils import DEFAULT_CFG
from geotrax.utils.config_utils import load_config, resolve_class_names, resolve_model_path
from geotrax.utils.logging_utils import setup_logger


def parse_class_conf(pairs: List[str]) -> Dict[int, float]:
    """Parse CLASS_ID=THRESHOLD pairs (e.g. ['0=0.3', '1=0.5']) into a {class_id: threshold} dict."""
    class_conf: Dict[int, float] = {}
    for pair in pairs:
        try:
            class_id_str, threshold_str = pair.split('=')
            class_conf[int(class_id_str)] = float(threshold_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid --class-conf entry '{pair}'. Expected format: CLASS_ID=THRESHOLD (e.g. 0=0.3)."
            )
    return class_conf


def run_annotator(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Run the annotator."""
    if args.cfg:
        config = load_config(args.cfg, logger)
    else:
        logger.error("Error loading the configuration.")
        return

    # The model and class-rename overrides live in the 'extraction:' section of a pipeline config
    # (absent for a flat Ultralytics YAML); capture them before narrowing to 'ultralytics:' below.
    extraction_cfg = config.get('extraction', {}) if isinstance(config, dict) else {}
    cfg_model = extraction_cfg.get('model')
    cfg_class_rename = extraction_cfg.get('class_rename')

    # Use the 'ultralytics:' section of a geo-trax pipeline config; fall back to a flat
    # Ultralytics YAML if the user supplies one directly.
    config = config.get('ultralytics', config)
    config['mode'] = 'predict'  # the pipeline config uses mode: track; the annotator only predicts
    # A CLI --model override takes precedence over the config value; either form (local path or
    # hf:// reference) is resolved to a concrete local file (auto-downloaded and cached if hf://).
    config['model'] = str(resolve_model_path(args.model or cfg_model or config.get('model'), logger))

    # Apply CLI overrides to config
    if args.conf is not None:
        config['conf'] = args.conf
    if args.iou is not None:
        config['iou'] = args.iou
    if args.imgsz is not None:
        config['imgsz'] = args.imgsz
    if args.augment is not None:
        config['augment'] = args.augment
    if args.max_det is not None:
        config['max_det'] = args.max_det
    if args.agnostic_nms is not None:
        config['agnostic_nms'] = args.agnostic_nms
    if args.classes is not None:
        config['classes'] = args.classes

    # Parse per-class confidence thresholds
    class_conf = parse_class_conf(args.class_conf) if args.class_conf else {}
    base_conf = config.get('conf', 0.25)

    model = load_detector(config, logger)

    # Apply a class-name override (CLI > config) to the model so saved visualizations use the custom
    # labels. Without an override the model keeps its own embedded names.
    if args.class_names is not None or cfg_class_rename is not None:
        model.names, _ = resolve_class_names(
            Path(config['model']), args.class_names, cfg_class_rename, config.get('classes'), logger
        )

    output_dir = Path(args.annotations) if args.annotations else args.source.parent / "pre-labels"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Resolve visualization and masked output directories
    if args.save_viz:
        viz_dir = Path(args.viz_dir) if args.viz_dir else output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True, parents=True)
    if args.save_masked:
        masked_dir = output_dir / "masked"
        masked_dir.mkdir(exist_ok=True, parents=True)

    det_results = model(args.source, **config, stream=True)

    logger.info(f"Annotating images in '{args.source}'...")
    written = skipped = 0
    for result in det_results:
        annotation_path = output_dir / f"{Path(result.path).stem}.txt"
        if annotation_path.exists() and not args.overwrite:
            logger.info(f"Annotation already exists, skipping '{annotation_path.name}' (use --overwrite to regenerate).")
            skipped += 1
            continue

        # Apply per-class confidence filtering (post-inference).
        # Filters result.boxes so that annotations, visualizations, and masked images stay consistent.
        if class_conf:
            all_class_ids = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()
            keep_indices = [
                i for i, (cls_id, conf_val) in enumerate(zip(all_class_ids, confs))
                if conf_val >= class_conf.get(cls_id, base_conf)
            ]
            result.boxes = result.boxes[keep_indices]

        # Images with no (surviving) detections are still written: an empty annotation file is a
        # valid YOLO "background" label, and the saved visualization / masked image is just the plain
        # frame. This also ensures --overwrite clears stale boxes from a previous run.
        class_ids = result.boxes.cls.int().tolist()
        boxes_xywhn = result.boxes.xywhn
        boxes_xywh = result.boxes.xywh
        box_confs = result.boxes.conf.tolist()

        # Save YOLO-format annotations (empty file when there are no detections)
        with open(annotation_path, "w") as f:
            for box, class_id, conf_val in zip(boxes_xywhn, class_ids, box_confs):
                line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                if args.save_conf:
                    line += f" {conf_val:.6f}"
                f.write(line + "\n")
        written += 1

        # Save visualization with colored bounding boxes
        if args.save_viz:
            plot_args = {
                'conf': args.show_conf,
                'labels': args.show_labels,
            }
            if args.line_width is not None:
                plot_args['line_width'] = args.line_width
            annotated_img = result.plot(**plot_args)
            cv2.imwrite(str(viz_dir / Path(result.path).name), annotated_img)

        # Save masked images with blacked-out regions
        if args.save_masked:
            img = cv2.imread(result.path)
            for box in boxes_xywh:
                xc, yc, w, h = box.int().tolist()
                w = int(w * (1 + args.margin))
                h = int(h * (1 + args.margin))
                x = int(xc - w / 2)
                y = int(yc - h / 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
            cv2.imwrite(str(masked_dir / Path(result.path).name), img)

    logger.notice(f"Annotations saved to '{output_dir}' ({written} written, {skipped} skipped).")
    if skipped:
        logger.notice(f"{skipped} existing annotation(s) were left unchanged — pass --overwrite to regenerate them.")
    if args.save_viz:
        logger.notice(f"Visualizations saved to '{viz_dir}'.")
    if args.save_masked:
        logger.notice(f"Masked images saved to '{masked_dir}'.")


def load_detector(config: Dict, logger: logging.Logger) -> YOLO:
    """Load the detection model."""
    try:
        model = YOLO(model=config['model'], task=config['task'])
    except Exception as e:
        logger.error(f"Error loading detection model: {e}")
        return
    else:
        logger.info(f"Detection model '{config['model']}' loaded successfully.")

    check_yolo(device=config['device'])

    return model


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Annotate images with vehicle bounding boxes using a YOLOv8 model.')

    # Required
    parser.add_argument('source', type=Path,
                        help='Path to the directory containing images to annotate')

    # Output paths
    parser.add_argument('--annotations', '-a', type=Path,
                        help='Directory to save annotation .txt files (default: <source>/../pre-labels)')
    parser.add_argument('--cfg', '-c', type=Path, default=DEFAULT_CFG,
                        help="Pipeline config (a bundled preset name or a path) or a flat Ultralytics YAML; "
                             "the annotator uses its 'ultralytics:' detection settings.")
    parser.add_argument('--model', '-m', nargs='+', default=None, metavar='MODEL',
                        help="Detection model overriding the config — a local file path OR an "
                             "'hf://<org>/<name>/<file>.pt' Hugging Face reference (auto-downloaded and cached).")
    parser.add_argument('--class-names', '-cn', nargs='+', default=None, metavar='ID=NAME|FILE',
                        help="Override class-id -> name labels on saved visualizations: a .yaml/.json mapping "
                             "file or inline ID=NAME pairs (e.g. -cn 0=car 1=bus).")

    # Output modes
    parser.add_argument('--save-viz', '-v', action='store_true',
                        help='Save images with colored bounding boxes overlaid')
    parser.add_argument('--viz-dir', '-z', type=Path,
                        help='Directory to save visualization images (default: <annotations>/visualizations)')
    parser.add_argument('--save-masked', '-mk', action='store_true',
                        help='Save images with blacked-out vehicle regions')
    parser.add_argument('--margin', '-g', type=float, default=0.0,
                        help='Margin factor to enlarge bounding boxes for masked images only (default: 0.0)')
    parser.add_argument('--save-conf', '-s', action='store_true',
                        help='Append detection confidence to each annotation line')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Regenerate and overwrite existing annotation files (default: skip images already annotated)')

    # Inference overrides (each overrides the matching key in the config's 'ultralytics:' section)
    parser.add_argument('--conf', '-f', type=float, default=None,
                        help='Override the confidence threshold from the config file')
    parser.add_argument('--iou', '-i', type=float, default=None,
                        help='Override the IoU threshold for NMS from the config file')
    parser.add_argument('--imgsz', '-sz', type=int, default=None,
                        help='Override the inference image size [px]; higher detects smaller/farther vehicles, slower')
    parser.add_argument('--augment', '-ag', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable test-time augmentation (--augment / --no-augment); can improve recall')
    parser.add_argument('--max-det', '-md', type=int, default=None,
                        help='Override the maximum number of detections per image')
    parser.add_argument('--agnostic-nms', '-an', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable class-agnostic NMS (--agnostic-nms / --no-agnostic-nms)')
    parser.add_argument('--classes', '-k', type=int, nargs='+', metavar='CLASS_ID',
                        help='Restrict inference to the given class IDs (e.g. -k 0 1 2)')
    parser.add_argument('--class-conf', '-t', nargs='+', metavar='CLASS_ID=THRESHOLD',
                        help='Per-class confidence thresholds, e.g. -t 0=0.3 1=0.5. '
                             'Classes not listed fall back to the base --conf threshold.')

    # Visualization style
    parser.add_argument('--hide-conf', '-hc', dest='show_conf', action='store_false', default=True,
                        help='Hide confidence scores on visualizations')
    parser.add_argument('--hide-labels', '-hl', dest='show_labels', action='store_false', default=True,
                        help='Hide class labels on visualizations')
    parser.add_argument('--line-width', '-w', type=int, default=None,
                        help='Line thickness for bounding boxes in visualizations (default: auto-scaled)')

    # Logging
    parser.add_argument('--log-path', '-lp', type=Path, default=None,
                        help='Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce console verbosity to important messages only (default: show INFO-level detail).')

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_cli_args()
    if isinstance(args.model, list):
        args.model = ' '.join(args.model)
    logger = setup_logger(Path(__file__).stem, verbose=not args.quiet, log_path=args.log_path)

    run_annotator(args, logger)


if __name__ == '__main__':
    main()
