#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
batch_process.py - Primary Entry Point for the Full Geo-trax Pipeline

This is the go-to script for running Geo-trax. It orchestrates the complete pipeline —
detection/tracking/stabilization, georeferencing, visualization, and plotting — for a single
video file or an entire directory tree of videos. Key features:
- Smart skip-if-exists: each stage checks for its output before running; only missing results
  are (re-)computed, making re-runs fast and safe.
- Selective stage control: --viz-only, --geo-only, --no-geo, and --plot-only let you re-run
  or skip any subset of stages without touching the rest.
- Dry-run mode (--dry-run): prints exactly which files and stages would run — nothing is
  executed. Combine with --verbose for a full preview before committing to a long batch job.
- Batch-directory mode: recursively finds all videos under the input path, with optional
  exclusion of specific sub-folders (--folders-exclude) or filename patterns
  (--exclude-patterns). Note: --cut-frame-right is silently ignored in batch-directory mode
  so that every video is always processed in full.

Usage:
  geotrax batch <input> [options]

Arguments:
  input : Path to a video file (.mp4, .mov, .avi, .mkv) or a directory of video files
          (searched recursively).

Batch Processing Options:
    --help, -h          : Show this help message and exit.
    --yes, -y           : Skip all overwrite confirmation prompts (use with --overwrite).
    --overwrite, -o     : Re-run stages whose output already exists and overwrite it.
    --dry-run, -dr      : Preview which files and stages would be processed without
                          executing anything. Add --verbose for full per-stage detail.
    --viz-only, -vo     : Skip detection, tracking, stabilization, and georeferencing;
                          only (re-)run visualization. Requires existing .txt tracking results.
    --geo-only, -go     : Skip detection, tracking, and stabilization; only (re-)run
                          georeferencing. Requires existing .txt tracking results.
    --plot-only, -po    : Skip all pipeline stages; only (re-)generate plots from existing
                          results.
    --no-geo, -ng       : Skip georeferencing; run detection/tracking/stabilization and
                          visualization only (pixel-coordinate output).
    --folders-exclude, -fe <str> [<str> ...] : Sub-folder names to skip when scanning for
                          videos. Defaults to cfg -> batch -> folders_exclude (['results']).
    --exclude-patterns, -ep <str> [<str> ...] : Skip videos whose filename contains any of
                          these substrings (e.g., --exclude-patterns test temp).
                          Defaults to cfg -> batch -> exclude_patterns.

Shared Options:
    --cfg, -c <path>    : Path to the main geo-trax configuration file (default: geotrax/cfg/default.yaml).
    --log-path, -lp <str> : Where to write logs: a directory or a full file path; defaults to a platform-specific log directory.
    --verbose, -v       : Set print verbosity level to INFO (default: WARNING).

Processing Options:
    --conf, -co <float>   : Detection confidence threshold. Defaults to cfg -> cfg_ultralytics -> conf.
    --classes, -cls <int> [<int> ...] : Vehicle class IDs to extract (e.g., --classes 0 1 2).
                          Defaults to cfg -> cfg_ultralytics -> classes.
    --cut-frame-left, -cfl <int> : Skip the first N frames. Defaults to cfg -> processing -> cut_frame_left.
    --cut-frame-right, -cfr <int> : Stop at this frame number. Applies to single-file input only;
                          silently ignored in batch-directory mode. Defaults to cfg -> processing -> cut_frame_right.
    For full detection and tracking control (model, IoU, image size, tracker settings, etc.),
    edit geotrax/cfg/ultralytics/default.yaml and the linked tracker config.

Georeferencing Options:
    --ortho-folder, -of <path>     : Path to the folder with orthophotos (.png, .tif, .txt).
                          Defaults to cfg -> folders -> ortho_folder, then 'ORTHOPHOTOS' at the
                          same level as 'PROCESSED' in input.
    --geo-source, -gs <choice>     : Georeferencing parameter source: metadata-tif, text-file, or
                          center-text-file. Auto-detected if omitted.
                          Defaults to cfg -> georef -> processing -> geo_source.
    --ref-frame, -rf <int>         : Reference frame number; must match the value used for
                          stabilization. Defaults to cfg -> georef -> processing -> ref_frame.
    --no-master, -nm               : Disable the master-frame approach regardless of config.
                          When not set, cfg -> georef -> processing -> use_master applies.
    --master-folder, -mf <path>    : Path to the folder containing master frame files (.png).
                          Defaults to cfg -> folders -> master_folder, then '<ortho-folder>/master_frames'.
    --recompute, -r                : Force recomputation of the master→orthophoto homography
                          even if a cached result exists.
                          Defaults to cfg -> georef -> processing -> recompute.
    --segmentation-folder, -osf <path> : Path to the folder with road segmentation CSV files
                          (used for lane assignment during georeferencing) and, when
                          --plot-segmentations is enabled, the corresponding overlay PNG files
                          used as plot backgrounds. Defaults to cfg -> folders -> segmentation_folder,
                          then '<ortho-folder>/segmentations'.

Visualization Options:
    --save / --no-save, -s  : Save the annotated output video to file.
                          Defaults to cfg -> visualization -> save.
    --show / --no-show, -sh : Open a live preview window during processing.
                          Defaults to cfg -> visualization -> show.
    --viz-mode, -vm <int> [<int> ...] : Annotation source frame(s): 0=original, 1=stabilized,
                          2=reference frame. Accepts multiple values (e.g. 0 1 2) to render one
                          video per mode. Defaults to cfg -> visualization -> viz_mode.
    --plot-trajectories, -pt : Overlay all trajectory positions on the first frame before the
                          main annotation pass. Defaults to cfg -> visualization -> plot_trajectories.
    --plot-delay, -pd <int> : Frames to hold the trajectory overlay; only relevant with
                          --plot-trajectories. Defaults to cfg -> visualization -> plot_delay.
    --show-conf, -sc    : Include detection confidence score in bounding-box labels.
                          Defaults to cfg -> visualization -> show_conf.
    --show-lanes, -sl   : Include lane ID in bounding-box labels (requires georeferencing).
                          Defaults to cfg -> visualization -> show_lanes.
    --show-class-names, -scn : Include vehicle class name in bounding-box labels.
                          Defaults to cfg -> visualization -> show_class_names.
    --hide-labels, -hl  : Suppress all text label overlays.
                          Defaults to cfg -> visualization -> hide_labels.
    --hide-tracks, -ht  : Suppress track tail lines.
                          Defaults to cfg -> visualization -> hide_tracks.
    --hide-speed, -hs   : Suppress speed values in labels (requires georeferencing).
                          Defaults to cfg -> visualization -> hide_speed.
    --speed-unit, -su <choice> : Speed display unit: km/h or mi/h.
                          Defaults to cfg -> visualization -> speed_unit.
    --speed-deadzone, -sdz <float> : Floor displayed speeds <= this value (in the chosen speed unit) to 0; 0 disables.
                          Defaults to cfg -> visualization -> speed_deadzone.
    --class-filter, -cf <int> [<int> ...] : Vehicle class IDs to exclude from visualization
                          (e.g., -cf 1 2 hides buses and trucks).
                          Defaults to cfg -> visualization -> class_filter.

Plotting Options:
    --plot-save / --no-plot-save, -ps  : Save plots as PDF files.
                          Defaults to cfg -> plotting -> save.
    --plot-show / --no-plot-show, -psh : Show plots interactively.
                          Defaults to cfg -> plotting -> show.
    --plot-aggregate, -pa : When the input is a folder, merge trajectories from all videos
                          sharing the same location ID into one combined plot per location.
                          Defaults to cfg -> plotting -> aggregate.
    --plot-points, -pp  : Plot discrete trajectory points instead of connected lines.
                          Defaults to cfg -> plotting -> plot_points.
    --plot-segmentations, -pseg : Produce an additional trajectory plot overlaid on the lane
                          segmentation overlay PNG (pre-generate with tools/viz_segmentations.py).
                          Defaults to cfg -> plotting -> use_segmentations.
    --plot-class-filter, -pcf <int> [<int> ...] : Vehicle class IDs to exclude from plots
                          (e.g., -pcf 1 2 excludes buses and trucks).
                          Defaults to cfg -> plotting -> class_filter.

Examples:
  1. Full pipeline on a single video — detect, track, stabilize, georeference, save annotated video with lane
     IDs, and generate plots:
        geotrax batch data/video.mp4 --show-lanes

  2. Pixel-coordinate only (no georeferencing) — detect, track, stabilize, save annotated
     video with class names and confidence scores, and generate plots:
        geotrax batch data/U_video_cut.mp4 --no-geo --show-class-names --show-conf

  3. Dry-run on a directory to preview what would be processed without executing anything:
        geotrax batch path/to/PROCESSED/ --dry-run --verbose

  4. Re-run visualization only (existing .txt results) — save with lane IDs, stabilized view:
        geotrax batch data/video.mp4 --viz-only --save --show-lanes --viz-mode 1

  5. Re-run georeferencing only with a forced homography recompute:
        geotrax batch data/video.mp4 --geo-only -of --recompute

  6. Batch-process a directory, overwrite all existing results without prompts, save videos:
        geotrax batch path/to/PROCESSED/ --overwrite --yes

  7. Generate aggregated trajectory plots only from existing results, excluding buses and trucks:
        geotrax batch path/to/PROCESSED/ --plot-only --plot-aggregate \
            --plot-class-filter 1 2

  8. Use a lenient detection preset and skip videos matching 'test' or 'tmp' in their name:
        geotrax batch path/to/videos/ -c geotrax/cfg/lenient.yaml \
            --exclude-patterns test tmp --no-geo

Notes:
  - Skip-if-exists: each stage silently skips if its output file already exists. Use
    --overwrite (with optional --yes) to force re-execution of any stage.
  - Stage dependencies: georeferencing and visualization both require .txt tracking results.
    If these are missing, the dependent stage is skipped with an error message.
  - --cut-frame-right is reset to null in batch-directory mode; all videos are always
    processed through to their last frame.
  - --dry-run does not create the log file; use it freely before committing to a long run.
  - For full detection, tracking, and stabilization control (model, IoU, image size, tracker
    algorithm, stabilizer settings), edit the linked sub-configs in geotrax/cfg/.
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from geotrax.extract import add_processing_args, detect_track_stabilize
from geotrax.georeference import add_georeferencing_args, georeference
from geotrax.plot import add_plotting_args, default_plot_args, generate_plots
from geotrax.utils.cli_utils import add_common_args
from geotrax.utils.config_utils import backfill_args_from_config, load_config
from geotrax.utils.constants import VIDEO_FORMATS
from geotrax.utils.file_utils import check_if_results_exist, determine_suffix_and_fourcc
from geotrax.utils.logging_utils import bcolors, setup_logger
from geotrax.visualize import add_visualization_args, resolve_viz_modes, visualize_results

# Stage labels: used both as display text and as the dispatch key in should_process_file
ACTION_EXTRACT = "Detecting, tracking, and stabilizing"
ACTION_GEOREF = "Georeferencing"
ACTION_VISUALIZE = "Visualizing"


def process_input(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process the input file or directory.
    """
    input_path = args.input
    if not input_path.exists():
        logger.critical(f"File or directory '{input_path}' not found.")
        return

    batch_cfg = load_config(args.cfg, logger)['batch']
    backfill_args_from_config(args, {
        'folders_exclude': batch_cfg['folders_exclude'],
        'exclude_patterns': batch_cfg['exclude_patterns'],
    })

    try:
        if input_path.is_file() and input_path.suffix.lower() in VIDEO_FORMATS:
            process_file(input_path, args, logger)
        elif input_path.is_dir():
            logger.notice(f"{bcolors.OKGREEN}Batch processing all videos in: '{input_path}'{bcolors.ENDC}")
            args.cut_frame_right = None
            potential_files_to_process = [file for file in input_path.rglob('*') if file.is_file() and file.suffix.lower() in VIDEO_FORMATS]
            files_to_process = filter_files_to_process(potential_files_to_process, args, logger)
            files_to_process = sorted(files_to_process)

            pbar = tqdm(files_to_process, unit="video")
            for file in files_to_process:
                pbar.set_description(f"Processing: '{file}'")
                process_file(file, args, logger)
                pbar.update(1)
    except KeyboardInterrupt:
        logger.error("Batch processing interrupted by user.")
        return

    if (args.plot_save is not False or args.plot_show is not False) and not args.viz_only and not args.geo_only and input_path.is_dir():
        run_plotting(input_path, args, logger)


def run_plotting(path: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the plotting function for the given path.
    """
    logger.info(f"Generating plots for: '{path}'")
    if not args.dry_run:
        plot_args = default_plot_args(
            input=path,
            save=args.plot_save,
            show=args.plot_show,
            cfg=args.cfg,
            log_path=args.log_path,
            verbose=args.verbose,
            aggregate=args.plot_aggregate,
            ortho_folder=args.ortho_folder,
            segmentation_folder=args.segmentation_folder,
            segmentations=args.plot_segmentations,
            points=args.plot_points,
            class_filter=args.plot_class_filter,
        )
        generate_plots(plot_args, logger)


def process_file(file: Path, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process the file if it is a video file and not in the results directory.
    """
    try:
        logger.info(f"Processing: '{file}'")
        if not args.viz_only and not args.geo_only and not args.plot_only:
            process_step(file, args, logger, ACTION_EXTRACT, detect_track_stabilize)

        if not args.viz_only and not args.no_geo and not args.plot_only:
            process_step(file, args, logger, ACTION_GEOREF, georeference)

        if (args.save is not False or args.show is not False) and not args.plot_only:
            process_step(file, args, logger, ACTION_VISUALIZE, visualize_results)

        if (args.plot_save is not False or args.plot_show is not False) and not args.viz_only and not args.geo_only and not args.input.is_dir():
            run_plotting(file, args, logger)

    except Exception as e:
        logger.error(f"Error with {file}: {e}")


def process_step(file: Path, args: argparse.Namespace, logger: logging.Logger, action: str, func) -> None:
    """
    Process a specific step (processing, georeferencing, visualizing) for the file.
    """
    if should_process_file(file, args, logger, action):
        logger.info(f"{action}: '{file}'")
        if not args.dry_run:
            args.source = file
            func(args, logger)


def filter_files_to_process(files: list, args: argparse.Namespace, logger: logging.Logger) -> list:
    """
    Filter files based on exclusion criteria (folders and patterns).
    """
    filtered_files = []
    for file in files:
        if file.parent.name in args.folders_exclude:
            logger.info(f"Skipping '{file}' as it's in an excluded folder.")
            continue

        if args.exclude_patterns and any(pattern in file.name for pattern in args.exclude_patterns):
            logger.info(f"Skipping '{file}' due to matching exclusion pattern.")
            continue

        filtered_files.append(file)

    return filtered_files


def should_process_file(file: Path, args: argparse.Namespace, logger: logging.Logger, action: str) -> bool:
    """
    Determine if the video should be processed, georeferenced, or visualized based on the existing results and user input.
    """
    txt_exists = check_if_results_exist(file, "processed")[0]
    processing_steps = "detection, tracking, and stabilization"

    if action == ACTION_EXTRACT:
        return handle_existing_results(file, args, logger, txt_exists, processing_steps)
    elif action == ACTION_GEOREF:
        if not txt_exists:
            logger.error(f"'{file}' - No {processing_steps} results found. Skipping georeferencing.")
            return False
        csv_exists = check_if_results_exist(file, "georeferenced")[0]
        return handle_existing_results(file, args, logger, csv_exists, action)
    elif action == ACTION_VISUALIZE:
        if not txt_exists:
            logger.error(f"'{file}' - No {processing_steps} results found. Skipping visualization.")
            return False
        # Results count as present only if every mode the visualizer will render exists
        suffix = determine_suffix_and_fourcc()[0]
        viz_modes = resolve_viz_modes(args, logger)
        vid_exists = all(check_if_results_exist(file, "visualized", mode, suffix)[0] for mode in viz_modes)
        return handle_existing_results(file, args, logger, vid_exists, action)
    return False


def handle_existing_results(file: Path, args: argparse.Namespace, logger: logging.Logger, exists: bool, action: str) -> bool:
    """
    Handle existing results based on user input and overwrite options.
    """
    if exists and not args.overwrite:
        logger.warning(f"'{file}' - {action} results already exist and overwrite not allowed.")
        return False
    elif exists and args.overwrite and not args.yes:
        user_input = input(f"{bcolors.BOLD}Overwrite {action} results for: '{file}'? [y/n]: {bcolors.ENDC}").lower()
        return user_input == 'y'
    return True


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Primary entry point for the full Geo-trax pipeline. Runs detection/tracking/stabilization, georeferencing, visualization, and plotting for a single video file or an entire directory tree. Stages are skipped if their output already exists; use --overwrite to force re-execution.')

    parser.add_argument('input', type=Path, help='Path to a video file or a directory of video files (searched recursively).')

    batch = parser.add_argument_group('Batch processing options')
    batch.add_argument('--yes', '-y', action='store_true', help='Automatically confirm prompts without waiting for user input')
    batch.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing files that have already been processed')
    batch.add_argument('--dry-run', '-dr', action='store_true', help='Preview which files and stages would be processed without executing anything. Add --verbose for full per-stage detail.')
    batch.add_argument('--viz-only', '-vo', action='store_true', help='Skip detection, tracking, stabilization, and georeferencing; only (re-)run visualization. Requires existing .txt tracking results.')
    batch.add_argument('--geo-only', '-go', action='store_true', help='Only run georeferencing; skip detection, tracking, and stabilization')
    batch.add_argument('--plot-only', '-po', action='store_true', help='Only generate plots; skip processing, georeferencing, and visualization')
    batch.add_argument('--no-geo', '-ng', action='store_true', help='Do not georeference the tracking data')
    batch.add_argument("--folders-exclude", "-fe", type=str, nargs='+', default=None, help="Folders to exclude from the batch processing. Defaults to cfg -> batch -> folders_exclude.")
    batch.add_argument("--exclude-patterns", "-ep", type=str, nargs='+', default=None, help="File name patterns to exclude (e.g., --exclude-patterns car_test drone_2023). Defaults to cfg -> batch -> exclude_patterns.")

    shared = parser.add_argument_group('Shared options')
    add_common_args(shared)

    processing = parser.add_argument_group('Processing options',
        'For full detection and tracking control (model, IoU, image size, tracker settings, etc.), '
        'edit geotrax/cfg/ultralytics/default.yaml and the linked tracker config.')
    add_processing_args(processing)

    georef = parser.add_argument_group('Georeferencing options')
    add_georeferencing_args(georef)

    viz = parser.add_argument_group('Visualization options')
    add_visualization_args(viz, include_frame_range=False)  # cut-frame args come from the processing group above

    plotting = parser.add_argument_group('Plotting options')
    add_plotting_args(plotting, dest_prefix='plot_')  # plot_* dests so they don't clash with the visualization args

    return parser.parse_args()


def main() -> None:
    """
    Main function to process the input file or directory.
    """
    args = parse_cli_args()
    logger = setup_logger(__name__, args.verbose, args.log_path, args.dry_run)

    process_input(args, logger)


if __name__ == "__main__":
    main()
