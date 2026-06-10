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
  python batch_process.py <input_path> [options]

Arguments:
  input_path : Path to a video file (.mp4, .mov, .avi, .mkv) or a directory that is searched
               recursively for video files.

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
                          videos (default: ['results']).
    --exclude-patterns, -ep <str> [<str> ...] : Skip videos whose filename contains any of
                          these substrings (e.g., --exclude-patterns test temp).

Shared Options:
    --cfg, -c <path>    : Path to the main geo-trax configuration file (default: cfg/default.yaml).
    --log-file, -lf <str> : Filename to save detailed logs. Saved in the 'logs' folder (default: None).
    --verbose, -v       : Set print verbosity level to INFO (default: WARNING).

Processing Options:
    --conf, -co <float>   : Detection confidence threshold. Defaults to cfg -> cfg_ultralytics -> conf.
    --classes, -cls <int> [<int> ...] : Vehicle class IDs to extract (e.g., --classes 0 1 2).
                          Defaults to cfg -> cfg_ultralytics -> classes.
    --cut-frame-left, -cfl <int> : Skip the first N frames. Defaults to cfg -> processing -> cut_frame_left.
    --cut-frame-right, -cfr <int> : Stop at this frame number. Applies to single-file input only;
                          silently ignored in batch-directory mode. Defaults to cfg -> processing -> cut_frame_right.
    For full detection and tracking control (model, IoU, image size, tracker settings, etc.),
    edit cfg/ultralytics/default.yaml and the linked tracker config.

Georeferencing Options:
    --ortho-folder, -of <path>     : Path to the folder with orthophotos (.png, .tif, .txt).
                          Defaults to 'ORTHOPHOTOS' at the same level as 'PROCESSED' in input.
    --geo-source, -gs <choice>     : Georeferencing parameter source: metadata-tif, text-file, or
                          center-text-file. Auto-detected if omitted.
                          Defaults to cfg -> georef -> processing -> geo_source.
    --ref-frame, -rf <int>         : Reference frame number; must match the value used for
                          stabilization. Defaults to cfg -> georef -> processing -> ref_frame.
    --no-master, -nm               : Disable the master-frame approach regardless of config.
                          When not set, cfg -> georef -> processing -> use_master applies.
    --master-folder, -mf <path>    : Path to the folder containing master frame files (.png).
                          Defaults to '<ortho-folder>/master_frames'.
    --recompute, -r                : Force recomputation of the master→orthophoto homography
                          even if a cached result exists.
                          Defaults to cfg -> georef -> processing -> recompute.
    --segmentation-folder, -osf <path> : Path to the folder with road segmentation CSV files
                          (used for lane assignment during georeferencing) and, when
                          --segmentations is enabled, the corresponding overlay PNG files used
                          as plot backgrounds. Defaults to '<ortho-folder>/segmentations'.

Visualization Options:
    --save / --no-save, -s  : Save the annotated output video to file.
                          Defaults to cfg -> visualization -> save.
    --show / --no-show, -sh : Open a live preview window during processing.
                          Defaults to cfg -> visualization -> show.
    --viz-mode, -vm <int>   : Annotation source frame: 0=original, 1=stabilized, 2=reference
                          frame. Defaults to cfg -> visualization -> viz_mode.
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
    --class-filter, -cf <int> [<int> ...] : Vehicle class IDs to exclude from visualization
                          (e.g., -cf 1 2 hides buses and trucks).
                          Defaults to cfg -> visualization -> class_filter.

Plotting Options:
    --plot-save / --no-plot-save, -ps  : Save plots as PDF files.
                          Defaults to cfg -> plotting -> save.
    --plot-show / --no-plot-show, -psh : Show plots interactively.
                          Defaults to cfg -> plotting -> show.
    --aggregate, -a     : When the input is a folder, merge trajectories from all videos
                          sharing the same location ID into one combined plot per location.
                          Defaults to cfg -> plotting -> aggregate.
    --points, -p        : Plot discrete trajectory points instead of connected lines.
                          Defaults to cfg -> plotting -> plot_points.
    --segmentations, -seg : Produce an additional trajectory plot overlaid on the lane
                          segmentation overlay PNG (pre-generate with tools/viz_segmentations.py).
                          Defaults to cfg -> plotting -> use_segmentations.
    --plot-class-filter, -pcf <int> [<int> ...] : Vehicle class IDs to exclude from plots
                          (e.g., -pcf 1 2 excludes buses and trucks).
                          Defaults to cfg -> plotting -> class_filter.

Examples:
  1. Full pipeline on a single video — detect, track, stabilize, georeference, save annotated video with lane
     IDs, and generate plots:
        python batch_process.py data/video.mp4 --show-lanes

  2. Pixel-coordinate only (no georeferencing) — detect, track, stabilize, save annotated
     video with class names and confidence scores, and generate plots:
        python batch_process.py data/U_video_cut.mp4 --no-geo --show-class-names --show-conf

  3. Dry-run on a directory to preview what would be processed without executing anything:
        python batch_process.py path/to/PROCESSED/ --dry-run --verbose

  4. Re-run visualization only (existing .txt results) — save with lane IDs, stabilized view:
        python batch_process.py data/video.mp4 --viz-only --save --show-lanes --viz-mode 1

  5. Re-run georeferencing only with a forced homography recompute:
        python batch_process.py data/video.mp4 --geo-only -of --recompute

  6. Batch-process a directory, overwrite all existing results without prompts, save videos:
        python batch_process.py path/to/PROCESSED/ --overwrite --yes

  7. Generate aggregated trajectory plots only from existing results, excluding buses and trucks:
        python batch_process.py path/to/PROCESSED/ --plot-only --aggregate \
            --plot-class-filter 1 2

  8. Use a lenient detection preset and skip videos matching 'test' or 'tmp' in their name:
        python batch_process.py path/to/videos/ -c cfg/lenient.yaml \
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
    algorithm, stabilizer settings), edit the linked sub-configs in cfg/.
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from detect_track_stabilize import detect_track_stabilize
from georeference import georeference
from plot import generate_plots
from utils.file_utils import check_if_results_exist, determine_suffix_and_fourcc
from utils.logging_utils import bcolors, setup_logger
from visualize import visualize_results

VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv'}


def process_input(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Process the input file or directory.
    """
    input_path = args.input
    if not input_path.exists():
        logger.critical(f"File or directory '{input_path}' not found.")
        return

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
        plot_args = argparse.Namespace(
            input=path,
            save=args.plot_save,
            show=args.plot_show,
            cfg=args.cfg,
            log_file=args.log_file,
            verbose=args.verbose,
            aggregate=args.aggregate,
            ortho_folder=args.ortho_folder,
            segmentation_folder=args.segmentation_folder,
            segmentations=args.segmentations,
            id=0,
            points=args.points,
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
            process_step(file, args, logger, "Detecting, tracking, and stabilizing", detect_track_stabilize)

        if not args.viz_only and not args.no_geo and not args.plot_only:
            process_step(file, args, logger, "Georeferencing", georeference)

        if (args.save is not False or args.show is not False) and not args.plot_only:
            process_step(file, args, logger, "Visualizing", visualize_results)

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
    suffix = determine_suffix_and_fourcc()[0]
    txt_exists = check_if_results_exist(file, "processed")[0]
    csv_exists = check_if_results_exist(file, "georeferenced")[0]
    vid_exists = check_if_results_exist(file, "visualized", args.viz_mode if args.viz_mode is not None else 0, suffix)[0]

    processing_steps = "detection, tracking, and stabilization"
    if action == "Detecting, tracking, and stabilizing":
        return handle_existing_results(file, args, logger, txt_exists, processing_steps)
    elif action == "Georeferencing":
        if not txt_exists:
            logger.error(f"'{file}' - No {processing_steps} results found. Skipping georeferencing.")
            return False
        return handle_existing_results(file, args, logger, csv_exists, action)
    elif action == "Visualizing":
        if not txt_exists:
            logger.error(f"'{file}' - No {processing_steps} results found. Skipping visualization.")
            return False
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

    parser.add_argument('input', type=Path, help='Path to the input directory or video file (e.g., path/to/video_dir/)')

    batch = parser.add_argument_group('Batch processing options')
    batch.add_argument('--yes', '-y', action='store_true', help='Automatically confirm prompts without waiting for user input')
    batch.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing files that have already been processed')
    batch.add_argument('--dry-run', '-dr', action='store_true', help='Preview which files and stages would be processed without executing anything. Add --verbose for full per-stage detail.')
    batch.add_argument('--viz-only', '-vo', action='store_true', help='Skip detection, tracking, stabilization, and georeferencing; only (re-)run visualization. Requires existing .txt tracking results.')
    batch.add_argument('--geo-only', '-go', action='store_true', help='Only run georeferencing; skip detection, tracking, and stabilization')
    batch.add_argument('--plot-only', '-po', action='store_true', help='Only generate plots; skip processing, georeferencing, and visualization')
    batch.add_argument('--no-geo', '-ng', action='store_true', help='Do not georeference the tracking data')
    batch.add_argument("--folders-exclude", "-fe", type=str, nargs='+', default=['results'], help="Folders to exclude from the batch processing")
    batch.add_argument("--exclude-patterns", "-ep", type=str, nargs='+', default=None, help="File name patterns to exclude (e.g., --exclude-patterns car_test drone_2023)")

    shared = parser.add_argument_group('Shared options')
    shared.add_argument('--cfg', '-c', type=Path, default='cfg/default.yaml', help='Path to the main geo-trax configuration file')
    shared.add_argument('--log-file', '-lf', type=str, default=None, help="Filename to save detailed logs. Saved in the 'logs' folder.")
    shared.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING)')

    processing = parser.add_argument_group('Processing options',
        'For full detection and tracking control (model, IoU, image size, tracker settings, etc.), '
        'edit cfg/ultralytics/default.yaml and the linked tracker config.')
    processing.add_argument('--conf', '-co', type=float, default=None, help='Detection confidence threshold. Defaults to cfg -> cfg_ultralytics -> conf.')
    processing.add_argument('--classes', '-cls', nargs='+', type=int, default=None, help='Class IDs to extract (e.g., --classes 0 1 2). Defaults to cfg -> cfg_ultralytics -> classes.')
    processing.add_argument('--cut-frame-left', '-cfl', type=int, default=None, help='Skip the first N frames. Defaults to cfg -> processing -> cut_frame_left.')
    processing.add_argument('--cut-frame-right', '-cfr', type=int, default=None, help='Stop processing after this frame. Only considered if input is a single video file. Defaults to cfg -> processing -> cut_frame_right.')

    georef = parser.add_argument_group('Georeferencing options')
    georef.add_argument("--ortho-folder", "-of", type=Path, default=None, help="Custom path to the folder with orthophotos (.png, .tif, .txt). Defaults to 'ORTHOPHOTOS' at the same level as 'PROCESSED' in 'input'.")
    georef.add_argument("--geo-source", "-gs", choices=['metadata-tif', 'text-file', 'center-text-file'], default=None, help="Source of georeferencing parameters. Defaults to cfg -> georef -> processing -> geo_source.")
    georef.add_argument("--ref-frame", "-rf", type=int, default=None, help="Reference frame number (must match stabilization setting). Defaults to cfg -> georef -> processing -> ref_frame.")
    georef.add_argument("--no-master", "-nm", action="store_const", const=True, default=None, help="Disable the master frame approach regardless of config. When not set, cfg -> georef -> processing -> use_master applies.")
    georef.add_argument("--master-folder", "-mf", type=Path, default=None, help="Custom path to the folder containing master frame files (.png). If not provided, '--ortho-folder / master_frames' will be used.")
    georef.add_argument("--recompute", "-r", action="store_const", const=True, default=None, help="Force recompute master->ortho homography even if cached. Defaults to cfg -> georef -> processing -> recompute.")
    georef.add_argument("--segmentation-folder", "-osf", type=Path, default=None, help="Path to the folder containing lane segmentation CSV files (used for lane assignment during georeferencing) and, when --segmentations is enabled, the corresponding overlay PNG files used as plot backgrounds. Defaults to '<ortho-folder>/segmentations'.")

    viz = parser.add_argument_group('Visualization options')
    viz.add_argument('--save', '-s', action=argparse.BooleanOptionalAction, default=None, help='Save the annotated output video to file. Defaults to cfg -> visualization -> save.')
    viz.add_argument('--show', '-sh', action=argparse.BooleanOptionalAction, default=None, help='Open a live preview window during processing. Defaults to cfg -> visualization -> show.')
    viz.add_argument('--viz-mode', '-vm', type=int, default=None, choices=[0, 1, 2], help='Frame source for annotation: 0=original, 1=stabilized, 2=reference frame. Defaults to cfg -> visualization -> viz_mode.')
    viz.add_argument("--plot-trajectories", "-pt", action=argparse.BooleanOptionalAction, default=None, help='Overlay trajectory positions on the first frame. Defaults to cfg -> visualization -> plot_trajectories.')
    viz.add_argument("--plot-delay", "-pd", type=int, default=None, help='Number of frames to display the trajectory overlay; only relevant when --plot-trajectories is enabled. Defaults to cfg -> visualization -> plot_delay.')
    viz.add_argument("--show-conf", "-sc", action=argparse.BooleanOptionalAction, default=None, help='Include detection confidence in labels. Defaults to cfg -> visualization -> show_conf.')
    viz.add_argument("--show-lanes", "-sl", action=argparse.BooleanOptionalAction, default=None, help='Include lane ID in labels. Defaults to cfg -> visualization -> show_lanes.')
    viz.add_argument("--show-class-names", "-scn", action=argparse.BooleanOptionalAction, default=None, help='Include class name in labels. Defaults to cfg -> visualization -> show_class_names.')
    viz.add_argument("--hide-labels", "-hl", action=argparse.BooleanOptionalAction, default=None, help='Suppress all label text. Defaults to cfg -> visualization -> hide_labels.')
    viz.add_argument("--hide-tracks", "-ht", action=argparse.BooleanOptionalAction, default=None, help='Suppress track tail lines. Defaults to cfg -> visualization -> hide_tracks.')
    viz.add_argument("--hide-speed", "-hs", action=argparse.BooleanOptionalAction, default=None, help='Suppress speed values in labels. Defaults to cfg -> visualization -> hide_speed.')
    viz.add_argument('--speed-unit', '-su', type=str, default=None, choices=['km/h', 'mi/h'], help='Speed display unit: km/h or mi/h. Defaults to cfg -> visualization -> speed_unit.')
    viz.add_argument('--class-filter', '-cf', type=int, nargs='+', default=None, help='Class IDs to exclude from visualization (e.g., -cf 1 2). Defaults to cfg -> visualization -> class_filter.')

    plotting = parser.add_argument_group('Plotting options')
    plotting.add_argument('--plot-save', '-ps', action=argparse.BooleanOptionalAction, default=None, help='Save plots as PDF files. Defaults to cfg -> plotting -> save.')
    plotting.add_argument('--plot-show', '-psh', action=argparse.BooleanOptionalAction, default=None, help='Show plots in an interactive window. Defaults to cfg -> plotting -> show.')
    plotting.add_argument('--aggregate', '-a', action=argparse.BooleanOptionalAction, default=None, help='When the input is a folder, merge trajectories from all videos sharing the same location ID into a single plot per location. Defaults to cfg -> plotting -> aggregate.')
    plotting.add_argument('--points', '-p', action=argparse.BooleanOptionalAction, default=None, help='Plot trajectory points instead of lines. Defaults to cfg -> plotting -> plot_points.')
    plotting.add_argument('--segmentations', '-seg', action=argparse.BooleanOptionalAction, default=None, help='Produce an additional trajectory plot overlaid on the lane segmentation overlay PNG (from --segmentation-folder), alongside the standard plain-orthophoto plot. Requires pre-generated overlays (run: python tools/viz_segmentations.py <ortho_folder>/). Defaults to cfg -> plotting -> use_segmentations.')
    plotting.add_argument('--plot-class-filter', '-pcf', type=int, nargs='+', default=None, help='Class IDs to exclude from plots (e.g., -pcf 1 2). Defaults to cfg -> plotting -> class_filter.')

    return parser.parse_args()


def main() -> None:
    """
    Main function to process the input file or directory.
    """
    args = parse_cli_args()
    logger = setup_logger(Path(__file__).name, args.verbose, args.log_file, args.dry_run)

    process_input(args, logger)


if __name__ == "__main__":
    main()
