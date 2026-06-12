#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
viz_dimension_estimation.py - Vehicle Dimension Estimation Visualizer

Visualizes the step-by-step process of the azimuth-based vehicle dimension
estimation algorithm implemented in the geo-trax pipeline (geotrax/extract.py).
For a selected vehicle ID, it renders two diagnostic plots:
  1. The vehicle trajectory overlaid with bounding boxes, colour-coded by their
     acceptance or rejection at each filtering stage.
  2. The distribution of candidate bounding boxes alongside the final dimension
     estimate (25th-percentile length and width).

This tool was developed in the context of the Songdo (Korea) intersection experiment
(DOI: 10.1016/j.trc.2025.105205) to validate and compare the azimuth-based estimator
against the naive bounding-box baseline (see compute_naive_bb_dimensions.py).

Usage:
  python tools/viz_dimension_estimation.py <source> [options]

Arguments:
  source : Path to the video file. Tracking results must exist in a 'results/'
           subfolder next to the video (produced by 'geotrax extract').

Options:
  -h, --help        : Show this help message and exit.
  --id, -i <int>    : Vehicle ID to visualize. Prompted interactively if omitted or 0 (default: 0).
  --show            : Display plots interactively (default: False).
  --save, -s        : Save plots as PDF files to <video_dir>/results/plots/ (default: False).

Examples:
1. Visualize vehicle ID 42 and show plots interactively:
   python tools/viz_dimension_estimation.py path/to/video.mp4 --id 42 --show

2. Visualize vehicle ID 42 and save plots to PDF:
   python tools/viz_dimension_estimation.py path/to/video.mp4 --id 42 --save

3. Prompt for vehicle ID and show:
   python tools/viz_dimension_estimation.py path/to/video.mp4 --show

Input:
- Video file (any format supported by OpenCV)
- Tracking result file: results/<video_stem>.txt  (comma-delimited, produced by
  'geotrax extract'; columns: frame, id, x, y, w, h [, x_stab, y_stab,
  w_stab, h_stab, conf, class, length, width])

Output:
- 'trajectory_with_dimensions': vehicle trajectory with bounding boxes colour-coded
  by filtering stage (red = rejected, green = accepted, bright = azimuth keyframe)
- 'dimensions_distribution': all accepted candidate boxes and the final estimate (blue)

Notes:
- GSD and tau_c constants below are tuned to the Songdo experiment (4K drone footage
  at 140–150 m altitude). Adjust them for other datasets.
- The azimuth-based algorithm mirrors geotrax/extract.py exactly; any changes
  to the pipeline estimator should be reflected here for consistency.
- Stabilised coordinates (columns 6/7) are used when available; raw coordinates
  (columns 2/3) are used as fallback.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from geotrax.utils.file_utils import detect_delimiter

# Ground sampling distance constants for the Songdo experiment
# (4K resolution, 140–150 m flight altitude)
GSD_150 = 0.0282  # m/px at 150 m
GSD_140 = 0.0263  # m/px at 140 m
GSD = (GSD_150 + GSD_140) / 2

# Length-to-width ratio thresholds per vehicle class (used when azimuth is unavailable)
# Tuned for the Songdo experiment; adjust for other datasets.
TAU_C = {
    0: 2.0,   # car (incl. vans, SUVs)
    1: 2.5,   # bus
    2: 2.1,   # truck
    3: 1.9,   # motorcycle
    -1: 1.7,  # unknown / fallback
}


def visualize_dimension_estimation(args: argparse.Namespace) -> None:
    tracks = load_tracks(args)
    args = resolve_vehicle_id(tracks, args)
    visualize_id(tracks, args)


def load_tracks(args: argparse.Namespace) -> np.ndarray:
    tracks_file = args.source.parent / 'results' / f'{args.source.stem}.txt'
    if not tracks_file.exists():
        print(f"\033[91mTracking results not found: '{tracks_file}'.\033[0m")
        print("Run 'geotrax extract' on the video first.")
        sys.exit(1)
    delimiter = detect_delimiter(tracks_file)
    tracks = np.loadtxt(tracks_file, delimiter=delimiter)
    if tracks.ndim == 1:
        tracks = tracks.reshape(1, -1)
    return tracks


def resolve_vehicle_id(tracks: np.ndarray, args: argparse.Namespace) -> argparse.Namespace:
    unique_ids = np.unique(tracks[:, 1]).astype(int)
    vehicle_id = args.id
    if vehicle_id == 0:
        while vehicle_id not in unique_ids:
            try:
                vehicle_id = int(input(f"Enter a vehicle ID to visualize {unique_ids.tolist()}: "))
            except ValueError:
                pass
            if vehicle_id not in unique_ids:
                print(f"ID {vehicle_id} not found in the tracks. Available IDs: {unique_ids.tolist()}")
        args.id = vehicle_id
    elif vehicle_id not in unique_ids:
        print(f"\033[91mID {vehicle_id} not found in the tracks. Available IDs: {unique_ids.tolist()}\033[0m")
        sys.exit(1)
    return args


def plot_trajectory(tracks: np.ndarray, idx_x: int, idx_y: int) -> None:
    plt.figure()
    for vehicle_id in np.unique(tracks[:, 1]):
        mask = tracks[:, 1] == vehicle_id
        plt.plot(tracks[mask, idx_x], tracks[mask, idx_y], 'k-', alpha=0.9, linewidth=1)


def plot_boxes(tracks: np.ndarray, idx_x: int, idx_y: int, color: str = 'r', lw: float = 0.5, alpha: float = 0.1) -> None:
    for track in tracks:
        x_c, y_c = track[idx_x], track[idx_y]
        w, h = track[4], track[5]
        plt.plot(
            [x_c - w/2, x_c + w/2, x_c + w/2, x_c - w/2, x_c - w/2],
            [y_c - h/2, y_c - h/2, y_c + h/2, y_c + h/2, y_c - h/2],
            linewidth=lw, alpha=alpha, color=color,
        )


def plot_dimensions(
    tracks: np.ndarray,
    id2lengths: dict, id2widths: dict,
    id2length: dict, id2width: dict,
    args: argparse.Namespace,
    idx_x: int, idx_y: int,
) -> None:
    plt.figure()
    zoom_factor = 5
    x_center = 1.55 * np.max(tracks[:, idx_x])
    y_center = (np.max(tracks[:, idx_y]) + np.min(tracks[:, idx_y])) / 2

    lengths = id2lengths[args.id] * zoom_factor
    widths  = id2widths[args.id]  * zoom_factor
    length_est = id2length[args.id] * zoom_factor
    width_est  = id2width[args.id]  * zoom_factor

    for l, w in zip(lengths, widths):
        plt.plot(
            [x_center - l/2, x_center + l/2, x_center + l/2, x_center - l/2, x_center - l/2],
            [y_center - w/2, y_center - w/2, y_center + w/2, y_center + w/2, y_center - w/2],
            linewidth=0.5, alpha=0.1, color='g',
        )

    plt.plot(
        [x_center - length_est/2, x_center + length_est/2, x_center + length_est/2, x_center - length_est/2, x_center - length_est/2],
        [y_center - width_est/2,  y_center - width_est/2,  y_center + width_est/2,  y_center + width_est/2,  y_center - width_est/2],
        linewidth=1.5, alpha=0.9, color='b', linestyle='dashed',
    )


def visualize_id(
    tracks: np.ndarray,
    args: argparse.Namespace,
    eps: int = 4,
    r0: float = 1.25,
    gsd: float = GSD,
    theta_bar_deg: float = 15.0,
    tau_c: dict = TAU_C,
) -> None:
    radius_threshold = r0 / gsd
    theta_bar = np.deg2rad(theta_bar_deg)

    # Get video frame dimensions
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        print(f"\033[91mCould not open video: '{args.source}'.\033[0m")
        sys.exit(1)
    w_I = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_I = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Step 0: isolate the chosen vehicle's tracks
    valid_tracks = tracks[tracks[:, 1] == args.id]
    if len(valid_tracks) == 0:
        print(f"\033[91mNo tracks found for ID {args.id}.\033[0m")
        sys.exit(1)

    # Choose stabilised or raw coordinate columns
    if valid_tracks.shape[1] > 8:
        idx_x, idx_y, idx_c = 6, 7, 10   # stabilised coordinates available
    else:
        idx_x, idx_y, idx_c = 2, 3, 6    # fall back to raw coordinates

    # Step 1: visibility filtering (always applied on raw bounding box columns)
    mask = (
        (valid_tracks[:, 2] - valid_tracks[:, 4] / 2 > eps) &
        (valid_tracks[:, 3] - valid_tracks[:, 5] / 2 > eps) &
        (valid_tracks[:, 2] + valid_tracks[:, 4] / 2 < w_I - 1 - eps) &
        (valid_tracks[:, 3] + valid_tracks[:, 5] / 2 < h_I - 1 - eps)
    )

    plot_trajectory(valid_tracks, idx_x, idx_y)
    plot_boxes(valid_tracks[~mask], idx_x, idx_y, color='r', alpha=0.3)   # outside-frame boxes
    valid_tracks = valid_tracks[mask]

    # Step 2: collect per-id bounding box sides and trajectory centres
    unique_ids = np.unique(valid_tracks[:, 1]).astype(int)
    id2lengths   = {vid: [] for vid in unique_ids}
    id2widths    = {vid: [] for vid in unique_ids}
    id2x_centers = {vid: [] for vid in unique_ids}
    id2y_centers = {vid: [] for vid in unique_ids}
    id2class     = {}

    for track in valid_tracks:
        vid = int(track[1])
        w, h = track[4], track[5]
        id2lengths[vid].append(max(w, h))
        id2widths[vid].append(min(w, h))
        id2x_centers[vid].append(track[idx_x])
        id2y_centers[vid].append(track[idx_y])
        id2class.setdefault(vid, int(track[idx_c]))

    # Step 3: azimuth-based filtering
    for vid in unique_ids:
        x_centers = id2x_centers[vid]
        y_centers = id2y_centers[vid]
        lengths   = np.array(id2lengths[vid])
        widths    = np.array(id2widths[vid])

        azimuth  = None
        idx_prev = 0
        x_c_prev, y_c_prev = x_centers[idx_prev], y_centers[idx_prev]
        mask_accept  = np.zeros(len(lengths), dtype=bool)
        mask_keyframe = mask_accept.copy()
        mask_keyframe[0] = True

        for idx, (x_c, y_c) in enumerate(zip(x_centers[1:], y_centers[1:]), start=1):
            distance = np.hypot(x_c - x_c_prev, y_c - y_c_prev)
            if distance >= radius_threshold:
                mask_keyframe[idx - 1] = True
                azimuth = np.arctan2(-(y_c - y_c_prev), x_c - x_c_prev)
                x_c_prev, y_c_prev = x_c, y_c
                cardinal_diffs = np.abs(azimuth - np.array([0, np.pi/2, np.pi, -np.pi/2, -np.pi]))
                if np.any(cardinal_diffs <= theta_bar):
                    mask_accept[idx_prev:idx] = True
                idx_prev = idx

        if azimuth is None:
            # No sufficient displacement: fall back to length/width ratio threshold
            ratio_threshold = tau_c.get(id2class.get(vid, -1), tau_c[-1])
            mask_accept = lengths >= widths * ratio_threshold

        id2lengths[vid] = lengths[mask_accept]
        id2widths[vid]  = widths[mask_accept]

        # Colour-coded bounding boxes: red=rejected, green=accepted, brighter=keyframe
        plot_boxes(valid_tracks[~mask_accept & ~mask_keyframe], idx_x, idx_y, color='r', lw=0.5, alpha=0.1)
        plot_boxes(valid_tracks[ mask_accept & ~mask_keyframe], idx_x, idx_y, color='g', lw=0.5, alpha=0.1)
        plot_boxes(valid_tracks[~mask_accept &  mask_keyframe], idx_x, idx_y, color='r', lw=1.0, alpha=0.5)
        plot_boxes(valid_tracks[ mask_accept &  mask_keyframe], idx_x, idx_y, color='g', lw=1.0, alpha=0.5)

    # Step 4: final dimension estimate (25th percentile)
    id2length, id2width = {}, {}
    for vid in unique_ids:
        id2length[vid] = np.percentile(id2lengths[vid], 25) if len(id2lengths[vid]) > 0 else np.nan
        id2width[vid]  = np.percentile(id2widths[vid],  25) if len(id2widths[vid])  > 0 else np.nan

    print(f"\nID {int(args.id)} | Length: {id2length[args.id]:.2f} px | Width: {id2width[args.id]:.2f} px\n")

    save_or_show_plot(args, 'trajectory_with_dimensions')

    plot_dimensions(valid_tracks, id2lengths, id2widths, id2length, id2width, args, idx_x, idx_y)
    save_or_show_plot(args, 'dimensions_distribution')


def save_or_show_plot(args: argparse.Namespace, filename: str) -> None:
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    if args.save:
        img_dir = args.source.parent / 'results' / 'plots'
        img_dir.mkdir(parents=True, exist_ok=True)
        img_filepath = img_dir / f"{args.source.stem}_{filename}_ID-{args.id}.pdf"
        plt.savefig(str(img_filepath), bbox_inches='tight', pad_inches=0, transparent=False)
        print(f"Plot saved to '{img_filepath}'")
    if args.show:
        plt.show()
    plt.close()


def get_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the azimuth-based vehicle dimension estimation algorithm step-by-step."
    )
    parser.add_argument(
        'source', type=Path,
        help="Path to the video file. Tracking results must exist in results/<video_stem>.txt."
    )
    parser.add_argument(
        '--id', '-i', type=int, default=0,
        help="Vehicle ID to visualize. Prompted interactively if 0 (default: 0)."
    )
    parser.add_argument(
        '--show', action='store_true',
        help="Display plots interactively."
    )
    parser.add_argument(
        '--save', '-s', action='store_true',
        help="Save plots as PDF files to <video_dir>/results/plots/."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    if not args.show and not args.save:
        print("Warning: neither --show nor --save specified. Plots will not be displayed or saved.")
    visualize_dimension_estimation(args)
