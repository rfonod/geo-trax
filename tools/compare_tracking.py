#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Compare tracking results across two or more vehicle tracking algorithms.

This script analyzes and compares the performance of any number of tracking algorithms
by evaluating trajectory length distributions and missing frame patterns. It processes
tracking results from multiple videos, computes statistical metrics (including pairwise
KL divergence), and generates visualizations to highlight differences between algorithms.

By default it compares the six trackers geo-trax ships with (BoT-SORT, ByteTrack, OC-SORT,
Deep OC-SORT, FastTracker, TrackTrack), but any tracker name is accepted as long as a
matching results folder exists.

Usage:
    python tools/compare_tracking.py INPUT [--trackers NAME ...] [--show] [--save]

Arguments:
    INPUT                    Path to folder containing video files and per-tracker results

Options:
    --trackers NAME ...      Tracker names to compare (default: the six geo-trax trackers).
                             Each name must have a matching `results_<name>/` subfolder;
                             names without one are skipped. At least two must remain.
    --show                   Display the generated plots interactively
    --save                   Save plots to 'plots/' subdirectory as PNG files

Input Format:
    For each tracker NAME, results live in a sibling folder of the input videos:
    - results_<NAME>/{video_name}.txt   (e.g. results_botsort/, results_ocsort/)

    Each line contains 14 comma-separated values:
    frame_number,vehicle_id,x,y,width,height,x_stab,y_stab,width_stab,height_stab,class_id,confidence,vehicle_length,vehicle_width

Output:
    - Per-tracker statistics for trajectory lengths and missing frames
    - Pairwise KL divergence measures between distributions
    - Multi-panel visualization (violin plots, CDFs, histograms, and density comparison)
    - Optional PNG plot files saved to plots/ directory

Notes:
    - Videos starting with 'P' are automatically skipped
    - A video is only included when every selected tracker has a results file for it
    - Uses KDE-based density estimation for detailed distribution analysis
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add project root directory to Python path
from utils.file_utils import detect_delimiter

# Trackers geo-trax ships with, in display order. Used as the default comparison set.
TRACKER_DISPLAY_NAMES = {
    "botsort": "BoT-SORT",
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "deepocsort": "Deep OC-SORT",
    "fasttrack": "FastTracker",
    "tracktrack": "TrackTrack",
}
DEFAULT_TRACKERS = list(TRACKER_DISPLAY_NAMES)

# Stable, high-contrast color per known tracker; unknown names fall back to the palette below.
TRACKER_COLORS = {
    "botsort": "#3A6DAA",  # deep blue
    "bytetrack": "#E57200",  # deep orange
    "ocsort": "#2CA02C",  # green
    "deepocsort": "#9467BD",  # purple
    "fasttrack": "#D62728",  # red
    "tracktrack": "#17BECF",  # cyan
}
FALLBACK_COLORS = ["#8C564B", "#BCBD22", "#7F7F7F", "#E377C2", "#1F77B4", "#FF7F0E"]


def display_name(tracker: str) -> str:
    """Human-readable label for a tracker name."""
    return TRACKER_DISPLAY_NAMES.get(tracker, tracker)


def color_for(tracker: str, index: int) -> str:
    """Plot color for a tracker, deterministic for known names and cycled for the rest."""
    return TRACKER_COLORS.get(tracker, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def compare_tracks(args):
    """
    Compare tracking results across the selected trackers.
    """
    # Keep only trackers that actually have a results folder, preserving the requested order
    trackers = []
    for tracker in args.trackers:
        if (args.input / f"results_{tracker}").is_dir():
            trackers.append(tracker)
        else:
            print(f"Warning: no 'results_{tracker}/' folder found in {args.input}; skipping {display_name(tracker)}")
    if len(trackers) < 2:
        sys.exit("Need at least two trackers with available results to compare.")

    print(f"Comparing trackers: {', '.join(display_name(t) for t in trackers)}\n")

    # Accumulate per-tracker trajectory lengths and missing-frame counts across all videos
    lengths = {tracker: [] for tracker in trackers}
    missing = {tracker: [] for tracker in trackers}

    video_files = sorted(args.input.glob("*.MP4"))
    for video_file in video_files:
        if video_file.stem[0] == "P":
            continue

        result_paths = {t: args.input / f"results_{t}" / f"{video_file.stem}.txt" for t in trackers}
        absent = [display_name(t) for t, p in result_paths.items() if not p.exists()]
        if absent:
            print(f"Skipping {video_file.stem}: missing results for {', '.join(absent)}")
            continue

        print(f"Comparing tracking results for video: {video_file.stem}")
        for tracker, path in result_paths.items():
            tracks = np.loadtxt(path, delimiter=detect_delimiter(path), dtype=np.float64, ndmin=2)
            if tracks.size == 0:
                continue
            lengths[tracker].extend(compute_trajectory_lengths(tracks))
            missing[tracker].extend(find_missing_frames(tracks))

    if any(len(v) == 0 for v in lengths.values()):
        empty = [display_name(t) for t, v in lengths.items() if len(v) == 0]
        sys.exit(f"No usable tracking results found for: {', '.join(empty)}.")

    print_metric_analysis("Trajectory Length Analysis", lengths, trackers)
    print_metric_analysis("Missing Frames Analysis", missing, trackers)

    if args.show or args.save:
        plot_trajectory_length_distributions(lengths, trackers, args)


def print_metric_analysis(title, values_by_tracker, trackers):
    """
    Print per-tracker summary statistics and pairwise KL divergence for one metric.
    """
    print(f"\n{title}:\n")
    for tracker in trackers:
        values = values_by_tracker[tracker]
        print(f"{display_name(tracker)}: count={len(values)}, mean={np.mean(values):.2f}, std={np.std(values):.2f}")

    print("\nPairwise KL divergence:")
    for a, b in itertools.combinations(trackers, 2):
        kl_ab = compute_kl_divergence(values_by_tracker[a], values_by_tracker[b])
        kl_ba = compute_kl_divergence(values_by_tracker[b], values_by_tracker[a])
        kl_avg = (kl_ab + kl_ba) / 2
        print(
            f"  {display_name(a)} ↔ {display_name(b)}: "
            f"{display_name(a)}→{display_name(b)}={kl_ab:.4f}, "
            f"{display_name(b)}→{display_name(a)}={kl_ba:.4f}, avg={kl_avg:.4f}"
        )


def find_missing_frames(tracks):
    """
    Count the number of missing frames in each vehicle's trajectory
    """
    vehicle_ids = np.unique(tracks[:, 1])
    missing_frames = []
    for vehicle_id in vehicle_ids:
        vehicle_tracks = tracks[tracks[:, 1] == vehicle_id]
        frame_numbers = vehicle_tracks[:, 0]
        frame_difference = frame_numbers[-1] + 1 - frame_numbers[0]
        missing_frames.append(frame_difference - len(frame_numbers))

    return missing_frames


def compute_trajectory_lengths(tracks):
    """
    Compute the trajectory lengths for each vehicle in the tracks
    """
    vehicle_ids = np.unique(tracks[:, 1])
    trajectory_lengths = []
    for vehicle_id in vehicle_ids:
        vehicle_tracks = tracks[tracks[:, 1] == vehicle_id]
        trajectory_lengths.append(len(vehicle_tracks))

    return trajectory_lengths


def compute_kl_divergence(p, q, epsilon=1e-10):
    """
    Compute the KL divergence between two distributions p and q
    """

    # Create histograms with identical bins
    min_val = min(np.min(p), np.min(q))
    max_val = max(np.max(p), np.max(q))
    if max_val == min_val:
        return 0.0  # both distributions are a single shared value; no divergence
    bins = np.linspace(min_val, max_val, 50)

    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    # Add small epsilon to avoid division by zero
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon

    # Normalize
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)

    # Compute the KL divergence
    kl_divergence = np.sum(p_hist * np.log(p_hist / q_hist))

    return kl_divergence


def plot_trajectory_length_distributions(lengths_by_tracker, trackers, args):
    """
    Plot the trajectory length distributions for the selected trackers with enhanced
    visualization techniques to highlight subtle differences between similar distributions.

    Works for any number of trackers. When exactly two are compared, the bottom two panels
    use the richer mirrored-histogram and signed-density-difference views; for three or more
    they fall back to overlaid histograms and KDE curves.
    """
    colors = {t: color_for(t, i) for i, t in enumerate(trackers)}
    names = {t: display_name(t) for t in trackers}
    all_data = np.concatenate([lengths_by_tracker[t] for t in trackers])

    # Create a figure with a custom, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1.5, 1])

    # Top left: Violin plot across all trackers
    ax1 = fig.add_subplot(gs[0, 0])
    df = pd.DataFrame(
        [{"Algorithm": names[t], "Trajectory Length": v} for t in trackers for v in lengths_by_tracker[t]]
    )
    order = [names[t] for t in trackers]
    sns.violinplot(
        x='Algorithm',
        y='Trajectory Length',
        data=df,
        order=order,
        hue='Algorithm',
        palette={names[t]: colors[t] for t in trackers},
        legend=False,
        inner='quartile',
        cut=0,
        ax=ax1,
    )
    means = [np.mean(lengths_by_tracker[t]) for t in trackers]
    ax1.scatter(range(len(trackers)), means, color='white', s=30, zorder=3)
    ax1.scatter(range(len(trackers)), means, color='black', s=15, zorder=4)
    ax1.set_title("Distribution Comparison (Violin Plot)", fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', labelrotation=15 if len(trackers) > 3 else 0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Top right: Cumulative distribution function
    ax2 = fig.add_subplot(gs[0, 1])
    for tracker in trackers:
        sns.ecdfplot(lengths_by_tracker[tracker], label=names[tracker], color=colors[tracker], lw=2, ax=ax2)
    ax2.set_title("Cumulative Distribution Function", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Trajectory Length", fontsize=12)
    ax2.set_ylabel("Cumulative Probability", fontsize=12)
    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Middle: histogram comparison (mirrored for two trackers, overlaid otherwise)
    ax3 = fig.add_subplot(gs[1, :])
    bins = np.linspace(min(all_data), max(all_data), 40)
    width = bins[1] - bins[0]
    if len(trackers) == 2:
        a, b = trackers
        a_hist, _ = np.histogram(lengths_by_tracker[a], bins=bins)
        b_hist, _ = np.histogram(lengths_by_tracker[b], bins=bins)
        a_hist = a_hist / len(lengths_by_tracker[a]) * 100
        b_hist = -b_hist / len(lengths_by_tracker[b]) * 100  # Negative for bottom direction
        ax3.bar(bins[:-1], a_hist, width=width, color=colors[a], alpha=0.7, align='edge', label=names[a])
        ax3.bar(bins[:-1], b_hist, width=width, color=colors[b], alpha=0.7, align='edge', label=names[b])
        ax3.axvline(np.mean(lengths_by_tracker[a]), color=colors[a], linestyle='-', lw=2)
        ax3.axvline(np.mean(lengths_by_tracker[b]), color=colors[b], linestyle='-', lw=2)
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{abs(y):.0f}%'))
        ax3.set_title("Mirrored Histogram Comparison", fontsize=16, fontweight='bold')
    else:
        for tracker in trackers:
            hist, _ = np.histogram(lengths_by_tracker[tracker], bins=bins)
            hist = hist / len(lengths_by_tracker[tracker]) * 100
            ax3.step(bins[:-1], hist, where='post', color=colors[tracker], lw=2, label=names[tracker])
            ax3.axvline(np.mean(lengths_by_tracker[tracker]), color=colors[tracker], linestyle='--', lw=1.5, alpha=0.7)
        ax3.set_title("Histogram Comparison", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Trajectory Length", fontsize=14)
    ax3.set_ylabel("Percentage (%)", fontsize=14)
    ax3.legend(loc='upper right', frameon=True, fontsize=12)
    ax3.spines['top'].set_visible(False)
    ax3.grid(True, linestyle='--', alpha=0.3)

    # Bottom: density comparison (signed difference for two trackers, overlaid KDE otherwise)
    ax4 = fig.add_subplot(gs[2, :])
    x_range = np.linspace(min(all_data), max(all_data), 1000)
    if len(trackers) == 2:
        a, b = trackers
        diff = stats.gaussian_kde(lengths_by_tracker[a])(x_range) - stats.gaussian_kde(lengths_by_tracker[b])(x_range)
        ax4.fill_between(
            x_range, diff, 0, where=(diff > 0), color=colors[a], alpha=0.7, label=f"{names[a]} higher density"
        )
        ax4.fill_between(
            x_range, diff, 0, where=(diff < 0), color=colors[b], alpha=0.7, label=f"{names[b]} higher density"
        )
        ax4.axhline(y=0, color='black', linestyle='-', lw=1)
        ax4.set_title(f"Density Difference ({names[a]} - {names[b]})", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Density Difference", fontsize=12)
    else:
        for tracker in trackers:
            ax4.plot(
                x_range,
                stats.gaussian_kde(lengths_by_tracker[tracker])(x_range),
                color=colors[tracker],
                lw=2,
                label=names[tracker],
            )
        ax4.set_title("Density Comparison (KDE)", fontsize=14, fontweight='bold')
        ax4.set_ylabel("Density", fontsize=12)
    ax4.set_xlabel("Trajectory Length", fontsize=12)
    ax4.legend(loc='best', frameon=True, fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Add per-tracker summary statistics
    summary_lines = ["Statistics Summary (trajectory length):"]
    for tracker in trackers:
        values = lengths_by_tracker[tracker]
        summary_lines.append(
            f"{names[tracker]} - Mean: {np.mean(values):.2f}, Median: {np.median(values):.2f}, "
            f"Std: {np.std(values):.2f}, Count: {len(values)}"
        )
    plt.figtext(
        0.5,
        0.005,
        "\n".join(summary_lines),
        ha="center",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5, "boxstyle": "round,pad=0.5"},
    )

    plt.suptitle("Trajectory Length Distribution Analysis", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)

    if args.show:
        plt.show()

    if args.save:
        save_path = args.input / "plots" / "trajectory_length_distribution_comparison.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")


def get_cli_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Compare two or more tracking algorithms")
    parser.add_argument("input", type=Path, help="Path to the folder containing video files with results to compare")
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=DEFAULT_TRACKERS,
        metavar="NAME",
        help="Tracker names to compare; each expects a 'results_<name>/' subfolder. "
        f"Default: the geo-trax trackers ({', '.join(DEFAULT_TRACKERS)}). Any names are accepted.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot")
    parser.add_argument("--save", action="store_true", help="Save the plot")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    compare_tracks(args)
