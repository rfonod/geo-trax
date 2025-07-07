"""
Compare tracking results from two different vehicle tracking algorithms.

This script analyzes and compares the performance of BoT-SORT and ByteTrack tracking
algorithms by evaluating trajectory length distributions and missing frame patterns.
It processes tracking results from multiple videos, computes statistical metrics, and
generates comprehensive visualizations to highlight differences between the algorithms.

Usage:
    python compare_tracking.py INPUT [--show] [--save]

Arguments:
    INPUT                    Path to folder containing video files with tracking results

Options:
    --show                   Display the generated plots interactively
    --save                   Save plots to 'plots/' subdirectory as PNG files

Input Format:
    Tracking results should be in text files with the following structure:
    - results_botsort/{video_name}.txt
    - results_bytetrack/{video_name}.txt

    Each line contains 14 comma-separated values:
    frame_number,vehicle_id,x,y,width,height,x_stab,y_stab,width_stab,height_stab,class_id,confidence,vehicle_length,vehicle_width

Output:
    - Statistical comparison of trajectory lengths and missing frames
    - KL divergence measures between distributions
    - Multi-panel visualization with violin plots, CDFs, mirrored histograms, and density differences
    - Optional PNG plot files saved to plots/ directory

Notes:
    - Videos starting with 'P' are automatically skipped
    - Requires both tracking result files to exist for comparison
    - Uses KDE-based density estimation for detailed distribution analysis
    - Generates publication-ready plots with enhanced styling
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add project root directory to Python path
from utils.utils import detect_delimiter


def compare_tracks(args):
    """
    Compare tracking results
    """
    # Get all the files in the input folder
    video_files = sorted(list(args.input.glob("*.MP4")))

    # Initialize lists to store the trajectory lengths for each tracking algorithm
    botsort_trajectory_lengths = []
    bytetrack_trajectory_lengths = []

    # Initialize lists to store the missing frames for each tracking algorithm
    botsort_missing_frames = []
    bytetrack_missing_frames = []

    # Iterate over each video file
    for video_file in video_files:
        if video_file.stem[0] == "P":
            continue
        print(f"Comparing tracking results for video: {video_file.stem}")

        botsort_results_filepath = args.input / "results_botsort" / f"{video_file.stem}.txt"
        bytetrack_results_filepath = args.input / "results_bytetrack" / f"{video_file.stem}.txt"

        # Load the results
        if not botsort_results_filepath.exists() or not bytetrack_results_filepath.exists():
            print(f"Results files not found for video: {video_file.stem}")
            continue
        delimiter = detect_delimiter(botsort_results_filepath)
        botsort_tracks = np.loadtxt(botsort_results_filepath, delimiter=delimiter, dtype=np.float64)
        delimiter = detect_delimiter(bytetrack_results_filepath)
        bytetrack_tracks = np.loadtxt(bytetrack_results_filepath, delimiter=delimiter, dtype=np.float64)

        # Compute the trajectory length distribution for each tracking algorithm
        botsort_trajectory_lengths.extend(compute_trajectory_lengths(botsort_tracks))
        bytetrack_trajectory_lengths.extend(compute_trajectory_lengths(bytetrack_tracks))

        # Determine the missing frames for each tracking algorithm
        botsort_missing_frames.extend(find_missing_frames(botsort_tracks))
        bytetrack_missing_frames.extend(find_missing_frames(bytetrack_tracks))

    print("\nTrajectory Length Analysis:\n")

    # Print the total number of vehicles tracked by each tracking algorithm
    print(f"Total number of vehicles tracked by BoT-SORT: {len(botsort_trajectory_lengths)}")
    print(f"Total number of vehicles tracked by ByteTrack: {len(bytetrack_trajectory_lengths)}")

    # Compute the average trajectory length for each tracking algorithm
    botsort_avg_trajectory_length = np.mean(botsort_trajectory_lengths)
    bytetrack_avg_trajectory_length = np.mean(bytetrack_trajectory_lengths)

    print(f"Average trajectory length for BoT-SORT: {botsort_avg_trajectory_length:.2f}")
    print(f"Average trajectory length for ByteTrack: {bytetrack_avg_trajectory_length:.2f}")

    # Compute the average trajectory length difference between the two tracking algorithms
    avg_trajectory_length_diff = np.abs(botsort_avg_trajectory_length - bytetrack_avg_trajectory_length)
    print(f"Average trajectory length difference: {avg_trajectory_length_diff:.2f}")

    # Compute the standard deviation of the trajectory lengths for each tracking algorithm
    botsort_std_dev = np.std(botsort_trajectory_lengths)
    bytetrack_std_dev = np.std(bytetrack_trajectory_lengths)

    print(f"Standard deviation of trajectory lengths for BoT-SORT: {botsort_std_dev:.2f}")
    print(f"Standard deviation of trajectory lengths for ByteTrack: {bytetrack_std_dev:.2f}")

    # Compute the standard deviation difference between the two tracking algorithms
    std_dev_diff = np.abs(botsort_std_dev - bytetrack_std_dev)
    print(f"Standard deviation difference: {std_dev_diff:.2f}")

    # Compute the KL divergence between the two trajectory length distributions
    kl_bs_bt = compute_kl_divergence(botsort_trajectory_lengths, bytetrack_trajectory_lengths)
    kl_bt_bs = compute_kl_divergence(bytetrack_trajectory_lengths, botsort_trajectory_lengths)
    kl_divergence = (kl_bs_bt + kl_bt_bs) / 2
    print(f"KL divergence from BoT-SORT to ByteTrack: {kl_bs_bt:.4f}")
    print(f"KL divergence from ByteTrack to BoT-SORT: {kl_bt_bs:.4f}")
    print(f"Average KL divergence: {kl_divergence:.4f}")

    print("\nMissing Frames Analysis:\n")

    # Compute the average number of missing frames for each tracking algorithm
    botsort_avg_missing_frames = np.mean(botsort_missing_frames)
    bytetrack_avg_missing_frames = np.mean(bytetrack_missing_frames)

    print(f"Average number of missing frames for BoT-SORT: {botsort_avg_missing_frames:.2f}")
    print(f"Average number of missing frames for ByteTrack: {bytetrack_avg_missing_frames:.2f}")

    # Compute the standard deviation of the missing frames for each tracking algorithm
    botsort_missing_frames_std_dev = np.std(botsort_missing_frames)
    bytetrack_missing_frames_std_dev = np.std(bytetrack_missing_frames)

    print(f"Standard deviation of missing frames for BoT-SORT: {botsort_missing_frames_std_dev:.2f}")
    print(f"Standard deviation of missing frames for ByteTrack: {bytetrack_missing_frames_std_dev:.2f}")

    kl_bs_bt = compute_kl_divergence(botsort_missing_frames, bytetrack_missing_frames)
    kl_bt_bs = compute_kl_divergence(bytetrack_missing_frames, botsort_missing_frames)
    kl_divergence = (kl_bs_bt + kl_bt_bs) / 2

    print(f"KL divergence from BoT-SORT to ByteTrack (missing frames): {kl_bs_bt:.4f}")
    print(f"KL divergence from ByteTrack to BoT-SORT (missing frames): {kl_bt_bs:.4f}")
    print(f"Average KL divergence (missing frames): {kl_divergence:.4f}")

    # Plot the trajectory length distributions
    if args.show or args.save:
        plot_trajectory_length_distributions(botsort_trajectory_lengths, bytetrack_trajectory_lengths, args)


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


def plot_trajectory_length_distributions(botsort_trajectory_lengths, bytetrack_trajectory_lengths, args):
    """
    Plot the trajectory length distributions for the two tracking algorithms with enhanced
    visualization techniques to highlight subtle differences between similar distributions.
    """

    # Convert the trajectory lengths from number of frames to seconds
    # botsort_trajectory_lengths = np.array(botsort_trajectory_lengths) / 29.97
    # bytetrack_trajectory_lengths = np.array(bytetrack_trajectory_lengths) / 29.97

    # Create a figure with a custom, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 10), dpi=100)

    # Create a custom grid layout
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1.5, 1])

    # Calculate statistics for annotations
    bs_mean = np.mean(botsort_trajectory_lengths)
    bt_mean = np.mean(bytetrack_trajectory_lengths)
    bs_median = np.median(botsort_trajectory_lengths)
    bt_median = np.median(bytetrack_trajectory_lengths)
    bs_std = np.std(botsort_trajectory_lengths)
    bt_std = np.std(bytetrack_trajectory_lengths)

    # Calculate KL divergence
    # First create histograms with identical bins
    min_val = min(np.min(botsort_trajectory_lengths), np.min(bytetrack_trajectory_lengths))
    max_val = max(np.max(botsort_trajectory_lengths), np.max(bytetrack_trajectory_lengths))
    bins = np.linspace(min_val, max_val, 50)

    bs_hist, _ = np.histogram(botsort_trajectory_lengths, bins=bins, density=True)
    bt_hist, _ = np.histogram(bytetrack_trajectory_lengths, bins=bins, density=True)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    bs_hist = bs_hist + epsilon
    bt_hist = bt_hist + epsilon

    # Normalize
    bs_hist = bs_hist / np.sum(bs_hist)
    bt_hist = bt_hist / np.sum(bt_hist)

    # Calculate KL divergence both ways
    kl_bs_bt = np.sum(bs_hist * np.log(bs_hist / bt_hist))
    kl_bt_bs = np.sum(bt_hist * np.log(bt_hist / bs_hist))

    # Colors with better contrast
    botsort_color = '#3A6DAA'  # Deeper blue
    bytetrack_color = '#E57200'  # Deeper orange

    # Top left: Violin plot instead of Ridge view
    ax1 = fig.add_subplot(gs[0, 0])

    # Create a DataFrame for violin plot
    import pandas as pd

    df = pd.DataFrame({'BoT-SORT': botsort_trajectory_lengths, 'ByteTrack': bytetrack_trajectory_lengths})
    df_melted = pd.melt(df, var_name='Algorithm', value_name='Trajectory Length')

    # Create violin plot
    sns.violinplot(
        x='Algorithm',
        y='Trajectory Length',
        data=df_melted,
        palette={'BoT-SORT': botsort_color, 'ByteTrack': bytetrack_color},
        inner='quartile',
        cut=0,
        ax=ax1,
    )

    # Add mean points
    ax1.scatter([0, 1], [bs_mean, bt_mean], color='white', s=30, zorder=3)
    ax1.scatter([0, 1], [bs_mean, bt_mean], color='black', s=15, zorder=4)

    ax1.set_title("Distribution Comparison (Violin Plot)", fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Top right: Cumulative distribution function
    ax2 = fig.add_subplot(gs[0, 1])
    sns.ecdfplot(botsort_trajectory_lengths, label="BoT-SORT", color=botsort_color, lw=2, ax=ax2)
    sns.ecdfplot(bytetrack_trajectory_lengths, label="ByteTrack", color=bytetrack_color, lw=2, ax=ax2)
    ax2.set_title("Cumulative Distribution Function", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Trajectory Length", fontsize=12)
    ax2.set_ylabel("Cumulative Probability", fontsize=12)
    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Middle: Double-sided histogram for direct comparison
    ax3 = fig.add_subplot(gs[1, :])

    # Create bins for the histograms
    all_data = np.concatenate([botsort_trajectory_lengths, bytetrack_trajectory_lengths])
    bins = np.linspace(min(all_data), max(all_data), 40)

    # Calculate histograms
    bs_hist, _ = np.histogram(botsort_trajectory_lengths, bins=bins)
    bt_hist, _ = np.histogram(bytetrack_trajectory_lengths, bins=bins)

    # Normalize to get percentages
    bs_hist = bs_hist / len(botsort_trajectory_lengths) * 100
    bt_hist = -bt_hist / len(bytetrack_trajectory_lengths) * 100  # Negative for bottom direction

    # Plot the mirrored histogram
    width = bins[1] - bins[0]
    ax3.bar(bins[:-1], bs_hist, width=width, color=botsort_color, alpha=0.7, align='edge', label="BoT-SORT")
    ax3.bar(bins[:-1], bt_hist, width=width, color=bytetrack_color, alpha=0.7, align='edge', label="ByteTrack")

    # Add mean lines
    ax3.axvline(bs_mean, color=botsort_color, linestyle='-', lw=2)
    ax3.axvline(bt_mean, color=bytetrack_color, linestyle='-', lw=2)

    # Add text annotations for means
    ax3.annotate(
        f'BoT-SORT Mean: {bs_mean:.2f}',
        xy=(bs_mean, 5),
        xytext=(bs_mean + 1, 5),
        color=botsort_color,
        fontweight='bold',
        ha='left',
        va='center',
        fontsize=10,
    )
    ax3.annotate(
        f'ByteTrack Mean: {bt_mean:.2f}',
        xy=(bt_mean, -5),
        xytext=(bt_mean + 1, -5),
        color=bytetrack_color,
        fontweight='bold',
        ha='left',
        va='center',
        fontsize=10,
    )

    # Set the y-ticks to absolute values
    yticks = ax3.get_yticks()
    ax3.set_yticklabels([f'{abs(y):.0f}%' for y in yticks])

    ax3.set_title("Mirrored Histogram Comparison", fontsize=16, fontweight='bold')
    ax3.set_xlabel("Trajectory Length", fontsize=14)
    ax3.set_ylabel("Percentage (%)", fontsize=14)
    ax3.legend(loc='upper right', frameon=True, fontsize=12)
    ax3.spines['top'].set_visible(False)
    ax3.grid(True, linestyle='--', alpha=0.3)

    # Bottom: Difference plot to highlight where distributions differ
    ax4 = fig.add_subplot(gs[2, :])

    # Create KDEs for both distributions
    kde_bs = stats.gaussian_kde(botsort_trajectory_lengths)
    kde_bt = stats.gaussian_kde(bytetrack_trajectory_lengths)

    # Calculate the difference between densities
    x_range = np.linspace(min(all_data), max(all_data), 1000)
    diff = kde_bs(x_range) - kde_bt(x_range)

    # Plot the difference
    ax4.fill_between(
        x_range, diff, 0, where=(diff > 0), color=botsort_color, alpha=0.7, label="BoT-SORT higher density"
    )
    ax4.fill_between(
        x_range, diff, 0, where=(diff < 0), color=bytetrack_color, alpha=0.7, label="ByteTrack higher density"
    )

    ax4.axhline(y=0, color='black', linestyle='-', lw=1)
    ax4.set_title("Density Difference (BoT-SORT - ByteTrack)", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Trajectory Length", fontsize=12)
    ax4.set_ylabel("Density Difference", fontsize=12)
    ax4.legend(loc='best', frameon=True, fontsize=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Add summary statistics with standard deviation and KL divergence
    plt.figtext(
        0.5,
        0.005,
        f"Statistics Summary:\n"
        f"BoT-SORT - Mean: {bs_mean:.2f}, Median: {bs_median:.2f}, Std: {bs_std:.2f}, Count: {len(botsort_trajectory_lengths)}\n"
        f"ByteTrack - Mean: {bt_mean:.2f}, Median: {bt_median:.2f}, Std: {bt_std:.2f}, Count: {len(bytetrack_trajectory_lengths)}\n"
        f"KL Divergence: BoT-SORT→ByteTrack: {kl_bs_bt:.4f}, ByteTrack→BoT-SORT: {kl_bt_bs:.4f}",
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
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")


def get_cli_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Compare various tracking outputs")
    parser.add_argument("input", type=Path, help="Path to the folder containing video files with results to compare")
    parser.add_argument("--show", action="store_true", help="Show the plot")
    parser.add_argument("--save", action="store_true", help="Save the plot")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cli_arguments()
    compare_tracks(args)
