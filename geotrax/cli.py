# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
cli.py - Umbrella command-line interface for the geo-trax pipeline.

Dispatches the 'geotrax' console command to the individual pipeline stages. Each
subcommand accepts exactly the same arguments as its underlying module; run
'geotrax <command> --help' for the full per-command reference.

Usage:
    geotrax <command> [options]

Commands:
    batch        : Run the full pipeline (extract -> georeference -> visualize -> plot)
                   for a single video or an entire directory tree (primary entry point).
    extract      : Stage 1 - vehicle detection, tracking, and trajectory stabilization.
    georeference : Stage 2 - map stabilized trajectories to real-world coordinates.
    aggregate    : Stage 3 - merge georeferenced results across drones/flights.
    visualize    : Overlay extracted tracks on the video (original/stabilized/reference frame).
    plot         : Plot trajectories and kinematic/dimension distributions.

Options:
    --help, -h     : Show this help message and exit.
    --version, -V  : Show the installed geo-trax version and exit.

Examples:
  1. Process a video end-to-end without georeferencing:
        geotrax batch path/to/video.mp4 --no-geo

  2. Run only the extraction stage with a custom config:
        geotrax extract path/to/video.mp4 -c path/to/custom.yaml

  3. Render annotated videos for multiple visualization modes:
        geotrax visualize path/to/video.mp4 --save --viz-mode 0 1 2
"""

import importlib
import sys

from geotrax import __version__

# Subcommand -> (module with a main() entry point, one-line description).
# Modules are imported lazily so that 'geotrax --help' stays fast.
COMMANDS = {
    'batch': ('geotrax.batch_process', 'Run the full pipeline for a video or directory tree (primary entry point)'),
    'extract': ('geotrax.detect_track_stabilize', 'Stage 1: vehicle detection, tracking, and stabilization'),
    'georeference': ('geotrax.georeference', 'Stage 2: map stabilized trajectories to real-world coordinates'),
    'aggregate': ('geotrax.aggregate', 'Stage 3: merge georeferenced results across drones/flights'),
    'visualize': ('geotrax.visualize', 'Overlay extracted tracks on the video'),
    'plot': ('geotrax.plot', 'Plot trajectories and distributions'),
}


def build_usage() -> str:
    """Build the top-level usage/help message."""
    lines = [
        'usage: geotrax <command> [options]',
        '',
        'Geo-trax: georeferenced vehicle trajectory extraction from drone imagery.',
        '',
        'commands:',
    ]
    width = max(len(name) for name in COMMANDS)
    lines += [f'  {name:<{width}}  {description}' for name, (_, description) in COMMANDS.items()]
    lines += [
        '',
        "Run 'geotrax <command> --help' for command-specific options.",
    ]
    return '\n'.join(lines)


def main() -> None:
    """Entry point for the 'geotrax' console command."""
    argv = sys.argv[1:]

    if not argv or argv[0] in ('-h', '--help'):
        print(build_usage())
        return
    if argv[0] in ('-V', '--version'):
        print(f'geo-trax {__version__}')
        return

    command = argv[0]
    if command not in COMMANDS:
        print(f"geotrax: error: unknown command '{command}'\n\n{build_usage()}", file=sys.stderr)
        sys.exit(2)

    module = importlib.import_module(COMMANDS[command][0])
    sys.argv = [f'geotrax {command}'] + argv[1:]
    module.main()


if __name__ == '__main__':
    main()
