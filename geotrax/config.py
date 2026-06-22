#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
config.py - Pipeline Configuration Management

The whole pipeline is driven by a single, self-contained config file (the "pipeline config").
The defaults ship bundled inside the installed package, so there is no path to edit directly.
This command lets you locate, inspect, and copy them.

Usage:
  geotrax config show [preset]
  geotrax config copy [--output-dir DIR] [--overwrite]

Subcommands:
  show          Explain the config system, list available presets, and show how to use them.
  show <preset> Print the full contents of one bundled preset (default, confident, lenient, stable).
  copy          Copy the bundled presets into a local directory (default: the current directory)
                as <name>_copy.yaml files, ready to edit and pass to any command via -c.

Options (copy):
  -o, --output-dir <path> : Destination directory for the copies (default: current directory).
  --overwrite             : Overwrite existing <name>_copy.yaml files in the destination.

Examples:
1. Overview of the config system — presets, how to select and customise them:
   geotrax config show

2. Print the full default pipeline config (or browse it on GitHub — see 'geotrax config show'):
   geotrax config show default

3. Copy all presets into the current directory for editing:
   geotrax config copy

4. Copy into a specific directory, overwriting any previous copies:
   geotrax config copy --output-dir ~/my_project --overwrite

Then edit a copy and pass it to any command:
   geotrax extract video.mp4 -c default_copy.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from geotrax import CFG_DIR

# Pipeline config presets bundled with the package (top-level YAML files only).
_PRESETS = ['default', 'confident', 'lenient', 'stable']

_PRESET_DESCRIPTIONS = {
    'default':   'Balanced baseline; suitable for most scenes.',
    'confident': 'Precision: higher conf (0.4), tighter NMS, longer min track; fewer, more reliable detections.',
    'lenient':   'Recall: relaxes every threshold (conf 0.15, max_det 1500, looser NMS/association/track-init).',
    'stable':    'Stabilization quality: full res + more features + CLAHE + stricter matching; slower, more accurate.',
}

_COPY_SUFFIX = '_copy'


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='geotrax config',
        description='Pipeline configuration management (locate, inspect, and copy the bundled configs).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='action', metavar='<subcommand>')

    # --- show ---
    show_p = subparsers.add_parser(
        'show',
        help='Explain the config system and list presets, or print the full contents of one preset.',
        description='Without an argument: explain the pipeline config system, list available '
                    'presets, and show how to inspect, copy, and select them. '
                    'With a preset name: print that preset in full.',
    )
    show_p.add_argument(
        'preset', nargs='?', default=None, choices=_PRESETS, metavar='preset',
        help=f"Optional preset to print in full: {', '.join(_PRESETS)}.",
    )

    # --- copy ---
    copy_p = subparsers.add_parser(
        'copy',
        help='Copy the bundled presets locally (as <name>_copy.yaml) for editing.',
        description='Copy the bundled pipeline config presets into a local directory as '
                    '<name>_copy.yaml files, ready to edit and pass to any command via -c.',
    )
    copy_p.add_argument(
        '--output-dir', '-o', type=Path, default=Path('.'), metavar='DIR',
        help='Destination directory for the copies (default: the current directory).',
    )
    copy_p.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing <name>_copy.yaml files in the destination.',
    )

    args = parser.parse_args()
    if args.action is None:
        parser.print_help()
        sys.exit(0)
    return args


def _error(message: str) -> None:
    """Print an error to stderr and exit with a non-zero status."""
    print(f"geotrax config: error: {message}", file=sys.stderr)
    sys.exit(1)


def _run_show(preset: Optional[str]) -> None:
    if preset is not None:
        src = CFG_DIR / f'{preset}.yaml'
        if not src.is_file():
            _error(f"bundled preset not found: '{src}'.")
        print(src.read_text())
        return

    print("geo-trax is driven by a single pipeline config that controls every stage: detection,")
    print("tracking, stabilization, georeferencing, visualization, and plotting. Select a preset")
    print("with -c on any command, e.g.:")
    print()
    print("  geotrax extract video.mp4 -c confident")
    print("  geotrax batch   video.mp4 -c lenient -orf data/orthophotos")
    print()
    print("Available presets:")
    width = max(len(n) for n in _PRESETS)
    for name in _PRESETS:
        marker = '  [missing]' if not (CFG_DIR / f'{name}.yaml').is_file() else ''
        print(f"  {name:<{width}}  {_PRESET_DESCRIPTIONS[name]}{marker}")
    print()
    print("Inspect a preset in full:")
    print("  geotrax config show default")
    print()
    print("To customise, copy a preset locally, edit it, then pass it via -c:")
    print("  geotrax config copy                        # writes *_copy.yaml to the current directory")
    print("  geotrax extract video.mp4 -c default_copy.yaml")
    print()
    print("  See 'geotrax config copy --help' for output-directory and overwrite options.")
    print()
    print("Browse presets online (alternative to 'geotrax config show <preset>'):")
    print("  https://github.com/rfonod/geo-trax/tree/main/geotrax/cfg")
    print()
    print(f"Bundled config location (installed package):\n  {CFG_DIR}")


def _run_copy(output_dir: Path, overwrite: bool) -> None:
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    targets = {name: out / f'{name}{_COPY_SUFFIX}.yaml' for name in _PRESETS}

    if not overwrite:
        existing = [t.name for t in targets.values() if t.exists()]
        if existing:
            _error(
                f"these files already exist in '{out}': {', '.join(existing)}. "
                "Use --overwrite to replace them."
            )

    copied = []
    for name, dst in targets.items():
        src = CFG_DIR / f'{name}.yaml'
        if not src.is_file():
            print(f"Warning: bundled preset not found, skipping: '{src}'.", file=sys.stderr)
            continue
        shutil.copy2(src, dst)
        copied.append(dst)

    if not copied:
        _error("no preset files were copied.")

    print(f"Copied {len(copied)} pipeline config preset(s) to: {out}/\n")
    width = max(len(p.name) for p in copied)
    for dst in copied:
        name = dst.stem[: -len(_COPY_SUFFIX)] if dst.stem.endswith(_COPY_SUFFIX) else dst.stem
        print(f"  {dst.name:<{width}}  {_PRESET_DESCRIPTIONS.get(name, '')}")
    print()
    print("Edit a copy, then pass it to any command with -c, e.g.:")
    example = (Path(output_dir) / f'default{_COPY_SUFFIX}.yaml')
    print(f"  geotrax extract video.mp4 -c {example}")


def main() -> None:
    args = parse_cli_args()
    if args.action == 'show':
        _run_show(args.preset)
    elif args.action == 'copy':
        _run_copy(args.output_dir, args.overwrite)


if __name__ == '__main__':
    main()
