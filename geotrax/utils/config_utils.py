# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""YAML configuration loading and path resolution for the geo-trax pipeline."""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Union

import yaml
from ultralytics import YOLO

from geotrax import CFG_DIR, PACKAGE_DIR

ROOT_DIR = PACKAGE_DIR.parent  # repository root (source checkout) or site-packages (installed)


def resolve_config_path(cfg_filepath: Union[str, Path]) -> Path:
    """Resolve a configuration file path.

    Tries, in order: the path as given (absolute or relative to the current working directory),
    the path relative to the package parent directory, and the path inside the bundled
    configuration directory (geotrax/cfg). A missing '.yaml' suffix and a legacy leading 'cfg/'
    component are tolerated, so e.g. 'confident', 'cfg/default.yaml', and 'lenient' all resolve
    to the bundled presets. Returns the path unchanged if no candidate exists.
    """
    path = Path(cfg_filepath)
    if not path.suffix:
        path = path.with_suffix('.yaml')

    candidates = [path]
    if not path.is_absolute():
        bundled = Path(*path.parts[1:]) if path.parts[0] == 'cfg' else path
        candidates += [ROOT_DIR / path, CFG_DIR / bundled]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(cfg_filepath)


def resolve_asset_path(filepath: Union[str, Path]) -> Path:
    """Resolve a non-config asset path (e.g., model weights) against the cwd and the package parent.

    Returns the path unchanged if no candidate exists, leaving error reporting to the caller.
    """
    path = Path(filepath)
    if not path.is_absolute() and not path.is_file() and (ROOT_DIR / path).is_file():
        return ROOT_DIR / path
    return path


def load_config_all(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """Load the unified pipeline configuration file and return a nested dict.

    The pipeline config is a single YAML file with top-level sections: folders, processing,
    batch, extraction, stabilo, georef, visualization, plotting, ultralytics, tracker.
    The tracker section holds an 'active' selector plus a full parameter block per supported
    tracker; the active block is written to a temporary YAML file so Ultralytics can read it
    as a file path (its required interface).
    """
    full = load_config(args.cfg, logger)

    kwargs_tracker     = full.get('tracker', {})
    kwargs_stabilo     = full.get('stabilo', {})
    kwargs_ultralytics = dict(full.get('ultralytics', {}))
    kwargs_georef      = full.get('georef', {})
    kwargs_main        = {k: v for k, v in full.items()
                          if k not in ('tracker', 'stabilo', 'ultralytics', 'georef')}

    kwargs_ultralytics['tracker'] = str(_write_tracker_yaml(kwargs_tracker, args.cfg, logger))
    kwargs_ultralytics['model'] = str(resolve_asset_path(kwargs_ultralytics['model']))

    kwargs_main['class_names'] = load_class_names_from_model(Path(kwargs_ultralytics['model']), logger)
    kwargs_main['args'] = args

    keys_to_update = ['classes', 'conf', 'show']
    for arg, value in vars(args).items():
        if value is not None and arg in keys_to_update:
            kwargs_ultralytics[arg] = value
            logger.info(f"The default ultralytics value for {arg} has been updated to the provided CLI argument: {value}.")

    logger.info(f"Pipeline configuration loaded from: '{args.cfg}'.")

    return {
        'main': kwargs_main,
        'stabilo': kwargs_stabilo,
        'ultralytics': kwargs_ultralytics,
        'georef': kwargs_georef,
    }


def _write_tracker_yaml(tracker_section: dict, cfg_name: Union[str, Path], logger: logging.Logger) -> Path:
    """Select the active tracker block and write it to a temporary YAML file; return its path.

    The pipeline config's ``tracker`` section holds an ``active`` selector plus one parameter
    block per supported tracker. Only the active block is passed to Ultralytics, which requires
    a file path for the tracker config; this bridges the unified config to that interface.
    The temp file persists until OS cleanup.
    """
    active = tracker_section.get('active')
    if active is None:
        logger.critical(f"No 'active' tracker selector found in the 'tracker' section of '{cfg_name}'.")
        sys.exit(1)
    if active not in tracker_section:
        available = [k for k in tracker_section if k != 'active']
        logger.critical(
            f"Active tracker '{active}' has no parameter block in the 'tracker' section of "
            f"'{cfg_name}'. Available: {available}."
        )
        sys.exit(1)

    tracker_cfg = tracker_section[active]
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, prefix='geotrax_tracker_', encoding='utf-8'
        ) as tmp:
            yaml.dump(tracker_cfg, tmp, default_flow_style=False, allow_unicode=True)
            return Path(tmp.name)
    except OSError as exc:
        logger.critical(f"Failed to write temporary tracker config: {exc}")
        sys.exit(1)


def load_config(cfg_filepath: Union[str, Path], logger: logging.Logger) -> dict:
    """Load a configuration file and return the contents as a dictionary."""
    resolved_filepath = resolve_config_path(cfg_filepath)
    try:
        with open(resolved_filepath, 'r') as f:
            kwargs = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"Configuration file '{cfg_filepath}' not found.")
        sys.exit(1)
    return kwargs


def backfill_args_from_config(args: argparse.Namespace, mapping: dict) -> None:
    """Set each ``args.arg_name`` from ``mapping[arg_name]`` when the arg is still ``None``
    (i.e. not overridden on the command line)."""
    for arg_name, config_value in mapping.items():
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, config_value)


def load_class_names_from_model(model_path: Path, logger: logging.Logger) -> dict:
    """Load class names embedded in a YOLO model file."""
    try:
        names = YOLO(str(model_path)).names
        logger.info(f"Class names loaded from model: '{model_path}'.")
        return names
    except Exception as e:
        logger.error(f"Failed to load class names from '{model_path}': {e}. Using default class names.")
        return {i: f'class_{i}' for i in range(100)}
