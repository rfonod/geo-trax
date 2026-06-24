# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""YAML configuration loading and path resolution for the geo-trax pipeline."""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import yaml

from geotrax import CFG_DIR, PACKAGE_DIR

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE as _HF_HUB_CACHE
except ImportError:
    hf_hub_download = None
    try_to_load_from_cache = None
    _HF_HUB_CACHE = None

ROOT_DIR = PACKAGE_DIR.parent  # repository root (source checkout) or site-packages (installed)

# Scheme prefix for Hugging Face Hub model references in the config, e.g.
# 'hf://rfonod/geo-trax/geotrax_hbb_yolov8s_1920_v1.pt' -> repo 'rfonod/geo-trax', file '...pt'.
HF_PREFIX = 'hf://'


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


def resolve_model_path(model_ref: Union[str, Path], logger: logging.Logger) -> Path:
    """Resolve a model reference to a local file path, downloading from Hugging Face if needed.

    Two forms are supported via the same config/CLI entry:
      * A Hugging Face reference ``hf://<org>/<repo>/<path/to/file>`` (e.g.
        ``hf://rfonod/geo-trax/geotrax_hbb_yolov8s_1920_v1.pt``). The weight is downloaded once and
        served from the standard Hugging Face Hub cache (``~/.cache/huggingface/hub``, overridable via
        ``HF_HOME``/``HF_HUB_CACHE``); ``hf_hub_download`` revalidates by etag, so repeat runs do not
        re-download. The cache location is identical for every install mode (PyPI, source, editable).
      * A local path (absolute or relative), which keeps the historical behaviour via
        :func:`resolve_asset_path` and is never downloaded.
    """
    model_str = str(model_ref).strip()
    if model_str.startswith('hf download '):
        model_str = model_str[len('hf download '):].strip()
    if not model_str.startswith(HF_PREFIX):
        return resolve_asset_path(model_str)

    if hf_hub_download is None:
        logger.critical(
            f"Model '{model_str}' is a Hugging Face reference but 'huggingface_hub' is not installed. "
            "Install it (it is a core dependency: `python -m pip install -e .`) or point the config "
            "`ultralytics -> model` (or --model) at a local weights file."
        )
        sys.exit(1)

    parts = model_str[len(HF_PREFIX):].split('/')
    if len(parts) < 3:
        logger.critical(
            f"Malformed Hugging Face model reference '{model_str}'. Expected "
            f"'{HF_PREFIX}<org>/<repo>/<path/to/file>' (e.g. '{HF_PREFIX}rfonod/geo-trax/geotrax_hbb_yolov8s_1920_v1.pt')."
        )
        sys.exit(1)

    repo_id = '/'.join(parts[:2])
    filename = '/'.join(parts[2:])
    cached = try_to_load_from_cache(repo_id=repo_id, filename=filename) if try_to_load_from_cache else None
    is_cached = isinstance(cached, str)
    if not is_cached:
        cache_hint = str(_HF_HUB_CACHE) if _HF_HUB_CACHE else '~/.cache/huggingface/hub'
        logger.notice(
            f"Downloading '{filename}' from Hugging Face (repo: '{repo_id}') → "
            f"cache: '{cache_hint}' (override via HF_HOME or HF_HUB_CACHE) ..."
        )
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        logger.critical(f"Failed to download model '{filename}' from Hugging Face repo '{repo_id}': {e}")
        sys.exit(1)
    if is_cached:
        logger.info(f"Model '{filename}' loaded from cache: '{local_path}'.")
    return Path(local_path)


def load_config_all(args: argparse.Namespace, logger: logging.Logger, needs_model: bool = True) -> dict:
    """Load the unified pipeline configuration file and return a nested dict.

    The pipeline config is a single YAML file with top-level sections: input, output,
    processing, batch, extraction, stabilo, georef, visualization, plotting, ultralytics,
    tracker. The tracker section holds an 'active' selector plus a full parameter block per
    supported tracker; the active block is written to a temporary YAML file so Ultralytics can
    read it as a file path (its required interface).

    Set ``needs_model=False`` for stages (e.g. georeferencing) that never use the detection
    model or class names. This skips the tracker YAML, model path resolution, and HF download
    for those stages so a missing or unavailable model does not abort them.
    """
    full = load_config(args.cfg, logger)

    kwargs_tracker     = full.get('tracker', {})
    kwargs_stabilo     = full.get('stabilo', {})
    kwargs_ultralytics = dict(full.get('ultralytics', {}))
    kwargs_georef      = full.get('georef', {})
    kwargs_main        = {k: v for k, v in full.items()
                          if k not in ('tracker', 'stabilo', 'ultralytics', 'georef')}

    if needs_model:
        kwargs_ultralytics['tracker'] = str(_write_tracker_yaml(kwargs_tracker, args.cfg, logger))
        # The model and class-rename mapping live in the 'extraction:' section (the 'ultralytics:'
        # section keeps only a pointer comment). A CLI --model override takes precedence over the
        # config value, then the reference (local path or hf:// auto-download) is resolved to a
        # concrete local file for Ultralytics.
        extraction_cfg = full.get('extraction', {})
        raw_model = getattr(args, 'model', None)
        if isinstance(raw_model, list):
            raw_model = ' '.join(raw_model)
        model_ref = raw_model or extraction_cfg.get('model') or kwargs_ultralytics.get('model')
        kwargs_main['model_configured'] = str(model_ref)
        kwargs_ultralytics['model'] = str(resolve_model_path(model_ref, logger))
        kwargs_main['class_names'], kwargs_main['class_names_source'] = resolve_class_names(
            Path(kwargs_ultralytics['model']),
            getattr(args, 'class_names', None),
            extraction_cfg.get('class_rename'),
            kwargs_ultralytics.get('classes'),
            logger,
        )
        active = kwargs_tracker.get('active')
        kwargs_main['tracker_active'] = active
        kwargs_main['tracker_params'] = kwargs_tracker.get(active, {}) if active else {}
    else:
        kwargs_main['class_names'] = {}
        kwargs_main['class_names_source'] = None
        kwargs_main['model_configured'] = None
        kwargs_main['tracker_active'] = None
        kwargs_main['tracker_params'] = {}

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


def load_class_names_from_model(model_path: Path, logger: logging.Logger) -> Optional[dict]:
    """Load the class-id -> name mapping embedded in a YOLO model file.

    Returns ``None`` when the names cannot be obtained (ultralytics missing or the model fails to
    load), letting the caller fall back to a config/CLI mapping or integer labels.
    """
    if YOLO is None:
        logger.error("ultralytics is not installed; cannot load class names from model.")
        return None
    try:
        names = YOLO(str(model_path)).names
        logger.info(f"Class names loaded from model: '{model_path}'.")
        return names
    except Exception as e:
        logger.error(f"Failed to load class names from '{model_path}': {e}.")
        return None


def _load_class_names_mapping(value: Union[str, Path, dict, list], logger: logging.Logger) -> Optional[dict]:
    """Coerce a class-names override into a ``{int: str}`` mapping.

    Accepts an inline ``dict`` (from the config), a path to a ``.yaml``/``.json`` mapping file, or a
    list of ``ID=NAME`` tokens (from the CLI, e.g. ``['0=car', '1=bus']``). Returns ``None`` on failure.
    """
    mapping = None
    if isinstance(value, dict):
        mapping = value
    elif isinstance(value, list):  # CLI ID=NAME pairs, or a single-element [path]
        if len(value) == 1 and Path(value[0]).is_file():
            return _load_class_names_mapping(value[0], logger)
        mapping = {}
        for token in value:
            if '=' not in token:
                logger.error(f"Invalid --class-names entry '{token}'. Expected ID=NAME (e.g. 0=car) or a file path.")
                return None
            key, name = token.split('=', 1)
            mapping[key] = name
    else:  # str/Path file
        path = Path(value)
        if not path.is_file():
            logger.error(f"Class names file '{path}' not found.")
            return None
        try:
            with open(path, 'r') as f:
                mapping = json.load(f) if path.suffix.lower() == '.json' else yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read class names from '{path}': {e}.")
            return None
    if not isinstance(mapping, dict) or not mapping:
        logger.error(f"Class names override '{value}' did not yield a non-empty mapping.")
        return None
    try:
        return {int(k): str(v) for k, v in mapping.items()}
    except (ValueError, TypeError) as e:
        logger.error(f"Class names override '{value}' has non-integer keys: {e}.")
        return None


def resolve_class_names(
    model_path: Path,
    cli_value: Optional[Union[list, str]],
    cfg_value: Optional[Union[dict, str]],
    classes: Optional[list],
    logger: logging.Logger,
) -> tuple:
    """Resolve the class-id -> name mapping by precedence: CLI > config > model > integer fallback.

    The CLI (``--class-names``) and config (``class_names:``) overrides accept an inline mapping, a
    ``.yaml``/``.json`` file path, or ``ID=NAME`` pairs. When none of CLI, config, or the model yields a
    mapping, integer labels (``{id: str(id)}``) are used over the configured ``classes`` ids (or
    ``range(100)``) and a warning is logged.

    Returns a ``(mapping, source_label)`` tuple where ``source_label`` is one of
    ``'cli'``, ``'config'``, ``'model'``, or ``'fallback'``.
    """
    for source_label, log_tag, value in (
        ('cli', '--class-names', cli_value),
        ('config', 'config class_names', cfg_value),
    ):
        if value is not None:
            mapping = _load_class_names_mapping(value, logger)
            if mapping is not None:
                logger.info(f"Class names taken from {log_tag}: {mapping}.")
                return mapping, source_label

    model_names = load_class_names_from_model(model_path, logger)
    if model_names:
        return model_names, 'model'

    ids = classes if classes else range(100)
    logger.warning(
        "No class-name mapping found (CLI, config, or model); falling back to integer class IDs. "
        "Provide one via cfg -> class_names or --class-names to label classes."
    )
    return {int(i): str(int(i)) for i in ids}, 'fallback'
