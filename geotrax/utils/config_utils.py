# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""YAML configuration loading and path resolution for the geo-trax pipeline."""

import argparse
import atexit
import difflib
import functools
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional, Union

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

    A path with no final component ('', '.', '..') gets no suffix appended, since with_suffix
    rejects an empty name; it falls through unchanged and load_config reports it.
    """
    path = Path(cfg_filepath)
    if not path.suffix and path.name:
        path = path.with_suffix('.yaml')

    candidates = [path]
    if not path.is_absolute():
        bundled = Path(*path.parts[1:]) if path.parts[:1] == ('cfg',) else path
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

    Both override mechanisms are resolved here, before the config is split into sections, so that
    every consumer downstream (and the saved run metadata) sees the effective configuration:
    ``--set`` first, then the dedicated CLI flags via :func:`sync_args_with_config`.
    """
    full = load_config(args.cfg, logger, args)
    sync_args_with_config(args, full, logger)

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
    The file must outlive this call (Ultralytics reads it later), so it cannot be removed here;
    it is unlinked at interpreter exit instead, which keeps a long 'batch' run from leaving one
    stray file per video behind.
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
            tracker_path = Path(tmp.name)
    except OSError as exc:
        logger.critical(f"Failed to write temporary tracker config: {exc}")
        sys.exit(1)

    atexit.register(_unlink_quietly, tracker_path)
    return tracker_path


def _unlink_quietly(path: Path) -> None:
    """Remove *path*, ignoring an already-removed file or an unwritable temp directory."""
    try:
        path.unlink()
    except OSError:
        pass


def load_config(cfg_filepath: Union[str, Path], logger: logging.Logger,
                args: Optional[argparse.Namespace] = None) -> dict:
    """Load a configuration file and return the contents as a dictionary.

    When *args* is given, its ``--set KEY=VALUE`` overrides are applied to the loaded config
    (see :func:`apply_cli_overrides`). Callers without a ``--set`` flag can omit it.

    Every way the file can fail to yield a config mapping is reported as a fatal, named error:
    a hand-edited config that is unreadable, malformed or empty would otherwise surface much
    later as an opaque traceback from whichever stage first indexed the missing section.
    """
    resolved_filepath = resolve_config_path(cfg_filepath)
    try:
        with open(resolved_filepath, 'r') as f:
            kwargs = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"Configuration file '{cfg_filepath}' not found.")
        sys.exit(1)
    except OSError as exc:
        logger.critical(f"Configuration file '{resolved_filepath}' could not be read: {exc}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        logger.critical(f"Configuration file '{resolved_filepath}' is not valid YAML: {exc}")
        sys.exit(1)

    if not isinstance(kwargs, dict):
        found = 'nothing' if kwargs is None else f'a {type(kwargs).__name__}'
        logger.critical(
            f"Configuration file '{resolved_filepath}' does not contain a top-level mapping of "
            f"sections (found {found}). Run 'geotrax config copy' for a valid starting point."
        )
        sys.exit(1)

    if args is not None:
        apply_cli_overrides(kwargs, getattr(args, 'set', None), args, logger)
    return kwargs


def backfill_args_from_config(args: argparse.Namespace, mapping: dict) -> None:
    """Set each ``args.arg_name`` from ``mapping[arg_name]`` when the arg is still ``None``
    (i.e. not overridden on the command line).

    Retained for the ``tools/`` scripts. The pipeline stages use :func:`sync_args_with_config`,
    which derives the same mapping from the parser's ``_cfg_paths`` and also writes CLI values
    back into the config so the saved run metadata records what actually ran.
    """
    for arg_name, config_value in mapping.items():
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, config_value)


# --- CLI <-> config plumbing -------------------------------------------------------------------

_MISSING = object()  # sentinel: distinguishes "key absent from the config" from "key set to null"


def _iter_leaf_paths(cfg: dict, prefix: str = '') -> Iterator[str]:
    """Yield the dotted path of every leaf in *cfg*; a leaf is any non-dict value (``null`` included)."""
    for key, value in (cfg or {}).items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            yield from _iter_leaf_paths(value, f'{path}.')
        else:
            yield path


def _resolve_key(node: dict, key: str) -> Any:
    """Return the key of *node* that the dotted-path component *key* names, or ``_MISSING``.

    Dotted paths are strings, but one config block is keyed by integer class id
    (``extraction -> dimension_estimation -> tau_c``), and ``_iter_leaf_paths`` renders those
    keys through an f-string. Matching on the string form keeps the three path helpers agreeing
    on the same key, so an override of an integer-keyed leaf reaches the value that the pipeline
    actually reads instead of writing a string-keyed duplicate beside it.
    """
    if key in node:
        return key
    for candidate in node:
        if not isinstance(candidate, str) and str(candidate) == key:
            return candidate
    return _MISSING


def _get_by_path(cfg: dict, path: str, default: Any = _MISSING) -> Any:
    """Return the value at the dotted *path*, or *default* if any component is absent."""
    node = cfg
    for key in path.split('.'):
        if not isinstance(node, dict):
            return default
        actual = _resolve_key(node, key)
        if actual is _MISSING:
            return default
        node = node[actual]
    return node


@functools.lru_cache(maxsize=1)
def _bundled_defaults() -> dict:
    """The shipped default.yaml, used as a fallback for a key absent from a loaded custom config."""
    with open(CFG_DIR / 'default.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_bundled_defaults(section: Optional[dict], path: str) -> dict:
    """Return *section* with every key missing from it filled in from the bundled default.yaml.

    ``sync_args_with_config`` already gives that fallback to each registered CLI flag, so a
    custom config predating a new key still loads with a sane value. A block whose keys are
    config-only has no registered dest and so is skipped by it, and would reach its consumer
    half-populated. Use this wherever such a block is read as a whole.
    """
    defaults = _get_by_path(_bundled_defaults(), path, default={}) or {}
    return {**defaults, **(section or {})}


def _set_by_path(cfg: dict, path: str, value: Any) -> None:
    """Set the value at the dotted *path*, creating intermediate dicts as needed."""
    *parents, leaf = path.split('.')
    node = cfg
    for key in parents:
        actual = _resolve_key(node, key)
        child = node[actual] if actual is not _MISSING else None
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    actual = _resolve_key(node, leaf)
    node[leaf if actual is _MISSING else actual] = value


def sync_args_with_config(args: argparse.Namespace, cfg: dict, logger: logging.Logger) -> None:
    """Reconcile the parsed CLI arguments with the pipeline config, in both directions.

    Every config-backed flag defaults to ``None`` (see ``cli_utils.add_cfg_arg``), so for each
    ``dest -> CfgArg`` entry the parser recorded in ``args._cfg_paths``:

      * the argument is ``None`` -> pull the config value into it (the historical backfill), or
      * the argument was given  -> push it into the config.

    The second direction is what makes the config dict authoritative: everything downstream, the
    saved run-metadata YAML included, then sees the configuration the run actually used rather
    than the file's version of it. Flags marked ``no_sync`` are skipped in both directions; their
    precedence is resolved elsewhere (see :class:`~geotrax.utils.cli_utils.CfgArg`).

    Config keys absent from the file fall back to the bundled default.yaml, so a custom config
    predating a newly added key still loads with a sane value instead of leaving the argument
    (and every downstream consumer of it) silently at ``None``.

    A dest listed in ``args._cfg_pinned`` is always pushed, never pulled, even when it is
    ``None``. That is how a caller suppresses a config key outright: 'batch' clears
    ``cut_frame_right`` in directory mode, and without the pin the very next stage would read the
    value straight back out of the config it just re-loaded.
    """
    pinned = getattr(args, '_cfg_pinned', ())
    for dest, spec in getattr(args, '_cfg_paths', {}).items():
        if spec.no_sync or not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        if value is None and dest not in pinned:
            config_value = _get_by_path(cfg, spec.path)
            if config_value is _MISSING:
                config_value = _get_by_path(_bundled_defaults(), spec.path)
                if config_value is _MISSING:
                    continue
            if spec.invert:
                config_value = not config_value
            elif spec.coerce is not None and config_value is not None:
                config_value = spec.coerce(config_value)
            setattr(args, dest, config_value)
        else:
            _set_by_path(cfg, spec.path, not value if spec.invert and value is not None else value)
            if dest in pinned:  # the caller suppressed the key; it did not come from the CLI
                continue
            provided = getattr(args, '_cli_provided', None)
            if provided is None or dest in provided:  # a pulled value pushed back is not news
                logger.info(f"CLI argument applied to the configuration: {spec.path} = {value}.")


def apply_cli_overrides(cfg: dict, overrides: Optional[list], args: Optional[argparse.Namespace],
                        logger: logging.Logger) -> None:
    """Apply ``--set KEY=VALUE`` overrides to the loaded config, in place.

    KEY is matched against the config that was actually loaded, so a custom config with extra
    keys works without any change here. It may be a full dotted path ('ultralytics.conf') or any
    unambiguous tail of one ('conf', 'matching.gpu'); an unknown or ambiguous key aborts the run
    rather than being silently dropped, since an override that quietly does nothing is worse than
    one that stops.

    VALUE is parsed with YAML rules, matching the config file it overrides, and is rejected if
    its type is incompatible with the value already in place.

    When *args* carries a ``_cfg_paths`` map, a key that is also targeted by a dedicated flag on
    the same command line is a conflict and aborts. Nothing is written until every override has
    been validated, so a rejected command line leaves the config untouched.
    """
    if not overrides:
        return

    leaf_paths = list(_iter_leaf_paths(cfg))
    flag_for_path = _dedicated_flags_in_use(args)
    resolved = []

    for token in overrides:
        if '=' not in token:
            logger.critical(f"Invalid --set entry '{token}'. Expected KEY=VALUE (e.g. --set conf=0.35).")
            sys.exit(1)
        key, raw_value = token.split('=', 1)
        key = key.strip()
        path = _resolve_override_key(key, leaf_paths, cfg, logger)

        if path in flag_for_path:
            logger.critical(
                f"Conflicting overrides for '{path}':\n"
                f"  --set {key}={raw_value}\n"
                f"  {flag_for_path[path]}\n"
                "Pass only one."
            )
            sys.exit(1)

        if raw_value == '':
            value = ''  # '--set tracks_postfix=' clears a string; YAML would read it as null
        else:
            try:
                value = yaml.safe_load(raw_value)
            except yaml.YAMLError:
                value = raw_value  # an unquoted string that is not valid YAML is still a string
        current = _get_by_path(cfg, path)
        value = _coerce_override_value(path, value, current, raw_value, logger)
        resolved.append((path, value, current))

    # The config is re-loaded for every stage and, under 'batch', for every video; announce each
    # override once at NOTICE (visible without --verbose) and drop to INFO on the repeats.
    first_time = not getattr(args, '_overrides_announced', False)
    for path, value, current in resolved:
        _set_by_path(cfg, path, value)
        message = f"CLI override: {path} = {value!r} (was {current!r})."
        logger.notice(message) if first_time else logger.info(message)
    if args is not None:
        args._overrides_announced = True


def _dedicated_flags_in_use(args: Optional[argparse.Namespace]) -> dict:
    """Map each config path to the dedicated flag that set it on this command line, if any.

    Driven by ``_cli_provided`` (recorded at parse time by ``cli_utils.finalize_cli_args``) rather
    than by the current argument values: once ``sync_args_with_config`` has pulled the config into
    the namespace, a value that came from the config file is indistinguishable from a typed one.
    """
    provided = getattr(args, '_cli_provided', None)
    if provided is None:
        return {}
    flags = {}
    for dest, spec in getattr(args, '_cfg_paths', {}).items():
        if dest not in provided:
            continue
        value = getattr(args, dest, None)
        flags[spec.path] = spec.flag if isinstance(value, bool) else f'{spec.flag} {value}'
    return flags


def _resolve_override_key(key: str, leaf_paths: list, cfg: dict, logger: logging.Logger) -> str:
    """Resolve a ``--set`` key to exactly one dotted config path, or abort with a usable message."""
    if key in leaf_paths:
        return key

    segments = key.split('.')
    matches = [p for p in leaf_paths if p.split('.')[-len(segments):] == segments]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        if _get_by_path(cfg, key) is not _MISSING:
            logger.critical(
                f"--set key '{key}' is a config section, not a single value. Set the keys inside it "
                f"individually, e.g. --set {key}.<name>=<value>."
            )
            sys.exit(1)
        # Compare against the leaf names as well as the full paths: a user typing 'cnf' is close
        # to 'conf' but not to 'ultralytics.conf', so path-only matching finds nothing useful.
        by_leaf = {}
        for path in leaf_paths:
            by_leaf.setdefault(path.split('.')[-1], []).append(path)
        suggestions = [p for name in difflib.get_close_matches(segments[-1], by_leaf, n=3, cutoff=0.6)
                       for p in by_leaf[name]]
        suggestions += [p for p in difflib.get_close_matches(key, leaf_paths, n=3, cutoff=0.6)
                        if p not in suggestions]
        suggestions += [p for p in leaf_paths if p.split('.')[-1].startswith(segments[-1]) and p not in suggestions]
        hint = ('\n  Did you mean:\n    ' + '\n    '.join(suggestions[:5])) if suggestions else ''
        logger.critical(f"Unknown --set key '{key}'.{hint}")
        sys.exit(1)

    active_tracker = (cfg.get('tracker') or {}).get('active')
    listed = '\n    '.join(
        f'{p}{"   <- the active tracker" if active_tracker and p == f"tracker.{active_tracker}.{segments[-1]}" else ""}'
        for p in matches
    )
    logger.critical(
        f"--set key '{key}' is ambiguous; it matches {len(matches)} config keys:\n    {listed}\n"
        "  Qualify it with enough of the path to be unique."
    )
    sys.exit(1)


def _coerce_override_value(path: str, value: Any, current: Any, raw_value: str, logger: logging.Logger) -> Any:
    """Check an overridden value against the type already in the config, coercing where unambiguous."""
    if current is None:
        return value  # a null in the config carries no type information

    if value is None:
        if _get_by_path(_bundled_defaults(), path, default=None) is None:
            return value  # the key is documented as nullable in the bundled config
        expected = type(current).__name__
        logger.critical(
            f"--set {path}={raw_value} has the wrong type: expected {expected} (current value: {current!r}), "
            "got NoneType."
        )
        sys.exit(1)

    is_bool, is_number = isinstance(value, bool), isinstance(value, (int, float)) and not isinstance(value, bool)

    if isinstance(current, bool):
        if is_bool:
            return value
        if is_number and value in (0, 1):
            return bool(value)  # 0/1 is a common spelling of a flag
    elif isinstance(current, float):
        if is_number:
            return float(value)
    elif isinstance(current, int):
        # Numbers stay interchangeable: whether a key reads as int or float in the config is often
        # incidental (stationary_speed_cutoff ships as '1', yet 1.5 is a perfectly good value), so
        # rejecting a fractional value here would block valid input. Whole values keep their
        # int-ness; fractional ones are passed through as the user wrote them, exactly as editing
        # the config file by hand would. The check below still catches str/bool/list confusion.
        if is_number:
            return int(value) if float(value).is_integer() else value
    elif isinstance(current, str):
        if isinstance(value, str):
            return value
    elif isinstance(current, list):
        if isinstance(value, list):
            return value
    elif isinstance(value, type(current)):
        return value

    expected = type(current).__name__
    logger.critical(
        f"--set {path}={raw_value} has the wrong type: expected {expected} (current value: {current!r}), "
        f"got {type(value).__name__}."
    )
    sys.exit(1)


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
