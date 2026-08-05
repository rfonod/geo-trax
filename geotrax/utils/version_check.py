# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
version_check.py - Non-intrusive check for a newer geo-trax release on PyPI.

The check is best-effort: it never raises, is silent on any failure or when offline,
caches the last result for 24 hours, and can be disabled via the
GEOTRAX_DISABLE_UPDATE_CHECK environment variable.

Invoked once per process from setup_logger() (logging_utils.py), so every pipeline
stage and tools/ script performs it with a fully configured logger. Note that stabilo
runs an equivalent check of its own from Stabilizer.__init__; it is silenced separately,
via STABILO_DISABLE_UPDATE_CHECK.
"""

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from geotrax.utils.constants import MACOS, WINDOWS

PYPI_URL = "https://pypi.org/pypi/geo-trax/json"
CACHE_TTL = 24 * 3600
FETCH_TIMEOUT = 2.0
ENV_OPT_OUT = "GEOTRAX_DISABLE_UPDATE_CHECK"

_state = {'checked': False}
_lock = threading.Lock()


def _opted_out() -> bool:
    return bool(os.environ.get(ENV_OPT_OUT))


def _cache_dir() -> Path:
    """Return the platform-native cache directory for geo-trax (sibling of default_log_dir)."""
    if WINDOWS:  # %LOCALAPPDATA%\geo-trax\Cache
        base = Path(os.environ.get('LOCALAPPDATA') or (Path.home() / 'AppData' / 'Local'))
        return base / 'geo-trax' / 'Cache'
    if MACOS:  # ~/Library/Caches/geo-trax
        return Path.home() / 'Library' / 'Caches' / 'geo-trax'
    # Linux and other Unix (XDG base directory spec): ~/.cache/geo-trax
    base = Path(os.environ.get('XDG_CACHE_HOME') or (Path.home() / '.cache'))
    return base / 'geo-trax'


def _cache_file() -> Path:
    return _cache_dir() / "update_check.json"


def _read_cache():
    try:
        with open(_cache_file(), "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(latest: str) -> None:
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        with open(_cache_file(), "w") as f:
            json.dump({"last_check": time.time(), "latest_version": latest}, f)
    except Exception:
        pass


def _parse_version(version: str):
    """Parse a version string into a comparable tuple of ints; never raises."""
    parts = []
    for chunk in str(version).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _fetch_latest() -> str:
    with urllib.request.urlopen(PYPI_URL, timeout=FETCH_TIMEOUT) as response:
        data = json.load(response)
    return data["info"]["version"]


def _notify_if_newer(latest: str, logger) -> None:
    # Deliberately local: geotrax/__init__ -> ... -> logging_utils -> this module, so a
    # top-level import of __version__ would be a cycle.
    from geotrax import __version__  # noqa: PLC0415

    if latest and _parse_version(latest) > _parse_version(__version__):
        message = (
            f"A newer geo-trax version ({latest}) is available on PyPI (installed: {__version__}). "
            f"Upgrade with: pip install -U geo-trax (set {ENV_OPT_OUT}=1 to silence)."
        )
        if logger is not None:
            logger.warning(message)


def _do_check(logger) -> None:
    try:
        cache = _read_cache()
        if cache and (time.time() - cache.get("last_check", 0)) < CACHE_TTL:
            latest = cache.get("latest_version")
        else:
            latest = _fetch_latest()
            _write_cache(latest)
        _notify_if_newer(latest, logger)
    except Exception:
        pass


def check_for_updates(logger=None, blocking: bool = False) -> None:
    """
    Warn (once) if a newer geo-trax release is available on PyPI. Never raises.

    When the cache is fresh the comparison is done in-process; otherwise the network
    fetch runs in a daemon thread unless `blocking` is True.
    """
    if _opted_out():
        return
    if blocking:
        _do_check(logger)
    else:
        threading.Thread(target=_do_check, args=(logger,), daemon=True).start()


def check_for_updates_once(logger=None) -> None:
    """Run check_for_updates at most once per process."""
    with _lock:
        if _state['checked']:
            return
        _state['checked'] = True
    check_for_updates(logger)
