# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the best-effort PyPI update check."""

import io
import json
import logging

import pytest

import geotrax.utils.version_check as vc


@pytest.fixture
def cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, '_cache_dir', lambda: tmp_path)
    vc._state['checked'] = False
    monkeypatch.delenv(vc.ENV_OPT_OUT, raising=False)
    yield tmp_path
    vc._state['checked'] = False


class _CaptureLogger:
    def __init__(self):
        self.logger = logging.getLogger('geotrax_test_version_check')
        self.logger.handlers.clear()
        self.messages = []
        parent = self

        class _H(logging.Handler):
            def emit(self, record):
                parent.messages.append(record.getMessage())

        self.logger.addHandler(_H())
        self.logger.setLevel(logging.WARNING)


def _fake_urlopen(version):
    def _open(url, timeout=None):
        return io.BytesIO(json.dumps({'info': {'version': version}}).encode())

    return _open


def test_newer_version_warns(cache_in_tmp, monkeypatch):
    monkeypatch.setattr(vc.urllib.request, 'urlopen', _fake_urlopen('999.0.0'))
    cap = _CaptureLogger()
    vc.check_for_updates(cap.logger, blocking=True)
    assert any('999.0.0' in m for m in cap.messages)


def test_same_version_silent(cache_in_tmp, monkeypatch):
    from geotrax import __version__  # noqa: PLC0415 - read at call time, mirroring _notify_if_newer

    monkeypatch.setattr(vc.urllib.request, 'urlopen', _fake_urlopen(__version__))
    cap = _CaptureLogger()
    vc.check_for_updates(cap.logger, blocking=True)
    assert cap.messages == []


def test_older_version_silent(cache_in_tmp, monkeypatch):
    monkeypatch.setattr(vc.urllib.request, 'urlopen', _fake_urlopen('0.0.1'))
    cap = _CaptureLogger()
    vc.check_for_updates(cap.logger, blocking=True)
    assert cap.messages == []


def test_network_error_silent(cache_in_tmp, monkeypatch):
    def _raise(url, timeout=None):
        raise OSError('offline')

    monkeypatch.setattr(vc.urllib.request, 'urlopen', _raise)
    cap = _CaptureLogger()
    vc.check_for_updates(cap.logger, blocking=True)
    assert cap.messages == []


def test_bad_json_silent(cache_in_tmp, monkeypatch):
    monkeypatch.setattr(vc.urllib.request, 'urlopen', lambda url, timeout=None: io.BytesIO(b'not json'))
    cap = _CaptureLogger()
    vc.check_for_updates(cap.logger, blocking=True)
    assert cap.messages == []


def test_opt_out_skips_network(cache_in_tmp, monkeypatch):
    def _open(url, timeout=None):
        raise AssertionError('should not be called')

    monkeypatch.setattr(vc.urllib.request, 'urlopen', _open)
    monkeypatch.setenv(vc.ENV_OPT_OUT, '1')
    cap = _CaptureLogger()
    vc.check_for_updates(cap.logger, blocking=True)
    assert cap.messages == []


def test_fresh_cache_skips_network(cache_in_tmp, monkeypatch):
    vc._write_cache('999.0.0')
    called = []

    def _open(url, timeout=None):
        called.append(url)
        raise AssertionError('should not be called')

    monkeypatch.setattr(vc.urllib.request, 'urlopen', _open)
    cap = _CaptureLogger()
    vc.check_for_updates(cap.logger, blocking=True)
    assert called == []
    assert any('999.0.0' in m for m in cap.messages)


def test_stale_cache_fetches(cache_in_tmp, monkeypatch):
    (cache_in_tmp / 'update_check.json').write_text(json.dumps({'last_check': 0, 'latest_version': '0.0.1'}))
    monkeypatch.setattr(vc.urllib.request, 'urlopen', _fake_urlopen('999.0.0'))
    cap = _CaptureLogger()
    vc.check_for_updates(cap.logger, blocking=True)
    assert any('999.0.0' in m for m in cap.messages)


def test_check_once(cache_in_tmp, monkeypatch):
    calls = []
    monkeypatch.setattr(vc, 'check_for_updates', lambda logger=None: calls.append(1))
    vc.check_for_updates_once()
    vc.check_for_updates_once()
    assert len(calls) == 1


def test_parse_version_never_raises():
    assert vc._parse_version('1.4.0') == (1, 4, 0)
    assert vc._parse_version('1.4.0.dev1') == (1, 4, 0, 0)
    assert vc._parse_version('2.0.0rc1') == (2, 0, 0)
    assert vc._parse_version('') == (0,)


def test_cache_dir_is_platform_specific(monkeypatch):
    """The cache lives under the platform cache root, never the log root."""
    monkeypatch.setattr(vc, 'WINDOWS', False)
    monkeypatch.setattr(vc, 'MACOS', True)
    assert vc._cache_dir().as_posix().endswith('Library/Caches/geo-trax')

    monkeypatch.setattr(vc, 'MACOS', False)
    monkeypatch.setenv('XDG_CACHE_HOME', '/tmp/xdg-cache')
    assert vc._cache_dir().as_posix() == '/tmp/xdg-cache/geo-trax'

    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)
    assert vc._cache_dir().as_posix().endswith('.cache/geo-trax')
