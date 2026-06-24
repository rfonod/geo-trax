# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for visualization helpers."""

import logging

import numpy as np
import pytest

from geotrax.visualize import normalize_viz_modes, read_transforms

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    'given, expected',
    [
        (0, [0]),                  # scalar from config
        ([1], [1]),                # single-element list from CLI
        ([0, 1, 2], [0, 1, 2]),    # multiple modes
        ((2, 0), [2, 0]),          # tuple, order preserved
        ([0, 0, 1], [0, 1]),       # duplicates removed
    ],
)
def test_normalize_viz_modes(given, expected):
    assert normalize_viz_modes(given, logger) == expected


@pytest.mark.parametrize('given', [3, -1, [0, 3], 'all'])
def test_normalize_viz_modes_invalid_exits(given):
    with pytest.raises(SystemExit):
        normalize_viz_modes(given, logger)


# --- read_transforms ---------------------------------------------------------

def test_read_transforms_none_returns_none():
    assert read_transforms(None, logger) is None


def test_read_transforms_valid_file(tmp_path):
    # Two frames with identity matrices (det=1>0), space-delimited, 10 columns each.
    content = '0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n1 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n'
    filepath = tmp_path / 'transforms.txt'
    filepath.write_text(content)
    result = read_transforms(filepath, logger)
    assert set(result.keys()) == {0, 1}
    np.testing.assert_allclose(result[0], np.eye(3))
    np.testing.assert_allclose(result[1], np.eye(3))


def test_read_transforms_wrong_column_count_exits(tmp_path):
    content = '0 1.0 0.0 0.0\n'
    filepath = tmp_path / 'bad.txt'
    filepath.write_text(content)
    with pytest.raises(SystemExit):
        read_transforms(filepath, logger)


def test_read_transforms_singular_matrix_exits(tmp_path):
    # All-zero matrix → det=0 → invalid → SystemExit.
    content = '0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n'
    filepath = tmp_path / 'singular.txt'
    filepath.write_text(content)
    with pytest.raises(SystemExit):
        read_transforms(filepath, logger)


def test_read_transforms_non_consecutive_frames_warns(tmp_path, caplog):
    # Frames 0 and 2 (frame 1 missing) → warning logged.
    content = '0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n2 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n'
    filepath = tmp_path / 'gap.txt'
    filepath.write_text(content)
    with caplog.at_level(logging.WARNING):
        result = read_transforms(filepath, logger)
    assert set(result.keys()) == {0, 2}
    assert any('Missing frame' in r.message for r in caplog.records)
