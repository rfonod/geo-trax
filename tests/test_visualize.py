# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for visualization helpers."""

import logging

import pytest

from geotrax.visualize import normalize_viz_modes

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
