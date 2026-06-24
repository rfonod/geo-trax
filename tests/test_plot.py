# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure label and filtering helpers in plot.py."""

import pandas as pd
import pytest

from geotrax.plot import filter_classes, get_xlabel, get_ylabel


@pytest.mark.parametrize(
    'key, expected',
    [
        ('X_stabilized', 'X stabilized [px]'),
        ('Ortho_X', 'Ortho X [px]'),
        ('Longitude', 'Longitude [deg]'),
        ('Local_X', 'Local X [m]'),
    ],
)
def test_get_xlabel(key, expected):
    assert get_xlabel(key) == expected


@pytest.mark.parametrize(
    'key, expected',
    [
        ('Y_unstabilized', 'Y unstabilized [px]'),
        ('Ortho_Y', 'Ortho Y [px]'),
        ('Latitude', 'Latitude [deg]'),
        ('Local_Y', 'Local Y [m]'),
    ],
)
def test_get_ylabel(key, expected):
    assert get_ylabel(key) == expected


def test_filter_classes_excludes_listed():
    df = pd.DataFrame({'Vehicle_Class': [0, 1, 2, 3], 'v': [10, 20, 30, 40]})
    result = filter_classes(df, [1, 2])
    assert sorted(result['Vehicle_Class']) == [0, 3]


def test_filter_classes_accepts_string_ids():
    df = pd.DataFrame({'Vehicle_Class': [0, 1, 2], 'v': [1, 2, 3]})
    result = filter_classes(df, ['1'])
    assert sorted(result['Vehicle_Class']) == [0, 2]


def test_filter_classes_empty_filter_is_noop():
    df = pd.DataFrame({'Vehicle_Class': [0, 1, 2]})
    result = filter_classes(df, [])
    assert len(result) == 3
