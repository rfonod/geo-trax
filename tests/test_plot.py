# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the pure label and filtering helpers in plot.py."""

import pandas as pd
import pytest

from geotrax.plot import AGG_STEM_MAX_LEN, aggregated_stem, filter_classes, get_xlabel, get_ylabel, merge_coordinates


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


# --- aggregation helpers -----------------------------------------------------

def test_aggregated_stem_joins_stems():
    assert aggregated_stem('A', ['2022-10-07_A_AM1', '2022-10-07_A_PM1']) == 'agg_2022-10-07_A_AM1_2022-10-07_A_PM1'


def test_aggregated_stem_falls_back_when_too_long():
    stems = [f'2022-10-04_A_AM{i}_D3' for i in range(20)]
    stem = aggregated_stem('A', stems)
    assert stem == 'agg_A_20_files'
    assert len(stem) <= AGG_STEM_MAX_LEN


def test_merge_coordinates_first_file_without_results():
    img = {'Unstabilized image coordinates': ['X_unstabilized', 'Y_unstabilized']}
    assert merge_coordinates(None, img) == img
    assert merge_coordinates(img, None) == img


def test_merge_coordinates_keeps_only_shared_systems():
    unstab = {'Unstabilized image coordinates': ['X_unstabilized', 'Y_unstabilized']}
    both = {**unstab, 'Stabilized image coordinates': ['X_stabilized', 'Y_stabilized']}
    assert merge_coordinates(both, unstab) == unstab
