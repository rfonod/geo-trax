# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests for the 'geotrax' umbrella command-line interface."""

import subprocess
import sys

import pytest

from geotrax.cli import COMMANDS, build_usage


def run_cli(*args):
    """Run 'python -m geotrax' with the given arguments and return the completed process."""
    return subprocess.run(
        [sys.executable, '-m', 'geotrax', *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_all_commands():
    result = run_cli('--help')
    assert result.returncode == 0
    for command in COMMANDS:
        assert command in result.stdout


def test_no_arguments_prints_usage():
    result = run_cli()
    assert result.returncode == 0
    assert 'usage: geotrax' in result.stdout


def test_version():
    result = run_cli('--version')
    assert result.returncode == 0
    assert result.stdout.startswith('geo-trax ')


def test_unknown_command_fails():
    result = run_cli('frobnicate')
    assert result.returncode == 2
    assert "unknown command 'frobnicate'" in result.stderr


@pytest.mark.parametrize('command', list(COMMANDS))
def test_subcommand_help(command):
    result = run_cli(command, '--help')
    assert result.returncode == 0
    assert 'usage:' in result.stdout


def test_subcommand_without_required_args_fails():
    result = run_cli('visualize')
    assert result.returncode == 2
    assert 'required' in result.stderr or 'arguments' in result.stderr


def test_build_usage_contains_all_commands():
    usage = build_usage()
    assert isinstance(usage, str)
    for name in COMMANDS:
        assert name in usage
