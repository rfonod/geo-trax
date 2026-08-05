# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Shared pytest fixtures."""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_update_checks():
    """
    Keep the suite hermetic: never let a PyPI update check reach the network.

    Silences both geo-trax's own check (fired from setup_logger) and stabilo's (fired from
    Stabilizer.__init__). setdefault, not assignment, so test_version_check.py can still
    delete the variable via monkeypatch and exercise the real code path. Session-scoped and
    autouse so it is set before test_cli.py spawns its subprocesses, which inherit the env.
    """
    os.environ.setdefault("GEOTRAX_DISABLE_UPDATE_CHECK", "1")
    os.environ.setdefault("STABILO_DISABLE_UPDATE_CHECK", "1")
