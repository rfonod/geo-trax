# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Tests that every third-party package the code imports is declared in pyproject.toml."""

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / 'pyproject.toml'
PACKAGE_DIR = REPO_ROOT / 'geotrax'

IMPORT_TO_DISTRIBUTION = {
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'yaml': 'pyyaml',
}
"""Top-level import names whose distribution name differs from the import name."""

OPTIONAL_IMPORTS = {
    'sahi': 'sahi',
}
"""Imports that may come from an extra instead of the core dependencies (import name -> extra)."""


def normalize(name: str) -> str:
    """Normalize a distribution name as PEP 503 does."""
    return re.sub(r'[-_.]+', '-', name).lower()


def requirement_name(requirement: str) -> str:
    """Return the normalized distribution name of a PEP 508 requirement string."""
    return normalize(re.match(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*)', requirement).group(1))


def third_party_imports() -> dict[str, set[str]]:
    """Map each top-level third-party module imported by the package to the files importing it.

    Function-level (lazy) imports count too, since a missing package fails them just the same.
    """
    imports: dict[str, set[str]] = {}
    for path in PACKAGE_DIR.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split('.')[0]
                if top not in sys.stdlib_module_names and top != 'geotrax':
                    imports.setdefault(top, set()).add(str(path.relative_to(REPO_ROOT)))
    return imports


@pytest.fixture(scope='module')
def project():
    if not PYPROJECT.is_file():
        pytest.skip('pyproject.toml is only available in a source checkout')
    return tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['project']


def test_scan_finds_imports():
    assert {'numpy', 'yaml', 'ultralytics', 'stabilo'} <= set(third_party_imports())


def test_every_import_is_declared(project):
    declared = {requirement_name(req) for req in project['dependencies']}
    extras = {
        extra: {requirement_name(req) for req in reqs} for extra, reqs in project['optional-dependencies'].items()
    }
    missing = []
    for module, files in sorted(third_party_imports().items()):
        distribution = normalize(IMPORT_TO_DISTRIBUTION.get(module, module))
        if module in OPTIONAL_IMPORTS:
            found = distribution in extras.get(OPTIONAL_IMPORTS[module], set())
        else:
            found = distribution in declared
        if not found:
            missing.append(f'{module} ({distribution}) imported by {", ".join(sorted(files))}')
    assert not missing, 'undeclared dependencies:\n' + '\n'.join(missing)
