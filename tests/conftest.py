"""Test configuration for metagen-dsl.

Adds parent dir to sys.path so the in-tree metagen_dsl package is importable
without installation. Also makes legacy `from metagen import *` in fixture
code.py files resolve to metagen_dsl so we don't need to mutate those files.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Sibling-repo build dirs for the optional native dependencies. This lets
# `pytest` work in a development checkout where metagen-kernel and
# metagen-simulator are sibling submodules with built extensions in
# ./build/, without requiring `pip install -e` of those repos.
for _sibling in ('metagen-kernel', 'metagen-simulator'):
    _build_dir = REPO_ROOT.parent / _sibling / 'build'
    if _build_dir.is_dir() and str(_build_dir) not in sys.path:
        sys.path.insert(0, str(_build_dir))

import metagen_dsl  # noqa: E402
sys.modules.setdefault('metagen', metagen_dsl)

from . import fixture_data as fd  # noqa: E402


def pytest_addoption(parser):
    parser.addoption('--no-skip-default', action='store_true',
                     help='Include cases marked skip_default '
                          '(known-buggy TPMS+Conjugation cases).')


def pytest_configure(config):
    config.addinivalue_line('markers', 'tier1: fast cases (<5s mean trial time)')
    config.addinivalue_line('markers', 'tier2: medium cases (<60s)')
    config.addinivalue_line('markers', 'tier3: slow cases (>60s)')
    config.addinivalue_line('markers',
        'skip_default: cases default-flagged as having geometric '
        'inconsistency; not run unless --no-skip-default is passed')
    config.addinivalue_line('markers',
        'pyrender: needs the optional metagen-dsl[pyrender] extra')


def pytest_collection_modifyitems(config, items):
    """Tag (case, res)-parametrized tests with tier and skip_default markers,
    then apply the actual skip if --no-skip-default wasn't given."""
    fixtures_ready = fd.fixtures_initialized()
    include_default = config.getoption('--no-skip-default')
    skip_marker = pytest.mark.skip(
        reason='broken geometry; pass --no-skip-default to include')

    for item in items:
        callspec = getattr(item, 'callspec', None)
        if callspec is not None and fixtures_ready:
            case = callspec.params.get('case')
            res = callspec.params.get('resolution')
            if case and res:
                try:
                    meta = fd.load_meta(case, int(res))
                except FileNotFoundError:
                    meta = None
                if meta:
                    if meta.get('skip_by_default'):
                        item.add_marker(pytest.mark.skip_default)
                    item.add_marker(getattr(pytest.mark, fd.runtime_tier(case, int(res))))

        if not include_default and 'skip_default' in item.keywords:
            item.add_marker(skip_marker)
