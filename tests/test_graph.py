"""Verify code.py → graph.json compilation matches the committed reference.

This is the original 50/50 regression test that we already use to validate
that the DSL extraction preserved upstream behavior.
"""

import json

import pytest

from . import fixture_data as fd


def _all_cases():
    """Every case in the fixtures repo, irrespective of skip-by-default
    flags — graph.json compilation is unrelated to the geometry/sim
    instabilities those flags address."""
    if not fd.fixtures_initialized():
        return []
    return [c['name'] for c in fd.manifest()['cases']]


@pytest.mark.tier1
@pytest.mark.parametrize('case', _all_cases())
def test_graph_json_matches_reference(case):
    code_py = fd.load_code_py(case)
    program = (
        'from metagen_dsl import *\n'
        f'{code_py}\n'
        'output = ProcMetaTranslator(make_structure())'
    )
    env = {}
    exec(program, env)
    produced = json.loads(env['output'].to_json())
    expected = json.loads(fd.load_graph_json(case))
    assert produced == expected, f'graph.json mismatch for {case}'
