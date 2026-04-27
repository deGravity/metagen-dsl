"""Global display options (pandas-style).

Controls what Structure._repr_mimebundle_ computes and shows in Jupyter,
what resolution/backend default for explicit method calls, and how many
cached results each Structure keeps.
"""

from contextlib import contextmanager


_DEFAULTS = {
    'display.resolution': 65,            # for _repr_mimebundle_
    'display.resolution_direct': 97,     # for explicit .geometry()/.simulate()
    'display.backend': 'auto',           # 'auto' | 'gpu' | 'cpu'
    'display.show': 'auto',              # 'auto' | 'render' | 'sim' | 'interactive' | 'all' | 'none'
    'display.simulate_in_repr': 'auto',  # 'auto' (if GPU) | True | False
    'display.render_size': (400, 400),
    'display.render_views': ('top_right', 'top', 'front', 'right'),
    'cache.max_entries': 4,
    # Selects which optimizer the kernel uses inside the conjugate-TPMS solve.
    #   'current'      — BOBYQA local refinement only. Fast (~seconds for typical
    #                    cases); non-deterministic for TPMS+Conjugation patterns
    #                    on prism/cuboid bounding volumes.
    #   'global'       — Adds a 500-eval ESCH evolution-strategy phase. Effectively
    #                    deterministic; substantially slower (often 10× or more).
    #   'experimental' — Reserved for ongoing stable-but-fast algorithm work.
    #                    Currently identical to 'current'.
    'tpms.optimizer_mode': 'current',
}

_options = dict(_DEFAULTS)


def _check_key(name):
    if name not in _DEFAULTS:
        raise KeyError(f"unknown option {name!r}. Known: {sorted(_DEFAULTS)}")


def set_option(name, value):
    _check_key(name)
    _options[name] = value


def get_option(name):
    _check_key(name)
    return _options[name]


def reset_option(name):
    _check_key(name)
    _options[name] = _DEFAULTS[name]


@contextmanager
def option_context(*args):
    """Temporarily set options. Usage:
        with option_context('display.show', 'sim', 'display.resolution', 33):
            display(s)
    """
    if len(args) % 2 != 0:
        raise ValueError("option_context takes pairs of (name, value)")
    pairs = list(zip(args[::2], args[1::2]))
    for name, _ in pairs:
        _check_key(name)
    saved = {name: _options[name] for name, _ in pairs}
    try:
        for name, value in pairs:
            _options[name] = value
        yield
    finally:
        _options.update(saved)
