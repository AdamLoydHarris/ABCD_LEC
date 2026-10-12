"""Restore the NumPy 1.x aliases that NumPy 2 removed, for Kilosort 4.0.18.

Written 2026-09-28. The `maze_ephys` env was upgraded to numpy 2.0.2 some time after the
August 2025 sorts, and kilosort 4.0.18 predates numpy 2. Two separate breakages, both of
which made every sort in this environment fail -- mEC and LEC alike, old blocks included:

1. `probeinterface.write_prb` formats channel ids with repr, and numpy 2 reprs a scalar as
   "np.int64(0)" instead of "0". kilosort's `io.load_probe` reads that file back with
   `exec(contents, {}, metadata)` -- empty globals -- so it dies with
   `NameError: name 'np' is not defined` within minutes. Fixed below by
   `np.set_printoptions(legacy="1.25")` on numpy 2 only; formatting, never arithmetic.
2. `kilosort/swarmsplitter.py:96` assigns `score = np.NaN`, removed in numpy 2. This one is
   reached inside clustering, 10-24 h into a 6-day sort, which is how it cost five jobs
   (3660041-45, 2026-09-23/24) a full day each before showing itself.

This module restores every removed alias that appears anywhere in the installed kilosort,
spikeinterface or probeinterface, not only the one observed, so the next one does not cost
another day. Every entry is a documented 1:1 rename from the numpy 2 migration guide -- the
objects are identical, so nothing numerical changes; `np.NaN is np.nan` was always true.

Import it before running a sorter:

    import numpy2_compat  # noqa: F401  (restores np.NaN etc. for kilosort 4.0.18)

Checked on numpy 2.0.2: of these, only `in1d`, `row_stack` and `trapz` still existed.
"""
from __future__ import annotations

import numpy as np

#: removed name -> the object it was always an alias for
_ALIASES = {
    "NaN": np.nan, "NAN": np.nan,
    "Inf": np.inf, "Infinity": np.inf, "infty": np.inf, "PINF": np.inf,
    "NINF": -np.inf,
    "float_": np.float64, "complex_": np.complex128,
    "unicode_": np.str_, "string_": np.bytes_,
    "int0": np.intp, "uint0": np.uintp, "bool8": np.bool_,
    "product": np.prod, "cumproduct": np.cumprod,
    "alltrue": np.all, "sometrue": np.any,
    "round_": np.round,
}


def apply() -> list[str]:
    """Add back any missing alias. Returns the names restored, for the job log."""
    restored = []
    for name, obj in _ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, obj)
            restored.append(name)
    # kilosort's io.load_probe execs a probe file written with repr(); numpy 2's scalar repr
    # ("np.int64(0)") is not valid in that empty-globals exec. Formatting only.
    # numpy 2 ONLY: on 1.26.4 "1.25" is not a legacy mode, gets stored as a raw string, and
    # the first float format (h5py's import) dies on `'1.25' <= 113` (smoke job 3665856).
    if int(np.__version__.split(".")[0]) >= 2:
        np.set_printoptions(legacy="1.25")
    return restored


RESTORED = apply()
