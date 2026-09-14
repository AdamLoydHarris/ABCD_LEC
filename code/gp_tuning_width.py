"""W5 -- goal-progress tuning: per-leg phase curves, split-half peak vs width, the time-cell
null, the beta route, and sorted beta heatmaps.

STANDALONE by instruction (2026-09-08): this module does not modify `glm_plots.py`,
`pfc_glm_plots.py`, `check_mirror_parity.py` or the V3 notebooks. It only READS the V3 fits
and the data. It is byte-identical in `code/` and `mFC_data/code/` and detects which tree it
is in (`IS_PFC`), like `glm_v3_synthetics.py`; `assert_mirror()` checks the two copies and the
notebooks call it, because `check_mirror_parity.BYTE_PAIRS` was deliberately not extended.

    python code/gp_tuning_width.py --build            # per-recday leg-curve caches for this tree
    python code/gp_tuning_width.py --verify           # LEC: rebuild vs norm_neurons_dic / gp_curves
    python code/gp_tuning_width.py --check-mirror     # the two copies are byte-identical

THE CONFOUND THIS ANALYSIS IS BUILT AROUND (docs/handoff/W5_gp_tuning_width.md section 2).
`goal_progress` is time normalised by leg duration D, and D varies (LEC median 9.2 s, p90
25.6 s). A cell locked to a fixed TIME t0 after reward sits at phase t0/D on each leg: its
phase peak is ~t0/median(D) and its phase WIDTH grows with t0 x spread(1/D). A cell locked to
a fixed time before the next reward mirrors that from the leg's end. So a population of time
cells produces a V-shaped width-vs-peak relation in phase space (narrow at both ends, broad in
the middle), a population of phase cells produces no dependence at all, and a peak-width
correlation is the NULL EXPECTATION under time coding, not evidence of anything. Every real
relation here is read against the curve `simulate_population` produces from each recday's REAL
legs through this same code.

A second trap: a peak and a width from the SAME noisy curve are coupled by noise (a noise spike
is both a peak and narrow). Peak comes from the odd legs, width from the even legs.

DECISIONS TAKEN WITH THE USER, 2026-09-08 (do not re-derive)
  * goal-progress cells = Freedman-Lane p < 0.05 on CPD for `goal_progress` in the gp+tfr
    uniform/30 arm (`SELECTING_ARM`); gp-only/30 membership is a robustness column.
  * per-leg 90-bin phase curves, leg cap 30 s (whole legs dropped), circular phase axis,
    sigma = 3 bins (sensitivity at 2 and 5); split = odd vs even legs, interleaved over the
    pooled GLM sessions in temporal order.
  * peak = circular argmax of the odd-leg curve (bin only; nothing else comes from that half);
    width = length of the contiguous circular run of even-leg bins with
    (B - min B)/(max B - min B) >= 0.5 that CONTAINS the odd-leg peak bin. If the even curve, on
    its own scale, is below 0.5 at that bin the peak did not reproduce: width NaN, flag
    `peak_not_reproduced`, counted and reported, excluded from every rho.
  * cross-session peak stability: per-session peaks on the circular 90-bin axis; a cell
    "remaps" if the MAX PAIRWISE circular distance between its per-session peaks exceeds
    30 bins (a third of the leg). The circular SD in bins is reported beside it; it saturates
    (two peaks 30 bins apart give SD ~ 17 bins), which is why it is not the flag.
  * beta route from the existing 10-bin betas only (selecting arm, and gp-only/30 as the
    "any within-leg structure" comparison); no refit.
  * units: neuron within mouse; per-(recday, region) Spearman rho -> mouse mean -> mean over
    mice with the mice shown; LEC per region under docs/handoff/README.md section 3; PFC pooled.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import glm_analysis_v3 as glm                           # noqa: E402
import w1_refit as w                                    # noqa: E402
import run_glm_batch as rb                              # noqa: E402

try:                                                    # the plot module of THIS tree
    import glm_plots as P                               # noqa: E402
except ImportError:                                     # pragma: no cover - PFC tree
    import pfc_glm_plots as P                           # noqa: E402

IS_PFC = hasattr(glm, 'build_data_dic_from_pfc')
DATASET = 'PFC' if IS_PFC else 'LEC'


def _repo_root():
    p = HERE
    for _ in range(4):
        if os.path.isdir(os.path.join(p, 'code')) and os.path.isdir(os.path.join(p, 'mFC_data')):
            return p
        p = os.path.dirname(p)
    raise RuntimeError('cannot locate the repo root from ' + HERE)


REPO = _repo_root()
DATA_ROOT = os.path.join(REPO, 'mFC_data', 'data') if IS_PFC else os.path.join(REPO, 'data')
CACHE_DIR = os.path.join(DATA_ROOT, 'processed_data', 'gp_leg_curves')
FIG_DIR = os.path.join(DATA_ROOT, 'figures', 'gp_tuning_width')
SYNTH_OUT = os.path.join(DATA_ROOT, 'processed_data', 'gp_tuning_width_synthetics.pkl')

BIN_MS = 25
N_BINS = 90
CAP_S = 30.0
SIGMA = 3
SIGMA_SENSITIVITY = (2, 5)
REMAP_BINS = 30            #: max pairwise circular peak distance (bins of 90) that flags a cell

SELECTING_ARM = 'core_progress_time__matched_250ms_decile_cap30s_tfrU10b'
GP_ONLY_ARM = 'core_progress_only__matched_250ms_decile_cap30s'
CORE5 = list(w.SECTIONS['core_progress_time']['regressors'])
CORE4 = list(w.SECTIONS['core_progress_only']['regressors'])
ARMS = ('core_progress_time__matched_250ms_decile_cap30s_tfrD10b',
        'core_progress_time__matched_250ms_decile_cap30s_tfrU10b',
        'core_progress_time__matched_250ms_decile_cap60s_tfrD10b',
        'core_progress_time__matched_250ms_decile_cap60s_tfrU10b',
        'core_progress_only__matched_250ms_decile_cap30s',
        'core_progress_only__matched_250ms_decile_cap60s')

# colours: the tree's plot module palette; region colours come from anatomy_split (LEC only)
C_TIME = P.C_PAST          # retrospective / time-from-reward
C_LEAD = P.C_FUTURE        # prospective / time-to-reward
C_PHASE = '#2C2C2A'        # planted phase cells
C_NEUTRAL = P.C_NEUTRAL
C_STONE = P.C_STONE


# ---------------------------------------------------------------------------
# filesystem
# ---------------------------------------------------------------------------

def _ensure_writable_dir(path):
    """`mFC_data/data/**` is periodically set read-only after the fact; re-open it."""
    os.makedirs(path, exist_ok=True)
    if not os.access(path, os.W_OK):
        os.chmod(path, os.stat(path).st_mode | 0o200)


def _save(fig, out_path):
    if not out_path:
        return
    _ensure_writable_dir(os.path.dirname(os.path.abspath(out_path)) or '.')
    with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig.savefig(out_path, bbox_inches=None, dpi=300)


def _style():
    glm.apply_gridmaze_style()


def _asp():
    """`anatomy_split` lives only in `code/`; import it from there in either tree (LEC only)."""
    p = os.path.join(REPO, 'code')
    if p not in sys.path:
        sys.path.insert(0, p)
    import anatomy_split
    return anatomy_split


def assert_mirror():
    """The two copies of this module and its synthetics must be byte-identical.

    `check_mirror_parity.BYTE_PAIRS` was not extended (standalone instruction), so the
    notebooks call this instead. Returns the list of pairs checked."""
    pairs = [('code/gp_tuning_width.py', 'mFC_data/code/gp_tuning_width.py'),
             ('code/gp_tuning_width_synthetics.py', 'mFC_data/code/gp_tuning_width_synthetics.py')]
    for a, b in pairs:
        pa, pb = os.path.join(REPO, a), os.path.join(REPO, b)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            raise AssertionError(f'{a} / {b}: one copy is missing')
        with open(pa, 'rb') as fa, open(pb, 'rb') as fb:
            if fa.read() != fb.read():
                raise AssertionError(f'{a} != {b}: copy code/ -> mFC_data/code/ before trusting a result')
    return pairs


# ---------------------------------------------------------------------------
# B1. per-leg phase curves
# ---------------------------------------------------------------------------

#: How to upsample a leg shorter than `n_bins` samples before rebinning. THE REPO HAS TWO
#: CONVENTIONS AND THEY DIFFER BY A FACTOR OF TEN.
#:
#:  'rate'         -- `np.repeat(seg, ceil(n_bins / L))`, no rescaling, as
#:                    `remapping_rotation_analysis.raw_to_norm` does (the plan's reference).
#:                    Repeating a sample k times leaves every bin mean equal to the local
#:                    firing rate, so a short leg's curve is on the same scale as a long one's.
#:                    DEFAULT, and the only rule used for a result.
#:  'legacy_div10' -- `np.repeat(seg, 10) / 10`, as `preprocessing/build_data_dic.normalise`
#:                    (which produced `norm_neurons_dic.pkl`, `Neurons_norm` and, through
#:                    `Smoothed_norm`, `gp_curves.pkl`) and `mFC glm_analysis_v3._raw_to_norm`
#:                    do. The division makes sense for counts-per-sub-bin under a max
#:                    aggregator; under the MEAN it puts a leg shorter than `n_bins` samples
#:                    (2.25 s at 25 ms) on a scale ten times too small. Kept ONLY so
#:                    `verify_against_legacy` can reproduce the legacy arrays exactly and
#:                    show that this is the entire difference between them.
SHORT_LEG_RULES = ('rate', 'legacy_div10')


def _rebin_leg(seg, n_bins, short_leg_rule='rate'):
    """(n_neurons, L) -> (n_neurons, n_bins) bin means, reproducing
    `scipy.stats.binned_statistic(np.arange(L), x, 'mean', bins=n_bins)` after the short-leg
    upsampling rule (`SHORT_LEG_RULES`)."""
    n, L = seg.shape
    if L < n_bins:
        if short_leg_rule == 'legacy_div10':
            seg = np.repeat(seg, 10, axis=1) / 10.0
        elif short_leg_rule == 'rate':
            seg = np.repeat(seg, int(np.ceil(n_bins / max(L, 1))), axis=1)
        else:
            raise ValueError(f'short_leg_rule must be one of {SHORT_LEG_RULES}, got {short_leg_rule!r}')
        L = seg.shape[1]
        if L < n_bins:                       # legacy rule on a leg of < 9 samples
            seg = np.repeat(seg, int(np.ceil(n_bins / L)), axis=1)
            L = seg.shape[1]
    idx = np.arange(L)
    edges = np.linspace(0, L - 1, n_bins + 1)
    b = np.digitize(idx, edges) - 1
    b[b >= n_bins] = n_bins - 1                 # scipy: the right edge belongs to the last bin
    starts = np.searchsorted(b, np.arange(n_bins))
    counts = np.diff(np.append(starts, L))
    if not (counts > 0).all():                  # cannot happen after the repeat rule
        raise AssertionError(f'empty phase bin for a leg of {L} samples')
    return np.add.reduceat(seg, starts, axis=1) / counts


def leg_phase_curves(neuron_raw, trial_times, *, n_bins=N_BINS, cap_s=CAP_S, bin_ms=BIN_MS,
                     short_leg_rule='rate'):
    """Every leg `[tt[r, s], tt[r, s+1])` of every trial, rebinned to `n_bins` equal-width phase
    bins, in Hz.

    Returns `curves (n_neurons, n_legs, n_bins) float32` and a dict of per-leg arrays
    (`duration_s`, `state`, `trial`, `start_bin`) plus the counts of legs dropped: `n_cap`
    (longer than `cap_s`; dropped WHOLE, like the GLM's `max_transition_seconds`, so kept legs
    put exactly 1/n_bins of their samples in every bin) and `n_invalid` (end <= start, or
    beyond the recording), and `n_short` (legs of fewer than `n_bins` samples, upsampled by
    `short_leg_rule`). `cap_s=None` keeps every valid leg (used by the legacy cross-check).
    Legs are in temporal order. The state label is the trial-times COLUMN, never `i % 4`
    (the 2026-07-30 labelling bug)."""
    NR = np.asarray(neuron_raw)
    T = NR.shape[1]
    tt = np.asarray(trial_times).astype(int)
    cap_bins = None if cap_s is None else int(round(cap_s * 1000.0 / bin_ms))
    hz = 1000.0 / bin_ms
    curves, dur, state, trial, start = [], [], [], [], []
    n_cap = n_invalid = n_short = 0
    for r in range(tt.shape[0]):
        for s in range(tt.shape[1] - 1):
            a, b = int(tt[r, s]), int(tt[r, s + 1])
            if not (b > a and a >= 0 and b <= T):
                n_invalid += 1
                continue
            if cap_bins is not None and (b - a) > cap_bins:
                n_cap += 1
                continue
            n_short += (b - a) < n_bins
            curves.append(_rebin_leg(NR[:, a:b].astype(np.float32), n_bins, short_leg_rule) * hz)
            dur.append((b - a) * bin_ms / 1000.0)
            state.append(s); trial.append(r); start.append(a)
    if curves:
        C = np.stack(curves, axis=1).astype(np.float32)
    else:
        C = np.zeros((NR.shape[0], 0, n_bins), np.float32)
    meta = {'duration_s': np.asarray(dur, float), 'state': np.asarray(state, int),
            'trial': np.asarray(trial, int), 'start_bin': np.asarray(start, int),
            'n_cap': n_cap, 'n_invalid': n_invalid, 'n_short': int(n_short)}
    return C, meta


def build_curves(session_arrays, *, cap_s=CAP_S, n_bins=N_BINS, bin_ms=BIN_MS,
                 short_leg_rule='rate'):
    """Per-leg curves for one recday from `{session: {'Neuron_raw', 'Trial_times'}}`, sessions
    in ascending (temporal) order, legs pooled. The ONE code path for real and synthetic data
    and for both datasets. Returns the cache dict (see `build_recday_cache`)."""
    sessions = sorted(session_arrays)
    C, sess, dur, state, trial, start = [], [], [], [], [], []
    n_cap = n_invalid = n_short = 0
    spikes = None; seconds = 0.0
    tts, Ts = {}, {}
    for s in sessions:
        NR = np.asarray(session_arrays[s]['Neuron_raw'])
        tt = np.asarray(session_arrays[s]['Trial_times']).astype(int)
        c, m = leg_phase_curves(NR, tt, n_bins=n_bins, cap_s=cap_s, bin_ms=bin_ms,
                                short_leg_rule=short_leg_rule)
        C.append(c)
        sess.append(np.full(c.shape[1], s, int))
        dur.append(m['duration_s']); state.append(m['state']); trial.append(m['trial'])
        start.append(m['start_bin'])
        n_cap += m['n_cap']; n_invalid += m['n_invalid']; n_short += m['n_short']
        sp = NR.sum(axis=1).astype(float)
        spikes = sp if spikes is None else spikes + sp
        seconds += NR.shape[1] * bin_ms / 1000.0
        tts[s] = tt; Ts[s] = int(NR.shape[1])
    curves = np.concatenate(C, axis=1) if C else np.zeros((0, 0, n_bins), np.float32)
    return {'curves': curves,
            'leg_session': np.concatenate(sess) if sess else np.zeros(0, int),
            'leg_duration_s': np.concatenate(dur) if dur else np.zeros(0),
            'leg_state': np.concatenate(state) if state else np.zeros(0, int),
            'leg_trial': np.concatenate(trial) if trial else np.zeros(0, int),
            'leg_start_bin': np.concatenate(start) if start else np.zeros(0, int),
            'n_legs_dropped_cap': int(n_cap), 'n_legs_dropped_invalid': int(n_invalid),
            'n_legs_short': int(n_short),
            'mean_rate_hz': (spikes / seconds) if spikes is not None else np.zeros(0),
            'sessions': sessions, 'trial_times': tts, 'T': Ts,
            'cap_s': cap_s, 'n_bins': n_bins, 'bin_ms': bin_ms, 'short_leg_rule': short_leg_rule}


def build_recday_cache(data_rd, rd, *, cap_s=CAP_S, n_bins=N_BINS):
    """`build_curves` on the GLM's session set (`get_sessions_for_glm`: ABCD sessions with
    trials, deduplicated by task), plus provenance."""
    sessions, tasks = glm.get_sessions_for_glm(data_rd)
    arrays = {s: {'Neuron_raw': data_rd[s]['Neuron_raw'], 'Trial_times': data_rd[s]['Trial_times']}
              for s in sessions}
    out = build_curves(arrays, cap_s=cap_s, n_bins=n_bins)
    out.update({'recday': rd, 'dataset': DATASET, 'mouse': rd.split('_')[0],
                'tasks': {s: str(data_rd[s].get('Task')) for s in sessions},
                'n_trials': {s: int(data_rd[s].get('num_trials', np.asarray(data_rd[s]['Trial_times']).shape[0]))
                             for s in sessions},
                'built': time.strftime('%Y-%m-%d %H:%M:%S')})
    return out


def cache_path(rd, cache_dir=CACHE_DIR):
    return os.path.join(cache_dir, f'{rd}.pkl')


def save_cache(cache, cache_dir=CACHE_DIR):
    _ensure_writable_dir(cache_dir)
    with open(cache_path(cache['recday'], cache_dir), 'wb') as fh:
        pickle.dump(cache, fh, protocol=4)


def load_cache(rd, cache_dir=CACHE_DIR):
    with open(cache_path(rd, cache_dir), 'rb') as fh:
        return pickle.load(fh)


def load_all_caches(cache_dir=CACHE_DIR):
    """`{recday: cache}` for every cached recday of this tree, excluded recdays dropped."""
    import glob
    out = {}
    excluded = set(glm._stale_or_excluded(SELECTING_ARM))
    for p in sorted(glob.glob(os.path.join(cache_dir, '*.pkl'))):
        rd = os.path.basename(p)[:-4]
        if rd in excluded or rd.startswith('_'):        # `_verify_legacy.pkl` etc. are not caches
            continue
        with open(p, 'rb') as fh:
            c = pickle.load(fh)
        if isinstance(c, dict) and 'curves' in c:
            out[rd] = c
    return out


def build_all(cap_s=CAP_S, n_bins=N_BINS, recdays=None, verbose=True):
    """Build and save every recday's cache for this tree. LEC loads the 3.8 GB `data_dic`
    once; PFC is per recday."""
    t0 = time.time()
    _ensure_writable_dir(CACHE_DIR)
    if IS_PFC:
        rds = recdays or w.pfc_recdays()
        for rd in rds:
            dd = glm.build_data_dic_from_pfc(w.DATA_FOLDER, [rd], verbose=False)
            if rd not in dd:
                print(f'  {rd}: no data, skipped'); continue
            c = build_recday_cache(dd[rd], rd, cap_s=cap_s, n_bins=n_bins)
            save_cache(c)
            if verbose:
                print(f'  {rd}: {c["curves"].shape[0]} neurons x {c["curves"].shape[1]} legs '
                      f'({c["n_legs_dropped_cap"]} > cap, {c["n_legs_dropped_invalid"]} invalid), '
                      f'sessions {c["sessions"]}  ({time.time() - t0:.0f} s)')
    else:
        dd = glm.load_data_dic(validate=True, apply_exclusions=True, verbose=verbose)
        rds = recdays or sorted(dd)
        for rd in rds:
            c = build_recday_cache(dd[rd], rd, cap_s=cap_s, n_bins=n_bins)
            save_cache(c)
            if verbose:
                print(f'  {rd}: {c["curves"].shape[0]} neurons x {c["curves"].shape[1]} legs '
                      f'({c["n_legs_dropped_cap"]} > cap, {c["n_legs_dropped_invalid"]} invalid), '
                      f'sessions {c["sessions"]}  ({time.time() - t0:.0f} s)')
    print(f'{DATASET}: {len(rds)} recdays cached under {CACHE_DIR} in {(time.time() - t0) / 60:.1f} min')


# ---------------------------------------------------------------------------
# B1. split half, peak, width
# ---------------------------------------------------------------------------

def smooth_circular(x, sigma):
    """Gaussian smoothing along the last axis with wrap-around (the leg end runs into the next
    leg's start through the reward). Identical to `remapping_rotation_analysis.smooth_circular`
    for sigma << n_bins; `sigma=0` returns the input."""
    x = np.asarray(x, float)
    if not sigma:
        return x
    return gaussian_filter1d(x, sigma, axis=-1, mode='wrap')


def split_half(curves, *, sigma=SIGMA, order=None):
    """Odd legs -> curve A, even legs -> curve B, each the mean over its legs then circularly
    smoothed; plus the split-half Pearson r per neuron between the two.

    `order` is the temporal order of the legs (default: the stored order, which is already
    temporal: sessions ascending, legs within session ascending). Odd/even is by POSITION in
    that order, so the two halves interleave across the whole recday and session drift (leg
    duration falls with session time, rho ~ -0.3) does not become a difference between halves.
    The halves are asserted disjoint and interleaved."""
    n_legs = curves.shape[1]
    order = np.arange(n_legs) if order is None else np.asarray(order)
    idx_a, idx_b = order[0::2], order[1::2]
    if set(idx_a.tolist()) & set(idx_b.tolist()):
        raise AssertionError('split halves overlap')
    if len(idx_a) + len(idx_b) != n_legs:
        raise AssertionError('split halves do not cover the legs')
    if n_legs >= 2 and not np.all(np.diff(np.sort(np.concatenate([idx_a, idx_b]))) == 1):
        raise AssertionError('split halves are not a partition of 0..n-1')
    if n_legs >= 3 and not np.all(idx_b - idx_a[:len(idx_b)] == 1):
        raise AssertionError('split halves are not interleaved')
    A = smooth_circular(np.nanmean(curves[:, idx_a, :], axis=1), sigma)
    B = smooth_circular(np.nanmean(curves[:, idx_b, :], axis=1), sigma)
    r = _rowwise_pearson(A, B)
    return A, B, r, {'n_odd': int(len(idx_a)), 'n_even': int(len(idx_b))}


def _rowwise_pearson(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    a = A - A.mean(axis=1, keepdims=True); b = B - B.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (a * b).sum(1) / den
    r[~np.isfinite(r)] = np.nan
    return r


def circ_mean_phase(weights, n_bins=N_BINS):
    """Circular mean direction of a non-negative weight vector over bin CENTRES, as a phase in
    [0, 1); and the resultant length R. Rows of a 2-D input are handled independently."""
    W = np.atleast_2d(np.asarray(weights, float))
    theta = 2 * np.pi * (np.arange(W.shape[1]) + 0.5) / n_bins
    z = (W * np.exp(1j * theta)).sum(axis=1)
    tot = W.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = z / tot
    ph = (np.angle(z) / (2 * np.pi)) % 1.0
    R = np.abs(z)
    ph[~np.isfinite(R)] = np.nan
    return ph, R


def circ_sd_fraction(weights, n_bins=N_BINS):
    """Circular SD, sqrt(-2 ln R), of a weight vector treated as a density on the phase circle,
    expressed as a fraction of the leg (radians / 2 pi). The threshold-free width."""
    _, R = circ_mean_phase(weights, n_bins)
    with np.errstate(invalid='ignore', divide='ignore'):
        sd = np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0)))
    return sd / (2 * np.pi)


def _runs_on_circle(mask):
    """Number of contiguous runs of True on a circular boolean vector (0 if none, 1 if all)."""
    m = np.asarray(mask, bool)
    if m.all():
        return 1
    if not m.any():
        return 0
    return int(np.sum(m & ~np.roll(m, 1)))


def _run_containing(mask, k, circular=True):
    """Length of the contiguous run of True containing index k (0 if mask[k] is False)."""
    m = np.asarray(mask, bool)
    n = len(m)
    if not m[k]:
        return 0
    if m.all():
        return n
    length = 1
    j = k
    while True:
        j2 = (j + 1) % n if circular else j + 1
        if (not circular and j2 >= n) or j2 == k or not m[j2]:
            break
        length += 1; j = j2
    j = k
    while True:
        j2 = (j - 1) % n if circular else j - 1
        if (not circular and j2 < 0) or j2 == k or not m[j2]:
            break
        length += 1; j = j2
    return length


def peak_and_width(A, B, *, n_bins=None, edge_frac=0.1, ramp_width=0.6):
    """Per neuron: peak from half A, width from half B around A's peak (see module docstring).

    EDGE ARTEFACTS. The half-max run is taken on the CIRCLE, so a field sitting on the leg
    boundary is never truncated and no peak position has less room than any other: the estimator
    has no edge by construction, and synthetic control 3 measures that directly (planted fields of
    width 0.05-0.5 come back within a bin of their planted width at every planted peak from 0.1 to
    0.9, with zero peak-dependence). `width_hm_linear` is the same run WITHOUT the wrap and is
    reported once so the doc can quantify what a non-circular definition would have cost -- it
    truncates exactly the fields that straddle the reward, which is the artefact the circular axis
    avoids. Two things that are NOT estimator edges but do live at the boundary and must be read as
    such: the animal is stationary at the port around phase 0 and 1, so firing there is
    behaviourally special; and the smoothing wraps within a leg, which is legitimate only because
    consecutive legs are pooled (the end of one leg and the start of the next are the same moment,
    and both are in the average).

    Columns: `peak_bin`, `peak_phase` (bin centre), `peak_phase_cm` (circular mean of the
    baseline-subtracted A curve, noise-robust alternative), `width_hm` (half-max run in
    fraction of the leg, circular), `width_hm_linear` (the same run not allowed to wrap --
    reported once so the doc can say what the circular axis changes), `width_csd` (circular SD
    of the baseline-subtracted B curve as a density, fraction of the leg), `above_half_frac`
    (fraction of B bins above half max regardless of contiguity), flags `peak_not_reproduced`
    (B on its own scale < 0.5 at A's peak bin -> width NaN), `multimodal` (the above-half-max
    set of B is not one circular run), `ramp` (peak within `edge_frac` of either end of the leg
    and width > `ramp_width`), `flat` (B has zero range)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    n, nb = A.shape
    n_bins = n_bins or nb
    peak_bin = np.argmax(A, axis=1)
    peak_bin_B = np.argmax(B, axis=1)
    peak_phase = (peak_bin + 0.5) / n_bins
    ph_cm, _ = circ_mean_phase(A - A.min(axis=1, keepdims=True), n_bins)
    rows = []
    for i in range(n):
        b = B[i]
        rng = b.max() - b.min()
        if not np.isfinite(rng) or rng <= 0:
            rows.append(dict(width_hm=np.nan, width_hm_linear=np.nan, above_half_frac=np.nan,
                             peak_not_reproduced=False, multimodal=False, flat=True))
            continue
        bn = (b - b.min()) / rng
        above = bn >= 0.5
        k = int(peak_bin[i])
        if not above[k]:
            rows.append(dict(width_hm=np.nan, width_hm_linear=np.nan,
                             above_half_frac=float(above.mean()),
                             peak_not_reproduced=True, multimodal=_runs_on_circle(above) > 1,
                             flat=False))
            continue
        rows.append(dict(width_hm=_run_containing(above, k, True) / n_bins,
                         width_hm_linear=_run_containing(above, k, False) / n_bins,
                         above_half_frac=float(above.mean()),
                         peak_not_reproduced=False, multimodal=_runs_on_circle(above) > 1,
                         flat=False))
    out = pd.DataFrame(rows)
    out.insert(0, 'peak_bin', peak_bin)
    out.insert(1, 'peak_phase', peak_phase)
    out.insert(2, 'peak_phase_cm', ph_cm)
    #: the two halves' peaks and their circular distance -- the A-B PEAK-DISAGREEMENT gate
    #: (`ab_gate_sweep`) thresholds this. It is NOT applied by default: the primary rule is the
    #: half-max one above. See GP_TUNING_WIDTH.md section 2.0.1 -- the distance is correlated
    #: with width (disagreeing cells are broad), so gating on it selects on something related to
    #: the outcome and belongs in a robustness variant, not in the primary analysis.
    out['peak_bin_B'] = peak_bin_B
    out['ab_offset_bins'] = circ_dist_bins(peak_bin, peak_bin_B, n_bins)
    out['width_csd'] = circ_sd_fraction(B - B.min(axis=1, keepdims=True), n_bins)
    edge = (peak_phase < edge_frac) | (peak_phase >= 1.0 - edge_frac)
    out['ramp'] = edge & (out['width_hm'] > ramp_width)
    out['peak_hz_A'] = A.max(axis=1)
    out['range_hz_B'] = B.max(axis=1) - B.min(axis=1)
    return out


def plot_width_explainer(A, B, table, *, index=None, n_bins=N_BINS, out_path=None, seed=0):
    """Worked examples of the width rule, one panel per neuron, so the definition can be read off
    a picture rather than a docstring.

    Each panel draws the odd-leg curve A (grey) and the even-leg curve B (black) in Hz, marks A's
    peak bin, draws B's own minimum and maximum and the half-max level **half way between them**,
    and shades the contiguous circular run of B above that level which contains A's peak bin. The
    length of that shaded run, as a fraction of the leg, is the reported width.

    THE THRESHOLD IS B'S OWN HALF-MAX, NOT HALF OF B'S VALUE AT A'S PEAK. Concretely: if B at A's
    peak bin sits at 0.8 of B's min-to-max range, the cutoff is still 0.5 of that range, not 0.4.
    Two consequences, both deliberate and both visible in the panels:
      * the shaded run is generally NOT symmetric about A's peak -- it is B's own field, and A's
        peak only selects WHICH field (which run) to measure, so that noise in B's argmax cannot
        make the estimator jump to a different bump;
      * a cell whose B curve is below 0.5 at A's peak has no run to measure: width is NaN and the
        cell is flagged `peak_not_reproduced` (bottom-right panel).

    `A`, `B` are the (n_neurons, n_bins) matrices from `stack_curves`; `table` the matching
    per-neuron table (same row order). `index` picks the rows to draw; by default one example of
    each of: narrow, asymmetric (A's peak far from B's), wrapping the reward, multimodal, broad,
    and not reproduced."""
    _style()
    rng = np.random.default_rng(seed)
    t = table.reset_index(drop=True)
    pkA = np.argmax(A, axis=1)
    pkB = np.argmax(B, axis=1)
    off = circ_dist_bins(pkA, pkB, n_bins)
    wraps = (t['width_hm'] > t['width_hm_linear'] + 1e-9).to_numpy()
    ok = np.isfinite(t['width_hm']).to_numpy()

    def pick(mask, note):
        idx = np.flatnonzero(mask)
        return (int(rng.choice(idx)), note) if len(idx) else None

    if index is None:
        cand = [
            pick(ok & (t['width_hm'] < 0.25).to_numpy() & (off < 3) & ~t['multimodal'].to_numpy(),
                 'narrow, A and B agree'),
            pick(ok & (off >= 8) & ~t['multimodal'].to_numpy(),
                 "asymmetric: A's peak is off-centre in B's field"),
            pick(ok & wraps, 'field wraps the reward (0.9 → 0.1)'),
            pick(ok & t['multimodal'].to_numpy(), 'multimodal: B has more than one run'),
            pick(ok & (t['width_hm'] > 0.6).to_numpy(), 'broad'),
            pick(t['peak_not_reproduced'].to_numpy(), 'NOT reproduced → width NaN'),
        ]
        picks = [c for c in cand if c is not None]
    else:
        picks = [(int(i), '') for i in index]
    n = len(picks)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.0 * nrow), squeeze=False)
    x = (np.arange(n_bins) + 0.5) / n_bins
    for ax, (i, note) in zip(axes.ravel(), picks):
        a_, b_ = A[i], B[i]
        lo, hi = b_.min(), b_.max()
        thr = lo + 0.5 * (hi - lo)
        k = int(pkA[i])
        ax.plot(x, a_, color=C_STONE, lw=1.0, label='odd legs (A)')
        ax.plot(x, b_, color='#2C2C2A', lw=1.3, label='even legs (B)')
        ax.axhline(thr, color=C_TIME, lw=0.8, ls='--')
        ax.axhline(lo, color=C_NEUTRAL, lw=0.5, ls=':')
        ax.axhline(hi, color=C_NEUTRAL, lw=0.5, ls=':')
        ax.axvline(x[k], color=C_LEAD, lw=1.0)
        above = (b_ - lo) / (hi - lo) >= 0.5 if hi > lo else np.zeros(n_bins, bool)
        if above[k]:
            run = np.zeros(n_bins, bool)
            j = k
            while True:
                run[j] = True
                j = (j + 1) % n_bins
                if j == k or not above[j]:
                    break
            j = k
            while True:
                run[j] = True
                j = (j - 1) % n_bins
                if j == k or not above[j]:
                    break
            for s0 in np.flatnonzero(run & ~np.roll(run, 1)):
                L = _run_containing(run, int(s0), True)
                for m in range(L):
                    ax.axvspan(x[(s0 + m) % n_bins] - 0.5 / n_bins,
                               x[(s0 + m) % n_bins] + 0.5 / n_bins,
                               color=C_TIME, alpha=0.13, lw=0)
        r = t.iloc[i]
        w = r['width_hm']
        ax.set_title(f"{note}\nwidth = {'NaN' if not np.isfinite(w) else f'{w:.3f}'}"
                     f"   |  B at A's peak = {((b_[k] - lo) / (hi - lo) if hi > lo else np.nan):.2f}"
                     f" of B's range", loc='left', fontsize=6)
        ax.set_xlim(0, 1)
        ax.set_xlabel('goal progress (fraction of leg)', fontsize=6)
        ax.set_ylabel('Hz', fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    axes.ravel()[0].legend(frameon=False, fontsize=5, loc='upper right')
    fig.suptitle("width = length of the contiguous circular run of B above B's OWN half-max "
                 "(red dashed) that contains A's peak (blue line)", fontsize=6, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_path)
    return fig, [i for i, _ in picks]


def circ_dist_bins(a, b, n_bins=N_BINS):
    d = np.abs(np.asarray(a) - np.asarray(b)) % n_bins
    return np.minimum(d, n_bins - d)


def per_session_peaks(curves, leg_session, *, sigma=SIGMA, n_bins=N_BINS, min_legs=8):
    """Peak bin per neuron per session (all legs of that session, smoothed), and two spread
    statistics across sessions: `max_pair_dist_bins` (max pairwise circular distance -- the
    flag statistic) and `circ_sd_bins` (sqrt(-2 ln R) x n_bins / 2 pi, reported, saturating).
    Sessions with fewer than `min_legs` kept legs are skipped."""
    sessions = sorted(set(leg_session.tolist()))
    peaks = {}
    for s in sessions:
        m = leg_session == s
        if m.sum() < min_legs:
            continue
        c = smooth_circular(np.nanmean(curves[:, m, :], axis=1), sigma)
        peaks[s] = np.argmax(c, axis=1)
    if len(peaks) < 2:
        n = curves.shape[0]
        return pd.DataFrame({'n_sessions_peaked': [len(peaks)] * n,
                             'max_pair_dist_bins': [np.nan] * n, 'circ_sd_bins': [np.nan] * n,
                             'remaps': [False] * n}), peaks
    Pk = np.stack([peaks[s] for s in peaks], axis=1)          # (n, n_sessions)
    mx = np.zeros(Pk.shape[0])
    for i in range(Pk.shape[1]):
        for j in range(i + 1, Pk.shape[1]):
            mx = np.maximum(mx, circ_dist_bins(Pk[:, i], Pk[:, j], n_bins))
    theta = 2 * np.pi * (Pk + 0.5) / n_bins
    R = np.abs(np.exp(1j * theta).mean(axis=1))
    sd = np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0))) * n_bins / (2 * np.pi)
    return pd.DataFrame({'n_sessions_peaked': Pk.shape[1], 'max_pair_dist_bins': mx,
                         'circ_sd_bins': sd, 'remaps': mx > REMAP_BINS}), peaks


def peaks_by_state(curves, leg_state, leg_duration_s, *, sigma=SIGMA, min_legs=8):
    """Peak phase per neuron per task state (legs of that state only), with the state's median
    leg duration -- the B5 discriminator (time cells shift with D, phase cells do not)."""
    out = {}
    for s in sorted(set(leg_state.tolist())):
        m = leg_state == s
        if m.sum() < min_legs:
            continue
        c = smooth_circular(np.nanmean(curves[:, m, :], axis=1), sigma)
        out[s] = {'peak_phase': (np.argmax(c, axis=1) + 0.5) / curves.shape[2],
                  'median_duration_s': float(np.median(leg_duration_s[m])), 'n_legs': int(m.sum())}
    return out


def curve_route(cache, *, sigma=SIGMA):
    """B1 end to end for one recday cache -> per-neuron DataFrame (recday, mouse, neuron, ...)."""
    C = cache['curves']
    A, B, r, halves = split_half(C, sigma=sigma)
    tab = peak_and_width(A, B)
    tab['splithalf_r'] = r
    stab, _ = per_session_peaks(C, cache['leg_session'], sigma=sigma)
    tab = pd.concat([tab, stab], axis=1)
    tab.insert(0, 'neuron', np.arange(C.shape[0]))
    tab.insert(0, 'mouse', cache.get('mouse', cache['recday'].split('_')[0]))
    tab.insert(0, 'recday', cache['recday'])
    tab['mean_rate_hz'] = cache['mean_rate_hz']
    tab['n_legs'] = C.shape[1]
    tab['n_odd'] = halves['n_odd']; tab['n_even'] = halves['n_even']
    tab['sigma'] = sigma
    return tab, (A, B)


# ---------------------------------------------------------------------------
# B2. the beta route
# ---------------------------------------------------------------------------

def load_arm(section, out_dir=None):
    return glm.load_glm_results(out_dir or rb.DEFAULT_OUT, section, apply_exclusions=True, verbose=False)


def regs_for(section):
    return CORE5 if section.startswith('core_progress_time') else CORE4


def gp_col_idx(section):
    return glm._resolve_regressor_groups(regs_for(section), parameterization='reference_coded')[0]['goal_progress']


def beta_profile(params, col_idx, center='reference'):
    """Reference-coded beta vector with the reference bin (= 0) prepended; `center='mean'`
    subtracts the mean over bins."""
    beta = np.concatenate([[0.0], np.asarray(params, float)[np.asarray(col_idx)]])
    if center == 'mean':
        beta = beta - np.nanmean(beta)
    return beta


def beta_peak_width(beta):
    """Peak = signed argmax (bin 0 included); width = contiguous circular run around the peak
    with (beta - min)/(max - min) >= 0.5, in fractions of the leg; multimodal flag."""
    beta = np.asarray(beta, float)
    n = len(beta)
    k = int(np.argmax(beta))
    rng = beta.max() - beta.min()
    if not np.isfinite(rng) or rng <= 0:
        return k, np.nan, False
    above = (beta - beta.min()) / rng >= 0.5
    return k, _run_containing(above, k, True) / n, _runs_on_circle(above) > 1


def beta_table(res, section, *, alpha=0.05, null='freedman_lane', p_stat='cpd', prefix=''):
    """Per-neuron beta-route table for one arm: p, CPD, delta r2 (gp, and tfr if present), the
    gp-vs-tfr split, significance, beta peak / width / multimodality, the unit-norm profile."""
    col_idx = gp_col_idx(section)
    rows = []
    for rd, cv in res['cv_results'].items():
        if rd not in res['glm_results']:
            continue
        pv = P.p_values(cv, p_stat, null)
        keys = sorted(res['glm_results'][rd])
        has_tfr = 'time_from_reward' in cv['delta_r2_cv']
        for k, nrn in enumerate(keys):
            beta = beta_profile(res['glm_results'][rd][nrn], col_idx)
            m = np.nanmax(np.abs(beta))
            pk, wd, mm = beta_peak_width(beta)
            rows.append({'recday': rd, 'neuron': k,
                         f'{prefix}p_gp': float(pv['goal_progress'][k]),
                         f'{prefix}sig_gp': bool(pv['goal_progress'][k] < alpha),
                         f'{prefix}cpd_gp': float(cv['cpd_cv']['goal_progress'][k]),
                         f'{prefix}dr2_gp': float(cv['delta_r2_cv']['goal_progress'][k]),
                         f'{prefix}dr2_tfr': float(cv['delta_r2_cv']['time_from_reward'][k]) if has_tfr else np.nan,
                         f'{prefix}beta_peak_bin': pk,
                         f'{prefix}beta_peak_phase': (pk + 0.5) / len(beta),
                         f'{prefix}beta_width': wd, f'{prefix}beta_multimodal': mm,
                         f'{prefix}beta_max': float(m)})
    df = pd.DataFrame(rows)
    df[f'{prefix}split'] = df[f'{prefix}dr2_gp'] - df[f'{prefix}dr2_tfr']
    return df


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

def assemble(caches, res_sel, res_gp, *, sigma=SIGMA, unit_regions=None, alpha=0.05):
    """Join curve route (all caches) with the beta route of the selecting arm (`sel_`) and the
    gp-only arm (`gpo_`), and with anatomy (LEC). Positional join asserted per recday."""
    tabs = []
    for rd in sorted(caches):
        t, _ = curve_route(caches[rd], sigma=sigma)
        tabs.append(t)
    cur = pd.concat(tabs, ignore_index=True)
    sel = beta_table(res_sel, SELECTING_ARM, alpha=alpha, prefix='sel_')
    gpo = beta_table(res_gp, GP_ONLY_ARM, alpha=alpha, prefix='gpo_')
    for name, bt in (('selecting arm', sel), ('gp-only arm', gpo)):
        n_cur = cur.groupby('recday').size(); n_bt = bt.groupby('recday').size()
        common = n_cur.index.intersection(n_bt.index)
        bad = [rd for rd in common if n_cur[rd] != n_bt[rd]]
        if bad:
            raise AssertionError(f'{name}: neuron count differs from the curves in {bad} -- positional join refused')
    out = cur.merge(sel, on=['recday', 'neuron'], how='inner').merge(gpo, on=['recday', 'neuron'], how='inner')
    out['abs_peak_off_centre'] = np.abs(out['peak_phase'] - 0.5)
    if unit_regions is not None:
        reg = []
        for rd, ur in unit_regions.items():
            if rd in set(out['recday']):
                n = int((out['recday'] == rd).sum())
                if n != len(ur):
                    raise AssertionError(f'{rd}: {n} neurons vs {len(ur)} unit_regions rows')
                d = ur[['group', 'y_um', 'shank']].copy().reset_index(drop=True)
                d.insert(0, 'neuron', np.arange(n)); d.insert(0, 'recday', rd)
                reg.append(d)
        out = out.merge(pd.concat(reg, ignore_index=True), on=['recday', 'neuron'], how='left')
    else:
        out['group'] = 'PFC'
    return out


def agreement(table, *, circular_n=1.0):
    """Beta route vs curve route per neuron: circular correlation of peaks (Fisher-Lee) and
    Spearman of widths, on the gp-significant cells with a defined width in both routes."""
    d = table[table['sel_sig_gp'] & np.isfinite(table['width_hm']) & np.isfinite(table['sel_beta_width'])]
    a = 2 * np.pi * d['peak_phase'].to_numpy(); b = 2 * np.pi * d['sel_beta_peak_phase'].to_numpy()
    rc = circular_corr(a, b)
    rs = spearmanr(d['width_hm'], d['sel_beta_width']).correlation if len(d) > 2 else np.nan
    dist = circ_dist_bins(d['peak_phase'] * 10, d['sel_beta_peak_phase'] * 10, 10)  # tenths
    return {'n': int(len(d)), 'circ_corr_peaks': rc, 'spearman_widths': float(rs),
            'peak_within_1_tenth': float(np.mean(dist <= 1.0)) if len(d) else np.nan,
            'median_peak_dist_tenths': float(np.median(dist)) if len(d) else np.nan}


def circular_corr(a, b):
    """Fisher-Lee circular correlation of two angle vectors (radians)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    am = np.angle(np.exp(1j * a).mean()); bm = np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - am), np.sin(b - bm)
    den = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / den) if den > 0 else np.nan


# ---------------------------------------------------------------------------
# statistics: neuron within mouse
# ---------------------------------------------------------------------------

def per_recday_rho(table, x, y, *, groups=None, min_n=8):
    """Spearman rho(x, y) per (recday, group) over neurons with both defined."""
    rows = []
    d = table.dropna(subset=[x, y])
    if groups is not None:
        d = d[d['group'].isin(groups)]
    for (rd, g), sub in d.groupby(['recday', 'group']):
        if len(sub) < min_n:
            continue
        r = spearmanr(sub[x], sub[y]).correlation
        rows.append({'recday': rd, 'mouse': rd.split('_')[0], 'group': g, 'n': len(sub),
                     'rho': float(r) if np.isfinite(r) else np.nan})
    return pd.DataFrame(rows)


def per_mouse_rho(table, x, y, *, groups=None, min_n=8):
    """recday rho -> mouse mean; DataFrame(mouse, group, rho, n_recdays, n_units)."""
    pr = per_recday_rho(table, x, y, groups=groups, min_n=min_n)
    if not len(pr):
        return pr
    return (pr.dropna(subset=['rho']).groupby(['mouse', 'group'])
            .agg(rho=('rho', 'mean'), n_recdays=('rho', 'size'), n_units=('n', 'sum')).reset_index())


def rho_report(table, x, y, *, groups=None, min_n=8, n_boot=10000, seed=0):
    """Every group on its own: mean over mice of the per-mouse rho, mouse bootstrap CI, the
    mice listed. A 1-mouse group is descriptive."""
    pm = per_mouse_rho(table, x, y, groups=groups, min_n=min_n)
    rng = np.random.default_rng(seed)
    rows = []
    for g in (groups or sorted(pm['group'].unique())):
        sub = pm[pm['group'] == g]
        if not len(sub):
            continue
        eff = sub['rho'].to_numpy(float)
        if len(eff) > 1:
            boot = np.array([np.mean(rng.choice(eff, len(eff), replace=True)) for _ in range(n_boot)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
        else:
            lo = hi = np.nan
        rows.append({'group': g, 'n_mice': len(eff), 'n_recdays': int(sub['n_recdays'].sum()),
                     'n_units': int(sub['n_units'].sum()), 'rho_mean_over_mice': float(np.mean(eff)),
                     'ci_lo': lo, 'ci_hi': hi,
                     'per_mouse': {m: round(float(r), 3) for m, r in zip(sub['mouse'], eff)},
                     'evidence': ('primary-capable' if len(eff) >= 3 else 'secondary' if len(eff) == 2
                                  else 'descriptive (1 mouse)')})
    return pd.DataFrame(rows)


def rho_contrast(table, x, y, group_a, group_b, *, min_n=8, n_boot=10000, seed=0):
    """Mouse-resampling bootstrap of rho(group_a) - rho(group_b), mice with both regions."""
    pm = per_mouse_rho(table, x, y, groups=[group_a, group_b], min_n=min_n)
    if not len(pm):
        return {'n_mice': 0}
    wide = pm.pivot(index='mouse', columns='group', values='rho').dropna()
    if not len(wide):
        return {'n_mice': 0}
    d = (wide[group_a] - wide[group_b]).to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'contrast': f'{group_a} - {group_b}', 'n_mice': int(len(d)),
            'per_mouse_diff': {m: round(float(v), 3) for m, v in zip(wide.index, d)},
            'mean_diff': float(np.mean(d)), 'ci_lo': float(lo), 'ci_hi': float(hi)}


# ---------------------------------------------------------------------------
# B3. the simulation null
# ---------------------------------------------------------------------------

LATENCIES_S = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0)
#: Planted phase-field widths. 0.4 and 0.5 are past the plan's 0.05-0.3 range and are there
#: because the REAL cells' median half-max width is ~0.49: a calibration that stops at 0.3 says
#: nothing about the regime the data occupies, and it is exactly in the wide regime that an edge
#: effect would appear if the width estimator had one.
PHASE_WIDTHS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
PHASE_CENTRES = tuple(np.round(np.arange(0.1, 0.95, 0.1), 2))


def simulate_population(trial_times_by_session, T_by_session, *, latencies=LATENCIES_S,
                        phase_widths=PHASE_WIDTHS, phase_centres=PHASE_CENTRES, n_noise=20,
                        base_hz=1.0, amp_hz=15.0, noise_hz=2.0, window_s=1.0, seed=0, bin_ms=BIN_MS):
    """Poisson cells on a recday's REAL legs: time-from-reward cells (`amp_hz` for `window_s`
    starting `t0` after each reward), time-to-reward cells (the same window ending `t1` before
    the next reward), phase cells (`amp_hz` over a phase window of width w centred at p,
    circular), and constant-rate noise. Returns `{'arrays': {session: {'Neuron_raw',
    'Trial_times'}}, 'cells': DataFrame(kind, param, param2)}` -- feed `arrays` to
    `build_curves`, the same door the data goes through."""
    rng = np.random.default_rng(seed)
    cells = ([{'kind': 'tfr', 'param': float(t), 'param2': np.nan} for t in latencies]
             + [{'kind': 'ttr', 'param': float(t), 'param2': np.nan} for t in latencies]
             + [{'kind': 'phase', 'param': float(p), 'param2': float(wd)} for wd in phase_widths for p in phase_centres]
             + [{'kind': 'noise', 'param': np.nan, 'param2': np.nan} for _ in range(n_noise)])
    cells = pd.DataFrame(cells)
    arrays = {}
    for s in sorted(trial_times_by_session):
        tt = np.asarray(trial_times_by_session[s]).astype(int)
        T = int(T_by_session[s])
        _, gp, _, tf, tr = glm.compute_task_state_arrays(tt, num_bins=10)
        n = min(T, len(tf))
        tf_s = np.full(T, np.nan); tf_s[:n] = tf[:n] * bin_ms / 1000.0
        tr_s = np.full(T, np.nan); tr_s[:n] = tr[:n] * bin_ms / 1000.0
        gp_c = np.full(T, np.nan); gp_c[:n] = gp[:n]
        # outside every leg (before the first boundary, after the last) nothing is defined
        inleg = np.zeros(T, bool)
        for r in range(tt.shape[0]):
            for k in range(tt.shape[1] - 1):
                a, b = int(tt[r, k]), int(tt[r, k + 1])
                if b > a and a >= 0:
                    inleg[a:min(b, T)] = True
        rates = np.full((len(cells), T), noise_hz)
        for i, c in cells.iterrows():
            if c['kind'] == 'tfr':
                on = inleg & (tf_s >= c['param']) & (tf_s < c['param'] + window_s)
            elif c['kind'] == 'ttr':
                on = inleg & (tr_s > c['param']) & (tr_s <= c['param'] + window_s)
            elif c['kind'] == 'phase':
                d = np.abs(gp_c - c['param']); d = np.minimum(d, 1.0 - d)
                on = inleg & (d < c['param2'] / 2.0)
            else:
                continue
            rates[i] = base_hz + amp_hz * on
        arrays[s] = {'Neuron_raw': rng.poisson(rates * bin_ms / 1000.0).astype(np.uint16),
                     'Trial_times': tt}
    return {'arrays': arrays, 'cells': cells}


def simulate_recday(cache, *, seed=0, sigma=SIGMA, sigmas=None, cap_s=None, **kw):
    """`simulate_population` on a recday cache's real legs -> curve-route table for the planted
    cells (with `kind`, `param`, `param2`), plus per-state peaks (the B5 discriminator).
    `sigmas` scores the SAME Poisson draw at several smoothing widths (a `sigma` column)."""
    cap_s = cache['cap_s'] if cap_s is None else cap_s
    sim = simulate_population(cache['trial_times'], cache['T'], seed=seed, **kw)
    c = build_curves(sim['arrays'], cap_s=cap_s, n_bins=cache['n_bins'])
    c['recday'] = cache['recday']; c['mouse'] = cache.get('mouse', cache['recday'].split('_')[0])
    tabs = []
    for sg in (sigmas or (sigma,)):
        tab, _ = curve_route(c, sigma=sg)
        tab = pd.concat([sim['cells'].reset_index(drop=True), tab.reset_index(drop=True)], axis=1)
        tab['seed'] = seed
        tab['median_leg_s'] = float(np.median(c['leg_duration_s']))
        tab['p90_leg_s'] = float(np.percentile(c['leg_duration_s'], 90))
        pbs = peaks_by_state(c['curves'], c['leg_state'], c['leg_duration_s'], sigma=sg)
        for s, v in pbs.items():
            tab[f'peak_state{s}'] = v['peak_phase']
            tab[f'medD_state{s}'] = v['median_duration_s']
        tabs.append(tab)
    return pd.concat(tabs, ignore_index=True)


def null_curves(sim_table, *, q=(0.25, 0.5, 0.75)):
    """The expected (peak, width) per planted cell type, pooled over recdays and seeds:
    median and IQR of the recovered peak phase and half-max width for every (kind, param)."""
    keys = ['kind', 'param', 'param2']
    tot = sim_table.groupby(keys, dropna=False).size().rename('n_total').reset_index()
    d = sim_table.dropna(subset=['width_hm'])
    g = d.groupby(keys, dropna=False)
    out = g.agg(n=('width_hm', 'size'),
                peak_med=('peak_phase', 'median'), peak_lo=('peak_phase', lambda v: v.quantile(q[0])),
                peak_hi=('peak_phase', lambda v: v.quantile(q[2])),
                width_med=('width_hm', 'median'), width_lo=('width_hm', lambda v: v.quantile(q[0])),
                width_hi=('width_hm', lambda v: v.quantile(q[2])),
                csd_med=('width_csd', 'median')).reset_index()
    out = tot.merge(out, on=keys, how='left')          # a group with no defined width keeps its row
    out['n'] = out['n'].fillna(0).astype(int)
    out['frac_not_reproduced'] = 1.0 - out['n'] / out['n_total']
    return out


# ---------------------------------------------------------------------------
# Part A. sorted beta heatmaps
# ---------------------------------------------------------------------------

def _profile_rows(glm_results, cv_results, col_idx, regressor, *, only_significant, alpha, null,
                  p_stat, center, region_of, keep_keys=None):
    """Unit-norm beta profiles, their labels and recdays; mirrors `plot_beta_profile`'s
    selection exactly so the row count equals its legend n -- unless `keep_keys` narrows it
    further (see `plot_beta_heatmap`). Returns the rows and the count `keep_keys` removed."""
    rows, labels, rds, nrns = [], [], [], []
    n_dropped = 0
    for rd, r in cv_results.items():
        if rd not in glm_results or not glm_results[rd]:
            continue
        pv = P.p_values(r, p_stat, null)
        p = np.asarray(pv[regressor], float) if pv is not None and regressor in pv else None
        keys = sorted(glm_results[rd])
        lab = (np.asarray(region_of[rd]) if region_of is not None else np.array(['all'] * len(keys)))
        for k, nrn in enumerate(keys):
            if only_significant and p is not None and not (p[k] < alpha):
                continue
            if keep_keys is not None and (rd, int(k)) not in keep_keys:
                n_dropped += 1
                continue
            beta = beta_profile(glm_results[rd][nrn], col_idx, center)
            m = np.nanmax(np.abs(beta))
            if not np.isfinite(m) or m == 0:
                continue
            rows.append(beta / m); labels.append(str(lab[k])); rds.append(rd); nrns.append(k)
    M = np.asarray(rows) if rows else np.zeros((0, len(col_idx) + 1))
    return M, np.asarray(labels), np.asarray(rds), np.asarray(nrns), n_dropped


def sort_by_peak(M):
    """Row order: bin of highest beta (signed argmax, bin 0 included), ties broken by the
    circular centre of mass of the POSITIVE part relative to the peak, so rows within a peak bin
    order by skew (left-skewed first). Returns the permutation."""
    if not len(M):
        return np.zeros(0, int)
    n_bins = M.shape[1]
    peak = np.argmax(M, axis=1)
    pos = np.clip(M, 0, None)
    ph, _ = circ_mean_phase(pos, n_bins)
    off = ((ph - (peak + 0.5) / n_bins) + 0.5) % 1.0 - 0.5          # in (-0.5, 0.5]
    off = np.nan_to_num(off, nan=0.0)
    return np.lexsort((off, peak))


def plot_beta_heatmap(glm_results, cv_results, col_idx, regressor, *, only_significant=True,
                      alpha=0.05, null='freedman_lane', p_stat='cpd', center='reference',
                      sort='peak', region_of=None, groups=None, colors=None, out_path=None,
                      title=None, max_rows=None, keep_keys=None):
    """Neurons x bins heatmap of unit-norm reference-coded beta profiles, sorted by peak.

    Rows are neurons (`regressor`-significant by default; `only_significant=False` for all),
    columns the regressor's bins with bin 0 (the reference, 0 by construction under
    `center='reference'`; `center='mean'` subtracts each neuron's mean first). Each row is
    divided by its own max|beta|, so colour is SHAPE, not rate. Diverging `RdBu_r` centred on 0,
    symmetric limits. `region_of={recday: labels}` gives one block per region in `groups` order,
    each sorted independently, a colour strip on the left, block boundaries, n per block.

    READ WITH CARE: sorting by peak manufactures a diagonal for ANY population, noise included
    (`gp_region_heatmaps` carries the same warning). The panel shows where the peaks fall and
    how wide the bumps around them are; it is never evidence of tuning on its own. The synthetic
    gate in `gp_tuning_width_synthetics.py` plants a phase population and a noise population.

    `keep_keys`: `None` (default) reproduces `plot_beta_profile`'s selection exactly, which is what
    the primary analysis uses and what the notebook's row-count assertion checks. A set of
    `(recday, neuron)` keys narrows the rows to those neurons as well. `neuron` is the position in
    `sorted(glm_results[recday])`, which `anatomy_split.assert_glm_keys_contiguous` guarantees is the
    `Neuron_raw` row index -- the same key `assemble` puts in `T['neuron']`, so a mask built from the
    per-neuron table addresses these rows directly.

    **`keep_keys` exists for the A-B peak-disagreement variant only** (GP_TUNING_WIDTH.md 4.8), so
    that every figure in that run describes one population. Do not use it in the primary analysis:
    the gate it carries is correlated with tuning width, and narrowing a figure by it is the
    selection-on-the-outcome problem section 2.0.1 warns about.

    Returns `(fig, info)`, `info={'n_rows', 'n_per_block', 'order', 'n_dropped_by_keep_keys'}`; the
    notebook asserts `n_per_block` against the `plot_beta_profile` legend for the same arm.
    """
    _style()
    if center not in ('reference', 'mean'):
        raise ValueError(f"center must be 'reference' or 'mean', got {center!r}")
    M, labels, rds, nrns, n_dropped = _profile_rows(
        glm_results, cv_results, col_idx, regressor, only_significant=only_significant, alpha=alpha,
        null=null, p_stat=p_stat, center=center, region_of=region_of, keep_keys=keep_keys)
    n_bins = M.shape[1]
    if region_of is not None:
        order_groups = [g for g in (groups or sorted(set(labels))) if g in set(labels)]
    else:
        order_groups = ['all']
    blocks, n_per = [], {}
    for g in order_groups:
        idx = np.flatnonzero(labels == g)
        if sort == 'peak':
            idx = idx[sort_by_peak(M[idx])]
        elif sort is not None:
            raise ValueError(f"sort must be 'peak' or None, got {sort!r}")
        blocks.append(idx); n_per[g] = int(len(idx))
    order = np.concatenate(blocks) if blocks else np.zeros(0, int)
    Ms = M[order]
    if max_rows and len(Ms) > max_rows:          # thin for display only; counts stay exact
        keep = np.linspace(0, len(Ms) - 1, max_rows).astype(int)
        Ms = Ms[keep]; shown_labels = labels[order][keep]
    else:
        shown_labels = labels[order]

    # axis labels as plot_beta_profile draws them
    edges_s, scheme = [], None
    for r in cv_results.values():
        bo = (r.get('bin_occupancy') or {}).get(regressor)
        if bo is not None:
            scheme = bo.get('scheme', scheme)
            if bo.get('edges_s') is not None:
                edges_s.append(bo['edges_s'])
    if regressor == 'time_from_reward' and edges_s:
        e = np.nanmedian(np.asarray(edges_s, float), axis=0)
        lower = np.concatenate([[0.0], e[1:-1]])
        xlab = [f'{lo:.0f}' for lo in lower]
        xname = 'time from reward (s, lower bin edge' + (', median over recdays)' if scheme == 'decile' else ')')
    elif regressor in P._PROFILE_AXES:
        xname, fn = P._PROFILE_AXES[regressor]; xlab = fn(n_bins)
    else:
        xlab = [str(i) for i in range(n_bins)]; xname = f'{regressor} bin'

    h = 3.4 if region_of is None else 4.6
    fig = plt.figure(figsize=(3.6, h))
    if region_of is not None:
        gs = fig.add_gridspec(1, 2, width_ratios=[0.06, 1], wspace=0.04, left=0.16, right=0.86,
                              top=0.90, bottom=0.14)
        strip = fig.add_subplot(gs[0, 0]); a = fig.add_subplot(gs[0, 1], sharey=strip)
    else:
        a = fig.add_axes([0.20, 0.14, 0.62, 0.76]); strip = None
    v = 1.0
    im = a.imshow(Ms, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, interpolation='none')
    a.set_xticks(np.arange(n_bins)); a.set_xticklabels(xlab, fontsize=6)
    a.set_xlabel(xname)
    a.tick_params(axis='y', left=False, labelleft=(strip is None))
    if strip is not None:
        col_of = colors or {}
        rgb = np.array([mpl.colors.to_rgb(col_of.get(g, '#B4B2A9')) for g in shown_labels])
        strip.imshow(rgb[:, None, :], aspect='auto', interpolation='none')
        strip.set_xticks([]); strip.set_yticks([])
        for sp in strip.spines.values():
            sp.set_visible(False)
        # block boundaries and labels
        starts = np.cumsum([0] + [n_per[g] for g in order_groups])
        scale = len(Ms) / max(len(order), 1)
        for g, s0, s1 in zip(order_groups, starts[:-1], starts[1:]):
            y0, y1 = s0 * scale - 0.5, s1 * scale - 0.5
            if s1 < starts[-1]:
                a.axhline(y1, color='white', lw=0.6)
            strip.text(-0.7, (y0 + y1) / 2, f'{g}\n(n={n_per[g]})', ha='right', va='center',
                       fontsize=6, color=col_of.get(g, '#555555'))
        a.set_ylabel('')
    else:
        a.set_ylabel(f'neuron, sorted by peak bin (n={len(order)})')
        a.set_yticks([])
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03)
    cb.set_label(r'$\beta$ / max|$\beta$|' if center == 'reference' else r'($\beta-\bar\beta$) / max|$\cdot$|', fontsize=7)
    cb.outline.set_visible(False)
    sel = (f'{regressor}-significant (p<{alpha:g}, {null})' if only_significant else 'all neurons')
    a.set_title(title or f'{sel}; bin 0 = reference (0)' if center == 'reference' else
                title or f'{sel}; mean-centred', loc='left', fontsize=7)
    _save(fig, out_path)
    return fig, {'n_rows': int(len(order)), 'n_per_block': n_per, 'order': order,
                 'n_dropped_by_keep_keys': int(n_dropped),
                 'recday': rds[order] if len(order) else rds, 'neuron': nrns[order] if len(order) else nrns}


def stack_curves(caches, *, sigma=SIGMA):
    """Per-neuron 90-bin curves for every cached recday, as three aligned matrices.

    `A` (odd legs), `B` (even legs) and `full` (all legs), each `(n_neurons, n_bins)` and
    smoothed at `sigma` bins -- `sigma=0` gives the raw bin means, which is what the
    "unsmoothed" heatmap shows. Rows are in the same order as `assemble`'s table (recday
    ascending, neuron ascending), and the index columns are returned so the two can be
    subset together."""
    A, B, F, rds, nrns = [], [], [], [], []
    for rd in sorted(caches):
        C = caches[rd]['curves']
        a, b, _, _ = split_half(C, sigma=sigma)
        A.append(a); B.append(b)
        F.append(smooth_circular(np.nanmean(C, axis=1), sigma))
        rds.append(np.array([rd] * C.shape[0])); nrns.append(np.arange(C.shape[0]))
    return {'A': np.vstack(A), 'B': np.vstack(B), 'full': np.vstack(F),
            'recday': np.concatenate(rds), 'neuron': np.concatenate(nrns), 'sigma': sigma}


def plot_curve_heatmap(M_sort, M_show, *, labels=None, groups=None, colors=None, normalise='zscore',
                       out_path=None, title=None, xlabel='goal progress (fraction of leg)',
                       cbar_label=None, max_rows=None):
    """Neurons x 90 phase bins, rows sorted by the peak of `M_sort` and coloured from `M_show`.

    Pass the SAME matrix twice for the classic within-data picture -- and then the diagonal is
    circular, guaranteed even for noise, exactly as in `plot_beta_heatmap` and
    `gp_region_heatmaps`. Pass the odd-leg curves as `M_sort` and the even-leg curves as
    `M_show` for the cross-validated version, where a diagonal is evidence: the sort knows
    nothing about the legs being displayed.

    `normalise='zscore'` (per row, diverging colours) or `'minmax'` (per row to [0, 1]).
    `labels` splits into blocks in `groups` order with a colour strip, as `plot_beta_heatmap`."""
    _style()
    S = np.asarray(M_sort, float); D = np.asarray(M_show, float)
    if S.shape != D.shape:
        raise ValueError(f'sort and show matrices differ: {S.shape} vs {D.shape}')
    n_bins = D.shape[1]
    if normalise == 'zscore':
        mu, sd = D.mean(1, keepdims=True), D.std(1, keepdims=True)
        Z = (D - mu) / np.where(sd > 0, sd, np.nan)
        cmap, sym = 'RdBu_r', True
        cbar_label = cbar_label or 'firing rate (z per neuron)'
    elif normalise == 'minmax':
        lo, hi = D.min(1, keepdims=True), D.max(1, keepdims=True)
        Z = (D - lo) / np.where(hi > lo, hi - lo, np.nan)
        cmap, sym = 'magma', False
        cbar_label = cbar_label or 'firing rate (min-max per neuron)'
    else:
        raise ValueError("normalise must be 'zscore' or 'minmax'")
    ok = np.isfinite(Z).all(axis=1)
    Z, S = Z[ok], S[ok]
    lab = (np.asarray(labels)[ok] if labels is not None else np.array(['all'] * len(Z)))
    order_groups = ([g for g in (groups or sorted(set(lab))) if g in set(lab)]
                    if labels is not None else ['all'])
    blocks, n_per = [], {}
    for g in order_groups:
        idx = np.flatnonzero(lab == g)
        idx = idx[np.argsort(np.argmax(S[idx], axis=1), kind='stable')]
        blocks.append(idx); n_per[g] = int(len(idx))
    order = np.concatenate(blocks) if blocks else np.zeros(0, int)
    Zs = Z[order]
    shown = lab[order]
    if max_rows and len(Zs) > max_rows:
        keep = np.linspace(0, len(Zs) - 1, max_rows).astype(int)
        Zs, shown = Zs[keep], shown[keep]
    v = float(np.nanpercentile(np.abs(Zs), 99)) if sym else None
    fig = plt.figure(figsize=(3.4, 4.4 if labels is not None else 3.4))
    if labels is not None:
        gs = fig.add_gridspec(1, 2, width_ratios=[0.06, 1], wspace=0.04, left=0.18, right=0.86,
                              top=0.90, bottom=0.14)
        strip = fig.add_subplot(gs[0, 0]); a = fig.add_subplot(gs[0, 1], sharey=strip)
    else:
        a = fig.add_axes([0.18, 0.14, 0.66, 0.76]); strip = None
    im = a.imshow(Zs, aspect='auto', cmap=cmap, interpolation='none',
                  extent=[0, 1, len(Zs), 0], **({'vmin': -v, 'vmax': v} if sym else {'vmin': 0, 'vmax': 1}))
    a.set_xlabel(xlabel)
    a.tick_params(axis='y', left=False, labelleft=(strip is None))
    if strip is not None:
        col_of = colors or {}
        rgb = np.array([mpl.colors.to_rgb(col_of.get(g, '#B4B2A9')) for g in shown])
        strip.imshow(rgb[:, None, :], aspect='auto', interpolation='none')
        strip.set_xticks([]); strip.set_yticks([])
        for sp in strip.spines.values():
            sp.set_visible(False)
        starts = np.cumsum([0] + [n_per[g] for g in order_groups])
        scale = len(Zs) / max(len(order), 1)
        for g, s0, s1 in zip(order_groups, starts[:-1], starts[1:]):
            if s1 < starts[-1]:
                a.axhline(s1 * scale, color='white', lw=0.6)
            strip.text(-0.7, (s0 + s1) / 2 * scale, f'{g}\n(n={n_per[g]})', ha='right', va='center',
                       fontsize=6, color=col_of.get(g, '#555555'))
    else:
        a.set_ylabel(f'neuron, sorted by peak (n={len(order)})')
        a.set_yticks([])
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03)
    cb.set_label(cbar_label, fontsize=7); cb.outline.set_visible(False)
    if title:
        a.set_title(title, loc='left', fontsize=7)
    _save(fig, out_path)
    return fig, {'n_rows': int(len(order)), 'n_per_block': n_per, 'order': order}


def legend_n(fig_profile):
    """`{label: n}` parsed from a `plot_beta_profile` figure legend ('ENTl-deep (n=425)')."""
    import re
    out = {}
    leg = fig_profile.axes[0].get_legend()
    if leg is None:
        return out
    for t in leg.get_texts():
        m = re.match(r'(.+?) \(n=(\d+)\)$', t.get_text())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


# ---------------------------------------------------------------------------
# figures for B4
# ---------------------------------------------------------------------------

def plot_peak_vs_width(table, null, *, groups=None, colors=None, color_col='sel_split',
                       width_col='width_hm', title=None, out_path=None, planted_width=0.1,
                       ax=None, show_marginals=True, point_color=None, legend=True, overlay=True,
                       running_median=True, n_running=10):
    """Scatter of width vs peak phase per neuron, coloured by the gp-vs-tfr split (diverging,
    centred on 0), with marginal histograms. `null` is `null_curves(...)`.

    Two curves are drawn over the cloud, and the point of the figure is to compare them:
    `running_median` (default on) is the DATA's own median width per peak bin with its
    interquartile band, and `overlay` (default on) is the simulated time-cell V for both anchors
    plus the planted phase-cell line. `overlay=False` produces the `_nooverlay` twin -- the same
    data with no simulation drawn -- for looking at the measurement alone. `point_color` replaces
    the split colouring (and its colourbar) with one flat colour, which is what the per-region
    panels use."""
    _style()
    d = table.dropna(subset=['peak_phase', width_col])
    if groups is not None:
        d = d[d['group'].isin(groups)]
    if ax is None:
        fig = plt.figure(figsize=(3.6, 3.6))
        if show_marginals:
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.28], height_ratios=[0.28, 1],
                                  wspace=0.05, hspace=0.05, left=0.16, right=0.97, top=0.93, bottom=0.14)
            a = fig.add_subplot(gs[1, 0]); ax_top = fig.add_subplot(gs[0, 0], sharex=a)
            ax_right = fig.add_subplot(gs[1, 1], sharey=a)
        else:
            a = fig.add_subplot(111); ax_top = ax_right = None
    else:
        a = ax; fig = ax.figure; ax_top = ax_right = None
    c = d[color_col].to_numpy(float) if (point_color is None and color_col in d) else None
    if point_color is not None:
        a.scatter(d['peak_phase'], d[width_col], color=point_color, s=5, alpha=0.7,
                  edgecolor='none', rasterized=True, zorder=2)
    elif c is not None and np.isfinite(c).any():
        lim = np.nanpercentile(np.abs(c), 95) or 1e-6
        sc = a.scatter(d['peak_phase'], d[width_col], c=c, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       s=5, alpha=0.7, edgecolor='none', rasterized=True, zorder=2)
        cb = fig.colorbar(sc, ax=a, fraction=0.05, pad=0.02)
        cb.set_label(r'$\Delta r^2_{gp} - \Delta r^2_{tfr}$ (uniform/30)', fontsize=6)
        cb.outline.set_visible(False)
    else:
        a.scatter(d['peak_phase'], d[width_col], color=C_NEUTRAL, s=5, alpha=0.7, edgecolor='none',
                  rasterized=True, zorder=2)
    if running_median and len(d) > n_running:
        # the DATA's own width-vs-peak curve, so the reader compares two curves rather than
        # a curve against a cloud
        agg = peak_count_vs_width(d, n_bins=n_running, width_col=width_col)
        ok = agg['n_peaking'] > 0
        a.plot(agg.loc[ok, 'phase'], agg.loc[ok, 'width_median'], color='#2C2C2A', lw=1.6,
               marker='o', ms=3.5, zorder=6, label='data, median per bin')
        a.fill_between(agg.loc[ok, 'phase'], agg.loc[ok, 'width_lo'], agg.loc[ok, 'width_hi'],
                       color='#2C2C2A', alpha=0.12, lw=0, zorder=5)
    if overlay and null is not None:
        overlay_null(a, null, planted_width=planted_width, legend=legend)
    elif legend and running_median:
        a.legend(frameon=False, fontsize=5, loc='upper left')
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel('peak phase (odd legs, bin centre)')
    a.set_ylabel('half-max width (even legs, fraction of leg)')
    a.set_title(title or f'n={len(d)} cells', loc='left', fontsize=7)
    if ax_top is not None:
        ax_top.hist(d['peak_phase'], bins=np.linspace(0, 1, 21), color=C_NEUTRAL, edgecolor='none')
        ax_top.axis('off')
        ax_right.hist(d[width_col], bins=np.linspace(0, 1, 21), orientation='horizontal',
                      color=C_NEUTRAL, edgecolor='none')
        ax_right.axis('off')
    _save(fig, out_path)
    return fig


def overlay_null(a, null, *, planted_width=0.1, lw=1.2, legend=True):
    """Draw the simulated time-cell V (both anchors, median with IQR band) and the planted
    phase-cell width line on axes `a`."""
    for kind, col, lab in (('tfr', C_TIME, 'time-from-reward cells (sim)'),
                           ('ttr', C_LEAD, 'time-to-reward cells (sim)')):
        s = null[null['kind'] == kind].sort_values('peak_med')
        if len(s):
            a.plot(s['peak_med'], s['width_med'], color=col, lw=lw, zorder=4, label=lab)
            a.fill_between(s['peak_med'], s['width_lo'], s['width_hi'], color=col, alpha=0.15, lw=0, zorder=3)
    ph = null[(null['kind'] == 'phase') & np.isclose(null['param2'], planted_width)].sort_values('peak_med')
    if len(ph):
        a.plot(ph['peak_med'], ph['width_med'], color=C_PHASE, lw=lw, ls='--', zorder=4,
               label=f'phase cells, planted width {planted_width:g} (sim)')
    if legend:
        a.legend(frameon=False, fontsize=5, loc='upper left')


def peak_count_vs_width(table, *, n_bins=10, width_col='width_hm'):
    """Per goal-progress bin: how many cells peak there, and how wide those cells are.

    Returns a DataFrame with one row per bin (`bin`, `phase` centre, `n_peaking`,
    `frac_peaking`, `width_median`, `width_lo`/`width_hi` quartiles, `width_mean`) and the
    Spearman correlation across bins between the count and the median width, in `.attrs`.

    The two are not independent by construction, but neither is the relation forced: the count
    per bin says where fields sit, the width how wide they are there. Its SIGN is not predictable
    in advance -- it is the V multiplied by the peak distribution. Under the V (narrow at both
    ends of the leg, broad in the middle) a population whose peaks pile up at the leg edges gives
    a NEGATIVE count-width correlation, and one whose peaks pile up mid-leg gives a positive one.
    Measured on LEC (2026-09-08): peaks pile at the edges and rho = -0.82. So this figure is a
    joint description of the peak distribution and the V, not independent evidence; and its n is
    the number of BINS, so it carries no per-mouse inference."""
    d = table.dropna(subset=['peak_phase', width_col])
    b = np.clip((d['peak_phase'].to_numpy() * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        w = d[width_col].to_numpy()[b == k]
        rows.append({'bin': k, 'phase': (k + 0.5) / n_bins, 'n_peaking': int(len(w)),
                     'frac_peaking': len(w) / max(len(d), 1),
                     'width_median': float(np.median(w)) if len(w) else np.nan,
                     'width_lo': float(np.percentile(w, 25)) if len(w) else np.nan,
                     'width_hi': float(np.percentile(w, 75)) if len(w) else np.nan,
                     'width_mean': float(np.mean(w)) if len(w) else np.nan})
    out = pd.DataFrame(rows)
    ok = out['n_peaking'] > 0
    r = spearmanr(out.loc[ok, 'n_peaking'], out.loc[ok, 'width_median']) if ok.sum() > 2 else None
    out.attrs['spearman_count_vs_width'] = float(r.correlation) if r is not None else np.nan
    out.attrs['p'] = float(r.pvalue) if r is not None else np.nan
    out.attrs['n_cells'] = int(len(d))
    return out


def plot_peak_count_vs_width(table, *, n_bins=10, width_col='width_hm', sim_table=None,
                             groups=None, colors=None, title=None, out_path=None):
    """Left: cells peaking in each goal-progress bin (bars) with their median width over it
    (line, right axis). Right: the two against each other, one point per bin, with the Spearman
    across bins; simulated populations overlaid as open markers when `sim_table` is given.

    With `groups` the left panel draws one width line per region over the pooled bars and the
    right panel one series per region."""
    _style()
    agg = peak_count_vs_width(table, n_bins=n_bins, width_col=width_col)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.0, 2.6))
    a.bar(agg['phase'], agg['n_peaking'], width=0.9 / n_bins, color=C_STONE, edgecolor='none',
          zorder=2, label='cells peaking here')
    a.set_xlabel('peak phase (goal-progress bin)')
    a.set_ylabel('cells peaking in the bin')
    a2 = a.twinx()
    a2.spines['right'].set_visible(True)
    per_group = {}
    if groups:
        for g in groups:
            sub = table[table['group'] == g]
            if len(sub) < 10:
                continue
            ag = peak_count_vs_width(sub, n_bins=n_bins, width_col=width_col)
            per_group[g] = ag
            a2.plot(ag['phase'], ag['width_median'], color=(colors or {}).get(g, C_NEUTRAL),
                    lw=1.0, marker='o', ms=2.5, zorder=4)
    else:
        a2.plot(agg['phase'], agg['width_median'], color=C_PHASE, lw=1.4, marker='o', ms=3, zorder=4)
        a2.fill_between(agg['phase'], agg['width_lo'], agg['width_hi'], color=C_PHASE, alpha=0.15,
                        lw=0, zorder=3)
    a2.set_ylabel('median half-max width', color=C_PHASE)
    a2.set_ylim(0, 1)
    a.set_title(title or f'n={agg.attrs["n_cells"]} cells', loc='left', fontsize=7)

    if per_group:
        for g, ag in per_group.items():
            b.plot(ag['n_peaking'], ag['width_median'], 'o', ms=3,
                   color=(colors or {}).get(g, C_NEUTRAL),
                   label=f'{g} (ρ={ag.attrs["spearman_count_vs_width"]:+.2f})')
    else:
        b.scatter(agg['n_peaking'], agg['width_median'], s=22, c=agg['phase'], cmap='viridis',
                  zorder=3)
        for _, r_ in agg.iterrows():
            b.annotate(f"{r_['phase']:.2f}", (r_['n_peaking'], r_['width_median']), fontsize=5,
                       xytext=(2, 2), textcoords='offset points', color=C_NEUTRAL)
    if sim_table is not None:
        for kind, col, lab in (('tfr', C_TIME, 'sim time-from-reward'),
                               ('ttr', C_LEAD, 'sim time-to-reward'),
                               ('phase', C_PHASE, 'sim phase cells')):
            sub = sim_table[sim_table['kind'] == kind]
            if not len(sub):
                continue
            ag = peak_count_vs_width(sub, n_bins=n_bins, width_col=width_col)
            b.plot(ag['n_peaking'] / max(ag['n_peaking'].sum(), 1) * agg['n_peaking'].sum(),
                   ag['width_median'], 'x', ms=4, color=col, alpha=0.8,
                   label=f'{lab} (ρ={ag.attrs["spearman_count_vs_width"]:+.2f}, count rescaled)')
    b.set_xlabel('cells peaking in the bin')
    b.set_ylabel('median half-max width')
    b.set_ylim(0, 1)
    b.set_title(f'across bins: ρ={agg.attrs["spearman_count_vs_width"]:+.2f}, '
                f'p={agg.attrs["p"]:.3g}', loc='left', fontsize=7)
    b.legend(frameon=False, fontsize=5, loc='lower right')
    fig.tight_layout()
    _save(fig, out_path)
    return fig, agg


#: Thresholds swept by `ab_gate_sweep`, in bins of 90. `None` is the ungated primary analysis.
AB_GATE_THRESHOLDS = (None, 45, 30, 20, 15, 10)


def ab_gate_sweep(table, sim=None, thresholds=AB_GATE_THRESHOLDS, *, groups=None, min_n=8,
                  width_col='width_hm'):
    """How both correlations move as the A-B peak-disagreement gate tightens.

    The gate keeps cells whose odd-leg and even-leg peaks are within `t` bins of each other on the
    circular 90-bin axis. `None` keeps everything (the primary analysis).

    WHY THIS IS A SWEEP AND NOT A CHOICE. The offset is correlated with width -- disagreeing cells
    are the broad ones -- so the gate preferentially removes wide cells, and wide cells are
    disproportionately mid-leg, which is where the V lives. Conditioning on a variable related to
    the outcome can bias the relation either way, so the honest object is the whole curve rather
    than one cut. `sim`, if given, is gated IDENTICALLY and reported beside the real cells: gating
    the data but not the planted populations would compare a filtered measurement against an
    unfiltered prediction."""
    rows = []
    for t in thresholds:
        for label, tab in (('real', table), ('sim time cells', None if sim is None else sim[sim['kind'].isin(['tfr', 'ttr'])]),
                           ('sim phase cells', None if sim is None else sim[sim['kind'] == 'phase'])):
            if tab is None:
                continue
            d = tab.dropna(subset=['peak_phase', width_col])
            n0 = len(d)
            if t is not None:
                d = d[d['ab_offset_bins'] <= t]
            if len(d) < 5:
                continue
            off = np.abs(d['peak_phase'] - 0.5)
            rec = {'threshold_bins': ('none' if t is None else t), 'population': label,
                   'n': int(len(d)), 'kept': len(d) / max(n0, 1),
                   'median_width': float(d[width_col].median()),
                   'rho_peak_width': float(spearmanr(d['peak_phase'], d[width_col]).correlation),
                   'rho_absoff_width': float(spearmanr(off, d[width_col]).correlation)}
            if label == 'real' and 'group' in d.columns:
                rep = rho_report(d.assign(abs_peak_off_centre=off), 'abs_peak_off_centre', width_col,
                                 groups=groups, min_n=min_n)
                for _, r in rep.iterrows():
                    rec[f'V {r["group"]}'] = round(r['rho_mean_over_mice'], 3)
                    rec[f'mice {r["group"]}'] = int(r['n_mice'])
            rows.append(rec)
    return pd.DataFrame(rows)


def plot_ab_gate_sweep(sweep, *, out_path=None, width_col='width_hm'):
    """The sweep as a figure: both correlations, and the fraction of cells kept, against the gate."""
    _style()
    order = [t for t in sweep['threshold_bins'].unique()]
    x = np.arange(len(order))
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 2.4))
    for pop, col, mk in (('real', C_PHASE, 'o'), ('sim time cells', C_TIME, 's'),
                         ('sim phase cells', C_LEAD, '^')):
        s = sweep[sweep['population'] == pop].set_index('threshold_bins').reindex(order)
        if s['n'].isna().all():
            continue
        a.plot(x, s['rho_absoff_width'], color=col, lw=1.2, marker=mk, ms=3.5, label=pop)
        b.plot(x, s['rho_peak_width'], color=col, lw=1.2, marker=mk, ms=3.5, label=pop)
    for ax, lab in ((a, r'$\rho$(|peak $-$ 0.5|, width)  — the V'),
                    (b, r'$\rho$(peak, width)  — anchor asymmetry')):
        ax.axhline(0, color='#2C2C2A', lw=0.6, ls=':')
        ax.set_xticks(x); ax.set_xticklabels([str(o) for o in order])
        ax.set_xlabel('A–B peak-disagreement gate (bins of 90 kept)')
        ax.set_ylabel(lab, fontsize=6)
        ax.set_ylim(-1, 1)
    real = sweep[sweep['population'] == 'real'].set_index('threshold_bins').reindex(order)
    a2 = a.twinx(); a2.spines['right'].set_visible(True)
    a2.plot(x, real['kept'], color=C_STONE, lw=1.0, ls='--')
    a2.set_ylabel('fraction of cells kept', color=C_STONE, fontsize=6)
    a2.set_ylim(0, 1.05)
    a.legend(frameon=False, fontsize=5, loc='lower left')
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_peak_distribution_by_region(table, *, groups=None, colors=None, n_bins=10,
                                     width_col='width_hm', show_width=True, overlay=True,
                                     out_path=None, title=None):
    """One panel per region: where that region's cells peak in the leg, and how wide they are there.

    `plot_peak_count_vs_width(..., groups=...)` pools the bars over every region and draws only the
    width line per region, so the *peak distributions* cannot be compared. This is the same content
    with the aggregation undone: each panel is one region's own histogram.

    Bars are the **fraction** of that region's cells peaking in each goal-progress bin, not the
    count, so regions with very different n (ENTl-deep 519 against CA1/HPF 119) sit on the same
    axis; the count and the number of mice are in the panel title. With `show_width` a line on a
    twin axis gives that region's median half-max width per bin, as in the pooled figure. With
    `overlay` a final panel puts every region's fraction curve together.

    Read the panels with `docs/handoff/README.md` section 3 in hand: ENTl-sup is one mouse (ah08)
    and ENTm two, so a difference between those panels and the rest is a difference between animals
    as much as between regions."""
    _style()
    groups = [g for g in (groups or sorted(table['group'].dropna().unique()))
              if (table['group'] == g).any()]
    n = len(groups) + (1 if overlay else 0)
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    aggs = {}
    for i, g in enumerate(groups):
        sub = table[table['group'] == g]
        ag = peak_count_vs_width(sub, n_bins=n_bins, width_col=width_col)
        aggs[g] = ag
        a = axes[i]
        col = (colors or {}).get(g, C_NEUTRAL)
        a.bar(ag['phase'], ag['frac_peaking'], width=0.9 / n_bins, color=col, edgecolor='none',
              zorder=2)
        a.set_xlabel('peak phase', fontsize=6)
        a.set_xlim(0, 1)
        if i == 0:
            a.set_ylabel('fraction of the region\'s cells peaking')
        a.set_title(f"{g}\nn={ag.attrs['n_cells']}, {sub['mouse'].nunique()} mice", loc='left',
                    fontsize=6, color=col)
        if show_width:
            a2 = a.twinx()
            a2.spines['right'].set_visible(True)
            a2.plot(ag['phase'], ag['width_median'], color='#2C2C2A', lw=1.0, marker='o', ms=2.5,
                    zorder=4)
            a2.set_ylim(0, 1)
            a2.tick_params(labelsize=5)
            if i == len(groups) - 1 and not overlay:
                a2.set_ylabel('median half-max width', fontsize=6)
            else:
                a2.set_yticklabels([])
    if overlay:
        a = axes[-1]
        for g in groups:
            a.plot(aggs[g]['phase'], aggs[g]['frac_peaking'], lw=1.2, marker='o', ms=2.5,
                   color=(colors or {}).get(g, C_NEUTRAL), label=g)
        a.set_xlim(0, 1)
        a.set_xlabel('peak phase', fontsize=6)
        a.set_title('all regions', loc='left', fontsize=6)
        a.legend(frameon=False, fontsize=5)
    if title:
        fig.suptitle(title, x=0.01, ha='left', fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.94] if title else None)
    _save(fig, out_path)
    return fig, pd.concat([ag.assign(group=g) for g, ag in aggs.items()], ignore_index=True)


def plot_calibration(sim_table, *, out_path=None):
    """Width-estimator calibration on planted phase cells: recovered vs planted width at every
    planted peak (one line per planted width), and recovered vs planted peak."""
    _style()
    d = sim_table[sim_table['kind'] == 'phase']
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.2, 2.4))
    for wd, sub in d.groupby('param2'):
        g = sub.groupby('param')['width_hm'].agg(['median', lambda v: v.quantile(0.25), lambda v: v.quantile(0.75)])
        g.columns = ['med', 'lo', 'hi']
        a.plot(g.index, g['med'], marker='o', ms=3, lw=1, label=f'planted {wd:g}')
        a.fill_between(g.index, g['lo'], g['hi'], alpha=0.15, lw=0)
        a.axhline(wd, color=C_STONE, lw=0.5, ls=':')
    a.set_xlabel('planted peak phase'); a.set_ylabel('recovered half-max width')
    a.set_ylim(0, 0.6); a.legend(frameon=False, fontsize=5)
    a.set_title('width calibration (dotted = planted)', loc='left', fontsize=7)
    g = d.groupby('param')['peak_phase'].agg(['median', lambda v: v.quantile(0.25), lambda v: v.quantile(0.75)])
    g.columns = ['med', 'lo', 'hi']
    b.plot([0, 1], [0, 1], color=C_STONE, lw=0.6, ls='--')
    b.errorbar(g.index, g['med'], yerr=[g['med'] - g['lo'], g['hi'] - g['med']], fmt='o', ms=3,
               color=C_PHASE, lw=0.8)
    b.set_xlabel('planted peak phase'); b.set_ylabel('recovered peak phase (odd legs)')
    b.set_xlim(0, 1); b.set_ylim(0, 1)
    b.set_title('peak calibration', loc='left', fontsize=7)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_time_cell_null(null, sim_table, *, out_path=None):
    """The V: recovered width vs recovered peak for the planted time cells, both anchors, with
    the phase-cell grid flat beside it; and peak vs planted latency."""
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.2, 2.4))
    overlay_null(a, null, planted_width=0.1)
    ph = null[null['kind'] == 'phase']
    a.scatter(ph['peak_med'], ph['width_med'], s=6, color=C_PHASE, alpha=0.5, zorder=3)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel('recovered peak phase'); a.set_ylabel('recovered half-max width')
    a.set_title('planted populations through the pipeline', loc='left', fontsize=7)
    for kind, col in (('tfr', C_TIME), ('ttr', C_LEAD)):
        s = null[null['kind'] == kind].sort_values('param')
        b.errorbar(s['param'], s['peak_med'], yerr=[s['peak_med'] - s['peak_lo'], s['peak_hi'] - s['peak_med']],
                   fmt='o-', ms=3, lw=0.8, color=col)
    b.set_xlabel('planted latency / lead (s)'); b.set_ylabel('recovered peak phase')
    b.set_ylim(0, 1)
    b.set_title('time cells: phase peak vs seconds', loc='left', fontsize=7)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_rho_by_group(rep, *, colors=None, ylabel=r'Spearman $\rho$', title=None, out_path=None,
                      sim_rows=None, ax=None):
    """Per-group mean-over-mice rho with mouse points and the bootstrap CI; optional simulated
    reference values (`sim_rows={'label': rho}`) as dashed lines."""
    _style()
    fig, a = (plt.subplots(figsize=(3.8, 2.6)) if ax is None else (ax.figure, ax))
    for i, row in rep.reset_index(drop=True).iterrows():
        col = (colors or {}).get(row['group'], C_NEUTRAL)
        pm = list(row['per_mouse'].values())
        a.scatter(np.full(len(pm), i), pm, s=18, color=col, zorder=3, clip_on=False)
        a.scatter([i], [row['rho_mean_over_mice']], s=70, marker='_', color='#2C2C2A', zorder=4)
        if np.isfinite(row['ci_lo']):
            a.plot([i, i], [row['ci_lo'], row['ci_hi']], color=C_STONE, lw=1.2, zorder=2)
        a.annotate(f"{int(row['n_mice'])} m", (i, -0.95), ha='center', va='bottom', fontsize=6, color=C_NEUTRAL)
    for lab, v in (sim_rows or {}).items():
        a.axhline(v, color=C_TIME if 'time' in lab else C_PHASE, lw=0.8, ls='--')
        a.text(len(rep) - 0.5, v, lab, fontsize=5, va='bottom', ha='right')
    a.axhline(0, color='#2C2C2A', lw=0.6, ls=':')
    a.set_xticks(range(len(rep))); a.set_xticklabels(rep['group'], rotation=45, ha='right')
    a.set_ylim(-1, 1); a.set_ylabel(ylabel)
    if title:
        a.set_title(title, loc='left', fontsize=7)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# verification 1 (LEC): the rebuild against the legacy pickles
# ---------------------------------------------------------------------------

#: One entry per figure this analysis produces: the filename stem (`{prefix}_<stem>.pdf`), what is
#: on each axis, how a row/point is normalised, which cells are in it, and what it does NOT show.
#: `region` is True for a figure that has a per-region version or a per-region line.
FIGURE_GUIDE = [
    dict(stem='{p}_gp_beta_heatmap_{tag}[_all][_regions]', region=True,
         title='Sorted beta heatmap, goal progress',
         shows='One row per neuron, ten columns = the ten goal-progress bins of the GLM design, '
               'bin 0 at the left. Colour is the fitted reference-coded beta divided by that '
               "neuron's own max|beta|, on a diverging scale centred at 0 (blue below the "
               'consumption bin, red above it). Rows are sorted by the bin of highest beta, ties '
               'broken by the circular centre of mass of the positive part.',
         cells='`_all` = every neuron; otherwise only neurons whose `goal_progress` beats its '
               'Freedman-Lane null at p < 0.05 in that arm. `_regions` stacks one block per '
               'region with a colour strip and block counts.',
         units='No aggregation: one row is one unit-recording. Recdays are pooled, so a mouse '
               'with more units contributes more rows.',
         not_shows='NOT smoothed and NOT evidence of tuning. Sorting by peak produces a clean '
                   'diagonal for any population including pure noise (synthetic control 7); only '
                   'the structure AROUND the diagonal (how wide the bright band is, whether the '
                   'row has a second lobe) carries information. Bin 0 is exactly 0 for every '
                   'neuron by construction, so the colour there is not data.'),
    dict(stem='{p}_tfr_beta_heatmap_{tag}[_all][_regions]', region=True,
         title='Sorted beta heatmap, time from reward',
         shows='As above for the `time_from_reward` block; columns are seconds (lower bin edge), '
               'median over recdays under decile coding.',
         cells='tfr-significant neurons of the gp+tfr arms (four of the six).',
         units='As above.',
         not_shows='As above. Under uniform coding the late bins hold under 1 % of rows '
                   '(GLM_V3.md 9.1), so a late peak rests on very few samples.'),
    dict(stem='{p}_curve_heatmap_sigma{s}', region=False,
         title='Full-resolution curve heatmap, sorted on itself',
         shows='One row per gp-significant neuron, 90 columns = phase bins of the leg. Colour is '
               'the mean firing rate over all kept legs, z-scored per neuron. Rows sorted by that '
               "same curve's peak. `sigma0` is the raw bin means with no smoothing at all; "
               '`sigma3` is the same data after 3-bin circular smoothing.',
         cells='gp-significant in the uniform/30 arm.',
         units='One row per unit-recording, recdays pooled.',
         not_shows='The diagonal is CIRCULAR here: the sort is defined on the matrix being shown, '
                   'so it is guaranteed even for noise. Use the cross-validated version for '
                   'evidence.'),
    dict(stem='{p}_curve_heatmap_xval_sigma{s}[_regions]', region=True,
         title='Cross-validated curve heatmap',
         shows='Rows sorted by the ODD legs\' peak, colours from the EVEN legs (z-scored per '
               'neuron). 90 phase bins.',
         cells='gp-significant in the uniform/30 arm; the `_regions` version blocks by region.',
         units='One row per unit-recording.',
         not_shows='Here a diagonal IS evidence -- the sort knows nothing about the legs being '
                   'displayed -- so the honest reading is how much sharper the self-sorted '
                   'version is than this one.'),
    dict(stem='{p}_width_calibration', region=False,
         title='Width-estimator calibration (simulated)',
         shows='Left: recovered half-max width against planted peak phase, one line per planted '
               'field width, with the planted value as a dotted line. Right: recovered peak phase '
               'against planted peak phase.',
         cells='Planted Poisson phase cells on this dataset\'s real legs, through the same code.',
         units='Pooled over recdays and seeds; band is the interquartile range.',
         not_shows='Nothing about the data. It is the licence to read the width axis at all: a '
                   'flat line here means the estimator has no peak-dependent bias, i.e. no edge '
                   'artefact.'),
    dict(stem='{p}_time_cell_null', region=False,
         title='The time-cell null',
         shows='Left: recovered width against recovered peak for planted fixed-latency '
               '(retrospective) and fixed-lead (prospective) cells, with the planted phase cells '
               'as a flat reference. Right: recovered peak phase against the planted latency in '
               'seconds.',
         cells='Planted cells only.',
         units='Median over recdays and seeds, IQR band.',
         not_shows='This is the null, not a result. The V it traces is what time coding alone '
                   'produces once time is divided by a variable leg duration.'),
    dict(stem='{p}_peak_vs_width[_nooverlay][_by_region][_gponly]', region=True,
         title='Peak phase against tuning width',
         shows='One point per neuron: x = peak phase from the odd legs, y = half-max width from '
               'the even legs. Colour is the gp-vs-tfr split of the selecting arm '
               '(red leans phase, blue leans absolute time). Marginal histograms on the pooled '
               'version. The overlay is the simulated time-cell V and the planted phase-cell '
               'line; `_nooverlay` is the same data with no simulation drawn.',
         cells='gp-significant in the uniform/30 arm and with a defined width; `_gponly` repeats '
               'it for the gp-only/30 significant set.',
         units='One point per unit-recording -- this panel is descriptive. The inference is the '
               'per-mouse rho figures.',
         not_shows='A relation here is NOT evidence of anything until it is compared with the '
                   'overlaid V, which time coding produces on its own.'),
    dict(stem='{p}_peak_count_vs_width', region=True,
         title='How many cells peak in a bin against how wide they are',
         shows='Left: bars = number of gp-significant cells whose peak falls in each of ten '
               'goal-progress bins; line (right axis) = the median half-max width of exactly '
               'those cells, one line per region when regions are drawn. Right: the same two '
               'quantities against each other, one point per bin, labelled by phase, with the '
               'Spearman across bins and the simulated populations as crosses.',
         cells='gp-significant in the uniform/30 arm with a defined width.',
         units='Bins pool every unit-recording; the correlation is across the ten bins, not '
               'across cells, so its n is 10 and it carries no per-mouse inference.',
         not_shows='Count and width are two summaries of the same cells: this correlation is the '
                   'V multiplied by the peak distribution, so its sign follows from where peaks '
                   'pile up (at the leg edges in this data, hence negative) and it is not '
                   'independent evidence for anything.'),
    dict(stem='{p}_peak_distribution_by_region', region=True,
         title='Peak distribution, one panel per region',
         shows='One panel per region. Bars are the fraction of THAT region\'s cells peaking in each '
               'of ten goal-progress bins — a fraction, not a count, so regions with very different '
               'n share an axis. The black line on the right axis is that region\'s median half-max '
               'width per bin. A final panel overlays every region\'s fraction curve.',
         cells='gp-significant in the uniform/30 arm with a defined width.',
         units='Bars pool every unit-recording of the region; the panel title gives n cells and n '
               'mice. No per-mouse inference is drawn here — it is a description.',
         not_shows='The companion figure `{p}_peak_count_vs_width` pools the bars over all regions '
                   'and is the one to read for the count-width relation; this one exists to compare '
                   'the peak DISTRIBUTIONS, which that figure hides. A difference between the '
                   'single-mouse panels (ENTl-sup, and ENTm on two) and the rest is a difference '
                   'between animals as much as between regions.'),
    dict(stem='{p}_rho_peak_phase / {p}_rho_abs_peak_off_centre', region=True,
         title='The two correlations, per region, mice as the unit',
         shows='Per region: each point is one mouse (its recdays\' Spearman rho averaged), the '
               'wide dash the mean over mice, the vertical bar a mouse-resampling bootstrap CI, '
               'the number below the count of mice. Dashed horizontal lines are the same '
               'statistic on the simulated time-cell and phase-cell populations.',
         cells='gp-significant in the uniform/30 arm; a (recday, region) cell needs at least '
               '8 neurons to contribute.',
         units='THIS is the inference panel: neuron -> recday rho -> mouse mean -> mean over '
               'mice.',
         not_shows='A region with one mouse is descriptive and labelled so. rho(peak, width) is '
                   'the anchor asymmetry; rho(|peak - 0.5|, width) is the V.'),
    dict(stem='{p}_peak_stability', region=True,
         title='Cross-session peak stability',
         shows='Left: distribution of the maximum pairwise circular distance between a cell\'s '
               'per-session peaks, in bins of 90, with the 30-bin flag line. Right: that '
               'statistic against the circular SD of the same peaks, showing the SD saturating.',
         cells='gp-significant cells with peaks in at least two sessions.',
         units='One point per unit-recording; the per-region fractions are tabulated beside it.',
         not_shows='A large value is not necessarily remapping: it also arises when a cell is '
                   'weakly tuned and its peak is noise. Read it with the split-half r.'),
    dict(stem='{p}_beta_vs_curve', region=False,
         title='Beta route against curve route',
         shows='Three scatters: beta peak against curve peak, beta width against curve width, and '
               'beta width against beta peak. One point per neuron.',
         cells='gp-significant in the uniform/30 arm with both routes defined.',
         units='Per unit-recording.',
         not_shows='Disagreement is a result, not an error: the betas are what goal progress '
                   'explains GIVEN place, speed, acceleration and time from reward, while the '
                   'curves are the raw rate.'),
]


def write_region_reports(table, sim, out_dir, *, prefix, groups=None, colors=None, n_bins=10,
                         null=None, extra_note=''):
    """One markdown file per region (plus `_index.md`) describing exactly what every figure shows
    for that region, with that region's own measured numbers filled in.

    Written from the executed table so the guide cannot drift from the figures. `table` is
    `assemble`'s output restricted to the cells the figures show; `sim` is the simulated-cell
    table at the analysis sigma."""
    os.makedirs(out_dir, exist_ok=True)
    groups = groups or sorted(table['group'].dropna().unique())
    sim_ref = {}
    for lab, sub in (('time cells (both anchors)', sim[sim['kind'].isin(['tfr', 'ttr'])]),
                     ('retrospective only', sim[sim['kind'] == 'tfr']),
                     ('prospective only', sim[sim['kind'] == 'ttr']),
                     ('phase cells', sim[sim['kind'] == 'phase']),
                     ('noise cells', sim[sim['kind'] == 'noise'])):
        sub = sub.dropna(subset=['width_hm'])
        if len(sub) < 5:
            continue
        sim_ref[lab] = (float(spearmanr(sub['peak_phase'], sub['width_hm']).correlation),
                        float(spearmanr(np.abs(sub['peak_phase'] - 0.5), sub['width_hm']).correlation),
                        float(sub['width_hm'].median()))
    single_mouse = {}
    try:
        single_mouse = _asp().SINGLE_MOUSE_GROUPS
    except Exception:                                            # noqa: BLE001 - PFC has no anatomy
        pass
    written = []
    for g in groups:
        d = table[table['group'] == g]
        if not len(d):
            continue
        dw = d.dropna(subset=['width_hm'])
        rep_a = rho_report(d, 'peak_phase', 'width_hm', groups=[g])
        rep_b = rho_report(d, 'abs_peak_off_centre', 'width_hm', groups=[g])
        cw = peak_count_vs_width(d, n_bins=n_bins)
        lines = [f'# {g} — what each W5 figure shows', '',
                 f'Generated by `gp_tuning_width.write_region_reports` on '
                 f'{time.strftime("%Y-%m-%d %H:%M")}, from the executed notebook. Figures live in '
                 f'`{FIG_DIR}`; the method is `code/GP_TUNING_WIDTH.md`, the plan '
                 f'`docs/handoff/W5_gp_tuning_width.md`.', '']
        if g in single_mouse:
            lines += [f'> **Single-mouse region.** {g} is dominated by {single_mouse[g]}. Every '
                      f'number here is that animal wearing a region label; it grounds no contrast.', '']
        if extra_note:
            lines += [extra_note, '']
        lines += ['## This region in one table', '',
                  '| | |', '|---|---|',
                  f'| unit-recordings shown | {len(d)} |',
                  f'| mice / recdays | {d["mouse"].nunique()} / {d["recday"].nunique()} |',
                  f'| width defined (peak reproduced in the held-out half) | {np.isfinite(d["width_hm"]).mean():.1%} |',
                  f'| median split-half r | {d["splithalf_r"].median():.3f} |',
                  f'| median peak phase | {d["peak_phase"].median():.3f} |',
                  f'| median half-max width (fraction of leg) | {dw["width_hm"].median():.3f} |',
                  f'| median circular-SD width | {d["width_csd"].median():.3f} |',
                  f'| multimodal / ramp | {d["multimodal"].mean():.1%} / {d["ramp"].mean():.1%} |',
                  f'| peak moves > {REMAP_BINS} bins across sessions | {d["remaps"].mean():.1%} |',
                  f'| median firing rate (Hz) | {d["mean_rate_hz"].median():.2f} |', '',
                  '## The two correlations for this region', '',
                  '| statistic | mean over mice | 95 % CI (mouse bootstrap) | mice | per mouse |',
                  '|---|---|---|---|---|']
        for name, rep in (('ρ(peak, width) — anchor asymmetry', rep_a),
                          ('ρ(\\|peak − 0.5\\|, width) — the V', rep_b)):
            if len(rep):
                r0 = rep.iloc[0]
                ci = ('—' if not np.isfinite(r0['ci_lo'])
                      else f"[{r0['ci_lo']:+.3f}, {r0['ci_hi']:+.3f}]")
                lines.append(f"| {name} | {r0['rho_mean_over_mice']:+.3f} | {ci} | "
                             f"{int(r0['n_mice'])} | {r0['per_mouse']} |")
        lines += ['', 'For comparison, the same statistics on the planted populations that went '
                      'through this exact code:', '',
                  '| planted population | ρ(peak, width) | ρ(\\|peak − 0.5\\|, width) | median width |',
                  '|---|---|---|---|']
        for lab, (ra, rb, mw) in sim_ref.items():
            lines.append(f'| {lab} | {ra:+.3f} | {rb:+.3f} | {mw:.3f} |')
        lines += ['', '## Where this region\'s cells peak, and how wide they are there', '',
                  '| goal-progress bin | cells peaking | share | median width | IQR |',
                  '|---|---|---|---|---|']
        for _, r_ in cw.iterrows():
            lines.append(f"| {r_['bin']} ({r_['phase']:.2f}) | {int(r_['n_peaking'])} | "
                         f"{r_['frac_peaking']:.1%} | "
                         + (f"{r_['width_median']:.3f} | [{r_['width_lo']:.3f}, {r_['width_hi']:.3f}] |"
                            if np.isfinite(r_['width_median']) else '— | — |'))
        lines += ['', f"Spearman across the {n_bins} bins between the count and the median width: "
                      f"**{cw.attrs['spearman_count_vs_width']:+.3f}** (p = {cw.attrs['p']:.3g}). "
                      f"n here is {n_bins} bins, not {len(dw)} cells.", '',
                  '## The figures, one by one', '']
        for f in FIGURE_GUIDE:
            stem = f['stem'].replace('{p}', prefix)
            lines += [f"### {f['title']}", '', f"`{stem}.pdf`"
                      + ('' if f['region'] else '  — pooled only, this region is not separated in it'),
                      '', f"**Shows.** {f['shows']}", '', f"**Which cells.** {f['cells']}", '',
                      f"**Unit of inference.** {f['units']}", '',
                      f"**What it does not show.** {f['not_shows']}", '']
        p = os.path.join(out_dir, f"{g.replace('/', '-')}.md")
        with open(p, 'w') as fh:
            fh.write('\n'.join(lines))
        written.append(p)
    idx = ['# W5 figure guide, by region', '',
           f'Written {time.strftime("%Y-%m-%d %H:%M")} from the executed notebook. One file per '
           f'region: what every figure shows, which cells are in it, what it is not evidence for, '
           f'and that region\'s own numbers.', '']
    for p in written:
        g = os.path.basename(p)[:-3]
        d = table[table['group'] == g.replace('-', '/')] if g.replace('-', '/') in set(table['group']) else table[table['group'] == g]
        idx.append(f'- [{g}]({os.path.basename(p)}) — {len(d)} unit-recordings, '
                   f'{d["mouse"].nunique()} mice')
    ip = os.path.join(out_dir, '_index.md')
    with open(ip, 'w') as fh:
        fh.write('\n'.join(idx) + '\n')
    return written + [ip]


def verify_against_legacy(data_dic, *, recdays=('ah10_20250616_20250617', 'ah10_20250618_20250619'),
                          norm_path=None, gp_path=None, verbose=True):
    """The rebuild against `norm_neurons_dic.pkl` and `gp_curves.pkl`, per session, per leg.

    Two comparisons, both reported:
      * `short_leg_rule='legacy_div10'` -- must reproduce the legacy arrays EXACTLY (the legacy
        pipeline's own convention), which is what proves the leg partition, the rebinning and
        the (trial, state) ordering agree;
      * `short_leg_rule='rate'` (the analysis default) -- must differ from them ONLY on legs of
        fewer than 90 samples, whose count is reported.

    `norm_neurons_dic[rd]` is a LIST over sessions that have trials, in session order but NOT
    indexed by session number (plan section 4); entry *i* here is the *i*-th such session, and
    the leg order within an entry is (trial, state).

    NOTE ON CURRENCY (measured 2026-09-08, correcting the plan): the 6 Sep salvage of
    `ah10_20250618_20250619` session 5 recovered `Locs_raw`/`XY_raw`/`HD_raw`; it did NOT change
    `Neuron_raw` or `Trial_times`, which are all `norm_neurons_dic` depends on. So that pickle's
    per-leg curves are NOT stale, and "must differ on s5" cannot be the currency test. What the
    salvage changes for us is the SESSION SET -- `get_sessions_for_glm` needs `Locs_raw`, so
    session 5 was invisible before it and is a fold now. `verify_session_set` is that test."""
    root = os.path.join(REPO, 'data', 'processed_data')
    norm_path = norm_path or os.path.join(root, 'norm_neurons_dic.pkl')
    gp_path = gp_path or os.path.join(root, 'gp_curves.pkl')
    with open(norm_path, 'rb') as fh:
        norm = pickle.load(fh)
    with open(gp_path, 'rb') as fh:
        gpc = pickle.load(fh)
    hz = 1000.0 / BIN_MS
    report = {}
    for rd in recdays:
        with_trials = [s for s in sorted(data_dic[rd])
                       if data_dic[rd][s].get('Neuron_raw') is not None
                       and data_dic[rd][s].get('Trial_times') is not None
                       and np.asarray(data_dic[rd][s]['Trial_times']).ndim == 2
                       and np.asarray(data_dic[rd][s]['Trial_times']).shape[0] > 0]
        entries = norm[rd]
        sess_report = {}
        for i, s in enumerate(with_trials):
            sd = data_dic[rd][s]
            tt = np.asarray(sd['Trial_times']).astype(int)
            rec = {'entry': i, 'n_trials': int(tt.shape[0])}
            C, meta = leg_phase_curves(sd['Neuron_raw'], tt, cap_s=None)                    # 'rate'
            Cl, _ = leg_phase_curves(sd['Neuron_raw'], tt, cap_s=None, short_leg_rule='legacy_div10')
            rec.update({'n_legs': int(C.shape[1]), 'n_short_legs': int(meta['n_short'])})
            if i < len(entries):
                e = np.asarray(entries[i], float)
                if e.shape[0] == C.shape[0] and e.shape[1] == tt.shape[0]:
                    legs_e = e.reshape(e.shape[0], e.shape[1] * 4, N_BINS)
                    idx = meta['trial'] * 4 + meta['state']
                    ref = legs_e[:, idx, :]
                    rec['legacy_rule_max_abs_diff'] = float(np.nanmax(np.abs(ref - Cl / hz)))
                    rec['rate_rule_max_abs_diff'] = float(np.nanmax(np.abs(ref - C / hz)))
                    d = np.abs(ref - C / hz) > 1e-6
                    rec['rate_rule_legs_differing'] = sorted(np.unique(np.where(d)[1]).tolist())
                else:
                    rec['norm_shape'] = tuple(e.shape)
            if rd in gpc and s in gpc[rd]:
                g = np.asarray(gpc[rd][s], float)
                rec['gp_curves_max_abs_diff'] = float(np.nanmax(np.abs(
                    g - smooth_circular(np.nanmean(Cl, axis=1) / hz, 10))))
            sess_report[s] = rec
            if verbose:
                print(f'  {rd} s{s}: {rec}')
        report[rd] = sess_report
    return report


def verify_session_set(data_dic, pre_path=None,
                       rd='ah10_20250618_20250619', session=5, verbose=True):
    """The currency test: the salvage added tracking to one session, which puts it INTO the
    GLM's session set (`get_sessions_for_glm` requires `Locs_raw`). Compares the session set and
    the per-session arrays of the current `data_dic` against the pre-salvage pickle."""
    pre_path = pre_path or os.path.join(REPO, 'data', 'processed_data',
                                        'data_dic_lec.pkl.PRE_salvage_ah10_20250618_20250619_s5')
    with open(pre_path, 'rb') as fh:
        pre = pickle.load(fh)
    cur_sess, _ = glm.get_sessions_for_glm(data_dic[rd])
    pre_sess, _ = glm.get_sessions_for_glm(pre[rd])
    changed = {}
    for k in ('Neuron_raw', 'Trial_times', 'Locs_raw', 'XY_raw', 'HD_raw'):
        a, b = data_dic[rd][session].get(k), pre[rd][session].get(k)
        if a is None and b is None:
            changed[k] = 'absent in both'
        elif b is None:
            changed[k] = f'ADDED by the salvage {np.shape(a)}'
        elif a is None:
            changed[k] = 'removed'
        else:
            A, B = np.asarray(a, float), np.asarray(b, float)
            changed[k] = ('same shape, identical' if A.shape == B.shape
                          and np.array_equal(np.nan_to_num(A), np.nan_to_num(B))
                          else f'differs ({A.shape} vs {B.shape})')
    out = {'recday': rd, 'session': session, 'sessions_now': sorted(cur_sess),
           'sessions_pre_salvage': sorted(pre_sess),
           'session_in_glm_now': session in cur_sess, 'session_in_glm_pre': session in pre_sess,
           'arrays': changed}
    if verbose:
        for k, v in out.items():
            print(f'  {k}: {v}')
    return out


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='build per-recday leg-curve caches for this tree')
    ap.add_argument('--verify', action='store_true', help='LEC: compare the rebuild with the legacy pickles')
    ap.add_argument('--check-mirror', action='store_true')
    ap.add_argument('--recday', action='append', default=None)
    ap.add_argument('--cap-s', type=float, default=CAP_S)
    args = ap.parse_args(argv)
    if args.check_mirror:
        print('mirror OK:', assert_mirror())
    if args.verify:
        if IS_PFC:
            print('--verify is LEC-only (the legacy pickles are LEC)'); return 1
        dd = glm.load_data_dic(validate=True, apply_exclusions=True, verbose=True)
        print('\n--- rebuild vs norm_neurons_dic / gp_curves ---')
        leg = verify_against_legacy(dd)
        print('\n--- currency: the salvage and the GLM session set ---')
        ses = verify_session_set(dd)
        _ensure_writable_dir(CACHE_DIR)
        with open(os.path.join(CACHE_DIR, '_verify_legacy.pkl'), 'wb') as fh:
            pickle.dump({'legacy': leg, 'session_set': ses,
                         'built': time.strftime('%Y-%m-%d %H:%M:%S')}, fh)
        print('saved', os.path.join(CACHE_DIR, '_verify_legacy.pkl'))
    if args.build:
        build_all(cap_s=args.cap_s, recdays=args.recday)
    return 0


if __name__ == '__main__':
    sys.exit(main())
