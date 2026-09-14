"""Synthetic controls for GLM V3, run through the REAL pipeline.

Repo practice: a synthetic must enter at the same door the data does. Here that means
replacing `Neuron_raw` in one real recday with Poisson cells built from that recday's REAL
behaviour (`Locs_raw`, `Trial_times`) and letting `glm_analysis_v3.run_glm_analysis` do the
filtering, binning, encoding, cross-validation and Freedman-Lane null -- rather than handing
the analysis a ready-made design matrix.

    python code/glm_v3_synthetics.py              # all controls, ~15 min (the data load is 4 min)
    python code/glm_v3_synthetics.py --quick      # fewer permutations / noise cells, skips control 1

This file is byte-identical in `code/` and `mFC_data/code/` (`check_mirror_parity.py`
asserts it); it detects which tree it is in and loads that dataset. The equivalence
controls compare THIS tree's v3 against THIS tree's v2.

| # | control | a failure would mean |
|---|---|---|
| 1 | v3 with default flags reproduces v2 on one recday for the production matched-13 design: every key of cv/cpd/glm/permutation results equal, max|diff| = 0 | the seed changed the numbers |
| 1b | v3's decile arm (core-5, cap 60, tfr decile via `binning_overrides`) equals v2 with the same five regressors, max|diff| = 0 | the override path touches the decile coding |
| 2 | fixed-range edges: bin centres land in bins 0..9 at cap 30 and 60; edges ignore the data and are identical from the LEC and PFC modules; `fixed_range` with `scheme='decile'` raises | the encoder or the unit conversion is wrong |
| 3 | latency cell (fires 6-9 s after reward): tfr significant, tfr >> gp, beta peak in the 6-9 s bin (uniform) / a decile overlapping 6-9 s | tfr does not capture absolute time |
| 4 | phase cell (fires at gp in [0.4, 0.6] whatever the duration): gp significant, gp >> tfr, both codings | gp does not capture phase, or the two are not separable on this behaviour |
| 5 | consumption cell (fires 0-1.5 s after reward): tfr significant with the FIRST bin dominant -- every other beta negative | consumption does not land where GLM_V3.md caveat 3 says |
| 6 | place cell: place significant, gp and tfr both < 20 % of it | the confounds do not absorb what they should |
| 7 | Poisson-noise cells: fraction p<0.05 near 0.05 for every regressor, both codings | the Freedman-Lane null is miscalibrated in the reduced design |
| 8 | the fit's `bin_occupancy` matches the trial-times measurement (uniform: rows per bin within 0.02, legs reaching each bin exact; decile: ~10 % per bin); out-of-range rows < 1 % | rows or units are wrong |
| 9 | `is_post_refit_section` is True for the six V3 names and both production names, and `_stale_or_excluded` drops nothing for them while still dropping `ly05_20250618_20250619` for a pre-refit name | the ly05 silent drop |
| 10 | `--tfr-bins 6`: 53-column full-rank design, tfr occupies the last 5 columns, `compute_tuning_arrays` accepts the override | the `n_cols_override` plumbing is inconsistent |
| 11 | gp-only design: the phase cell is gp-significant (the sanity arm works) AND the latency cell is gp-significant too, losing most of that gp variance once tfr is added -- the reading rule of GLM_V3.md caveat 9 | the gp-only arm would be read as evidence of phase-locking, which it cannot be |

Controls 1 and 1b must pin EVERY flag whose default has moved (none should) or they silently
stop testing equivalence and start testing the new defaults.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import glm_analysis_v3 as g3                               # noqa: E402
import glm_analysis_v2 as g2                               # noqa: E402
import w1_refit as w                                       # noqa: E402
import recday_registry as rr                               # noqa: E402

IS_PFC = hasattr(g3, 'build_data_dic_from_pfc')
DATASET = 'PFC' if IS_PFC else 'LEC'


def _repo_root():
    p = HERE
    for _ in range(4):
        if os.path.isdir(os.path.join(p, 'code')) and os.path.isdir(os.path.join(p, 'mFC_data')):
            return p
        p = os.path.dirname(p)
    raise RuntimeError('cannot locate the repo root from ' + HERE)


REPO = _repo_root()
OUT_DIR = os.path.join(HERE, '..', 'data', 'processed_data')
OUT = os.path.join(OUT_DIR, 'glm_v3_synthetics.pkl')

BIN_MS = 25
CORE5 = ['place', 'goal_progress', 'speed', 'acceleration', 'time_from_reward']
CORE4 = ['place', 'goal_progress', 'speed', 'acceleration']
JOINT = [('progress_or_time', ['goal_progress', 'time_from_reward'])]
CELLS = ('latency_6-9s', 'phase_0.4-0.6', 'consumption_0-1.5s', 'place_node')
LAT, PHASE, CONS, PLACE = 0, 1, 2, 3

V3_NAMES = ('core_progress_time__matched_250ms_decile_cap30s_tfrU10b',
            'core_progress_time__matched_250ms_decile_cap30s_tfrD10b',
            'core_progress_time__matched_250ms_decile_cap60s_tfrU10b',
            'core_progress_time__matched_250ms_decile_cap60s_tfrD10b',
            'core_progress_only__matched_250ms_decile_cap30s',
            'core_progress_only__matched_250ms_decile_cap60s')
PROD_NAMES = ('all_regressors__full_250ms_decile', 'all_regressors__matched_250ms_decile')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def uniform_override(cap_s, n_bins=10):
    return {'time_from_reward': {'scheme': 'uniform', 'range_s': (0.0, float(cap_s)),
                                 'n_bins': int(n_bins)}}


def decile_override(n_bins=10):
    return {'time_from_reward': {'scheme': 'decile', 'range_s': None, 'n_bins': int(n_bins)}}


def fit(glm, data, rd, regs, *, cap_s, override=None, joint=None, n_perm=100,
        num_permutations=5, seed=123):
    """One `run_glm_analysis` call with the V3 production flags; dict of artefacts."""
    np.random.seed(seed)   # the in-sample permutation loop draws from the global RNG
    kw = dict(num_permutations=num_permutations, regressors_to_include=regs,
              joint_drop_groups=joint or None, filter_correct_paths=False,
              max_transition_seconds=float(cap_s), compute_cpd=True,
              parameterization='reference_coded', downsample_factor=10, downsample_mode='bin',
              continuous_binning='decile', cross_validate=True, cv_n_perm=int(n_perm),
              cv_nulls=('freedman_lane',), cv_center_within_sessions=True)
    if override is not None:
        kw['binning_overrides'] = override
    out = glm.run_glm_analysis([rd], data, **kw)
    return dict(zip(['glm_results', 'permutation_results', 'cpd_results', 'cv_results'], out))


def max_abs_diff(a, b, path='', ignore=()):
    """Max |a - b| over every numeric leaf; raises on a structural mismatch.

    `ignore` names dict keys that are not part of the result: the V3-only keys (`engine`,
    `bin_occupancy`, `n_rows`) and `elapsed_s`, the wall-clock time `cv_scores` records, which
    differs between any two runs and once made controls 1 and 1b report a spurious 11-second
    "difference" between numerically identical fits.
    """
    if isinstance(a, dict):
        keys = [k for k in a if k not in ignore]
        missing = [k for k in keys if k not in b]
        if missing:
            raise AssertionError(f'{path}: keys missing in the second result: {missing}')
        return max([max_abs_diff(a[k], b[k], f'{path}/{k}', ignore) for k in keys],
                   default=0.0)
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError(f'{path}: length {len(a)} vs {len(b)}')
        return max([max_abs_diff(x, y, f'{path}[{i}]', ignore)
                    for i, (x, y) in enumerate(zip(a, b))], default=0.0)
    if a is None or isinstance(a, (str, bool)):
        if a != b:
            raise AssertionError(f'{path}: {a!r} vs {b!r}')
        return 0.0
    A, B = np.atleast_1d(np.asarray(a, dtype=float)), np.atleast_1d(np.asarray(b, dtype=float))
    if A.shape != B.shape:
        raise AssertionError(f'{path}: shape {A.shape} vs {B.shape}')
    if A.size == 0:
        return 0.0
    d = np.abs(A - B)
    d[np.isnan(A) & np.isnan(B)] = 0.0     # NaN in both places is agreement
    if np.isnan(d).any():
        raise AssertionError(f'{path}: NaN in one result but not the other')
    return float(np.max(d))


def load_real_recday(recday=None):
    """One real recday of this tree's dataset, as a `{recday: {session: ...}}` data_dic."""
    if IS_PFC:
        rds = w.pfc_recdays()
        rd = recday or rds[0]
        dd = g3.build_data_dic_from_pfc(w.DATA_FOLDER, [rd], verbose=False)
    else:
        dd_all = g3.load_data_dic(validate=True, apply_exclusions=True, verbose=False)
        rd = recday or ('ah10_20250616_20250617' if 'ah10_20250616_20250617' in dd_all
                        else sorted(dd_all)[0])
        dd = {rd: dd_all[rd]}
        del dd_all
    return rd, dd


def synthetic_recday(data, rd, *, n_noise=60, base_hz=1.0, amp_hz=15.0, noise_hz=2.0, seed=0):
    """Replace `Neuron_raw` in every GLM session of `rd` with Poisson cells driven by that
    session's REAL behaviour. Cells 0-3 are the named cells (CELLS), the rest constant-rate
    noise. Returns (synthetic data_dic, place node used)."""
    rng = np.random.default_rng(seed)
    sessions, _ = g3.get_sessions_for_glm(data[rd])
    # the most-visited node across the GLM sessions is the place cell's field
    counts = np.zeros(10)
    for s in sessions:
        locs = np.asarray(data[rd][s]['Locs_raw'], dtype=float)
        ok = np.isfinite(locs) & (locs >= 1) & (locs <= 9)
        counts += np.bincount(locs[ok].astype(int), minlength=10)[:10]
    node = int(np.argmax(counts))
    syn = {}
    for s in data[rd]:
        sd = data[rd][s]
        if s not in sessions:
            syn[s] = sd
            continue
        T = int(sd['Neuron_raw'].shape[1])
        tt = np.asarray(sd['Trial_times']).astype(int)
        _, gp, _, tf, _ = g3.compute_task_state_arrays(tt, num_bins=10)
        n = min(T, len(tf))
        tf_s = np.full(T, np.nan); tf_s[:n] = tf[:n] * BIN_MS / 1000.0
        gp_c = np.full(T, np.nan); gp_c[:n] = gp[:n]
        locs = np.full(T, np.nan)
        L = np.asarray(sd['Locs_raw'], dtype=float)
        m = min(T, len(L)); locs[:m] = L[:m]
        rates = np.full((4 + n_noise, T), noise_hz)
        rates[LAT] = base_hz + amp_hz * ((tf_s >= 6.0) & (tf_s < 9.0))
        rates[PHASE] = base_hz + amp_hz * ((gp_c >= 0.4) & (gp_c < 0.6))
        rates[CONS] = base_hz + amp_hz * (tf_s < 1.5)
        rates[PLACE] = base_hz + amp_hz * (locs == node)
        new = dict(sd)
        new['Neuron_raw'] = rng.poisson(rates * BIN_MS / 1000.0).astype(np.uint16)
        syn[s] = new
    return {rd: syn}, node


def expected_occupancy(data, rd, cap_s, n_bins=10):
    """Trial-times measurement of the uniform-bin occupancy (raw-rate samples in kept legs)
    and the legs reaching each bin -- the same quantities `run_glm_analysis` stores."""
    sessions, _ = g3.get_sessions_for_glm(data[rd])
    capb = int(round(cap_s * 1000.0 / BIN_MS))
    edges = np.linspace(0, capb, n_bins + 1)
    rows = np.zeros(n_bins); durs = []
    for s in sessions:
        sd = data[rd][s]
        tt = np.asarray(sd['Trial_times']).astype(int)
        _, _, _, tf, _ = g3.compute_task_state_arrays(tt, num_bins=10)
        locs = sd['Locs_raw']
        mask, _ = g3.compute_transition_filter_mask(tt, locs, sd['Task'],
                                                    require_shortest_path=False,
                                                    max_duration_bins=capb)
        n = min(len(mask), len(tf))
        v = np.asarray(tf[:n], float)[mask[:n]]
        idx = np.clip(np.digitize(v, edges) - 1, 0, n_bins - 1)
        rows += np.bincount(idx, minlength=n_bins)[:n_bins]
        b = np.sort(tt.flatten()); d = np.diff(b); d = d[d > 0]
        durs.extend(d[d <= capb].tolist())
    durs = np.asarray(durs, float)
    return rows / rows.sum(), [int((durs > e).sum()) for e in edges[:-1]], int(len(durs))


def _cv(F, rd):
    return F['cv_results'][rd]


def _dr2(F, rd, g, i):
    return float(_cv(F, rd)['delta_r2_cv'][g][i])


def _p(F, rd, g, i):
    return float(_cv(F, rd)['p_freedman_lane__cpd'][g][i])


def _tfr_beta(F, rd, i, regs, n_bins=10):
    groups, _ = g3._resolve_regressor_groups(
        regs, parameterization='reference_coded',
        n_cols_override={'time_from_reward': n_bins} if n_bins != 10 else None)
    params = np.asarray(F['glm_results'][rd][i], float)
    return np.concatenate([[0.0], params[groups['time_from_reward']]])


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def control_1_v2_equivalence(data, rd):
    regs = w.matched_regressors(w.SECTIONS['all_regressors']['regressors'])
    joint = [j for j in w.SECTIONS['all_regressors']['joint_drop_groups']
             if all(r in regs for r in j[1])]
    a = fit(g2, data, rd, regs, cap_s=60.0, joint=joint, n_perm=3, num_permutations=3)
    b = fit(g3, data, rd, regs, cap_s=60.0, joint=joint, n_perm=3, num_permutations=3)
    d = max_abs_diff(a, b, ignore=('engine', 'bin_occupancy', 'n_rows', 'elapsed_s'))
    return d == 0.0, {'max_abs_diff': d, 'n_regressors': len(regs),
                      'n_neurons': len(a['glm_results'][rd])}


def control_1b_decile_arm_equivalence(data, rd):
    a = fit(g2, data, rd, CORE5, cap_s=60.0, joint=JOINT, n_perm=5, num_permutations=3)
    b = fit(g3, data, rd, CORE5, cap_s=60.0, joint=JOINT, n_perm=5, num_permutations=3,
            override=decile_override())
    d = max_abs_diff(a, b, ignore=('engine', 'bin_occupancy', 'n_rows', 'elapsed_s'))
    occ = _cv(b, rd)['bin_occupancy']['time_from_reward']
    dec_ok = all(0.06 <= f <= 0.14 for f in occ['frac_rows'])
    return d == 0.0 and dec_ok, {'max_abs_diff': d,
                                 'decile_frac_rows': [round(f, 3) for f in occ['frac_rows']]}


def control_2_fixed_edges():
    info = {}
    ok = True
    for cap in (30.0, 60.0):
        hi = cap * 1000.0 / BIN_MS
        e = g3.compute_decile_edges(np.zeros(3), n_bins=10, scheme='uniform', fixed_range=(0, hi))
        e2 = g3.compute_decile_edges(np.random.default_rng(1).gamma(2, 200, 5000), n_bins=10,
                                     scheme='uniform', fixed_range=(0, hi))
        centres = (np.arange(10) + 0.5) * hi / 10
        ok &= bool((g3.apply_onehot(centres, e).argmax(1) == np.arange(10)).all())
        ok &= bool(np.array_equal(e, e2))                         # edges ignore the data
        ok &= e[0] == -np.inf and e[-1] == np.inf
        info[f'edges_s_cap{cap:g}'] = [round(float(x) * BIN_MS / 1000, 2) for x in e[1:-1]]
    # the other tree's module gives the same edges
    other = os.path.join(REPO, 'code' if IS_PFC else os.path.join('mFC_data', 'code'),
                         'glm_analysis_v3.py')
    spec = importlib.util.spec_from_file_location('glm_analysis_v3_other', other)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    e_here = g3.compute_decile_edges(np.zeros(3), n_bins=10, scheme='uniform', fixed_range=(0, 1200))
    e_there = mod.compute_decile_edges(np.zeros(3), n_bins=10, scheme='uniform', fixed_range=(0, 1200))
    ok &= bool(np.array_equal(e_here, e_there))
    try:
        g3.compute_decile_edges(np.zeros(30), scheme='decile', fixed_range=(0, 1))
        ok = False; info['decile_fixed_range'] = 'did not raise'
    except ValueError:
        info['decile_fixed_range'] = 'raises'
    return ok, info


def control_3_latency(F_u, F_d, rd, occ_d):
    out = {}
    ok = True
    for lab, F in (('uniform', F_u), ('decile', F_d)):
        p_t, d_t, d_g = _p(F, rd, 'time_from_reward', LAT), _dr2(F, rd, 'time_from_reward', LAT), _dr2(F, rd, 'goal_progress', LAT)
        beta = _tfr_beta(F, rd, LAT, CORE5)
        k = int(np.argmax(beta))
        if lab == 'uniform':
            peak_ok = k == 2                              # 3 s bins at cap 30: 6-9 s is bin 2
        else:
            e = occ_d['edges_s']; lo = 0.0 if k == 0 else e[k]; hi_ = e[k + 1]
            peak_ok = (lo <= 9.0) and (hi_ >= 6.0)        # the peak decile overlaps 6-9 s
        c_ok = (p_t < 0.05) and (d_t > 3 * max(d_g, 1e-9)) and peak_ok
        ok &= c_ok
        out[lab] = {'p_tfr': p_t, 'dr2_tfr': d_t, 'dr2_gp': d_g, 'peak_bin': k, 'peak_ok': peak_ok}
    return ok, out


def control_4_phase(F_u, F_d, rd):
    out = {}; ok = True
    for lab, F in (('uniform', F_u), ('decile', F_d)):
        p_g, d_g, d_t = _p(F, rd, 'goal_progress', PHASE), _dr2(F, rd, 'goal_progress', PHASE), _dr2(F, rd, 'time_from_reward', PHASE)
        c_ok = (p_g < 0.05) and (d_g > 3 * max(d_t, 1e-9))
        ok &= c_ok
        out[lab] = {'p_gp': p_g, 'dr2_gp': d_g, 'dr2_tfr': d_t}
    return ok, out


def control_5_consumption(F_u, F_d, rd):
    out = {}; ok = True
    for lab, F in (('uniform', F_u), ('decile', F_d)):
        p_t = _p(F, rd, 'time_from_reward', CONS)
        beta = _tfr_beta(F, rd, CONS, CORE5)
        if lab == 'uniform':
            first_ok = bool(np.all(beta[1:] < 0))          # bin 0 (reference) dominates
        else:
            first_ok = bool(np.all(beta[2:] < 0)) and int(np.argmax(beta)) in (0, 1)
        c_ok = (p_t < 0.05) and first_ok
        ok &= c_ok
        out[lab] = {'p_tfr': p_t, 'beta': [round(float(b), 3) for b in beta], 'first_bin_dominant': first_ok}
    return ok, out


def control_6_place(F_u, F_d, rd):
    out = {}; ok = True
    for lab, F in (('uniform', F_u), ('decile', F_d)):
        p_pl, d_pl = _p(F, rd, 'place', PLACE), _dr2(F, rd, 'place', PLACE)
        d_g, d_t = _dr2(F, rd, 'goal_progress', PLACE), _dr2(F, rd, 'time_from_reward', PLACE)
        c_ok = (p_pl < 0.05) and (d_g < 0.2 * d_pl) and (d_t < 0.2 * d_pl)
        ok &= c_ok
        out[lab] = {'p_place': p_pl, 'dr2_place': d_pl, 'dr2_gp': d_g, 'dr2_tfr': d_t}
    return ok, out


def control_7_noise(F_u, F_d, rd, n_noise):
    out = {}; ok = True
    idx = np.arange(4, 4 + n_noise)
    per_reg_cap = 0.13 if n_noise >= 60 else 0.20
    for lab, F in (('uniform', F_u), ('decile', F_d)):
        cv = _cv(F, rd)
        fr = {g: float(np.mean(np.asarray(cv['p_freedman_lane__cpd'][g])[idx] < 0.05))
              for g in CORE5}
        pooled = float(np.mean(list(fr.values())))
        c_ok = all(v <= per_reg_cap for v in fr.values()) and 0.01 <= pooled <= 0.10
        ok &= c_ok
        centred = {g: float(np.mean(np.asarray(cv['delta_r2_cv'][g])[idx]
                                    - np.asarray(cv['null_mean_freedman_lane__delta_r2'][g])[idx]))
                   for g in CORE5}
        out[lab] = {'frac_sig': {g: round(v, 3) for g, v in fr.items()}, 'pooled': round(pooled, 3),
                    'mean_obs_minus_null': {g: f'{v:+.2e}' for g, v in centred.items()}}
    return ok, out


def control_8_occupancy(F_u, F_d, data, rd, cap_s):
    occ_u = _cv(F_u, rd)['bin_occupancy']['time_from_reward']
    occ_d = _cv(F_d, rd)['bin_occupancy']['time_from_reward']
    frac_exp, legs_exp, n_legs_exp = expected_occupancy(data, rd, cap_s)
    max_dev = float(np.max(np.abs(np.asarray(occ_u['frac_rows']) - frac_exp)))
    legs_ok = list(occ_u['legs_reaching']) == legs_exp and occ_u['n_legs'] == n_legs_exp
    out_frac = occ_u['n_out_of_range'] / max(occ_u['n_rows'], 1)
    dec_ok = all(0.06 <= f <= 0.14 for f in occ_d['frac_rows'])
    ok = (max_dev < 0.02) and legs_ok and (out_frac < 0.01) and dec_ok
    return ok, {'max_abs_dev_frac_rows': round(max_dev, 4), 'legs_reaching_exact': legs_ok,
                'n_out_of_range': occ_u['n_out_of_range'], 'n_rows': occ_u['n_rows'],
                'out_of_range_frac': round(out_frac, 5),
                'uniform_frac_rows': [round(f, 3) for f in occ_u['frac_rows']],
                'decile_frac_rows': [round(f, 3) for f in occ_d['frac_rows']],
                'decile_edges_s': [round(e, 2) for e in occ_d['edges_s'][1:-1]]}


def control_9_registry():
    ok = True; info = {}
    for n in V3_NAMES + PROD_NAMES:
        ok &= rr.is_post_refit_section(n)
        dropped = g3._stale_or_excluded(n)
        ok &= 'ly05_20250618_20250619' not in dropped
        info[n] = sorted(dropped)
    pre = g3._stale_or_excluded('all_regressors')
    ok &= (not rr.is_post_refit_section('all_regressors')) and 'ly05_20250618_20250619' in pre
    info['pre-refit all_regressors drops'] = sorted(pre)
    return ok, info


def control_10_n_cols_override(F6, rd):
    cv = _cv(F6, rd)
    groups, names = g3._resolve_regressor_groups(CORE5, parameterization='reference_coded',
                                                 n_cols_override={'time_from_reward': 6})
    n_cols = 1 + sum(len(v) for v in groups.values())
    tuned = g3.compute_tuning_arrays(F6['glm_results'], F6['permutation_results'], CORE5,
                                     parameterization='reference_coded',
                                     n_cols_override={'time_from_reward': 6})
    ok = (cv['rank']['n_cols'] == 53 and cv['rank']['full_rank'] and n_cols == 53
          and list(groups['time_from_reward']) == list(range(48, 53))
          and tuned[rd].shape[1] == 5 and len(_tfr_beta(F6, rd, LAT, CORE5, n_bins=6)) == 6)
    return ok, {'rank': cv['rank'], 'tfr_cols': list(groups['time_from_reward']),
                'tuned_shape': tuple(tuned[rd].shape),
                'occupancy_6bins': [round(f, 3) for f in cv['bin_occupancy']['time_from_reward']['frac_rows']]}


def control_11_gp_only(F_gp, F_u, rd):
    p_phase = _p(F_gp, rd, 'goal_progress', PHASE)
    p_lat = _p(F_gp, rd, 'goal_progress', LAT)
    d_lat_gp_only = _dr2(F_gp, rd, 'goal_progress', LAT)
    d_lat_with_tfr = _dr2(F_u, rd, 'goal_progress', LAT)
    ok = (p_phase < 0.05) and (p_lat < 0.05) and (d_lat_with_tfr < 0.2 * d_lat_gp_only)
    return ok, {'phase_cell_p_gp': p_phase, 'latency_cell_p_gp': p_lat,
                'latency_cell_dr2_gp_gp_only': d_lat_gp_only,
                'latency_cell_dr2_gp_with_tfr': d_lat_with_tfr}


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='30 perms, 30 noise cells, skip control 1')
    ap.add_argument('--recday', default=None)
    ap.add_argument('--skip-equivalence', action='store_true', help='skip controls 1 and 1b')
    ap.add_argument('--only-equivalence', action='store_true',
                    help='run only controls 1 and 1b (and the cheap 2 and 9); merge into the saved pickle')
    args = ap.parse_args(argv)
    n_perm = 30 if args.quick else 100
    n_noise = 30 if args.quick else 60
    cap = 30.0

    results = {}
    if args.only_equivalence and os.path.exists(OUT):
        with open(OUT, 'rb') as fh:
            results = pickle.load(fh).get('results', {})
    t0 = time.time()

    def record(name, fn, *a, **k):
        t = time.time()
        try:
            ok, info = fn(*a, **k)
        except Exception as exc:                                   # noqa: BLE001
            ok, info = False, {'error': f'{type(exc).__name__}: {exc}'}
        results[name] = {'pass': bool(ok), 'info': info, 'elapsed_s': round(time.time() - t, 1)}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({results[name]['elapsed_s']} s)")
        for kk, vv in info.items():
            print(f'        {kk}: {vv}')

    print(f'GLM V3 synthetic controls — {DATASET} tree, n_perm={n_perm}, noise cells={n_noise}')
    record('2  fixed-range edges', control_2_fixed_edges)
    record('9  registry / stale list', control_9_registry)

    rd, data = load_real_recday(args.recday)
    print(f'real recday: {rd}   ({time.time() - t0:.0f} s)')

    if not args.quick and not args.skip_equivalence:
        record('1  v3 defaults == v2 (matched-13, cap 60)', control_1_v2_equivalence, data, rd)
    if not args.skip_equivalence:
        record('1b v3 decile arm == v2 (core-5, cap 60)', control_1b_decile_arm_equivalence, data, rd)

    if args.only_equivalence:
        n_fail = sum(not r['pass'] for r in results.values())
        print('=' * 78)
        print(f"{len(results) - n_fail}/{len(results)} controls passed  ({(time.time() - t0) / 60:.1f} min)")
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT, 'wb') as fh:
            pickle.dump({'dataset': DATASET, 'recday': rd, 'n_perm': n_perm, 'n_noise': n_noise,
                         'results': results}, fh)
        print(f'saved {OUT}')
        return 1 if n_fail else 0

    syn, node = synthetic_recday(data, rd, n_noise=n_noise)
    print(f'synthetic cells built on {rd} behaviour (place node {node}); fitting 4 designs at cap {cap:g} s')
    F_u = fit(g3, syn, rd, CORE5, cap_s=cap, override=uniform_override(cap), joint=JOINT, n_perm=n_perm)
    F_d = fit(g3, syn, rd, CORE5, cap_s=cap, override=decile_override(), joint=JOINT, n_perm=n_perm)
    F_gp = fit(g3, syn, rd, CORE4, cap_s=cap, joint=None, n_perm=n_perm)
    F6 = fit(g3, syn, rd, CORE5, cap_s=cap, override=uniform_override(cap, 6), joint=JOINT, n_perm=0)
    occ_d = _cv(F_d, rd)['bin_occupancy']['time_from_reward']

    record('3  latency cell -> tfr, peak at 6-9 s', control_3_latency, F_u, F_d, rd, occ_d)
    record('4  phase cell -> gp', control_4_phase, F_u, F_d, rd)
    record('5  consumption cell -> first tfr bin', control_5_consumption, F_u, F_d, rd)
    record('6  place cell -> place only', control_6_place, F_u, F_d, rd)
    record('7  noise cells: FL null calibration', control_7_noise, F_u, F_d, rd, n_noise)
    record('8  bin occupancy == trial-times measurement', control_8_occupancy, F_u, F_d, data, rd, cap)
    record('10 n_cols_override (tfr x6)', control_10_n_cols_override, F6, rd)
    record('11 gp-only arm reading rule', control_11_gp_only, F_gp, F_u, rd)

    n_fail = sum(not r['pass'] for r in results.values())
    print('=' * 78)
    print(f"{len(results) - n_fail}/{len(results)} controls passed  ({(time.time() - t0) / 60:.1f} min)")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, 'wb') as fh:
        pickle.dump({'dataset': DATASET, 'recday': rd, 'n_perm': n_perm, 'n_noise': n_noise,
                     'results': results}, fh)
    print(f'saved {OUT}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
