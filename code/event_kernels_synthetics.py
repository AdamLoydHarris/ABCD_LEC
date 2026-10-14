"""Synthetic controls for `event_kernels`, through the real loader (WP1 gate).

One real recday's behaviour (timings, tracking, place, pokes, wall clock) enters through
`event_kernels.load_recday`; its spikes are replaced by Poisson units with planted dependencies,
and Stage 1, the reset test and Stage 2 run unchanged.

    python code/event_kernels_synthetics.py --dataset lec --quick     # ~10 min, n_perm 20
    sbatch ... event_kernels_synthetics.py --dataset lec              # full, n_perm 100, 50-80 min
    ... --kin strict                                                  # the conservative kinematic control

| # | planted | must hold |
|---|---|---|
| G1 | Poisson noise | locked (p < 0.05 and cpd > 0) not above the permutation null: one-sided binomial p >= 0.01 per block, and for any event block or the A+reward joint |
| G2 | entry ramps up and down, tau 30 s - 2 h | `ev_entry` significant in >= 80% (tau <= 30 min) |
| G3 | A-locked ramps, tau 2 - 30 s | `ev_A` or the A+reward joint in >= 80% |
| G4 | reward ramps, bumps, superposition traces, progress / distance clocks | reward or A+reward in >= 80% |
| G5 | speed-history units (tau 2 s, 128 s) | per tau, any event or the joint not above noise with the bank (one-sided Fisher p >= 0.01); >= 50% without kinematics (not vacuous) |
| G6 | OU drift, fresh path per session | `ev_entry` not above the null rate (binomial p >= 0.01): session-unique change is not a locked clock |
| G7 | day-trend units vs entry ramps | reset test: day beats entry in >= 80% of trend units, entry beats day in >= 80% of ramps |
| G8 | bank initialisation | zero-init raises spurious `ev_entry` in speed units relative to steady-state (reported) |
| G9 | place cells | place in >= 80%; any event not above noise (Fisher p >= 0.01); the rate without the place block reported |
| G10 | Stage 2 on ramps and bumps | right class in >= 70%, tau within 2x in >= 70% inside the grid |
| G11 | Freedman-Lane with a 2 min minimum shift | the local re-implementation at 0 reproduces `glm_cv` p exactly; power and FPR reported |

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binom, fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import glm_cv                                                          # noqa: E402

IS_PFC = 'mFC_data' in HERE
DEFAULT_RECDAY = {'lec': 'ah10_20250616_20250617', 'pfc': 'me11_05122021_06122021'}
BASE_HZ = 2.0
OUT_DIR = os.path.join(ek.REPO, 'data', 'event_kernels', 'synthetics')


def _versioned(path: str) -> str:
    stem, ext = os.path.splitext(path)
    k = 0
    while os.path.exists(p := stem + (f'.v{k}' if k else '') + ext):
        k += 1
    return p


# ---------------------------------------------------------------------------
# planted units
# ---------------------------------------------------------------------------

def _ou(T, tau_s, rng):
    a = np.exp(-ek.BIN_S / tau_s)
    e = rng.standard_normal(T) * np.sqrt(1 - a ** 2)
    x = np.empty(T)
    x[0] = rng.standard_normal()
    for t in range(1, T):
        x[t] = a * x[t - 1] + e[t]
    return x


def _leg_progress(sess, T):
    """(progress in the current leg 0..1, NaN outside legs) at 25 ms."""
    ev = ek.event_bins(sess, 'reward')
    prog = np.full(T, np.nan)
    for a, b in zip(ev[:-1], ev[1:]):
        if b > a:
            prog[a:b] = (np.arange(a, b) - a) / (b - a)
    return prog


def _distance_since(sess, speed, T):
    ev = np.sort(ek.event_bins(sess, 'reward'))
    d = np.full(T, np.nan)
    sp = np.nan_to_num(speed) * ek.BIN_S
    for j, a in enumerate(ev):
        b = ev[j + 1] if j + 1 < len(ev) else T
        d[a:b] = np.cumsum(sp[a:b]) - sp[a]
    return d


def unit_specs(dataset: str, quick: bool) -> list:
    """(type, params, expected) per planted unit."""
    reps = 1 if quick else 2
    U = []
    for tau in (30, 120, 480, 1800, 7200):
        for sign in ('down', 'up'):
            U += [('entry_ramp', dict(tau=tau, sign=sign))] * reps
    for tau in (2, 8, 30):
        U += [('A_ramp', dict(tau=tau))] * reps
    for tau in (0.5, 2, 8):
        U += [('reward_ramp', dict(tau=tau))] * reps
    for tau in (8, 30, 120):
        U += [('reward_superposition', dict(tau=tau))] * reps
    for peak in (1, 3, 8):
        U += [('reward_bump', dict(peak=peak, k=4))] * reps
    for peak in (2, 6):
        U += [('reward_fixed_width', dict(peak=peak, sigma=1.0))] * reps
    U += [('progress_clock', dict(centre=0.5, sigma=0.12))] * reps
    U += [('distance_clock', dict(length_cm=30.0))] * reps
    for tau in (2, 128):
        U += [('speed_history', dict(tau=tau))] * (4 * reps)
    U += [('day_trend', dict(per_hour=-0.3))] * (3 * reps)
    for tau in (60, 300):
        U += [('ou_drift', dict(tau=tau, sd=0.5))] * (2 * reps)
    for node in (5, 1):
        U += [('place', dict(node=node))] * (4 * reps)
    if dataset == 'lec':
        U += [('consumption', dict())] * reps
    U += [('noise', dict())] * (30 if quick else 60)
    return U


def leaky_specs(dataset: str) -> list:
    """Planted units for the entry-clock vs leaky-reward-count gate (`run_event_kernels --gate-leaky`):
    entry ramps and leaky reward counts at matched tau, both signs, and noise. `unit_specs` is
    untouched, so the M1 gate pickles still reproduce."""
    U = []
    for tau in (120, 300, 600, 1200):
        for sign in ('down', 'up'):
            U += [('entry_ramp', dict(tau=tau, sign=sign))] * 2
            U += [('leaky_count', dict(tau=tau, sign=sign))] * 2
    return U + [('noise', dict())] * 20


def planted_log_rate(kind, p, sess, feats, rng):
    """Unit-free modulation f(t) at 25 ms; the rate is BASE_HZ * exp(amp * f)."""
    T = sess['counts'].shape[1]
    if kind == 'entry_ramp':
        f = np.exp(-feats['t_entry'] / p['tau'])
        return f if p['sign'] == 'down' else -f
    if kind == 'A_ramp':
        return np.nan_to_num(np.exp(-feats['t_A'] / p['tau']))
    if kind == 'reward_ramp':
        return np.nan_to_num(np.exp(-feats['t_rew'] / p['tau']))
    if kind == 'reward_superposition':
        ev = ek.event_bins(sess, 'reward')
        x = np.zeros(T)
        x[ev[(ev >= 0) & (ev < T)]] = 1.0
        a = np.exp(-ek.BIN_S / p['tau'])
        from scipy.signal import lfilter
        y = lfilter([1.0], [1, -a], x)
        return y / max(y.max(), 1e-9)
    if kind == 'reward_bump':
        return ek.bump(feats['t_rew'], p['peak'], p['k'])
    if kind == 'reward_fixed_width':
        return np.nan_to_num(np.exp(-0.5 * ((feats['t_rew'] - p['peak']) / p['sigma']) ** 2))
    if kind == 'progress_clock':
        return np.nan_to_num(np.exp(-0.5 * ((feats['progress'] - p['centre']) / p['sigma']) ** 2))
    if kind == 'distance_clock':
        return np.nan_to_num(np.exp(-feats['dist'] / p['length_cm']))
    if kind == 'speed_history':
        b = ek.exp_bank(feats['speed'], (p['tau'],), init='steady')[:, 0]
        return (b - feats['speed_mean']) / feats['speed_sd']
    if kind == 'day_trend':
        return feats['hours'] * p['per_hour']
    if kind == 'ou_drift':
        return p['sd'] * _ou(T, p['tau'], rng)
    if kind == 'place':
        return (sess['locs'] == p['node']).astype(float)
    if kind == 'leaky_count':
        f = ek.leaky_trace(ek.event_bins(sess, 'reward'), T, p['tau'])   # unnormalised per session
        return f if p['sign'] == 'up' else -f
    if kind == 'consumption':
        x = np.zeros(T)
        pk = sess.get('pokes')
        if pk is not None:
            for a, b in pk[pk[:, 3] == 1][:, :2]:
                x[max(a, 0):min(b + 1, T)] = 1.0
        return x
    return np.zeros(T)


AMP = {'day_trend': 1.0, 'ou_drift': 1.0, 'speed_history': 0.5, 'noise': 0.0}


def make_synthetic(rec, sessions, specs, seed=0):
    """Copies of `sessions` with planted Poisson counts; truth table (one row per unit)."""
    rng = np.random.default_rng(seed)
    all_speed = np.concatenate([ek.kinematics(s['xy'])[0] for s in sessions])
    s_mean, s_sd = np.nanmean(all_speed), np.nanstd(all_speed)
    out = []
    for sess in sessions:
        T = sess['counts'].shape[1]
        speed = ek.kinematics(sess['xy'])[0]
        feats = dict(t_entry=np.arange(T) * ek.BIN_S, t_A=ek.time_since(ek.event_bins(sess, 'A'), T),
                     t_rew=ek.time_since(ek.event_bins(sess, 'reward'), T), speed=speed,
                     speed_mean=s_mean, speed_sd=s_sd, progress=_leg_progress(sess, T),
                     dist=_distance_since(sess, speed, T),
                     hours=sess.get('hours_since_day_start', 0.0) + np.arange(T) * ek.BIN_S / 3600.0)
        counts = np.empty((len(specs), T), dtype=np.uint16)
        for u, (kind, p) in enumerate(specs):
            f = planted_log_rate(kind, p, sess, feats, rng)
            amp = AMP.get(kind, np.log(3.0))
            lam = BASE_HZ * ek.BIN_S * np.exp(np.clip(amp * np.nan_to_num(f), -5, 5))
            counts[u] = rng.poisson(lam)
        out.append({**sess, 'counts': counts})
    truth = pd.DataFrame([dict(unit=u, kind=k, **p) for u, (k, p) in enumerate(specs)])
    return out, truth


# ---------------------------------------------------------------------------
# Freedman-Lane with a minimum shift (G11)
# ---------------------------------------------------------------------------

def fl_pvalues(X, FR, sids, group_cols, n_perm, seed, min_shift_rows=0):
    """Leave-one-session-out FL p for ONE group, as `glm_cv.cv_scores` (centred FR), with shifts
    drawn the same way when `min_shift_rows` is 0 and redrawn until >= min_shift_rows from
    identity (circularly) otherwise."""
    FRc = glm_cv.center_within_session(FR, sids)
    Xr = np.delete(X, group_cols, axis=1)
    rng = np.random.default_rng(seed)
    T = X.shape[0]
    n_s = len(np.unique(sids))
    shifts = rng.integers(0, T, size=(n_perm, n_s))
    if min_shift_rows:
        lens = np.array([np.sum(sids == s) for s in np.unique(sids)])
        for i in range(n_perm):
            for k in range(n_s):
                while True:
                    r = shifts[i, k] % lens[k]
                    if min(r, lens[k] - r) >= min_shift_rows:
                        break
                    shifts[i, k] = rng.integers(0, T)
    yhat = (Xr @ (glm_cv._pinv(Xr) @ FRc.T)).T
    resid = FRc - yhat
    obs_red = obs_full = 0.0
    null_red = np.zeros((n_perm, FR.shape[0]))
    null_full = np.zeros((n_perm, FR.shape[0]))
    for f in np.unique(sids):
        te, tr = sids == f, sids != f
        Pf, Pr = glm_cv._pinv(X[tr]), glm_cv._pinv(Xr[tr])
        Yte, Ytr = FRc[:, te].T, FRc[:, tr].T
        obs_full = obs_full + np.sum((Yte - X[te] @ (Pf @ Ytr)) ** 2, axis=0)
        obs_red = obs_red + np.sum((Yte - Xr[te] @ (Pr @ Ytr)) ** 2, axis=0)
        for i in range(n_perm):
            ys = yhat + glm_cv.roll_within_sessions(resid, sids, shifts[i])
            Ytr_, Yte_ = ys[:, tr].T, ys[:, te].T
            null_full[i] += np.sum((Yte_ - X[te] @ (Pf @ Ytr_)) ** 2, axis=0)
            null_red[i] += np.sum((Yte_ - Xr[te] @ (Pr @ Ytr_)) ** 2, axis=0)
    obs = (obs_red - obs_full) / obs_red
    nul = (null_red - null_full) / null_red
    return (1 + np.sum(nul >= obs[None, :], axis=0)) / (1 + n_perm)


# ---------------------------------------------------------------------------
# run + gates
# ---------------------------------------------------------------------------

def locked(res, g):
    """Per unit: block `g` explains held-out variance (cpd > 0) beyond its Freedman-Lane null
    (p < 0.05). At n_perm 20 the p floor is 1/21, reached by units with cpd <= 0."""
    return (res['p_freedman_lane__cpd'][g] < 0.05) & (res['cpd_cv'][g] > 0)


def alpha_null(n_perm: int) -> float:
    """Exact null rate of p < 0.05 for p = (1 + #null >= obs) / (1 + n_perm): 1/21 at 20, 5/101 at 100."""
    return float((np.ceil(0.05 * (n_perm + 1)) - 1) / (n_perm + 1))


def summarise(truth, res, label):
    """Fraction locked (p < 0.05 and cpd > 0) per (kind, block)."""
    rows = []
    blocks = [g for g in res['p_freedman_lane__cpd']]
    for kind, sub in truth.groupby('kind'):
        u = sub['unit'].to_numpy()
        row = dict(arm=label, kind=kind, n=len(u))
        for g in blocks:
            row[g] = float(np.mean(locked(res, g)[u]))
        rows.append(row)
    return pd.DataFrame(rows)


def run(dataset: str, recday: str, quick: bool, seed: int = 0, kin: str = 'full') -> dict:
    t0 = time.time()
    n_perm = 20 if quick else 100
    rec = ek.load_recday(recday, dataset)
    ek.add_day_clock(rec)
    sessions = ek.analysis_sessions(rec)
    specs = unit_specs(dataset, quick)
    syn, truth = make_synthetic(rec, sessions, specs, seed=seed)
    print(f'{dataset} {recday}: {len(syn)} sessions, {len(truth)} planted units', flush=True)

    out = dict(dataset=dataset, recday=recday, quick=quick, truth=truth, n_perm=n_perm, kin=kin)
    D = ek.build_design(syn, kin=kin)
    out['stage1'] = ek.stage1(syn, n_perm=n_perm, seed=seed, design=D)
    print(f'  stage1 done {time.time() - t0:.0f}s', flush=True)
    out['stage1_nokin'] = ek.stage1(syn, n_perm=n_perm, seed=seed, kin='none')
    print(f'  stage1 kin=none done {time.time() - t0:.0f}s', flush=True)
    out['stage1_zeroinit'] = ek.stage1(syn, n_perm=n_perm, seed=seed, kin=kin, bank_init='zero')
    print(f'  stage1 zero-init done {time.time() - t0:.0f}s', flush=True)
    out['stage1_noplace'] = ek.stage1(syn, n_perm=n_perm, seed=seed, kin=kin, place=False)
    print(f'  stage1 place=False done {time.time() - t0:.0f}s', flush=True)
    out['reset'] = ek.reset_test(rec, syn, kin=kin)
    print(f'  reset test done {time.time() - t0:.0f}s', flush=True)

    tab = pd.concat([summarise(truth, out['stage1'], 'bank'),
                     summarise(truth, out['stage1_nokin'], 'no_kin'),
                     summarise(truth, out['stage1_zeroinit'], 'zero_init'),
                     summarise(truth, out['stage1_noplace'], 'no_place')], ignore_index=True)
    out['table'] = tab

    s2 = []
    want = truth[truth['kind'].isin(['entry_ramp', 'A_ramp', 'reward_ramp', 'reward_bump',
                                     'reward_fixed_width', 'reward_superposition'])]
    if quick:
        want = want.groupby('kind').head(3)
    ev_of = {'entry_ramp': 'entry', 'A_ramp': 'A'}
    for r in want.itertuples():
        e = ev_of.get(r.kind, 'reward')
        s2.append({**ek.stage2_unit(D, int(r.unit), e), 'kind': r.kind,
                   'tau_true': getattr(r, 'tau', np.nan), 'peak_true': getattr(r, 'peak', np.nan),
                   'sign_true': getattr(r, 'sign', None)})
    out['stage2'] = pd.DataFrame(s2)
    print(f'  stage2 done {time.time() - t0:.0f}s', flush=True)

    Xc = ek.center_within_sessions(D['X'], D['session_ids'])
    cols = D['groups']['ev_entry']
    sub = truth[truth['kind'].isin(['entry_ramp', 'ou_drift', 'noise'])]['unit'].to_numpy()
    p0 = fl_pvalues(Xc, D['FR'][sub], D['session_ids'], cols, n_perm, seed)
    p2 = fl_pvalues(Xc, D['FR'][sub], D['session_ids'], cols, n_perm, seed,
                    min_shift_rows=int(120 / (ek.BIN_S * ek.BLOCK)))
    ref = glm_cv.cv_scores(Xc, D['FR'][sub], D['session_ids'], {'ev_entry': cols}, n_perm=n_perm,
                           seed=seed, center_within_sessions=True, nulls=('freedman_lane',))
    out['fl_min_shift'] = dict(units=sub, p0=p0, p2=p2, p_glm_cv=ref['p_freedman_lane__cpd']['ev_entry'])
    out['elapsed_s'] = time.time() - t0
    out['gates'] = gates(out)
    return out


def gates(out) -> pd.DataFrame:
    tab, truth = out['table'], out['truth']
    bank = tab[tab['arm'] == 'bank'].set_index('kind')
    zero = tab[tab['arm'] == 'zero_init'].set_index('kind')
    rows = []

    def add(gid, desc, value, ok):
        rows.append(dict(gate=gid, check=desc, value=value, passed=bool(ok)))

    s1, s_nk = out['stage1'], out['stage1_nokin']
    units = lambda kind, **kw: truth[(truth['kind'] == kind) & np.all(
        [truth[k] <= v for k, v in kw.items()], axis=0)]['unit'].to_numpy() if kw else \
        truth[truth['kind'] == kind]['unit'].to_numpy()
    events = [g for g in ('ev_entry', 'ev_A', 'ev_reward', 'ev_A+reward') if g in s1['cpd_cv']]

    # The A+reward joint counts as a hit in G3/G4, so it counts as a leak too: without it the
    # full PFC run passed G5 while 5/8 tau-2 s speed units were locked to the joint alone.
    def any_event(res, u):
        hit = np.zeros(len(u), bool)
        for g in events:
            hit |= locked(res, g)[u]
        return hit

    # Chance rows are exact tests: fixed cut-offs (<= 0.10 per block, noise + 0.10) failed a null
    # process 16-29% of the time at these n (PFC quick G9: 1 of 2 place units, Fisher p 0.24).
    a_null = alpha_null(out['n_perm'])
    nz = units('noise')

    def excess_p(hit, rate):
        return float(binom.sf(hit.sum() - 1, len(hit), rate))

    def vs_noise(hit, hit_noise):
        t = [[hit.sum(), (~hit).sum()], [hit_noise.sum(), (~hit_noise).sum()]]
        return float(fisher_exact(t, alternative='greater').pvalue)

    for g in events + ['kin_fast', 'kin_slow', 'place']:
        if g in s1['cpd_cv']:
            h = locked(s1, g)[nz]
            p = excess_p(h, a_null)
            add('G1', f'noise FPR {g} (binomial p vs {a_null:.3f})', f'{h.mean():.2f} (p={p:.2f})', p >= 0.01)
    hit_noise = any_event(s1, nz)
    a_any = 1 - (1 - a_null) ** len(events)
    p = excess_p(hit_noise, a_any)
    add('G1', f'noise: any event block or the A+reward joint (vs {a_any:.3f})', f'{hit_noise.mean():.2f} (p={p:.2f})',
        p >= 0.01)
    er = units('entry_ramp', tau=1800)
    v = float(locked(s1, 'ev_entry')[er].mean())
    add('G2', 'entry ramps tau <= 30 min: ev_entry', v, v >= 0.8)
    joint = 'ev_A+reward' if 'ev_A+reward' in s1['cpd_cv'] else None
    for kind, gid in (('A_ramp', 'G3'), ('reward_ramp', 'G4'), ('reward_bump', 'G4'),
                      ('reward_fixed_width', 'G4'), ('reward_superposition', 'G4'),
                      ('progress_clock', 'G4'), ('distance_clock', 'G4')):
        u = units(kind)
        if not len(u):
            continue
        hit = locked(s1, 'ev_A')[u] | locked(s1, 'ev_reward')[u]
        if joint:
            hit |= locked(s1, joint)[u]
        add(gid, f'{kind}: A, reward or joint', float(hit.mean()), hit.mean() >= 0.8)
    sp = units('speed_history')
    for tau in sorted(truth.loc[truth['kind'] == 'speed_history', 'tau'].unique()):
        h = any_event(s1, truth[(truth['kind'] == 'speed_history') & (truth['tau'] == tau)]['unit'].to_numpy())
        p = vs_noise(h, hit_noise)
        add('G5', f'speed units tau {tau:g} s: any event with the bank (Fisher p vs noise)',
            f'{h.mean():.2f} vs {hit_noise.mean():.2f} (p={p:.3f})', p >= 0.01)
    v_nk = float(any_event(s_nk, sp).mean())
    add('G5', 'speed units: any event without kinematics', v_nk, v_nk >= 0.5)
    h = locked(s1, 'ev_entry')[units('ou_drift')]
    p = excess_p(h, a_null)
    add('G6', 'OU drift: ev_entry (binomial p)', f'{h.mean():.2f} (p={p:.2f})', p >= 0.01)
    rs = out['reset']
    if 'skipped' in rs:
        add('G7', 'reset test', rs['skipped'], False)
    else:
        r = rs['entry_beats_day']
        dt = units('day_trend')
        add('G7', 'day trend: day beats entry', float(np.mean(~r[dt])), np.mean(~r[dt]) >= 0.8)
        add('G7', 'entry ramps tau <= 30 min: entry beats day', float(np.mean(r[er])), np.mean(r[er]) >= 0.8)
    if 'ev_entry' in zero.columns:
        add('G8', 'speed units ev_entry, zero init vs steady', f"{zero.loc['speed_history', 'ev_entry']:.2f} vs "
            f"{bank.loc['speed_history', 'ev_entry']:.2f}", True)
    pl = units('place')
    v = float(locked(s1, 'place')[pl].mean())
    add('G9', 'place cells: place', v, v >= 0.8)
    h = any_event(s1, pl)
    p = vs_noise(h, hit_noise)
    add('G9', 'place cells: any event (Fisher p vs noise)', f'{h.mean():.2f} vs {hit_noise.mean():.2f} (p={p:.2f})',
        p >= 0.01)
    if 'stage1_noplace' in out:
        h = any_event(out['stage1_noplace'], pl)
        add('G9', 'place cells: any event without the place block (reported)', float(h.mean()), True)
    s2 = out['stage2']
    if len(s2):
        ramps = s2[s2['kind'].isin(['entry_ramp', 'A_ramp', 'reward_ramp'])]
        want = np.where(ramps['kind'] == 'entry_ramp',
                        np.where(ramps['sign_true'] == 'down', 'ramp_down', 'ramp_up'), 'ramp_down')
        ok = ramps['shape'].to_numpy() == want
        add('G10', 'ramps: class and sign', float(ok.mean()), ok.mean() >= 0.7)
        err = np.abs(np.log2(ramps['tau_median'] / ramps['tau_true']))
        add('G10', 'ramps: tau within 2x', float(np.mean(err <= 1)), np.mean(err <= 1) >= 0.7)
        bumps = s2[s2['kind'] == 'reward_bump']
        if len(bumps):
            ok = bumps['shape'] == 'bump'
            add('G10', 'reward bumps: class', float(ok.mean()), ok.mean() >= 0.7)
    fl = out['fl_min_shift']
    same = float(np.max(np.abs(fl['p0'] - fl['p_glm_cv'])))
    add('G11', 'local FL at min shift 0 == glm_cv (max |dp|)', same, same == 0.0)
    kinds = truth.set_index('unit').loc[fl['units'], 'kind'].to_numpy()
    for kind in ('entry_ramp', 'ou_drift', 'noise'):
        m = kinds == kind
        add('G11', f'{kind}: p<0.05 at shift>=0 vs >=2 min',
            f'{np.mean(fl["p0"][m] < 0.05):.2f} vs {np.mean(fl["p2"][m] < 0.05):.2f}', True)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=('lec', 'pfc'), default='pfc' if IS_PFC else 'lec')
    ap.add_argument('--recday', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kin', choices=('full', 'strict'), default='full',
                    help='kinematic nuisance for every arm with kinematics (see build_design)')
    args = ap.parse_args(argv)
    rd = args.recday or DEFAULT_RECDAY[args.dataset]
    out = run(args.dataset, rd, args.quick, args.seed, kin=args.kin)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = ('_quick' if args.quick else '') + ('' if args.kin == 'full' else f'_kin-{args.kin}')
    p = _versioned(os.path.join(OUT_DIR, f"synthetics_{args.dataset}_{rd}{tag}.pkl"))
    with open(p, 'wb') as fh:
        pickle.dump(out, fh)
    pd.set_option('display.width', 200)
    print(out['table'].round(2).to_string(index=False))
    if len(out['stage2']):
        print(out['stage2'][['kind', 'event', 'tau_true', 'tau_median', 'peak_true', 'peak_median',
                             'shape', 'frac_ramp', 'frac_bump']].round(2).to_string(index=False))
    print(out['gates'].to_string(index=False))
    print(f'-> {p}  ({out["elapsed_s"]:.0f} s)')
    return 0 if out['gates']['passed'].all() else 1


if __name__ == '__main__':
    sys.exit(main())
