"""Synthetic controls for `unit_timescales` and the ramp test (M3 part 2): planted Poisson units on
one real recday per dataset, through `run_unit_timescales.fit_recday`.

    python code/run_unit_timescales.py --dataset lec --gate [--quick]

| row | population | planted | must hold |
|---|---|---|---|
| T1 | tk | entry ramps, tau 300 s (the real Stage 2 median, 320 s), both signs | `entry_time` selected in >= 0.8; tau 60 and 120 s reported |
| T2 | tk | day trends | `day_time` in >= 0.8 |
| T3 | tk | reward ramps, tau 8 s | `leg_time` or `place` in >= 0.8 (`leg_time` alone reported) |
| T3b | tk | A ramps, tau 30 s | reported: `loop_time`, `leg_time`, `place` |
| T3c | tk | reward bumps, peak 1 s | reported: any of `leg_time`, `loop_time`, `place` |
| T4 | tk | place cells | `place` in >= 0.8; any time term <= 0.1 |
| T5 | tk | task offsets | `task` in >= 0.8 |
| T6 | tk | noise | any term <= 0.05 (the control for T1-T5) |
| T7 | tk | OU drift, tau 30-300 s | reported: how often drift is called a time cell |
| T8 | tk | pure exponentials in rate, tau 300 / 600 s | Tsao and Kanter fits within 1.25x of tau in >= 0.8 |
| K1 | tk | entry ramps | Kanter GLM elapsed-time p < 0.05 in >= 0.8 of unit-sessions |
| K2 | tk | noise | GLM p < 0.05 in <= 0.1; median log Fano within 0.1 of 0 |
| K3 | tk | OU drift, tau 30 s | ACF tau within 2x in >= 0.8; median log Fano > 0. 120 and 300 s are reported: the 1-90 s fit's
|    |    |                   | finite-session bias (simulated 2026-10-14, 20-min sessions: 0.75x, 0.45x, 0.32x of tau at 30, 120, 300 s) |
| K4 | tk | entry ramps vs OU drift | residual ACF tail (30-90 s) / raw: <= 0.5 for ramps, 0.75-1.5 for drift |
| K5 | ramp | slow entry ramps, tau 190-900 s | held-out accuracy > chance + 0.05 raw; <= chance + 0.05 after the shape-free division and the 60 and 120 s slow-ramp ones (240 s reported) |
| K5b | transient | entry ramps, tau 15 s | the shape-free division leaves < half the first-minute held-out advantage, the 240 s slow-ramp one >= half (60, 120 s reported) |
| K5c | drift | OU drift, every unit divided | held-out <= chance + 0.05 both ways; within-session changes < 10% |

Revised 2026-10-14 after the quick gate (LEC 3766694 failed three rows; PFC 3766695 passed all):
- T1 was 60-300 s. One linear entry_time term cannot reach adjusted R^2 +0.01 for a ramp mostly
  over in 2-3 min of a 20-min session (LEC: 60 s 0/2, 120 s 1/2, 300 s 2/2): Tsao's recipe's floor,
  now reported as the envelope.
- K4's drift window was 0.75-1.25 (LEC 1.30, PFC 0.88). The held-out model carries the average of
  the training sessions' independent drifts, ~1/(n-1) of the drift variance (1.14x at 8 sessions),
  plus slow-bank overfit; the residual tail of an unlocked unit is inflated, not removed.
- K5b demanded <= 0.05 after the shape-free division (LEC 0.069, from 0.236). Its first lag bin is
  10 s wide, so a 15 s transient is removed only at that resolution; the test is the contrast.
Revised again after the full gate (LEC 3766748): a slow ramp with tau >= 60 s removed 65% of a 15 s
transient's first-minute advantage (0.382 -> 0.132), because it still weights minute 1 most. The
slow ramp now comes at floors 60, 120 and 240 s; real data are read against this envelope.

Existing kinds come from `event_kernels_synthetics.make_synthetic`, which is not edited. Task
offsets and pure exponentials in rate are planted here.

This file is byte-identical in `code/` and `mFC_data/code/`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import event_kernels as ek                                             # noqa: E402
import event_kernels_synthetics as syn                                 # noqa: E402

POPULATIONS = ('tk', 'ramp', 'transient', 'drift')
PURE_EXP_R0_HZ = 6.0
TASK_GAIN = 3.0


def tk_specs(quick: bool) -> list:
    """(kind, params) for the T and K rows; local kinds ('task_offset', 'pure_exp') included."""
    r = 1 if quick else 2
    U = []
    for tau in (60, 120, 300):
        for sign in ('down', 'up'):
            U += [('entry_ramp', dict(tau=tau, sign=sign))] * r
    for per_hour in (-0.3, 0.3):
        U += [('day_trend', dict(per_hour=per_hour))] * (2 * r)
    U += [('reward_ramp', dict(tau=8))] * (3 * r)
    U += [('A_ramp', dict(tau=30))] * (2 * r)
    U += [('reward_bump', dict(peak=1, k=4))] * (2 * r)
    for node in (5, 1):
        U += [('place', dict(node=node))] * (1 if quick else 3)
    for tau in (30, 120, 300):
        U += [('ou_drift', dict(tau=tau, sd=0.5))] * (2 * r)
    U += [('task_offset', dict())] * (3 * r)
    for tau in (300, 600):
        U += [('pure_exp', dict(tau=tau))] * (1 if quick else 3)
    return U + [('noise', dict())] * (15 if quick else 30)


def ramp_specs(which: str, quick: bool) -> list:
    """The ramp test's populations: 40 planted units (20 when quick) plus noise."""
    n = 20 if quick else 40
    if which == 'ramp':
        taus = np.geomspace(190, 900, n)
        U = [('entry_ramp', dict(tau=float(t), sign=('down', 'up')[i % 2])) for i, t in enumerate(taus)]
    elif which == 'transient':
        U = [('entry_ramp', dict(tau=15.0, sign='down'))] * n
    elif which == 'drift':
        U = [('ou_drift', dict(tau=180.0, sd=0.8))] * n
    else:
        raise ValueError(which)
    return U + [('noise', dict())] * (n // 2)


LOCAL_KINDS = ('task_offset', 'pure_exp')


def make_population(rec: dict, sessions: list, which: str, seed: int = 0, quick: bool = False):
    """Planted copies of `sessions` and the truth table. Kinds `make_synthetic` knows are planted by
    it; local kinds are appended after, from their own random stream."""
    specs = tk_specs(quick) if which == 'tk' else ramp_specs(which, quick)
    known = [(k, p) for k, p in specs if k not in LOCAL_KINDS]
    local = [(k, p) for k, p in specs if k in LOCAL_KINDS]
    synth, truth = syn.make_synthetic(rec, sessions, known, seed=seed)
    if local:
        rng = np.random.default_rng([seed, 7])
        task0 = str(sessions[0]['task'])
        for s in synth:
            T = s['counts'].shape[1]
            t = np.arange(T) * ek.BIN_S
            rows = []
            for k, p in local:
                if k == 'task_offset':
                    lam = np.full(T, syn.BASE_HZ * (TASK_GAIN if str(s['task']) == task0 else 1.0))
                else:
                    lam = PURE_EXP_R0_HZ * np.exp(-t / p['tau'])
                rows.append(rng.poisson(lam * ek.BIN_S))
            s['counts'] = np.vstack([s['counts'], np.array(rows, dtype=np.uint16)])
        extra = pd.DataFrame([dict(unit=len(known) + i, kind=k, **p) for i, (k, p) in enumerate(local)])
        truth = pd.concat([truth, extra], ignore_index=True)
    return synth, truth


def locked_units(truth: pd.DataFrame, which: str) -> np.ndarray:
    """The units the ramp test divides: the planted entry ramps or transients; every unit for drift
    (the worst case for noise added by division)."""
    if which == 'drift':
        return truth['unit'].to_numpy(int)
    return truth.loc[truth['kind'] == 'entry_ramp', 'unit'].to_numpy(int)


# ---------------------------------------------------------------------------
# gate rows
# ---------------------------------------------------------------------------

TIME_FLAGS = ('has_day_time', 'has_entry_time', 'has_leg_time', 'has_loop_time')


def _within(est, truth, factor):
    est, truth = np.asarray(est, float), np.asarray(truth, float)
    return np.isfinite(est) & (np.abs(np.log(est / truth)) <= np.log(factor))


def tk_rows(d: dict, truth: pd.DataFrame) -> list:
    """T1-T8 and K1-K4 from one fitted 'tk' shard."""
    ts = d['tsao'].merge(truth, on='unit')
    ka = d['kanter'].merge(truth, on='unit')
    rows = []

    def add(gid, check, value, ok, must=True):
        rows.append(dict(gate=gid, check=check, value=value, passed=bool(ok) if must else True, must=must))

    def frac(mask, col):
        return float(ts.loc[mask, col].astype(bool).mean())

    k = ts['kind']
    for tau in (60, 120, 300):
        v = frac((k == 'entry_ramp') & (ts['tau'] == tau), 'has_entry_time')
        add('T1', f'entry ramps tau {tau} s: entry_time selected', v, v >= 0.8, must=tau == 300)
    v = frac(k == 'day_trend', 'has_day_time')
    add('T2', 'day trends: day_time selected', v, v >= 0.8)
    rr = k == 'reward_ramp'
    v = float((ts.loc[rr, 'has_leg_time'] | ts.loc[rr, 'has_place']).mean())
    add('T3', 'reward ramps: leg_time or place selected', v, v >= 0.8)
    add('T3', 'reward ramps: leg_time alone selected', frac(rr, 'has_leg_time'), True, must=False)
    ar = k == 'A_ramp'
    add('T3b', 'A ramps: loop_time / leg_time / place',
        f"{frac(ar, 'has_loop_time'):.2f} / {frac(ar, 'has_leg_time'):.2f} / {frac(ar, 'has_place'):.2f}", True, must=False)
    rb = k == 'reward_bump'
    v = float((ts.loc[rb, ['has_leg_time', 'has_loop_time', 'has_place']].any(axis=1)).mean())
    add('T3c', 'reward bumps: any of leg_time, loop_time, place', v, True, must=False)
    pl = k == 'place'
    v1 = frac(pl, 'has_place')
    v2 = float(ts.loc[pl, list(TIME_FLAGS)].any(axis=1).mean())
    add('T4', 'place cells: place selected / any time term', f'{v1:.2f} / {v2:.2f}', v1 >= 0.8 and v2 <= 0.1)
    v = frac(k == 'task_offset', 'has_task')
    add('T5', 'task offsets: task selected', v, v >= 0.8)
    nz = k == 'noise'
    v = float(ts.loc[nz, [c for c in ts.columns if c.startswith('has_')]].any(axis=1).mean())
    add('T6', 'noise: any term selected', v, v <= 0.05)
    ou = k == 'ou_drift'
    add('T7', 'OU drift: day_time or entry_time selected',
        float(ts.loc[ou, ['has_day_time', 'has_entry_time']].any(axis=1).mean()), True, must=False)
    pe = ts[k == 'pure_exp']
    ok_t = _within(pe.get('entry_time_tau_s', pd.Series(np.nan, index=pe.index)), pe['tau'], 1.25) & \
        pe['has_entry_time'].astype(bool).to_numpy()
    kg = ka[(ka['kind'] == 'pure_exp') & (ka['glm_p'] < 0.05)].groupby('unit').agg(tau=('tau', 'first'), est=('glm_tau_s', 'median'))
    ok_k = _within(kg['est'], kg['tau'], 1.25)
    add('T8', 'pure exponentials: Tsao entry_time tau within 1.25x / Kanter GLM tau within 1.25x',
        f'{np.mean(ok_t):.2f} / {np.mean(ok_k) if len(kg) else np.nan:.2f}',
        np.mean(ok_t) >= 0.8 and len(kg) and np.mean(ok_k) >= 0.8)
    kk = ka['kind']
    v = float((ka.loc[(kk == 'entry_ramp'), 'glm_p'] < 0.05).mean())
    add('K1', 'entry ramps: GLM elapsed-time p < 0.05 (unit-sessions)', v, v >= 0.8)
    v1 = float((ka.loc[kk == 'noise', 'glm_p'] < 0.05).mean())
    v2 = float(ka.loc[kk == 'noise', 'log_fano'].median())
    add('K2', 'noise: GLM p < 0.05 / median log Fano', f'{v1:.3f} / {v2:+.3f}', v1 <= 0.1 and abs(v2) <= 0.1)
    for tau in (30, 120, 300):
        sub = ka[(kk == 'ou_drift') & (ka['tau'] == tau)].groupby('unit')['acf_tau_s'].median()
        w = float(np.mean(_within(sub.to_numpy(), tau, 2.0))) if len(sub) else np.nan
        add('K3', f'OU drift tau {tau} s: ACF tau within 2x (median ACF tau {np.nanmedian(sub):.0f} s)', w,
            w >= 0.8, must=tau == 30)
    v = float(ka.loc[kk == 'ou_drift', 'log_fano'].median())
    add('K3', 'OU drift: median log Fano > 0', v, v > 0)
    rat = {}
    for kind in ('entry_ramp', 'ou_drift'):
        g = ka[kk == kind].groupby('unit')[['acf_tail', 'res_acf_tail']].mean()
        rat[kind] = float(g['res_acf_tail'].median() / g['acf_tail'].median())
    add('K4', 'residual / raw ACF tail: entry ramps (<= 0.5) / OU drift (0.75-1.5)',
        f"{rat['entry_ramp']:.2f} / {rat['ou_drift']:.2f}", rat['entry_ramp'] <= 0.5 and 0.75 <= rat['ou_drift'] <= 1.5)
    return rows


def ramp_summary(d: dict) -> pd.DataFrame:
    """Per input on the 'all' set: held-out and within-session accuracy, overall and in the first
    1-min epoch, pooled over sessions; chance = 1 / number of epochs."""
    rows = []
    for r in d['ramp']['knn']:
        if r['set'] != 'all':
            continue
        y = np.concatenate([p['y'] for p in r['preds']])
        w = np.concatenate([p['within'] for p in r['preds']])
        a = np.concatenate([p['across'] for p in r['preds'] if p['across'] is not None])
        ya = np.concatenate([p['y'] for p in r['preds'] if p['across'] is not None])
        rows.append(dict(input=r['input'], within=float(np.mean(w == y)), across=float(np.mean(a == ya)),
                         within_e0=float(np.mean(w[y == 0] == 0)), across_e0=float(np.mean(a[ya == 0] == 0)),
                         chance=1.0 / len(np.unique(y))))
    return pd.DataFrame(rows).set_index('input')


def ramp_rows(which: str, d: dict) -> list:
    """K5, K5b or K5c from one fitted ramp-test shard."""
    s = ramp_summary(d)
    ch = float(s['chance'].iloc[0])
    rows = []

    def add(gid, check, value, ok):
        rows.append(dict(gate=gid, check=check, value=value, passed=bool(ok), must=True))

    if which == 'ramp':
        must = ('div_entry', 'div_ramp60', 'div_ramp120')
        add('K5', 'slow ramps: held-out accuracy raw / ' + ' / '.join(must) + ' (chance %.3f)' % ch,
            ' / '.join(f"{s.loc[i, 'across']:.3f}" for i in ('raw',) + must),
            s.loc['raw', 'across'] > ch + 0.05 and all(s.loc[i, 'across'] <= ch + 0.05 for i in must))
        rows.append(dict(gate='K5', check='slow ramps: held-out accuracy after div_ramp240 (reported)',
                         value=f"{s.loc['div_ramp240', 'across']:.3f}", passed=True, must=False))
    elif which == 'transient':
        adv = {i: s.loc[i, 'across_e0'] - ch for i in s.index}
        add('K5b', 'transient: first-minute held-out advantage over chance raw / div_entry / div_ramp240',
            f"{adv['raw']:.3f} / {adv['div_entry']:.3f} / {adv['div_ramp240']:.3f}",
            adv['raw'] > 0.1 and adv['div_entry'] < 0.5 * adv['raw'] and adv['div_ramp240'] >= 0.5 * adv['raw'])
        rows.append(dict(gate='K5b', check='transient: advantage left after div_ramp60 / div_ramp120 (reported)',
                         value=f"{adv['div_ramp60']:.3f} / {adv['div_ramp120']:.3f}", passed=True, must=False))
    else:
        rel = abs(s.loc['div_entry', 'within'] - s.loc['raw', 'within']) / s.loc['raw', 'within']
        add('K5c', 'drift: held-out raw / div_entry (chance %.3f); within-session change' % ch,
            f"{s.loc['raw', 'across']:.3f} / {s.loc['div_entry', 'across']:.3f}; {rel:.3f}",
            s.loc['raw', 'across'] <= ch + 0.05 and s.loc['div_entry', 'across'] <= ch + 0.05 and rel < 0.10)
    return rows
