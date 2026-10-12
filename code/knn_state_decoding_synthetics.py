"""Synthetic controls for knn_state_decoding, through the real sample builder.

    python knn_state_decoding_synthetics.py [--quick]

Every case is a data_dic-shaped recday from ccgp_state_pairs.make_synthetic_recday
(trial_times='session': real-time, drifting legs), run through run_recday exactly as the data are.
Exit 0 only if every check passes, including those that must NOT reject (noise) and those that
must show a control is not vacuous (the drift artefact, the tone at trim 0).
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np

import ccgp_state_pairs as cc
import knn_state_decoding as kn

#: sha256[:16] of the legacy generator's seed-0 output, from HEAD before the session mode was
#: added; the legacy path must never move (CCGP_STATE_PAIRS.md §7 depends on it)
LEGACY_DIGEST = {'abstract': '5c097752a25208f8', 'place': '913cfee865046a6f',
                 'tone': '2a03ea898292a664', 'noise': 'cadced726aa39302'}
CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append((name, bool(ok)))
    print(f'  [{"PASS" if ok else "FAIL"}] {name}  {detail}')


def run(kind, cfg, seed=0, trim=15, **kw):
    dd = cc.make_synthetic_recday(kind, trial_times='session', seed=seed, **kw)
    c = kn.KNNConfig(**{**cfg.__dict__, 'trim_bins': trim})
    res = kn.run_recday(dd, f'synth_{kind}', c)
    rm = kn.recday_means(res['rows']).set_index(['analysis', 'pool', 'variant']).acc
    con = kn.contrasts(kn.recday_means(res['rows'])).set_index('variant')
    return res, rm, con


def legacy_digest(kind):
    dd = cc.make_synthetic_recday(kind, seed=0)
    m = hashlib.sha256()
    for si in sorted(dd[f'synth_{kind}']):
        sd = dd[f'synth_{kind}'][si]
        for key in sorted(sd):
            m.update(np.ascontiguousarray(np.asarray(sd[key], float)).tobytes())
    return m.hexdigest()[:16]


def main(quick):
    cfg = kn.KNNConfig(n_null=200 if quick else 1000, n_pool_draws=10 if quick else 50,
                       n_null_within=50 if quick else 200)
    seeds = (0, 1, 2) if quick else tuple(range(10))

    print('(0) unit checks')
    for k, d in LEGACY_DIGEST.items():
        got = legacy_digest(k)
        check(f'legacy generator unchanged: {k}', got == d, got)
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 4, (2000, 5))
    p, _ = kn.vote(lab)
    eqv = all(np.array_equal(kn.vote(s[lab])[0], s[p])
              for s in (rng.permutation(4) for _ in range(20)))
    check('vote is label-permutation equivariant', eqv)
    try:
        kn._check_trial_level(np.array([1., 2., 3., 4.]), np.zeros(4, int))
        check('sample-level detrend regressor is refused', False)
    except ValueError:
        check('sample-level detrend regressor is refused', True)
    try:
        kn.run_recday(cc.make_synthetic_recday('abstract', seed=0), 'synth_abstract', cfg)
        check('legacy restarting Trial_times are refused', False)
    except AssertionError:
        check('legacy restarting Trial_times are refused', True)

    print('(a) noise: must not reject anywhere')
    x0, w0, pvals = {v: [] for v in kn.VARIANTS}, {v: [] for v in kn.VARIANTS}, []
    for sd in seeds:
        res, rm, _ = run('noise', cfg, seed=sd)
        for v in kn.VARIANTS:
            x0[v].append(rm[('X0', 'all', v)])
            w0[v].append(rm[('W0', 'all', v)])
            kn.check_null(res['null_cross'][v], res['n_cross_tested'])
        nul = res['null_cross']['raw']
        pvals.append((1 + np.sum(nul >= rm[('X0', 'all', 'raw')])) / (1 + len(nul)))
    for v in kn.VARIANTS:
        check(f'X0 {v} at chance', abs(np.mean(x0[v]) - 0.25) < 0.05, f'{np.mean(x0[v]):.3f}')
        check(f'W0 {v} at chance', abs(np.mean(w0[v]) - 0.25) < 0.05, f'{np.mean(w0[v]):.3f}')
    check('role null calibrated (per-recday p < .05 rarely)',
          np.mean(np.asarray(pvals) < 0.05) <= 0.34, f'{np.round(pvals, 3).tolist()}')

    print('(b) abstract code: detected everywhere, time controls keep it')
    res, rm, con = run('abstract', cfg, n_repeat=2)
    for v in kn.VARIANTS:
        check(f'X0 {v} >= 0.9', rm[('X0', 'all', v)] >= 0.9, f'{rm[("X0", "all", v)]:.3f}')
        check(f'W0 {v} >= 0.9', rm[('W0', 'all', v)] >= 0.9, f'{rm[("W0", "all", v)]:.3f}')
    check('W2 near - rand = 0 (stable code)', abs(con.loc['raw', 'W2_near-rand']) < 0.03,
          f'{con.loc["raw", "W2_near-rand"]:+.3f}')
    check('X4 repeat-session ceiling >= 0.9', rm[('X4', 'repeat', 'raw')] >= 0.9,
          f'{rm[("X4", "repeat", "raw")]:.3f}')

    print('(c) corridor (place) code: within yes, across no, and the place split sees it')
    res, rm, _ = run('corridor', cfg)
    x3 = res['x3']
    lm, nm = x3[x3.leg_match].correct.mean(), x3[~x3.leg_match].correct.mean()
    check('W0 raw >= 0.8', rm[('W0', 'all', 'raw')] >= 0.8, f'{rm[("W0", "all", "raw")]:.3f}')
    check('X0 raw <= 0.30', rm[('X0', 'all', 'raw')] <= 0.30, f'{rm[("X0", "all", "raw")]:.3f}')
    check('leg-matched >= unmatched + 0.2', lm >= nm + 0.2, f'{lm:.3f} vs {nm:.3f}')

    print('(d) pure shared drift: time-local neighbours; centring manufactures "state", project not')
    Lr, Lc, d_cen, d_prj = [], [], [], []
    for sd in seeds:
        res, rm, _ = run('drift_axis', cfg, seed=sd)
        w = res['rows'][res['rows'].analysis == 'W0'].groupby('variant').L.mean()
        Lr.append(w['raw'])
        Lc.append(w['centred'])
        d_cen.append(rm[('X0', 'all', 'centred')] - rm[('X0', 'all', 'raw')])
        d_prj.append(rm[('X0', 'all', 'project')] - rm[('X0', 'all', 'raw')])
    check('within locality L raw > 0.3', np.mean(Lr) > 0.3, f'{np.mean(Lr):.2f}')
    check('within locality L centred ~ 0', abs(np.mean(Lc)) < 0.1, f'{np.mean(Lc):.2f}')
    check('centred X0 inflated by the within-trial ramp (the artefact exists)',
          np.mean(d_cen) > 0.03, f'{np.mean(d_cen):+.3f}')
    check('project X0 not inflated', abs(np.mean(d_prj)) < 0.05, f'{np.mean(d_prj):+.3f}')

    print('(e) abstract + per-session drift: raw degraded, detrend / project rescue')
    res, rm, con = run('abstract+drift_session', cfg, gain=1.0, drift_gain=30.0)
    for v in ('detrend', 'project'):
        dv = rm[('X0', 'all', v)] - rm[('X0', 'all', 'raw')]
        check(f'X0 {v} - raw > 0.1', dv > 0.1, f'{dv:+.3f}')
    check('W2 near - rand raw > 0.03 (additive drift)', con.loc['raw', 'W2_near-rand'] > 0.03,
          f'{con.loc["raw", "W2_near-rand"]:+.3f}')
    check('W2 near - rand centred ~ 0', abs(con.loc['centred', 'W2_near-rand']) < 0.03,
          f'{con.loc["centred", "W2_near-rand"]:+.3f}')

    print('(f) drifting code: near beats rand, and centring cannot remove it')
    res, rm, con = run('drift_code', cfg)
    for v in ('raw', 'centred'):
        check(f'W2 near - rand {v} > 0.03', con.loc[v, 'W2_near-rand'] > 0.03,
              f'{con.loc[v, "W2_near-rand"]:+.3f}')
    check('X1 near - rand > 0.03', con.loc['raw', 'X1_near-rand'] > 0.03,
          f'{con.loc["raw", "X1_near-rand"]:+.3f}')

    print('(g) time-since-A clock: transfers across tasks, errors adjacent (phase, not state)')
    res, rm, _ = run('clock', cfg)
    x3 = res['x3']
    err = x3[~x3.correct]
    adj = np.mean(((err.pred - err.state) % 4).isin([1, 3])) if len(err) else 1.0
    check('X0 raw > 0.5', rm[('X0', 'all', 'raw')] > 0.5, f'{rm[("X0", "all", "raw")]:.3f}')
    check('errors adjacent (chance 2/3)', adj > 0.8, f'{adj:.2f} of {len(err)}')

    print('(h) reward tone at A: decodable untrimmed, removed by the 15-bin trim')
    r0 = run('tone', cfg, trim=0)[0]['x3']
    r15 = run('tone', cfg, trim=15)[0]['x3']
    a0, a15 = r0[r0.state == 0].correct.mean(), r15[r15.state == 0].correct.mean()
    check('A recall at trim 0 > 0.8', a0 > 0.8, f'{a0:.3f}')
    check('A recall at trim 15 at chance', abs(a15 - 0.25) < 0.1, f'{a15:.3f}')

    n_fail = sum(not ok for _, ok in CHECKS)
    print(f'\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    sys.exit(main(ap.parse_args().quick))
