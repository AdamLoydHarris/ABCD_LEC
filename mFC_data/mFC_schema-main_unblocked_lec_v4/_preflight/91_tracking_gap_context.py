#!/usr/bin/env python3
"""Are the tracking gaps in engaged (high reward rate) or disengaged periods?

    python 91_tracking_gap_context.py [--window-bins 2400]
    -> logs/tracking_gaps_context.csv, and a verdict on stdout

Why: a gap that falls in a run of rewarded trials removes data the tuning and regression
actually use. A gap in a long unrewarded stretch is the animal being disengaged, and costs
almost nothing. The two have opposite implications for whether the v4 tracking gaps matter.

Reward events are taken as the `trialtimes_` state boundaries: each row is one ABCD loop and
each of its five entries is a goal poke, so the flattened unique set is the reward train. Rates
are per minute over the trial span (first to last boundary); bins outside it are not comparable.

Two measures per gap, because the gap itself is too short to carry the answer: at a typical
4 rewards/min a 9 s gap expects 0.6 rewards, so its own count is mostly Poisson noise. The
CONTEXT window (+/- 60 s around the gap, excluding the gap) is the one that says whether the
animal was working at the time; the in-gap count is reported alongside it.

The control is a matched-window null of the same width drawn from the same session's trial
span, with a MID-P rule for ties: `mean(null < x) + 0.5 * mean(null == x)`. Plain `<=` is
wrong here -- in a low-rate session most windows contain zero rewards, so a zero-reward gap
would tie with them and score p ~ 0.95, reading as "engaged" when it is simply unremarkable.

I/O: reads logs/tracking_gaps.csv and <mirror>/trialtimes_<rd>_<s>.npy; writes the context CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

REGION = 'lec'
ROOT = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data'
RUN = 'lec_replication_run_v4'
MIRROR = os.path.join(ROOT, RUN, 'data', 'Intermediate_objects')
LOGDIR = os.path.join(ROOT, RUN, 'logs')
BIN_MS = 25
N_NULL = 500
CONTEXT_S = 60          # half-width of the engagement window around each gap


def rate_per_min(events, lo, hi):
    """Rewards per minute in [lo, hi)."""
    dur_min = (hi - lo) * BIN_MS / 1000 / 60
    if dur_min <= 0:
        return None
    return int(((events >= lo) & (events < hi)).sum()) / dur_min


def mid_p(null, x):
    """P(null < x) + 0.5 P(null == x). Ties are half-counted: with a zero-reward gap in a
    low-rate session, a plain `<=` would count every zero-reward window as below it."""
    null = np.asarray(null, dtype=float)
    return float((null < x).mean() + 0.5 * (null == x).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    src = os.path.join(LOGDIR, 'tracking_gaps.csv')
    if not os.path.exists(src):
        print(f'NOT RUN: {src} missing -- run 15_export_lec_to_flat.py first')
        return 1
    rows = list(csv.DictReader(open(src)))
    if not rows:
        print('NOT RUN: 0 gaps recorded')
        return 1

    rng = np.random.default_rng(args.seed)
    out, engaged, disengaged, outside = [], 0, 0, 0
    for r in rows:
        rd, s = r['recday'], int(r['session'])
        a, b = int(r['start_bin']), int(r['end_bin'])
        f = os.path.join(MIRROR, f'trialtimes_{rd}_{s}.npy')
        rec = dict(r)
        if not os.path.exists(f):
            rec.update(verdict='no trials in this session', gap_rate_per_min='',
                       session_rate_per_min='', null_p='', n_rewards_in_gap='')
            out.append(rec); outside += 1
            continue
        tt = np.load(f).astype(float) / BIN_MS          # ms -> 25 ms bins, as the notebooks do
        ev = np.unique(tt.ravel())
        lo, hi = float(ev.min()), float(ev.max())
        gap_rate = rate_per_min(ev, a, b + 1)
        sess_rate = rate_per_min(ev, lo, hi)
        n_in = int(((ev >= a) & (ev <= b)).sum())
        # Context: +/- CONTEXT_S around the gap, with the gap's own span removed from both the
        # count and the duration, so a long gap cannot dilute its own context rate.
        half = int(CONTEXT_S * 1000 / BIN_MS)
        c0, c1 = max(int(lo), a - half), min(int(hi), b + 1 + half)
        gap_len = b - a + 1
        ctx_bins = max(c1 - c0 - gap_len, 1)
        n_ctx = int(((ev >= c0) & (ev < c1)).sum()) - n_in
        ctx_rate = n_ctx / (ctx_bins * BIN_MS / 1000 / 60)
        p = None
        if hi - lo > ctx_bins + gap_len:
            W = ctx_bins + gap_len
            starts = rng.integers(int(lo), int(hi - W), size=N_NULL)
            null = np.array([int(((ev >= x) & (ev < x + W)).sum()) / (ctx_bins * BIN_MS / 1000 / 60)
                             for x in starts], dtype=float)
            p = mid_p(null, ctx_rate)
        if b < lo or a > hi:
            v = 'outside the trial span'; outside += 1
        elif p is not None and p <= 0.05:
            v = 'DISENGAGED (context reward rate in the bottom 5%)'; disengaged += 1
        elif p is not None and p >= 0.95:
            v = 'ENGAGED (context reward rate in the top 5%)'; engaged += 1
        else:
            v = 'typical for this session'
        rec.update(verdict=v, n_rewards_in_gap=n_in, n_rewards_in_context=n_ctx,
                   gap_rate_per_min=None if gap_rate is None else round(gap_rate, 2),
                   context_rate_per_min=round(ctx_rate, 2),
                   session_rate_per_min=None if sess_rate is None else round(sess_rate, 2),
                   null_p=None if p is None else round(p, 3))
        out.append(rec)

    dst = os.path.join(LOGDIR, 'tracking_gaps_context.csv')
    cols = list(rows[0].keys()) + ['n_rewards_in_gap', 'gap_rate_per_min', 'n_rewards_in_context',
                                   'context_rate_per_min', 'session_rate_per_min', 'null_p', 'verdict']
    with open(dst, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(out, key=lambda x: -int(x['n_bins'])):
            w.writerow({c: r.get(c, '') for c in cols})

    print(f'=== tracking gaps vs reward rate ({len(out)} gaps) ===')
    print(f'  {"session":32} {"s":>6} {"rew":>4} {"ctx/min":>8} {"sess/min":>9} {"p":>6}  verdict')
    for r in sorted(out, key=lambda x: -int(x['n_bins'])):
        print(f"  {r['recday']+'_'+str(r['session']):32} {float(r['seconds']):6.1f} "
              f"{str(r['n_rewards_in_gap']):>4} {str(r.get('context_rate_per_min')):>8} "
              f"{str(r['session_rate_per_min']):>9} {str(r['null_p']):>6}  {r['verdict']}")
    in_trial = sum(1 for r in out if r['inside_a_trial'] == 'True')
    print(f'\n  {in_trial}/{len(out)} gaps overlap a trial; '
          f'{disengaged} disengaged, {engaged} engaged, {outside} outside the trial span')
    print(f'  wrote {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
