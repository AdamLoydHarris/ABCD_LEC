"""One-factor-at-a-time ladder from our V4 configuration toward El-Gaby's executed one.

**Does not modify `elasticnet_regression_v4`.** Every rung is a plain `RegressionConfigV4`
built from settings that already exist; the module is a driver plus a summary.

    python code/elgaby_ladder.py                 # 6 recdays, PFC, past lags
    python code/elgaby_ladder.py --all-recdays   # all 25 (slow)
    python code/elgaby_ladder.py --rungs R0 R1   # just some

The question
------------
Our PFC past sweep gives n = 155 non-zero-lag state-tuned neurons at mean r = +0.057 against a
reported n = 329 and a markedly right-shifted distribution. Re-scoring the existing betas
(`elgaby_figure5.py`) already showed the **count** is explained by the scoring criterion --
{0,11} instead of {0,1,11} gives 271, and his per-fold semantics give 486, bracketing 329 --
but the **right-shift is not**. So the remaining factors all concern the fit, and each needs a
re-fit to test:

    R0  v4 as shipped: ElasticNet alpha=0.01, 12 lags, preferred phase from TRAINING folds
    R1  + Poisson                     his executed estimator; never zeroes a coefficient,
                                      so the 33% of units with no surviving beta come back
    R2  + pref_phase_source='test'    his cell 21 reads the HELD-OUT session's phase tuning,
                                      for both the fit and the scoring -- leakage
    R3  + num_lags=24  (_beyond)      cells 32/38, which savefig the published histograms,
                                      are set to limited=False

Rungs are cumulative and differ by exactly one setting each, so the first rung that moves the
distribution is the answer. Every rung writes a `run_config.json`, and
`elasticnet_v4_compare.check_comparable` will flag any rung that accidentally moved two.

Each rung is scored under both lag sets x both semantics (that axis is free), so the criterion
and the fit are never confounded in the readout.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import elasticnet_regression_v4 as v4                                    # noqa: E402
import elgaby_figure5 as eg                                              # noqa: E402

PFC_DATA = os.path.join(REPO_ROOT, 'mFC_data', 'data')
PFC_CODE = os.path.join(REPO_ROOT, 'mFC_data', 'code')

#: Six recdays spanning every mouse and the whole size range (unit counts in comments).
LADDER_RECDAYS = [
    'ah04_01122021_02122021',   # 117
    'ah07_01092023_02092023',   #  97
    'ab03_29082023_30082023',   #  70
    'me11_01122021_02122021',   #  53
    'me08_06092021_09092021',   #  23
    'ah03_12082021_13082021',   #  16
]

#: Cumulative: each rung adds exactly one setting to the one above it.
#: Cumulative: each rung adds exactly one setting to the one above it. Re-aimed after the paper
#: settled the criterion questions -- the `_beyond` (24-lag) rung is gone because the methods
#: state 12 lags explicitly, and the defaults now start reference-matched, so the last rung
#: REMOVES the leakage rather than adding it.
RUNGS = {
    'R0': ('v4 as shipped (reference-matched)', {}),
    'R1': ('+ Poisson', {'use_poisson': True}),
    'R2': ("+ poisson_link='log' (the paper's LNP)",
           {'use_poisson': True, 'poisson_link': 'log'}),
    'R3': ("+ pref phase from TRAIN (leakage removed)",
           {'use_poisson': True, 'poisson_link': 'log', 'pref_phase_source': 'train'}),
}


def load_pfc_module():
    """The PFC loader, by explicit path -- both trees have a `glm_analysis_v2`."""
    spec = importlib.util.spec_from_file_location(
        'pfc_glm_analysis', os.path.join(PFC_CODE, 'glm_analysis_v2.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pfc_glm_analysis'] = mod
    spec.loader.exec_module(mod)
    return mod


def build_sessions(data_dic):
    """Dedup by exact task equality (== his `non_repeat_ses_maker`), then his hand-exclusion."""
    vs = {}
    for mr in data_dic:
        keep, seen = [], []
        for s in sorted(x for x in data_dic[mr] if x != 'valid_sessions'):
            sd = data_dic[mr][s]
            if sd.get('num_trials', 0) < 5:
                continue
            if not any(np.array_equal(sd['Task'], c) for c in seen):
                seen.append(sd['Task'])
                keep.append(s)
        vs[mr] = keep
    return v4.apply_excluded_sessions(vs, verbose=False)


def score_rung(data_dic, runs, lag_sets, tuning_key='state_tuned_mask'):
    """n / mean r / frac>0 / p for one rung, under each lag set x semantics."""
    from scipy import stats as st
    rows = []
    prep_cache = {}
    for lag_bins in lag_sets:
        acc = {'per-neuron mask': [0, []], 'El-Gaby per-fold': [0, []]}
        for recday, res in runs.items():
            cfg = eg._config_from(res)
            if recday not in prep_cache:
                prep_cache[recday] = eg._prepare_test_sessions(data_dic, recday, res, cfg)
            sc = eg.score_elgaby(data_dic, recday, res, lag_bins=lag_bins, config=cfg,
                                 preps=prep_cache[recday])
            mask = eg.score_ours(res, lag_bins, config=cfg)['mask']
            sel = mask & res[tuning_key] & np.isfinite(sc['mean_raw'])
            acc['per-neuron mask'][0] += int(sel.sum())
            acc['per-neuron mask'][1].append(sc['mean_raw'][sel])

            sel_eg = (sc['counts'] > 0) & res[tuning_key] & np.isfinite(sc['mean_nozero'])
            acc['El-Gaby per-fold'][0] += int(sel_eg.sum())
            acc['El-Gaby per-fold'][1].append(sc['mean_nozero'][sel_eg])
        for label, (n, rr) in acc.items():
            r = np.concatenate(rr) if rr else np.array([])
            r = r[np.isfinite(r)]
            t, p = st.ttest_1samp(r, 0) if len(r) > 1 else (np.nan, np.nan)
            rows.append({'lag_set': '{' + ','.join(map(str, lag_bins)) + '}',
                         'semantics': label, 'n': n,
                         'mean_r': round(float(r.mean()), 4) if len(r) else np.nan,
                         'median_r': round(float(np.median(r)), 4) if len(r) else np.nan,
                         'frac_pos': round(float(np.mean(r > 0)), 3) if len(r) else np.nan,
                         'p': float(p),
                         '_r': r})
    return rows


def plot_ladder(results_by_rung, out_path):
    """Histogram of the per-neuron r at each rung, El-Gaby semantics, lags {0,11}."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from glm_analysis_v2 import apply_gridmaze_style

    apply_gridmaze_style()
    rungs = [k for k in RUNGS if k in results_by_rung]
    fig, axes = plt.subplots(1, len(rungs), figsize=(2.5 * len(rungs), 2.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, rung in zip(axes, rungs):
        row = next((r for r in results_by_rung[rung]
                    if r['lag_set'] == '{0,11}' and r['semantics'] == 'El-Gaby per-fold'), None)
        if row is None or not len(row['_r']):
            ax.axis('off')
            continue
        r = row['_r']
        ax.hist(r, bins=np.linspace(-1, 1, 41), color='#888780', edgecolor='#2C2C2A', lw=0.3)
        ax.axvline(0, color='#2C2C2A', ls='--', lw=0.8)
        ax.axvline(r.mean(), color='#C03030', lw=1.4)
        ax.set_title(f"{rung} {RUNGS[rung][0]}\nn={row['n']}  mean={r.mean():+.3f}  "
                     f"{row['frac_pos']:.0%}>0", fontsize=7)
        ax.set_xlabel('pref-phase state r (n=4)')
    axes[0].set_ylabel('# neurons')
    fig.suptitle('El-Gaby Figure 5 ladder - PFC, past lags, lags {0,11}, his per-fold semantics',
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        for ext in ('svg', 'png'):
            fig.savefig(f'{out_path}.{ext}', bbox_inches=None, dpi=300)
    return fig


def main():
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rungs', nargs='*', default=list(RUNGS))
    ap.add_argument('--all-recdays', action='store_true')
    ap.add_argument('--n-jobs', type=int, default=6)
    ap.add_argument('--out-root', default=os.path.join(REPO_ROOT, 'data', 'figures'))
    ap.add_argument('--resume-into', default=None,
                    help='an existing elgaby_ladder_* folder: rungs already exported there are '
                         're-scored instead of re-fitted')
    ap.add_argument('--refit', action='store_true', help='ignore existing exports')
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = args.resume_into or os.path.join(args.out_root, f'elgaby_ladder_{stamp}')
    os.makedirs(out_root, exist_ok=True)
    print(f'outputs -> {out_root}\n')

    pfc = load_pfc_module()
    recdays = (list(np.load(os.path.join(PFC_DATA, 'MetaData', 'combined_ABCDonly_days.npy'))
                    .astype(str)) if args.all_recdays else LADDER_RECDAYS)
    t0 = time.time()
    data_dic = pfc.build_data_dic_from_pfc(PFC_DATA, recdays, compute_norm=False, verbose=False)
    valid_sessions = build_sessions(data_dic)
    n_units = sum(data_dic[mr][min(data_dic[mr])]['Neuron_raw'].shape[0] for mr in data_dic)
    print(f'{len(data_dic)} recdays, {n_units} units, loaded in {time.time() - t0:.0f}s')
    print('folds/recday: ' + ', '.join(f'{k}={len(v)}' for k, v in valid_sessions.items()) + '\n')

    all_rows, results_by_rung = [], {}
    for rung in args.rungs:
        label, overrides = RUNGS[rung]
        # merge first: an override that sets use_poisson would otherwise collide with the
        # explicit keyword and raise TypeError
        kw = {'lag_direction': 'past', 'use_poisson': False, 'regularize': True, **overrides}
        cfg = v4.RegressionConfigV4(**kw)
        out_dir = os.path.join(out_root, f'{rung}_{v4.run_dir_name(cfg, stamp=stamp)}')
        print(f'{"=" * 70}\n{rung}  {label}   ({v4.estimator_name(cfg)}, '
              f'{cfg.num_lags} lags, pref={cfg.pref_phase_source})\n{"=" * 70}', flush=True)

        # Resume: a rung whose exports already exist is re-scored, not re-fitted. Poisson
        # rungs cost 30-90 min each, so a crash in the scoring layer must not throw the fits
        # away. `--out-root <previous run>` picks up where an interrupted ladder stopped.
        existing = sorted(glob.glob(os.path.join(out_root, f'{rung}_*', '*_arrays.npz')))
        if existing and not args.refit:
            prev = os.path.dirname(existing[0])
            results = eg.load_run(prev)
            print(f'  RESUMED from {os.path.basename(prev)} ({len(results)} recdays, no re-fit)',
                  flush=True)
        else:
            t0 = time.time()
            results, _, diag = v4.run_and_summarise_all_mice_v4(
                data_dic, cfg, valid_sessions_dic=valid_sessions, save_dir=out_dir,
                export_dir=out_dir, make_pdfs=False, n_jobs=args.n_jobs, verbose=False)
            print(f'  fitted in {time.time() - t0:.0f}s', flush=True)

        worst = 0.0
        for rd, res in results.items():
            worst = max(worst, eg.verify_against_v4(data_dic, rd, res)[0])
        print(f'  regressor gate: worst max|diff| = {worst:.3g}', flush=True)

        # the 90-degree set depends on the lag count: cell 26 overwrites it with
        # np.arange(12) in the _beyond branch, so it must be built per rung
        lag_sets = (eg.BINS_30, (0, 1, 11), eg.bins_90_for(cfg.num_lags))
        rows = score_rung(data_dic, results, lag_sets)
        results_by_rung[rung] = rows
        for r in rows:
            all_rows.append({'rung': rung, 'label': label,
                             'estimator': v4.estimator_name(cfg), 'num_lags': cfg.num_lags,
                             'pref_phase': cfg.pref_phase_source,
                             **{k: v for k, v in r.items() if k != '_r'}})
        print(pd.DataFrame([{k: v for k, v in r.items() if k != '_r'} for r in rows])
              .to_string(index=False), flush=True)
        print()

    table = pd.DataFrame(all_rows)
    table.to_csv(os.path.join(out_root, 'ladder_summary.csv'), index=False)
    print('=' * 70)
    print(table.to_string(index=False))
    print('\nEl-Gaby reports n = 329 for the nonzero (30 deg) quantity, lags {0,11}')
    if len(results_by_rung) > 1:
        plot_ladder(results_by_rung, os.path.join(out_root, 'ladder'))
    print(f'\n-> {out_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
