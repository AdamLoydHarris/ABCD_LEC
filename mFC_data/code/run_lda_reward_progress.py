"""Joint trial x goal-progress LDA (lda_reward_goalprogress.py) for one dataset, under sbatch.

    python run_lda_reward_progress.py --dataset lec|pfc [--conjunction trial_progress|reward_progress]
        [--min-trials 10] [--n-shuffles 1000] [--ridge-alpha 1.0] [--recdays rd1,rd2] [--out PATH]
        [--legacy-decoding]

Same file in code/ and mFC_data/code/. LEC reads data_dic_lec.pkl; PFC builds the same shape
with build_data_dic_from_pfc(compute_norm=True). Per recday: dataset -> PCA -> joint LDA, then
run_joint_lda_readout (held-out time and progress scores with a circular-shift null).

Output: pickle {'dataset', 'conjunction', 'min_trials', 'n_shuffles', 'ridge_alpha', 'recdays',
'valid_sessions_dic', 'results_by_recday', 'readout', 'skipped', 'elapsed_s'} (+ the old
'decoding_results' with --legacy-decoding) and a per-recday CSV from reward_progress_table.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

REPO = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
OUT_DEFAULT = {
    'lec': os.path.join(REPO, 'data', 'glm_outputs', 'LEC_lda', 'reward_progress.pkl'),
    'pfc': os.path.join(REPO, 'mFC_data', 'glm_outputs', 'PFC_lda', 'reward_progress.pkl'),
}


#: alias -> set of `unit_regions` group labels (anatomy_split.REGION_COLORS keys)
REGION_GROUPS = {
    'ENTl': {'ENTl-sup', 'ENTl-deep'},
    'ENTl-deep': {'ENTl-deep'},
    'ENTl-sup': {'ENTl-sup'},
    'ENTm': {'ENTm'},
    'SUB': {'SUB/ProS'},
    'CA1': {'CA1/HPF'},
    'SUBCA1': {'SUB/ProS', 'CA1/HPF'},
}


def hrs(t0):
    return f'{(time.time() - t0) / 3600:.2f} h'


def load_lec(only=None):
    with open(os.path.join(REPO, 'data', 'processed_data', 'data_dic_lec.pkl'), 'rb') as f:
        data_dic = pickle.load(f)
    recdays = sorted(k for k in data_dic if '_sb' not in k)
    return data_dic, [r for r in recdays if not only or r in only]


def load_pfc(only=None):
    sys.path.insert(0, os.path.join(REPO, 'mFC_data', 'code'))
    from glm_analysis_v2 import build_data_dic_from_pfc
    data_folder = os.path.join(REPO, 'mFC_data', 'data')
    recdays = list(np.load(os.path.join(data_folder, 'MetaData', 'combined_ABCDonly_days.npy')).astype(str))
    if only:
        recdays = [r for r in recdays if r in only]
    data_dic = build_data_dic_from_pfc(data_folder, recdays, compute_norm=True, verbose=False)
    return data_dic, sorted(data_dic.keys())


def build_valid_sessions(data_dic, recdays, min_trials=5):
    """One session per unique task structure with >= min_trials trials (the notebooks' cell 6)."""
    out = {}
    for rd in recdays:
        valid, tasks = [], []
        for session in data_dic[rd]:
            if session == 'valid_sessions':
                continue
            sd = data_dic[rd][session]
            if sd['num_trials'] < min_trials:
                continue
            if 'defaultdict' in str(sd['Task']):
                continue
            if not any(np.array_equal(sd['Task'], t) for t in tasks):
                tasks.append(sd['Task'])
                valid.append(session)
        out[rd] = valid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=('lec', 'pfc'))
    ap.add_argument('--conjunction', default='trial_progress', choices=('trial_progress', 'reward_progress'))
    ap.add_argument('--min-trials', type=int, default=10, help='first N trials per session')
    ap.add_argument('--n-shuffles', type=int, default=1000)
    ap.add_argument('--ridge-alpha', type=float, default=1.0)
    ap.add_argument('--n-pcs', type=int, default=None,
                    help='fixed LDA input dimension for every recday (default: 75 %% variance); '
                         'the pickle gets a _pcs<n> suffix unless --out is given')
    ap.add_argument('--subsample-neurons', type=int, default=None,
                    help='random neuron subset of this size per recday (recdays with fewer keep '
                         'all); repeated --n-draws times; pickle suffix _sub<n>')
    ap.add_argument('--n-draws', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--region-group', default=None, choices=sorted(REGION_GROUPS),
                    help='LEC only: restrict to units of this anatomical group (unit_regions.pkl); '
                         'combine with --subsample-neurons for a count-matched readout; suffix _grp<name>')
    ap.add_argument('--recdays', default='', help='comma-separated subset (smoke test)')
    ap.add_argument('--out', default='')
    ap.add_argument('--legacy-decoding', action='store_true',
                    help='also run the old separate progress / reward-number decoding LDAs')
    args = ap.parse_args()
    t0 = time.time()

    tree = 'code' if args.dataset == 'lec' else os.path.join('mFC_data', 'code')
    sys.path.insert(0, os.path.join(REPO, tree))
    import lda_reward_goalprogress as lrg

    only = set(args.recdays.split(',')) if args.recdays else None
    data_dic, recdays = (load_lec if args.dataset == 'lec' else load_pfc)(only)
    valid_sessions_dic = build_valid_sessions(data_dic, recdays)
    print(f'[{hrs(t0)}] {args.dataset.upper()}: {len(recdays)} recdays loaded', flush=True)

    unit_regions = None
    if args.region_group is not None:
        if args.dataset != 'lec':
            raise SystemExit('--region-group needs per-unit anatomy, which only the LEC dataset has')
        import elasticnet_regression_v5 as v5
        unit_regions = v5.load_unit_regions()

    results_by_recday, readout, decoding_results, skipped = {}, {}, {}, {}
    rng = np.random.default_rng(args.seed)
    for rd in recdays:
        n_neurons = next(v for k, v in data_dic[rd].items()
                         if k != 'valid_sessions' and isinstance(v, dict))['Neurons_norm'].shape[0]
        pool = np.arange(n_neurons)
        if unit_regions is not None:
            groups = np.asarray(unit_regions[rd]['group'])
            assert len(groups) == n_neurons, (rd, len(groups), n_neurons)
            pool = np.flatnonzero(np.isin(groups, list(REGION_GROUPS[args.region_group])))
            if args.subsample_neurons is not None and len(pool) < args.subsample_neurons:
                skipped[rd] = f'{len(pool)} {args.region_group} units < {args.subsample_neurons}'
                print(f'  SKIP {rd}: {skipped[rd]}', flush=True)
                continue
            if len(pool) < 2:
                skipped[rd] = f'{len(pool)} {args.region_group} units'
                continue
        if args.subsample_neurons is not None and len(pool) > args.subsample_neurons:
            subsets = [np.sort(rng.choice(pool, args.subsample_neurons, replace=False))
                       for _ in range(args.n_draws)]
        elif unit_regions is not None:
            subsets = [pool]          # exactly the target (or no subsampling): the whole group once
        else:
            subsets = [None]          # fewer neurons than the target: one full draw
        draws, res = [], None
        for subset in subsets:
            res = lrg.run_reward_progress_lda_analysis(
                data_dic, rd, valid_sessions=valid_sessions_dic[rd], neuron_subset=subset,
                min_trials=args.min_trials, conjunction=args.conjunction, plot=False,
                n_pcs=args.n_pcs)
            plt.close('all')
            if res is None:
                break
            draws.append(lrg.run_joint_lda_readout(res, n_shuffles=args.n_shuffles,
                                                   ridge_alpha=args.ridge_alpha,
                                                   verbose=len(subsets) == 1))
        if res is None or not draws:
            skipped[rd] = 'run_reward_progress_lda_analysis returned None (see log)'
            continue
        res['n_neurons_total'] = n_neurons
        res['n_neurons_pool'] = int(len(pool))
        res['region_group'] = args.region_group
        res['neuron_subsets'] = None if (len(subsets) == 1 and subsets[0] is None) else subsets
        results_by_recday[rd] = res
        readout[rd] = draws[0] if len(draws) == 1 else lrg.aggregate_draw_readouts(draws)
        if len(draws) > 1:
            print(f"  {len(draws)} draws of {args.subsample_neurons}/{n_neurons} neurons: "
                  f"seconds r {readout[rd]['sec_r']:.2f} (draw sd {np.nanstd(readout[rd]['sec_r_draws']):.2f}), "
                  f"trial MAE {readout[rd]['key_mae']:.2f}, progress acc {readout[rd]['prog_acc']:.3f}")
        if args.legacy_decoding:
            decoding_results[rd] = {}
            for target in ('progress', 'reward'):
                real_acc, null_accs, p_val = lrg.run_reward_progress_decoding(
                    res, decode_target=target, mouse_recday=rd, n_shuffles=args.n_shuffles)
                plt.close('all')
                decoding_results[rd][target] = {'real_acc': real_acc, 'null_accs': null_accs,
                                                'p_value': p_val}
        print(f'[{hrs(t0)}] done {rd}', flush=True)

    out = args.out or OUT_DEFAULT[args.dataset]
    if not args.out and args.n_pcs is not None:
        out = out.replace('.pkl', f'_pcs{args.n_pcs}.pkl')
    if not args.out and args.region_group is not None:
        out = out.replace('.pkl', f'_grp{args.region_group}.pkl')
    if not args.out and args.subsample_neurons is not None:
        out = out.replace('.pkl', f'_sub{args.subsample_neurons}.pkl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = dict(dataset=args.dataset, conjunction=args.conjunction, min_trials=args.min_trials,
                   n_pcs=args.n_pcs, subsample_neurons=args.subsample_neurons,
                   region_group=args.region_group,
                   n_draws=args.n_draws if args.subsample_neurons is not None else 1, seed=args.seed,
                   n_shuffles=args.n_shuffles, ridge_alpha=args.ridge_alpha, recdays=recdays,
                   valid_sessions_dic=valid_sessions_dic, results_by_recday=results_by_recday,
                   readout=readout, skipped=skipped, elapsed_s=time.time() - t0)
    if args.legacy_decoding:
        payload['decoding_results'] = decoding_results
    with open(out, 'wb') as f:
        pickle.dump(payload, f)
    lrg.reward_progress_table(payload).to_csv(out.replace('.pkl', '.csv'), index=False)
    print(f'[{hrs(t0)}] wrote {out} ({len(results_by_recday)} recdays, '
          f'{len(skipped)} skipped: {skipped})', flush=True)


if __name__ == '__main__':
    main()
