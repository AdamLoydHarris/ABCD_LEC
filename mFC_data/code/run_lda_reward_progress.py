"""Run the reward x goal-progress LDA (lda_reward_goalprogress.py) for one dataset on the
cluster and pickle everything the notebooks' summary cells need.

    python run_lda_reward_progress.py --dataset lec|pfc [--min-trials 10] [--n-shuffles 1000]
                                      [--recdays rd1,rd2] [--out PATH]

Why a script: the analysis lived only in `LEC_lda_analyses.ipynb` / `PFC_lda_analyses.ipynb`
cell 30, where the 1000-shuffle null per recday x 2 targets made it long enough to be
interrupted (LEC, 3/25 recdays) and to crash on the last PFC recday (1 PC, fixed 2026-09-14
in the module). Neither region ever had a complete run. Running it here, under sbatch, keeps
it out of the 64 GB interactive cgroup and leaves a joinable pickle rather than 59 MB of
embedded notebook outputs. The notebooks load the pickle (`RUN_MODE = 'load'`).

Identical file in `code/` and `mFC_data/code/` (repo convention: duplication, not import);
`--dataset` picks the loader. LEC reads `data/processed_data/data_dic_lec.pkl` (the current
pickle, which carries `Neurons_norm`; the notebook used the Feb-2026 `data_dic_for_yaren.pkl`),
PFC builds the same shape with `glm_analysis_v2.build_data_dic_from_pfc(compute_norm=True)`.

Output pickle:
    {'dataset', 'min_trials', 'n_shuffles', 'recdays', 'valid_sessions_dic',
     'results_by_recday': {recday: run_reward_progress_lda_analysis(...) dict},
     'decoding_results':  {recday: {'progress'|'reward': {'real_acc','null_accs','p_value'}}},
     'skipped': {recday: reason}, 'elapsed_s'}
plus `<out>.csv` with one row per recday (n_neurons, n_pcs, n_lds, accuracies, p-values).
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


def hrs(t0):
    return f'{(time.time() - t0) / 3600:.2f} h'


def load_lec():
    with open(os.path.join(REPO, 'data', 'processed_data', 'data_dic_lec.pkl'), 'rb') as f:
        data_dic = pickle.load(f)
    recdays = sorted(k for k in data_dic if '_sb' not in k)
    return data_dic, recdays


def load_pfc():
    sys.path.insert(0, os.path.join(REPO, 'mFC_data', 'code'))
    from glm_analysis_v2 import build_data_dic_from_pfc
    data_folder = os.path.join(REPO, 'mFC_data', 'data')
    recdays = list(np.load(os.path.join(data_folder, 'MetaData', 'combined_ABCDonly_days.npy')).astype(str))
    data_dic = build_data_dic_from_pfc(data_folder, recdays, compute_norm=True, verbose=False)
    return data_dic, sorted(data_dic.keys())


def build_valid_sessions(data_dic, recdays, min_trials=5):
    """One session per unique task structure, >= min_trials trials -- the notebooks' cell 6."""
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
    ap.add_argument('--min-trials', type=int, default=10,
                    help='first N trials per session (the notebooks use 10)')
    ap.add_argument('--n-shuffles', type=int, default=1000)
    ap.add_argument('--recdays', default='', help='comma-separated subset (smoke test)')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    t0 = time.time()

    tree = 'code' if args.dataset == 'lec' else os.path.join('mFC_data', 'code')
    sys.path.insert(0, os.path.join(REPO, tree))
    import lda_reward_goalprogress as lrg

    data_dic, recdays = (load_lec if args.dataset == 'lec' else load_pfc)()
    if args.recdays:
        recdays = [r for r in recdays if r in args.recdays.split(',')]
    valid_sessions_dic = build_valid_sessions(data_dic, recdays)
    print(f'[{hrs(t0)}] {args.dataset.upper()}: {len(recdays)} recdays loaded', flush=True)

    results_by_recday, decoding_results, skipped, rows = {}, {}, {}, []
    for rd in recdays:
        res = lrg.run_reward_progress_lda_analysis(
            data_dic, rd, valid_sessions=valid_sessions_dic[rd],
            neuron_subset=None, min_trials=args.min_trials)
        plt.close('all')
        if res is None:
            skipped[rd] = 'run_reward_progress_lda_analysis returned None (see log)'
            continue
        results_by_recday[rd] = res
        decoding_results[rd] = {}
        for target in ('progress', 'reward'):
            real_acc, null_accs, p_val = lrg.run_reward_progress_decoding(
                res, decode_target=target, mouse_recday=rd, n_shuffles=args.n_shuffles)
            plt.close('all')
            decoding_results[rd][target] = {'real_acc': real_acc, 'null_accs': null_accs,
                                            'p_value': p_val}
        rows.append(dict(
            recday=rd, mouse=rd.split('_')[0], n_neurons=res['X'].shape[1],
            n_samples=res['X'].shape[0], n_pcs=res['X_pca'].shape[1],
            n_lds=res['X_rp_ld'].shape[1], n_sessions=len(res['filtered_sessions']),
            progress_acc=decoding_results[rd]['progress']['real_acc'],
            progress_null_mean=float(np.mean(decoding_results[rd]['progress']['null_accs'])),
            progress_p=decoding_results[rd]['progress']['p_value'],
            reward_acc=decoding_results[rd]['reward']['real_acc'],
            reward_null_mean=float(np.mean(decoding_results[rd]['reward']['null_accs'])),
            reward_p=decoding_results[rd]['reward']['p_value'],
            reward_chance=1.0 / len(np.unique(res['y_reward'])),
        ))
        print(f'[{hrs(t0)}] done {rd}', flush=True)

    out = args.out or OUT_DEFAULT[args.dataset]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = dict(dataset=args.dataset, min_trials=args.min_trials, n_shuffles=args.n_shuffles,
                   recdays=recdays, valid_sessions_dic=valid_sessions_dic,
                   results_by_recday=results_by_recday, decoding_results=decoding_results,
                   skipped=skipped, elapsed_s=time.time() - t0)
    with open(out, 'wb') as f:
        pickle.dump(payload, f)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out.replace('.pkl', '.csv'), index=False)
    print(f'[{hrs(t0)}] wrote {out} ({len(results_by_recday)} recdays, '
          f'{len(skipped)} skipped: {skipped})', flush=True)


if __name__ == '__main__':
    main()
