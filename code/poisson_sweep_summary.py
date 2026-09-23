"""Summarise a Poisson anchoring regularisation sweep into one table and figure.

    python poisson_sweep_summary.py [--region pfc|lec] [--out-dir DIR]

Discovers the reference arm and every sweep point on disk by directory name (PFC specs 2 and
14-17; LEC specs 12 and 18-21) and
reports, per penalty: the three El-Gaby panels (n and t, his semantics, linear readout), the
fraction of fits whose coefficients are all zero, and the median number of non-zero betas of
324. Under an L1 penalty a fixed alpha is a firing-rate filter, so n is only interpretable
beside the zero-fit fraction.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl                      # noqa: E402
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402

REPO = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
#: region -> (figures root, glob for the un-swept L2 reference arm)
REGIONS = {'pfc': (os.path.join(REPO, 'mFC_data', 'data', 'figures'), 'poisson_v5_past_*'),
           'lec': (os.path.join(REPO, 'data', 'figures'), 'repro_poisson_v5_past_*')}
PANELS = [('all', 'all state-tuned', 489, 10.7),
          ('30deg', 'non-zero-lag, 30 deg', 346, 4.74),
          ('90deg', 'non-zero-lag, 90 deg', 229, 2.81)]
#: GridMaze Stone (reference), an ordered ramp over decreasing alpha, Ultra Violet for EN
INK = '#2C2C2A'
PEN_COLOURS = {'L2': '#B4B2A9', 'EN': '#6B3FA0'}
L1_RAMP = ['#45B5AA', '#0F4C81', '#BE3455', '#9B1B30']


def colours(df):
    l1 = [a for p_, a in zip(df.penalty, df.alpha) if p_ == 'L1']
    ramp = {a: L1_RAMP[min(i, len(L1_RAMP) - 1)] for i, a in enumerate(sorted(l1, reverse=True))}
    return [PEN_COLOURS.get(p_, ramp.get(a, INK)) for p_, a in zip(df.penalty, df.alpha)]


def sparsity(run_dir):
    """(all-zero fit fraction, median non-zero betas) over every fitted neuron-fold."""
    nz, zero, fitted = [], 0, 0
    for f in sorted(glob.glob(os.path.join(run_dir, '*_past_arrays.npz'))):
        c = np.load(f, allow_pickle=True)['cv_coeffs']
        ok = np.isfinite(c).all(axis=2)
        n = (np.abs(c) > 1e-8).sum(axis=2)
        fitted += int(ok.sum())
        zero += int(((n == 0) & ok).sum())
        nz.append(n[ok])
    nz = np.concatenate(nz) if nz else np.array([np.nan])
    return (zero / fitted if fitted else np.nan), float(np.median(nz)), fitted


def discover(region):
    """(label, dir, alpha, penalty) for the reference arm and every sweep point on disk.

    Names carry everything needed: `alpha<a>_` for the sweep points and `_l1` / `_en<ratio>`
    for the penalty (`run_dir_name` + the queue's alpha prefix). Newest stamp per configuration.
    """
    fig, ref_glob = REGIONS[region]
    out = {}
    for d in sorted(glob.glob(os.path.join(fig, ref_glob))):
        b = os.path.basename(d)
        if b.startswith('alpha') or '_l1_' in b or '_en' in b:
            continue
        out[('L2', 1.0)] = ('L2 a=1', b, 1.0, 'L2')
    for d in sorted(glob.glob(os.path.join(fig, 'alpha*_poisson_*_v5_past_*'))):
        b = os.path.basename(d)
        m = re.match(r'alpha([\d.eE+-]+)_', b)
        if not m:
            continue
        alpha = float(m.group(1))
        if '_l1_' in b:
            pen, label = 'L1', f'L1 a={alpha:g}'
        else:
            r = re.search(r'_en([\d.]+)_', b)
            pen = 'EN'
            label = f'EN a={alpha:g} l1={r.group(1) if r else "?"}'
        out[(pen, alpha)] = (label, b, alpha, pen)
    rank = {'L2': 0, 'L1': 1, 'EN': 2}
    order = sorted(out.values(), key=lambda x: (rank[x[3]], -x[2]))
    if not order:
        raise SystemExit(f'no {region} Poisson runs found under {fig}')
    return order


def build_table(region='pfc'):
    fig_root = REGIONS[region][0]
    rows = []
    for label, d, alpha, pen in discover(region):
        run_dir = os.path.join(fig_root, d)
        t = pd.read_csv(os.path.join(run_dir, 'three_panel_summary_past.csv'))
        t = t[t.link.str.startswith('linear') & (t.semantics == 'elgaby')]
        zf, med, fitted = sparsity(run_dir)
        r = dict(run=label, alpha=alpha, penalty=pen, run_dir=d,
                 recdays=len(glob.glob(os.path.join(run_dir, '*_past_arrays.npz'))),
                 n_fits=fitted, zero_fit_frac=zf, median_nonzero_betas=med)
        for _, x in t.iterrows():
            key = 'all' if 'all state' in x.panel else ('30deg' if '30deg' in x.panel else '90deg')
            r[f'{key}_n'], r[f'{key}_t'], r[f'{key}_r'] = int(x.n), float(x.t), float(x.mean_r)
        rows.append(r)
    return pd.DataFrame(rows)


def plot(df, out_svg):
    try:
        import sys
        sys.path.insert(0, os.path.join(REPO, 'code'))
        from glm_analysis_v2 import apply_gridmaze_style
        apply_gridmaze_style()
    except Exception:
        pass
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.6))
    x = np.arange(len(df))
    for ax, (key, title, pn, pt) in zip(axes[:3], PANELS):
        ax.bar(x, df[f'{key}_n'], color=colours(df), width=0.72)
        ax.axhline(pn, ls='--', lw=0.8, color=INK)
        for i, (n, t) in enumerate(zip(df[f'{key}_n'], df[f'{key}_t'])):
            ax.text(i, n, f'{t:.1f}', ha='center', va='bottom', fontsize=6, color=INK)
        ax.set_title(f'{title}\nbar n, label t, dashed paper n', pad=6)
        ax.set_ylabel('neurons')
    ax = axes[3]
    ax.bar(x, df.zero_fit_frac * 100, color=colours(df), width=0.72)
    for i, m in enumerate(df.median_nonzero_betas):
        ax.text(i, df.zero_fit_frac.iloc[i] * 100, f'{m:.0f}', ha='center', va='bottom',
                fontsize=6, color=INK)
    ax.set_title('sparsity\nbar % all-zero fits, label median betas', pad=6)
    ax.set_ylabel('% all-zero fits')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df.run, rotation=45, ha='right', fontsize=6)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig.savefig(out_svg, bbox_inches=None)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default='pfc', choices=sorted(REGIONS))
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(REGIONS[args.region][0], 'poisson_sweep')
    df = build_table(args.region)
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, 'poisson_sweep_summary.csv')
    df.to_csv(csv, index=False)
    svg = os.path.join(out_dir, 'poisson_sweep_summary.svg')
    plot(df, svg)
    cols = ['run', 'recdays', 'zero_fit_frac', 'median_nonzero_betas',
            'all_n', 'all_t', '30deg_n', '30deg_t', '90deg_n', '90deg_t']
    print(df[cols].to_string(index=False))
    print(f'\npaper ED 8d: all n=489 t=10.7 | 30deg n=346 t=4.74 | 90deg n=229 t=2.81')
    print(f'wrote {csv}\nwrote {svg}')


if __name__ == '__main__':
    main()
