"""Round-2 summary figures (2026-10-12): maze claims against the field-shuffle, kinematic and speed nulls (B), the
maze-open-field link (C), the LEC object box (D) and the gate decisions behind the settings (V). Reads merged
tables, shards and gate logs only; no data_dic.

    sbatch --job-name=sp_round2_figs --partition=cpu_backfill --mem=64G sbatch_files/spatial.sbatch code/spatial_round2_figures.py
    python code/spatial_round2_figures.py --only C D V        # a subset

Writes <figure>.pdf and <figure>.png per figure to <LEC root>/data/spatial/figures/round2/, versioned and never
overwritten. code/ only: not mirrored to mFC_data/code/.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LEC = '/ceph/behrens/adam_harris/Taskspace_abstraction_lEC'
MEC = '/ceph/behrens/adam_harris/Taskspace_abstraction_mEC'
OUT = os.path.join(LEC, 'data', 'spatial', 'figures', 'round2')
LOGS = os.path.join(LEC, 'sbatch_files', 'slurm_outputs')
MAZE = {'B1': '53934257354e', 'B2': '05fd7f57354e', 'B2-head': '09482657354e'}
LINK_TAG, LINK_OF_TAG = 'f2d171c7b3d6', '8a49b0e2d93c'
OF_FINAL = 'a884c28b4e13'
OF_SENS = {'main': 'a884c28b4e13', 'smoothing 1.5 bins': 'c5f13b8b4e13', 'wall limit 1.5 cm': '7a55448b4e13',
           'pxd 7.5 cm': '0c43d68b4e13'}
NULL_COLOURS = {'place_shuffled': '#555555', 'kinematic': '#D9822B', 'speed': '#5B8DB8'}
NULL_LABELS = {'place_shuffled': 'field shuffle', 'kinematic': 'kinematic', 'speed': 'speed'}
#: the six mouse-level claims of maze_spatial_report, in the order of MAZE_SPATIAL.md
CLAIM_ORDER = ['dec_node_bridge_ba', 'rsa21_ED_X', 'rsa9_ED_X', 'dec_corner_cardinal_ba', 'dec_same_type_error_share',
               'rsa21_contrast']
MOVE_ORDER = ['generic', 'follow_slider', 'follow_diagonal', 'trace', 'vector_slider', 'vector_diagonal',
              'place_vacated', 'remap_E', 'E_change']
ROLE_COLOURS = {'V': '#C03030', 'S': '#2C2C2A', 'N': '#009B77', 'E': '#B4B2A9'}


def save(fig, name: str) -> str:
    """Write name.pdf and name.png at the first version free for both; returns the PDF path."""
    import matplotlib as mpl
    os.makedirs(OUT, exist_ok=True)
    k = 0
    while any(os.path.exists(os.path.join(OUT, name + (f'.v{k}' if k else '') + e)) for e in ('.pdf', '.png')):
        k += 1
    stem = os.path.join(OUT, name + (f'.v{k}' if k else ''))
    with mpl.rc_context({'savefig.bbox': 'tight'}):
        fig.savefig(stem + '.pdf')
        fig.savefig(stem + '.png', dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'wrote {stem}.pdf / .png', flush=True)
    return stem + '.pdf'


def wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def interval(ax, x, y, colour, width=0.18):
    """5-95 % bar, interquartile box and median tick of a null distribution at x."""
    lo, q1, md, q3, hi = np.nanpercentile(y, [5, 25, 50, 75, 95])
    ax.plot([x, x], [lo, hi], color=colour, lw=0.8)
    ax.add_patch(__import__('matplotlib').patches.Rectangle((x - width / 2, q1), width, q3 - q1, facecolor=colour,
                                                            alpha=0.35, edgecolor=colour, lw=0.6))
    ax.plot([x - width / 2, x + width / 2], [md, md], color=colour, lw=1.0)


# ---------------------------------------------------------------------------
# B: maze claims
# ---------------------------------------------------------------------------

def maze_run(dataset: str, tag: str) -> dict:
    import maze_spatial_report as r
    return r.load(dataset, tag)


def mouse_draws(run: dict, stat: str, null: str, exclude=()) -> dict:
    """mouse -> (real mean over recdays, null draws averaged over the same recdays), as mouse_null_tests."""
    pop, sims = run['population'], run['null_sims']
    real = pop[pop['recipe'] == 'decided'][['recday', 'mouse', stat]].dropna()
    s = sims[(sims['null'] == null) & (sims['stat'] == stat)]
    by_rd = {rd: g.sort_values('sim')['value'].to_numpy(float) for rd, g in s.groupby('recday')}
    out = {}
    for mouse, g in real.groupby('mouse'):
        if mouse in exclude:
            continue
        rds = [rd for rd in g['recday'] if rd in by_rd and np.isfinite(by_rd[rd]).any()]
        vals = [by_rd[rd][np.isfinite(by_rd[rd])] for rd in rds]
        n = min((len(v) for v in vals), default=0)
        if n:
            out[mouse] = (float(g.set_index('recday').loc[rds, stat].mean()), np.mean([v[:n] for v in vals], axis=0))
    return out


def fig_B1(runs: dict) -> None:
    """Per dataset, six claims: each mouse's real mean against its field-shuffle, kinematic and speed nulls (B1)."""
    import matplotlib.pyplot as plt
    import maze_spatial as ms
    import maze_spatial_report as r
    import spatial_plots as sp
    claims = {c[0]: c for c in r.MOUSE_CLAIMS}
    for d in ('mec', 'lec', 'pfc'):
        run = runs[('B1', d)]
        excl = r.UNINFORMATIVE.get(d, ())
        fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
        for ax, stat in zip(axs.ravel(), CLAIM_ORDER):
            _, label, chance, direction = claims[stat]
            mice = None
            for j, null in enumerate(('place_shuffled', 'kinematic', 'speed')):
                dr = mouse_draws(run, stat, null, excl)
                tests = ms.mouse_null_tests(run['population'], run['null_sims'], stat, null, direction, chance,
                                            exclude_mice=excl).set_index('mouse')
                mice = sorted(dr) if mice is None else mice
                for i, m in enumerate(mice):
                    if m not in dr:
                        continue
                    x = i + (j - 1) * 0.24
                    interval(ax, x, dr[m][1], NULL_COLOURS[null])
                    if bool(tests.loc[m, 'hit']):
                        ax.text(x, np.nanpercentile(dr[m][1], 95), '*', ha='center', va='bottom',
                                color=NULL_COLOURS[null], fontsize=9)
            for i, m in enumerate(mice):
                dr = mouse_draws(run, stat, 'place_shuffled', excl)
                ax.plot(i, dr[m][0], 'o', color=sp.TRUE, ms=4, zorder=5)
            ax.axhline(0.0 if chance is None else chance, color=sp.STONE, lw=0.6, ls='--', zorder=0)
            ax.set_xticks(range(len(mice)))
            ax.set_xticklabels(mice, rotation=45, fontsize=7)
            ax.set_title(label, fontsize=7)
        handles = [plt.Line2D([], [], marker='o', ls='', color=sp.TRUE, label='real, mean over recdays')] + \
                  [plt.Rectangle((0, 0), 1, 1, color=NULL_COLOURS[k], alpha=0.5, label=f'{NULL_LABELS[k]} null')
                   for k in NULL_COLOURS]
        fig.suptitle(f"{d.upper()} maze, run {MAZE['B1']}: bars 5-95 %, boxes 25-75 % of each mouse's null; "
                     f"* = mouse p <= 0.05" + (f"; {', '.join(excl)} not shown" if excl else ''), y=0.99, fontsize=7)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        save(fig, f'B1_maze_claims_{d}')


def fig_B2(runs: dict) -> None:
    """Scorecard: mice beyond each null per claim, dataset and run."""
    import matplotlib.pyplot as plt
    import maze_spatial_report as r
    cols = [('B1', 'place_shuffled', 'field shuffle'), ('B1', 'speed', 'speed'), ('B1', 'kinematic', 'kinematic'),
            ('B2', 'kinematic', 'kinematic,\ntravel-turn matched'), ('B2-head', 'kinematic', 'kinematic,\nhead-turn matched')]
    labels = {c[0]: c[1] for c in r.MOUSE_CLAIMS}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
    for ax, d in zip(axs, ('mec', 'lec', 'pfc')):
        M = np.full((len(CLAIM_ORDER), len(cols)), np.nan)
        T = [['' for _ in cols] for _ in CLAIM_ORDER]
        for j, (run_name, null, _) in enumerate(cols):
            if (run_name, d) not in runs:
                continue
            mc = r.mouse_claims(d, runs[(run_name, d)]).set_index(['statistic', 'null'])
            for i, stat in enumerate(CLAIM_ORDER):
                key = (labels[stat], null)
                if key in mc.index:
                    h, n = int(mc.loc[key, 'hits']), int(mc.loc[key, 'n_mice'])
                    M[i, j], T[i][j] = h / n, f'{h}/{n}'
        ax.imshow(M, cmap='Greens', vmin=0, vmax=1, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, T[i][j] or 'n/a', ha='center', va='center', fontsize=7,
                        color='white' if np.isfinite(M[i, j]) and M[i, j] > 0.6 else 'black')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[2] for c in cols], rotation=60, ha='right', fontsize=6)
        ax.set_title(d.upper())
        ax.spines[:].set_visible(False)
    axs[0].set_yticks(range(len(CLAIM_ORDER)))
    axs[0].set_yticklabels([labels[s] for s in CLAIM_ORDER], fontsize=7)
    fig.suptitle('Mice beyond each null (p <= 0.05, generalisation side); runs '
                 f"B1 {MAZE['B1']}, B2 {MAZE['B2']}, B2-head {MAZE['B2-head']}", fontsize=7)
    fig.tight_layout()
    save(fig, 'B2_maze_scorecard')


def fig_B3() -> None:
    """Turning and port QC from the B1 shards: median |turn| by location type per recday, and goal-arrival coverage."""
    import matplotlib.pyplot as plt
    import maze_spatial_report as r
    import spatial_plots as sp
    rows = []
    for d, root in r.ROOTS.items():
        for f in sorted(glob.glob(os.path.join(root, MAZE['B1'], 'shards', '*.pkl'))):
            sh = pickle.load(open(f, 'rb'))
            k = sh.get('kinematics', {})
            row = dict(dataset=d, recday=sh['recday'], coverage=k.get('reward_coverage', np.nan),
                       port_fraction=k.get('port_fraction', np.nan))
            for src in ('travel', 'head'):
                for typ, v in (k.get(f'median_turn_{src}') or {}).items():
                    row[f'{src}_{typ}'] = v
            rows.append(row)
            del sh
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
    types = ['corner', 'cardinal', 'centre', 'bridge']
    for k_, (src, ax) in enumerate((('travel', axs[0]), ('head', axs[1]))):
        for j, d in enumerate(('mec', 'lec', 'pfc')):
            g = df[df.dataset == d]
            Y = g[[f'{src}_{t}' for t in types]].to_numpy(float)
            if not np.isfinite(Y).any():
                continue
            x = np.arange(len(types)) + (j - 1) * 0.22
            for y in Y:
                ax.plot(x, y, color=sp.DATASET_COLOURS[d], alpha=0.25, lw=0.5)
            ax.plot(x, np.nanmedian(Y, axis=0), 'o-', color=sp.DATASET_COLOURS[d], ms=3, lw=1.0, label=d.upper())
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types)
        ax.set_ylabel(f'median |{src} turn|, deg/s')
        ax.set_title(f'{src} turning above 10 cm/s, per recday', fontsize=7)
    axs[0].legend(frameon=False, fontsize=7)
    ax = axs[2]
    for j, d in enumerate(('mec', 'lec', 'pfc')):
        g = df[df.dataset == d]
        ax.scatter(np.full(len(g), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(g)), g['coverage'],
                   color=sp.DATASET_COLOURS[d], s=8)
    ax.axhline(0.9, color=sp.STONE, ls='--', lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['mEC', 'LEC', 'PFC'])
    ax.set_ylabel('goal arrivals with an at-port\nepisode within 1 s')
    ax.set_title('at-port state, per recday', fontsize=7)
    fig.suptitle(f"Kinematic inputs, run {MAZE['B1']}: mEC 12, LEC 25, PFC 24 recdays; PFC has no head direction",
                 fontsize=7)
    fig.tight_layout()
    save(fig, 'B3_maze_turning_and_ports')


# ---------------------------------------------------------------------------
# C: maze-open-field link
# ---------------------------------------------------------------------------

def link_inputs():
    lk = {os.path.basename(f)[:-4]: pickle.load(open(f, 'rb'))
          for f in sorted(glob.glob(os.path.join(MEC, 'data', 'spatial', 'link', LINK_TAG, 'shards', '*.pkl')))}
    of = {rd: pickle.load(open(os.path.join(MEC, 'data', 'spatial', 'of', LINK_OF_TAG, 'shards', f'{rd}.pkl'), 'rb'))
          for rd in lk}
    return lk, of


def fig_C1(lk: dict, of: dict) -> None:
    """Per recday and arena session: each unit's offset arena - s * maze for every maze session, with the fitted rotation."""
    import matplotlib.pyplot as plt
    panels = [(rd, a) for rd, sh in lk.items() if not rd.startswith('bp02')
              for a in sorted({p['arena'] for p in sh['pairs']})]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0), subplot_kw=dict(projection='polar'))
    cmap = plt.get_cmap('tab10')
    for ax, (rd, a) in zip(axs.ravel(), panels):
        sh = lk[rd]
        s = sh['reflection']['s']
        ap = np.asarray(of[rd]['sessions'][a]['details']['pxd_pref'], float)
        pairs = [p for p in sh['pairs'] if p['arena'] == a and p['selection'] == 'hd_both' and 'phi_deg' in p]
        for k, p in enumerate(pairs):
            u = np.asarray(p['units'], int)
            mp = np.asarray(sh['maze_hd'][p['maze']]['pref'], float)
            delta = np.radians(wrap_deg(np.degrees(ap[u] - s * mp[u])))
            ax.scatter(delta, np.full(len(u), 1.0 + k), s=6, color=cmap(k % 10), alpha=0.8)
            ax.plot([np.radians(p['phi_deg'])] * 2, [0, len(pairs) + 1], color=cmap(k % 10), lw=0.7)
        ax.set_yticks([])
        ax.set_title(f"{rd[:4]} {rd[9:13]}/{rd[18:22]} arena s{a}\n{len(pairs)} maze sessions; "
                     f"R {np.min([p['R'] for p in pairs]):.2f}-{np.max([p['R'] for p in pairs]):.2f}" if pairs else
                     f'{rd[:4]} {rd[9:13]} arena s{a}: no testable pair', fontsize=6, pad=14)
    for ax in axs.ravel()[len(panels):]:
        ax.axis('off')
    fig.suptitle(f'Maze to open field: each ring is one maze session; dots are units HD-tuned in both; '
                 f'offset = arena - s x maze, s = -1; lines = fitted rotation (run {LINK_TAG})', fontsize=7)
    fig.tight_layout()
    save(fig, 'C1_link_hd_offsets')


def pooled(items: list) -> tuple:
    """(real mean, null draw i averaged) over a mouse's pairs or recdays, as maze_of_link.mouse_pool."""
    n = min(len(nl) for _, nl in items)
    return float(np.mean([r_ for r_, _ in items])), np.mean([np.asarray(nl, float)[:n] for _, nl in items], axis=0)


def fig_C2(lk: dict) -> None:
    """Mouse-level tests: HD coherence R against unit shuffles, and the class links against label shuffles."""
    import matplotlib.pyplot as plt
    import spatial_plots as sp
    mice = ['bp01', 'bp04']
    tests = [('pairs', 'hd_both', 'HD coherence R,\nunits HD in both'),
             ('pairs', 'hd_open_field', 'HD coherence R,\nopen-field HD units'),
             ('links', 'maze_R_by_of_hd', 'maze HD R: open-field\nHD cells - others'),
             ('links', 'edge_by_of_border', 'maze edge index: open-field\nborder cells - others'),
             ('links', 'of_speed_by_maze_node', 'open-field speed factor:\nothers - maze node cells'),
             ('links', 'of_si_vs_maze_stab', 'open-field information vs\nmaze stability, Spearman r')]
    fig, axs = plt.subplots(1, len(tests), figsize=(7.6, 2.6))
    for ax, (kind, key, label) in zip(axs, tests):
        ticks = []
        for i, m in enumerate(mice):
            items = []
            for rd, sh in lk.items():
                if not rd.startswith(m):
                    continue
                if kind == 'pairs':
                    items += [(p['R'], p['null_R']) for p in sh['pairs'] if p['selection'] == key and 'R' in p]
                elif 'real' in sh['links'].get(key, {}):
                    items.append((sh['links'][key]['real'], sh['links'][key]['null']))
            if not items:
                ticks.append(f'{m}\nnone')
                continue
            x, y = pooled(items)
            interval(ax, i, y, sp.SHUFFLE, width=0.3)
            p = (1 + (y >= x).sum()) / (1 + len(y))
            ax.plot(i, x, 'o', color=sp.TRUE, ms=4)
            ticks.append(f'{m}\np {p:.2g}\nn {len(items)}')
        ax.set_xticks(range(len(mice)))
        ax.set_xticklabels(ticks, fontsize=6)
        ax.set_xlim(-0.6, len(mice) - 0.4)
        ax.set_title(label, fontsize=6)
    fig.suptitle(f'Per mouse: real (red) against the shuffle null (grey bar 5-95 %, box 25-75 %); n = pairs or recdays; '
                 f'run {LINK_TAG}', fontsize=7)
    fig.tight_layout()
    save(fig, 'C2_link_mouse_tests')


def fig_C3(lk: dict) -> None:
    """The fitted rotation per maze-arena pair by day, both mice; maze-maze rotations; closure errors."""
    import matplotlib.pyplot as plt
    rows = []
    for rd, sh in lk.items():
        if rd.startswith('bp02'):
            continue
        for p in sh['pairs']:
            if p['selection'] == 'hd_both' and 'phi_deg' in p:
                rows.append(dict(mouse=rd[:4], recday=rd, arena=p['arena'], maze=p['maze'], phi=p['phi_deg'], p=p['p'], n=p['n']))
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[2.2, 1.2, 1]))
    ax = axs[0]
    keys = sorted({(r.recday[5:], r.arena) for r in df.itertuples()})
    colours = {'bp01': '#BE3455', 'bp04': '#0F4C81'}
    for r in df.itertuples():
        x = keys.index((r.recday[5:], r.arena)) + (-0.12 if r.mouse == 'bp01' else 0.12)
        ax.scatter(x, r.phi, s=10 + r.n, color=colours[r.mouse], alpha=1.0 if r.p <= 0.05 else 0.3,
                   edgecolor='none')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k[0][4:8]}/{k[0][13:17]}\narena s{k[1]}" for k in keys], fontsize=6)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel('rotation phi, deg')
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=c, label=m) for m, c in colours.items()],
              frameon=False, fontsize=7)
    ax.set_title('one dot per maze session; faded p > 0.05; size ~ units', fontsize=6)
    ax = axs[1]
    for rd, sh in lk.items():
        if rd.startswith('bp02'):
            continue
        ph = [mp['phi_deg'] for mp in sh['maze_pairs'] if 'phi_deg' in mp]
        x = np.full(len(ph), sorted(k for k in lk if not k.startswith('bp02')).index(rd))
        ax.scatter(x + np.random.default_rng(0).uniform(-0.15, 0.15, len(ph)), ph, s=6,
                   color=[('#C03030' if abs(v) > 45 else '#555555') for v in ph])
    names = sorted(k for k in lk if not k.startswith('bp02'))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f'{k[:4]}\n{k[9:13]}' for k in names], fontsize=6)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_title('maze-maze rotations; red |phi| > 45', fontsize=6)
    ax = axs[2]
    cl = pd.concat([pd.DataFrame(sh['closure']).assign(recday=rd) for rd, sh in lk.items()
                    if sh['closure'] and not rd.startswith('bp02')])
    for j, kind in enumerate(('maze pair', 'arena pair')):
        v = np.abs(cl[cl.kind == kind]['error_deg'].to_numpy(float))
        ax.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=6, color='#555555')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['maze pairs', 'arena pair\nbp01 04-02/03'], fontsize=6)
    ax.set_ylabel('|closure error|, deg')
    ax.set_title('closure', fontsize=6)
    fig.suptitle(f'The rotation between rigs is shared by both mice on 03-24/25; bp01 03-21/23 maze s3 is turned '
                 f'180 deg (run {LINK_TAG})', fontsize=7)
    fig.tight_layout()
    save(fig, 'C3_link_rotation_by_day')


# ---------------------------------------------------------------------------
# D: LEC object box
# ---------------------------------------------------------------------------

def of_shards(tag: str, dataset: str = 'lec') -> dict:
    root = os.path.join(LEC if dataset == 'lec' else MEC, 'data', 'spatial', 'of', tag, 'shards')
    return {os.path.basename(f)[:-4]: pickle.load(open(f, 'rb')) for f in sorted(glob.glob(os.path.join(root, '*.pkl')))}


def within_day_moves(shards: dict):
    """(recday, s1, s2, object_move result) for the within-day pairs: the ones with a moving object."""
    for rd, sh in shards.items():
        for (s1, s2), rot in sorted(sh['rotations'].items()):
            m_ = rot.get('object_move') if isinstance(rot, dict) else None
            if m_ and 'classes' in m_ and m_['layout']['movers']:
                yield rd, s1, s2, m_


def role_of(cls: str, layout: dict) -> str:
    if '[' not in cls:
        return cls
    kind, lab = cls.split('[')[0], cls.split('[')[1].rstrip(']')
    return f"{kind}_{'slider' if layout['movers'][lab]['frm'] == 'V' else 'diagonal'}"


def fig_D1(shards_lec: dict) -> None:
    """Per session, the fraction of broad eligible units in each class against one-shift-for-all chance."""
    import matplotlib.pyplot as plt
    import spatial_plots as sp
    for d in ('lec', 'mec'):
        root = os.path.join(LEC if d == 'lec' else MEC, 'data', 'spatial', 'of', OF_FINAL, 'merged')
        sessions = pd.read_csv(sorted(glob.glob(os.path.join(root, 'of_sessions*.csv')))[-1])
        # grid is scored but not classified in the 50 cm box
        classes = ['border', 'spatial', 'hd', 'speed'] if d == 'lec' else ['grid', 'border', 'spatial', 'hd', 'speed']
        df = sessions[(sessions['pool'] == 'broad') & sessions['cls'].isin(classes)]
        keys = df[['recday', 'session']].drop_duplicates().sort_values(['recday', 'session'])
        labels = [f"{r.recday[:4]} {r.recday[9:13]} s{r.session}" if d == 'mec' else f"{r.recday[:4]} s{r.session}"
                  for r in keys.itertuples()]
        fig, axs = plt.subplots(1, len(classes), figsize=(7.4, 2.9), sharey=False)
        for ax, cls in zip(axs, classes):
            g = df[df['cls'] == cls].set_index(['recday', 'session']).loc[list(map(tuple, keys.to_numpy()))]
            x = np.arange(len(g))
            frac = g['observed'] / g['n_units']
            ax.bar(x, frac, color=sp.CLASS_COLOURS[cls], width=0.75)
            ax.scatter(x, g['chance_p95'] / g['n_units'], marker='_', color=sp.SHUFFLE, s=30, zorder=5)
            sig = g['p'].to_numpy() <= 0.05
            ax.scatter(x[sig], frac.to_numpy()[sig] + 0.02 * max(frac.max(), 0.05), marker='*', color='k', s=8)
            ax.set_title(f"{cls}: {int(g['observed'].sum())} units", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
        axs[0].set_ylabel('fraction of broad units')
        fig.suptitle(f'{d.upper()} open field, run {OF_FINAL}: bar = observed, grey dash = chance p95 (one shift for all '
                     f'units), * = p <= 0.05' + ('; grid is not classified in the 50 cm box' if d == 'lec' else ''),
                     fontsize=7)
        fig.tight_layout()
        save(fig, f'D1_{d}_session_classes')


def fig_D2(shards_lec: dict) -> None:
    """Object layouts of the 10 within-day pairs: corners V, S, N, E and the two moves."""
    import matplotlib.pyplot as plt
    moves = list(within_day_moves(shards_lec))
    fig, axs = plt.subplots(2, 5, figsize=(7.4, 3.3))
    for ax, (rd, s1, s2, m_) in zip(axs.T.ravel(), moves):
        lay = m_['layout']
        side = shards_lec[rd]['sessions'][s1]['side_cm']
        ax.add_patch(plt.Rectangle((0, 0), side, side, fill=False, color='#2C2C2A', lw=0.8))
        for L, ro in lay['roles'].items():
            ax.text(*ro['xy'], L, color=ROLE_COLOURS[L], ha='center', va='center', fontsize=9, fontweight='bold')
        for lab, mv in lay['movers'].items():
            a, b = np.asarray(mv['A'], float), np.asarray(mv['B'], float)
            ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', lw=1.0,
                                                           color='#C03030' if mv['frm'] == 'V' else '#0F4C81'))
            ax.text(*((a + b) / 2), lab.split('_', 2)[-1], fontsize=5, ha='center', va='bottom')
        ax.set_xlim(-2, side + 2)
        ax.set_ylim(side + 2, -2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{rd[:4]} s{s1} -> s{s2}', fontsize=7)
        ax.spines[:].set_visible(False)
    fig.suptitle('Within-day object moves: V vacated, S held by the other object, N newly occupied, E empty; '
                 'red = slider V -> S, blue = diagonal S -> N; camera frame, y down', fontsize=7)
    fig.tight_layout()
    save(fig, 'D2_lec_object_layouts')


def move_table(shards: dict) -> pd.DataFrame:
    rows = []
    for rd, s1, s2, m_ in within_day_moves(shards):
        for c, v in m_['classes'].items():
            rows.append(dict(mouse=rd[:4], recday=rd, pair=f'{s1}->{s2}', role=role_of(c, m_['layout']),
                             obs=int(np.sum(v)), chance=np.asarray(m_['chance'][c]),
                             n_elig=int(np.sum(m_['eligible'][c]))))
    return pd.DataFrame(rows)


def fig_D3(shards_lec: dict) -> None:
    """Per class and mouse: units over both day-pairs against the pooled paired-shift chance distribution."""
    import matplotlib.pyplot as plt
    import spatial_plots as sp
    from scipy import stats
    t = move_table(shards_lec)
    mice = sorted(t.mouse.unique())
    fig, axs = plt.subplots(2, 5, figsize=(7.6, 3.8))
    for ax, role in zip(axs.ravel(), MOVE_ORDER):
        g0 = t[t.role == role]
        hits = 0
        for i, m in enumerate(mice):
            g = g0[g0.mouse == m]
            n = min(len(c) for c in g.chance)
            ch = np.sum([c[:n] for c in g.chance], axis=0)
            obs = int(g.obs.sum())
            p = (1 + (ch >= obs).sum()) / (1 + n)
            hits += p <= 0.05
            interval(ax, i, ch, sp.SHUFFLE, width=0.4)
            ax.plot(i, obs, 'o', color=sp.TRUE if p <= 0.05 else '#2C2C2A', ms=4, zorder=5)
        ax.set_xticks(range(len(mice)))
        ax.set_xticklabels(mice, rotation=45, fontsize=6)
        tag = 'reported' if role in ('place_vacated', 'remap_E', 'E_change') else 'primary'
        ax.set_title(f"{role.replace('_', ' ')} ({tag})\n{hits}/5 mice, binomial p "
                     f"{stats.binom.sf(hits - 1, 5, 0.05):.2g}", fontsize=6)
    ax = axs.ravel()[-1]
    g = t[t.role.isin(['generic', 'remap_E'])].groupby(['mouse', 'role']).obs.sum().unstack()
    x = np.arange(len(mice))
    ax.bar(x - 0.18, g.loc[mice, 'generic'], 0.36, color='#9B1B30', label='generic, V -> N')
    ax.bar(x + 0.18, g.loc[mice, 'remap_E'], 0.36, color='#B4B2A9', label='remap, V -> E')
    ax.set_xticks(x)
    ax.set_xticklabels(mice, rotation=45, fontsize=6)
    ax.set_ylim(0, 1.6 * np.nanmax(g.to_numpy(float)))
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    ax.set_title('generic against its matched\nremapping control', fontsize=6)
    axs[0, 0].set_ylabel('units, both day-pairs')
    axs[1, 0].set_ylabel('units, both day-pairs')
    fig.suptitle(f'LEC object moves per mouse: dot = units (red p <= 0.05), grey = chance, bar 5-95 %; '
                 f'run {OF_FINAL}', fontsize=7)
    fig.tight_layout()
    save(fig, 'D3_lec_move_classes')


def draw_map(ax, sess: dict, u: int, roles: dict, title: str) -> None:
    rate = sess['details']['rate'][u]
    side = sess['side_cm']
    ax.imshow(rate.T, origin='upper', cmap='viridis', extent=(0, side, side, 0), interpolation='nearest')
    for o in sess['objects']:
        ax.add_patch(__import__('matplotlib').patches.Circle(o['xy'], max(o['r_obj'], 2.0), fill=False, color='w', lw=0.8))
    for L, ro in roles.items():
        ax.text(ro['xy'][0], ro['xy'][1], L, color=ROLE_COLOURS[L], fontsize=5, ha='center',
                va='center', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    pk = np.nanmax(rate) if np.isfinite(rate).any() else 0.0
    ax.set_title(f'{title} {pk:.1f} Hz', fontsize=5)


def fig_D4(shards_lec: dict) -> None:
    """Rate maps before and after the move for every unit in a primary class, and the reported controls."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    picks = []
    for rd, s1, s2, m_ in within_day_moves(shards_lec):
        for c, v in m_['classes'].items():
            for u in np.nonzero(np.asarray(v, bool))[0]:
                picks.append((role_of(c, m_['layout']), rd, s1, s2, int(u), m_['layout']['roles']))
    order = {r: i for i, r in enumerate(MOVE_ORDER)}
    picks.sort(key=lambda x: (order.get(x[0], 99), x[1], x[2], x[4]))
    groups = [('generic object cells: firing leaves V and appears at N', [p for p in picks if p[0] == 'generic']),
              ('object-following, trace and object-vector units', [p for p in picks if p[0] in
               ('follow_slider', 'follow_diagonal', 'trace', 'vector_slider', 'vector_diagonal')]),
              ('reported only: place at V, remapping V -> E, E change',
               [p for p in picks if p[0] in ('place_vacated', 'remap_E', 'E_change')])]
    os.makedirs(OUT, exist_ok=True)
    k = 0
    while os.path.exists(os.path.join(OUT, 'D4_lec_move_units' + (f'.v{k}' if k else '') + '.pdf')):
        k += 1
    path = os.path.join(OUT, 'D4_lec_move_units' + (f'.v{k}' if k else '') + '.pdf')
    first_png = None
    with PdfPages(path) as pdf:
        for title, units in groups:
            for start in range(0, max(len(units), 1), 24):
                chunk = units[start:start + 24]
                fig, axs = plt.subplots(6, 8, figsize=(7.4, 6.4))
                for ax in axs.ravel():
                    ax.axis('off')
                for i, (role, rd, s1, s2, u, roles) in enumerate(chunk):
                    r_, c_ = divmod(i, 4)
                    sh = shards_lec[rd]
                    for j, s in enumerate((s1, s2)):
                        ax = axs[r_, 2 * c_ + j]
                        ax.axis('on')
                        draw_map(ax, sh['sessions'][s], u, roles,
                                 f"{rd[:4]} u{u} s{s}" + (f"\n{role.replace('_', ' ')}" if j == 0 else '\n'))
                fig.suptitle(f'LEC {title}; each pair = before | after; {len(units)} units; run {OF_FINAL}', fontsize=7)
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                pdf.savefig(fig)
                if first_png is None:
                    first_png = path[:-4] + '.png'
                    fig.savefig(first_png, dpi=150)
                plt.close(fig)
    print(f'wrote {path} (+ first page {first_png})', flush=True)


def fig_D5() -> None:
    """Session classes and object-move counts across the box sensitivity runs."""
    import matplotlib.pyplot as plt
    import spatial_plots as sp
    cls_rows, mv_rows = [], []
    for name, tag in OF_SENS.items():
        root = os.path.join(LEC, 'data', 'spatial', 'of', tag)
        s = pd.read_csv(sorted(glob.glob(os.path.join(root, 'merged', 'of_sessions*.csv')))[-1])
        b = s[(s.pool == 'broad') & s.cls.isin(['border', 'hd', 'spatial', 'speed'])]
        for c, g in b.groupby('cls'):
            cls_rows.append(dict(run=name, cls=c, units=int(g['observed'].sum())))
        t = move_table(of_shards(tag))
        for r_, g in t.groupby('role'):
            mv_rows.append(dict(run=name, role=r_, units=int(g.obs.sum())))
    a, m = pd.DataFrame(cls_rows), pd.DataFrame(mv_rows)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[1, 2]))
    cols = ['#2C2C2A', '#5B8DB8', '#D9822B', '#88B04B']
    for ax, df, key, order in ((axs[0], a, 'cls', ['border', 'hd', 'spatial', 'speed']), (axs[1], m, 'role', MOVE_ORDER)):
        x = np.arange(len(order))
        for j, name in enumerate(OF_SENS):
            y = df[df.run == name].set_index(key).reindex(order)['units'].to_numpy(float)
            ax.bar(x + (j - 1.5) * 0.2, y, 0.2, color=cols[j], label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace('_', ' ') for o in order], rotation=45, ha='right', fontsize=6)
        ax.set_ylabel('units summed over sessions' if key == 'cls' else 'units, 10 day-pairs')
    axs[1].legend(frameon=False, fontsize=6, loc='upper right')
    axs[0].set_title('session classes, broad units', fontsize=7)
    axs[1].set_title('object-move classes', fontsize=7)
    fig.suptitle('LEC box sensitivity: one box50 setting changed per run; main = 2.0 cm wall limit, 1-bin smoothing, '
                 'pxd 5 cm', fontsize=7)
    fig.tight_layout()
    save(fig, 'D5_lec_sensitivity')


# ---------------------------------------------------------------------------
# V: the gate measurements behind the settings
# ---------------------------------------------------------------------------

def log_lines(job: int, name_part: str) -> list:
    f = glob.glob(os.path.join(LOGS, f'*{name_part}*-{job}.out'))
    return open(f[0]).read().splitlines() if f else []


def fig_V1() -> None:
    """Box smoothing (job 3752965), conjunctive border x HD by preference (3752994), object-move test level (3755058)."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.8))
    ax = axs[0]
    rx = re.compile(r'sigma ([\d.]+) (.+?)\s*: planted border median ([-\d.]+), called ([\d.]+) .*finite shuffles ([\d.]+)')
    rows = [m.groups() for l in log_lines(3752965, 'diag_box_border2') if (m := rx.search(l))]
    d = pd.DataFrame(rows, columns=['sigma', 'kind', 'score', 'called', 'finite']).astype(
        dict(sigma=float, score=float, called=float, finite=float))
    for kind, mk, col in (('left lam 5', 'o', '#BE3455'), ('bottom+right lam 5', 's', '#0F4C81'),
                          ('near-wall place', '^', '#555555'), ('centre place', 'v', '#B4B2A9')):
        g = d[d.kind == kind].sort_values('sigma')
        ax.plot(g.sigma, g.called, marker=mk, color=col, ms=4, lw=0.8, label=f'{kind}: called border')
    g = d[d.kind == 'bottom+right lam 5'].sort_values('sigma')
    ax.plot(g.sigma, g.finite, 'k:', lw=0.8, label='two-wall: shifted maps with\na valid border field')
    ax.set_xlabel('box smoothing, bins of 2.5 cm')
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_ylabel('fraction')
    ax.legend(frameon=False, fontsize=5)
    ax.set_title('border detection in the 50 cm box', fontsize=7)
    ax = axs[1]
    ry = re.compile(r'HD pref\s+(\d+) deg \(dwell\s+([\d.]+) s\).*called border ([\d.]+)')
    lines = log_lines(3752994, 'diag_box_conj')
    i0 = next((i for i, l in enumerate(lines) if l.startswith('lam 5')), 0)
    rows = [m.groups() for l in lines[i0:i0 + 9] if (m := ry.search(l))]
    e = pd.DataFrame(rows, columns=['pref', 'dwell', 'called']).astype(float)
    ax.bar(e.pref, e.called, width=30, color='#555555')
    for r in e.itertuples():
        ax.text(r.pref, r.called + 0.03, f'{r.dwell:.0f} s', ha='center', fontsize=5)
    ax.set_xticks(e.pref)
    ax.set_xlabel('HD preference, deg; 0 = away from the wall')
    ax.set_ylabel('conjunctive cells called border')
    ax.set_ylim(0, 1.15)
    ax.set_title('border x HD cells, ah10 s1; label = seconds within\n5 cm of the wall facing that way, all speeds', fontsize=6)
    ax = axs[2]
    rz = re.compile(r'regulariser ([\d.]+) Hz, every term at p(\d+)')
    rk = re.compile(r'^\s+(\w+)\s+n\s+\d+: follow_slider ([\d.]+), follow_diag ([\d.]+), trace ([\d.]+), generic ([\d.]+) \| any ([\d.]+)')
    cur, rows = None, []
    for l in log_lines(3755058, 'diag_move_fire5'):
        if (m := rz.search(l)):
            cur = f'{float(m.group(1)):g} Hz, p{m.group(2)}'
        elif cur and (m := rk.search(l)):
            rows.append((cur, m.group(1)) + tuple(float(x) for x in m.groups()[1:]))
    f = pd.DataFrame(rows, columns=['variant', 'kind', 'fs', 'fd', 'tr', 'ge', 'any'])
    variants = list(dict.fromkeys(f.variant))
    neg = ['place_V', 'place_S', 'border_VS', 'border_VN', 'hd_VN', 'hd_V', 'slow', 'gain', 'anchor']
    x = np.arange(len(variants))
    for kind, col, lab, mk in (('follow_slider', '#BE3455', 'follow, slider', 'o'), ('follow_diag', '#0F4C81', 'follow, diagonal', 's'),
                               ('trace', '#009B77', 'trace', '^'), ('generic', '#9B1B30', 'generic', 'D')):
        y = [float(f[(f.variant == v) & (f.kind == kind)][{'follow_slider': 'fs', 'follow_diag': 'fd', 'trace': 'tr', 'generic': 'ge'}[kind]].iloc[0])
             for v in variants]
        ax.plot(x, y, marker=mk, color=col, ms=4, lw=0.8, label=lab)
    yneg = [f[(f.variant == v) & f.kind.isin(neg)]['any'].max() for v in variants]
    ax.plot(x, yneg, 'k--', lw=0.8, label='worst negative type, any class')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha='right', fontsize=6)
    ax.axhline(0.8, color='#B4B2A9', lw=0.6)
    ax.set_ylabel('planted cells detected')
    ax.legend(frameon=False, fontsize=5)
    ax.set_title('object-move tests, 20 planted cells per type; chosen:\n0.1 Hz, four-term classes at p95, trace at p99', fontsize=6)
    fig.suptitle('Gate measurements behind the box50 settings (diagnostic jobs 3752965, 3752994, 3755058)', fontsize=7)
    fig.tight_layout(w_pad=2.5)
    save(fig, 'V1_gate_decisions')


def main(argv=None) -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', nargs='+', default=['B', 'C', 'D', 'V'], choices=['B', 'C', 'D', 'V'])
    ap.add_argument('--out', default=OUT, help='output folder (a scratch folder for test runs)')
    args = ap.parse_args(argv)
    OUT = args.out
    import matplotlib
    matplotlib.use('Agg')
    import spatial_plots as sp
    sp.setup_style()
    if 'B' in args.only:
        runs = {(name, d): maze_run(d, tag) for name, tag in MAZE.items() for d in ('mec', 'lec', 'pfc')
                if not (name == 'B2-head' and d == 'pfc')}
        fig_B1(runs)
        fig_B2(runs)
        fig_B3()
    if 'C' in args.only:
        lk, of = link_inputs()
        fig_C1(lk, of)
        fig_C2(lk)
        fig_C3(lk)
    if 'D' in args.only:
        sh = of_shards(OF_FINAL)
        fig_D1(sh)
        fig_D2(sh)
        fig_D3(sh)
        fig_D4(sh)
        fig_D5()
    if 'V' in args.only:
        fig_V1()
    return 0


if __name__ == '__main__':
    sys.exit(main())
