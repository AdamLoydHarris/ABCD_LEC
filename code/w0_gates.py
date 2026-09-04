"""W0 gates for the anatomy split — every data-dependent check, in one data_dic load.

Run as a script. Loading `data_dic_lec.pkl` costs ~3.8 GB on disk and a few minutes, so
every gate that needs it lives here and the results are cached to
`data/processed_data/w0_gates.pkl` for the notebooks to read.

Gates (see docs/ANATOMY_SPLIT_PLAN.md and the approved plan, W0.1):

  1. length gate      len(unit_regions[rd]) == Neuron_raw.shape[0], 25/25, HARD ASSERT
  2. HD column check  earL2earR_deg - back2mid_deg ~ +/-90 deg  (gates the HD fix)
  3. depth ordering   y_um orders regions consistently with the insertion trajectory
  4. quality table    mean_rate_hz / sd / r2_full per region, published BEFORE any tuning

Gate 3 and 4 read `unit_regions.pkl` and `clustering__meta.pkl` and do not strictly need
`data_dic`, but they are cheap and belong in the same report.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

UNIT_REGIONS = os.path.join(REPO, 'data', 'processed_data', 'unit_regions.pkl')
CLUSTER_META = os.path.join(REPO, 'data', 'glm_outputs', 'LEC_selectivity_geometry',
                            'clustering__meta.pkl')
OUT_PATH = os.path.join(REPO, 'data', 'processed_data', 'w0_gates.pkl')

#: Coarse region labels, ordered superficial -> deep along a typical trajectory.
GROUP_ORDER = ['ENTl-sup', 'ENTl-deep', 'ENTm', 'SUB/ProS', 'CA1/HPF', 'fibre/other']


def _load_pickle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Gate 1 - length gate
# ---------------------------------------------------------------------------

def gate_length(data_dic, unit_regions, *, strict=True):
    """`len(unit_regions[rd])` must equal `Neuron_raw.shape[0]` for every recday.

    This is the check that originally caught the ly05 recday mismatch: the neural data is
    binned onto the behavioural timeline, so every other shape check passes even when the
    spikes come from the wrong day. Only the unit COUNT separates them.
    """
    rows = []
    for rd in sorted(unit_regions):
        n_reg = len(unit_regions[rd])
        if rd not in data_dic:
            rows.append({'recday': rd, 'n_unit_regions': n_reg, 'n_neuron_raw': None,
                         'status': 'ABSENT from data_dic'})
            continue
        sessions = [s for s in data_dic[rd] if isinstance(data_dic[rd][s], dict)]
        first = data_dic[rd][sessions[0]]
        n_raw = int(np.asarray(first['Neuron_raw']).shape[0])
        # every session of a recday is the same sorted block, so the count must be constant
        counts = {int(np.asarray(data_dic[rd][s]['Neuron_raw']).shape[0])
                  for s in sessions if 'Neuron_raw' in data_dic[rd][s]}
        rows.append({'recday': rd, 'n_unit_regions': n_reg, 'n_neuron_raw': n_raw,
                     'n_sessions': len(sessions),
                     'consistent_across_sessions': len(counts) == 1,
                     'status': 'PASS' if n_reg == n_raw else 'FAIL'})
    df = pd.DataFrame(rows)
    n_pass = int((df['status'] == 'PASS').sum())
    print(f"\n[gate 1] length gate: {n_pass}/{len(df)} recdays PASS")
    bad = df[df['status'] != 'PASS']
    if len(bad):
        print(bad.to_string(index=False))
    if strict and n_pass != len(df):
        raise AssertionError(
            f"length gate FAILED for {len(bad)} recday(s) — the unit_regions <-> Neuron_raw "
            f"join is positional and is not safe until this is 25/25:\n{bad.to_string(index=False)}")
    return df


# ---------------------------------------------------------------------------
# Gate 2 - head-direction column identity
# ---------------------------------------------------------------------------

def gate_hd_columns(data_dic, *, tol_deg=25.0):
    """`HD_raw` is (T, 2) = [back2mid_deg, earL2earR_deg], two angles ~90 deg apart.

    The GLM currently does `HD_raw.flatten()`, which interleaves the columns; the fix is to
    take column 0. That fix is only correct if the columns are what the preprocessing
    docstring says, so check the signed circular difference clusters near +/-90.

    Returns a per-session table and a summary. `tol_deg` is how far the circular MEAN of
    the difference may sit from +/-90 for a session to count as consistent.
    """
    rows = []
    for rd in sorted(data_dic):
        for s in sorted(k for k in data_dic[rd] if isinstance(data_dic[rd][k], dict)):
            sd = data_dic[rd][s]
            hd = sd.get('HD_raw')
            if hd is None:
                rows.append({'recday': rd, 'session': s, 'shape': None, 'status': 'no HD'})
                continue
            hd = np.asarray(hd, dtype=float)
            if hd.ndim != 2 or hd.shape[1] != 2:
                rows.append({'recday': rd, 'session': s, 'shape': hd.shape,
                             'status': f'unexpected shape {hd.shape}'})
                continue
            back2mid, ear2ear = hd[:, 0], hd[:, 1]
            ok = np.isfinite(back2mid) & np.isfinite(ear2ear)
            if ok.sum() < 100:
                rows.append({'recday': rd, 'session': s, 'shape': hd.shape,
                             'status': 'too few finite samples'})
                continue
            # signed circular difference, wrapped to (-180, 180]
            d = np.angle(np.exp(1j * np.deg2rad(ear2ear[ok] - back2mid[ok])), deg=True)
            # circular mean of |d| is not meaningful across a +/-90 bimodal split, so
            # summarise the two lobes separately and report the dominant one
            circ_mean = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(d)))))
            frac_near_plus90 = float(np.mean(np.abs(d - 90) < tol_deg))
            frac_near_minus90 = float(np.mean(np.abs(d + 90) < tol_deg))
            frac_near_90 = frac_near_plus90 + frac_near_minus90
            rows.append({
                'recday': rd, 'session': s, 'shape': hd.shape, 'n_finite': int(ok.sum()),
                'circ_mean_diff_deg': round(float(circ_mean), 2),
                'median_abs_diff_deg': round(float(np.median(np.abs(d))), 2),
                'frac_within_tol_of_90': round(frac_near_90, 3),
                'status': 'PASS' if frac_near_90 > 0.5 else 'CHECK',
            })
    df = pd.DataFrame(rows)
    have = df[df['status'].isin(['PASS', 'CHECK'])]
    print(f"\n[gate 2] HD columns: {len(have)} sessions with (T,2) HD_raw")
    if len(have):
        n_pass = int((have['status'] == 'PASS').sum())
        print(f"  sessions where >50% of samples sit within {tol_deg} deg of +/-90: "
              f"{n_pass}/{len(have)}")
        print(f"  median |earL2earR - back2mid| across sessions: "
              f"{have['median_abs_diff_deg'].median():.1f} deg")
        print(f"  pooled median frac_within_tol_of_90: "
              f"{have['frac_within_tol_of_90'].median():.3f}")
        if n_pass < len(have):
            print("  sessions needing a look:")
            print(have[have['status'] == 'CHECK'].head(15).to_string(index=False))
    other = df[~df['status'].isin(['PASS', 'CHECK'])]
    if len(other):
        print(f"  {len(other)} session(s) without usable HD: "
              f"{other['status'].value_counts().to_dict()}")
    return df


def gate_hd_lengths(data_dic):
    """The fixed HD vector must be as long as the other per-bin arrays.

    The bug's second half: `truncate_all_arrays` cuts every array to the shortest, so a
    2T-length flattened HD silently truncated the session to T. After the fix, HD_raw[:, 0]
    is length T and must match Locs_raw.
    """
    rows = []
    for rd in sorted(data_dic):
        for s in sorted(k for k in data_dic[rd] if isinstance(data_dic[rd][k], dict)):
            sd = data_dic[rd][s]
            hd = sd.get('HD_raw')
            if hd is None:
                continue
            hd = np.asarray(hd)
            locs = np.asarray(sd['Locs_raw'])
            fr = np.asarray(sd['Neuron_raw'])
            rows.append({
                'recday': rd, 'session': s,
                'n_locs': len(locs), 'n_fr_bins': fr.shape[1],
                'hd_shape': hd.shape,
                'n_hd_fixed': hd.shape[0] if hd.ndim == 2 else len(hd),
                'n_hd_flattened': hd.size,
            })
    df = pd.DataFrame(rows)
    if len(df):
        df['fixed_matches_locs'] = df['n_hd_fixed'] == df['n_locs']
        df['old_truncated_to'] = np.minimum(df['n_hd_flattened'], df['n_locs'])
        # How much of the real HD TIMELINE the old code saw. flatten() interleaves the two
        # columns, so element 2k of the flattened vector is back2mid[k] and element 2k+1 is
        # earL2earR[k]. Truncating to T therefore reaches only k < T/2 of each column: half
        # the session, interleaved with a ~90-deg-rotated copy of itself, and offset from
        # every other regressor from sample 1 onward.
        df['old_timeline_covered'] = (df['old_truncated_to'] / 2 / df['n_locs']).round(3)
        print(f"\n[gate 2b] HD length: fixed HD matches Locs_raw in "
              f"{int(df['fixed_matches_locs'].sum())}/{len(df)} sessions")
        print(f"  flattened length == 2T in "
              f"{int((df['n_hd_flattened'] == 2 * df['n_hd_fixed']).sum())}/{len(df)}")
        print(f"  under the OLD flatten(), median fraction of the HD TIMELINE covered: "
              f"{df['old_timeline_covered'].median():.3f}")
    return df


# ---------------------------------------------------------------------------
# Gate 3 - depth ordering
# ---------------------------------------------------------------------------

def gate_depth_ordering(unit_regions, tol_um=50.0):
    """`y_um` (0 = shank tip) must order regions consistently within each mouse.

    Pure geometry: a probe enters through one structure and ends in another, so the median
    depth of each region should be monotone along the trajectory and should not disagree
    between recdays of the same mouse (same physical probe).
    """
    rows = []
    for rd, df in unit_regions.items():
        mouse = rd.split('_')[0]
        for grp, sub in df.groupby('group'):
            rows.append({'mouse': mouse, 'recday': rd, 'group': grp, 'n': len(sub),
                         'median_y_um': float(np.median(sub['y_um'])),
                         'q25_y_um': float(np.percentile(sub['y_um'], 25)),
                         'q75_y_um': float(np.percentile(sub['y_um'], 75))})
    per_recday = pd.DataFrame(rows)

    # per-mouse ordering, using only groups with >=10 units in that recday
    print("\n[gate 3] depth ordering (median y_um per region; 0 = shank tip)")
    orderings = {}
    for mouse, sub in per_recday.groupby('mouse'):
        piv = (sub[sub['n'] >= 10]
               .pivot_table(index='recday', columns='group', values='median_y_um'))
        piv = piv[[c for c in GROUP_ORDER if c in piv.columns]]
        # Judge consistency only on groups this mouse has in EVERY recday. A group that is
        # present in one recday and absent in another (n < 10) produces a NaN rank, which
        # is a gap in sampling, not a disagreement about anatomy -- scoring it as an
        # inconsistency would fail the gate on ly05/ly06/ly07 for the wrong reason.
        full = piv.dropna(axis=1, how='any')
        # A pair of groups "reverses" only if it is ordered one way in one recday and the
        # other way in another. Ties and sub-tolerance separations are not reversals: the
        # local deformation field is 8-13 um/voxel and structure boundaries carry more
        # uncertainty than that, so two medians within `tol_um` are not separable and a
        # swap between them says nothing about the fit. Scoring those as failures would
        # fail this gate on a tie at exactly 300 um (ah10) and on a 15-75 um ENTl-deep vs
        # ENTm wobble (ly07), neither of which is evidence of a bad registration.
        cols = list(full.columns)
        reversals, sub_tol, ties = [], [], []
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                d = (full[cols[a]] - full[cols[b]]).to_numpy()
                rec = {'pair': (cols[a], cols[b]),
                       'min_abs_sep_um': float(np.abs(d).min()),
                       'max_abs_sep_um': float(np.abs(d).max())}
                if (d > 0).any() and (d < 0).any():
                    (reversals if np.abs(d).max() > tol_um else sub_tol).append(rec)
                elif (d == 0).any():
                    ties.append(rec)
        consistent = len(reversals) == 0
        orderings[mouse] = {
            'table': piv,
            'groups_in_all_recdays': cols,
            'mean_order': list(piv.mean(axis=0).sort_values().index),
            'consistent_across_recdays': consistent,
            'reversals_beyond_tol': reversals,
            'reversals_within_tol': sub_tol,
            'ties': ties,
            'tol_um': tol_um,
        }
        verdict = 'PASS' if consistent else f'REVERSAL > {tol_um:.0f}um'
        print(f"\n  {mouse}: shallow->deep by median y_um = "
              f"{' < '.join(orderings[mouse]['mean_order'])}"
              f"   [{verdict}, judged on {len(cols)} group(s) present in all recdays]")
        print(piv.round(0).to_string())
        for f in reversals:
            print(f"    REVERSAL: {f['pair'][0]} vs {f['pair'][1]} — separation "
                  f"{f['min_abs_sep_um']:.0f}-{f['max_abs_sep_um']:.0f} um (> {tol_um:.0f})")
        for f in sub_tol:
            print(f"    order swaps within tolerance: {f['pair'][0]} vs {f['pair'][1]} — "
                  f"separation {f['min_abs_sep_um']:.0f}-{f['max_abs_sep_um']:.0f} um "
                  f"(<= {tol_um:.0f}, not separable)")
        for f in ties:
            print(f"    tie: {f['pair'][0]} vs {f['pair'][1]} — equal medians in >=1 recday")
    n_fail = sum(1 for v in orderings.values() if not v['consistent_across_recdays'])
    print(f"\n  [gate 3] {len(orderings) - n_fail}/{len(orderings)} mice PASS "
          f"(no region-order reversal beyond {tol_um:.0f} um)")
    return per_recday, orderings


# ---------------------------------------------------------------------------
# Gate 4 - unit quality per region
# ---------------------------------------------------------------------------

def gate_quality_by_region(unit_regions, meta_path=CLUSTER_META):
    """Publish `mean_rate_hz`, `sd`, `r2_full` per region BEFORE any tuning result.

    Region is derived from the unit's max-amplitude channel, and depth along a shank drives
    spike amplitude, isolation and yield. So a "regional difference" in tuning can be a
    recording-quality gradient wearing an anatomical label. This table is how a reader
    sizes that risk, and it is why it comes first.
    """
    if not os.path.exists(meta_path):
        print(f"\n[gate 4] SKIPPED — {meta_path} not found")
        return None
    meta = _load_pickle(meta_path)
    meta = pd.DataFrame(meta)
    frames = []
    for rd, reg in unit_regions.items():
        sub = meta[meta['recday'] == rd]
        if not len(sub):
            continue
        # meta's `neuron` is the Neuron_raw row index, which is unit_regions' row order
        idx = sub['neuron'].to_numpy(dtype=int)
        keep = idx < len(reg)
        sub = sub.loc[keep]
        idx = idx[keep]
        j = sub.copy()
        j['group'] = reg['group'].to_numpy()[idx]
        j['y_um'] = reg['y_um'].to_numpy()[idx]
        frames.append(j)
    if not frames:
        print("\n[gate 4] SKIPPED — no overlap between clustering meta and unit_regions")
        return None
    joined = pd.concat(frames, ignore_index=True)
    cols = [c for c in ('mean_rate_hz', 'sd', 'r2_full') if c in joined.columns]
    summary = (joined.groupby('group')[cols]
               .agg(['count', 'median', 'mean', 'std'])
               .reindex([g for g in GROUP_ORDER if g in set(joined['group'])]))
    print("\n[gate 4] unit quality by region (from clustering__meta.pkl)")
    print(summary.round(3).to_string())
    print("\n  Read this before any tuning result: if groups differ in rate, every")
    print("  rate-dependent statistic downstream inherits that difference.")
    return joined, summary


# ---------------------------------------------------------------------------

def main(strict=True):
    import glm_analysis_v2 as glm

    print("=" * 78)
    print("W0 GATES — anatomy split")
    print("=" * 78)

    unit_regions = _load_pickle(UNIT_REGIONS)
    print(f"unit_regions: {len(unit_regions)} recdays")

    print("\nloading data_dic (3.8 GB, this takes a few minutes) ...")
    data_dic = glm.load_data_dic(validate=True, apply_exclusions=True, verbose=True)

    out = {}
    out['length'] = gate_length(data_dic, unit_regions, strict=strict)
    out['hd_columns'] = gate_hd_columns(data_dic)
    out['hd_lengths'] = gate_hd_lengths(data_dic)
    out['depth_per_recday'], out['depth_orderings'] = gate_depth_ordering(unit_regions)
    q = gate_quality_by_region(unit_regions)
    if q is not None:
        out['quality_joined'], out['quality_summary'] = q

    with open(OUT_PATH, 'wb') as fh:
        pickle.dump(out, fh)
    print(f"\nsaved gate results -> {OUT_PATH}")
    return out


if __name__ == '__main__':
    main(strict='--no-strict' not in sys.argv)
