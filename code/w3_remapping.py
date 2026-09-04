"""W3 -- generalising state cells and coherent remapping, split by brain region.

Two questions that are one analysis seen from both ends (`docs/handoff/W3_remapping.md`):

  GENERALISING STATE CELLS -- does a neuron keep the same task-state preference ACROSS
  tasks? `A` is a different physical port in every one of a recday's 6 tasks, so state
  labels are purely ordinal and a within-task preference is just "fires on one leg" with an
  arbitrary letter attached. What makes the letters mean anything is constancy across ports.

  COHERENT REMAPPING -- even where cells do remap, is the PAIRWISE angle between two cells
  preserved, i.e. does the population rotate as a rigid body?

Both reduce to a remapping angle on the 360-bin task-space rate map (1 bin = 1 deg, one
state = 90 deg). Generalising = angle near 0. Coherent = pairwise angle constant.

The new thing here is anatomy: the recorded "LEC" bank spans ENTl-sup/deep, ENTm, SUB/ProS
and CA1/HPF, and no remapping result has ever read it. The headline that only this
workstream can produce is figure 5 -- whether coherently-remapping pairs are same-region
(anatomically-defined independent modules) or cross-region.

Three things about the measurement that the reader needs to hold
---------------------------------------------------------------
**1. The estimator is a 4-way vote, not a continuous angle.** The task-phase spectrum is
dominated by harmonic 4 (`TASKPHASE_PERIODICITY.md`: power fraction .098 against .057 at
h=1, the largest harmonic in 84% of sessions). A curve dominated by h=4 has a
cross-correlation with four near-equal peaks at 0/90/180/270, so the argmax is a
state-identity vote with a noise tie-break. That is why chance is 1/4, and it is why the
null here is a CELL-IDENTITY SHUFFLE (neuron i in task X against neuron j in task Y) rather
than a uniform circle -- only the shuffle preserves the h=4 structure it has to.

**2. The sign of an angle is not interpretable, and that is a bug detector.** Task ordering
within a recday is arbitrary, so P(+90) must equal P(-90) and every pooled polar histogram
is symmetric by construction. `assert_symmetric` checks it; asymmetry means the pipeline is
broken, not that the brain prefers one direction.

**3. Rate is confounded with region, and it acts on this measure through reliability.** A
low-rate region gives noisier curves, hence a noisier argmax, hence a FLATTER polar
histogram -- which reads as "more remapping" with no biology behind it. Every per-region
panel therefore carries that region's own split-half noise floor, and the primary contrast
is repeated under `anatomy_split.rate_match`.

Run order
---------
    python w3_remapping.py --synthetic     # blocking controls, exits non-zero on failure
    python w3_remapping.py --run           # builds data/processed_data/w3_remapping.pkl

then `code/LEC_anatomy_state_remapping.ipynb` for the figures.
"""

from __future__ import annotations

import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import anatomy_split as asplit                                    # noqa: E402
import remapping_rotation_analysis as rr                          # noqa: E402

CACHE_PATH = os.path.join(REPO, 'data', 'processed_data', 'w3_remapping.pkl')
FIG_DIR = os.path.join(REPO, 'data', 'figures', 'anatomy_split')
W0_GATES = os.path.join(REPO, 'data', 'processed_data', 'w0_gates.pkl')

#: Half a task state. A cell that has moved to a neighbouring state cannot pass.
GEN_THRESHOLD_DEG = 45.0

#: 1/num_task_states -- the chance level the h=4 quantisation imposes (see docstring).
CHANCE = 0.25

METHODS = ('xcorr', 'peak')


def config(**kw):
    """`RemapConfig` at W3 settings: 1 deg resolution, every non-reference task used.

    `rotation_step_single` is irrelevant to this module (angles come from the FFT
    implementation in `anatomy_split`, which is exact at 1 bin) but is set to 1 so that
    anything reaching for `rr.best_rotation` agrees. `max_comparisons=None` uses all 5
    non-reference tasks rather than El-Gaby's 2; the 2-comparison parity number is recovered
    afterwards by slicing, not by re-running.
    """
    cfg = rr.RemapConfig(rotation_step_single=1, max_comparisons=None)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Session selection
# ---------------------------------------------------------------------------

def session_plan(recday_data, min_trials=5):
    """Split a recday's 8 sessions into 6 unique tasks + the exact repeats.

    Returns `{'unique': [...], 'repeats': [(first, repeat), ...], 'tasks': {session: task}}`.
    The repeats are the X-vs-X' control: the same physical task run twice, so the remapping
    angle between them is the CEILING for this estimator -- whatever it returns there is
    measurement noise, not remapping. Every recday has two such pairs (sessions [0,3] and
    [4,7] in the canonical ordering, but they are found by comparing `Task`, not assumed).
    """
    unique, repeats, tasks, seen = [], [], {}, []
    for s in recday_data:
        if s == 'valid_sessions' or not isinstance(recday_data[s], dict):
            continue
        sd = recday_data[s]
        if sd.get('num_trials', 0) < min_trials:
            continue
        task = sd.get('Task')
        if task is None or 'defaultdict' in str(type(task)) or 'defaultdict' in str(task):
            continue
        task = np.asarray(task)
        if task.ndim != 1 or task.size == 0:
            continue
        tasks[s] = task
        match = next((u for u in seen if np.array_equal(tasks[u], task)), None)
        if match is None:
            unique.append(s)
            seen.append(s)
        else:
            repeats.append((match, s))
    return {'unique': unique, 'repeats': repeats, 'tasks': tasks}


# ---------------------------------------------------------------------------
# Tuning curves + the gate, in one pass
# ---------------------------------------------------------------------------

def build_session(session_data, cfg):
    """`(tuning (n, 360), gate (n,) bool, pref_state (n,))` -- one pass over the trials.

    `rr.build_session_tuning` calls `raw_to_norm` twice per neuron: once with
    `return_mean=False` inside `identify_state_tuned_neurons`, and again with
    `return_mean=True` for the curve. The mean curve is exactly `nanmean` of the per-trial
    array, so one pass gives both and halves an 8-minute cohort pass. `synthetic_controls`
    asserts this function reproduces `rr.build_session_tuning` exactly rather than trusting
    that argument.

    The gate is the El-Gaby peak-z t-test at parity (`identify_state_tuned_neurons`): peak
    firing per state per trial -> z across states within trial -> mean across trials ->
    preferred state = argmax -> one-sample t-test of that state's per-trial z against 0.
    """
    from scipy import stats

    for k in ('Neuron_raw', 'Trial_times'):
        if k not in session_data or session_data[k] is None:
            return None
    raw = np.asarray(session_data['Neuron_raw'], dtype=float)
    tt = np.asarray(session_data['Trial_times'], dtype=int)
    if raw.ndim != 2 or tt.ndim != 2 or tt.shape[0] < 2:
        return None

    n, nstates, nbps = raw.shape[0], cfg.num_task_states, cfg.num_bins_per_state
    tuning = np.full((n, cfg.total_bins), np.nan)
    gate = np.zeros(n, dtype=bool)
    pref = np.full(n, np.nan)

    for ni in range(n):
        per_trial = rr.raw_to_norm(raw[ni], tt, cfg, return_mean=False)
        if per_trial is None:
            continue
        curve = np.nanmean(per_trial, axis=0)
        if not np.all(np.isnan(curve)):
            tuning[ni] = rr.smooth_circular(np.nan_to_num(curve, nan=0.0),
                                            sigma=cfg.smoothing_sigma)
        if per_trial.shape[0] < 3:
            continue
        peak = np.full((per_trial.shape[0], nstates), np.nan)
        for s in range(nstates):
            peak[:, s] = np.nanmax(per_trial[:, s * nbps:(s + 1) * nbps], axis=1)
        rm = np.nanmean(peak, axis=1, keepdims=True)
        rs = np.nanstd(peak, axis=1, keepdims=True)
        rs[rs == 0] = np.nan
        z = (peak - rm) / rs
        if np.all(np.isnan(z)):
            continue
        p_state = int(np.nanargmax(np.nanmean(z, axis=0)))
        pref[ni] = p_state
        zp = z[:, p_state][~np.isnan(z[:, p_state])]
        if len(zp) >= 3 and stats.ttest_1samp(zp, 0)[1] < cfg.state_tuning_p_threshold:
            gate[ni] = True
    return tuning, gate, pref


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

def _wrap(deg):
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def dist_to_multiple_of_90(deg):
    """Distance in degrees to the nearest multiple of 90, in [0, 45].

    The h=4-dominated spectrum quantises the rotation estimate onto multiples of 90 -- one
    whole task state. So an angle splits into a STATE-IDENTITY part (which multiple of 90 it
    landed on) and a WITHIN-LEG PROGRESS-PHASE part (what is left over). This is the second
    part: near 0 means the cell moved by whole states, larger means its phase within the leg
    shifted too.
    """
    return np.abs((np.asarray(deg, dtype=float) + 45.0) % 90.0 - 45.0)


def angles_between(curves_a, curves_b, method='xcorr'):
    """Row-wise remapping angle: `out[i]` rotates `curves_a[i]` onto `curves_b[i]`.

    Signed degrees in (-180, 180], NaN where either curve is flat or all-NaN. Thin loop over
    `anatomy_split.remapping_angle` -- the one tested implementation, and cheap enough here
    (a few hundred 360-point FFTs per recday) that batching buys nothing.
    """
    a, b = np.asarray(curves_a, dtype=float), np.asarray(curves_b, dtype=float)
    return np.array([asplit.remapping_angle(a[i], b[i], method=method)
                     for i in range(a.shape[0])])


def _derangement(n, rng):
    """A permutation with no fixed point (n >= 2), for the cell-identity shuffle."""
    if n < 2:
        return np.arange(n)
    for _ in range(100):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    p = rng.permutation(n)                       # fall back: rotate away the fixed points
    fixed = np.flatnonzero(p == np.arange(n))
    for i in fixed:
        j = (i + 1) % n
        p[i], p[j] = p[j], p[i]
    return p


# ---------------------------------------------------------------------------
# Per-recday run
# ---------------------------------------------------------------------------

def run_recday(recday_data, cfg, seed=0, reference=None, verbose=False):
    """Everything W3 needs for one recday, keyed to `Neuron_raw` rows.

    The join keys are `included_idx` (rows of `Neuron_raw`, hence rows of
    `unit_regions[recday]`, that passed the gate) and `pair_idx` (pairs of POSITIONS into
    `included_idx`, in `np.triu_indices` order). Everything else is positional against those
    two, and the join asserts lengths -- see `anatomy_split.join_regions`.
    """
    plan = session_plan(recday_data)
    unique = plan['unique']
    if len(unique) < 2:
        return None

    built = {}
    n_neurons = None
    for s in unique + [r for _, r in plan['repeats']]:
        b = build_session(recday_data[s], cfg)
        if b is None:
            continue
        if n_neurons is None:
            n_neurons = b[0].shape[0]
        elif b[0].shape[0] != n_neurons:
            if verbose:
                print(f"    neuron-count mismatch at session {s}, skipping recday")
            return None
        built[s] = b
    unique = [s for s in unique if s in built]
    if len(unique) < 2:
        return None

    tuning = {s: built[s][0] for s in built}
    gate = {s: built[s][1] for s in unique}
    pref = {s: built[s][2] for s in unique}

    # Included = state-tuned in >= half of the unique tasks (El-Gaby parity).
    frac = np.mean(np.vstack([gate[s] for s in unique]), axis=0)
    included = np.flatnonzero(frac >= cfg.min_tasks_tuned_frac)
    gate_frac_per_neuron = frac

    ref = unique[0] if reference is None else reference
    others = [s for s in unique if s != ref]
    out = {
        'n_neurons': n_neurons,
        'unique_sessions': unique,
        'reference_session': ref,
        'comparison_sessions': others,
        'repeats': plan['repeats'],
        'tasks': {s: plan['tasks'][s] for s in built},
        'gate_frac': gate_frac_per_neuron,
        'gate_any': np.any(np.vstack([gate[s] for s in unique]), axis=0),
        'pref_state': np.vstack([pref[s] for s in unique]),          # (n_tasks, n_neurons)
        'included_idx': included,
        'n_included': len(included),
        'tuning': tuning,
    }
    if len(included) == 0:
        out['empty'] = True
        return out

    rng = np.random.default_rng(seed)
    ref_curves = tuning[ref][included]

    # --- single-neuron rotations vs the reference task, and the two reference bands ---
    for m in METHODS:
        out[f'angles_{m}'] = np.column_stack(
            [angles_between(ref_curves, tuning[s][included], method=m) for s in others]
        ) if others else np.zeros((len(included), 0))

        # CEILING: X vs X'. The same physical task twice, so anything but 0 is noise.
        xs = [r for f, r in plan['repeats'] if r in tuning]
        out[f'angles_X_{m}'] = np.column_stack(
            [angles_between(tuning[f][included], tuning[r][included], method=m)
             for f, r in plan['repeats'] if r in tuning and f in tuning]
        ) if xs else np.zeros((len(included), 0))

        # FLOOR: cell-identity shuffle. Neuron i's reference curve against neuron pi(i)'s
        # curve in the comparison task -- same curves, same h=4 structure, no correspondence.
        out[f'angles_shuffle_{m}'] = np.column_stack(
            [angles_between(ref_curves, tuning[s][included][_derangement(len(included), rng)],
                            method=m) for s in others]
        ) if others else np.zeros((len(included), 0))

    # --- pairwise: the within-task angle between two neurons, in EVERY task ---
    # This feeds the DIRECT metric. "Direct" does not mean reference-free -- it still measures
    # change against the reference task. What it drops is the PER-NEURON rotation anchoring:
    # it never estimates any single neuron's rotation, only the geometry of the pair inside
    # each task. See `rel_pairs_direct` below.
    out['pair_idx'] = np.column_stack(np.triu_indices(len(included), k=1))
    out['pair_angles'] = {s: asplit.pairwise_angles(tuning[s][included], method='xcorr')
                          for s in unique}
    iu = np.triu_indices(len(included), k=1)
    ref_pair = out['pair_angles'][ref][iu]
    out['pair_tuning_dist'] = np.abs(ref_pair)          # |initial angle| in the ref task

    # relative pair rotation, reference-anchored: Delta r = r_j - r_i
    a = out['angles_xcorr']
    out['rel_pairs'] = _wrap(a[iu[1], :] - a[iu[0], :])              # (n_pairs, n_comp)
    aX = out['angles_X_xcorr']
    out['rel_pairs_X'] = (_wrap(aX[iu[1], :] - aX[iu[0], :]) if aX.shape[1]
                          else np.zeros((len(iu[0]), 0)))
    aS = out['angles_shuffle_xcorr']
    out['rel_pairs_shuffle'] = (_wrap(aS[iu[1], :] - aS[iu[0], :]) if aS.shape[1]
                                else np.zeros((len(iu[0]), 0)))
    # DIRECT: Delta_ij(t) - Delta_ij(ref), where Delta_ij(t) is the shift aligning neuron i's
    # curve onto neuron j's WITHIN task t. Algebraically equal to r_j - r_i whenever both
    # neurons' own rotations are well defined -- if curve_i(t) = roll(curve_i(ref), r_i) then
    # Delta_ij(t) = Delta_ij(ref) + r_j - r_i.
    #
    # They differ in HOW THEY FAIL, which is the reason both are computed. The reference-
    # anchored version routes through each neuron's individual rotation, so a single neuron
    # with a broad or multi-peaked curve -- whose argmax is a coin flip -- corrupts every one
    # of the n-1 pairs it belongs to. The direct version never estimates a single neuron's
    # rotation, so two individually-ambiguous cells can still have a well-determined angle
    # between them. Where the two disagree, suspect the per-neuron rotation estimates.
    out['rel_pairs_direct'] = np.column_stack(
        [_wrap(out['pair_angles'][s][iu] - ref_pair) for s in others]
    ) if others else np.zeros((len(iu[0]), 0))
    return out


def run_all(data_dic, cfg=None, seed=0, verbose=True, reference=None):
    """`run_recday` over every recday, with progress."""
    cfg = cfg or config()
    results = {}
    for i, rd in enumerate(sorted(data_dic)):
        if verbose:
            print(f"[{i + 1}/{len(data_dic)}] {rd} ...", flush=True)
        res = run_recday(data_dic[rd], cfg, seed=seed, reference=reference, verbose=verbose)
        if res is None:
            if verbose:
                print("    unusable, skipped")
            continue
        results[rd] = res
        if verbose:
            print(f"    {res['n_neurons']} neurons, {len(res['unique_sessions'])} tasks, "
                  f"{res['n_included']} included "
                  f"({res['n_included'] / max(res['n_neurons'], 1):.1%} pass the gate)",
                  flush=True)
    return results


def recompute_with_reference(res, ref_session, seed=0):
    """Re-derive the angles for one recday against a DIFFERENT reference task.

    Reference anchoring has one specific failure mode: every neuron's rotation and therefore
    every pair's relative rotation is measured against one task, so a single unrepresentative
    reference corrupts the whole recday at once. The check is to rotate which task plays the
    reference and confirm the headline does not move.

    This is cheap because the cache holds the tuning curves and the per-task pairwise angle
    matrices -- only the comparisons against the anchor have to be redone, not the binning,
    warping, smoothing or gating, which is where all the time goes.
    """
    inc = res.get('included_idx', np.array([], dtype=int))
    if not len(inc) or ref_session not in res['tuning']:
        return None
    others = [s for s in res['unique_sessions'] if s != ref_session]
    if not others:
        return None
    rng = np.random.default_rng(seed)
    tuning, ref_curves = res['tuning'], res['tuning'][ref_session][inc]
    out = dict(res)
    out['reference_session'] = ref_session
    out['comparison_sessions'] = others
    for m in METHODS:
        out[f'angles_{m}'] = np.column_stack(
            [angles_between(ref_curves, tuning[s][inc], method=m) for s in others])
        out[f'angles_shuffle_{m}'] = np.column_stack(
            [angles_between(ref_curves, tuning[s][inc][_derangement(len(inc), rng)],
                            method=m) for s in others])
    iu = np.triu_indices(len(inc), k=1)
    a, aS = out['angles_xcorr'], out['angles_shuffle_xcorr']
    out['rel_pairs'] = _wrap(a[iu[1], :] - a[iu[0], :])
    out['rel_pairs_shuffle'] = _wrap(aS[iu[1], :] - aS[iu[0], :])
    ref_pair = res['pair_angles'][ref_session][iu]
    out['rel_pairs_direct'] = np.column_stack(
        [_wrap(res['pair_angles'][s][iu] - ref_pair) for s in others])
    out['pair_tuning_dist'] = np.abs(ref_pair)
    return out


def reference_robustness(results, unit_regions, max_refs=6):
    """The headline rates recomputed with every task in turn playing the reference."""
    rows = []
    n_refs = max(len(r['unique_sessions']) for r in results.values())
    for k in range(min(n_refs, max_refs)):
        recomputed = {}
        for rd, res in results.items():
            uniq = res['unique_sessions']
            if k >= len(uniq) or not res.get('n_included'):
                continue
            alt = recompute_with_reference(res, uniq[k])
            if alt is not None:
                recomputed[rd] = alt
        if not recomputed:
            continue
        import w3_figures as wf
        n = neuron_frame(recomputed, unit_regions)
        p = pair_frame(recomputed, unit_regions)
        inc = n[n['included']]
        rows.append({
            'reference_rank': k, 'n_recdays': len(recomputed),
            'single_real': wf.per_comparison_rate(inc, 'angles_xcorr'),
            'single_shuffle': wf.per_comparison_rate(inc, 'angles_shuffle_xcorr'),
            'pair_real': wf.per_comparison_rate(p, 'rel'),
            'pair_shuffle': wf.per_comparison_rate(p, 'relS'),
            'pair_direct': wf.per_comparison_rate(p, 'relD'),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df['single_gap'] = df['single_real'] - df['single_shuffle']
        df['pair_gap'] = df['pair_real'] - df['pair_shuffle']
    return df


# ---------------------------------------------------------------------------
# Joining to anatomy
# ---------------------------------------------------------------------------

def neuron_frame(results, unit_regions, rates=None):
    """One row per unit (all units, not just gated), joined to region.

    Built full-length and NaN-filled for the units that failed the gate, so the positional
    length assert in `anatomy_split.join_regions` still does its job -- that assert is what
    catches a cached result computed against a different unit count, which would otherwise
    misalign every neuron after the first mismatch.
    """
    per_rd = {}
    for rd, res in results.items():
        n = res['n_neurons']
        inc = res['included_idx']
        n_comp = res.get('angles_xcorr', np.zeros((0, 0))).shape[1]
        cols = {'gate_frac': res['gate_frac'], 'included': np.isin(np.arange(n), inc)}
        for m in METHODS:
            for tag in ('angles', 'angles_X', 'angles_shuffle'):
                src = res.get(f'{tag}_{m}')
                if src is None or not len(inc):
                    continue
                for c in range(src.shape[1]):
                    full = np.full(n, np.nan)
                    full[inc] = src[:, c]
                    cols[f'{tag}_{m}_{c}'] = full
        # preferred state in the reference task, for the "which letter" analysis
        ref_pos = res['unique_sessions'].index(res['reference_session'])
        cols['pref_state_ref'] = res['pref_state'][ref_pos]
        cols['n_comparisons'] = np.full(n, n_comp)

        # Generalising is decided HERE, per recday, not after the concat. Recdays differ in
        # task count (5 or 6 usable tasks), so after concatenation a 5-task recday has a
        # NaN in the last angle column -- and `all(isfinite)` would then silently score
        # every one of its neurons as non-generalising.
        for m in METHODS:
            A = res.get(f'angles_{m}')
            if A is None or not len(inc):
                continue
            full = np.zeros(n, dtype=bool)
            full[inc] = asplit.is_generalising(A, GEN_THRESHOLD_DEG)
            cols[f'generalising_{m}'] = full
            ma = np.full(n, np.nan)
            ma[inc] = np.nanmean(np.abs(A), axis=1)
            cols[f'mean_abs_angle_{m}'] = ma
            # distance to the nearest multiple of 90: state-identity remapping lands ON a
            # multiple, so what is left over is the within-leg progress-phase shift
            rz = np.full(n, np.nan)
            rz[inc] = np.nanmean(dist_to_multiple_of_90(A), axis=1)
            cols[f'mean_abs_resid90_{m}'] = rz
        per_rd[rd] = cols
        per_rd[rd]['_n'] = n
        per_rd[rd]['_n_comp'] = n_comp

    # join once on a single column so the length assert runs, then widen by merge
    base = asplit.join_regions({rd: c['gate_frac'] for rd, c in per_rd.items()},
                               unit_regions, value_name='gate_frac')
    extras = []
    for rd, c in per_rd.items():
        df = pd.DataFrame({k: v for k, v in c.items()
                           if not k.startswith('_') and k != 'gate_frac'})
        df.insert(0, 'recday', rd)
        df.insert(1, 'neuron', np.arange(c['_n']))
        extras.append(df)
    out = base.merge(pd.concat(extras, ignore_index=True), on=['recday', 'neuron'],
                     how='left', validate='one_to_one')

    if rates is not None:
        out = out.merge(rates[['recday', 'neuron', 'mean_rate_hz']],
                        on=['recday', 'neuron'], how='left', validate='one_to_one')

    out['included'] = out['included'].fillna(False).astype(bool)
    for m in METHODS:
        col = f'generalising_{m}'
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool) & out['included']
    # agreement between the two metrics, over the cells both are defined for
    both = out['included'] & out[[f'mean_abs_angle_{m}' for m in METHODS]].notna().all(axis=1)
    out['metrics_agree'] = np.where(
        both, out['generalising_xcorr'] == out['generalising_peak'], np.nan)
    return out


def pair_frame(results, unit_regions, min_included=2):
    """One row per simultaneously-recorded pair of included neurons, with both regions.

    `pair_type` is `within-<G>` or `<A>x<B>` with the two region names sorted, so that
    within- and cross-region pairs are directly comparable. Only pairs from the SAME recday
    exist -- a cross-region pair is meaningless unless the two units were recorded at once.
    """
    frames = []
    for rd, res in results.items():
        inc = res.get('included_idx', np.array([], dtype=int))
        if len(inc) < min_included or 'pair_idx' not in res:
            continue
        reg = unit_regions[rd]
        if len(reg) != res['n_neurons']:
            raise AssertionError(
                f"pair_frame: {rd} has {res['n_neurons']} neurons but unit_regions has "
                f"{len(reg)} — the join is positional and would misalign every pair.")
        pi = res['pair_idx']
        ni, nj = inc[pi[:, 0]], inc[pi[:, 1]]
        gi = reg['group'].to_numpy()[ni]
        gj = reg['group'].to_numpy()[nj]
        same = gi == gj
        ptype = np.where(same, np.char.add('within-', gi.astype(str)),
                         ['x'.join(sorted((a, b))) for a, b in zip(gi, gj)])
        df = pd.DataFrame({
            'recday': rd, 'mouse': rd.split('_')[0],
            'neuron_i': ni, 'neuron_j': nj, 'group_i': gi, 'group_j': gj,
            'same_region': same, 'pair_type': ptype,
            'tuning_dist': res['pair_tuning_dist'],
        })
        for tag, key in (('rel', 'rel_pairs'), ('relX', 'rel_pairs_X'),
                         ('relS', 'rel_pairs_shuffle'), ('relD', 'rel_pairs_direct')):
            M = res.get(key)
            if M is None or not M.shape[1]:
                continue
            for c in range(M.shape[1]):
                df[f'{tag}_{c}'] = M[:, c]
            # Coherence is decided HERE, per recday. Recdays differ in task count, so after
            # the concat a 5-task recday carries a NaN in the last comparison column and
            # `all(isfinite)` would score every one of its pairs as incoherent.
            df[f'coherent_{tag}'] = _coherent(M)
            if tag == 'rel':
                df['n_comparisons'] = M.shape[1]
                if M.shape[1] >= 2:
                    # El-Gaby parity: the first two comparisons only (chance 1/16)
                    df['coherent_rel_parity'] = _coherent(M[:, :2])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for c in [c for c in out.columns if c.startswith('coherent_')]:
        out[c] = out[c].fillna(False).astype(bool)
    return out


def _coherent(rel, threshold_deg=GEN_THRESHOLD_DEG):
    """A pair is coherent if its relative rotation stays within threshold in EVERY comparison."""
    R = np.asarray(rel, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    ok = np.abs(R) < threshold_deg
    return np.where(np.all(np.isfinite(R), axis=1), np.all(ok, axis=1), False)


# ---------------------------------------------------------------------------
# The symmetry assert (see module docstring, point 2)
# ---------------------------------------------------------------------------

def assert_symmetric(angles, label='', tol=0.06, min_n=200, raise_on_fail=False):
    """Task ordering is arbitrary, so a pooled angle histogram must be symmetric about 0.

    Compares the mass at +theta against -theta outside the near-zero window (where a genuine
    generalising peak sits and sign is meaningless anyway). Returns the signed imbalance
    (positive = excess clockwise). This is a pipeline check, not a result: a real asymmetry
    would mean the reference task is being compared in a consistent direction somewhere it
    should not be.
    """
    a = np.asarray(angles, dtype=float).ravel()
    a = a[np.isfinite(a) & (np.abs(a) >= GEN_THRESHOLD_DEG) & (np.abs(a) < 180)]
    if len(a) < min_n:
        return np.nan
    pos, neg = np.sum(a > 0), np.sum(a < 0)
    imbalance = (pos - neg) / max(pos + neg, 1)
    msg = (f"assert_symmetric{' [' + label + ']' if label else ''}: "
           f"{pos} positive vs {neg} negative angles, imbalance {imbalance:+.3f} "
           f"(tol {tol}) — task order is arbitrary, so this should be ~0.")
    if abs(imbalance) > tol:
        if raise_on_fail:
            raise AssertionError(msg)
        print('WARNING: ' + msg)
    return imbalance


# ---------------------------------------------------------------------------
# Do the coherent modules line up with anatomy?
# ---------------------------------------------------------------------------

def cluster_vs_anatomy(results, unit_regions, cfg=None, n_perm=200, seed=0, min_neurons=10):
    """Cluster neurons by how coherently they rotate, then ask if the clusters are regions.

    This is the strongest form of the anatomy question, and the one place W3 flags as most
    likely to yield a new claim: rather than testing a within-vs-cross contrast we have
    chosen, let the coherence structure define its own modules (agglomerative clustering on
    the incoherence matrix, k by max silhouette) and ask whether those modules recover the
    region labels. Adjusted Rand against a within-recday permutation of the labels.

    A null result here is informative: it says the population rotates as one thing, or as
    modules that cut across the anatomy rather than following it.
    """
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    cfg = cfg or config()
    rng = np.random.default_rng(seed)
    rows = []
    for rd, res in sorted(results.items()):
        a = res.get('angles_xcorr')
        inc = res.get('included_idx', np.array([], dtype=int))
        if a is None or not len(inc) or a.shape[1] < 2:
            continue
        valid = ~np.any(np.isnan(a), axis=1)
        if valid.sum() < min_neurons:
            continue
        labels, k, sil = rr.cluster_and_silhouette(
            rr.incoherence_matrix(a[valid], cfg), cfg)
        if labels is None:
            continue
        groups = unit_regions[rd]['group'].to_numpy()[inc[valid]]
        if len(set(groups)) < 2:
            continue
        ari = adjusted_rand_score(groups, labels)
        ami = adjusted_mutual_info_score(groups, labels)
        null = np.array([adjusted_rand_score(rng.permutation(groups), labels)
                         for _ in range(n_perm)])
        rows.append({'recday': rd, 'mouse': rd.split('_')[0], 'n_neurons': int(valid.sum()),
                     'n_clusters': k, 'silhouette': sil, 'n_regions': len(set(groups)),
                     'ari': ari, 'ami': ami, 'ari_null_mean': float(null.mean()),
                     'ari_null_sd': float(null.std()),
                     'p_perm': float((1 + np.sum(null >= ari)) / (1 + len(null)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CCGP — the population form of the same question
# ---------------------------------------------------------------------------

def ccgp_by_region(data_dic, unit_regions, groups=None, n_neurons=20, n_draws=5,
                   full_n=False, n_shuffles=50, seed=0, verbose=True):
    """State-pair CCGP per region: a decoder trained on some tasks, tested on a held-out one.

    The population form of Stage B. State identity is abstract across tasks only if such a
    decoder generalises, so this is an independent check on the single-neuron result -- it
    can find structure that is distributed across cells and invisible one neuron at a time.

    Two modes, and they answer different questions:

    - **matched n (default)** -- every region subsampled to the same neuron count, so
      regions can be COMPARED. Decoder accuracy scales with n and per-recday region counts
      differ two-fold, so an unmatched comparison measures yield.
    - **`full_n=True`** -- each region at its actual count. This is a WITHIN-region existence
      claim against that region's own null, never a cross-region comparison: full-n
      ENTl-deep at 80 units beating full-n SUB/ProS at 41 is expected from n alone.

    Subsamples are RANDOM, never `arange(n)`. The neuron index is depth-ordered (corr of row
    index with `y_um` = +0.976 to +0.988), so a leading slice is a superficial-biased sample
    -- in ah10 it catches zero of 29 SUB/ProS and zero of 21 CA1/HPF units.
    """
    import ccgp_state_pairs as ccgp

    groups = groups or asplit.ANALYSIS_GROUPS
    cfg = ccgp.CCGPConfig(n_shuffles=n_shuffles, random_state=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for rd in sorted(data_dic):
        if rd not in unit_regions:
            continue
        reg = unit_regions[rd]['group'].to_numpy()
        for g in groups:
            idx = np.flatnonzero(reg == g)
            if len(idx) < max(cfg.min_neurons, n_neurons if not full_n else 0):
                continue
            draws = ([idx] if full_n
                     else [rng.choice(idx, n_neurons, replace=False) for _ in range(n_draws)])
            for d, subset in enumerate(draws):
                tasks = ccgp.build_task_state_matrices(data_dic, rd, cfg,
                                                       neuron_subset=np.sort(subset))
                if not tasks:
                    continue
                # the join key the CCGP change exists for: every surviving column must
                # actually be a unit of this region
                assert set(reg[tasks[0]['neuron_idx']]) == {g}, \
                    f'{rd}/{g}: CCGP columns are not all from this region'
                out = ccgp.run_ccgp_recday(data_dic, rd, cfg, tasks=tasks,
                                           with_transfer=False)
                df = out['ccgp']
                if not len(df):
                    continue
                rows.append({'recday': rd, 'mouse': rd.split('_')[0], 'group': g,
                             'draw': d, 'n_units': len(tasks[0]['neuron_idx']),
                             'mode': 'full_n' if full_n else f'matched_{n_neurons}',
                             'acc': float(df['acc'].mean()),
                             'null_mean': float(df['null_mean'].mean()),
                             'ceiling': float(df['ceiling'].mean()),
                             'n_test_tasks': int(df['test_session'].nunique())})
            if verbose and rows and rows[-1]['recday'] == rd:
                r = rows[-1]
                print(f"  {rd} {g:>10}: acc {r['acc']:.3f} null {r['null_mean']:.3f} "
                      f"ceiling {r['ceiling']:.3f} (n={r['n_units']})", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def save_cache(results, path=CACHE_PATH, meta=None):
    payload = {'results': results, 'meta': meta or {}}
    with open(path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=4)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB, {len(results)} recdays)")


def load_cache(path=CACHE_PATH):
    with open(path, 'rb') as fh:
        payload = pickle.load(fh)
    return payload['results'], payload.get('meta', {})


def load_rates(path=W0_GATES):
    """Per-unit `mean_rate_hz` from the W0 quality table, for `anatomy_split.rate_match`."""
    with open(path, 'rb') as fh:
        g = pickle.load(fh)
    return g['quality_joined'][['recday', 'neuron', 'mean_rate_hz']].copy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--synthetic' in argv:
        import w3_synthetics
        return w3_synthetics.run_all(verbose=True)
    if '--run' in argv or '--extras' in argv:
        import glm_analysis_v2 as glm
        data_dic = glm.load_data_dic(validate=True, apply_exclusions=True, verbose=True)
        unit_regions = asplit.load_unit_regions()
        for rd in data_dic:
            if rd not in unit_regions:
                raise AssertionError(f"{rd} has no region labels")
        cfg = config()

    if '--run' in argv:
        results = run_all(data_dic, cfg, verbose=True)
        save_cache(results, meta={'n_recdays': len(results),
                                  'gen_threshold_deg': GEN_THRESHOLD_DEG,
                                  'min_tasks_tuned_frac': cfg.min_tasks_tuned_frac})

    if '--extras' in argv:
        # Everything else that needs the 3.8 GB data_dic, in the same load. Each stage is
        # saved as it finishes: CCGP takes hours and the split-half floors take minutes, and
        # a single save at the end would hold the fast result hostage to the slow one.
        import w3_figures as wf
        path = os.path.join(REPO, 'data', 'processed_data', 'w3_extras.pkl')
        extras = {}
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                extras = pickle.load(fh)

        def _flush():
            with open(path, 'wb') as fh:
                pickle.dump(extras, fh, protocol=4)
            print(f'  -> saved {path} ({", ".join(sorted(extras))})', flush=True)

        stages = [s for s in ('floors', 'ccgp') if f'--{s}' in argv] or ['floors', 'ccgp']
        results, _ = load_cache()

        if 'floors' in stages:
            print('\n=== split-half reliability floor by region ===', flush=True)
            extras['splithalf_floors'] = wf.splithalf_floor_by_region(
                data_dic, neuron_frame(results, unit_regions))
            print(f"  {len(extras['splithalf_floors'])} neuron-sessions over "
                  f"{extras['splithalf_floors']['group'].nunique()} regions")
            _flush()

        if 'ccgp' in stages:
            # n_shuffles=20 and 3 draws: the role-permutation null is the dominant cost
            # (6 test tasks x 6 pairs x n_shuffles SVM fits per draw) and 20 draws is
            # already enough to place a mean, which is all the regional summary uses.
            print('\n=== CCGP, matched n=20 ===', flush=True)
            extras['ccgp_matched'] = ccgp_by_region(data_dic, unit_regions, n_neurons=20,
                                                    n_draws=3, n_shuffles=20)
            _flush()
            print('\n=== CCGP, full n ===', flush=True)
            extras['ccgp_full'] = ccgp_by_region(data_dic, unit_regions, full_n=True,
                                                 n_shuffles=20)
            _flush()

    if '--run' not in argv and '--extras' not in argv:
        print(__doc__)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
