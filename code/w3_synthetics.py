"""Synthetic controls for W3, run through the REAL pipeline.

Repo practice, and it has earned its place: gating every new analysis on synthetics pushed
through the production code path has caught six plausible-looking design errors so far. The
rule is that a synthetic must enter at the same door the data does -- here that means
generating `Neuron_raw` and `Trial_times` and letting `w3_remapping.run_recday` bin, warp,
smooth, gate and rotate them, rather than handing the analysis a ready-made tuning curve.

    python w3_remapping.py --synthetic          # or: python w3_synthetics.py

| # | control | what a failure would mean |
|---|---|---|
| 1 | known rotations are recovered exactly | the angle estimator is wrong |
| 2 | rigid population ~100% coherent, independent ~1/4 | the coherence metric is wrong |
| 3 | ordinal-state cell generalises, place cell does not | the cascade measures the wrong thing |
| 4 | two rigid modules -> within-region coherent, cross-region chance | the pair->region join is wrong |
| 5 | the cell-identity shuffle preserves h=4 and sits at 1/4 | the null is not the null we claim |
| 6 | the one-pass tuning build equals `rr.build_session_tuning` | the speed-up changed the data |
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import anatomy_split as asplit                       # noqa: E402
import remapping_rotation_analysis as rr             # noqa: E402
import w3_remapping as w3                            # noqa: E402


# ---------------------------------------------------------------------------
# A synthetic recday that enters through the same door as the data
# ---------------------------------------------------------------------------

def make_recday(phases, tasks=None, n_trials=26, leg_bins=(150, 500), amp=8.0,
                base=0.25, width_deg=22.0, seed=0, n_repeat_tasks=1):
    """Build a `data_dic`-shaped recday whose neurons have KNOWN task-space phases.

    Parameters
    ----------
    phases : (n_tasks, n_neurons) preferred phase in degrees on the 360-bin task-space loop,
        NaN for a neuron with no task-space tuning. This is the ground truth: neuron i's
        rotation from task 0 to task t is `phases[t, i] - phases[0, i]`.
    n_repeat_tasks : int
        How many tasks are additionally run a second time, to exercise the X-vs-X' path.

    Activity is Poisson around a von-Mises-ish bump placed at the neuron's phase *in
    normalised task space*, then written back into RAW time. That matters: the bump lands at
    a different number of raw bins into each leg because leg durations differ, which is
    exactly the warping `raw_to_norm` has to undo. Handing the pipeline pre-warped curves
    would skip the step most likely to be wrong.
    """
    rng = np.random.default_rng(seed)
    phases = np.asarray(phases, dtype=float)
    n_tasks, n_neurons = phases.shape
    if tasks is None:
        # Distinct by construction: `session_plan` deduplicates on `Task`, so two tasks that
        # happen to coincide would silently reduce the recday's task count and make any
        # count-dependent control test the wrong thing.
        from itertools import permutations
        pool = [np.array(p) for p in permutations([1, 2, 3, 4])]
        tasks = [pool[i] for i in rng.permutation(len(pool))[:n_tasks]]
        if n_tasks > len(pool):
            raise ValueError(f'only {len(pool)} distinct 4-port tasks exist')

    order = list(range(n_tasks)) + list(range(n_repeat_tasks))    # repeats at the end
    recday, t_cursor = {}, 0
    for sess, ti in enumerate(order):
        legs = rng.integers(leg_bins[0], leg_bins[1], size=n_trials * 4)
        bounds = np.concatenate([[0], np.cumsum(legs)])
        total = int(bounds[-1])
        trial_times = np.column_stack([bounds[np.arange(n_trials) * 4 + k] for k in range(5)])

        # normalised phase of every raw bin
        phase = np.full(total, np.nan)
        for k in range(n_trials * 4):
            a, b = int(bounds[k]), int(bounds[k + 1])
            phase[a:b] = 90.0 * (k % 4) + 90.0 * (np.arange(b - a) / (b - a))

        rate = np.full((n_neurons, total), base)
        for ni in range(n_neurons):
            p = phases[ti, ni]
            if not np.isfinite(p):
                continue
            d = (phase - p + 180.0) % 360.0 - 180.0
            rate[ni] += amp * np.exp(-0.5 * (d / width_deg) ** 2)
        recday[sess] = {
            'Neuron_raw': rng.poisson(rate).astype(np.uint16),
            'Trial_times': trial_times.astype(int),
            'Task': np.asarray(tasks[ti]),
            'num_trials': n_trials,
        }
        t_cursor += total
    return recday


def _fake_unit_regions(recday_name, groups):
    return {recday_name: pd.DataFrame({
        'group': list(groups),
        'mouse': recday_name.split('_')[0],
        'acronym': list(groups),
        'shank': 0,
        'y_um': np.arange(len(groups), dtype=float),
        'ap_i': 0,
    })}


# ---------------------------------------------------------------------------
# 1 — known rotations are recovered exactly
# ---------------------------------------------------------------------------

def test_angle_recovery():
    rng = np.random.default_rng(0)
    base = rr.smooth_circular(np.exp(-0.5 * ((np.arange(360) - 100) / 20.0) ** 2), sigma=10)
    bad = []
    for true_shift in (0, 30, -30, 45, -45, 90, 180):
        rotated = np.roll(base, true_shift)
        got = asplit.remapping_angle(base, rotated, method='xcorr')
        want = (true_shift + 180) % 360 - 180
        if not np.isclose(abs(got), abs(want), atol=1e-6):
            bad.append(f'xcorr {true_shift}: got {got}')
    a, b = np.zeros(360), np.zeros(360)
    a[10], b[100] = 1, 1
    got = asplit.remapping_angle(rr.smooth_circular(a, 10), rr.smooth_circular(b, 10),
                                 method='peak')
    if not np.isclose(got, 90.0, atol=1e-6):
        bad.append(f'peak 10->100: got {got}')
    del rng
    return (not bad), '; '.join(bad) or 'all rotations recovered exactly (xcorr + peak)'


# ---------------------------------------------------------------------------
# 2 — the coherence metric separates rigid from independent
# ---------------------------------------------------------------------------

def test_pair_coherence():
    rng = np.random.default_rng(1)
    n, n_tasks = 30, 4
    prof = np.stack([rr.smooth_circular(
        np.exp(-0.5 * (((np.arange(360) - p + 180) % 360 - 180) / 22.0) ** 2), sigma=10)
        for p in rng.uniform(0, 360, n)])

    def curves(rot):                       # rot: (n_tasks, n) rotation in degrees
        return {t: np.stack([np.roll(prof[i], int(rot[t, i])) for i in range(n)])
                for t in range(n_tasks)}

    rigid_rot = np.repeat(rng.choice([0, 90, 180, 270], n_tasks)[:, None], n, axis=1)
    rigid_rot[0] = 0
    indep_rot = rng.choice([0, 90, 180, 270], (n_tasks, n))
    indep_rot[0] = 0

    import w3_figures as wf

    out = {}
    for name, rot in (('rigid', rigid_rot), ('independent', indep_rot)):
        C = curves(rot)
        iu = np.triu_indices(n, k=1)
        r = np.column_stack([w3.angles_between(C[0], C[t]) for t in range(1, n_tasks)])
        rel = w3._wrap(r[iu[1], :] - r[iu[0], :])                       # reference-anchored
        pa = {t: asplit.pairwise_angles(C[t]) for t in range(n_tasks)}
        direct = np.column_stack([w3._wrap(pa[t][iu] - pa[0][iu]) for t in range(1, n_tasks)])
        # the DUAL criterion: coherent in X->Y *and* X->Z, chance (1/4)^2 = 1/16
        out[name] = (float(np.nanmean(wf.dual_per_row(rel))),
                     float(np.nanmean(wf.dual_per_row(direct))))

    bad = []
    if out['rigid'][0] < 0.99 or out['rigid'][1] < 0.99:
        bad.append(f"rigid should be ~1.00, got {out['rigid']}")
    # An independently-rotating population must land on the ANALYTIC dual chance, 1/16.
    # This is the control that pins the chance level the headline is measured against.
    if not all(0.03 < v < 0.11 for v in out['independent']):
        bad.append(f"independent should be near {wf.DUAL_CHANCE:.4f}, got {out['independent']}")
    if abs(out['rigid'][0] - out['rigid'][1]) > 0.02 or \
       abs(out['independent'][0] - out['independent'][1]) > 0.05:
        bad.append(f'reference-anchored and direct metrics disagree: {out}')
    detail = (f"DUAL: rigid {out['rigid'][0]:.3f}/{out['rigid'][1]:.3f}, "
              f"independent {out['independent'][0]:.4f}/{out['independent'][1]:.4f} "
              f"(ref-anchored/direct; analytic dual chance {wf.DUAL_CHANCE:.4f})")
    return (not bad), '; '.join(bad) or detail


# ---------------------------------------------------------------------------
# 3 — ordinal-state cells generalise, place cells do not
# ---------------------------------------------------------------------------

def _place_phases(tasks, port):
    """Phase at which a cell tuned to physical `port` fires: arrival at that port.

    State i is the leg from `task[i]` to `task[i+1]`, so a port-tuned cell fires at the END
    of the leg whose destination is its port -- normalised phase 90*(i+1). Because the port
    sits at a different ordinal position in each task, that phase MOVES across tasks, which
    is the whole point: a place cell must not be scored as a generalising state cell.
    """
    out = []
    for task in tasks:
        task = np.asarray(task)
        idx = int(np.flatnonzero(task == port)[0])
        out.append((90.0 * idx) % 360.0)
    return np.array(out)


def test_generalising_vs_place():
    n_tasks = 6
    rng = np.random.default_rng(2)
    ports = np.array([1, 2, 3, 4])
    tasks = [ports.copy()]
    while len(tasks) < n_tasks:
        cand = rng.permutation(ports)
        if not any(np.array_equal(cand, t) for t in tasks):
            tasks.append(cand)

    n_state, n_place = 8, 8
    phases = np.zeros((n_tasks, n_state + n_place))
    for i in range(n_state):                       # ordinal-state cells: fixed phase
        phases[:, i] = 45.0 + 90.0 * (i % 4)
    for j in range(n_place):                       # place cells: phase follows the port
        phases[:, n_state + j] = _place_phases(tasks, ports[j % 4])

    recday = make_recday(phases, tasks=tasks, seed=3)
    res = w3.run_recday(recday, w3.config(), seed=0)
    if res is None or not res['n_included']:
        return False, 'no neurons passed the gate — the synthetic is not tuned enough'

    inc = res['included_idx']
    gen = asplit.is_generalising(res['angles_xcorr'], w3.GEN_THRESHOLD_DEG)
    is_state = inc < n_state
    gen_state = gen[is_state].mean() if is_state.sum() else np.nan
    gen_place = gen[~is_state].mean() if (~is_state).sum() else np.nan

    bad = []
    if not (is_state.sum() and (~is_state).sum()):
        bad.append(f'gate kept only one kind of cell ({is_state.sum()} state, '
                   f'{(~is_state).sum()} place)')
    if not (gen_state > 0.9):
        bad.append(f'ordinal-state cells should generalise, got {gen_state:.2f}')
    if not (gen_place < 0.1):
        bad.append(f'place cells must NOT generalise, got {gen_place:.2f}')
    detail = (f'gate passed {len(inc)}/{n_state + n_place}; generalising: '
              f'state {gen_state:.2f}, place {gen_place:.2f}')
    return (not bad), '; '.join(bad) or detail


# ---------------------------------------------------------------------------
# 4 — the pair -> region join: two rigid modules
# ---------------------------------------------------------------------------

def test_anatomy_modules():
    """Two regions, each rigidly rotating, independent of each other.

    Within-region pairs must come out coherent and cross-region pairs at chance. This is the
    only control that exercises the pair -> region join, which is the new code in this
    workstream and so the likeliest place for an off-by-one: `pair_idx` indexes POSITIONS in
    `included_idx`, which indexes rows of `Neuron_raw`, which is the row order of
    `unit_regions`. Get any of those three wrong and the figure still plots.
    """
    n_tasks, n_per = 6, 12
    rng = np.random.default_rng(4)
    rot_a = np.concatenate([[0], rng.choice([90, 180, 270], n_tasks - 1)])
    rot_b = np.concatenate([[0], rng.choice([90, 180, 270], n_tasks - 1)])
    while np.any(rot_a[1:] == rot_b[1:]):          # the modules must actually differ
        rot_b = np.concatenate([[0], rng.choice([90, 180, 270], n_tasks - 1)])

    base_a = 45.0 + 90.0 * (np.arange(n_per) % 4)
    base_b = 45.0 + 90.0 * (np.arange(n_per) % 4)
    phases = np.zeros((n_tasks, 2 * n_per))
    for t in range(n_tasks):
        phases[t, :n_per] = (base_a + rot_a[t]) % 360
        phases[t, n_per:] = (base_b + rot_b[t]) % 360

    recday = make_recday(phases, seed=5)
    res = w3.run_recday(recday, w3.config(), seed=0)
    if res is None or res['n_included'] < 6:
        return False, 'too few neurons passed the gate'

    rd = 'synth_00000000_00000000'
    groups = np.array(['ENTl-deep'] * n_per + ['SUB/ProS'] * n_per)
    pairs = w3.pair_frame({rd: res}, _fake_unit_regions(rd, groups))
    if not len(pairs):
        return False, 'pair_frame returned nothing'

    within = pairs.loc[pairs['same_region'], 'coherent_rel'].mean()
    cross = pairs.loc[~pairs['same_region'], 'coherent_rel'].mean()
    # the join must also survive a shuffle of the labels: with region assignment destroyed,
    # within- and cross-region coherence have to converge
    perm = np.random.default_rng(6).permutation(groups)
    shuf = w3.pair_frame({rd: res}, _fake_unit_regions(rd, perm))
    shuf_gap = (shuf.loc[shuf['same_region'], 'coherent_rel'].mean()
                - shuf.loc[~shuf['same_region'], 'coherent_rel'].mean())

    bad = []
    if not (within > 0.9):
        bad.append(f'within-region coherence should be ~1.0, got {within:.2f}')
    if not (cross < 0.1):
        bad.append(f'cross-region coherence should be ~0, got {cross:.2f}')
    if abs(shuf_gap) > 0.25:
        bad.append(f'shuffled region labels still give a gap of {shuf_gap:+.2f} — '
                   f'the join is not reading the labels it claims to')
    detail = (f'within {within:.2f}, cross {cross:.2f}, '
              f'label-shuffled gap {shuf_gap:+.2f} ({len(pairs)} pairs)')
    return (not bad), '; '.join(bad) or detail


# ---------------------------------------------------------------------------
# 5 — the cell-identity shuffle is the null we claim it is
# ---------------------------------------------------------------------------

def test_h4_shuffle():
    """The null must preserve the h=4 structure and land at 1/4.

    A uniform-circle null would be far too easy to beat: because the task-phase spectrum is
    h=4-dominated, the cross-correlation has four near-equal peaks and the argmax is
    quantised near multiples of 90 whatever the neurons are doing. So the shuffle has to
    reproduce BOTH facts -- mass concentrated at multiples of 90, and a generalising
    fraction at chance -- or it is testing something other than what we claim.
    """
    n_tasks, n = 6, 40
    rng = np.random.default_rng(7)
    phases = rng.choice([45.0, 135.0, 225.0, 315.0], (n_tasks, n))
    recday = make_recday(phases, seed=8)
    res = w3.run_recday(recday, w3.config(), seed=0)
    if res is None or res['n_included'] < 10:
        return False, 'too few neurons passed the gate'

    sh = res['angles_shuffle_xcorr'].ravel()
    sh = sh[np.isfinite(sh)]
    gen_shuffle = asplit.is_generalising(res['angles_shuffle_xcorr'],
                                         w3.GEN_THRESHOLD_DEG).mean()
    # distance to the nearest multiple of 90 — the quantisation signature
    resid = np.abs((np.abs(sh) + 45.0) % 90.0 - 45.0)
    quantised = float(np.mean(resid < 20.0))
    per_comparison = float(np.mean(np.abs(sh) < w3.GEN_THRESHOLD_DEG))

    bad = []
    if quantised < 0.8:
        bad.append(f'shuffle is not h=4-quantised ({quantised:.2f} within 20 deg of a '
                   f'multiple of 90) — it is not preserving the structure it must')
    if not (0.10 < per_comparison < 0.45):
        bad.append(f'per-comparison shuffle rate {per_comparison:.2f} is not near chance '
                   f'{w3.CHANCE}')
    detail = (f'{quantised:.2f} quantised to multiples of 90; per-comparison '
              f'{per_comparison:.2f} (chance {w3.CHANCE}); all-{n_tasks - 1}-comparison '
              f'generalising {gen_shuffle:.3f}')
    return (not bad), '; '.join(bad) or detail


# ---------------------------------------------------------------------------
# 6 — the one-pass tuning build did not change the data
# ---------------------------------------------------------------------------

def test_build_session_equivalence():
    """`w3.build_session` must equal `rr.build_session_tuning` bit for bit.

    `build_session` folds the gate and the mean curve into one pass over the trials, halving
    an 8-minute cohort pass. That is only legitimate if it is the same computation -- so
    assert it against the tested original rather than reasoning about it.
    """
    phases = np.random.default_rng(9).choice([30.0, 120.0, 210.0, 300.0, np.nan], (3, 25))
    recday = make_recday(phases, n_trials=14, seed=10)
    cfg = w3.config()
    bad = []
    for sess in list(recday)[:2]:
        tuning, gate, _ = w3.build_session(recday[sess], cfg)
        ref_tuning, ref_gate = rr.build_session_tuning(recday[sess], cfg)
        if not np.allclose(tuning, ref_tuning, equal_nan=True):
            bad.append(f'session {sess}: tuning curves differ '
                       f'(max {np.nanmax(np.abs(tuning - ref_tuning)):.3g})')
        if not np.array_equal(gate, ref_gate):
            bad.append(f'session {sess}: gate differs on {int((gate != ref_gate).sum())} neurons')
    return (not bad), '; '.join(bad) or 'one-pass build matches rr.build_session_tuning exactly'


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def test_ragged_task_counts():
    """A recday with fewer tasks must not be scored as uniformly non-generalising.

    Recdays do not all have 6 usable tasks. The per-recday angle matrices are therefore
    ragged, and concatenating them into one frame pads the short recday's last comparison
    column with NaN. Since both "generalising" and "coherent" require EVERY comparison to be
    within threshold, a NaN column silently condemns every neuron and every pair of the
    short recday -- a whole animal-day quietly reading as fully remapping. Both flags are
    therefore decided per recday, before the concat, and this is the control that says so.
    """
    phases_6 = np.tile(45.0 + 90.0 * (np.arange(14) % 4), (6, 1))
    phases_5 = np.tile(45.0 + 90.0 * (np.arange(14) % 4), (5, 1))
    results = {'aa00_20250101_20250102': w3.run_recday(
                   make_recday(phases_6, seed=11), w3.config(), seed=0),
               'aa00_20250103_20250104': w3.run_recday(
                   make_recday(phases_5, seed=12), w3.config(), seed=0)}
    if any(r is None or not r['n_included'] for r in results.values()):
        return False, 'a synthetic recday produced no included neurons'

    groups = np.array(['ENTl-deep'] * 7 + ['SUB/ProS'] * 7)
    ur = {}
    for rd in results:
        ur.update(_fake_unit_regions(rd, groups))
    neurons = w3.neuron_frame(results, ur)
    pairs = w3.pair_frame(results, ur)

    bad = []
    ncomp = neurons.groupby('recday')['n_comparisons'].max().to_dict()
    if sorted(ncomp.values()) != [4, 5]:
        bad.append(f'expected 4 and 5 comparisons, got {ncomp}')
    for rd, sub in neurons[neurons['included']].groupby('recday'):
        frac = sub['generalising_xcorr'].mean()
        if frac < 0.9:
            bad.append(f'{rd} ({ncomp[rd]} comparisons): generalising {frac:.2f}, '
                       f'every cell was built to generalise')
    for rd, sub in pairs.groupby('recday'):
        frac = sub['coherent_rel'].mean()
        if frac < 0.9:
            bad.append(f'{rd}: coherent {frac:.2f}, every pair was built to be coherent')
    detail = ('ragged recdays scored independently: '
              + ', '.join(f'{rd.split("_")[1]} {ncomp[rd]}c '
                          f'gen={neurons[neurons.included & (neurons.recday == rd)]["generalising_xcorr"].mean():.2f} '
                          f'coh={pairs[pairs.recday == rd]["coherent_rel"].mean():.2f}'
                          for rd in sorted(results)))
    return (not bad), '; '.join(bad) or detail


CONTROLS = [
    ('1 rotation recovery', test_angle_recovery),
    ('2 pair coherence: rigid vs independent', test_pair_coherence),
    ('3 ordinal-state generalises, place does not', test_generalising_vs_place),
    ('4 anatomy join: two rigid modules', test_anatomy_modules),
    ('5 cell-identity shuffle preserves h=4', test_h4_shuffle),
    ('6 one-pass tuning build equivalence', test_build_session_equivalence),
    ('7 ragged task counts across recdays', test_ragged_task_counts),
]


def run_all(verbose=True):
    """Run every control. Returns 0 if all pass, 1 otherwise (CLI exit code)."""
    failures = 0
    for name, fn in CONTROLS:
        try:
            ok, detail = fn()
        except Exception as exc:                       # a control that errors is a failure
            ok, detail = False, f'{type(exc).__name__}: {exc}'
        failures += (not ok)
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}", flush=True)
    if verbose:
        print(f"\n{len(CONTROLS) - failures}/{len(CONTROLS)} controls passed")
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(run_all())
