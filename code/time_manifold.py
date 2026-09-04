"""
Joint geometry of two nested time codes: session time T and time-since-reward tau.

Tests the hypotheses in `mohamady_time_idea/time_torus{1,2}.pdf`. The question is
not whether LEC codes both variables (it does -- `nn_time_decoder` decodes T under
leave-one-session-out CV, `glm_analysis_v2` finds time_from_reward tuning) but what
JOINT manifold the two axes make, and whether the fast axis closes.

The candidate answers, and what separates them
----------------------------------------------
  sheet     (R1 x R1)  tau is an open arc that RESETS at reward       beta1 = 0
  cylinder  (S1 x R1)  tau is a phase that CLOSES at the leg end      beta1 = 1
  torus     (S1 x S1)  both close -- requires session time to close   beta1 = 2

Report 1's central methodological claims, all of which shape this module:

  1. TOPOLOGY CANNOT ANSWER THE INTERESTING QUESTION. Separable, sheared, warped
     and conjunctive codes are ALL beta1 = 0 sheets with identical topology but
     completely different computational content. Only cross-condition
     generalisation separates them (see `factorisation`).

  2. BIN THE FAST AXIS BOTH WAYS. A phase-like code reads closure ~1.00 binned by
     leg phase and ~0.50 binned by absolute tau; a true sheet reads ~0.00 under
     both. Binning one way lets you declare "sheet" on data that is a cylinder.
     Every condition tensor in this module is therefore built as a PAIR
     (`tau_binning='abs'` and `'phase'`), and the plotting helpers refuse to show
     one without the other.

  3. THE JUMP-MAGNITUDE STATISTIC IS NOT IDENTIFIED. Report 1 section 4 measures
     1.15 to 3.37 for the SAME model depending on which tau the baseline is
     matched to. `reset_jump` is provided for the figure panel only, returns all
     three baselines, and is never used as evidence.

  4. PERSISTENT HOMOLOGY FAILS SILENTLY unless the metric is kNN-geodesic (the
     Euclidean metric saturates on a tiled code) and the manifold is sampled
     uniformly in NEURAL ARCLENGTH rather than in seconds (a Weber-scaled code
     puts a huge neural gap between 100 s and 225 s and almost none between
     3000 s and 3125 s; a seconds-uniform grid contains one enormous gap that
     sets the connectivity scale and buries every real loop). Both failures look
     like clean negative results.

  5. SMALL POPULATIONS MANUFACTURE RINGS. Report 1 detects a spurious ring in
     4/6 runs at 25 cells and 0/6 at 400. Always report the run-to-run spread of
     H1, and treat a large spread as evidence AGAINST the feature.

Report 2's rule, which applies to every folded statistic here: any analysis that
conditions on an estimated parameter -- a period, a phase, a module assignment --
must have its null passed through the same conditioning. A shuffle that skips the
folding step is not a null for a folded statistic.

What this module deliberately does NOT do
-----------------------------------------
  * read participation ratio / number of significant PCs as evidence about
    manifold dimension (report 1 section 5: PR ranges 2.0-14.6 across models that
    are ALL 2-D manifolds)
  * report gridness or module structure without the random non-negative weight
    null on a bandwidth-matched basis (report 2 section 2: random weights match
    the learned gridness, 1.21 vs 1.27, and pass the module test in 100% of draws)
  * report beta1 from folded data without a folded null (report 2 section 4)

Task-specific notes (this is not report 1's simulation)
-------------------------------------------------------
Measured on the real stores, 2026-08-19:

  session length          ~19-21 s * 60          (his sim: 60 min)
  sessions per recday     6-9, each a diff. task (his sim: 1)
  inter-reward interval   p10 4.4 / med 8.2 / p90 18.4 s   (his: 12-75 s)
  intervals per session    ~72-168                (his: 123)
  neurons                 LEC 151-183; PFC median 53

The 6-9 sessions per recday are an asset his simulation lacks: they turn "does the
slow axis reset?" into a direct measurement, and are the cleanest available
separation of a genuine session-time code from electrode drift or satiety.
`cross_session_reset` is that test.

Reuses from glm_analysis_v2 (same directory): get_sessions_for_glm,
prepare_session_data, truncate_all_arrays, downsample_session_data.
Ports, with attribution, from mohamady_time_idea/time_manifold_code_1/{analysis,topo}.py
and time_manifold_code_part2/compress.py.
"""

from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np

from glm_analysis_v2 import (
    get_sessions_for_glm,
    prepare_session_data,
    truncate_all_arrays,
    downsample_session_data,
)

BIN_SIZE_MS = 25  # native acquisition bin (40 Hz)

# The three coordinates the fast axis can be binned in. 'abs' and 'phase' are
# report 1's pair -- absolute seconds since reward vs fraction of the current leg.
# 'loop_phase' is ours: position in the A->B->C->D->A cycle, the only one of the
# three that closes as a TASK variable, and the coordinate the existing
# `analyse_taskphase_ring` result lives on.
FAST_AXIS_BINNINGS = ('abs', 'phase', 'loop_phase')


# ============================================================================
# Palette (GridMaze Colors skill). Kept consistent with
# time_vs_progress_dissociation.py so the two modules' figures compose.
# ============================================================================

AXIS_COLORS = {
    'T':   '#2A6FB5',   # session time (cool)
    'tau': '#C03030',   # reward time  (warm)
}
BINNING_COLORS = {
    'abs':   '#C03030',   # binned in absolute seconds
    'phase': '#0F4C81',   # binned in leg phase
}
TRUTH_COLORS = {
    'sheet':    '#2A6FB5',
    'cylinder': '#E07B39',
    'torus':    '#3F8F5B',
}
REGION_COLORS = {'LEC': '#6667AB', 'PFC': '#FFA500'}   # Very Peri / Saffron
NULL_GREY = '#555555'
NEUTRAL = '#2C2C2A'   # Caviar


# ============================================================================
# Config
# ============================================================================

@dataclass
class TimeManifoldConfig:
    """Parameters for the joint (T, tau) manifold battery.

    Attributes
    ----------
    downsample_factor : int
        Applied to the raw 25 ms bins by `downsample_session_data`. 10 -> 250 ms,
        matching `time_vs_progress_dissociation` so sample selection is identical.
    n_T, n_tau : int
        Condition-tensor grid. Kept modest because our tau axis spans only
        ~0-18 s: a finer grid buys empty cells, not resolution.
    tau_range : (lo, hi) or None
        Absolute-tau window in seconds. None -> (0.0, p90 of leg duration),
        computed per recday. Report 1 section 3: the closure index reads the ends
        of THIS WINDOW, not the ends of the axis, so it must be reported.
    T_range_frac : (lo, hi)
        Fraction of the session used for the slow axis, trimming the ragged ends.
    T_range_sec : (lo, hi) or None
        Absolute-seconds window for the slow axis when `T_binning='sec'`.
        None -> (0, SHORTEST session duration), so every bin is supported by
        every session. See `build_condition_tensor`.
    min_samples_per_cell : int
        Condition-tensor cells with fewer samples are masked. The high-tau corner
        is thin here for the same reason it is in report 1's Fig. 1: long legs are
        rare.
    trim_seconds : float
        Trimmed from each end of every leg. Specified in SECONDS, not bins.
        CCGP_STATE_PAIRS.md uses 15 bins, but that is on the phase-warped grid
        (90 bins per leg), i.e. one sixth of a leg. Copying "15 bins" onto this
        250 ms substrate would remove 3.75 s from each end of a median 8-12 s leg
        -- over half the data, including the entire tau ~ 0 region the reset test
        anchors on. 1.0 s matches the peri-reward `event_window_s` that
        `time_vs_progress_dissociation.build_blocks` uses to absorb the
        reward-locked transient, which is this repo's own estimate of how long
        that transient lasts.
    smooth_sigma_bins : float
        Gaussian smoothing (in downsampled bins) applied per neuron for manifold
        work. NOT applied for decoding.
    n_pca : int
        PCA dimensionality for the manifold point cloud.
    geodesic_k : int
        k for the kNN geodesic graph (report 1: Euclidean saturates on a tiled
        code, so long-range structure needs graph distance).
    ph_thresh : float
        A H1 bar counts as a loop above this multiple of the connectivity scale.
        2.5 is report 1's threshold.
    min_local_structure : float
        The closure index is (ends - far) / (neighbour - far); below this value of
        the denominator it is not computed and NaN is returned instead. The axis
        has no local structure to normalise by, so the ratio is dominated by its
        own noise. MEASURED (6 LEC templates x 10 models x 3 seeds x 3 binnings,
        section 8.9): a tau-code binned by `loop_phase` averages over the four legs
        of the loop and flattens, dropping the denominator from ~0.54 under `abs`
        to ~0.15 -- and at 0.15 the `ramps` sheet read closure 0.37 to 0.82 across
        eight seeds, exceeding the 0.5 cut in 4/8. Separation is clean and wide:

            highest denominator among falsely-closing sheets   0.159
            lowest  denominator among true closures (torus)    0.275

        0.20 sits inside that window. The old value of 0.05 admitted the noise
        regime and produced a coin-flip gate failure on ah10. `cyl_wrap` is
        unaffected: it reads NaN here by design (`closure_index_blind`).
    n_arclength : int
        Points per axis after arclength-uniform resampling.
    min_loops_for_loop_topology : int
        Below this many A->A loops, the loop-phase topology test is underpowered
        and is reported as such rather than run. MEASURED on the `cyl_loop`
        synthetic (a true ring by construction) injected into real tables:

            recday                 n_loops   detected
            ah08_20250613_20250615    63       4/8
            ah08_20250616_20250617    73       7/8
            ah08_20250618_20250619    99       7/8

        At 63 loops the true ring (H1 3.55 +- 3.55) is not separable from a sheet
        under the same binning (sheet_orth 3.19 +- 1.45). Detection rises with
        loop count, which is the signature of underpowering rather than of a
        spurious feature -- report 1's finding #2 run in reverse.
    ridge_alpha : float or sequence
        Ridge penalty for the transfer decoders. A sequence is cross-validated on
        the training fold. A single fixed value is wrong here: LEC fires at ~0.075
        counts per 250 ms bin (0.3 Hz), so the optimal penalty spans orders of
        magnitude depending on how much smoothing is applied, and alpha=10 gives
        NEGATIVE held-out R2 on real data at every smoothing level.
    decode_smooth_seconds : float
        Smoothing for the FAST axis, applied within leg boundaries only.
        Necessary and not optional: at 0.075 counts per 250 ms bin a single-bin
        population vector is essentially Poisson noise. Smoothing ACROSS leg
        boundaries would mix the end of one leg with the start of the next and
        destroy the fast axis by construction, so `smooth_within_legs` does it
        segment-wise — which also caps the usable width at the leg duration
        (~8-12 s).
    decode_smooth_seconds_T : float
        Smoothing for the SLOW axis, applied within SESSION boundaries — legs may
        be crossed, because session time is continuous through a reward.

        The two axes genuinely need different substrates, and using one for both
        understates the slow axis by about 4x. Session time is a slow variable and
        keeps improving with wide smoothing (within-condition R2 = +0.02
        unsmoothed, +0.59 at 4 s across legs), while the same smoothing destroys
        the fast axis. `nn_time_decoder` bins at 10 s for session time for exactly
        this reason. Each decoder therefore gets the resolution matched to its own
        variable's timescale.
    n_boot, n_shuffle : int
        Bootstrap / shuffle counts.
    random_state : int
    """
    downsample_factor: int = 10
    n_T: int = 12
    n_tau: int = 10
    tau_range: tuple = None
    T_range_frac: tuple = (0.02, 0.98)
    T_range_sec: tuple = None
    min_samples_per_cell: int = 8
    trim_seconds: float = 1.0
    smooth_sigma_bins: float = 2.0
    n_pca: int = 8
    geodesic_k: int = 6
    ph_thresh: float = 2.5
    min_local_structure: float = 0.20
    n_arclength: int = 40
    min_loops_for_loop_topology: int = 90
    ridge_alpha: tuple = (1e1, 1e2, 1e3, 1e4, 1e5)
    decode_smooth_seconds: float = 1.0
    decode_smooth_seconds_T: float = 4.0
    n_boot: int = 200
    n_shuffle: int = 100
    random_state: int = 0

    @property
    def bin_seconds(self) -> float:
        return self.downsample_factor * BIN_SIZE_MS / 1000.0

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# A. Substrate -- the joint (T, tau) object
# ============================================================================

def build_time_tables(mouse_recdays, data_dic, config=None, *,
                      filter_correct_paths=False,
                      max_transition_seconds=60,
                      min_sessions=2,
                      verbose=True):
    """Long-format per-sample table of the two time axes, per recday.

    Mirrors `time_vs_progress_dissociation.build_design_tables` -- same pooling
    logic, same session dedup (`get_sessions_for_glm` keeps one session per unique
    task), same node filter (`Locs <= 21`) intersected with the transition mask --
    with one addition that exists nowhere else in the repo:

        T_sec, elapsed time since SESSION START in real seconds.

    `glm_analysis_v2` has no session-time regressor, and
    `time_coding_analysis.calculate_time_variables` computes it as
    `trial_index * 360 / 40`, i.e. trial index on the phase-warped substrate.
    Here it is the raw bin index times the bin width, so it is genuine elapsed
    seconds and is unaffected by how many samples the node filter removes.

    Unlike `build_design_tables`, session boundaries are RETAINED (`session_id`),
    because with 6-9 sessions per recday the cross-session reset is the strongest
    test available for the slow axis.

    Returns
    -------
    tables : dict {recday: table}
        Per-sample 1-D arrays unless noted:
          T_sec        elapsed seconds since session start
          T_frac       T_sec / session duration, in [0, 1]
          tau          elapsed seconds since the last reward
          time_to      seconds until the next reward
          D            leg duration (s) = tau + time_to
          phase_leg    tau / D in [0, 1]
          dist_from    path length since last reward
          dist_to      path length to next reward
          speed, acc   kinematics
          locs         node id (1..21)
          state        task state (leg A->B = 0, ...)
          interval_id  globally-unique leg index
          loop_id      globally-unique loop (A->A) index
          session_id   metadata-ordinal session index
          within_leg_i sample index within its leg (for trimming)
          leg_n        number of samples in its leg
          FR           spike counts [n_neurons x n_samples]   *** 2-D ***
          session_dur  dict {session_id: duration in seconds}
          bin_seconds  float
    """
    cfg = config or TimeManifoldConfig()
    binsec = BIN_SIZE_MS / 1000.0          # raw bin width; tau/time_to are in raw bins
    dt = cfg.bin_seconds                   # downsampled sample spacing
    tables = {}

    any_filter = filter_correct_paths or max_transition_seconds is not None

    for mr in mouse_recdays:
        if mr not in data_dic:
            continue
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        if len(sessions) < min_sessions:
            continue

        cols = defaultdict(list)
        fr_blocks = []
        iid_offset = 0
        lid_offset = 0
        session_dur = {}

        for sess in sessions:
            sd = data_dic[mr][sess]
            prep = prepare_session_data(
                sd,
                task=sd.get('Task') if any_filter else None,
                filter_correct_paths=filter_correct_paths,
                max_transition_seconds=max_transition_seconds,
                bin_size_ms=BIN_SIZE_MS,
            )
            prep = truncate_all_arrays(prep)
            prep = downsample_session_data(prep, cfg.downsample_factor)

            tf = np.asarray(prep['time_from_reward'], float)
            tt = np.asarray(prep['time_to_reward'], float)
            n = len(tf)
            if n == 0:
                continue

            # --- the new variable ------------------------------------------
            # Sample index -> seconds. Computed BEFORE the validity filter so
            # that removing samples never shifts the clock.
            T_sec = np.arange(n, dtype=float) * dt
            dur = float(n * dt)

            # interval (leg) id: a new leg starts wherever elapsed time fails to
            # increase. Same rule as build_design_tables.
            resets = np.r_[True, np.diff(tf) <= 0]
            iid = np.cumsum(resets) - 1

            # loop (A->A) id: a new loop starts at each entry into state 0.
            state = np.asarray(prep['State'], int)
            loop_resets = np.r_[True, (state[1:] == 0) & (state[:-1] != 0)]
            lid = np.cumsum(loop_resets) - 1

            # position within leg, and leg length, for the end-trims.
            # `resets[0]` is True by construction, so maximum.accumulate carries
            # the index of the most recent leg start forward.
            leg_start = np.maximum.accumulate(np.where(resets, np.arange(n), 0))
            within = np.arange(n) - leg_start
            leg_n = np.bincount(iid, minlength=iid.max() + 1)[iid]

            nf = np.asarray(prep['Locs']) <= 21
            vtm = prep.get('valid_transition_mask')
            if vtm is not None:
                vtm = np.asarray(vtm, bool)
                if len(vtm) == len(nf):
                    nf = nf & vtm
                elif len(vtm) > len(nf):
                    nf = nf & vtm[:len(nf)]
                else:
                    pad = np.zeros(len(nf), bool)
                    pad[:len(vtm)] = vtm
                    nf = nf & pad

            D = tf + tt
            valid = nf & (D > 0)
            if not valid.any():
                continue

            cols['T_sec'].append(T_sec[valid])
            cols['T_frac'].append(T_sec[valid] / max(dur, 1e-9))
            cols['tau'].append(tf[valid] * binsec)
            cols['time_to'].append(tt[valid] * binsec)
            cols['D'].append(D[valid] * binsec)
            cols['phase_leg'].append(np.clip(tf[valid] / D[valid], 0, 1))
            cols['dist_from'].append(np.asarray(prep['dist_from_reward'], float)[valid])
            cols['dist_to'].append(np.asarray(prep['dist_to_reward'], float)[valid])
            cols['speed'].append(np.asarray(prep['Speed'], float)[valid])
            cols['acc'].append(np.asarray(prep['Acc'], float)[valid])
            cols['locs'].append(np.asarray(prep['Locs'], float)[valid])
            cols['state'].append(state[valid].astype(float))
            # loop phase: position within the A->B->C->D->A cycle. This is the
            # THIRD clock, and the one that genuinely can close -- the task
            # returns to A, so a code tracking position in the loop has a real
            # S1. The existing task-phase ring (`analyse_taskphase_ring`) lives
            # on this variable, which is why it needs its own binning rather
            # than being folded into the leg-phase one.
            loop_ph = np.zeros(n)
            lstart = np.maximum.accumulate(np.where(loop_resets, np.arange(n), 0))
            lcount = np.bincount(lid, minlength=lid.max() + 1)[lid]
            loop_ph = (np.arange(n) - lstart) / np.maximum(lcount, 1)
            cols['loop_phase'].append(np.clip(loop_ph[valid], 0, 1))
            cols['within_leg_i'].append(within[valid].astype(float))
            cols['leg_n'].append(leg_n[valid].astype(float))
            cols['interval_id'].append(iid[valid] + iid_offset)
            cols['loop_id'].append(lid[valid] + lid_offset)
            cols['session_id'].append(np.full(int(valid.sum()), float(sess)))

            iid_offset += int(iid[valid].max()) + 1
            lid_offset += int(lid[valid].max()) + 1
            session_dur[int(sess)] = dur

            fr_blocks.append(np.asarray(prep['FR'], float)[:, valid])

        if not fr_blocks or sum(b.shape[1] for b in fr_blocks) == 0:
            continue

        table = {k: np.concatenate(v) for k, v in cols.items()}
        table['FR'] = np.concatenate(fr_blocks, axis=1)
        table['session_dur'] = session_dur
        table['bin_seconds'] = dt
        table['recday'] = mr
        tables[mr] = table

        if verbose:
            n_neu, n_samp = table['FR'].shape
            print(f"  {mr}: {n_neu} neurons x {n_samp} samples, "
                  f"{len(session_dur)} sessions, "
                  f"{len(np.unique(table['interval_id']))} legs, "
                  f"{len(np.unique(table['loop_id']))} loops")

    return tables


def leg_duration_stats(table):
    """p10 / median / p90 of leg duration (s). Sets the default tau window and is
    the leverage the whole clock-vs-phase dissociation rides on."""
    _, first = np.unique(table['interval_id'], return_index=True)
    D = np.asarray(table['D'], float)[first]
    D = D[np.isfinite(D) & (D > 0)]
    if D.size == 0:
        return dict(p10=np.nan, median=np.nan, p90=np.nan, n=0, ratio=np.nan)
    p10, med, p90 = np.percentile(D, [10, 50, 90])
    return dict(p10=float(p10), median=float(med), p90=float(p90),
                n=int(D.size), ratio=float(p90 / max(p10, 1e-9)))


def trim_mask(table, trim_seconds=None, config=None):
    """Drop `trim_seconds` from each end of every leg.

    The reward transient lives in exactly those windows, so any result that
    survives only WITHOUT the trim is a reward-transient result. But the trim also
    removes the tau ~ 0 anchor the reset test needs, so every reset figure must be
    shown both ways rather than trimmed by default.

    Legs shorter than 2 * trim_seconds are dropped entirely (they contain nothing
    but the two transients).
    """
    cfg = config or TimeManifoldConfig()
    ts = cfg.trim_seconds if trim_seconds is None else trim_seconds
    if ts <= 0:
        return np.ones(len(table['tau']), bool)
    tb = int(round(ts / table['bin_seconds']))
    i = np.asarray(table['within_leg_i'], float)
    n = np.asarray(table['leg_n'], float)
    return (i >= tb) & (i < n - tb)


# ---------------------------------------------------------------- tensors

def _edges(lo, hi, n):
    return np.linspace(lo, hi, n + 1)


def build_condition_tensor(table, config=None, *, tau_binning='abs',
                           T_binning='frac', mask=None, sessions=None,
                           zscore=True):
    """Condition-mean population vectors on the (T, tau) grid.

    ALWAYS build both binnings (`build_tensor_pair` does it for you). Report 1's
    single most important practical finding is that the binning variable decides
    the answer more than the neuron count does: a genuinely phase-like code reads
    closure ~1.00 under phase binning and ~0.50 under absolute binning, while a
    true sheet reads ~0.00 under both. One binning alone cannot tell those apart.

    Parameters
    ----------
    tau_binning : {'abs', 'phase'}
        'abs'   -- bin tau in absolute seconds over `config.tau_range`
        'phase' -- bin tau/D in [0, 1]
    T_binning : {'frac', 'sec'}
        'frac' -- bin T/session_duration in [0, 1] (default; what every geometry
                  statistic in this module uses). Every session contributes to
                  every bin, so the grid is evenly supported.
        'sec'  -- bin T in absolute seconds over `config.T_range_sec`, which
                  defaults to (0, SHORTEST session duration). The cap matters:
                  sessions run 1022-1281 s here, so binning to the longest one
                  would build the right-hand edge of the map out of the long
                  sessions only, and the tail would get noisier for a reason that
                  has nothing to do with the neurons.
    mask : bool array, optional
        Per-sample inclusion (e.g. `trim_mask(table)`).
    sessions : sequence, optional
        Restrict to these session ids.
    zscore : bool
        z-score each neuron across the occupied grid cells before returning.

    Returns
    -------
    dict with
      M          [n_T, n_tau, N] condition means, NaN where unoccupied
      occupancy  [n_T, n_tau] sample counts
      valid      [n_T, n_tau] bool, occupancy >= min_samples_per_cell
      T_centres, tau_centres
      tau_window (lo, hi) -- REPORT THIS: the closure index reads its ends
      T_window   (lo, hi) in the units of `T_binning`
      tau_binning, T_binning
    """
    cfg = config or TimeManifoldConfig()
    if tau_binning not in FAST_AXIS_BINNINGS:
        raise ValueError(f"tau_binning must be one of {FAST_AXIS_BINNINGS}")
    if T_binning not in ('frac', 'sec'):
        raise ValueError("T_binning must be 'frac' or 'sec'")

    keep = np.ones(len(table['tau']), bool) if mask is None else np.asarray(mask, bool)
    if sessions is not None:
        keep = keep & np.isin(table['session_id'], np.asarray(sessions, float))

    if T_binning == 'frac':
        Tv = np.asarray(table['T_frac'], float)[keep]
        T_lo, T_hi = cfg.T_range_frac
    else:
        Tv = np.asarray(table['T_sec'], float)[keep]
        if cfg.T_range_sec is not None:
            T_lo, T_hi = cfg.T_range_sec
        else:
            durs = list(table['session_dur'].values())
            T_lo, T_hi = 0.0, float(min(durs)) if durs else 1.0
    if not np.isfinite(T_hi) or T_hi <= T_lo:
        T_hi = T_lo + 1.0

    if tau_binning == 'abs':
        tv = np.asarray(table['tau'], float)[keep]
        if cfg.tau_range is not None:
            lo, hi = cfg.tau_range
        else:
            lo, hi = 0.0, float(leg_duration_stats(table)['p90'])
    elif tau_binning == 'phase':
        tv = np.asarray(table['phase_leg'], float)[keep]
        lo, hi = 0.0, 1.0
    else:                                   # 'loop_phase'
        tv = np.asarray(table['loop_phase'], float)[keep]
        lo, hi = 0.0, 1.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    FR = np.asarray(table['FR'], float)[:, keep]

    T_edges = _edges(T_lo, T_hi, cfg.n_T)
    tau_edges = _edges(lo, hi, cfg.n_tau)

    iT = np.digitize(Tv, T_edges) - 1
    it = np.digitize(tv, tau_edges) - 1
    ok = (iT >= 0) & (iT < cfg.n_T) & (it >= 0) & (it < cfg.n_tau)

    N = FR.shape[0]
    M = np.full((cfg.n_T, cfg.n_tau, N), np.nan)
    occ = np.zeros((cfg.n_T, cfg.n_tau), int)

    flat = iT[ok] * cfg.n_tau + it[ok]
    counts = np.bincount(flat, minlength=cfg.n_T * cfg.n_tau)
    occ = counts.reshape(cfg.n_T, cfg.n_tau)
    # sum per cell per neuron, vectorised over neurons
    sums = np.zeros((cfg.n_T * cfg.n_tau, N))
    np.add.at(sums, flat, FR[:, ok].T)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = sums / counts[:, None]
    means[counts == 0] = np.nan
    M = means.reshape(cfg.n_T, cfg.n_tau, N)

    valid = occ >= cfg.min_samples_per_cell
    M[~valid] = np.nan

    if zscore:
        flatM = M.reshape(-1, N)
        mu = np.nanmean(flatM, axis=0)
        sd = np.nanstd(flatM, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        M = (M - mu) / sd

    return dict(M=M, occupancy=occ, valid=valid,
                T_centres=0.5 * (T_edges[:-1] + T_edges[1:]),
                tau_centres=0.5 * (tau_edges[:-1] + tau_edges[1:]),
                tau_window=(float(lo), float(hi)),
                T_window=(float(T_lo), float(T_hi)),
                tau_binning=tau_binning, T_binning=T_binning,
                n_neurons=int(N))


def build_tensor_pair(table, config=None, binnings=('abs', 'phase'), **kw):
    """Several fast-axis binnings at once. Use this, not `build_condition_tensor`.

    Report 1 finding #1: running both is a clean disambiguation, not a fishing
    expedition -- a true sheet gives ~0.00 under both with zero false positives in
    60 runs, so the pair has no cost and one binning alone can be actively wrong.

    Pass `binnings=FAST_AXIS_BINNINGS` to add the loop-phase axis, which is where
    the ABCD cycle genuinely closes.
    """
    return {b: build_condition_tensor(table, config, tau_binning=b, **kw)
            for b in binnings}


def fill_tensor(tensor):
    """Condition tensor with unoccupied cells filled by nearest-valid interpolation.

    Every downstream geometry statistic needs a complete grid. Filling is done
    once, here, so that the amount of imputation is visible and reportable rather
    than hidden inside each statistic.

    Returns (M_filled [n_T, n_tau, N], frac_imputed).
    """
    M = np.array(tensor['M'], float)
    valid = np.asarray(tensor['valid'], bool)
    nT, nTau, N = M.shape
    if valid.all():
        return M, 0.0
    if not valid.any():
        raise ValueError('no valid condition cells -- lower min_samples_per_cell')

    ii, jj = np.nonzero(valid)
    qi, qj = np.nonzero(~valid)
    d = (qi[:, None] - ii[None, :]) ** 2 + (qj[:, None] - jj[None, :]) ** 2
    nearest = np.argmin(d, axis=1)
    M[qi, qj] = M[ii[nearest], jj[nearest]]
    return M, float((~valid).sum() / valid.size)


# ============================================================================
# B. The reset test -- the headline (report 1 section 4)
# ============================================================================

def reset_return_curve(table, config=None, *, binning='abs', n_bins=None,
                       mask=None, anchor_frac=0.1, smooth=True):
    """Population-vector correlation with the just-after-reward state, vs tau.

    THE single most informative plot available from this data, and it needs no
    dimensionality reduction (report 1 section 4). The logic:

        if the fast axis CLOSES, the state at the end of the interval IS the state
        at the beginning, so the correlation comes back up towards 1;
        if it RESETS, the correlation decays to a plateau and stays there.

    The plateau sits well above zero (~0.3 in report 1's simulations) because a
    tiled code shares a mean across all states. **The diagnostic is the SHAPE, not
    the floor.** Do not threshold the plateau height.

    Report 1 explicitly warns that the alternative statistic -- the size of the
    population jump across a reward -- is NOT identified (1.15 to 3.37 for one
    model depending on the baseline). See `reset_jump`, which exists only to draw
    that ambiguity rather than to resolve it.

    Parameters
    ----------
    binning : {'abs', 'phase'}
        Which fast-axis coordinate to bin. RUN BOTH. A phase code returns to ~1
        under 'phase' binning by construction; if it also returns under 'abs' the
        closure is a property of the neural code rather than of the warp.
    anchor_frac : float
        Fraction of the axis defining the "just after reward" anchor state.
    mask : bool array, optional
        Per-sample inclusion. Pass `trim_bins=0` trims to see the reward
        transient, and the default trim to see the code without it.

    Returns
    -------
    dict with centres, corr, n (samples per bin), anchor_n, binning.
    """
    cfg = config or TimeManifoldConfig()
    nb = cfg.n_tau * 2 if n_bins is None else n_bins

    keep = np.ones(len(table['tau']), bool) if mask is None else np.asarray(mask, bool)
    if binning == 'abs':
        x = np.asarray(table['tau'], float)[keep]
        lo, hi = (cfg.tau_range if cfg.tau_range is not None
                  else (0.0, float(leg_duration_stats(table)['p90'])))
    elif binning == 'phase':
        x = np.asarray(table['phase_leg'], float)[keep]
        lo, hi = 0.0, 1.0
    elif binning == 'loop_phase':
        # anchored on the A collection rather than on any reward, so this asks
        # whether the population returns to its A-onset state after a full loop
        x = np.asarray(table['loop_phase'], float)[keep]
        lo, hi = 0.0, 1.0
    else:
        raise ValueError(f'binning must be one of {FAST_AXIS_BINNINGS}')

    FR = np.asarray(table['FR'], float)[:, keep]
    if smooth and cfg.smooth_sigma_bins > 0:
        from scipy.ndimage import gaussian_filter1d
        FR = gaussian_filter1d(FR, cfg.smooth_sigma_bins, axis=1)

    # z-score per neuron so the correlation is over the coding pattern, not rate
    mu = FR.mean(1, keepdims=True)
    sd = FR.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (FR - mu) / sd

    edges = _edges(lo, hi, nb)
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < nb)

    anchor_sel = ok & (x <= lo + anchor_frac * (hi - lo))
    if anchor_sel.sum() < 5:
        return dict(centres=np.array([]), corr=np.array([]), n=np.array([]),
                    anchor_n=int(anchor_sel.sum()), binning=binning)
    anchor = Z[:, anchor_sel].mean(1)
    anchor = anchor - anchor.mean()
    an = np.linalg.norm(anchor)

    centres = 0.5 * (edges[:-1] + edges[1:])
    corr = np.full(nb, np.nan)
    counts = np.zeros(nb, int)
    for b in range(nb):
        sel = ok & (idx == b)
        counts[b] = int(sel.sum())
        if counts[b] < 3:
            continue
        v = Z[:, sel].mean(1)
        v = v - v.mean()
        corr[b] = float(anchor @ v / (an * np.linalg.norm(v) + 1e-12))

    return dict(centres=centres, corr=corr, n=counts,
                anchor_n=int(anchor_sel.sum()), binning=binning,
                window=(float(lo), float(hi)))


def return_curve_shape(curve):
    """Reduce a return curve to the two numbers that distinguish sheet from ring.

    `dip` is the minimum of the curve; `recovery` is how much of the way back to
    the anchor the curve climbs after that minimum:

        recovery = (corr_end - corr_min) / (1 - corr_min)

    A closed axis recovers towards 1; an open one stays at its plateau, so
    recovery ~ 0. Reporting both makes the shape claim quantitative without
    thresholding the plateau height, which report 1 warns against.
    """
    c = np.asarray(curve['corr'], float)
    good = np.isfinite(c)
    if good.sum() < 4:
        return dict(dip=np.nan, recovery=np.nan, corr_end=np.nan, i_min=-1)
    cc = c[good]
    i_min = int(np.argmin(cc))
    dip = float(cc[i_min])
    # tail = mean of the last 15% of finite bins, so a single noisy end bin
    # cannot manufacture a recovery
    n_tail = max(1, int(round(0.15 * cc.size)))
    corr_end = float(np.mean(cc[-n_tail:]))
    denom = 1.0 - dip
    rec = float((corr_end - dip) / denom) if denom > 1e-6 else np.nan
    return dict(dip=dip, recovery=rec, corr_end=corr_end, i_min=i_min)


def reset_jump(table, config=None, *, window_s=1.5, band_s=2.0, mask=None):
    """Population displacement across a reward, with ALL THREE baselines.

    PROVIDED FOR THE FIGURE PANEL ONLY -- this statistic is not identified.
    Report 1 section 4 measures ratios of 1.15 / 2.96 / 3.37 for the SAME
    Weber-scaled sheet depending on whether the within-trial baseline is matched
    to the post-reward tau, pooled over all tau, or matched to the pre-reward tau,
    "and there is no principled way to choose". Under Weber scaling the population
    moves several times faster just after a reward than late in a leg, and the
    across-reward comparison necessarily straddles both regimes.

    A value near 1.0 under EVERY baseline is still informative (closed models sit
    there); a value above 1 is not. Use `reset_return_curve` for the real test.

    Ported from mohamady_time_idea/time_manifold_code_1/analysis.py:reset_jump.
    """
    cfg = config or TimeManifoldConfig()
    dt = table['bin_seconds']
    keep = np.ones(len(table['tau']), bool) if mask is None else np.asarray(mask, bool)

    tau = np.asarray(table['tau'], float)
    iid = np.asarray(table['interval_id'], float)
    FR = np.asarray(table['FR'], float)
    mu = FR.mean(1, keepdims=True); sd = FR.std(1, keepdims=True); sd[sd == 0] = 1
    Z = ((FR - mu) / sd).T                       # [n_samples, N]

    lag = max(1, int(round(window_s / dt)))
    n = len(tau)

    # reward crossings = leg boundaries, taken only where the pre/post samples
    # are contiguous in the ORIGINAL recording (same session, adjacent legs)
    bnd = np.nonzero(np.diff(iid) == 1)[0] + 1
    pre, post = [], []
    for i in bnd:
        if i - lag >= 0 and i + lag < n and keep[i - lag] and keep[i + lag]:
            if iid[i - lag] == iid[i] - 1 and iid[i + lag] == iid[i]:
                pre.append(i - lag); post.append(i + lag)
    if len(pre) < 5:
        return dict(n_crossings=len(pre), ratio_pooled=np.nan,
                    ratio_post_matched=np.nan, ratio_pre_matched=np.nan)
    pre = np.array(pre); post = np.array(post)
    across = np.linalg.norm(Z[post] - Z[pre], axis=1)

    # within-leg pairs separated by the same elapsed time, no reward crossed
    j = np.arange(n - 2 * lag)
    same_leg = (iid[j + 2 * lag] == iid[j]) & keep[j] & keep[j + 2 * lag]
    j = j[same_leg]
    if len(j) < 20:
        return dict(n_crossings=len(pre), ratio_pooled=np.nan,
                    ratio_post_matched=np.nan, ratio_pre_matched=np.nan)
    within = np.linalg.norm(Z[j + 2 * lag] - Z[j], axis=1)
    start = tau[j]

    def matched(target):
        m = np.abs(start[:, None] - target[None, :]).min(1) < band_s
        return within[m] if m.sum() > 20 else within

    med = lambda v: float(np.median(v))
    return dict(n_crossings=int(len(pre)),
                across=across, within=within,
                ratio_pooled=med(across) / (med(within) + 1e-9),
                ratio_post_matched=med(across) / (med(matched(tau[post])) + 1e-9),
                ratio_pre_matched=med(across) / (med(matched(tau[pre])) + 1e-9),
                caveat='NOT IDENTIFIED -- report all three, use reset_return_curve instead')


# ============================================================================
# C. Closure index (report 1 section 3)
# ============================================================================

def _rowcorr(a, b):
    """Correlation between matched rows of two [..., N] stacks, mean-centred
    across neurons. Matches the convention in report 1's `closure_test`."""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (np.sum(a * b, -1) /
            (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12))


def closure_test(tensor, config=None):
    """Do the two ends of each axis meet?

    Scaled so that 0 means "the ends are as far apart as any two states on the
    manifold" and 1 means "the ends are the same state":

        closure = (ends - far) / (neighbour - far)

    CRITICAL CAVEAT, and it must go in every caption: this tests the ends of your
    analysis WINDOW, not the topology of the axis. Report 1's wrapped-clock model
    has a genuine ring (marginal beta1 = 1) but reads closure -0.40 because its
    25 s period does not divide the 0.3-40 s window. Only persistent homology of
    the marginal is window-independent. `tensor['tau_window']` is returned so the
    window is always reportable alongside the number.

    The index is undefined when the axis has no local structure (neighbour ~ far);
    `neighbour - far` is returned so that can be seen rather than inferred, and the
    index itself is NaN below `config.min_local_structure`. That floor is not
    cosmetic: binning a tau-code by `loop_phase` averages over the four legs and
    flattens the axis, and in that regime the `ramps` sheet read closure 0.37 to
    0.82 across eight seeds on the same template -- a coin flip against the gate's
    0.5 cut. See section 8.9 for the measured separation.

    Ported from mohamady_time_idea/time_manifold_code_1/analysis.py:closure_test.
    """
    cfg = config or TimeManifoldConfig()
    G, _ = fill_tensor(tensor)
    nT, nTau, _ = G.shape
    G = G - G.mean(axis=(0, 1))

    def axis_stats(H, n):
        """H indexed [outer, inner, N]; closure of the INNER axis."""
        ends = float(_rowcorr(H[:, 0], H[:, -1]).mean())
        neigh = float(_rowcorr(H[:, :-1], H[:, 1:]).mean())
        far_pairs = [_rowcorr(H[:, i], H[:, j]).mean()
                     for i in range(n) for j in range(n)
                     if abs(i - j) > n // 3
                     and not (min(i, j) == 0 and max(i, j) == n - 1)]
        far = float(np.mean(far_pairs)) if far_pairs else np.nan
        d = neigh - far
        idx = (float((ends - far) / d)
               if np.isfinite(d) and d > cfg.min_local_structure else np.nan)
        return ends, neigh, far, d, idx

    e_t, n_t, f_t, d_t, i_t = axis_stats(G, nTau)                    # tau axis
    e_T, n_T, f_T, d_T, i_T = axis_stats(np.swapaxes(G, 0, 1), nT)   # T axis

    return dict(
        tau_ends=e_t, tau_neighbour=n_t, tau_far=f_t,
        tau_local_structure=d_t, tau_closure=i_t,
        T_ends=e_T, T_neighbour=n_T, T_far=f_T,
        T_local_structure=d_T, T_closure=i_T,
        tau_binning=tensor['tau_binning'], tau_window=tensor['tau_window'],
        n_neurons=tensor['n_neurons'],
    )


def closure_both_binnings(table, config=None, **kw):
    """Closure of both axes under both fast-axis binnings.

    This is report 1's finding #1 in one call. Read the tau row: ~1.00 under
    'phase' with ~0.50 under 'abs' is a genuinely phase-like (cylinder) code;
    ~0.00 under both is a sheet. Either number alone is ambiguous.
    """
    pair = build_tensor_pair(table, config, **kw)
    return {b: closure_test(T, config) for b, T in pair.items()}


# ============================================================================
# D. Factorisation -- the test that actually separates the hypotheses
# ============================================================================
#
# Report 1 section 5: all four "plane" models are topologically identical and
# functionally completely different. Topology is the wrong instrument; cross-
# condition generalisation is the right one, and it is also the cheaper test and
# the one that separates at a smaller population.
# ============================================================================

def _additive_fit(G):
    """Least-squares additive model f(T) + g(tau) of a [nT, nTau, N] tensor."""
    grand = G.mean(axis=(0, 1))
    mT = G.mean(1) - grand
    mTau = G.mean(0) - grand
    return grand[None, None] + mT[:, None] + mTau[None, :]


def additive_r2_insample(tensor):
    """In-sample additive fit. BIASED BY NOISE -- use `additive_r2` instead.

    Kept because it is the literal port of report 1's statistic and is the right
    thing on a noiseless tensor. On our data it is not: it measures SNR as much as
    additivity. Measured on synthetics injected into a real table, where the true
    answer is 1.0 for both:

        sheet_orth (tiled, high SNR)   0.97
        ramps      (rank 2, low SNR)   0.65     <-- exactly additive, reads 0.65
        conjunctive (true answer 0.47) 0.49

    i.e. an exactly additive rank-2 code is barely distinguishable from a
    conjunctive one. The residual of the additive fit contains all the per-cell
    Poisson noise, and how much that costs depends on the code's dimensionality,
    not on whether it factorises.
    """
    G, _ = fill_tensor(tensor)
    grand = G.mean(axis=(0, 1))
    ss_res = float(np.sum((G - _additive_fit(G)) ** 2))
    ss_tot = float(np.sum((G - grand) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def additive_r2(table, config=None, *, tau_binning='abs', mask=None,
                n_splits=8, seed=0, **kw):
    """Cross-validated additive index: is the code f(T) + g(tau), or conjunctive?

    Report 1 section 5's cheapest discriminator, made noise-robust. Both an
    additive model and an unrestricted (full) model are fit on one half of the
    samples in every condition cell and scored on the held-out half, and the
    statistic is the RATIO

        additive_index = R2_additive_heldout / R2_full_heldout

    which is 1.0 for any additive code regardless of its SNR or dimensionality,
    and drops towards 0 as the code becomes conjunctive. The unrestricted model is
    the correct ceiling because it is the best any model can do given the same
    per-cell noise, so dividing by it cancels exactly the bias that breaks the
    in-sample version (see `additive_r2_insample` for the measured failure).

    Returns dict with additive_index, r2_additive, r2_full, n_splits_used.
    """
    cfg = config or TimeManifoldConfig()
    rng = np.random.default_rng(cfg.random_state if seed is None else seed)
    keep = np.ones(len(table['tau']), bool) if mask is None else np.asarray(mask, bool)

    idx_all = np.flatnonzero(keep)
    num_a, num_f, den = [], [], []
    for _ in range(n_splits):
        half = rng.random(len(idx_all)) < 0.5
        m1 = np.zeros(len(table['tau']), bool); m1[idx_all[half]] = True
        m2 = np.zeros(len(table['tau']), bool); m2[idx_all[~half]] = True
        t1 = build_condition_tensor(table, cfg, tau_binning=tau_binning,
                                    mask=m1, zscore=False, **kw)
        t2 = build_condition_tensor(table, cfg, tau_binning=tau_binning,
                                    mask=m2, zscore=False, **kw)
        both = t1['valid'] & t2['valid']
        if both.sum() < 8:
            continue
        A = t1['M'][both]                      # [n_cells, N] train half
        B = t2['M'][both]                      # held-out half
        # additive fit needs the grid shape, so fit on the filled train tensor
        G1, _ = fill_tensor(t1)
        fit_add = _additive_fit(G1)[both]
        num_a.append(np.sum((B - fit_add) ** 2))
        num_f.append(np.sum((B - A) ** 2))
        den.append(np.sum((B - B.mean(0)) ** 2))

    if not den:
        return dict(additive_index=np.nan, r2_additive=np.nan, r2_full=np.nan,
                    n_splits_used=0)
    r2_add = 1.0 - float(np.sum(num_a)) / float(np.sum(den))
    r2_full = 1.0 - float(np.sum(num_f)) / float(np.sum(den))
    # The ratio is only meaningful when there is a ceiling worth dividing by.
    # With r2_full near zero the condition means are pure noise and the ratio is
    # unstable (a rank-2 ramp synthetic returns 2.24 at r2_full = 0.03). Report
    # r2_full alongside so a low ceiling is visible rather than inferred.
    ratio = float(r2_add / r2_full) if r2_full > 0.05 else np.nan
    return dict(additive_index=ratio, r2_additive=r2_add, r2_full=r2_full,
                n_splits_used=len(den), tau_binning=tau_binning)


def smooth_within_legs(FR, leg_ids, sigma_bins):
    """Gaussian-smooth each neuron WITHIN each leg, never across leg boundaries.

    Smoothing across a reward would mix the end of one leg with the start of the
    next, destroying the fast axis by construction — the analysis would then be
    measuring its own preprocessing. Segments are smoothed independently with
    reflecting edges.
    """
    from scipy.ndimage import gaussian_filter1d
    if sigma_bins <= 0:
        return np.asarray(FR, float)
    out = np.array(FR, float)
    leg_ids = np.asarray(leg_ids)
    bounds = np.flatnonzero(np.diff(leg_ids) != 0) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(leg_ids)]):
        if b - a >= 2:
            out[:, a:b] = gaussian_filter1d(out[:, a:b], sigma_bins, axis=1,
                                            mode='reflect')
    return out


def _ridge_score(Xtr, ytr, Xte, yte, alpha=10.0):
    """Ridge R2 + median absolute error, standardised on TRAIN statistics only.

    Train-only standardisation is not cosmetic: `SELECTIVITY_GEOMETRY.md` section 4
    records that fitting the scaler on all data (or not at all) produced plausible
    near-chance accuracies on trivially separable data, because the solver
    silently failed to converge.

    `alpha` may be a sequence, in which case it is chosen by an inner split of the
    TRAINING fold only. A fixed alpha is not safe across substrates: at LEC firing
    rates alpha=10 returns negative held-out R2 at every smoothing level, while
    the same data at alpha=1e3 reaches +0.59.
    """
    from sklearn.linear_model import Ridge
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    alphas = np.atleast_1d(np.asarray(alpha, float))
    best_a, best_s = float(alphas[0]), -np.inf
    if len(alphas) > 1 and len(ytr) >= 40:
        cut = len(ytr) // 2
        inner = np.zeros(len(ytr), bool); inner[:cut] = True
        for a in alphas:
            m = Ridge(alpha=a).fit(Ztr[inner], ytr[inner])
            p = m.predict(Ztr[~inner])
            den = np.sum((ytr[~inner] - ytr[~inner].mean()) ** 2)
            s = 1 - np.sum((ytr[~inner] - p) ** 2) / den if den > 0 else -np.inf
            if s > best_s:
                best_s, best_a = s, float(a)
    m = Ridge(alpha=best_a).fit(Ztr, ytr)
    p = m.predict(Zte)
    denom = np.sum((yte - yte.mean()) ** 2)
    r2 = float(1 - np.sum((yte - p) ** 2) / denom) if denom > 0 else np.nan
    return dict(r2=r2, mae=float(np.median(np.abs(yte - p))),
                alpha=best_a, n_train=int(len(ytr)), n_test=int(len(yte)))


def factorisation(table, config=None, *, mask=None, n_neurons=None, seed=None):
    """Cross-condition transfer decoding, both ways.

    Four numbers, and the CONTRAST between each transfer score and its own
    within-condition ceiling is the result -- not the raw score:

      tau_within         ceiling: decode tau, train/test split BY LEG
      tau_across_T       decode tau trained early in the session, tested late
      T_within           ceiling: decode T at low tau, split by leg
      T_across_tau       decode T trained at low tau, tested at high tau

    Report 1's separation: orthogonal product code transfers at 0.86 (tau) and
    0.95 (T); the conjunctive code collapses to -0.17 +- 0.50 and 0.28 +- 0.23,
    i.e. to chance with a spread wide enough that a single-session estimate is
    meaningless in isolation. Run this per recday and read the distribution.

    Splits are BY LEG, never by sample. Samples within a leg are strongly
    autocorrelated; `selectivity_geometry.cv_decode` folds by trial for the same
    reason and documents the leakage.

    `n_neurons` subsamples the population, for N-matching between LEC (151-183)
    and PFC (median 53). Report 1's sweep shows the separable/conjunctive
    distinction appears at ~50 cells, so an unmatched region comparison is a
    statement about recording yield.

    Ported from mohamady_time_idea/time_manifold_code_1/analysis.py:ccgp_tests.
    """
    cfg = config or TimeManifoldConfig()
    rng = np.random.default_rng(cfg.random_state if seed is None else seed)

    keep = np.ones(len(table['tau']), bool) if mask is None else np.asarray(mask, bool)
    T = np.asarray(table['T_frac'], float)[keep]
    tau = np.asarray(table['tau'], float)[keep]
    leg = np.asarray(table['interval_id'], float)[keep]
    sid = np.asarray(table['session_id'], float)[keep]
    FR = np.asarray(table['FR'], float)[:, keep]

    # Each axis gets the smoothing its own timescale needs (see the config
    # docstring): the fast decoders may not cross a reward, the slow ones may not
    # cross a session boundary.
    X_tau = smooth_within_legs(FR, leg,
                               cfg.decode_smooth_seconds / table['bin_seconds']).T
    X_T = smooth_within_legs(FR, sid,
                             cfg.decode_smooth_seconds_T / table['bin_seconds']).T

    if n_neurons is not None and n_neurons < X_tau.shape[1]:
        sub = rng.choice(X_tau.shape[1], n_neurons, replace=False)
        X_tau, X_T = X_tau[:, sub], X_T[:, sub]

    legs = np.unique(leg)
    even = np.isin(leg, legs[::2])          # leg-level split for the ceilings

    early = T < 1 / 3
    late = T > 2 / 3
    mid = ~early & ~late

    tau_lo_thr, tau_hi_thr = np.percentile(tau, [33, 67])
    lo = tau < tau_lo_thr
    hi = tau > tau_hi_thr

    # ---- range matching. THIS IS NOT COSMETIC ---------------------------
    # In this dataset leg duration shrinks over the session (Spearman -0.22 to
    # -0.38 against session time, p < 1e-4 on every recday tested), so the
    # marginal distribution of tau ITSELF differs between the early and late
    # thirds -- early legs run to a p90 of ~33 s, late ones to ~20 s. An
    # unmatched early->late split therefore asks the decoder to extrapolate, and
    # reports a behavioural drift as a failure of the neural code. Measured on
    # synthetics injected into a real table, where the true answer is "transfers":
    #
    #                     unmatched   range-matched   ceiling(matched)
    #     sheet_orth        +0.19         +0.82           +0.96
    #     sheet_weber       +0.02         +0.85           +0.95
    #     conjunctive       -0.62         +0.46           +0.93
    #
    # Unmatched, a separable code is indistinguishable from a conjunctive one.
    # Report 1's simulation drew stationary intervals and so never hit this.
    # The same argument applies to T across tau: high-tau samples come from long
    # legs, which are early in the session.
    def _common(mask_a, mask_b, values, pct=(5, 95)):
        if mask_a.sum() < 10 or mask_b.sum() < 10:
            return mask_a, mask_b, (np.nan, np.nan)
        a_lo, a_hi = np.percentile(values[mask_a], pct)
        b_lo, b_hi = np.percentile(values[mask_b], pct)
        w_lo, w_hi = max(a_lo, b_lo), min(a_hi, b_hi)
        if not np.isfinite(w_lo) or w_hi <= w_lo:
            return mask_a, mask_b, (np.nan, np.nan)
        inside = (values >= w_lo) & (values <= w_hi)
        return mask_a & inside, mask_b & inside, (float(w_lo), float(w_hi))

    tau_tr, tau_te, tau_win = _common(early, late, tau)
    T_tr, T_te, T_win = _common(lo, hi, T)
    # the ceilings are computed on the SAME window, so the ratio is comparable
    tau_in = ((tau >= tau_win[0]) & (tau <= tau_win[1])
              if np.isfinite(tau_win[0]) else np.ones_like(early))
    T_in = ((T >= T_win[0]) & (T <= T_win[1])
            if np.isfinite(T_win[0]) else np.ones_like(lo))

    out = {}
    combos = {
        'tau_within':   (mid & even & tau_in, tau, mid & ~even & tau_in, tau, X_tau),
        'tau_across_T': (tau_tr, tau, tau_te, tau, X_tau),
        'T_within':     (lo & even & T_in, T, lo & ~even & T_in, T, X_T),
        'T_across_tau': (T_tr, T, T_te, T, X_T),
    }
    for name, (mtr, ytr, mte, yte, Xd) in combos.items():
        if mtr.sum() < 30 or mte.sum() < 30:
            out[name] = dict(r2=np.nan, mae=np.nan, n_train=int(mtr.sum()),
                             n_test=int(mte.sum()))
            continue
        out[name] = _ridge_score(Xd[mtr], ytr[mtr], Xd[mte], yte[mte], cfg.ridge_alpha)

    # ---- the headline: transfer AS A FRACTION OF ITS OWN CEILING --------
    # Report 1 stresses that the raw transfer R2 means nothing without the
    # within-condition ceiling beside it. The ratio is the comparable quantity
    # across recdays, regions and population sizes.
    def _ratio(t, c):
        rt, rc = out[t]['r2'], out[c]['r2']
        return float(rt / rc) if np.isfinite(rt) and np.isfinite(rc) and rc > 0.05 else np.nan

    out['tau_transfer_ratio'] = _ratio('tau_across_T', 'tau_within')
    out['T_transfer_ratio'] = _ratio('T_across_tau', 'T_within')
    out['n_neurons'] = int(X_tau.shape[1])
    out['tau_thresholds'] = (float(tau_lo_thr), float(tau_hi_thr))
    out['tau_match_window'] = tau_win
    out['T_match_window'] = T_win
    return out


def cross_session_transfer(table, config=None, *, mask=None, n_neurons=None,
                           seed=None):
    """Decode each axis trained on one session and tested on another.

    Not available in report 1's simulation (one session) and the strongest thing
    this dataset adds. Two distinct claims are tested:

      * tau transfers across sessions -> the fast code is session-invariant
      * T transfers across sessions   -> the slow code is a genuine time-in-session
        signal that RESETS, not a monotonic drift. Report 1 section 1 makes this
        the reason the slow axis cannot close: "the state at T recurs at the start
        of every session, which is a jump back to the origin, not a lap".

    Sessions here are different TASKS (`get_sessions_for_glm` dedups to one
    session per unique task), so cross-session transfer is also a cross-task
    generalisation test and inherits the remapping caveat from
    `remapping_rotation_analysis`.

    Returns one row per ordered (train, test) session pair.
    """
    cfg = config or TimeManifoldConfig()
    rng = np.random.default_rng(cfg.random_state if seed is None else seed)

    keep = np.ones(len(table['tau']), bool) if mask is None else np.asarray(mask, bool)
    sid = np.asarray(table['session_id'], float)[keep]
    Tf = np.asarray(table['T_frac'], float)[keep]
    tau = np.asarray(table['tau'], float)[keep]
    leg = np.asarray(table['interval_id'], float)[keep]
    FR = np.asarray(table['FR'], float)[:, keep]
    X_tau = smooth_within_legs(FR, leg,
                               cfg.decode_smooth_seconds / table['bin_seconds']).T
    X_T = smooth_within_legs(FR, sid,
                             cfg.decode_smooth_seconds_T / table['bin_seconds']).T
    if n_neurons is not None and n_neurons < X_tau.shape[1]:
        sub = rng.choice(X_tau.shape[1], n_neurons, replace=False)
        X_tau, X_T = X_tau[:, sub], X_T[:, sub]

    rows = []
    sessions = np.unique(sid)
    for a in sessions:
        for b in sessions:
            if a == b:
                continue
            ma, mb = sid == a, sid == b
            if ma.sum() < 50 or mb.sum() < 50:
                continue
            rows.append(dict(
                train=float(a), test=float(b),
                tau_r2=_ridge_score(X_tau[ma], tau[ma], X_tau[mb], tau[mb],
                                    cfg.ridge_alpha)['r2'],
                T_r2=_ridge_score(X_T[ma], Tf[ma], X_T[mb], Tf[mb],
                                  cfg.ridge_alpha)['r2'],
                n_train=int(ma.sum()), n_test=int(mb.sum()),
            ))
    return rows


# ============================================================================
# E. Subspace geometry (report 1 section 5)
# ============================================================================

def effective_rank(X, tol=1e-6):
    """Number of singular directions carrying real variance.

    Needed because a pure ramp code has exactly rank 1 per axis: asking for 4
    basis vectors returns 3 arbitrary null-space directions and every subsequent
    principal angle is numerical noise. Report 1's own adversarial audit found
    principal angles computed on a rank-deficient basis and corrected them.
    """
    Xc = np.asarray(X, float)
    Xc = Xc - Xc.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0:
        return 1
    return max(1, int(np.sum(sv > tol * sv[0])))


def principal_angles(A, B):
    """A, B: [d, k] bases -> principal angles in degrees (ascending)."""
    Qa = np.linalg.qr(np.asarray(A, float))[0]
    Qb = np.linalg.qr(np.asarray(B, float))[0]
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s))


def _top_components(X, k):
    Xc = np.asarray(X, float)
    Xc = Xc - Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T                       # [N, k]


def axis_geometry(tensor, k=4):
    """Angle between the T and tau coding subspaces, and drift of the tau subspace.

    Two numbers, and together they are what separates the three sheet regimes that
    topology cannot (report 1 section 5):

      angle_T_tau   90 deg -> orthogonal product code
                    ~87    -> gain-modulated
                    ~64    -> conjunctive
      tau_drift     rotation of the tau subspace between the first and last third
                    of the session: 0 deg orthogonal, ~16 gain-modulated,
                    ~90 conjunctive.

    tau_drift is the one worth looking for explicitly: it is exactly the signature
    of a fast code that is MODULATED BY rather than INDEPENDENT OF slow context,
    it is topologically invisible, and decoding transfer is too forgiving to catch
    it (the gain-modulated model still transfers at R2 = 0.87).
    """
    G, _ = fill_tensor(tensor)
    nT, nTau, _ = G.shape
    G = G - G.mean(axis=(0, 1))

    mT = G.mean(1)
    mTau = G.mean(0)
    kk = min(k, effective_rank(mT), effective_rank(mTau))
    ang = principal_angles(_top_components(mT, kk), _top_components(mTau, kk))

    third = max(1, nT // 3)
    early = G[:third].mean(0)
    late = G[-third:].mean(0)
    kd = min(k, effective_rank(early), effective_rank(late))
    drift = principal_angles(_top_components(early, kd), _top_components(late, kd))

    # ---- the null the raw drift angle needs ----------------------------
    # A subspace estimated from a noisy tensor differs from another estimate of
    # THE SAME subspace by a nonzero angle, and how much depends on SNR and on
    # `effective_rank`'s tolerance -- so a raw drift of 44 deg can arise from a
    # model whose true drift is exactly 0. Measured on synthetics injected into a
    # real table: sheet_weber (true drift 0) reads +44, ramps (true 0) reads +89.
    # The null is the same statistic between two INTERLEAVED halves of the same
    # session period, which carries the same noise but no drift; the interpretable
    # quantity is `drift_excess = drift - drift_null`.
    a = G[::2].mean(0)
    b = G[1::2].mean(0)
    kn = min(k, effective_rank(a), effective_rank(b))
    null = principal_angles(_top_components(a, kn), _top_components(b, kn))
    drift_null = float(np.max(null))
    drift_obs = float(np.max(drift))

    return dict(angle_T_tau=float(np.min(ang)), angles_T_tau=ang,
                tau_drift=drift_obs, tau_drift_angles=drift,
                tau_drift_null=drift_null,
                tau_drift_excess=drift_obs - drift_null,
                k_axes=int(kk), k_drift=int(kd),
                tau_binning=tensor['tau_binning'])


# ============================================================================
# F. Topology, done the way report 1 says it must be
# ============================================================================
#
# Two preprocessing choices decide whether ground-truth topology is recovered at
# all, and BOTH failures look like clean negative results rather than errors:
#
#   * Euclidean distance saturates. On a tiled code, states more than a field
#     width apart are all roughly equally far, so long-range structure has to be
#     recovered with kNN-geodesic distance. With raw Euclidean distance report 1's
#     cylinders read as sheets.
#   * Sampling uniformly in seconds is not uniform on the manifold. A Weber-scaled
#     code puts a large neural gap between 100 s and 225 s and almost none between
#     3000 s and 3125 s, so a grid uniform in seconds contains one enormous gap
#     that sets the connectivity scale and buries every real loop.
#
# With both corrections all eight of report 1's models recover correctly; without
# either, they do not. `persistent_homology_analysis` has the geodesic metric as a
# PHConfig option (default 'euclidean') but nothing equivalent to the arclength
# resampling, which is why it lives here.
# ============================================================================

def arclength_resample(curve, n):
    """Resample a [n_pts, N] curve at `n` points spaced equally along its own
    length in neural space.

    Returns (index_positions, resampled_curve). The index positions are in units
    of the original sample index, so they can be mapped back to seconds and the
    non-uniformity of the resampling reported.

    Ported from mohamady_time_idea/time_manifold_code_1/topo.py:_arclength_pick.
    """
    C = np.asarray(curve, float)
    d = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        idx = np.linspace(0, len(C) - 1, n)
        return idx, C[np.round(idx).astype(int)]
    targets = np.linspace(0, s[-1], n)
    idx = np.interp(targets, s, np.arange(len(C), dtype=float))
    out = np.stack([np.interp(idx, np.arange(len(C)), C[:, j])
                    for j in range(C.shape[1])], axis=1)
    return idx, out


def geodesic_matrix(X, k=6, _depth=0):
    """kNN-graph geodesic distance. Grows k until the graph is connected."""
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial.distance import pdist, squareform
    X = np.asarray(X, float)
    n = len(X)
    k = int(min(k, n - 1))
    D = squareform(pdist(X))
    idx = np.argsort(D, 1)[:, 1:k + 1]
    W = np.zeros_like(D)
    rows = np.repeat(np.arange(n), idx.shape[1])
    W[rows, idx.ravel()] = D[rows, idx.ravel()]
    W = np.maximum(W, W.T)
    G = shortest_path(W, method='D', directed=False)
    if not np.isfinite(G).all():
        if _depth > 8 or k >= n - 1:
            raise RuntimeError('geodesic graph will not connect')
        return geodesic_matrix(X, k + 3, _depth + 1)
    return G


def persistence(X, maxdim=1, k=6):
    """Persistent homology under the geodesic metric.

    Bar lifetimes are returned in units of the CONNECTIVITY SCALE (the largest H0
    death, i.e. the radius at which the cloud fuses into one component), so a bar
    longer than 1 is longer than the scale at which the cloud coalesces and the
    number is comparable across clouds of different size and density. Report the
    connectivity scale you normalised by.

    Ported from mohamady_time_idea/time_manifold_code_1/topo.py:ph.
    """
    from ripser import ripser
    G = geodesic_matrix(X, k)
    res = ripser(G, maxdim=maxdim, distance_matrix=True)
    h0 = res['dgms'][0][:, 1]
    conn = float(np.max(h0[np.isfinite(h0)]))
    out = {'conn': conn, 'dgms': res['dgms']}
    for d in range(1, maxdim + 1):
        dg = res['dgms'][d]
        L = (np.sort(dg[:, 1] - dg[:, 0])[::-1] / conn) if len(dg) else np.array([])
        out[d] = L
    return out


def betti(ph_out, d=1, thresh=2.5):
    L = ph_out.get(d, np.array([]))
    return dict(betti=int(np.sum(L > thresh)),
                bars=[float(x) for x in L[:4]],
                top=float(L[0]) if len(L) else 0.0,
                conn=ph_out.get('conn', np.nan))


def marginal_topology(tensor, config=None, axis='tau', maxdim=1):
    """Persistent homology of ONE axis's marginal, arclength-resampled.

    Only the marginal is window-independent (report 1 section 3): the closure
    index reads the ends of the analysis window, so a real ring whose period does
    not divide the window reads open. Report 1's wrapped-clock model is exactly
    that case -- closure -0.40, marginal beta1 = 1.

    Run this on BOTH tensors of a `build_tensor_pair`.
    """
    cfg = config or TimeManifoldConfig()
    G, frac = fill_tensor(tensor)
    curve = G.mean(1) if axis == 'T' else G.mean(0)
    idx, R = arclength_resample(curve, min(cfg.n_arclength, len(curve) * 4))
    ph = persistence(R, maxdim=maxdim, k=cfg.geodesic_k)
    out = betti(ph, 1, cfg.ph_thresh)
    out.update(axis=axis, tau_binning=tensor['tau_binning'],
               n_points=len(R), frac_imputed=frac,
               resample_index=idx, ph=ph)
    return out


def joint_topology(tensor, config=None, maxdim=2):
    """Persistent homology of the joint (T, tau) condition-mean manifold.

    Ground truth: sheet beta1 = 0, cylinder 1, torus 2. Remember report 1's
    ceiling: this cannot distinguish separable from conjunctive codes, which are
    both sheets. Use `factorisation` for that.
    """
    cfg = config or TimeManifoldConfig()
    G, frac = fill_tensor(tensor)
    nT, nTau, N = G.shape
    X = G.reshape(nT * nTau, N)
    ph = persistence(X, maxdim=maxdim, k=cfg.geodesic_k)
    out = betti(ph, 1, cfg.ph_thresh)
    out['H2'] = betti(ph, 2, cfg.ph_thresh) if maxdim >= 2 else None
    out.update(tau_binning=tensor['tau_binning'], frac_imputed=frac,
               n_points=nT * nTau, ph=ph)
    return out


def _resample_unit_for(tau_binning):
    """Which behavioural unit to bootstrap when testing a given fast-axis binning.

    The unit has to CONTAIN the structure under test. Bootstrapping legs while
    testing the loop-phase axis resamples the four legs of a loop independently,
    which scrambles loop membership and corrupts `loop_phase` itself -- the true
    loop ring then reads 3.54 +- 3.55 detected in 4/8 runs (bimodal: half the
    draws recover it perfectly, half destroy it) and is indistinguishable from a
    genuine artefact. With the loop as the unit it is stable.
    """
    return 'loop_id' if tau_binning == 'loop_phase' else 'interval_id'


def h1_stability(table, config=None, *, tau_binning='abs', n_runs=6,
                 n_neurons=None, frac_neurons=0.8, resample_units=True,
                 unit=None, axis='tau', seed=0, **kw):
    """Re-run the topology on independent resamples and report the SPREAD.

    Report 1 finding #2: a spurious ring is "detected" in 4/6 runs at 25 cells and
    0/6 at 400, with H1 = 3.0 +- 2.3 -- the point estimate looks like a ring and
    the run-to-run spread is as large as the effect. More data REMOVES a spurious
    loop rather than sharpening it. Never report a mean H1 without this spread,
    and treat a large spread as evidence AGAINST the feature.

    Both units of variability are resampled, because either alone leaves the
    result artificially stable:
      * neurons -- a subsample, `frac_neurons` of the population. Taking ALL N
        neurons in a random ORDER is not a resample: the condition tensor is
        invariant to neuron order, so every run returns bit-identical bars and
        the reported spread is exactly 0.00 whatever the data.
      * behavioural units -- bootstrapped with replacement. Samples within a leg
        are strongly autocorrelated, so the leg (or, for the loop-phase axis, the
        loop) is the real unit of independence. See `_resample_unit_for`.
    """
    cfg = config or TimeManifoldConfig()
    rng = np.random.default_rng(seed)
    N = table['FR'].shape[0]
    n_sub = (max(4, int(round(frac_neurons * N))) if n_neurons is None
             else min(n_neurons, N))
    if n_sub >= N and not resample_units:
        raise ValueError('nothing is being resampled: lower frac_neurons/n_neurons '
                         'or enable resample_units')

    unit_key = unit or _resample_unit_for(tau_binning)
    uid = np.asarray(table[unit_key], float)
    units = np.unique(uid)
    unit_index = {u: np.flatnonzero(uid == u) for u in units}

    tops, bettis = [], []
    for _ in range(n_runs):
        sub = dict(table)
        sub['FR'] = table['FR'][rng.choice(N, n_sub, replace=False)]
        if resample_units:
            draw = rng.choice(units, len(units), replace=True)
            sel = np.concatenate([unit_index[u] for u in draw])
            for k, v in list(sub.items()):
                if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(uid):
                    sub[k] = v[sel]
            sub['FR'] = sub['FR'][:, sel]
        tens = build_condition_tensor(sub, cfg, tau_binning=tau_binning, **kw)
        try:
            r = (marginal_topology(tens, cfg, axis=axis) if axis in ('tau', 'T')
                 else joint_topology(tens, cfg))
        except (RuntimeError, ValueError):
            continue
        tops.append(r['top']); bettis.append(r['betti'])
    if not tops:
        # Same keys as the success return below. Omitting stable_ring /
        # detection_rate here made `_model_stats` raise KeyError -- outside the try
        # that wraps this call -- on the first template where every resample hit
        # `geodesic graph will not connect`.
        return dict(n_runs=0, n_neurons=n_sub, axis=axis,
                    tau_binning=tau_binning, unit=unit_key,
                    H1_mean=np.nan, H1_sd=np.nan, H1_runs=[],
                    detected=0, detection_rate=np.nan, stable_ring=False)
    det = int(np.sum(np.array(bettis) >= 1))
    return dict(n_runs=len(tops), n_neurons=n_sub, axis=axis,
                tau_binning=tau_binning, unit=unit_key,
                H1_mean=float(np.mean(tops)), H1_sd=float(np.std(tops)),
                H1_runs=[float(x) for x in tops],
                detected=det, detection_rate=det / len(tops),
                # A real ring is large AND stable AND found every time. A spurious
                # one has a moderate mean, a spread comparable to it, and an
                # intermittent detection rate -- report 1 measured 3.0 +- 2.3 in
                # 4/6 runs for exactly that case.
                stable_ring=bool(det == len(tops) and np.mean(tops) > cfg.ph_thresh
                                 and np.std(tops) < 0.25 * max(np.mean(tops), 1e-9)))


# ============================================================================
# G. Compression tests (report 2)
# ============================================================================
#
# Report 2 withdrew its own torus claim after an adversarial audit. Only three
# statistics survived their nulls, and only those three are implemented here.
# ============================================================================

def covariance_spectrum(table, config=None, *, variable='tau', n_bins=24,
                        mask=None, n_shuffle=0, seed=0):
    """Eigenspectrum of the population covariance over one time variable.

    Report 2 section 5: under band-pass input the top eigenvalues come in
    near-degenerate PAIRS (measured ratios 1.026, 1.063, 1.099) against clearly
    split pairs for low-pass input (1.153, 1.23, 1.237). A degenerate pair at
    frequency f is a cosine and its quadrature sine, and the population state
    traces a CIRCLE in the plane they span -- so this is the linear-algebra
    statement of "this axis closes".

    It is the one signature in report 2 that no weight-vector null can reproduce,
    because it is a property of the covariance rather than of any weight vector.
    By the same token it is NOT measurable from tuning curves: it needs the
    population covariance itself.

    Report the whole spectrum and the consecutive ratios, not a single number --
    the claim is a relative one and "1.026 vs 1.153" only means anything as a
    comparison.

    `n_shuffle > 0` adds a leg-shuffled null (permute the leg each sample belongs
    to, preserving the marginal distribution of the binning variable).
    """
    cfg = config or TimeManifoldConfig()
    keep = np.ones(len(table['tau']), bool) if mask is None else np.asarray(mask, bool)

    if variable == 'tau':
        x = np.asarray(table['tau'], float)[keep]
        lo, hi = (cfg.tau_range if cfg.tau_range is not None
                  else (0.0, float(leg_duration_stats(table)['p90'])))
    elif variable == 'phase_leg':
        x = np.asarray(table['phase_leg'], float)[keep]
        lo, hi = 0.0, 1.0
    elif variable == 'T':
        x = np.asarray(table['T_frac'], float)[keep]
        lo, hi = cfg.T_range_frac
    else:
        raise ValueError('variable must be tau, phase_leg or T')

    FR = np.asarray(table['FR'], float)[:, keep]

    def spectrum(F):
        edges = _edges(lo, hi, n_bins)
        idx = np.digitize(x, edges) - 1
        ok = (idx >= 0) & (idx < n_bins)
        M = np.full((n_bins, F.shape[0]), np.nan)
        for b in range(n_bins):
            sel = ok & (idx == b)
            if sel.sum() >= 3:
                M[b] = F[:, sel].mean(1)
        good = np.isfinite(M).all(1)
        if good.sum() < 4:
            return None
        Mg = M[good]
        Mg = (Mg - Mg.mean(0)) / (Mg.std(0) + 1e-9)
        ev = np.linalg.svd(Mg - Mg.mean(0), compute_uv=False) ** 2
        ev = ev[ev > 0]
        return ev / ev[0]

    ev = spectrum(FR)
    if ev is None:
        return dict(eigenvalues=None)
    # consecutive ratios ev[i]/ev[i+1] for i = 0, 2, 4... -> the "pair" ratios
    pairs = [float(ev[i] / ev[i + 1]) for i in range(0, min(len(ev) - 1, 8), 2)]

    null_pairs = None
    if n_shuffle > 0:
        rng = np.random.default_rng(seed)
        leg = np.asarray(table['interval_id'], float)[keep]
        null_pairs = []
        for _ in range(n_shuffle):
            # permute whole legs against the neural data, preserving each leg's
            # internal temporal structure but destroying its alignment to tau
            legs = np.unique(leg)
            perm = {a: b for a, b in zip(legs, rng.permutation(legs))}
            order = np.argsort([perm[l] for l in leg], kind='stable')
            e = spectrum(FR[:, order])
            if e is not None and len(e) > 2:
                null_pairs.append([float(e[i] / e[i + 1])
                                   for i in range(0, min(len(e) - 1, 8), 2)])

    return dict(eigenvalues=ev, pair_ratios=pairs, variable=variable,
                n_bins=int(n_bins), null_pair_ratios=null_pairs,
                interpretation='pair ratio near 1.0 = quadrature pair = closed axis; '
                               'report 2 measured 1.026 band-pass vs 1.153 low-pass')


def fold_population(Z, x, period, n_bins=24):
    """Average the population state over laps at matched phase.

    Ported from mohamady_time_idea/time_manifold_code_part2/compress.py:fold.
    """
    ph = (x % period) / period
    lap = np.floor(x / period).astype(int)
    b = np.clip((ph * n_bins).astype(int), 0, n_bins - 1)
    laps = np.unique(lap)
    out = []
    for l in laps:
        m = np.full((n_bins, Z.shape[1]), np.nan)
        for j in range(n_bins):
            sel = (lap == l) & (b == j)
            if sel.sum():
                m[j] = Z[sel].mean(0)
        if np.isfinite(m).all():
            out.append(m)
    if not out:
        return None
    return np.stack(out)


def phase_consistency(Z, x, period, n_bins=24):
    """Do different laps put the population in the same state at the same phase?

    +1.5 is the ceiling (a perfect grid: same-phase r = 1, shifted r = -0.5);
    0 means the laps are unrelated, i.e. the code is not really periodic.

    THE ONLY tuning-curve statistic in report 2 that survived its nulls, and it
    survives because it has an INTERNAL CONTROL the others lack: fold at a wrong
    period and it must collapse. Report 2 measured +0.58 at the module's own
    period, -0.09 for the same module at a wrong period, and +0.01 for a
    non-periodic code. Always report the wrong-period value alongside
    (`phase_consistency_with_control`).

    Ported from compress.py:phase_consistency.
    """
    Z = np.asarray(Z, float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    laps = fold_population(Z, np.asarray(x, float), period, n_bins)
    if laps is None or len(laps) < 2:
        return np.nan
    same, shifted = [], []
    for i in range(len(laps)):
        for j in range(i + 1, len(laps)):
            same.append(_rowcorr(laps[i], laps[j]).mean())
            shifted.append(np.mean([_rowcorr(laps[i], np.roll(laps[j], k, 0)).mean()
                                    for k in range(n_bins // 4, 3 * n_bins // 4)]))
    return float(np.mean(same) - np.mean(shifted))


def phase_consistency_with_control(Z, x, period, n_bins=24,
                                   wrong_factors=(0.6, 1.5)):
    """Phase consistency at the estimated period AND at deliberately wrong ones.

    Report 2 section 3 -- this pairing IS the test. A real periodicity collapses
    at the wrong period; a folding artefact does not. Reporting the own-period
    number alone measures nothing, because folding any code into a small number of
    phase bins produces a short closed-ish polyline whatever went into it.
    """
    out = {'period': float(period),
           'own': phase_consistency(Z, x, period, n_bins)}
    for f in wrong_factors:
        out[f'wrong_{f:g}x'] = phase_consistency(Z, x, period * f, n_bins)
    wrongs = [v for k, v in out.items() if k.startswith('wrong_') and np.isfinite(v)]
    out['margin'] = float(out['own'] - max(wrongs)) if wrongs and np.isfinite(out['own']) else np.nan
    return out


def lattice_stats(field, extent=(1.0, 1.0)):
    """Grid statistics of a 2-D (T, tau) ratemap, the way the grid literature does.

        gridness = min(r60, r120) - max(r30, r90, r150)

    Hexagonal -> ~+0.5 to +1.4; square lattice or stripes -> negative; a single
    blob -> ~0. Report 2's strongest surviving result, and unlike everything in
    its section 2 it is a statement about the JOINT code rather than about the
    marginal periodicity of single cells.

    Ported from compress.py:lattice_stats.
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import rotate, maximum_filter
    f = np.asarray(field, float)
    f = f - np.nanmean(f)
    f = np.nan_to_num(f)
    A = fftconvolve(f, f[::-1, ::-1], mode='full')
    A /= (np.abs(A).max() + 1e-12)
    ny, nx = A.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)

    rad = np.arange(1, int(min(ny, nx) * 0.5))
    prof = np.array([A[(r >= k - .5) & (r < k + .5)].mean() for k in rad])
    dips = np.where((prof[1:-1] < prof[:-2]) & (prof[1:-1] < prof[2:]))[0]
    r_in = rad[dips[0] + 1] if len(dips) else min(ny, nx) * 0.06

    loc = (A == maximum_filter(A, size=max(3, int(min(ny, nx) * 0.06)))) & (r > r_in)
    ys, xs = np.where(loc)
    if len(ys) == 0:
        return dict(gridness=0.0, n_peaks=0, spacing=np.nan, autocorr=A)
    rr = np.hypot(ys - cy, xs - cx)
    vals = A[ys, xs]
    keep = vals > 0.1 * vals.max()
    rr = rr[keep]
    if len(rr) == 0:
        return dict(gridness=0.0, n_peaks=0, spacing=np.nan, autocorr=A)
    spacing = float(np.median(np.sort(rr)[:6]))
    ring = (r > r_in) & (r < min(1.6 * spacing, min(ny, nx) * 0.48))
    if ring.sum() < 50:
        return dict(gridness=0.0, n_peaks=len(rr), spacing=spacing, autocorr=A)

    def rot_corr(deg):
        B = rotate(A, deg, reshape=False, order=1, mode='nearest')
        a, b = A[ring], B[ring]
        a = a - a.mean(); b = b - b.mean()
        return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    g = min(rot_corr(60), rot_corr(120)) - max(rot_corr(30), rot_corr(90), rot_corr(150))
    n_peaks = int(np.sum(np.abs(np.sort(rr)[:8] - spacing) < 0.35 * spacing))
    return dict(gridness=float(g), n_peaks=n_peaks, spacing=spacing, autocorr=A)


def detectable_period_band(table, config=None):
    """The (T, tau) lattice periods this dataset could actually resolve.

    A lattice period is only detectable between about twice the bin width (below
    which it aliases) and about a third of the axis extent (above which fewer than
    ~3 repeats fit and the autocorrelogram has no ring to find).

    This matters: report 2's simulated lattice has a period of roughly 19 min in
    session time and 14 s in reward time. Our session is ~20 min and our tau axis
    spans ~0-30 s, so BOTH of those periods sit above our upper limit and would be
    invisible here. A null gridness result outside this band is uninformative, not
    a refutation, and must be reported as such.
    """
    cfg = config or TimeManifoldConfig()
    dur = float(np.mean(list(table['session_dur'].values())))
    T_extent = dur * (cfg.T_range_frac[1] - cfg.T_range_frac[0])
    tau_hi = (cfg.tau_range[1] if cfg.tau_range is not None
              else float(leg_duration_stats(table)['p90']))
    T_bin = T_extent / cfg.n_T
    tau_bin = tau_hi / cfg.n_tau
    return dict(T_band=(2 * T_bin, T_extent / 3.0),
                tau_band=(2 * tau_bin, tau_hi / 3.0),
                T_extent=T_extent, tau_extent=tau_hi,
                note='report 2 predicts ~19 min (T) and ~14 s (tau): both ABOVE '
                     'these bands, i.e. undetectable in this dataset')


def random_nonneg_weight_null(basis, n_out, seed=0, sparsity=0.15):
    """Random NON-NEGATIVE sparse weight vectors over a basis -- no learning.

    Report 2 section 2: any statistic this reproduces is a property of the input
    basis, not of the compression. Random weights matched the learned gridness
    (1.21 vs 1.27) and passed the module test in 100% of draws. Run this over a
    basis matched to YOUR OWN cells' measured bandwidth before concluding that
    multi-peaked tuning means periodicity.

    Ported from compress.py:random_weight_null.
    """
    rng = np.random.default_rng(seed)
    n_basis = np.asarray(basis).shape[1]
    J = rng.random((n_out, n_basis))
    J[J > sparsity] = 0.0
    for i in range(n_out):
        if J[i].sum() == 0:
            J[i, rng.integers(n_basis)] = 1.0
    J /= np.linalg.norm(J, axis=1, keepdims=True)
    return J @ np.asarray(basis, float).T          # [n_out, n_points]


# ============================================================================
# H. THE GATE -- synthetic controls with known ground truth
# ============================================================================
#
# Run this before trusting any number on real data. The synthetics are injected
# into a REAL table -- real leg boundaries, real leg-duration distribution, real
# session structure, real occupancy, real neuron count -- and flow through the
# same `build_condition_tensor` / `closure_test` / `factorisation` functions,
# unmodified. Per this repo's standing practice, gating on synthetics driven
# through the real pipeline has caught five plausible-looking design errors.
#
# The eight models are ported from
# mohamady_time_idea/time_manifold_code_1/models.py, re-parameterised to this
# task's timescales. Two more exist only because our task has a third clock:
# `cyl_loop` (tau coded as position in the A->B->C->D->A loop, which genuinely
# closes) and `sheet_3clock` (session + loop + leg, three OPEN axes -- report 1
# predicts extra clocks add dimensions, not loops).
# ============================================================================

# `closes_in` names the binning in which each closed model's fast axis actually
# closes, and it is NOT always 'phase'. `cyl_wrap` wraps at a fixed period in
# ABSOLUTE seconds, so it closes under 'abs' and reads open under 'phase' -- in
# report 1's own table it scores tau_closure -0.40 while its marginal beta1 is 1.
# Demanding that every cylinder close under phase binning is the mistake that
# statement is warning about.
GROUND_TRUTH = {
    'sheet_orth':   dict(topology='sheet',    b1=0, additive=1.0, drift=0,   closes_in=None),
    'sheet_weber':  dict(topology='sheet',    b1=0, additive=1.0, drift=0,   closes_in=None),
    'conjunctive':  dict(topology='sheet',    b1=0, additive=0.5, drift=90,  closes_in=None),
    'gain_mod':     dict(topology='sheet',    b1=0, additive=1.0, drift=16,  closes_in=None),
    'ramps':        dict(topology='sheet',    b1=0, additive=1.0, drift=0,   closes_in=None),
    'cyl_phase':    dict(topology='cylinder', b1=1, additive=1.0, drift=0,   closes_in='phase'),
    # cyl_wrap is report 1's documented blind spot for the closure index, and the
    # gate ASSERTS that blindness rather than working around it. Its 25 s period
    # does not divide the analysis window, so the two ends of the window are not
    # the same phase and the index reads open (report 1 measured -0.40) even
    # though the axis genuinely closes -- the marginal beta1 is 1. This is the
    # concrete demonstration that the closure index tests the ends of the WINDOW,
    # not the topology of the axis.
    'cyl_wrap':     dict(topology='cylinder', b1=1, additive=1.0, drift=0,
                         closes_in='abs', closure_index_blind=True),
    # cyl_loop codes position in the A->A cycle, so it closes in 'loop_phase' and
    # is INVISIBLE under 'phase': binning by within-leg phase averages over the
    # four legs of the loop and destroys the ring. That is not a failure of the
    # test, it is the same lesson as report 1 finding #1 -- the binning variable
    # has to be the one the code actually uses.
    'cyl_loop':     dict(topology='cylinder', b1=1, additive=1.0, drift=0,   closes_in='loop_phase'),
    'torus':        dict(topology='torus',    b1=2, additive=1.0, drift=0,   closes_in='phase'),
    'sheet_3clock': dict(topology='sheet',    b1=0, additive=None, drift=0,  closes_in=None),
}

SYNTHETIC_KINDS = tuple(GROUND_TRUTH)


def _gauss(x, c, s):
    return np.exp(-0.5 * ((np.asarray(x, float)[:, None] - c[None, :]) / s[None, :]) ** 2)


def _vonmises(phi, c, kappa):
    """phi, c in [0, 1) cycles."""
    return np.exp(kappa * (np.cos(2 * np.pi * (np.asarray(phi, float)[:, None]
                                               - c[None, :])) - 1))


def _log_centres(lo, hi, n):
    return np.exp(np.linspace(np.log(max(lo, 1e-6)), np.log(hi), n))


def make_synthetic_table(kind, template, config=None, *, n_neurons=None,
                         peak_hz=12.0, seed=0):
    """Replace a real table's FR with a synthetic code of known ground truth.

    Everything except the firing rates is the real recording: the same legs, the
    same leg-duration distribution, the same session boundaries, the same
    occupancy of the (T, tau) plane. That is the point -- a synthetic that invents
    its own behaviour cannot catch a bug that only fires on real sampling.
    """
    cfg = config or TimeManifoldConfig()
    rng = np.random.default_rng(seed)

    T = np.asarray(template['T_sec'], float)
    tau = np.asarray(template['tau'], float)
    phase = np.asarray(template['phase_leg'], float)
    dur = float(np.mean(list(template['session_dur'].values())))
    Tn = np.clip(T / max(dur, 1e-9), 0, 1)
    tau_max = float(np.percentile(tau, 95))
    N = int(template['FR'].shape[0] if n_neurons is None else n_neurons)
    nT, nTau = N // 2, N - N // 2

    # Loop phase (our third clock) is taken from the table, NOT recomputed here.
    # A synthetic generated from its own private copy of a variable cannot catch a
    # bug in the copy the analysis uses -- it would silently test agreement
    # between two implementations rather than the pipeline.
    loop_phase = np.asarray(template['loop_phase'], float)

    # shared parameter draws, so models stay comparable
    cT_lin = np.sort(rng.uniform(0, dur, nT))
    cTau_lin = np.sort(rng.uniform(0, tau_max, nTau))
    cT_log = np.sort(_log_centres(dur / 40, dur, nT) * rng.uniform(.85, 1.15, nT))
    cTau_log = np.sort(_log_centres(0.4, tau_max, nTau) * rng.uniform(.85, 1.15, nTau))
    ampT = rng.gamma(3, 1 / 3, nT)
    ampTau = rng.gamma(3, 1 / 3, nTau)
    sT_fix = np.full(nT, dur / 16)
    sTau_fix = np.full(nTau, tau_max / 14)
    sT_web = 0.22 * cT_log + dur / 144
    sTau_web = 0.22 * cTau_log + 0.6
    # Circular tuning centres are JITTERED-UNIFORM, not uniformly drawn.
    # Report 1 draws them from rng.uniform with 200 cells per axis, where gaps are
    # vanishingly unlikely. We have ~75, and a pure uniform draw then leaves a
    # visible hole in phase coverage often enough to matter: across six seeds the
    # cyl_phase ring measured H1 = 6.6, 6.3, 2.5, 6.9, 6.4, 4.4 -- i.e. one seed
    # in three produced a code that does not actually tile the circle, so its
    # "ring" was legitimately weak. That is a defect in the generator, not in the
    # analysis: a model whose ground truth is a ring has to cover the ring.
    cPh = (np.linspace(0, 1, nTau, endpoint=False)
           + rng.normal(0, 0.3 / nTau, nTau)) % 1.0
    cPh = np.sort(cPh)
    kap = np.full(nTau, 12.0)

    if kind == 'sheet_orth':
        R = np.hstack([_gauss(T, cT_lin, sT_fix) * ampT,
                       _gauss(tau, cTau_lin, sTau_fix) * ampTau])
    elif kind == 'sheet_weber':
        R = np.hstack([_gauss(T, cT_log, sT_web) * ampT,
                       _gauss(tau, cTau_log, sTau_web) * ampTau])
    elif kind == 'conjunctive':
        g = int(np.ceil(np.sqrt(N)))
        cc_T = np.repeat(np.linspace(0, dur, g), g)[:N] + rng.normal(0, dur / 40, N)
        cc_tau = np.tile(_log_centres(0.4, tau_max, g), g)[:N] * rng.uniform(.85, 1.15, N)
        ss_T = 0.22 * np.clip(cc_T, dur / 120, None) + dur / 60
        ss_tau = 0.22 * cc_tau + 0.8
        R = _gauss(T, cc_T, ss_T) * _gauss(tau, cc_tau, ss_tau) * rng.gamma(3, 1 / 3, N)
    elif kind == 'gain_mod':
        gainT = rng.uniform(-1, 1, nTau)
        b = _gauss(tau, cTau_log, sTau_web) * ampTau
        R = np.hstack([_gauss(T, cT_log, sT_web) * ampT,
                       b * (1 + 1.2 * gainT[None, :] * Tn[:, None])])
    elif kind == 'ramps':
        wT = rng.normal(0, 1, (1, nT)); wTau = rng.normal(0, 1, (1, nTau))
        offT = rng.uniform(.5, 1.5, nT); offTau = rng.uniform(.5, 1.5, nTau)
        R = np.hstack([offT + Tn[:, None] * wT,
                       offTau + (tau / tau_max)[:, None] * wTau]) * 0.5 + 0.5
    elif kind == 'cyl_phase':
        R = np.hstack([_gauss(T, cT_log, sT_web) * ampT,
                       _vonmises(phase, cPh, kap) * ampTau])
    elif kind == 'cyl_wrap':
        period = tau_max / 1.8
        R = np.hstack([_gauss(T, cT_log, sT_web) * ampT,
                       _vonmises((tau % period) / period, cPh, kap) * ampTau])
    elif kind == 'cyl_loop':
        R = np.hstack([_gauss(T, cT_log, sT_web) * ampT,
                       _vonmises(loop_phase, cPh, kap) * ampTau])
    elif kind == 'torus':
        cTs = np.sort((np.linspace(0, 1, nT, endpoint=False)
                       + rng.normal(0, 0.3 / nT, nT)) % 1.0)
        R = np.hstack([_vonmises(Tn, cTs, np.full(nT, 12.0)) * ampT,
                       _vonmises(phase, cPh, kap) * ampTau])
    elif kind == 'sheet_3clock':
        n3 = N // 3
        rest = N - 2 * n3
        cL = np.sort(rng.uniform(0, 1, rest))
        R = np.hstack([
            _gauss(T, cT_log[:n3], sT_web[:n3]) * ampT[:n3],
            _gauss(tau, cTau_log[:n3], sTau_web[:n3]) * ampTau[:n3],
            _gauss(loop_phase, cL, np.full(rest, 0.08)) * rng.gamma(3, 1 / 3, rest),
        ])
    else:
        raise ValueError(f'unknown synthetic kind {kind!r}; '
                         f'choose from {SYNTHETIC_KINDS}')

    dt = template['bin_seconds']
    lam = np.clip(R, 0, None) * peak_hz * dt
    out = dict(template)
    out['FR'] = rng.poisson(lam).astype(float).T          # [N, n_samples]
    out['synthetic_kind'] = kind
    out['ground_truth'] = GROUND_TRUTH[kind]
    return out


def _model_stats(kind, template, config, seed, run_topology, stability_runs):
    """All statistics for one synthetic model. Split out so the gate can compare
    models against each other rather than against hard-coded magnitudes."""
    cfg = config
    syn = make_synthetic_table(kind, template, cfg, seed=seed)
    pair = build_tensor_pair(syn, cfg, binnings=FAST_AXIS_BINNINGS)
    cl = {b: closure_test(pair[b], cfg) for b in pair}
    add = additive_r2(syn, cfg)
    geo = axis_geometry(pair['abs'])
    fac = factorisation(syn, cfg)

    stab = {}
    if run_topology:
        for b in FAST_AXIS_BINNINGS:
            # ONLY genuine per-run numerical failures may be absorbed into a NaN.
            # A bare `except Exception` here silently rewrote a missing `ripser`
            # install as H1_mean=nan for every model, which made `_sheet_ceiling`
            # nan, which made every "H1 clears every sheet" check read FAILED --
            # i.e. an absent dependency was reported as a scientific verdict. If
            # the topology cannot run at all, that must surface as a crash.
            try:
                stab[b] = h1_stability(syn, cfg, tau_binning=b, axis='tau',
                                       n_runs=stability_runs, seed=seed + 100)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                stab[b] = dict(H1_mean=np.nan, H1_sd=np.nan, detection_rate=np.nan,
                               stable_ring=False)

    row = dict(kind=kind, truth=GROUND_TRUTH[kind]['topology'],
               closes_in=GROUND_TRUTH[kind]['closes_in'],
               closure_T=cl['abs']['T_closure'],
               additive_index=add['additive_index'],
               additive_ceiling=add['r2_full'],
               angle_T_tau=geo['angle_T_tau'],
               tau_drift=geo['tau_drift'],
               tau_drift_excess=geo['tau_drift_excess'],
               tau_transfer=fac['tau_across_T']['r2'],
               tau_ceiling=fac['tau_within']['r2'],
               tau_transfer_ratio=fac['tau_transfer_ratio'],
               T_transfer_ratio=fac['T_transfer_ratio'])
    for b in FAST_AXIS_BINNINGS:
        row[f'closure_tau_{b}'] = cl[b]['tau_closure']
        # The closure denominator, recorded so a NaN index can be attributed to a
        # flattened axis rather than read as "does not close" (section 8.9).
        row[f'local_structure_{b}'] = cl[b]['tau_local_structure']
        if run_topology:
            row[f'H1_{b}'] = stab[b]['H1_mean']
            row[f'H1sd_{b}'] = stab[b]['H1_sd']
            row[f'stable_ring_{b}'] = stab[b]['stable_ring']
    return row


def run_synthetic_controls(template, config=None, *, kinds=None, seed=0,
                           verbose=True, run_topology=True, stability_runs=8):
    """THE GATE. Every statistic must recover ground truth before real numbers count.

    The checks are deliberately written as CONTRASTS BETWEEN MODELS rather than as
    thresholds on absolute values. Two reasons, both learned by watching an
    earlier threshold-based version pass on one recday and fail on the next:

      * our timescales are not report 1's, so his magnitudes are not the target;
        what has to survive the port is which model scores higher than which.
      * an absolute cut has to be tuned, and a gate tuned until it passes is not a
        gate. The conjunctive transfer ratio measured 0.54 / 0.08 / 0.65 on three
        real templates against 0.90-1.07 for the separable codes: no fixed cut is
        both safe and sensitive, but "conjunctive is below every separable model"
        holds on all three.

    Contrasts asserted (each must hold within a single run):

      additive index      separable, gain-modulated > conjunctive
      transfer ratio      separable, gain-modulated > conjunctive
      tau drift excess    conjunctive > separable
      closure             each closed model closes in ITS OWN binning
                          (`closes_in`) and no sheet closes in any binning
      topology            each closed model gives a stable ring in its own
                          binning; no sheet gives a stable ring anywhere

    Per-model absolute checks are kept only where the sign is the whole claim
    (the T axis must not close; `cyl_wrap`'s closure index must be blind).

    `stability_runs` sets the resampling depth of `h1_stability`. A SINGLE-RUN
    beta1 is deliberately not used anywhere: on real templates it flipped between
    0 and 1 for the same true ring across recdays, which is report 1's finding #2
    and the reason `h1_stability` exists.

    THREE OUTCOMES, NOT TWO. A check that could not be measured on this template
    is recorded in `not_evaluable` with its reason, never as a failure. Section 8.8
    is the cautionary tale (a missing `ripser` reported as a claim about tori) and
    8.6 the rule it violates ("treat a NaN ratio as 'the decoder does not work
    here', never as a weak effect"). `passed` is over the EVALUATED checks;
    `fully_evaluated` says whether anything was skipped. Read both.
    """
    cfg = config or TimeManifoldConfig()
    kinds = list(SYNTHETIC_KINDS if kinds is None else kinds)
    rows = [_model_stats(k, template, cfg, seed + i, run_topology, stability_runs)
            for i, k in enumerate(kinds)]
    by = {r['kind']: r for r in rows}

    n_loops = len(np.unique(template['loop_id']))
    loop_powered = n_loops >= cfg.min_loops_for_loop_topology

    SEPARABLE = [k for k in ('sheet_orth', 'sheet_weber', 'gain_mod') if k in by]
    CONJ = 'conjunctive' if 'conjunctive' in by else None

    checks = {}
    not_evaluable = {}

    def _lt(a, b):
        """a < b, or None if either side is undefined.

        A NaN is NOT a failed contrast. `factorisation._ratio` returns NaN exactly
        when the within-condition ceiling is non-positive, and section 8.6 is
        explicit: "Treat a NaN ratio as 'the decoder does not work here', never as
        a weak effect." Scoring it as a failed ordering is the same error as
        scoring a missing ripser as a failed topology check -- an un-measured
        quantity reported as a verdict. Measured on ah08_20250616: the conjunctive
        model's own tau ceiling was R2 = -56.3, so nothing could be compared, yet
        the check read FAILED.
        """
        if not (np.isfinite(a) and np.isfinite(b)):
            return None
        return a < b

    def _all_hold(results):
        """False if any comparison genuinely failed; None if none failed but some
        could not be evaluated; True only if every one held."""
        results = list(results)
        if any(r is False for r in results):
            return False
        if any(r is None for r in results):
            return None
        return True

    def _record(name, outcome, reason=''):
        if outcome is None:
            not_evaluable[name] = reason
        else:
            checks[name] = bool(outcome)

    if CONJ and SEPARABLE:
        _record('additive index: conjunctive < every separable',
                _all_hold(_lt(by[CONJ]['additive_index'], by[s]['additive_index'])
                          for s in SEPARABLE),
                'additive_index undefined for at least one model')
        _bad_ceiling = [f"{k} ceiling R2={by[k]['tau_ceiling']:+.3f}"
                        for k in [CONJ] + SEPARABLE
                        if not np.isfinite(by[k]['tau_transfer_ratio'])]
        _record('transfer ratio: conjunctive < every separable',
                _all_hold(_lt(by[CONJ]['tau_transfer_ratio'],
                              by[s]['tau_transfer_ratio']) for s in SEPARABLE),
                'decoder does not work on this template (section 8.6): '
                + '; '.join(_bad_ceiling))
        _record('tau drift: conjunctive > separable sheets',
                _all_hold(_lt(by[s]['tau_drift_excess'], by[CONJ]['tau_drift_excess'])
                          for s in ('sheet_orth', 'sheet_weber') if s in by),
                'tau_drift_excess undefined for at least one model')

    SHEETS = [k for k in by if GROUND_TRUTH[k]['closes_in'] is None]

    def _sheet_ceiling(binning):
        """Largest H1 any SHEET produces under this binning -- the bar a genuine
        ring has to clear. Using this rather than an absolute cut is the same
        argument as everywhere else in this gate, and it is also the only version
        that survived three real templates: a true ring resampled with leg or loop
        bootstrap routinely reads e.g. 5.2 +- 1.5 with 7/8 detection, which is not
        'stable' by a strict definition but is 5 sigma clear of every sheet."""
        vals = [by[s].get(f'H1_{binning}', np.nan) for s in SHEETS]
        vals = [v for v in vals if np.isfinite(v)]
        return max(vals) if vals else np.nan

    # "The topology never ran" and "the ring failed to clear the sheets" are
    # completely different verdicts and must not print the same way. With no finite
    # sheet H1 there is no ceiling, `_sheet_ceiling` returns nan, and every
    # `H1 under X clears every sheet` check silently reads FAILED -- which is how a
    # missing `ripser` install came to look like a result about cylinders and tori.
    # A sheet with no ring reads 0.0; nan everywhere means nothing was computed.
    if run_topology and not any(np.isfinite(_sheet_ceiling(b))
                                for b in FAST_AXIS_BINNINGS):
        raise RuntimeError(
            'topology was requested but not one sheet model produced a finite H1 in '
            'any binning, so there is no ceiling for a genuine ring to clear and '
            'every topology check would report FAILED for a reason that is not '
            'scientific. Check that `ripser` is importable in this kernel (the '
            'usual cause -- it lives in the `maze_ephys` env, not `maze_ephys_si104`), '
            'then that `geodesic_matrix` can connect. Pass run_topology=False to run '
            'the non-topological checks alone.')

    for r in rows:
        k, closes = r['kind'], r['closes_in']
        gt = GROUND_TRUTH[k]
        if closes is None:
            checks[f'{k}: does not close in any binning'] = all(
                not (np.isfinite(r[f'closure_tau_{b}']) and r[f'closure_tau_{b}'] > 0.5)
                for b in FAST_AXIS_BINNINGS)
            if run_topology:
                # no sheet may produce a ring that is BOTH large and stable
                checks[f'{k}: no stable ring in any binning'] = all(
                    not r.get(f'stable_ring_{b}', False)
                    for b in FAST_AXIS_BINNINGS
                    if b != 'loop_phase' or loop_powered)
        elif gt.get('closure_index_blind'):
            checks[f'{k}: closure index is blind (documented)'] = not (
                np.isfinite(r[f'closure_tau_{closes}']) and r[f'closure_tau_{closes}'] > 0.5)
        else:
            # A NaN index here means the axis was too flat under this binning for
            # the index to be defined -- not that the model failed to close. Same
            # rule as `_lt`: an un-measured quantity is not a verdict.
            _c, _d = r[f'closure_tau_{closes}'], r[f'local_structure_{closes}']
            _record(f'{k}: closes under {closes}',
                    None if not np.isfinite(_c) else bool(_c > 0.5),
                    f'closure index undefined under {closes}: local structure '
                    f'{_d:.3f} <= min_local_structure {cfg.min_local_structure}')
        if closes is not None and run_topology and (closes != 'loop_phase' or loop_powered):
            ceil = _sheet_ceiling(closes)
            checks[f'{k}: H1 under {closes} clears every sheet'] = bool(
                np.isfinite(r.get(f'H1_{closes}', np.nan)) and np.isfinite(ceil)
                and r[f'H1_{closes}'] > max(2.0 * ceil, cfg.ph_thresh))
        if k != 'torus':
            checks[f'{k}: T axis does not close'] = (
                not np.isfinite(r['closure_T']) or r['closure_T'] < 0.5)

    # `passed` is over the checks that could actually be EVALUATED. A contrast that
    # could not be measured on this template is reported separately rather than
    # counted as a failure -- but `fully_evaluated` says so, and the caller should
    # read both. Never let an un-measured quantity silently pass OR fail.
    passed = all(checks.values())
    fully_evaluated = not not_evaluable

    if verbose:
        hdr = (f"{'model':13s} {'truth':9s} "
               f"{'clos abs/ph/loop':>24s} {'addIdx':>7s} {'drift':>6s} "
               f"{'transR':>7s}")
        if run_topology:
            hdr += f" {'H1 abs/ph/loop':>22s}"
        print(hdr)
        for r in rows:
            line = (f"{r['kind']:13s} {r['truth']:9s} "
                    f"{_f(r['closure_tau_abs']):>7s}/{_f(r['closure_tau_phase']):>7s}"
                    f"/{_f(r['closure_tau_loop_phase']):>7s} "
                    f"{_f(r['additive_index']):>7s} "
                    f"{_f(r['tau_drift_excess'], 0):>6s} "
                    f"{_f(r['tau_transfer_ratio']):>7s}")
            if run_topology:
                line += ('  ' + '/'.join(
                    ('%.1f%s' % (r.get(f'H1_{b}', np.nan),
                                 '*' if r.get(f'stable_ring_{b}') else ' '))
                    for b in FAST_AXIS_BINNINGS))
            print(line)
        print(f"\n  n_loops = {n_loops}; loop-phase topology "
              f"{'powered' if loop_powered else 'UNDERPOWERED -> not asserted'} "
              f"(needs >= {cfg.min_loops_for_loop_topology})")
        print('  (* = stable ring: detected in every resample, sd < 25% of mean)\n')
        for name, ok in checks.items():
            if not ok:
                print(f"  FAILED: {name}")
        for name, why in not_evaluable.items():
            print(f"  NOT EVALUABLE: {name}\n                 {why}")
        print(f"\n{'GATE PASSES' if passed else 'GATE FAILS'} "
              f"({sum(checks.values())}/{len(checks)} checks evaluated"
              + (f", {len(not_evaluable)} NOT EVALUABLE)" if not_evaluable else ')'))

    return dict(rows=rows, checks=checks, passed=passed, n_loops=n_loops,
                loop_powered=loop_powered, not_evaluable=not_evaluable,
                fully_evaluated=fully_evaluated)


def _f(x, nd=2):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:+.{nd}f}'

# ============================================================================
# I. Plotting
# ============================================================================
#
# Every plotting helper here takes ALL fast-axis binnings rather than one, so a
# figure cannot accidentally make the single-binning claim report 1 warns
# against. Styling follows the gridmaze-plotter convention: call
# `glm_analysis_v2.apply_gridmaze_style()` once at the top of the notebook.
# ============================================================================

def plot_reset_curves(table, config=None, binnings=FAST_AXIS_BINNINGS,
                      axes=None, show_untrimmed=True):
    """The headline figure: population correlation with the post-reward state.

    One panel per binning, because the CONTRAST between them is the result. A
    closed axis comes back up; an open one decays to a plateau and stays. The
    plateau height is not the diagnostic -- a tiled code shares a mean, so the
    floor is arbitrary; the shape is what matters.

    The trimmed and untrimmed curves are drawn together: tau ~ 0 sits inside the
    reward window, so a recovery that exists only in the untrimmed curve is a
    reward transient rather than a coding result.
    """
    import matplotlib.pyplot as plt
    cfg = config or TimeManifoldConfig()
    binnings = list(binnings)
    if axes is None:
        _, axes = plt.subplots(1, len(binnings),
                               figsize=(2.6 * len(binnings), 2.4), squeeze=False)
        axes = np.ravel(axes)
    axes = np.ravel(axes)

    trimmed = trim_mask(table, config=cfg)
    for ax, b in zip(axes, binnings):
        if show_untrimmed:
            c0 = reset_return_curve(table, cfg, binning=b)
            ax.plot(c0['centres'], c0['corr'], color=NULL_GREY, lw=0.8,
                    ls='--', label='untrimmed')
        c = reset_return_curve(table, cfg, binning=b, mask=trimmed)
        s = return_curve_shape(c)
        ax.plot(c['centres'], c['corr'], color=BINNING_COLORS.get(b, NEUTRAL),
                lw=1.4, label=f'trim {cfg.trim_seconds:g}s')
        ax.axhline(0, color='0.7', lw=0.5)
        ax.set_title(f'{b}  (recovery {s["recovery"]:+.2f})')
        ax.set_xlabel('time since reward (s)' if b == 'abs' else f'{b} (fraction)')
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel('corr with state at $\\tau\\approx0$')
    return axes


def plot_closure_summary(table, config=None, ax=None):
    """Closure index of both axes under every binning, with the window annotated.

    The window annotation is not decoration: the index reads the ends of the
    analysis window, not the topology of the axis, and a real ring whose period
    does not divide the window reads open (report 1's wrapped-clock row, and the
    `cyl_wrap` gate model).
    """
    import matplotlib.pyplot as plt
    cfg = config or TimeManifoldConfig()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 2.4))
    res = {b: closure_test(build_condition_tensor(table, cfg, tau_binning=b), cfg)
           for b in FAST_AXIS_BINNINGS}
    x = np.arange(len(FAST_AXIS_BINNINGS))
    w = 0.38
    ax.bar(x - w / 2, [res[b]['tau_closure'] for b in FAST_AXIS_BINNINGS], w,
           color=AXIS_COLORS['tau'], label='fast axis')
    ax.bar(x + w / 2, [res[b]['T_closure'] for b in FAST_AXIS_BINNINGS], w,
           color=AXIS_COLORS['T'], label='session time')
    ax.axhline(1.0, ls='--', color='0.4', lw=0.7)
    ax.axhline(0.0, color='0.7', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n{res[b]['tau_window'][0]:.0f}-{res[b]['tau_window'][1]:.0f}"
                        for b in FAST_AXIS_BINNINGS])
    ax.set_ylabel('closure index')
    ax.set_title('1 = ends meet; reads the WINDOW, not the axis')
    ax.legend(frameon=False, fontsize=6)
    return ax, res


def plot_factorisation(results_by_recday, ax=None):
    """Transfer-as-fraction-of-ceiling, one point per recday.

    Report 1 stresses that a raw transfer R2 means nothing without its
    within-condition ceiling, and that a single-session estimate of the
    conjunctive case has a spread wide enough to be meaningless alone -- so this
    plots the distribution across recdays, not one number.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 2.4))
    tau = [r['tau_transfer_ratio'] for r in results_by_recday.values()]
    T = [r['T_transfer_ratio'] for r in results_by_recday.values()]
    for i, (vals, key) in enumerate(((tau, 'tau'), (T, 'T'))):
        v = np.array([x for x in vals if np.isfinite(x)])
        ax.scatter(np.full(len(v), i) + np.random.uniform(-.08, .08, len(v)), v,
                   s=12, color=AXIS_COLORS[key], alpha=0.75, edgecolors='none')
        if len(v):
            ax.hlines(np.median(v), i - .2, i + .2, color=NEUTRAL, lw=1.2)
    ax.axhline(1.0, ls='--', color='0.4', lw=0.7)
    ax.axhline(0.0, color='0.7', lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['$\\tau$ across session', 'T across $\\tau$'])
    ax.set_ylabel('transfer / within-condition ceiling')
    return ax


def plot_h1_stability(stab_by_binning, ax=None, config=None):
    """H1 bar length per resample. The SPREAD is the point, not the mean.

    Report 1 finding #2: a spurious ring reads 3.0 +- 2.3 detected in 4/6 runs --
    the point estimate looks like a ring and the spread is as large as the effect.
    Plotting the individual runs makes that visible; plotting a mean hides it.
    """
    import matplotlib.pyplot as plt
    cfg = config or TimeManifoldConfig()
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 2.4))
    for i, (b, s) in enumerate(stab_by_binning.items()):
        runs = s.get('H1_runs', [])
        ax.scatter(np.full(len(runs), i) + np.random.uniform(-.1, .1, len(runs)),
                   runs, s=14, color=BINNING_COLORS.get(b, NEUTRAL),
                   alpha=0.8, edgecolors='none')
        if runs:
            ax.hlines(np.mean(runs), i - .22, i + .22, color=NEUTRAL, lw=1.2)
            if s.get('stable_ring'):
                ax.text(i, max(runs) * 1.05, '*', ha='center', fontsize=11)
    ax.axhline(cfg.ph_thresh, ls='--', color='0.4', lw=0.7)
    ax.set_xticks(range(len(stab_by_binning)))
    ax.set_xticklabels(list(stab_by_binning))
    ax.set_ylabel('$H_1$ bar / connectivity scale')
    ax.set_title('each dot = one resample')
    return ax


def plot_occupancy(tensor, ax=None):
    """Samples per (T, tau) condition cell. The high-tau corner is always thin --
    long legs are rare -- and how thin decides how much of the tensor is imputed."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 2.4))
    im = ax.imshow(tensor['occupancy'].T, origin='lower', aspect='auto', cmap='Blues')
    ax.set_xlabel('session time (bin)')
    ax.set_ylabel(f"fast axis ({tensor['tau_binning']}) bin")
    ax.set_title(f"occupancy; {(~tensor['valid']).sum()} cells below threshold")
    plt.colorbar(im, ax=ax, label='samples')
    return ax


# ============================================================================
# J. Per-neuron 2-D rate maps over (session time x reward time)
# ============================================================================
#
# Descriptive, deliberately. No reliability statistic, no null, no significance
# -- these are for looking at the data before the geometry statistics are
# interpreted.
#
# TWO VARIANTS, and the pairing is the point:
#
#   'normalised'  fraction of session   x  fraction of the goal->goal transition
#   'absolute'    elapsed seconds       x  elapsed seconds since last reward
#
# The normalised pair aligns unequal sessions and unequal legs; the absolute pair
# is in real units. Comparing them is report 1's "bin the fast axis both ways"
# applied to both axes at once: a field that is sharp in the normalised map and
# smeared in the absolute one is tracking phase, and vice versa.
#
# THE THING TO KNOW BEFORE READING ANY OF THESE MAPS. Firing rates here are low,
# and LEC's are very low:
#
#     region   p10      median    p90
#     LEC      0.02 Hz  0.10 Hz   0.66 Hz
#     PFC      0.10 Hz  0.50 Hz   1.62 Hz
#
# On a 20x15 grid a cell holds 7.5-14 s of data, so a MEDIAN LEC NEURON
# CONTRIBUTES UNDER ONE SPIKE PER BIN (a median PFC neuron contributes ~10).
# Most single LEC maps are therefore Poisson noise, and nothing in this section
# distinguishes noise from a field -- that is what a split-half or shuffle would
# do, and it is deliberately not computed here. Read these as a first look, not
# as evidence.
# ============================================================================

RATEMAP_VARIANTS = {
    'normalised': dict(T_binning='frac', tau_binning='phase',
                       T_label='fraction of session',
                       fast_label='fraction of goal-goal transition'),
    'absolute':   dict(T_binning='sec', tau_binning='abs',
                       T_label='time in session (min)',
                       fast_label='time since reward (s)'),
}


def _smooth_occupancy_aware(spike_sums, occupancy, sigma_bins):
    """2-D Gaussian smoothing of a rate map that respects occupancy.

    Smooths the SUMMED SPIKES and the OCCUPANCY with the same kernel and divides,
    rather than smoothing the rate map itself. Smoothing a rate map directly
    treats an unvisited cell as if it had a rate, so empty corners bleed
    structure inward and the result invents fields where the animal never was.
    This is the standard place-field construction and it matters more than usual
    here, because at these rates the maps are mostly empty.

    Cells with no occupancy after smoothing stay NaN.
    """
    from scipy.ndimage import gaussian_filter
    if sigma_bins <= 0:
        with np.errstate(invalid='ignore', divide='ignore'):
            out = spike_sums / occupancy[..., None]
        out[occupancy == 0] = np.nan
        return out
    occ_s = gaussian_filter(occupancy.astype(float), sigma_bins, mode='nearest')
    num = np.stack([gaussian_filter(spike_sums[..., i], sigma_bins, mode='nearest')
                    for i in range(spike_sums.shape[-1])], axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / occ_s[..., None]
    out[occ_s <= 1e-9] = np.nan
    # a cell the animal genuinely never visited stays blank even if the kernel
    # leaked a little mass into it
    out[occupancy == 0] = np.nan
    return out


def neuron_ratemaps(table, config=None, *, variant='normalised',
                    smooth_bins=1.0, mask=None, sessions=None):
    """2-D rate map per neuron over (session time, reward time), in Hz.

    Pools every session of the recday. NOTE that sessions are different TASKS
    (`get_sessions_for_glm` dedups to one session per unique task), so pooling
    averages over the place<->state remapping `remapping_rotation_analysis`
    measures. For a pure time code that is what you want; if a neuron's time
    tuning is task-specific it will wash out. Pass `sessions=[s]` for a
    per-session version.

    Returns
    -------
    dict with
      maps        [N, n_T, n_fast] SMOOTHED firing rate in Hz, NaN where unvisited
      maps_raw    [N, n_T, n_fast] the same maps with no smoothing at all
      occupancy   [n_T, n_fast] samples per cell
      seconds     [n_T, n_fast] occupancy in seconds
      T_centres, fast_centres, T_label, fast_label, variant
      mean_hz, peak_hz   [N] per neuron

    `maps_raw` is returned alongside deliberately: at these rates the smoothed map
    is the only readable one, but it is also the one that can invent structure, so
    the raw map has to be visible next to it rather than a step the reader has to
    take on trust. Smoothed Poisson noise has a median CV of 0.22 on this grid
    against 0.58 for real LEC data -- structured, but not as structured.
    """
    cfg = config or TimeManifoldConfig()
    if variant not in RATEMAP_VARIANTS:
        raise ValueError(f'variant must be one of {list(RATEMAP_VARIANTS)}')
    spec = RATEMAP_VARIANTS[variant]

    # min_samples_per_cell=1 -> keep every visited cell; the occupancy map is
    # returned so thin cells stay visible rather than being silently dropped.
    cfg_rm = TimeManifoldConfig(**{**cfg.to_dict(), 'min_samples_per_cell': 1})
    tens = build_condition_tensor(table, cfg_rm, tau_binning=spec['tau_binning'],
                                  T_binning=spec['T_binning'], mask=mask,
                                  sessions=sessions, zscore=False)

    occ = tens['occupancy']
    dt = table['bin_seconds']
    # M is mean counts per bin; recover the summed counts the smoother needs
    sums = np.nan_to_num(tens['M']) * occ[..., None]
    rate = _smooth_occupancy_aware(sums, occ, smooth_bins) / dt   # counts/bin -> Hz
    rate_raw = _smooth_occupancy_aware(sums, occ, 0.0) / dt

    maps = np.moveaxis(rate, -1, 0)                               # [N, n_T, n_fast]
    maps_raw = np.moveaxis(rate_raw, -1, 0)

    # The `absolute` variant is drawn in MINUTES on the slow axis (a 0-1022 s
    # tick sequence is unreadable), so the window and the bin width convert too --
    # everything downstream must be in one consistent unit or the 2-sigma marker
    # lands in the wrong place.
    T_centres = tens['T_centres']
    T_window = tens['T_window']
    if variant == 'absolute':
        T_centres = T_centres / 60.0
        T_window = (T_window[0] / 60.0, T_window[1] / 60.0)
    fast_window = tens['tau_window']

    n_T, n_fast = maps.shape[1], maps.shape[2]
    T_bin = (T_window[1] - T_window[0]) / n_T
    fast_bin = (fast_window[1] - fast_window[0]) / n_fast

    FR = np.asarray(table['FR'], float)
    if mask is not None:
        FR = FR[:, np.asarray(mask, bool)]
    with warnings_suppressed():
        peak = np.nanmax(maps.reshape(len(maps), -1), axis=1)

    return dict(maps=maps, maps_raw=maps_raw, occupancy=occ, seconds=occ * dt,
                T_centres=T_centres, fast_centres=tens['tau_centres'],
                T_label=spec['T_label'], fast_label=spec['fast_label'],
                variant=variant, smooth_bins=smooth_bins,
                T_window=T_window, fast_window=fast_window,
                T_bin=T_bin, fast_bin=fast_bin,
                mean_hz=FR.mean(1) / dt, peak_hz=peak,
                n_neurons=int(len(maps)))


class warnings_suppressed:
    """All-NaN slices are expected (a neuron may have no visited cell in a thin
    corner); silence just that warning rather than globally."""
    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *a):
        self._c.__exit__(*a)


def build_ratemaps_all(tables, config=None, *, variant='normalised',
                       smooth_bins=1.0, mask_fn=None, autocorr=True,
                       verbose=True):
    """Rate maps for every neuron of every recday, stacked into one array.

    `mask_fn(table, config) -> bool array` optionally restricts samples
    (e.g. `trim_mask`). Recdays whose grids disagree are an error, not a silent
    skip -- a mismatched grid would otherwise be concatenated into a stack whose
    axes mean different things for different neurons.

    Returns
    -------
    out : dict with `maps` (smoothed), `maps_raw`, `autocorr` and `autocorr_raw`
          (or None), `meta` (DataFrame: recday, neuron, mean_hz, peak_hz) and
          `axes` (centres, labels, windows, bin widths and `smooth_bins`).
    """
    import pandas as pd
    cfg = config or TimeManifoldConfig()
    sm_stacks, raw_stacks, rows, axes = [], [], [], None

    for rd, t in tables.items():
        m = mask_fn(t, cfg) if mask_fn is not None else None
        rm = neuron_ratemaps(t, cfg, variant=variant, smooth_bins=smooth_bins,
                             mask=m)
        if axes is None:
            # `smooth_bins`, the windows and the bin widths travel with the axes:
            # the 2-sigma kernel marker has to be drawn in DATA coordinates, and
            # sigma is only meaningful once you know how wide a bin is on each
            # axis. Without them a caller can only draw a circle in axes-fraction
            # coordinates, which is wrong on both axes at once.
            axes = {k: rm[k] for k in ('T_centres', 'fast_centres', 'T_label',
                                       'fast_label', 'variant', 'smooth_bins',
                                       'T_window', 'fast_window',
                                       'T_bin', 'fast_bin')}
        elif rm['maps'].shape[1:] != sm_stacks[0].shape[1:]:
            raise ValueError(f'{rd}: grid {rm["maps"].shape[1:]} disagrees with '
                             f'{sm_stacks[0].shape[1:]}')
        sm_stacks.append(rm['maps'])
        raw_stacks.append(rm['maps_raw'])
        for i in range(rm['n_neurons']):
            rows.append(dict(recday=rd, neuron=i,
                             mean_hz=float(rm['mean_hz'][i]),
                             peak_hz=float(rm['peak_hz'][i])))
        if verbose:
            print(f'  {rd}: {rm["n_neurons"]} neurons, '
                  f'grid {rm["maps"].shape[1]}x{rm["maps"].shape[2]}, '
                  f'median {np.median(rm["seconds"]):.1f} s/cell')

    if not sm_stacks:
        return dict(maps=np.empty((0, 0, 0)), maps_raw=np.empty((0, 0, 0)),
                    autocorr=None, autocorr_raw=None,
                    meta=pd.DataFrame(), axes={})

    maps = np.concatenate(sm_stacks, 0).astype(np.float32)
    maps_raw = np.concatenate(raw_stacks, 0).astype(np.float32)
    # Both autocorrelograms, always. The smoothed one is the readable panel but a
    # Gaussian of sigma s imposes its own autocorrelation, so the raw one is what
    # says whether the structure is the neuron or the kernel.
    ac = ratemap_autocorr_stack(maps) if autocorr else None
    ac_raw = ratemap_autocorr_stack(maps_raw) if autocorr else None
    return dict(maps=maps, maps_raw=maps_raw, autocorr=ac, autocorr_raw=ac_raw,
                meta=pd.DataFrame(rows), axes=axes)


# ---------------------------------------------------------------- plotting

def _map_extent(axes):
    """imshow extent for a rate map, from the OUTER BIN EDGES.

    `imshow(extent=...)` wants the outer edges of the image, not the first and
    last bin centres. Passing centres shifts the whole map by half a bin on both
    axes -- small, but it is the difference between "this field peaks at 4 min"
    and "at 4.4 min", and it silently offsets the autocorrelogram's zero lag too.
    """
    T0, T1 = axes['T_window']
    f0, f1 = axes['fast_window']
    return [T0, T1, f0, f1]


def _lag_extent(axes, n_T, n_fast):
    """imshow extent for an autocorrelogram, in LAG units on both axes.

    An autocorrelogram of an [n_T, n_fast] map is [2*n_T-1, 2*n_fast-1] and spans
    +-(n-1) bins of lag. Without this the panel is drawn in pixel indices and no
    period can be read off it at all.
    """
    dT, df = axes['T_bin'], axes['fast_bin']
    return [-(n_T - 1) * dT, (n_T - 1) * dT,
            -(n_fast - 1) * df, (n_fast - 1) * df]


def _kernel_ellipse(ax, axes, n_sigma=2.0, color=NEUTRAL):
    """Mark the lags where the smoothing kernel dominates the autocorrelogram.

    A Gaussian kernel of width sigma has an autocorrelation that is itself
    Gaussian with std sqrt(2)*sigma, i.e. proportional to exp(-d^2 / 4 sigma^2).
    At d = 2 sigma that has fallen to e^-1 = 0.37 of the centre; at 3 sigma it is
    0.105. So the drawn contour is the e^-1 point of the kernel's OWN
    autocorrelation: structure inside it is substantially the smoothing, and only
    structure outside it can be attributed to the neuron.

    It is an ELLIPSE, not a circle. sigma is specified in bins, and the two axes
    have different bin widths -- at the 20x15 `absolute` default one bin is
    0.85 min on the slow axis and 1.99 s on the fast one, so 2 sigma is 1.70 min
    by 3.97 s. Drawing a circle (or drawing in axes-fraction coordinates) is wrong
    on both axes at once.

    Returns the semi-axes actually drawn, so a test can assert them.
    """
    from matplotlib.patches import Ellipse
    sig = axes.get('smooth_bins', 0.0)
    if not sig or sig <= 0:
        return None
    a = n_sigma * sig * axes['T_bin']          # semi-axis, slow axis
    b = n_sigma * sig * axes['fast_bin']       # semi-axis, fast axis
    ax.add_patch(Ellipse((0.0, 0.0), 2 * a, 2 * b, fill=False, lw=0.5,
                         ls='--', edgecolor=color, zorder=5))
    return a, b


def _ticks(ax, lo, hi, which='x', n=3, fontsize=4.5, label=None):
    """Two or three ticks with real units, small enough to fit a 1-inch panel."""
    vals = np.linspace(lo, hi, n)
    vals = vals + 0.0                      # normalise -0.0 -> 0.0, else a tick
    vals[np.abs(vals) < 1e-12] = 0.0       # at the origin renders as "-0"
    rng = abs(hi - lo)
    fmt = (lambda v: f'{v:.0f}') if rng >= 10 else (
        (lambda v: f'{v:.1f}') if rng >= 1 else (lambda v: f'{v:.2f}'))
    if which == 'x':
        ax.set_xticks(vals)
        ax.set_xticklabels([fmt(v) for v in vals], fontsize=fontsize)
        if label:
            ax.set_xlabel(label, fontsize=5, labelpad=1)
    else:
        ax.set_yticks(vals)
        ax.set_yticklabels([fmt(v) for v in vals], fontsize=fontsize)
        if label:
            ax.set_ylabel(label, fontsize=5, labelpad=1)
    ax.tick_params(which='both', length=1.5, width=0.4, pad=1)


def plot_ratemap_grid(maps, meta, axes, *, page=0, nrows=6, ncols=8,
                      normalise='peak', cmap='viridis', out_path=None,
                      panel_in=1.05, note=True):
    """A page of per-neuron (T, tau) rate maps, for scanning the population.

    Each panel is peak-normalised with its own peak rate printed in the title, so
    a 0.05 Hz cell and a 3 Hz cell are both readable but cannot be confused --
    without the printed rate a peak-normalised grid makes every neuron look
    equally active. `normalise='hz'` shares one absolute colour scale instead.

    `viridis` per the gridmaze-colors skill: this is an occupancy-normalised
    non-diverging scalar. `RdBu_r` is for z-scored maps only.

    Only the bottom-left panel carries ticks. Every panel shares the same axes, so
    ticking all 48 would be noise -- but ticking none (the previous behaviour)
    leaves no frame of reference for minutes or seconds anywhere on the page.

    Returns the Figure.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    per = nrows * ncols
    sel = np.arange(page * per, min((page + 1) * per, len(maps)))
    if len(sel) == 0:
        raise IndexError(f'page {page} is past the end ({len(maps)} neurons)')

    vmax_shared = np.nanpercentile(maps[sel], 99) if normalise == 'hz' else None
    ext = _map_extent(axes)

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(ncols * panel_in, nrows * panel_in + 0.6))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.set_axis_off()

    corner = (nrows - 1) * ncols          # bottom-left panel index
    for k, idx in enumerate(sel):
        ax = axs[k]
        ax.set_axis_on()
        m = maps[idx]
        pk = np.nanmax(m) if np.isfinite(m).any() else np.nan
        if normalise == 'peak':
            shown, vmin, vmax = (m / pk if pk and np.isfinite(pk) and pk > 0
                                 else m), 0.0, 1.0
        else:
            shown, vmin, vmax = m, 0.0, vmax_shared
        # .T so the fast axis is vertical and session time runs left->right
        ax.imshow(shown.T, origin='lower', aspect='auto', cmap=cmap,
                  vmin=vmin, vmax=vmax, extent=ext, interpolation='nearest')
        r = meta.iloc[idx]
        ax.set_title(f"{r['recday'][:6]}#{int(r['neuron'])}  {pk:.2f}Hz",
                     fontsize=5, pad=1.5)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        if k == corner:
            _ticks(ax, ext[0], ext[1], 'x', label=axes['T_label'])
            _ticks(ax, ext[2], ext[3], 'y', label=axes['fast_label'])
        else:
            ax.set_xticks([]); ax.set_yticks([])

    lo, hi = sel[0], sel[-1]
    title = (f"{axes['variant']} rate maps  |  x = {axes['T_label']},  "
             f"y = {axes['fast_label']}  |  neurons {lo}-{hi} of {len(maps)}")
    if note:
        title += ('\nper-panel peak-normalised, peak Hz in title; '
                  'no reliability measure computed - many LEC maps will be Poisson noise')
    fig.suptitle(title, fontsize=6, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))

    if out_path:
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                             'pdf.fonttype': 42, 'ps.fonttype': 42}):
            fig.savefig(out_path, bbox_inches=None)
    return fig


def plot_ratemap_summary(rm, meta=None, axes=None, fig=None):
    """Occupancy of the (T, tau) plane, and where peaks sit on each axis.

    The occupancy panel is the one to read first: it says which parts of the map
    are supported by data at all. The high-tau / late-session corner is always
    thin, because long legs are rare.
    """
    import matplotlib.pyplot as plt
    if fig is None:
        fig, _ = plt.subplots(1, 3, figsize=(8.4, 2.5))
    axs = np.atleast_1d(fig.axes).ravel()

    ext = [rm['T_centres'][0], rm['T_centres'][-1],
           rm['fast_centres'][0], rm['fast_centres'][-1]]
    im = axs[0].imshow(rm['seconds'].T, origin='lower', aspect='auto',
                       cmap='viridis', extent=ext, interpolation='nearest')
    axs[0].set_xlabel(rm['T_label']); axs[0].set_ylabel(rm['fast_label'])
    axs[0].set_title('occupancy (s per cell)')
    fig.colorbar(im, ax=axs[0])

    maps = rm['maps']
    with warnings_suppressed():
        flat = maps.reshape(len(maps), -1)
        good = np.isfinite(flat).any(1)
        arg = np.nanargmax(np.where(np.isfinite(flat), flat, -np.inf)[good], axis=1)
    iT, ifast = np.unravel_index(arg, maps.shape[1:])
    axs[1].hist(rm['T_centres'][iT], bins=len(rm['T_centres']),
                color=AXIS_COLORS['T'])
    axs[1].set_xlabel(rm['T_label']); axs[1].set_ylabel('neurons')
    axs[1].set_title('peak location, slow axis')
    axs[2].hist(rm['fast_centres'][ifast], bins=len(rm['fast_centres']),
                color=AXIS_COLORS['tau'])
    axs[2].set_xlabel(rm['fast_label'])
    axs[2].set_title('peak location, fast axis')
    for ax in axs[1:]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def ratemap_autocorr(m):
    """2-D autocorrelogram of one (T, tau) rate map.

    Standard construction: mean-centre, correlate the map with itself by FFT,
    normalise the peak to 1. Unvisited cells become 0 AFTER centring, i.e. they
    contribute nothing rather than contributing a spurious low value.

    Returns a [2*n_T-1, 2*n_fast-1] array. The centre is the zero-lag peak; a
    repeating field shows a ring of side peaks around it, and report 2's lattice
    prediction is six of them.

    This is the same autocorrelation `lattice_stats` computes internally; use
    `lattice_stats(m)` when the gridness score and peak count are wanted too.
    """
    from scipy.signal import fftconvolve
    f = np.asarray(m, float)
    n_T, n_f = f.shape
    # a neuron can be silent, or a map can be entirely unvisited; guard before
    # nanmean rather than after, or numpy warns on the empty slice
    if not np.isfinite(f).any():
        return np.zeros((2 * n_T - 1, 2 * n_f - 1))
    f = np.nan_to_num(f - np.nanmean(f))
    if not np.any(f):
        return np.zeros((2 * n_T - 1, 2 * n_f - 1))
    A = fftconvolve(f, f[::-1, ::-1], mode='full')
    return A / (np.abs(A).max() + 1e-12)


def ratemap_autocorr_stack(maps):
    """`ratemap_autocorr` over a [N, n_T, n_fast] stack -> [N, 2nT-1, 2nf-1]."""
    return np.stack([ratemap_autocorr(m) for m in maps]).astype(np.float32)


def plot_ratemap_panels(rm_or_maps, meta=None, axes=None, *, maps_raw=None,
                        autocorr=None, autocorr_raw=None, page=0, n_neurons=6,
                        out_path=None, panel_in=1.25, gridness=False,
                        n_sigma=2.0, note=True):
    """Per neuron, one row: raw map | smoothed map | autocorr(raw) | autocorr(smoothed).

    Four panels, and the pairing is the argument. At LEC rates the raw map is
    close to unreadable and the smoothed one is the only interpretable version --
    but smoothing is also what can invent a field, so each map sits beside its own
    autocorrelogram and the kernel's contribution is visible by comparison rather
    than taken on trust. The raw autocorrelogram will often be close to a delta at
    zero lag plus noise; that IS the honest picture at 0.75 spikes per bin.

    The dashed ellipse on the smoothed autocorrelogram marks `n_sigma` times the
    smoothing width (see `_kernel_ellipse`): at 2 sigma the kernel's own
    autocorrelation has fallen to e^-1 of the centre, so structure inside it is
    substantially the kernel and only structure outside it can be the neuron.

    Accepts either a `neuron_ratemaps` dict as the first argument, or explicit
    `maps` + `maps_raw` (+ optional precomputed autocorrelograms) for the pooled
    multi-recday case.

    `gridness=True` prints a gridness score on the smoothed autocorrelogram. It is
    UNCALIBRATED: report 2 section 2 found random non-negative weights on a
    band-pass basis matching learned gridness (1.21 vs 1.27), so a number here
    describes the map -- it is not evidence of periodicity, and nothing in this
    function computes the null that would make it evidence.

    Returns the Figure.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if isinstance(rm_or_maps, dict):
        rm = rm_or_maps
        maps, maps_raw = rm['maps'], rm['maps_raw']
        axes = axes or {k: rm[k] for k in
                        ('T_centres', 'fast_centres', 'T_label', 'fast_label',
                         'variant', 'smooth_bins', 'T_window', 'fast_window',
                         'T_bin', 'fast_bin')}
    else:
        maps = rm_or_maps
    if maps_raw is None:
        raise ValueError('need maps_raw (pass a neuron_ratemaps dict or the stack)')

    sel = np.arange(page * n_neurons, min((page + 1) * n_neurons, len(maps)))
    if len(sel) == 0:
        raise IndexError(f'page {page} is past the end ({len(maps)} neurons)')

    n_T, n_fast = maps.shape[1], maps.shape[2]
    ext_map = _map_extent(axes)
    ext_lag = _lag_extent(axes, n_T, n_fast)

    # lag-axis names, in the units the variant is drawn in
    if axes['variant'] == 'absolute':
        lx, ly = 'lag (min)', 'lag (s)'
    else:
        lx, ly = 'lag (frac session)', 'lag (frac leg)'
    # The axes themselves are now ticked and labelled on every row, so the header
    # no longer repeats the ranges -- it only names each panel.
    heads = ('raw', 'smoothed', 'autocorr (raw)',
             f'autocorr (smoothed)\ndashed = {n_sigma:g}$\\sigma$ kernel')

    # wider gutters than the default: every row now carries y ticks + a y label
    # on two of its four columns, and the neuron id sits outside column 0
    fig, axs = plt.subplots(len(sel), 4,
                            figsize=(4 * panel_in + 1.9,
                                     len(sel) * panel_in + 1.1),
                            squeeze=False)
    fig.subplots_adjust(wspace=0.42, hspace=0.18)
    last = len(sel) - 1

    for r, idx in enumerate(sel):
        raw, sm = maps_raw[idx], maps[idx]
        ac_s = (autocorr[idx] if autocorr is not None else ratemap_autocorr(sm))
        ac_r = (autocorr_raw[idx] if autocorr_raw is not None
                else ratemap_autocorr(raw))
        pk_raw = np.nanmax(raw) if np.isfinite(raw).any() else np.nan
        pk_sm = np.nanmax(sm) if np.isfinite(sm).any() else np.nan

        panels = (
            (raw, ext_map, 'viridis', 0, pk_raw, f'{pk_raw:.2f}Hz'),
            (sm, ext_map, 'viridis', 0, pk_sm, f'{pk_sm:.2f}Hz'),
            (ac_r, ext_lag, 'RdBu_r', -1, 1, ''),
            (ac_s, ext_lag, 'RdBu_r', -1, 1, ''),
        )
        for c, (img, e, cm, vmin, vmax, sub) in enumerate(panels):
            ax = axs[r, c]
            ax.imshow(np.asarray(img).T, origin='lower', aspect='auto', cmap=cm,
                      vmin=vmin, vmax=vmax, extent=e, interpolation='nearest')
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if c == 3:
                _kernel_ellipse(ax, axes, n_sigma=n_sigma)
                if gridness:
                    try:
                        sub = f"g={lattice_stats(sm)['gridness']:+.2f}"
                    except Exception:
                        sub = ''
            if r == 0:
                ax.set_title(heads[c], fontsize=5, pad=2.5)
            if sub:
                # boxed, because white-on-viridis is unreadable wherever the map
                # is bright and this text carries the absolute rate
                ax.text(0.03, 0.95, sub, transform=ax.transAxes, fontsize=4.5,
                        va='top', ha='left', color='w',
                        bbox=dict(facecolor='0.15', edgecolor='none',
                                  alpha=0.65, pad=0.8))
            # y ticks on EVERY row of the two leading columns; x ticks on the
            # bottom row only. Ticking just the bottom row (the previous
            # behaviour) leaves five of six rows with no frame of reference at
            # all -- you have to scan to the foot of the page to find out what
            # the vertical axis of the panel you are looking at means.
            is_map = c < 2
            if c in (0, 2):
                _ticks(ax, e[2], e[3], 'y',
                       label=(axes['fast_label'] if is_map else ly))
            else:
                ax.set_yticks([])
            if r == last:
                _ticks(ax, e[0], e[1], 'x',
                       label=(axes['T_label'] if is_map else lx))
            else:
                ax.set_xticks([])

        if meta is not None:
            # Neuron id as free text OUTSIDE the axes, never `set_ylabel` --
            # set_ylabel on column 0 silently overwrites the y-axis name, which
            # is how the bottom-left map ended up ticked 0/15/30 with nothing
            # saying those were seconds since reward.
            row = meta.iloc[idx]
            axs[r, 0].text(-0.62, 0.5, f"{row['recday'][:6]}\n#{int(row['neuron'])}",
                           transform=axs[r, 0].transAxes, fontsize=5,
                           ha='right', va='center')

    lo, hi = sel[0], sel[-1]
    title = (f"{axes['variant']}  |  maps: x = {axes['T_label']}, "
             f"y = {axes['fast_label']}  |  autocorr: lag on both axes"
             f"  |  neurons {lo}-{hi} of {len(maps)}")
    if note:
        title += ('\nraw beside smoothed because smoothing creates structure; '
                  'dashed ellipse = kernel-dominated lags; no null computed')
    fig.suptitle(title, fontsize=6, y=0.998)
    # no tight_layout: it fights the explicit gutters set above and squeezes the
    # y labels back out. Leave room for the suptitle and the neuron-id column.
    fig.subplots_adjust(top=0.90 if len(sel) > 3 else 0.82,
                        left=0.17, right=0.985, bottom=0.09)

    if out_path:
        with mpl.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0.0,
                             'pdf.fonttype': 42, 'ps.fonttype': 42}):
            fig.savefig(out_path, bbox_inches=None)
    return fig
