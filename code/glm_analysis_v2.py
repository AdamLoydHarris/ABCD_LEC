"""
GLM analysis with no-intercept, keep-all-bins parameterization.

Sibling of glm_analysis.py. Differences:
  1. No intercept column.
  2. One-hot encodings keep ALL bins (no reference-bin dropping):
     - place: 21 bins (locs 1..21)
     - head direction: 36 bins (10 deg bins)
     - goal progress: 10 bins (configurable via gp_n_bins; e.g. 20 to
       column-match the joint 20-col time_any / distance_any groups)
     - speed, acc, time/distance from/to reward: 10 decile bins each
     Total design matrix: 127 columns (137 when gp_n_bins=20).
  3. Design matrix is rank-deficient by 8 (each one-hot block sums to 1 per row,
     so without an intercept the blocks have 8 independent linear dependencies).
     np.linalg.lstsq returns the minimum-norm solution; individual betas are no
     longer uniquely identified, but predictions and RSS_full / RSS_reduced are
     invariant to the parameterization (the column space is the same). CPD, R²,
     and ΔR² therefore match the conventional reference-coded fit exactly. The
     F-statistic itself differs by a constant df-scaling because the default
     uses raw block sizes for df_num and T − 127 for df_resid (vs block_size−1
     and T − 119 in reference coding); but the permutation-based significance
     classification is essentially identical. Use parameterization='reference_coded'
     in run_glm_analysis to verify empirically.

Original glm_analysis.py is unchanged.
"""

import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


def _glm_cv():
    """Import `glm_cv` from this file's directory.

    Imported lazily and by path, like `_recday_registry`, so the mFC copy of this module --
    which sits beside no `glm_cv.py` -- degrades at the call site instead of failing at
    import time.
    """
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import glm_cv
    return glm_cv


# ============================================================================
# Task state / kinematic helpers
# ============================================================================

def compute_task_state_arrays(trial_times, num_bins=10):
    """Per-timepoint task state, goal progress, and time-to/from-reward.

    `trial_times` is (n_trials, num_states + 1); `trial_times[r, s]` is the time
    the animal is AT goal `s` on trial `r`, so **state s is the leg FROM goal s
    TO goal s+1** and spans `[trial_times[r, s], trial_times[r, s+1])`. The last
    column repeats the next row's first column (goal A of the following trial),
    which is why iteration is over (row, column) pairs rather than over the
    flattened+sorted boundary list.

    NOTE (fixed 2026-07-30): the previous implementation walked
    `np.sort(trial_times.flatten())` and assigned `state = i % num_states`, but
    `i` advanced even when the duplicated boundary `trial_times[r, -1] ==
    trial_times[r+1, 0]` was skipped. That rotated trial r's state labels by
    +r (mod num_states), mislabeling 72.9% of legs. Indexing by column instead
    makes the state label correct by construction. The goal-progress and
    time-to/from-reward outputs are unchanged (they never used `i`).
    """
    max_time = int(np.max(trial_times))
    num_states = trial_times.shape[1] - 1
    tt = trial_times.astype(int)

    state_array = np.zeros(max_time + 1, dtype=int)
    goal_progress_array = np.zeros(max_time + 1, dtype=float)
    goal_progress_binned = np.zeros(max_time + 1, dtype=int)
    time_from_last_reward = np.zeros(max_time + 1, dtype=int)
    time_to_next_reward = np.zeros(max_time + 1, dtype=int)

    for r in range(tt.shape[0]):
        for s in range(num_states):
            start_time = int(tt[r, s])
            end_time = int(tt[r, s + 1])
            if end_time <= start_time:
                continue
            state_array[start_time:end_time] = s
            time_range = np.arange(start_time, end_time)
            progress = (time_range - start_time) / (end_time - start_time)
            goal_progress_array[start_time:end_time] = progress
            goal_progress_binned[start_time:end_time] = np.floor(progress * num_bins).astype(int)
            time_from_last_reward[start_time:end_time] = time_range - start_time
            time_to_next_reward[start_time:end_time] = end_time - time_range

    # After the final boundary (an A collection — the last column is goal A of
    # the next trial), the animal is on the A→B leg, i.e. state 0.
    last_time = int(np.max(tt))
    state_array[last_time:] = 0
    goal_progress_array[last_time:] = 1.0
    goal_progress_binned[last_time:] = num_bins - 1
    time_from_last_reward[last_time:] = np.arange(0, max_time - last_time + 1)
    time_to_next_reward[last_time:] = 0

    return state_array, goal_progress_array, goal_progress_binned, time_from_last_reward, time_to_next_reward


def compute_since_A_arrays(trial_times, num_bins=10):
    """Compute time/progress variables anchored to reward A (state 0) collection.

    Unlike compute_task_state_arrays, which anchors to ANY reward boundary and
    resets 4× per loop, this anchors to state 0 (A→B leg) start only, testing
    whole-loop-scale time/progress. Returns time since last A (unbounded, decile-
    binned), time until next A (unbounded, decile-binned), and loop progress
    fraction since A (equal-width binned like goal_progress).
    """
    max_time = int(np.max(trial_times))

    time_since_A = np.zeros(max_time + 1, dtype=int)
    time_to_A = np.zeros(max_time + 1, dtype=int)
    progress_since_A_array = np.zeros(max_time + 1, dtype=float)
    progress_since_A_binned = np.zeros(max_time + 1, dtype=int)

    # Anchor only to reward A. Column 0 is goal A of this trial and the LAST
    # column is goal A of the next trial (they coincide: trial_times[r, -1] ==
    # trial_times[r+1, 0]). Taking column 0 alone therefore misses the final A
    # of the session, which sent the whole last loop into the after-last-A
    # fallback below (constant time_to_A=0, constant progress=num_bins-1) —
    # 9.7% of LEC timepoints. Union both columns and deduplicate.
    A_times = np.unique(
        np.concatenate([trial_times[:, 0], trial_times[:, -1]]).astype(int)
    )
    A_times = A_times[A_times >= 0]  # Guard against negative timestamps

    if len(A_times) == 0:
        # No valid A times; return zero-filled arrays.
        return time_since_A, time_to_A, progress_since_A_binned

    # For each inter-A interval, compute time/progress.
    for i in range(len(A_times) - 1):
        start_time = int(A_times[i])
        end_time = int(A_times[i + 1])
        if start_time == end_time:
            continue
        loop_duration = end_time - start_time
        time_range = np.arange(start_time, end_time)
        progress = (time_range - start_time) / loop_duration
        time_since_A[start_time:end_time] = time_range - start_time
        time_to_A[start_time:end_time] = end_time - time_range
        progress_since_A_array[start_time:end_time] = progress
        progress_since_A_binned[start_time:end_time] = np.floor(progress * num_bins).astype(int)

    # After the last A: time since keeps counting, time_to_A stays 0, progress stays 1.
    last_A_time = int(A_times[-1])
    time_since_A[last_A_time:] = np.arange(0, max_time - last_A_time + 1)
    time_to_A[last_A_time:] = 0
    progress_since_A_array[last_A_time:] = 1.0
    progress_since_A_binned[last_A_time:] = num_bins - 1

    return time_since_A, time_to_A, progress_since_A_binned


# ---------------------------------------------------------------------------
# Nose-poke occupancy (reward consumption vs reward timing)
# ---------------------------------------------------------------------------
#
# Poke event tables come from `code/preprocessing/extract_pokes.py`, one per
# session, columns [entry_bin, exit_bin, port, rewarded, state]. They are
# already on the same 25 ms grid and first-A_on origin as Neuron_raw and
# Trial_times, so no conversion is needed here.
#
# WHY THIS EXISTS: cells that look time-locked to reward (a peak around 2.5 s)
# may instead be reward-CONSUMPTION cells firing while the animal is in the
# port. Median rewarded-poke duration is 2.58 s, so the two hypotheses predict
# almost the same peri-reward response. Note that a rewarded poke's entry bin
# IS the reward bin (96.8% identical, 99.99% within one bin) — the state
# machine advances while the animal is already in the port — so `poke_rewarded`
# is collinear with the early `time_from_reward` bins BY CONSTRUCTION and its
# CPD is an arbitrary split of shared variance. The regressor is worth fitting,
# but the analysis that actually separates the hypotheses is the poke-DURATION
# split (`run_poke_duration_split`), which exploits the 3.5x spread in
# consumption-bout length.

_POKE_MAX_DURATION_BINS = 400   # 10 s; longer "pokes" are imputed-exit artifacts


def compute_poke_arrays(pokes, n_bins, max_duration_bins=_POKE_MAX_DURATION_BINS):
    """Per-timepoint in-port indicators, split by whether the poke was rewarded.

    Parameters
    ----------
    pokes : (n_pokes, 5) int array or None
        [entry_bin, exit_bin, port, rewarded, state] from `extract_pokes.py`.
        `exit_bin` is INCLUSIVE. May be empty — 9 of 191 LEC sessions have no
        completed trials and ship a legitimate (0, 5) table.
    n_bins : int
        Length of the returned arrays. Pass `Neuron_raw.shape[1]`, NOT
        `len(Locs_raw)`: tracking is not truncated to the neural recording and
        runs 200-4400 bins longer.
    max_duration_bins : int
        Drop pokes longer than this (default 400 = 10 s). 317 pokes dataset-wide
        exceed it, up to 646 s; they come from dangling entries whose exit was
        imputed at session end, and would otherwise paint huge blocks of the
        session as "in port".

    Returns
    -------
    poke_rewarded, poke_unrewarded : float 1-D arrays of length `n_bins`
        1.0 while the animal is inside a port on a rewarded / unrewarded poke.
        Overlapping or nested pokes (1.3% of same-port intervals) simply OR
        together. Returned 1-D deliberately — see `attach_pokes` for why a 2-D
        occupancy matrix must never enter the prepared-session dict.
    """
    poke_rewarded = np.zeros(n_bins, dtype=float)
    poke_unrewarded = np.zeros(n_bins, dtype=float)

    if pokes is None:
        return poke_rewarded, poke_unrewarded
    pokes = np.asarray(pokes)
    if pokes.size == 0:
        return poke_rewarded, poke_unrewarded
    if pokes.ndim != 2 or pokes.shape[1] < 4:
        raise ValueError(f"poke table must be (n_pokes, >=4), got {pokes.shape}")

    entry = pokes[:, 0].astype(int)
    exit_ = pokes[:, 1].astype(int)
    rewarded = pokes[:, 3].astype(int)

    # exit_bin is inclusive -> the occupied half-open span is [entry, exit + 1)
    start = np.clip(entry, 0, n_bins)
    stop = np.clip(exit_ + 1, 0, n_bins)

    duration = exit_ - entry + 1
    keep = (stop > start) & (duration <= max_duration_bins)

    for a, b, r in zip(start[keep], stop[keep], rewarded[keep]):
        if r == 1:
            poke_rewarded[a:b] = 1.0
        else:
            poke_unrewarded[a:b] = 1.0

    # A bin cannot be both; rewarded consumption wins where tables overlap.
    poke_unrewarded[poke_rewarded > 0] = 0.0
    return poke_rewarded, poke_unrewarded


def load_data_dic(path=None, *, validate=True, apply_exclusions=True, verbose=True):
    """Load a `data_dic` pickle, drop known-bad recdays, and check the pairing.

    Use this instead of a bare `pickle.load`. The two validators it runs are the only
    checks that can catch a recday whose neural data came from a different recording day
    than its behaviour: the neural data is binned onto the behavioural timeline, so every
    shape and length check passes regardless. See `recday_registry` and
    `docs/BUG_ly05_recday_mismatch.md`.

    Parameters
    ----------
    path : str, optional
        Defaults to `../data/processed_data/data_dic_lec.pkl` relative to this file.
    validate : bool, default True
        Raise if any recday's `Neuron_raw` row count disagrees with the QC unit count of
        the sorted block it is named after, or if any session's `Task` disagrees with its
        own day's pyControl `active_poke`.
    apply_exclusions : bool, default True
        Drop `recday_registry.EXCLUDE_RECDAYS`, with the reason printed.
    """
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data', 'processed_data', 'data_dic_lec.pkl')
    data_dic = _load_pickle(path)
    if verbose:
        print(f"load_data_dic: {len(data_dic)} recdays from {os.path.basename(path)}")

    registry = _recday_registry()
    if registry is None:
        if verbose:
            print("  recday_registry unavailable — exclusions and guards SKIPPED")
        return data_dic

    if apply_exclusions:
        data_dic = registry.apply_exclusions(data_dic, verbose=verbose)
    if validate:
        registry.validate_data_dic(data_dic, strict=True, verbose=verbose)
        registry.validate_tasks_against_pycontrol(data_dic, strict=True, verbose=verbose)
    return data_dic


def _recday_registry():
    """Import `recday_registry` from this file's directory, or None if unavailable.

    Imported lazily and by path so the PFC copy of this module, which sits beside no
    ephys tree, degrades to a no-op instead of failing at import time.
    """
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import recday_registry
        return recday_registry
    except Exception as exc:                      # noqa: BLE001 - advisory path
        print(f"_recday_registry: not available ({exc})")
        return None


def attach_pokes(data_dic, pokes_dir=None, verbose=True):
    """Load poke event tables into `data_dic[mouse_recday][session]['Pokes']`.

    Files are `pokes_{mouse}_{d1}_{d2}_{sess}.npy`, so the filename maps
    directly onto the existing keys: `mouse_recday` == `{mouse}_{d1}_{d2}` and
    the trailing integer is the session index.

    Stores the RAW (n_pokes, 5) table, not a dense occupancy matrix. This is
    load-bearing: `truncate_all_arrays` treats every 2-D array as
    (n_neurons, n_time) and truncates all arrays to the minimum `shape[1]`, so a
    (n_bins, 9) occupancy matrix — or this (n_pokes, 5) table — placed in the
    dict returned by `prepare_session_data` would silently truncate FR and every
    covariate to 9 (or 5) samples. The raw table therefore lives only in the
    INPUT `data_dic`, which is never truncated; `prepare_session_data` consumes
    it and returns only 1-D arrays.

    Returns the number of (recday, session) pairs that got a table.
    """
    import os
    import glob as _glob

    if pokes_dir is None:
        pokes_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'processed_data', 'trialtimes_raw_mingyutest',
        )

    paths = sorted(_glob.glob(os.path.join(pokes_dir, 'pokes_*.npy')))
    n_attached = 0
    n_empty = 0
    unmatched = []

    for path in paths:
        stem = os.path.basename(path)[len('pokes_'):-len('.npy')]
        mouse_recday, _, sess_str = stem.rpartition('_')
        try:
            session = int(sess_str)
        except ValueError:
            unmatched.append(stem)
            continue
        if mouse_recday not in data_dic or session not in data_dic[mouse_recday]:
            unmatched.append(stem)
            continue
        table = np.load(path)
        data_dic[mouse_recday][session]['Pokes'] = table
        n_attached += 1
        n_empty += int(table.size == 0)

    if verbose:
        n_sessions = sum(len(v) for v in data_dic.values())
        print(f"attach_pokes: {n_attached}/{len(paths)} poke files attached "
              f"({n_empty} empty tables) covering {n_attached}/{n_sessions} sessions")
        if unmatched:
            print(f"  {len(unmatched)} file(s) had no matching data_dic entry, "
                  f"e.g. {unmatched[:3]}")
    return n_attached


def compute_distance_to_rewards(trial_times, speed_array):
    reward_times = np.unique(np.sort(trial_times.flatten()))
    reward_times = reward_times[reward_times >= 0]

    max_time = int(reward_times[-1])
    T = len(speed_array)
    cumulative_distance = np.zeros(T + 1)
    for i in range(1, T + 1):
        cumulative_distance[i] = cumulative_distance[i - 1] + speed_array[i - 1]

    n_timepoints = min(T, max_time + 1)
    distance_from_last_reward = np.zeros(n_timepoints, dtype=float)
    distance_to_next_reward = np.zeros(n_timepoints, dtype=float)

    for t in range(n_timepoints):
        idx_last = np.searchsorted(reward_times, t, side='right') - 1
        last_reward_time = 0 if idx_last < 0 else int(reward_times[idx_last])
        t_clipped = min(t, T)
        last_clipped = min(last_reward_time, T)
        distance_from_last_reward[t] = cumulative_distance[t_clipped] - cumulative_distance[last_clipped]

        idx_next = np.searchsorted(reward_times, t, side='left')
        next_reward_time = t if idx_next >= len(reward_times) else int(reward_times[idx_next])
        next_clipped = min(next_reward_time, T)
        distance_to_next_reward[t] = cumulative_distance[next_clipped] - cumulative_distance[t_clipped]

    return distance_from_last_reward, distance_to_next_reward


def smooth_and_calculate_scalar_derivatives(data_matrix, sigma=3, dt=1.0):
    x_smoothed = gaussian_filter1d(data_matrix[:, 0], sigma=sigma)
    y_smoothed = gaussian_filter1d(data_matrix[:, 1], sigma=sigma)
    vx = np.gradient(x_smoothed, dt)
    vy = np.gradient(y_smoothed, dt)
    speed = np.sqrt(vx**2 + vy**2)
    acceleration = np.gradient(speed, dt)
    return np.column_stack((x_smoothed, y_smoothed, speed, acceleration))


# ============================================================================
# Maze graph + per-transition filter (correct-path + time-bounded)
# ============================================================================
#
# The 3×3 grid maze has 12 edges (matches the (12, 2) Edge_grid.npy file). Used
# to classify each inter-reward transition as "correct" (animal took a shortest
# path between the two consecutive reward ports) and/or "fast" (transition
# duration below a threshold). Samples in transitions that fail either filter
# are masked out before fitting.

_MAZE_EDGES = frozenset({
    (1, 2), (2, 3), (1, 4), (2, 5), (3, 6), (4, 5),
    (5, 6), (4, 7), (5, 8), (6, 9), (7, 8), (8, 9),
})


def _build_maze_graph(edges=_MAZE_EDGES):
    """Build a {node: set(neighbours)} adjacency dict from a set of node-pair
    tuples. Symmetric — each edge contributes both directions."""
    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def _shortest_distance(graph, src, dst):
    """BFS unweighted shortest distance between two nodes. Returns int or -1
    if unreachable. Trivially 0 when src == dst."""
    if src == dst:
        return 0
    if src not in graph or dst not in graph:
        return -1
    visited = {src}
    frontier = [(src, 0)]
    while frontier:
        node, d = frontier.pop(0)
        for nbr in graph[node]:
            if nbr == dst:
                return d + 1
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, d + 1))
    return -1


def compute_transition_filter_mask(
    trial_times_bins,
    locs,
    task,
    *,
    require_shortest_path=True,
    max_duration_bins=None,
    graph=None,
):
    """Build a per-sample boolean mask marking transitions to KEEP for the GLM.

    Iterates state segments in `trial_times_bins.flatten()` order (consecutive
    sorted boundaries). For each segment:
      - state index s = i % num_states; src = task[s], dst = task[(s+1) % num_states]
      - duration = boundary[i+1] - boundary[i] (raw bins)
      - path: extract Locs in the segment, keep entries in [1..9] (nodes only),
        compress consecutive duplicates. Pass if first==src, last==dst, and
        compressed length-1 equals graph shortest distance (= no backtracking).

    Samples in segments that pass BOTH filters are set True; the rest False.

    Parameters
    ----------
    trial_times_bins : array (n_trials, n_states+1) of bin indices (raw rate)
    locs             : array (T,) of Locs values (1..9 = nodes, 10..21 = bridges)
    task             : sequence of length num_states with reward node ids
    require_shortest_path : if True, apply correct-path filter
    max_duration_bins     : int or None; if int, apply duration <= bound filter
    graph                  : prebuilt maze graph or None (default 3×3 grid)

    Returns
    -------
    mask  : np.ndarray of dtype bool, length len(locs)
    stats : dict with aggregate + per-transition info (see module docs)
    """
    T = len(locs)
    mask = np.zeros(T, dtype=bool)
    if graph is None:
        graph = _build_maze_graph()

    task = list(task)
    num_states = trial_times_bins.shape[1] - 1
    boundaries = np.sort(trial_times_bins.flatten()).astype(int)

    n_transitions_total = 0
    n_pass_path = 0
    n_pass_time = 0
    n_pass_both = 0
    n_samples_kept = 0
    per_transition = []

    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end   = int(boundaries[i + 1])
        if start >= end or start >= T:
            continue
        end = min(end, T)
        if start < 0:
            start = 0

        n_transitions_total += 1
        state_idx = i % num_states
        src = int(task[state_idx])
        dst = int(task[(state_idx + 1) % num_states])
        duration_bins = end - start

        # Path filter
        seg_locs = locs[start:end]
        finite = np.isfinite(seg_locs)
        node_only = seg_locs[finite]
        # cast to int (may be float dtype if NaN present); guarantee 1..9 selection
        node_only = node_only.astype(int)
        node_only = node_only[(node_only >= 1) & (node_only <= 9)]
        # compress consecutive duplicates
        compressed = []
        for v in node_only:
            v_int = int(v)
            if not compressed or compressed[-1] != v_int:
                compressed.append(v_int)
        shortest = _shortest_distance(graph, src, dst)
        if compressed and compressed[0] == src and compressed[-1] == dst \
                and shortest >= 0 \
                and (len(compressed) - 1) == shortest:
            pass_path = True
        else:
            pass_path = False

        # Time filter
        if max_duration_bins is None:
            pass_time = True
        else:
            pass_time = duration_bins <= int(max_duration_bins)

        # Aggregate
        if pass_path:    n_pass_path += 1
        if pass_time:    n_pass_time += 1
        path_ok = (not require_shortest_path) or pass_path
        time_ok = pass_time
        if path_ok and time_ok:
            n_pass_both += 1
            mask[start:end] = True
            n_samples_kept += (end - start)

        per_transition.append({
            'src': src, 'dst': dst,
            'duration_bins': int(duration_bins),
            'n_unique_nodes': len(compressed),
            'shortest_dist': int(shortest),
            'pass_path': bool(pass_path),
            'pass_time': bool(pass_time),
        })

    stats = {
        'n_transitions_total': n_transitions_total,
        'n_pass_path':         n_pass_path,
        'n_pass_time':         n_pass_time,
        'n_pass_both':         n_pass_both,
        'n_samples_total':     int(T),
        'n_samples_kept':      int(n_samples_kept),
        'pct_samples_kept':    100.0 * n_samples_kept / max(1, T),
        'per_transition':      per_transition,
    }
    return mask, stats


def summarize_transition_filter_loss(
    mouse_recdays, data_dic,
    *,
    max_transition_seconds=60,
    require_shortest_path=True,
    bin_size_ms=25,
    verbose=True,
):
    """Diagnostic helper. Walks every (recday, session), computes the
    transition filter mask, and reports per-recday + pooled statistics.

    Does NOT fit anything — pure preview of "what fraction would we lose".

    Parameters
    ----------
    mouse_recdays, data_dic : as in `run_glm_analysis`.
    max_transition_seconds : float
        Drop transitions longer than this many seconds.
    require_shortest_path : bool
        Drop transitions whose Locs sequence isn't a shortest path between the
        consecutive reward ports.
    bin_size_ms : int
        Raw bin size in ms (LEC + PFC both = 25).
    verbose : bool
        Print per-recday and pooled summary.

    Returns
    -------
    summary : dict {recday: stats_dict}; also includes '__pooled__' key.
    """
    max_dur_bins = int(round(max_transition_seconds * 1000.0 / bin_size_ms))
    summary = {}
    pooled = {'n_transitions_total': 0, 'n_pass_path': 0, 'n_pass_time': 0,
              'n_pass_both': 0, 'n_samples_total': 0, 'n_samples_kept': 0}

    if verbose:
        print(f"Transition filter preview (max {max_transition_seconds}s, "
              f"require_shortest_path={require_shortest_path})")
        print(f"{'recday':30s}  {'sessions':>8s}  {'transitions':>11s}  "
              f"{'path%':>6s}  {'time%':>6s}  {'both%':>6s}  {'samples%':>8s}")
        print('-' * 90)

    for mr in mouse_recdays:
        if mr not in data_dic:
            continue
        try:
            sessions_for_glm, _ = get_sessions_for_glm(data_dic[mr])
        except Exception:
            continue
        rd = {'n_transitions_total': 0, 'n_pass_path': 0, 'n_pass_time': 0,
              'n_pass_both': 0, 'n_samples_total': 0, 'n_samples_kept': 0,
              'n_sessions': 0}
        for sess in sessions_for_glm:
            sd = data_dic[mr][sess]
            tt = sd.get('Trial_times')
            locs = sd.get('Locs_raw')
            task = sd.get('Task')
            if tt is None or locs is None or task is None:
                continue
            try:
                _, stats = compute_transition_filter_mask(
                    np.asarray(tt).astype(int), locs, task,
                    require_shortest_path=require_shortest_path,
                    max_duration_bins=max_dur_bins,
                )
            except Exception as e:
                if verbose:
                    print(f"  {mr} sess {sess}: error ({e})")
                continue
            rd['n_sessions'] += 1
            for k in ('n_transitions_total', 'n_pass_path', 'n_pass_time',
                      'n_pass_both', 'n_samples_total', 'n_samples_kept'):
                rd[k] += stats[k]

        if rd['n_transitions_total'] > 0:
            for k in pooled:
                pooled[k] += rd[k]
            summary[mr] = rd
            if verbose:
                ntt = rd['n_transitions_total']
                print(f"  {mr:30s}  {rd['n_sessions']:>8d}  {ntt:>11d}  "
                      f"{100 * rd['n_pass_path'] / ntt:>5.1f}%  "
                      f"{100 * rd['n_pass_time'] / ntt:>5.1f}%  "
                      f"{100 * rd['n_pass_both'] / ntt:>5.1f}%  "
                      f"{100 * rd['n_samples_kept'] / max(1, rd['n_samples_total']):>7.1f}%")

    summary['__pooled__'] = pooled
    if verbose and pooled['n_transitions_total'] > 0:
        ntt = pooled['n_transitions_total']
        print('-' * 90)
        print(f"  {'POOLED':30s}  {'':>8s}  {ntt:>11d}  "
              f"{100 * pooled['n_pass_path'] / ntt:>5.1f}%  "
              f"{100 * pooled['n_pass_time'] / ntt:>5.1f}%  "
              f"{100 * pooled['n_pass_both'] / ntt:>5.1f}%  "
              f"{100 * pooled['n_samples_kept'] / max(1, pooled['n_samples_total']):>7.1f}%")
    return summary


# ============================================================================
# Session selection and data preparation
# ============================================================================

def get_sessions_for_glm(recday_data):
    valid_sessions = {k: v for k, v in recday_data.items()
                      if 'num_trials' in v and v['num_trials'] > 2
                      and 'Neuron_raw' in v and v['Neuron_raw'] is not None
                      and 'Locs_raw' in v}
    sorted_sessions = sorted(valid_sessions.items(), key=lambda x: x[1]['num_trials'], reverse=True)
    unique_tasks = {}
    for session, data in sorted_sessions:
        task = str(data.get('Task', 'unknown'))
        if task not in unique_tasks:
            unique_tasks[task] = session
    return list(unique_tasks.values()), list(unique_tasks.keys())


#: Column of a (T, 2) `HD_raw` to use as head direction. Column 0 is `back2mid_deg`
#: (head_back -> head_mid), the body/head axis; column 1 is `earL2earR_deg`, which is the
#: same heading rotated ~90 deg. See `sleap_preprocess_lEC.ipynb`.
_HD_COLUMN = 0


def _extract_head_direction(hd_raw, n_expected):
    """Return a 1-D head-direction vector of length `n_expected` from `HD_raw`.

    `HD_raw` is stored as (T, 2) = [back2mid_deg, earL2earR_deg] -- two angles about 90 deg
    apart. This used to be `HD_raw.flatten()`, which INTERLEAVES the two columns into a
    2T-length vector; `truncate_all_arrays` then cut every array to the shortest (T), so
    the GLM received `HD[:T]` = `back2mid[:T/2]` interleaved with `earL2earR[:T/2]`. That
    is half the session, mixed with a 90-deg-rotated copy of itself, and misaligned with
    every other regressor from the second sample on. Every HD result fitted before this fix
    is noise (see docs/ANATOMY_SPLIT_PLAN.md section 6).

    Verified before the fix (code/w0_gates.py, gate 2, 187 sessions): the signed circular
    difference `earL2earR - back2mid` has median |diff| 84.8 deg and sits within 25 deg of
    +/-90 for 98.3% of samples, so the columns are what the preprocessing docstring says.
    """
    hd = np.asarray(hd_raw, dtype=float)
    if hd.ndim == 2:
        if hd.shape[1] == 0:
            return np.full(n_expected, np.nan)
        col = _HD_COLUMN if hd.shape[1] > _HD_COLUMN else 0
        hd = hd[:, col]
    else:
        hd = hd.ravel()
    if len(hd) < n_expected:
        hd = np.concatenate([hd, np.full(n_expected - len(hd), np.nan)])
    return hd[:n_expected]


def prepare_session_data(session_data, gp_n_bins=10,
                         task=None,
                         filter_correct_paths=False,
                         max_transition_seconds=None,
                         bin_size_ms=25):
    FR = session_data['Neuron_raw']
    Locs = session_data['Locs_raw']

    if 'XY_raw' in session_data and session_data['XY_raw'] is not None:
        XY = session_data['XY_raw']
        if XY.ndim == 1:
            XY = XY.reshape(-1, 2) if len(XY) % 2 == 0 else np.column_stack([XY, XY])
        processed_xy = smooth_and_calculate_scalar_derivatives(XY)
        Speed = processed_xy[:, 2]
        Acc = processed_xy[:, 3]
    else:
        Speed = np.zeros(len(Locs))
        Acc = np.zeros(len(Locs))

    if 'HD_raw' in session_data and session_data['HD_raw'] is not None:
        HD = _extract_head_direction(session_data['HD_raw'], len(Locs))
    else:
        HD = np.full(len(Locs), np.nan)

    if 'Trial_times' in session_data and session_data['Trial_times'] is not None:
        Trial_times = session_data['Trial_times']
        Trial_times_bins = Trial_times.astype(int)
        State, _, GP_binned, time_from, time_to = compute_task_state_arrays(
            Trial_times_bins, num_bins=gp_n_bins
        )
        time_since_A, time_to_A, progress_since_A_binned = compute_since_A_arrays(
            Trial_times_bins, num_bins=gp_n_bins
        )
        if len(Speed) > 0:
            dist_from, dist_to = compute_distance_to_rewards(Trial_times_bins, Speed)
        else:
            dist_from = np.zeros(len(State))
            dist_to = np.zeros(len(State))
    else:
        max_len = FR.shape[1]
        State = np.zeros(max_len, dtype=int)
        GP_binned = np.zeros(max_len, dtype=int)
        time_from = np.zeros(max_len)
        time_to = np.zeros(max_len)
        time_since_A = np.zeros(max_len, dtype=int)
        time_to_A = np.zeros(max_len, dtype=int)
        progress_since_A_binned = np.zeros(max_len, dtype=int)
        dist_from = np.zeros(max_len)
        dist_to = np.zeros(max_len)

    # Nose-poke occupancy. Optional, exactly like HD_raw above: absent for the
    # mFC/PFC dataset (no poke tables exist for it), in which case both arrays
    # are zeros and the regressors contribute nothing. Built at FR.shape[1] --
    # the neural recording length -- not len(Locs), which runs longer.
    if 'Pokes' in session_data and session_data['Pokes'] is not None:
        poke_rewarded, poke_unrewarded = compute_poke_arrays(
            session_data['Pokes'], n_bins=FR.shape[1]
        )
    else:
        poke_rewarded = np.zeros(FR.shape[1])
        poke_unrewarded = np.zeros(FR.shape[1])

    # Distance-based goal progress: fraction of the inter-reward path travelled.
    # NaN at samples where the animal didn't move at all in the interval
    # (total path = 0); zero-mask them downstream.
    total_path = dist_from + dist_to
    with np.errstate(divide='ignore', invalid='ignore'):
        gp_distance_cont = np.where(total_path > 0, dist_from / total_path, np.nan)

    # ---------------------------------------------------------------------
    # Per-sample transition-filter mask (correct-path + duration-bounded).
    # Default: all-True (no filter). When either filter is active and Task +
    # Trial_times are available, build the mask at raw bin rate so it can be
    # downsampled alongside other arrays.
    # ---------------------------------------------------------------------
    valid_transition_mask = np.ones(len(Locs), dtype=bool)
    filter_stats = None
    has_trial_times = ('Trial_times' in session_data
                       and session_data['Trial_times'] is not None)
    if (filter_correct_paths or max_transition_seconds is not None) \
            and has_trial_times and task is not None:
        if max_transition_seconds is None:
            max_dur_bins = None
        else:
            max_dur_bins = int(round(max_transition_seconds * 1000.0 / bin_size_ms))
        mask, filter_stats = compute_transition_filter_mask(
            Trial_times_bins, Locs, task,
            require_shortest_path=filter_correct_paths,
            max_duration_bins=max_dur_bins,
        )
        # mask length == len(Locs); ensure exact alignment
        if len(mask) == len(valid_transition_mask):
            valid_transition_mask = mask
        elif len(mask) < len(valid_transition_mask):
            valid_transition_mask[:len(mask)] = mask
            valid_transition_mask[len(mask):] = False
        else:
            valid_transition_mask = mask[:len(valid_transition_mask)]

    return {
        'FR': FR,
        'Locs': Locs,
        'HD': HD,
        'Speed': Speed,
        'Acc': Acc,
        'State': State,
        'GP_binned': GP_binned,
        'GP_dist_continuous': gp_distance_cont,
        'time_from_reward': time_from,
        'time_to_reward': time_to,
        'time_since_A': time_since_A,
        'time_to_A': time_to_A,
        'progress_since_A_binned': progress_since_A_binned,
        'poke_rewarded': poke_rewarded,
        'poke_unrewarded': poke_unrewarded,
        'dist_from_reward': dist_from,
        'dist_to_reward': dist_to,
        'valid_transition_mask': valid_transition_mask,
        'transition_filter_stats': filter_stats,
    }


def truncate_all_arrays(data_dict):
    lengths = []
    for key, arr in data_dict.items():
        if isinstance(arr, np.ndarray):
            lengths.append(arr.shape[1] if arr.ndim == 2 else len(arr))
    max_index = min(lengths) if lengths else 0
    truncated = {}
    for key, arr in data_dict.items():
        if isinstance(arr, np.ndarray):
            truncated[key] = arr[:, :max_index] if arr.ndim == 2 else arr[:max_index]
        else:
            truncated[key] = arr
    return truncated


#: How each `prepare_session_data` field is aggregated by `downsample_mode='bin'`.
#: Anything not listed falls back to the block's middle sample.
_DOWNSAMPLE_AGG = {
    'FR':                      'sum',      # spike counts -- the whole point
    'Locs':                    'mode',     # node id 1..21 (>21 = edge); a mean is meaningless
    'HD':                      'circmean',
    'Speed':                   'mean',
    'Acc':                     'mean',
    'State':                   'mode',     # 0..3 categorical
    'GP_binned':               'mode',     # ordinal bin index, used as a one-hot index
    'GP_dist_continuous':      'nanmean',
    'time_from_reward':        'mean',     # decile-binned downstream, so a mean is fine
    'time_to_reward':          'mean',
    'time_since_A':            'mean',
    'time_to_A':               'mean',
    'progress_since_A_binned': 'mode',
    'poke_rewarded':           'max',      # "in port at any point in the window"
    'poke_unrewarded':         'max',
    'valid_transition_mask':   'majority',  # most of the window must be valid, not all of it
}


def _aggregate_blocks(arr, factor, how):
    """Reduce consecutive blocks of `factor` samples along the time axis."""
    a = np.asarray(arr)
    twodim = a.ndim == 2
    T = a.shape[1] if twodim else len(a)
    n_blocks = T // factor
    if n_blocks == 0:
        return a[:, :0] if twodim else a[:0]
    a = a[:, :n_blocks * factor] if twodim else a[:n_blocks * factor]
    b = (a.reshape(a.shape[0], n_blocks, factor) if twodim
         else a.reshape(n_blocks, factor))
    ax = 2 if twodim else 1

    if how == 'sum':
        return b.sum(axis=ax)
    if how == 'mean':
        return b.astype(float).mean(axis=ax)
    if how == 'nanmean':
        with np.errstate(invalid='ignore'):
            return np.nanmean(b.astype(float), axis=ax)
    if how == 'max':
        return b.max(axis=ax)
    if how == 'all':
        return b.all(axis=ax)
    if how == 'majority':
        # A window whose covariates are AVERAGED over it is usable when most of it is
        # valid. Requiring every raw bin ('all') compounds brutally with an upstream
        # sample filter: at factor 10 it needs 10 consecutive valid bins, at factor 20
        # it needs 20, so a filter keeping ~10% of samples loses a further ~40% here.
        return b.mean(axis=ax) >= 0.5
    if how == 'circmean':
        z = np.exp(1j * np.deg2rad(b.astype(float)))
        with np.errstate(invalid='ignore'):
            m = np.nanmean(z, axis=ax)
        return np.rad2deg(np.angle(m)) % 360
    if how == 'mode':
        # Most-occupied value in the window. Non-finite entries are ignored unless the
        # whole block is non-finite, which is propagated as NaN so the downstream
        # `Locs <= 21` / finite filters still exclude it.
        bf = b.astype(float)
        out = np.full(bf.shape[:ax], np.nan)
        flat = bf.reshape(-1, factor)
        res = np.full(flat.shape[0], np.nan)
        for i in range(flat.shape[0]):
            v = flat[i][np.isfinite(flat[i])]
            if v.size:
                vals, cnt = np.unique(v, return_counts=True)
                res[i] = vals[np.argmax(cnt)]
        return res.reshape(out.shape)
    raise ValueError(f'unknown aggregation {how!r}')


def downsample_session_data(data_dict, factor, mode='stride'):
    """Downsample all arrays along the time axis.

    Parameters
    ----------
    mode : {'stride', 'bin'}, default 'stride'
        ``'stride'`` keeps every `factor`th sample -- the original behaviour, kept as the
        default so cached fits stay reproducible. It does NOT widen the bins: each retained
        sample is still one 25 ms bin, so with `factor=10` the GLM sees ~10% of the session's
        spikes and every other spike is discarded. `Neuron_raw` is integer counts (89% zeros,
        mean 0.131/bin), so a retained sample carries ~0.13 spikes.

        ``'bin'`` aggregates each block of `factor` samples instead: spike counts are SUMMED,
        continuous behaviour averaged, categorical behaviour taken as the block mode, head
        direction circular-averaged (see `_DOWNSAMPLE_AGG`). This yields the same number of
        samples, spaced identically -- so autocorrelation between samples is unchanged -- but
        each carries ~10x the spikes, for roughly sqrt(10) ~ 3x the SNR per observation at no
        cost. Under 'stride' the in-sample R^2 of these fits is ~0.011 against a p/T chance
        floor of ~0.004, i.e. only ~2.6x chance, which is the regime this is meant to improve.
    """
    if factor <= 1:
        return data_dict
    if mode not in ('stride', 'bin'):
        raise ValueError(f"mode must be 'stride' or 'bin', got {mode!r}")
    result = {}
    for key, arr in data_dict.items():
        if not isinstance(arr, np.ndarray):
            result[key] = arr
            continue
        if mode == 'stride':
            result[key] = arr[:, ::factor] if arr.ndim == 2 else arr[::factor]
        else:
            how = _DOWNSAMPLE_AGG.get(key)
            if how is None:
                # Unknown field: take the block's middle sample rather than guess.
                off = factor // 2
                result[key] = (arr[:, off::factor] if arr.ndim == 2 else arr[off::factor])
            else:
                result[key] = _aggregate_blocks(arr, factor, how)
    # Blocks must line up across fields: trim any middle-sample field that came out longer.
    lengths = [a.shape[1] if a.ndim == 2 else len(a)
               for a in result.values() if isinstance(a, np.ndarray)]
    if lengths:
        n = min(lengths)
        for k, a in result.items():
            if isinstance(a, np.ndarray):
                result[k] = a[:, :n] if a.ndim == 2 else a[:n]
    return result


# ============================================================================
# One-hot encoding helpers
# ============================================================================

def compute_decile_edges(values, n_bins=10, outlier_pct=1, scheme='decile'):
    """Compute bin edges from training data, excluding outliers.

    `scheme='decile'` places edges at quantiles, so every bin holds ~the same number of
    samples; `scheme='uniform'` places them at equal width across the trimmed range.

    Which to use is not cosmetic. The live GLM bins speed/acceleration/time_*/distance_* by
    QUANTILE while goal_progress, goal_progress_distance and progress_since_A are equal-width
    (they are already fractions on [0,1]) and head_direction is 36 fixed 10-degree bins -- so
    the design mixes two placement schemes and gives them different effective flexibility.
    Measured with train-only edges (`glm_cv_cpd`, binned data): all-quantile r2_cv +0.0546 vs
    all-uniform +0.0486, so quantile genuinely fits the skewed reward-relative variables
    better. But the shared variance moves: `goal_progress` CPD is -0.0003 under quantile and
    +0.0009 under uniform, because uniform binning handicaps its competitors rather than
    improving goal_progress itself. Report the scheme with any result that depends on it.
    """
    values = np.asarray(values).flatten()
    values = values[np.isfinite(values)]
    if len(values) < n_bins * 2:
        return None
    lo, hi = np.percentile(values, [outlier_pct, 100 - outlier_pct])
    clipped = values[(values >= lo) & (values <= hi)]
    if scheme == 'uniform':
        edges = np.linspace(clipped.min(), clipped.max(), n_bins + 1)
    elif scheme == 'decile':
        edges = np.percentile(clipped, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"scheme must be 'decile' or 'uniform', got {scheme!r}")
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def apply_onehot(values, edges):
    """Bin values using precomputed edges, one-hot encode keeping ALL bins (no reference drop)."""
    values = np.asarray(values, dtype=float)
    n_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)
    onehot = (bin_idx[:, None] == np.arange(0, n_bins)).astype(float)
    onehot[~np.isfinite(values)] = 0
    return onehot


# ============================================================================
# Raised-cosine basis (Pillow et al. 2008) — smooth alternative to one-hot bins
# ============================================================================

# Per-variable spacing for the raised-cosine basis. 'log' concentrates
# resolution near the low end of the range (i.e. near the reward event for
# reward-relative variables); 'linear' spreads resolution evenly.
_RAISED_COSINE_SPACING = {
    'speed':                  'linear',
    'acceleration':           'linear',
    'goal_progress':          'linear',
    'goal_progress_distance': 'linear',
    'time_from_reward':       'log',
    'time_to_reward':         'log',
    'time_since_A':           'log',
    'time_to_A':              'log',
    'progress_since_A':       'linear',
    'distance_from_reward':   'log',
    'distance_to_reward':     'log',
}


def make_raised_cosine_basis(values, n_basis=10, spacing='linear',
                             value_range=None, log_offset=None):
    """Raised-cosine basis for a continuous variable (Pillow et al. 2008).

    Returns an (n_samples, n_basis) matrix. Each column is a smooth cosine
    "bump"; adjacent bumps cross at half-height and sum to ≈ 1 in the interior,
    so the basis is a smooth analogue of one-hot binning.

    Parameters
    ----------
    values : array-like
        Raw continuous values (not pre-binned).
    n_basis : int, default 10
        Number of basis functions (columns). 10 matches the decile one-hot
        column count, keeping the design matrix column-structure identical.
    spacing : {'linear', 'log'}
        'linear' places centers evenly across the range.
        'log' places centers evenly in log-warped space, concentrating
        resolution near the low end of the range.
    value_range : (lo, hi) or None
        Range to cover. If None, uses 1st–99th percentile of finite values
        (matches `compute_decile_edges` outlier clipping).
    log_offset : float or None
        Offset added before the log warp so it's finite at the low edge (and
        at 0). Defaults to a small fraction of the range.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros((len(values), n_basis))

    if value_range is None:
        lo, hi = np.percentile(values[finite], [1, 99])
    else:
        lo, hi = value_range
    if hi <= lo:
        hi = lo + 1e-6

    if spacing == 'log':
        if log_offset is None:
            log_offset = max(1e-6, (hi - lo) / n_basis)
        warp = lambda x: np.log(np.clip(x, lo, hi) - lo + log_offset)
    elif spacing == 'linear':
        warp = lambda x: np.clip(x, lo, hi)
    else:
        raise ValueError(f"spacing must be 'linear' or 'log', got {spacing!r}")

    centers = np.linspace(warp(lo), warp(hi), n_basis)
    # Bump width = 2 × center spacing → adjacent bumps cross at half-height.
    db = centers[1] - centers[0] if n_basis > 1 else 1.0
    width = 2.0 * db if db > 0 else 1.0

    w_vals = warp(values)
    basis = np.zeros((len(values), n_basis))
    for j, cj in enumerate(centers):
        d = (w_vals - cj) * np.pi / width
        basis[:, j] = 0.5 * (1.0 + np.cos(np.clip(d, -np.pi, np.pi)))
    basis[~finite] = 0.0
    return basis


# ============================================================================
# Regressor group index map (no intercept, all bins kept)
# place(21) + HD(36) + GP(10) + speed(10) + acc(10)
# + time_from(10) + time_to(10) + dist_from(10) + dist_to(10) = 127 total
# ============================================================================

_N_TASK_STATES = 4

# Regressor groups for the default 9-regressor layout (127 cols).
# Optional regressors (goal_progress_distance, task_state, time_since_A,
# time_to_A, progress_since_A) are not listed here; they're added dynamically
# via _resolve_regressor_groups and _OPTIONAL_REGRESSORS.
regressor_groups = {
    'place':                list(range(0,   21)),
    'head_direction':       list(range(21,  57)),
    'goal_progress':        list(range(57,  67)),
    'speed':                list(range(67,  77)),
    'acceleration':         list(range(77,  87)),
    'time_from_reward':     list(range(87,  97)),
    'time_to_reward':       list(range(97,  107)),
    'distance_from_reward': list(range(107, 117)),
    'distance_to_reward':   list(range(117, 127)),
}

analysis_regressor_names = [
    'place',
    'head_direction',
    'goal_progress',
    'speed',
    'acceleration',
    'time_from_reward',
    'time_to_reward',
    'distance_from_reward',
    'distance_to_reward',
]


# Optional regressors. NOT in `analysis_regressor_names`, so
# `regressors_to_include=None` continues to give the 9-regressor default
# (backwards-compatible). To include one, list it explicitly in
# `regressors_to_include`. See each for description:
#   goal_progress_distance  : path-length analogue of goal_progress
#   task_state              : A/B/C/D leg identity (4 one-hot columns)
#   time_since_A            : raw bins elapsed since reward A (decile-binned, 10 cols)
#   time_to_A               : raw bins until next reward A (decile-binned, 10 cols)
#   progress_since_A        : loop-fraction progress (0→1, equal-width, gp_n_bins cols)
#   poke_rewarded           : in-port during a rewarded poke (binary, 1 col)
#   poke_unrewarded         : in-port during an unrewarded poke (binary, 1 col)
# NOTE poke_rewarded is collinear with early time_from_reward BY CONSTRUCTION
# (a rewarded poke's entry bin is the reward bin). Its CPD is an arbitrary split
# of shared variance; use run_poke_duration_split for the real dissociation.
_OPTIONAL_REGRESSORS = [
    'goal_progress_distance', 'task_state',
    'time_since_A', 'time_to_A', 'progress_since_A',
    'poke_rewarded', 'poke_unrewarded',
]
_ALL_REGRESSORS = analysis_regressor_names + _OPTIONAL_REGRESSORS

# Canonical column ordering when a fit includes a mix of default + optional
# regressors. task_state sits right after place (both are "which location/leg"
# categoricals); distance-GP sits adjacent to time-GP; the A-anchored time/progress
# sit right after the reward-anchored time regressors.
_CANONICAL_ORDER = [
    'place',
    'task_state',
    'poke_rewarded',
    'poke_unrewarded',
    'head_direction',
    'goal_progress',
    'goal_progress_distance',
    'speed',
    'acceleration',
    'time_from_reward',
    'time_to_reward',
    'time_since_A',
    'time_to_A',
    'progress_since_A',
    'distance_from_reward',
    'distance_to_reward',
]


# User-facing aliases → canonical names. Lets users type "time_since_reward"
# instead of "time_from_reward" (clearer phrasing, same underlying variable).
_REGRESSOR_NAME_ALIASES = {
    'time_since_reward':     'time_from_reward',
    'distance_since_reward': 'distance_from_reward',
    'state':                 'task_state',
}


# Pretty display labels for plots / printouts. Falls back to the canonical name
# if a key is missing.
_REGRESSOR_DISPLAY_NAMES = {
    'place':                  'place',
    'task_state':             'task state (A/B/C/D)',
    'poke_rewarded':          'rewarded poke (in port)',
    'poke_unrewarded':        'unrewarded poke (in port)',
    'head_direction':         'head direction',
    'goal_progress':          'goal progress',
    'goal_progress_distance': 'goal progress (distance)',
    'speed':                  'speed',
    'acceleration':           'acceleration',
    'time_from_reward':       'time since reward',
    'time_to_reward':         'time to reward',
    'time_since_A':           'time since reward A',
    'time_to_A':              'time to reward A',
    'progress_since_A':       'loop progress (since A)',
    'distance_from_reward':   'distance since reward',
    'distance_to_reward':     'distance to reward',
}


def _display(name):
    """Return a pretty label for a regressor name (canonical or alias)."""
    canonical = _REGRESSOR_NAME_ALIASES.get(name, name)
    return _REGRESSOR_DISPLAY_NAMES.get(canonical, canonical)


def _resolve_regressor_groups(regressors_to_include, gp_n_bins=10,
                              parameterization='all_bins'):
    """Resolve (regressor_groups, analysis_regressor_names) for a subset.

    If `regressors_to_include` is None and `gp_n_bins == 10` and
    `parameterization == 'all_bins'`, returns the module-level defaults
    (all 9 regressors, 127-col layout). Otherwise returns a NEW regressor_groups
    dict whose indices reflect a design matrix built from only the included
    regressors, in canonical order (the order they appear in
    `analysis_regressor_names`).

    Accepts both canonical names and aliases defined in `_REGRESSOR_NAME_ALIASES`
    (e.g. 'time_since_reward' is treated as 'time_from_reward').

    `gp_n_bins` sets the number of goal_progress one-hot columns (default 10).
    Pass the same value used in `run_glm_analysis` so the index map matches the
    fitted design matrix (e.g. gp_n_bins=20 to column-match GP to the joint
    20-col time_any / distance_any groups).

    `parameterization` selects the design-matrix encoding:
      - 'all_bins'        : no intercept, keep all bins per categorical block
                            (current default; rank-deficient by 8).
      - 'reference_coded' : intercept column at index 0 + drop the first bin of
                            every categorical block. Each per-regressor index
                            list shrinks by 1; the cursor starts at 1 to leave
                            room for the intercept. Total cols at gp_n_bins=10:
                            1 + 20 + 35 + 9 + 6·9 = 119 (full rank).
    """
    # Per-regressor column counts. Defaults come from the 9-regressor
    # `regressor_groups`; optional regressors (e.g. `goal_progress_distance`)
    # are not in that map, so add them explicitly.
    n_cols_per = {name: len(regressor_groups[name]) for name in analysis_regressor_names}
    n_cols_per['goal_progress'] = gp_n_bins
    n_cols_per['goal_progress_distance'] = gp_n_bins  # shares the gp_n_bins knob
    n_cols_per['task_state'] = _N_TASK_STATES
    n_cols_per['time_since_A'] = 10  # decile-binned
    n_cols_per['time_to_A'] = 10     # decile-binned
    n_cols_per['progress_since_A'] = gp_n_bins  # equal-width like goal_progress
    n_cols_per['poke_rewarded'] = 1    # binary indicator
    n_cols_per['poke_unrewarded'] = 1  # binary indicator
    if parameterization == 'reference_coded':
        # MIXED reference coding: drop a reference bin from every MULTI-column block, and
        # pass SINGLE-column indicators (the pokes) through untouched.
        #
        # The rank deficiency in 'all_bins' comes from multi-column one-hot blocks, each of
        # which sums to 1 per row, so k blocks contain k copies of the all-ones vector and
        # give k-1 dependencies. A binary poke indicator is 0 most of the time, does NOT sum
        # to 1, and so neither causes the deficiency nor needs a reference bin -- it simply
        # cannot have one taken (dropping its only column leaves nothing).
        #
        # This previously raised, forcing any poke-containing fit onto 'all_bins' and making
        # "the full regressor set" and "interpretable betas" look mutually exclusive. They
        # are not. Verified on a design of this shape: all_bins + pokes = 97 cols / rank 89
        # / deficiency 8; mixed reference coding = 89 cols / rank 89 / deficiency 0.
        n_cols_per = {k: (v - 1 if v > 1 else v) for k, v in n_cols_per.items()}
    elif parameterization != 'all_bins':
        raise ValueError(f"parameterization must be 'all_bins' or "
                         f"'reference_coded', got {parameterization!r}")

    if regressors_to_include is None:
        if gp_n_bins == 10 and parameterization == 'all_bins':
            return regressor_groups, list(analysis_regressor_names)
        ordered = list(analysis_regressor_names)
    else:
        # Map aliases → canonical names; validate against the full set
        # (default + optional regressors like goal_progress_distance).
        resolved = [_REGRESSOR_NAME_ALIASES.get(r, r) for r in regressors_to_include]

        invalid = [r for r in resolved if r not in _ALL_REGRESSORS]
        if invalid:
            raise ValueError(f"Unknown regressor names: {invalid}. "
                             f"Valid names: {_ALL_REGRESSORS} "
                             f"(aliases: {list(_REGRESSOR_NAME_ALIASES.keys())})")

        ordered = [r for r in _CANONICAL_ORDER if r in resolved]

    local = {}
    cursor = 1 if parameterization == 'reference_coded' else 0
    for name in ordered:
        n = n_cols_per[name]
        local[name] = list(range(cursor, cursor + n))
        cursor += n
    return local, ordered


# ============================================================================
# GLM analysis — no CV, fit on all available data per recording day
# ============================================================================

def run_glm_analysis(mouse_recdays, data_dic,
                     num_permutations=100, downsample_factor=10,
                     regressors_to_include=None,
                     compute_cpd=False,
                     joint_drop_groups=None,
                     continuous_basis='onehot',
                     gp_n_bins=10,
                     parameterization='all_bins',
                     filter_correct_paths=False,
                     max_transition_seconds=None,
                     bin_size_ms=25,
                     downsample_mode='stride',
                     continuous_binning='decile',
                     min_samples_per_param=10.0,
                     return_scales=False,
                     scales_only=False,
                     cross_validate=False,
                     cv_n_perm=0,
                     cv_poisson=False,
                     cv_poisson_neurons=None,
                     cv_center_within_sessions=False,
                     cv_zscore_within_sessions=False,
                     cv_within_session_folds=False,
                     cv_nulls=('freedman_lane',),
                     cv_only=False):
    """Fit per-neuron OLS GLM with permutation F-tests.

    Parameters
    ----------
    mouse_recdays, data_dic : see module docstring.
    cross_validate : bool, default False
        Additionally compute leave-one-session-out CPD and R^2 via `glm_cv.cv_scores`,
        returned as an extra `CV_results` element. Sessions are the fold unit because they
        are different tasks, so LOSO tests across-task generalisation. The in-sample
        quantities are still computed and returned unchanged, so the two are directly
        comparable on the same fit -- which is also what makes an old cache checkable
        against a new one. See `glm_cv` for why the in-sample CPD needed this.
    cv_n_perm : int, default 0
        Permutations for the cross-validated null (shifted WITHIN session). 0 skips it: a
        held-out CPD carries no in-sample optimism, so the null calibrates against
        autocorrelation rather than removing overfitting. Costs roughly `cv_n_perm/10` times
        the CV fit time.
    cv_poisson : bool, default False
        Also fit a leave-one-session-out Poisson GLM (deviance-based CPD) as a robustness
        check on the link function. Orders of magnitude slower than the linear path -- it
        cannot share a pseudo-inverse across neurons -- so pair it with
        `cv_poisson_neurons`.
    cv_poisson_neurons : array-like of int, optional
        Neuron subset for the Poisson CV, since fitting all of them is usually too slow.
    num_permutations : int, default 100
        Number of circular firing-rate shifts in the permutation null.
    continuous_binning : {'decile', 'uniform'}, default 'decile'
        Bin placement for speed / acceleration / time_* / distance_*. 'decile' is quantile
        placement (the historical behaviour); 'uniform' is equal width. goal_progress,
        goal_progress_distance and progress_since_A are equal-width regardless (they are
        already fractions on [0,1]) and head_direction is always 36 fixed 10-degree bins, so
        'uniform' is what makes the whole design one scheme. See `compute_decile_edges`.
    downsample_mode : {'stride', 'bin'}, default 'stride'
        How `downsample_factor` is applied. 'stride' keeps every Nth 25 ms bin and DISCARDS
        the rest (original behaviour, ~10% of spikes reach the GLM at factor=10); 'bin' sums
        spike counts over each block and aggregates behaviour appropriately, keeping every
        spike at the same sample count and spacing. See `downsample_session_data`.
    downsample_factor : int, default 10
    regressors_to_include : list of str or None
        Subset of `analysis_regressor_names` to include in the design matrix.
        Accepts canonical names AND aliases (e.g. 'time_since_reward'). None
        (default) keeps all 9 regressors.
    compute_cpd : bool, default False
        If True, also compute the Coefficient of Partial Determination (CPD)
        per neuron per regressor group: CPD = (RSS_reduced − RSS_full) / RSS_reduced.
        Return signature changes from a 2-tuple to a 3-tuple in this case.
    joint_drop_groups : list of (group_name, [regressor_names]) or None
        Each entry defines a joint test: fit a reduced model with ALL listed
        regressors dropped together, store F-stat (and CPD if compute_cpd=True)
        under `group_name` alongside the per-regressor results.
        Example: `[('time_any', ['time_from_reward', 'time_to_reward'])]`
        gives a joint test of "any time information" using a single F-stat
        on a 20-column reduction.
        Aliases are resolved in the listed regressor names.
    continuous_basis : {'onehot', 'raised_cosine'}, default 'onehot'
        Encoding for the continuous variables (speed, acceleration, GP, and the
        4 reward-relative variables). 'onehot' = decile bins (current behaviour).
        'raised_cosine' = 10 smooth raised-cosine bumps per variable
        (log-spaced for reward-relative variables, linear for speed/accel/GP).
        Place and head_direction are always one-hot. Because n_basis=10 matches
        the decile count, the design matrix is identical in shape (127 cols) and
        all downstream stats/plots are unchanged.
    gp_n_bins : int, default 10
        Number of goal_progress bins (one-hot columns, or raised-cosine bumps).
        The default 10 gives the standard 127-col design matrix. Setting 20
        column-matches goal_progress to the joint 20-col `time_any` /
        `distance_any` groups (design matrix → 137 cols), removing the
        column-count advantage from the headline GP-vs-time/distance comparison.
        Pass the SAME value to `compute_tuning_arrays`.
    parameterization : {'all_bins', 'reference_coded'}, default 'all_bins'
        Design-matrix encoding for categorical blocks:
          - 'all_bins'        : no intercept, keep all bins per block
                                (rank-deficient by 8; current default).
          - 'reference_coded' : prepend an intercept column and drop the FIRST
                                bin of every block (textbook coding; full rank).
        F-stats differ between modes because df accounting differs (block_size
        vs block_size−1 dropped per reduced model; T − 127 vs T − 119 residual
        df at gp_n_bins=10). RSS_full, RSS_reduced, CPD, R², ΔR², and the
        permutation-based significance flags are essentially identical across
        modes — this is intended as an empirical sanity check on the default.
        Pass the SAME value to `compute_tuning_arrays`.

    return_scales : bool, default False
        Also return `Neuron_scales`. Appended LAST, so the existing 2- and
        3-tuple returns are unchanged for callers that do not ask for it.
        Needed by selectivity_geometry.build_alpha_matrix: this GLM fits raw
        `Neuron_raw` counts, so betas carry each neuron's firing rate. Dividing
        by the SD makes coefficients unit-free and comparable across neurons —
        the equivalent of the per-neuron z-scoring Posani et al. (2026) apply
        before extracting selectivity. Without it, a selectivity-space
        clustering analysis clusters by firing rate.
    scales_only : bool, default False
        Build the design matrix and pooled FR exactly as usual, record the
        per-neuron SDs, then skip the OLS fit and permutation tests entirely.
        Use with `return_scales=True` to backfill `neuron_scales` onto an
        already-cached section in seconds rather than refitting it. Pass the
        SAME arguments used for the original fit, or the sample set — and hence
        the SDs — will not match. `GLM_results` etc. come back empty.

    Returns
    -------
    GLM_results : dict of {mouse_recday: {neuron_idx: params_array}}
    Permutation_results : dict of {mouse_recday: {neuron_idx: (F_real_dict, F_perm_dict)}}
    CPD_results (only if compute_cpd=True) : dict of {mouse_recday: {neuron_idx: cpd_dict}}
        Per-regressor CPDs plus reserved keys:
          '__r2_full__'  : full-model R² (scalar)
          '__delta_r2__' : {regressor: ΔRSS/TSS} unique R² per regressor
    Neuron_scales (only if return_scales=True)
        {mouse_recday: {neuron_idx: {'sd': float, 'mean': float}}} over exactly
        the samples entering the fit. Units are spikes per bin; multiply by
        1000/bin_size_ms for Hz.
    """
    local_groups, local_names = _resolve_regressor_groups(
        regressors_to_include, gp_n_bins=gp_n_bins,
        parameterization=parameterization,
    )

    # Resolve aliases in joint_drop_groups and validate
    joint_specs = []  # list of (group_name, [canonical_regressor_names])
    if joint_drop_groups is not None:
        for group_name, regs in joint_drop_groups:
            resolved_regs = [_REGRESSOR_NAME_ALIASES.get(r, r) for r in regs]
            for r in resolved_regs:
                if r not in local_names:
                    raise ValueError(
                        f"Joint group {group_name!r} references {r!r} which is not "
                        f"in the included regressors. Include it via "
                        f"regressors_to_include or remove from joint_drop_groups."
                    )
            joint_specs.append((group_name, resolved_regs))

    GLM_results = {}
    Permutation_results = {}
    CPD_results = {}
    Neuron_scales = {}
    CV_results = {}

    for mouse_recday in tqdm(mouse_recdays, desc="Processing recording days"):
        print(f"\n{mouse_recday}")

        GLM_results[mouse_recday] = {}
        Permutation_results[mouse_recday] = {}
        CPD_results[mouse_recday] = {}
        Neuron_scales[mouse_recday] = {}

        sessions_for_glm, _ = get_sessions_for_glm(data_dic[mouse_recday])

        if len(sessions_for_glm) < 2:
            print(f"  Skipping — not enough valid sessions ({len(sessions_for_glm)})")
            continue

        first_session = sessions_for_glm[0]
        num_neurons = data_dic[mouse_recday][first_session]['Neuron_raw'].shape[0]

        # Per-recday transition-filter accumulators (raw-rate stats)
        recday_filter_stats = {'n_transitions_total': 0, 'n_pass_path': 0,
                                'n_pass_time': 0, 'n_pass_both': 0,
                                'n_samples_total': 0, 'n_samples_kept': 0}
        any_filter_active = filter_correct_paths or max_transition_seconds is not None

        prepared_sessions = {}
        for session in sessions_for_glm:
            session_dict = data_dic[mouse_recday][session]
            prep_data = prepare_session_data(
                session_dict,
                gp_n_bins=gp_n_bins,
                task=session_dict.get('Task') if any_filter_active else None,
                filter_correct_paths=filter_correct_paths,
                max_transition_seconds=max_transition_seconds,
                bin_size_ms=bin_size_ms,
            )
            # Accumulate raw-rate filter stats before downsampling
            if any_filter_active and prep_data.get('transition_filter_stats') is not None:
                s = prep_data['transition_filter_stats']
                for k in ('n_transitions_total', 'n_pass_path', 'n_pass_time',
                          'n_pass_both', 'n_samples_total', 'n_samples_kept'):
                    recday_filter_stats[k] += s.get(k, 0)
            prep_data = truncate_all_arrays(prep_data)
            prepared_sessions[session] = downsample_session_data(
                prep_data, downsample_factor, mode=downsample_mode)

        if any_filter_active and recday_filter_stats['n_transitions_total'] > 0:
            ntt = recday_filter_stats['n_transitions_total']
            print(f"  transition filter: kept {recday_filter_stats['n_pass_both']}/{ntt} "
                  f"({100 * recday_filter_stats['n_pass_both'] / ntt:.1f}%)  "
                  f"path:{100 * recday_filter_stats['n_pass_path'] / ntt:.1f}%  "
                  f"time:{100 * recday_filter_stats['n_pass_time'] / ntt:.1f}%  "
                  f"samples:{100 * recday_filter_stats['n_samples_kept'] / max(1, recday_filter_stats['n_samples_total']):.1f}%")

        # ----------------------------------------------------------------
        # Pool all behavioral data across sessions (valid locations only)
        # ----------------------------------------------------------------
        all_speed, all_acc   = [], []
        all_tf, all_tt       = [], []
        all_tsa, all_tta, all_pga = [], [], []  # A-anchored time/progress
        all_pkr, all_pku     = [], []  # in-port indicators (rewarded/unrewarded)
        all_df, all_dt       = [], []
        all_locs, all_hd, all_gp = [], [], []
        all_gpd              = []  # distance-based goal progress (continuous 0–1, NaN where no motion)
        all_state            = []  # task state (0..3, one per leg)
        session_filters = []  # (session, node_filter) for reconstructing FR

        for session in sessions_for_glm:
            prep = prepared_sessions[session]
            nf   = prep['Locs'] <= 21
            # AND with valid-transition mask (default all-True when filter is off).
            # vtm came through the same truncate/downsample as Locs so lengths
            # should match; if not, pad with False to be safe.
            vtm = prep.get('valid_transition_mask')
            if vtm is not None:
                if len(vtm) == len(nf):
                    nf = nf & vtm.astype(bool)
                elif len(vtm) > len(nf):
                    nf = nf & vtm[:len(nf)].astype(bool)
                else:
                    padded = np.zeros(len(nf), dtype=bool)
                    padded[:len(vtm)] = vtm.astype(bool)
                    nf = nf & padded
            all_speed.append(prep['Speed'][nf])
            all_acc.append(prep['Acc'][nf])
            all_tf.append(prep['time_from_reward'][nf])
            all_tt.append(prep['time_to_reward'][nf])
            all_tsa.append(prep['time_since_A'][nf])
            all_tta.append(prep['time_to_A'][nf])
            all_pga.append(prep['progress_since_A_binned'][nf])
            all_pkr.append(prep['poke_rewarded'][nf])
            all_pku.append(prep['poke_unrewarded'][nf])
            all_df.append(prep['dist_from_reward'][nf])
            all_dt.append(prep['dist_to_reward'][nf])
            all_locs.append(prep['Locs'][nf])
            all_hd.append(prep['HD'][nf])
            all_gp.append(prep['GP_binned'][nf])
            all_gpd.append(prep['GP_dist_continuous'][nf])
            all_state.append(prep['State'][nf])
            session_filters.append((session, nf))

        speed_all = np.concatenate(all_speed)
        acc_all   = np.concatenate(all_acc)
        tf_all    = np.concatenate(all_tf)
        tt_all    = np.concatenate(all_tt)
        tsa_all   = np.concatenate(all_tsa)  # time_since_A
        tta_all   = np.concatenate(all_tta)  # time_to_A
        pga_all   = np.concatenate(all_pga)  # progress_since_A_binned
        pkr_all   = np.concatenate(all_pkr)  # poke_rewarded
        pku_all   = np.concatenate(all_pku)  # poke_unrewarded
        df_all    = np.concatenate(all_df)
        dt_all    = np.concatenate(all_dt)
        locs_all  = np.concatenate(all_locs).astype(int)
        hd_all    = np.concatenate(all_hd)
        gp_all    = np.concatenate(all_gp)
        gpd_all   = np.concatenate(all_gpd)
        state_all = np.concatenate(all_state).astype(int)
        assert np.all((state_all >= 0) & (state_all < _N_TASK_STATES)), \
            f"task state out of range [0, {_N_TASK_STATES}): got {np.unique(state_all)}"

        # ----------------------------------------------------------------
        # Decile edges from all available data
        # ----------------------------------------------------------------
        speed_edges = compute_decile_edges(speed_all, scheme=continuous_binning)
        acc_edges   = compute_decile_edges(acc_all, scheme=continuous_binning)
        tf_edges    = compute_decile_edges(tf_all, scheme=continuous_binning)
        tt_edges    = compute_decile_edges(tt_all, scheme=continuous_binning)
        tsa_edges   = compute_decile_edges(tsa_all, scheme=continuous_binning)
        tta_edges   = compute_decile_edges(tta_all, scheme=continuous_binning)
        df_edges    = compute_decile_edges(df_all, scheme=continuous_binning)
        dt_edges    = compute_decile_edges(dt_all, scheme=continuous_binning)

        if any(e is None for e in [speed_edges, acc_edges, tf_edges,
                                    tt_edges, tsa_edges, tta_edges,
                                    df_edges, dt_edges]):
            print("  Skipping — insufficient data for decile edges")
            continue

        # ----------------------------------------------------------------
        # Build design matrix (shared across all neurons)
        # No intercept; all bins kept per variable. Rank-deficient by 8.
        # ----------------------------------------------------------------
        # Place: nodes 1–21 → 21 cols
        place_onehot = (locs_all[:, None] == np.arange(1, 22)).astype(float)

        # Task state (A/B/C/D): states 0–3 → 4 cols
        state_onehot = (state_all[:, None] == np.arange(_N_TASK_STATES)).astype(float)

        # In-port indicators → 1 col each. Already binary, so they bypass the
        # continuous_basis branch entirely (no decile edges, no raised cosine),
        # exactly like place / task_state / HD.
        PKR_enc = pkr_all.astype(float)[:, None]
        PKU_enc = pku_all.astype(float)[:, None]

        # HD: 36 fixed 10° bins → 36 cols
        hd_bin_idx = np.clip(np.floor((hd_all % 360) / 10).astype(int), 0, 35)
        HD_onehot  = (hd_bin_idx[:, None] == np.arange(0, 36)).astype(float)
        HD_onehot[~np.isfinite(hd_all)] = 0

        if continuous_basis == 'onehot':
            # GP: gp_n_bins equal-width bins → gp_n_bins cols
            GP_enc = (gp_all[:, None] == np.arange(0, gp_n_bins)).astype(float)
            # Distance-based GP: same gp_n_bins, NaN samples zeroed out
            gpd_finite = np.isfinite(gpd_all)
            gpd_binned = np.zeros(len(gpd_all), dtype=int)
            gpd_binned[gpd_finite] = np.clip(
                np.floor(gpd_all[gpd_finite] * gp_n_bins).astype(int),
                0, gp_n_bins - 1,
            )
            GPd_enc = (gpd_binned[:, None] == np.arange(0, gp_n_bins)).astype(float)
            GPd_enc[~gpd_finite] = 0
            # Progress since A: equal-width like goal_progress (already binned upstream)
            PGA_enc = (pga_all[:, None] == np.arange(0, gp_n_bins)).astype(float)
            # Continuous vars: decile one-hot bins → 10 cols each
            Sp_enc = apply_onehot(speed_all, speed_edges)
            Ac_enc = apply_onehot(acc_all,   acc_edges)
            TF_enc = apply_onehot(tf_all,    tf_edges)
            TT_enc = apply_onehot(tt_all,    tt_edges)
            TSA_enc = apply_onehot(tsa_all,  tsa_edges)
            TTA_enc = apply_onehot(tta_all,  tta_edges)
            DF_enc = apply_onehot(df_all,    df_edges)
            DT_enc = apply_onehot(dt_all,    dt_edges)
        elif continuous_basis == 'raised_cosine':
            # Smooth raised-cosine basis (10 bumps each → same col count).
            # Per-variable spacing from _RAISED_COSINE_SPACING (log for
            # reward-relative variables, linear for speed/accel/GP).
            def _rc(vals, name):
                return make_raised_cosine_basis(
                    vals, n_basis=10, spacing=_RAISED_COSINE_SPACING[name]
                )
            # GP gets gp_n_bins bumps so its column count matches the one-hot path
            GP_enc = make_raised_cosine_basis(
                gp_all.astype(float), n_basis=gp_n_bins,
                spacing=_RAISED_COSINE_SPACING['goal_progress']
            )
            # Distance-GP raised-cosine: known [0,1] range; NaN handled by basis
            GPd_enc = make_raised_cosine_basis(
                gpd_all, n_basis=gp_n_bins,
                spacing=_RAISED_COSINE_SPACING['goal_progress_distance'],
                value_range=(0.0, 1.0),
            )
            # Progress since A: raised-cosine with linear spacing
            PGA_enc = make_raised_cosine_basis(
                pga_all.astype(float), n_basis=gp_n_bins,
                spacing=_RAISED_COSINE_SPACING['progress_since_A']
            )
            Sp_enc = _rc(speed_all,             'speed')
            Ac_enc = _rc(acc_all,               'acceleration')
            TF_enc = _rc(tf_all,                'time_from_reward')
            TT_enc = _rc(tt_all,                'time_to_reward')
            TSA_enc = _rc(tsa_all,              'time_since_A')
            TTA_enc = _rc(tta_all,              'time_to_A')
            DF_enc = _rc(df_all,                'distance_from_reward')
            DT_enc = _rc(dt_all,                'distance_to_reward')
        else:
            raise ValueError(
                f"continuous_basis must be 'onehot' or 'raised_cosine', "
                f"got {continuous_basis!r}"
            )

        onehots = {
            'place':                  place_onehot,
            'task_state':             state_onehot,
            'poke_rewarded':          PKR_enc,
            'poke_unrewarded':        PKU_enc,
            'head_direction':         HD_onehot,
            'goal_progress':          GP_enc,
            'goal_progress_distance': GPd_enc,
            'speed':                  Sp_enc,
            'acceleration':           Ac_enc,
            'time_from_reward':       TF_enc,
            'time_to_reward':         TT_enc,
            'time_since_A':           TSA_enc,
            'time_to_A':              TTA_enc,
            'progress_since_A':       PGA_enc,
            'distance_from_reward':   DF_enc,
            'distance_to_reward':     DT_enc,
        }
        if parameterization == 'reference_coded':
            # Mixed reference coding: drop the reference bin from MULTI-column blocks only,
            # leave single-column indicators (pokes) intact, prepend one intercept. See
            # `_resolve_regressor_groups` for why the pokes neither need nor can take a
            # reference bin. The index map there must agree with this exactly.
            blocks = [onehots[name][:, 1:] if onehots[name].shape[1] > 1 else onehots[name]
                      for name in local_names]
            X = np.column_stack([np.ones((blocks[0].shape[0], 1))] + blocks)
        else:
            X = np.column_stack([onehots[name] for name in local_names])
        X = np.nan_to_num(X)

        # Zero-variance guard. A constant column carries no information and costs exactly one
        # rank, so it turns a full-rank design deficient and makes its own CPD identically 0.
        # This is not hypothetical: `poke_unrewarded` is all-zero for any recday whose poke
        # tables were never attached (`attach_pokes`), and `poke_rewarded` would be too --
        # which is precisely how a "16-regressor" fit silently became a 13-regressor one with
        # two dead columns. Report rather than drop, so the caller sees the cause.
        _const = np.flatnonzero(X.std(axis=0) == 0)
        if len(_const):
            _owner = {}
            for _n in local_names:
                for _i in local_groups[_n]:
                    _owner[_i] = _n
            _by_reg = {}
            for _i in _const:
                _by_reg.setdefault(_owner.get(int(_i), 'intercept'), 0)
                _by_reg[_owner.get(int(_i), 'intercept')] += 1
            _by_reg.pop('intercept', None)
            if _by_reg:
                print(f"  WARNING: {len(_const)} zero-variance column(s) in the design "
                      f"{_by_reg} — their CPD will be exactly 0 and the design loses that "
                      f"much rank. For pokes this means `attach_pokes` was not called.")
        print(f"  Design matrix: {X.shape[0]} rows × {X.shape[1]} cols "
              f"(regressors: {local_names}, parameterization={parameterization})")

        if np.any(np.isnan(X)) or np.any(np.isinf(X)) or X.shape[0] < X.shape[1]:
            print(f"  Skipping — degenerate design matrix (shape {X.shape})")
            continue

        # Samples-per-parameter floor. `X.shape[0] < X.shape[1]` above only catches a design
        # that cannot be fitted AT ALL; it says nothing about one that fits and generalises
        # terribly. A production run with `filter_correct_paths=True` produced 376-3471 rows
        # against 123-160 columns and held-out R2 of -0.09 to -2.13 -- every job "succeeded".
        # Cross-validation makes it worse than the raw ratio suggests, because each training
        # fold sees only (n_folds-1)/n_folds of the rows.
        _spp = X.shape[0] / max(X.shape[1], 1)
        if _spp < min_samples_per_param:
            print(f"  Skipping — {X.shape[0]} rows for {X.shape[1]} columns "
                  f"({_spp:.1f} samples/parameter, floor {min_samples_per_param}). "
                  f"Check the transition filters: this is what an over-aggressive "
                  f"`filter_correct_paths` or `max_transition_seconds` looks like.")
            continue
        if _spp < 3 * min_samples_per_param:
            print(f"  WARNING: only {_spp:.1f} samples/parameter "
                  f"({X.shape[0]} rows, {X.shape[1]} cols) — fits will be unstable.")

        # Concatenate FR across sessions matching the same node filter
        FR_all = np.concatenate(
            [prepared_sessions[s]['FR'][:, nf] for s, nf in session_filters], axis=1
        )  # [n_neurons × T_total]

        # ----------------------------------------------------------------
        # Rank report + leave-one-session-out cross-validation
        # ----------------------------------------------------------------
        # The in-sample path below computes RSS, R2, CPD and the nested F on the same
        # samples it fits. That is fine for the permutation SIGNIFICANCE call (the null
        # goes through identical machinery, so the optimism cancels) but not for CPD
        # magnitudes, and the circular-shift null is permissive for slowly-varying
        # regressors -- 80% of LEC units come back "place-tuned", a range of only
        # 70-86% across brain regions, which is too saturated to compare regions with.
        # `cross_validate=True` adds a held-out version alongside, so the two are
        # directly comparable on the same fit.
        _rank = _glm_cv().check_rank(X, name=f'  {mouse_recday} design')
        print(f"  Design rank: {_rank['rank']}/{_rank['n_cols']}"
              + ('' if _rank['full_rank'] else
                 f" (deficient by {_rank['deficiency']}; RSS/R2/CPD valid, "
                 f"betas are min-norm)"))

        if cross_validate:
            cvmod = _glm_cv()
            sess_ids = cvmod.session_ids_from_filters(session_filters)
            n_sess = len(np.unique(sess_ids))
            if n_sess < 2:
                print(f"  CV skipped — only {n_sess} session(s)")
            else:
                t_cv = time.time()
                _fold_ids = (cvmod.within_session_folds(sess_ids)
                             if cv_within_session_folds else None)
                cv_out = cvmod.cv_scores(
                    X, FR_all, sess_ids, local_groups,
                    n_perm=cv_n_perm, joint_specs=joint_specs,
                    center_within_sessions=cv_center_within_sessions,
                    zscore_within_sessions=cv_zscore_within_sessions,
                    fold_ids=_fold_ids, nulls=cv_nulls,
                )
                if cv_poisson:
                    cv_out['poisson'] = cvmod.cv_scores_poisson(
                        X, FR_all, sess_ids, local_groups,
                        joint_specs=joint_specs, neuron_subset=cv_poisson_neurons,
                    )
                cv_out['elapsed_s'] = time.time() - t_cv
                cv_out['rank'] = _rank
                CV_results[mouse_recday] = cv_out
                print(f"  CV: {n_sess} folds, {len(local_groups)} groups, "
                      f"{cv_n_perm} perms — {cv_out['elapsed_s']:.1f}s"
                      f"  median r2_cv={np.nanmedian(cv_out['r2_cv']):.4f}")

        if cv_only:
            # Skip the in-sample per-neuron loop entirely. That loop re-decomposes the same
            # X inside `lstsq` for every neuron and every reduced model and costs ~680 s per
            # recday, against ~0 s for the CV path (which shares one pseudo-inverse per
            # fold/model). Same precedent as `scales_only`: everything that decides WHICH
            # samples enter the fit has already run.
            continue

        if scales_only:
            # Everything that decides WHICH samples enter the fit has already
            # run (session dedup, node filter, transition mask, downsample, and
            # the degenerate-design skip above), so the SDs recorded here are
            # exactly those of the fitted samples. Skipping straight past the
            # lstsq + permutation loop is what makes it cheap to backfill
            # `neuron_scales` onto sections that are already cached, instead of
            # paying for a full refit to recover one number per neuron.
            sd_all, mean_all = FR_all.std(axis=1), FR_all.mean(axis=1)
            for neuron in range(num_neurons):
                Neuron_scales[mouse_recday][neuron] = {
                    'sd': float(sd_all[neuron]), 'mean': float(mean_all[neuron])}
            continue

        # ----------------------------------------------------------------
        # Precompute reduced design matrices once (shared across neurons)
        # ----------------------------------------------------------------
        X_reduced_dict = {
            reg_name: np.delete(X, indices, axis=1)
            for reg_name, indices in local_groups.items()
        }

        # Joint reduced-model design matrices (drop multiple regressor groups together)
        X_reduced_joint = {}
        joint_n_dropped = {}
        for group_name, regs in joint_specs:
            joint_indices = sorted(set().union(*[local_groups[r] for r in regs]))
            X_reduced_joint[group_name] = np.delete(X, joint_indices, axis=1)
            joint_n_dropped[group_name] = len(joint_indices)

        # ----------------------------------------------------------------
        # Pseudo-inverses, computed ONCE per (recday, model) and reused for every neuron
        # and every permutation.
        #
        # The loop below used to call `np.linalg.lstsq` per neuron per model per
        # real/permuted pair -- with 16 regressors + 3 joint groups that is ~40 SVDs of the
        # SAME matrix per neuron, and the decomposition, not the solve, is the cost. One
        # LEC recday took ~1 h. `pinv` is the same minimum-norm solution `lstsq(rcond=None)`
        # returns, so results are unchanged; every fit becomes a matmul.
        #
        # Memory: each P is (n_cols, T) float64 -- ~38 MB at 160x30000, so ~800 MB across
        # ~20 models. That is why they are freed at the end of the recday.
        # ----------------------------------------------------------------
        _pinv = lambda A: np.linalg.pinv(A, rcond=1e-10)
        P_full = _pinv(X)
        P_reduced = {k: _pinv(v) for k, v in X_reduced_dict.items()}
        P_joint = {k: _pinv(v) for k, v in X_reduced_joint.items()}

        T, n_params = X.shape
        df_resid = T - n_params
        shifts = np.random.randint(0, T, size=num_permutations)

        # ----------------------------------------------------------------
        # Per-neuron: OLS fit + permutation F-test (vectorised)
        # ----------------------------------------------------------------
        for neuron in tqdm(range(num_neurons), desc="  Neurons", leave=False):
            frs = FR_all[neuron]
            try:
                # --- Full model ---
                params = P_full @ frs
                resid_full      = frs - X @ params
                rss_full        = resid_full @ resid_full

                # Total sum of squares & full-model R² (for CPD normalization)
                tss = float(np.sum((frs - frs.mean()) ** 2))
                r2_full = (1.0 - rss_full / tss) if tss > 0 else 0.0

                # --- Permuted full models (all at once) ---
                perm_frs        = np.stack([np.roll(frs, s) for s in shifts]).T  # [T × n_perms]
                beta_perms      = P_full @ perm_frs                              # [n_params × n_perms]
                resid_full_p    = perm_frs - X @ beta_perms                       # [T × n_perms]
                rss_full_p      = np.einsum('ij,ij->j', resid_full_p, resid_full_p)  # [n_perms]

                # --- F-stat per regressor group ---
                F_real = {}
                F_perm = {}
                CPD_real = {}
                delta_r2_real = {}   # unique R² = ΔRSS / TSS (for normalized CPD)
                for reg_name, X_r in X_reduced_dict.items():
                    df_num = len(local_groups[reg_name])

                    # Real reduced model
                    params_r = P_reduced[reg_name] @ frs
                    resid_r   = frs - X_r @ params_r
                    rss_r     = resid_r @ resid_r
                    F_real[reg_name] = ((rss_r - rss_full) / df_num) / (rss_full / df_resid)
                    if compute_cpd:
                        CPD_real[reg_name] = ((rss_r - rss_full) / rss_r) if rss_r > 0 else 0.0
                        delta_r2_real[reg_name] = ((rss_r - rss_full) / tss) if tss > 0 else 0.0

                    # Permuted reduced models (vectorised)
                    beta_perms_r = P_reduced[reg_name] @ perm_frs                 # [n_r × n_perms]
                    resid_r_p    = perm_frs - X_r @ beta_perms_r                   # [T × n_perms]
                    rss_r_p      = np.einsum('ij,ij->j', resid_r_p, resid_r_p)    # [n_perms]
                    F_perm[reg_name] = ((rss_r_p - rss_full_p) / df_num) / (rss_full_p / df_resid)

                # --- Joint reduced-model F-stats (and CPDs) ---
                for group_name, X_rj in X_reduced_joint.items():
                    df_num_j = joint_n_dropped[group_name]

                    params_rj = P_joint[group_name] @ frs
                    resid_rj  = frs - X_rj @ params_rj
                    rss_rj    = resid_rj @ resid_rj
                    F_real[group_name] = ((rss_rj - rss_full) / df_num_j) / (rss_full / df_resid)
                    if compute_cpd:
                        CPD_real[group_name] = ((rss_rj - rss_full) / rss_rj) if rss_rj > 0 else 0.0
                        delta_r2_real[group_name] = ((rss_rj - rss_full) / tss) if tss > 0 else 0.0

                    beta_perms_rj = P_joint[group_name] @ perm_frs
                    resid_rj_p    = perm_frs - X_rj @ beta_perms_rj
                    rss_rj_p      = np.einsum('ij,ij->j', resid_rj_p, resid_rj_p)
                    F_perm[group_name] = ((rss_rj_p - rss_full_p) / df_num_j) / (rss_full_p / df_resid)

                GLM_results[mouse_recday][neuron]         = params
                Permutation_results[mouse_recday][neuron] = (F_real, F_perm)
                # SD of this neuron over exactly the samples that were fitted
                # (same session dedup, node filter, transition filter and
                # downsample). Recomputing it outside this loop is the easiest
                # way to introduce a silent mismatch, which is why it is
                # recorded here, keyed identically to GLM_results.
                Neuron_scales[mouse_recday][neuron] = {
                    'sd': float(np.sqrt(tss / T)) if T > 0 else np.nan,
                    'mean': float(frs.mean()),
                }
                if compute_cpd:
                    # Reserved (__-prefixed) keys carry model-fit context alongside
                    # the per-regressor CPDs. Consumers iterating regressor names
                    # must skip keys starting with '__'.
                    CPD_real['__r2_full__'] = r2_full
                    CPD_real['__delta_r2__'] = delta_r2_real
                    CPD_results[mouse_recday][neuron] = CPD_real

            except Exception:
                continue

        del P_full, P_reduced, P_joint

    out = [GLM_results, Permutation_results]
    if compute_cpd:
        out.append(CPD_results)
    if return_scales:
        out.append(Neuron_scales)
    if cross_validate:
        out.append(CV_results)
    return tuple(out)


# ============================================================================
# Significance analysis
# ============================================================================

# Regressors where betas are ordered by bin — slope captures ramp direction
_ORDERED_REGRESSORS = {
    'goal_progress', 'goal_progress_distance',
    'speed', 'acceleration',
    'time_from_reward', 'time_to_reward',
    'time_since_A', 'time_to_A', 'progress_since_A',
    'distance_from_reward', 'distance_to_reward',
}


def _beta_direction(params, reg_name, groups=None):
    """
    Derive a sign (+1/-1) from the beta profile for a regressor group.
    Ordered variables: slope of a linear fit across all bin betas (all bins are
    present in this parameterization — no reference bin is dropped).
    Unordered variables (place, HD): sign of the mean beta.

    `groups` defaults to module-level `regressor_groups`; pass a subset-specific
    mapping (from `_resolve_regressor_groups`) when fitting a regressor subset.
    """
    if groups is None:
        groups = regressor_groups
    betas = params[groups[reg_name]]
    if reg_name in _ORDERED_REGRESSORS:
        slope = np.polyfit(np.arange(len(betas)), betas, 1)[0]
        return int(np.sign(slope)) or 1
    else:
        return int(np.sign(np.mean(betas))) or 1


def compute_tuning_arrays(GLM_results, Permutation_results, regressors_to_include=None,
                          gp_n_bins=10, parameterization='all_bins'):
    """
    Returns tuned_dict: {mouse_recday: tuning_array [n_neurons × n_regressors]}
    Values: +1 (positive / ramp-up tuning), -1 (negative / ramp-down), 0 (not significant).
    Significance: permutation F-test at p < 0.05 (one-sided on F).
    Direction: slope of beta profile for ordered regressors; sign of mean beta for unordered.

    Pass the same `regressors_to_include`, `gp_n_bins`, and `parameterization`
    used in `run_glm_analysis` so that `_beta_direction` indexes the regressor
    betas with the correct (subset-, bin-count-, and coding-specific) index map.
    """
    local_groups, local_names = _resolve_regressor_groups(
        regressors_to_include, gp_n_bins=gp_n_bins,
        parameterization=parameterization,
    )
    tuned_dict = {}

    for mouse_recday in GLM_results:
        if len(GLM_results[mouse_recday]) == 0:
            continue

        neuron_ids   = sorted(GLM_results[mouse_recday].keys())
        n_regressors = len(local_names)
        tuning_array = np.zeros((len(neuron_ids), n_regressors), dtype=int)

        for neuron_idx, neuron in enumerate(neuron_ids):
            params = GLM_results[mouse_recday].get(neuron)
            perm   = Permutation_results[mouse_recday].get(neuron)
            if params is None or perm is None:
                continue
            F_real, F_perm = perm

            for reg_idx, reg_name in enumerate(local_names):
                f_stat = F_real.get(reg_name)
                f_null = F_perm.get(reg_name)
                if f_stat is None or f_null is None:
                    continue

                # F-test: one-sided at p < 0.05
                if f_stat > np.percentile(f_null, 95):
                    tuning_array[neuron_idx, reg_idx] = _beta_direction(
                        params, reg_name, groups=local_groups
                    )

        tuned_dict[mouse_recday] = tuning_array
        print(f"{mouse_recday}: tuning array shape = {tuning_array.shape}")

    return tuned_dict


def group_tuning_by_mouse(tuned_dict):
    mouse_tuning = defaultdict(list)
    for mouse_recday, tuning_array in tuned_dict.items():
        mouse_tuning[mouse_recday[:4]].append(tuning_array)
    return {
        mouse_id: np.concatenate(arrays, axis=0)
        for mouse_id, arrays in sorted(mouse_tuning.items())
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_tuning_piecharts(mouse_tuning_concat, regressors_to_include=None):
    _, local_names = _resolve_regressor_groups(regressors_to_include)
    n_mice = len(mouse_tuning_concat)
    fig, axes = plt.subplots(
        len(local_names), n_mice,
        figsize=(4 * n_mice, 4 * len(local_names))
    )
    if n_mice == 1:
        axes = axes.reshape(-1, 1)
    if len(local_names) == 1:
        axes = axes.reshape(1, -1)

    for col, (mouse_id, all_tuning) in enumerate(sorted(mouse_tuning_concat.items())):
        n_total = all_tuning.shape[0]
        for row, reg_name in enumerate(local_names):
            ax = axes[row, col]
            reg_idx = local_names.index(reg_name)
            col_data = all_tuning[:, reg_idx]
            pos = np.sum(col_data == 1)
            neg = np.sum(col_data == -1)
            not_tuned = np.sum(col_data == 0)

            sizes  = [pos, neg, not_tuned]
            labels = ['Tuned (+)', 'Tuned (-)', 'Not Tuned']
            colors = ['#66b3ff', '#ff9999', '#d3d3d3']
            nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if nonzero:
                sizes, labels, colors = zip(*nonzero)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            if row == 0:
                ax.set_title(f"Mouse {mouse_id}\n(n={n_total})", fontsize=11)
            if col == 0:
                ax.set_ylabel(reg_name, fontsize=9)

    plt.suptitle("LEC: Fraction of neurons tuned per regressor", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_gp_overlap(mouse_tuning_concat, regressors_to_include=None):
    _, local_names = _resolve_regressor_groups(regressors_to_include)

    # Required regressors for this overlap plot
    required = ['goal_progress', 'place', 'time_from_reward',
                'distance_from_reward', 'speed', 'acceleration']
    missing = [r for r in required if r not in local_names]
    if missing:
        print(f"plot_gp_overlap: skipping — required regressors missing from subset: {missing}")
        return

    gp_idx        = local_names.index('goal_progress')
    place_idx     = local_names.index('place')
    time_from_idx = local_names.index('time_from_reward')
    dist_from_idx = local_names.index('distance_from_reward')
    speed_idx     = local_names.index('speed')
    acc_idx       = local_names.index('acceleration')

    n_mice = len(mouse_tuning_concat)
    fig, axes = plt.subplots(6, n_mice, figsize=(4 * n_mice, 24),
                             gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1, 1, 1]})
    if n_mice == 1:
        axes = axes.reshape(6, 1)

    bar_colors = ['#d3d3d3', '#66b3ff', '#ff9999', '#7a4f9e']
    bar_rows = [
        (['GP only', 'GP + time', 'GP + dist', 'GP + both'], time_from_idx, dist_from_idx),
        (['GP only', 'GP + speed', 'GP + acc',  'GP + both'], speed_idx,     acc_idx),
    ]

    for col, (mouse_id, all_tuning) in enumerate(sorted(mouse_tuning_concat.items())):
        n_total    = all_tuning.shape[0]
        gp_mask    = all_tuning[:, gp_idx]    != 0
        place_mask = all_tuning[:, place_idx] != 0
        speed_mask = all_tuning[:, speed_idx] != 0
        n_gp       = np.sum(gp_mask)

        # Row 0: GP pie
        ax = axes[0, col]
        pos = np.sum(all_tuning[:, gp_idx] == 1)
        neg = np.sum(all_tuning[:, gp_idx] == -1)
        sizes  = [pos, neg, n_total - n_gp]
        labels = ['Tuned (+)', 'Tuned (-)', 'Not Tuned']
        colors = ['#66b3ff', '#ff9999', '#d3d3d3']
        nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if nonzero:
            sizes, labels, colors = zip(*nonzero)
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(f"Mouse {mouse_id}\n(n={n_total} neurons)", fontsize=11)

        # Rows 1 & 2: 4-way bar charts among GP cells
        for row, (categories, idx_a, idx_b) in enumerate(bar_rows, start=1):
            ax = axes[row, col]
            if n_gp == 0:
                ax.axis('off')
                continue
            gp_tuned = all_tuning[gp_mask]
            a_tuned  = gp_tuned[:, idx_a] != 0
            b_tuned  = gp_tuned[:, idx_b] != 0
            counts = np.array([
                np.sum(~a_tuned & ~b_tuned),
                np.sum( a_tuned & ~b_tuned),
                np.sum(~a_tuned &  b_tuned),
                np.sum( a_tuned &  b_tuned),
            ])
            pcts = 100 * counts / n_gp
            bars = ax.bar(categories, pcts, color=bar_colors, edgecolor='white', linewidth=0.8)
            for bar, pct, cnt in zip(bars, pcts, counts):
                if pct > 3:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)
            if row == 1:
                ax.set_title(f"GP cells (n={n_gp})", fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of GP-tuned neurons" if col == 0 else "")
            ax.tick_params(axis='x', rotation=30)

        # Rows 3–5: single-variable overlap pies
        for row, (denom_mask, denom_label, cross_mask, slice_label, yes_color) in enumerate([
            (gp_mask,    'GP cells',    place_mask, 'Also place tuned', '#ff9999'),
            (speed_mask, 'Speed cells', gp_mask,    'Also GP tuned',    '#66b3ff'),
            (place_mask, 'Place cells', gp_mask,    'Also GP tuned',    '#66b3ff'),
        ], start=3):
            ax = axes[row, col]
            n_denom = np.sum(denom_mask)
            if n_denom == 0:
                ax.set_title(f"(no {denom_label.lower()})")
                ax.axis('off')
                continue
            yes = np.sum(denom_mask & cross_mask)
            no  = n_denom - yes
            sizes  = [yes, no]
            labels = [slice_label, 'Not tuned']
            colors = [yes_color, '#d3d3d3']
            nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if nonzero:
                sizes, labels, colors = zip(*nonzero)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.set_title(f"{denom_label} (n={n_denom})", fontsize=10)

    row_label_texts = [
        'Goal progress\ntuning', 'GP × time / dist', 'GP × speed / acc',
        'GP cells\n→ place overlap', 'Speed cells\n→ GP overlap', 'Place cells\n→ GP overlap',
    ]
    for row, text in enumerate(row_label_texts):
        axes[row, 0].annotate(text, xy=(0, 0.5), xytext=(-0.35, 0.5),
                              xycoords='axes fraction', textcoords='axes fraction',
                              fontsize=11, fontweight='bold', va='center', rotation=90)

    plt.suptitle("Goal progress neurons: overlap with place, temporal, and kinematic regressors",
                 fontsize=13)
    plt.tight_layout()
    plt.show()


def print_tuning_summary(mouse_tuning_concat, regressors_to_include=None):
    _, local_names = _resolve_regressor_groups(regressors_to_include)
    print("\nTuning Summary:")
    for mouse_id, all_tuning in sorted(mouse_tuning_concat.items()):
        total = all_tuning.shape[0]
        print(f"\nMouse {mouse_id} (n={total}):")
        for reg_idx, reg_name in enumerate(local_names):
            col = all_tuning[:, reg_idx]
            pos = np.sum(col == 1)
            neg = np.sum(col == -1)
            print(f"  {reg_name}: {pos+neg}/{total} ({100*(pos+neg)/total:.1f}%)  "
                  f"[{pos}+ / {neg}-]")


# ============================================================================
# Extended analysis: simplified pie charts, time-vs-progress overlap, CPD
# ============================================================================

def plot_tuning_piecharts_binary(mouse_tuning_concat, regressors_to_include=None):
    """Two-slice pie charts (Tuned vs Not Tuned) per regressor.

    Like `plot_tuning_piecharts` but collapses +1/-1 into a single 'tuned' slice.
    Uses pretty display names from `_REGRESSOR_DISPLAY_NAMES`.
    """
    _, local_names = _resolve_regressor_groups(regressors_to_include)
    n_mice = len(mouse_tuning_concat)
    fig, axes = plt.subplots(
        len(local_names), n_mice,
        figsize=(4 * n_mice, 4 * len(local_names))
    )
    if n_mice == 1:
        axes = axes.reshape(-1, 1)
    if len(local_names) == 1:
        axes = axes.reshape(1, -1)

    for col, (mouse_id, all_tuning) in enumerate(sorted(mouse_tuning_concat.items())):
        n_total = all_tuning.shape[0]
        for row, reg_name in enumerate(local_names):
            ax = axes[row, col]
            reg_idx = local_names.index(reg_name)
            col_data = all_tuning[:, reg_idx]
            tuned = int(np.sum(col_data != 0))
            not_tuned = int(np.sum(col_data == 0))

            sizes = [tuned, not_tuned]
            labels = ['Tuned', 'Not Tuned']
            colors = ['#4c9be8', '#d3d3d3']
            nonzero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if nonzero:
                sizes, labels, colors = zip(*nonzero)
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            if row == 0:
                ax.set_title(f"Mouse {mouse_id}\n(n={n_total})", fontsize=11)
            if col == 0:
                ax.set_ylabel(_display(reg_name), fontsize=10)

    plt.suptitle("LEC: Fraction of neurons significantly tuned per regressor", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_time_vs_progress_overlap(mouse_tuning_concat,
                                   regressors_to_include=None,
                                   use_joint_time=False):
    """Stacked-bar comparison of goal-progress vs time tuning, per mouse + aggregate.

    For each mouse, splits neurons into 4 categories:
      - GP only
      - Time only (tuned to time_from_reward OR time_to_reward — OR rule)
      - Both
      - Neither

    If `use_joint_time=True`, expects a column in `mouse_tuning_concat` named
    `time_any` (from `joint_drop_groups`) and uses that instead of the OR rule.
    """
    _, local_names = _resolve_regressor_groups(regressors_to_include)

    # Locate columns
    if 'goal_progress' not in local_names:
        print("plot_time_vs_progress_overlap: skipping — 'goal_progress' missing")
        return
    gp_col = local_names.index('goal_progress')

    if use_joint_time:
        if 'time_any' not in local_names:
            print("plot_time_vs_progress_overlap: 'time_any' not in tuning array; "
                  "fit with joint_drop_groups=[('time_any', [...])] and recompute "
                  "tuning_arrays with regressors_to_include including 'time_any'.")
            return
        time_col_indices = [local_names.index('time_any')]
    else:
        time_names_present = [r for r in ('time_from_reward', 'time_to_reward')
                              if r in local_names]
        if not time_names_present:
            print("plot_time_vs_progress_overlap: skipping — no time regressors present")
            return
        time_col_indices = [local_names.index(r) for r in time_names_present]

    n_mice = len(mouse_tuning_concat)
    mouse_ids = sorted(mouse_tuning_concat.keys())

    # Aggregate counts across mice for the summary panel
    agg_counts = np.zeros(4, dtype=int)  # [gp_only, time_only, both, neither]
    per_mouse = {}

    for mouse_id in mouse_ids:
        all_tuning = mouse_tuning_concat[mouse_id]
        gp_tuned = (all_tuning[:, gp_col] != 0)
        time_tuned = np.any(all_tuning[:, time_col_indices] != 0, axis=1)

        gp_only  = int(np.sum(gp_tuned & ~time_tuned))
        time_only = int(np.sum(~gp_tuned & time_tuned))
        both      = int(np.sum(gp_tuned & time_tuned))
        neither   = int(np.sum(~gp_tuned & ~time_tuned))

        per_mouse[mouse_id] = np.array([gp_only, time_only, both, neither])
        agg_counts += per_mouse[mouse_id]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.5 * n_mice + 4), 5),
                              gridspec_kw={'width_ratios': [n_mice, 1]})

    cat_labels = ['GP only', 'Time only', 'Both', 'Neither']
    cat_colors = ['#d62728', '#1f77b4', '#7a4f9e', '#d3d3d3']

    ax = axes[0]
    bar_bottom = np.zeros(n_mice)
    totals = np.array([per_mouse[m].sum() for m in mouse_ids])
    for k, (lbl, c) in enumerate(zip(cat_labels, cat_colors)):
        counts_k = np.array([per_mouse[m][k] for m in mouse_ids])
        pcts = 100 * counts_k / np.where(totals > 0, totals, 1)
        ax.bar(np.arange(n_mice), pcts, bottom=bar_bottom,
               color=c, edgecolor='white', linewidth=0.5, label=lbl)
        bar_bottom += pcts
    ax.set_xticks(np.arange(n_mice))
    ax.set_xticklabels([f"{m}\n(n={t})" for m, t in zip(mouse_ids, totals)],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('% of neurons')
    ax.set_ylim(0, 105)
    ax.set_title(
        f"GP vs time tuning per mouse "
        f"({'joint time' if use_joint_time else 'OR-rule across time vars'})"
    )
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)

    # Aggregate pie
    ax = axes[1]
    nonzero = [(c, l, col) for c, l, col in zip(agg_counts, cat_labels, cat_colors) if c > 0]
    if nonzero:
        sizes_, labels_, colors_ = zip(*nonzero)
        ax.pie(sizes_, labels=labels_, autopct='%1.1f%%', startangle=90, colors=colors_)
    total_agg = int(agg_counts.sum())
    ax.set_title(f'All mice pooled\n(n={total_agg})')

    plt.tight_layout()
    plt.show()


def _cpd_value(cpd, reg, normalize):
    """Per-regressor CPD value, optionally normalized to full-model R².

    normalize='none'    → standard CPD = ΔRSS / RSS_reduced
    normalize='r2_full' → ΔR²_reg / R²_full = unique variance as a fraction of
                          the model's total explainable variance.
    Returns np.nan if the regressor or required reserved keys are absent.
    """
    if normalize == 'none':
        return cpd.get(reg, np.nan)
    elif normalize == 'r2_full':
        dr2 = cpd.get('__delta_r2__', {})
        r2f = cpd.get('__r2_full__', np.nan)
        if reg not in dr2 or not np.isfinite(r2f) or r2f <= 0:
            return np.nan
        return dr2[reg] / r2f
    else:
        raise ValueError(f"normalize must be 'none' or 'r2_full', got {normalize!r}")


def plot_cpd_time_vs_progress(CPD_results, group_by_mouse=True, normalize='none'):
    """Compare per-neuron CPD for goal_progress vs time (any).

    CPD_time is taken from `time_any` if present (joint reduced model), else
    falls back to `max(CPD_time_from_reward, CPD_time_to_reward)`.

    Parameters
    ----------
    CPD_results : dict {mouse_recday: {neuron_idx: cpd_dict}}
        Output of `run_glm_analysis(..., compute_cpd=True)`. Reserved keys
        '__r2_full__' / '__delta_r2__' carry the model-fit context.
    group_by_mouse : bool, default True
        Color scatter points by mouse identity.
    normalize : {'none', 'r2_full'}, default 'none'
        'none'    → standard CPD (ΔRSS / RSS_reduced).
        'r2_full' → CPD normalized to full-model R² (fraction of the model's
                    explainable variance uniquely attributable to the regressor).
                    Requires `compute_cpd=True` was used in the fit.

    Returns
    -------
    cpd_gp, cpd_time : 1d arrays of per-neuron values (pooled across recdays)
    """
    cpd_gp = []
    cpd_time = []
    mouse_tag = []

    for mr, neuron_dict in CPD_results.items():
        mouse = mr.split('_')[0]
        for nidx, cpd in neuron_dict.items():
            g = _cpd_value(cpd, 'goal_progress', normalize)
            if 'time_any' in cpd:
                t = _cpd_value(cpd, 'time_any', normalize)
            else:
                t_f = _cpd_value(cpd, 'time_from_reward', normalize)
                t_t = _cpd_value(cpd, 'time_to_reward', normalize)
                t = np.nanmax([t_f, t_t]) if not (np.isnan(t_f) and np.isnan(t_t)) else np.nan
            cpd_gp.append(g)
            cpd_time.append(t)
            mouse_tag.append(mouse)

    cpd_gp = np.array(cpd_gp, dtype=float)
    cpd_time = np.array(cpd_time, dtype=float)
    mouse_tag = np.array(mouse_tag)

    valid = ~np.isnan(cpd_gp) & ~np.isnan(cpd_time)
    cpd_gp = cpd_gp[valid]
    cpd_time = cpd_time[valid]
    mouse_tag = mouse_tag[valid]

    metric_label = 'CPD' if normalize == 'none' else 'CPD / R²_full'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: scatter
    ax = axes[0]
    mice = sorted(np.unique(mouse_tag).tolist()) if group_by_mouse else [None]
    cmap = plt.get_cmap('tab10')
    for i, m in enumerate(mice):
        mask = (mouse_tag == m) if m is not None else slice(None)
        ax.scatter(cpd_gp[mask], cpd_time[mask], alpha=0.6, s=20,
                   color=cmap(i % 10), label=m)
    lim = max(0.001, max(np.nanmax(cpd_gp), np.nanmax(cpd_time)) * 1.05)
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f'{metric_label} goal progress')
    ax.set_ylabel(f'{metric_label} time (any)')
    ax.set_title(f'Per-neuron {metric_label}: goal progress vs time')
    if group_by_mouse:
        ax.legend(loc='upper right', fontsize=8, title='Mouse')

    # Panel 2: Δ histogram
    ax = axes[1]
    diffs = cpd_time - cpd_gp
    ax.hist(diffs, bins=40, color='darkorange', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    md = float(np.nanmean(diffs))
    ax.axvline(md, color='red', linewidth=2, label=f'Mean Δ = {md:+.4f}')
    ax.set_xlabel(f'{metric_label}_time − {metric_label}_progress')
    ax.set_ylabel('Number of neurons')
    ax.set_title('Δ (time − progress)')
    ax.legend()

    # Panel 3: mean metric per regressor across all neurons
    ax = axes[2]
    all_regressors_cpd = {}
    for mr, neuron_dict in CPD_results.items():
        for cpd in neuron_dict.values():
            for reg in cpd:
                if reg.startswith('__'):   # skip reserved keys
                    continue
                v = _cpd_value(cpd, reg, normalize)
                all_regressors_cpd.setdefault(reg, []).append(v)
    means = {r: float(np.nanmean(vs)) for r, vs in all_regressors_cpd.items()}
    sems = {r: float(np.nanstd(vs) / np.sqrt(max(1, np.sum(~np.isnan(vs)))))
            for r, vs in all_regressors_cpd.items()}
    regs_sorted = sorted(means.keys(), key=lambda r: -means[r])
    xs = np.arange(len(regs_sorted))
    ax.bar(xs,
           [means[r] for r in regs_sorted],
           yerr=[sems[r] for r in regs_sorted],
           color='steelblue', alpha=0.8, edgecolor='black', capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([_display(r) for r in regs_sorted], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'Mean {metric_label} across neurons')
    ax.set_title(f'Population-mean {metric_label} per regressor')

    plt.suptitle(
        f"{'Raw' if normalize == 'none' else 'R²-normalized'} CPD: time vs goal progress",
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return cpd_gp, cpd_time


def plot_cpd_pair(CPD_results, key_x, key_y, group_by_mouse=True,
                  normalize='none', label_x=None, label_y=None, title=None):
    """Per-neuron CPD scatter + Δ histogram for ANY two CPD keys.

    Generic sibling of `plot_cpd_time_vs_progress`, which hardcodes
    goal_progress vs time_any. Use this for regressor pairs that function
    doesn't know about — e.g. the loop-anchored group against the per-leg one:

        plot_cpd_pair(CPD, 'time_any', 'since_A_any')

    `key_x` / `key_y` are CPD dict keys, so they may be single regressors
    ('time_since_A') or joint groups declared via `joint_drop_groups`
    ('since_A_any'). Labels default to `_display(key)`.

    Returns (x, y) arrays of per-neuron values, pooled across recdays.
    """
    xs, ys, mouse_tag = [], [], []
    for mr, neuron_dict in CPD_results.items():
        mouse = mr.split('_')[0]
        for cpd in neuron_dict.values():
            xs.append(_cpd_value(cpd, key_x, normalize))
            ys.append(_cpd_value(cpd, key_y, normalize))
            mouse_tag.append(mouse)

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    mouse_tag = np.array(mouse_tag)
    valid = ~np.isnan(x) & ~np.isnan(y)
    x, y, mouse_tag = x[valid], y[valid], mouse_tag[valid]

    if x.size == 0:
        print(f"plot_cpd_pair: no neurons with both {key_x!r} and {key_y!r} "
              f"(present keys need joint_drop_groups / regressors_to_include "
              f"to include them)")
        return x, y

    lx = label_x or _display(key_x)
    ly = label_y or _display(key_y)
    metric = 'CPD' if normalize == 'none' else 'CPD / R²_full'

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    mice = sorted(np.unique(mouse_tag).tolist()) if group_by_mouse else [None]
    cmap = plt.get_cmap('tab10')
    for i, m in enumerate(mice):
        mask = (mouse_tag == m) if m is not None else slice(None)
        ax.scatter(x[mask], y[mask], alpha=0.6, s=20, color=cmap(i % 10), label=m)
    lim = max(0.001, max(np.nanmax(x), np.nanmax(y)) * 1.05)
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
    ax.set_xlabel(f'{metric} {lx}')
    ax.set_ylabel(f'{metric} {ly}')
    ax.set_title(f'Per-neuron {metric}')
    if group_by_mouse:
        ax.legend(loc='upper right', fontsize=8, title='Mouse')

    ax = axes[1]
    d = y - x
    ax.hist(d, bins=40, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    md = float(np.nanmean(d))
    ax.axvline(md, color='red', linewidth=2, label=f'Mean Δ = {md:+.4f}')
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(d)
        ax.set_title(f'Δ ({ly} − {lx})   Wilcoxon p = {p:.2g}')
    except Exception:
        ax.set_title(f'Δ ({ly} − {lx})')
    ax.set_xlabel(f'{metric}[{ly}] − {metric}[{lx}]')
    ax.set_ylabel('Number of neurons')
    ax.legend()

    plt.suptitle(title or f'{metric}: {ly} vs {lx}   (n = {x.size} neurons)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    return x, y


def plot_cpd_task_state_vs_place(CPD_results, group_by_mouse=True, normalize='none'):
    """Compare per-neuron CPD for task_state vs place.

    Task state (A/B/C/D) correlates with place within a task but should
    decorrelate across tasks (state remaps). This plot shows per-neuron
    variance explained by each, with one point per neuron.

    Parameters
    ----------
    CPD_results : dict {mouse_recday: {neuron_idx: cpd_dict}}
        Output of `run_glm_analysis(..., compute_cpd=True)` with task_state
        included in regressors.
    group_by_mouse : bool, default True
        If True, produce one scatter panel per mouse + a pooled panel.
        If False, produce a single pooled panel.
    normalize : {'none', 'r2_full'}, default 'none'
        'none': raw CPD = ΔRSS / RSS_reduced.
        'r2_full': ΔR²_reg / R²_full (unique variance as fraction of model's explainable variance).

    Returns
    -------
    cpd_ts, cpd_place : np.ndarray
        Pooled per-neuron CPD arrays (task_state and place).
    """
    cpd_ts, cpd_place, mouse_tag = [], [], []

    for mr, neuron_dict in CPD_results.items():
        mouse = mr.split('_')[0]
        for cpd in neuron_dict.values():
            ts = _cpd_value(cpd, 'task_state', normalize)
            pl = _cpd_value(cpd, 'place', normalize)
            if np.isfinite(ts) and np.isfinite(pl):
                cpd_ts.append(ts)
                cpd_place.append(pl)
                mouse_tag.append(mouse)

    cpd_ts = np.array(cpd_ts)
    cpd_place = np.array(cpd_place)
    mouse_tag = np.array(mouse_tag)

    mice = sorted(np.unique(mouse_tag))
    n_mice = len(mice)
    n_panels = n_mice + 1 if group_by_mouse else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes.flatten()

    # Per-mouse scatter plots
    for i, mouse in enumerate(mice):
        ax = axes[i]
        msk = mouse_tag == mouse
        ax.scatter(cpd_place[msk], cpd_ts[msk], alpha=0.6, s=18, color='steelblue')
        lim = max(0.001, max(cpd_place[msk].max(), cpd_ts[msk].max()) * 1.05)
        ax.plot([0, lim], [0, lim], 'k--', lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('CPD place')
        ax.set_ylabel('CPD task_state')
        ax.set_title(f'{mouse} (n={msk.sum()})')

    # Pooled scatter
    if group_by_mouse:
        ax = axes[-1]
        for i, mouse in enumerate(mice):
            msk = mouse_tag == mouse
            ax.scatter(cpd_place[msk], cpd_ts[msk], alpha=0.6, s=18,
                      color=plt.get_cmap('tab10')(i % 10), label=mouse)
        lim = max(0.001, max(cpd_place.max(), cpd_ts.max()) * 1.05)
        ax.plot([0, lim], [0, lim], 'k--', lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('CPD place')
        ax.set_ylabel('CPD task_state')
        ax.set_title(f'Pooled (n={len(cpd_ts)})')
        ax.legend(fontsize=8, title='Mouse')

    # Difference histogram (single panel below or on the right)
    d = cpd_ts - cpd_place
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(d, bins=40, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color='black', ls='--', lw=1.5)
    ax2.axvline(d.mean(), color='red', lw=2, label=f'Mean Δ = {d.mean():+.5f}')
    from scipy import stats as _st
    tval, pval = _st.ttest_1samp(d, 0)
    ax2.text(0.05, 0.95, f'n={len(d)}\nt={tval:.2f}\np={pval:.2e}',
            transform=ax2.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.set_xlabel('CPD_task_state − CPD_place')
    ax2.set_ylabel('# neurons')
    ax2.set_title('Task state vs place: ΔR² (task_state favours negative)')
    ax2.legend(fontsize=9)
    plt.tight_layout()

    fig.suptitle('Task state vs place: per-neuron CPD', fontweight='bold', y=1.02)
    plt.tight_layout()

    return cpd_ts, cpd_place


def plot_task_state_place_correlation_heatmap(mouse_recdays, data_dic,
                                               downsample_factor=10,
                                               max_recdays=None):
    """Heatmap of |correlations| between task_state and place one-hot columns.

    Shows a 4×21 matrix: task_state one-hots (rows; states A/B/C/D) vs
    place one-hots (columns; nodes 1-21). Entry [i,j] = |correlation|
    between state_i and place_j, pooled across all recdays/sessions/tasks.

    High correlation (warm colors) in a cell suggests that state i is
    systematically associated with a particular place; values close to 0
    (cool colors) suggest good decorrelation.

    Parameters
    ----------
    mouse_recdays, data_dic : as in `run_glm_analysis`.
    downsample_factor : int
        Match what you pass to `run_glm_analysis`.
    max_recdays : int or None
        Cap on recdays to pool (None = all).

    Returns
    -------
    corr_matrix : ndarray (4 × 21)
        Absolute correlation values.
    """
    recdays_used = mouse_recdays[:max_recdays] if max_recdays else list(mouse_recdays)
    all_state = []
    all_locs = []

    for mr in recdays_used:
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        for s in sessions:
            prep = prepare_session_data(data_dic[mr][s])
            prep = truncate_all_arrays(prep)
            prep = downsample_session_data(prep, downsample_factor)
            nf = prep['Locs'] <= 21
            all_state.append(prep['State'][nf])
            all_locs.append(prep['Locs'][nf])

    state_pooled = np.concatenate(all_state).astype(int)
    locs_pooled = np.concatenate(all_locs).astype(int)

    # One-hot encode
    state_onehot = (state_pooled[:, None] == np.arange(_N_TASK_STATES)).astype(float)
    place_onehot = (locs_pooled[:, None] == np.arange(1, 22)).astype(float)

    # Compute correlation matrix
    from scipy.stats import pearsonr
    corr_matrix = np.zeros((_N_TASK_STATES, 21))
    for i in range(_N_TASK_STATES):
        for j in range(21):
            r, _ = pearsonr(state_onehot[:, i], place_onehot[:, j])
            corr_matrix[i, j] = abs(r)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(corr_matrix, aspect='auto', cmap='hot', vmin=0, vmax=1)
    ax.set_xticks(np.arange(21))
    ax.set_xticklabels(np.arange(1, 22))
    ax.set_yticks(np.arange(_N_TASK_STATES))
    ax.set_yticklabels(['A (state 0)', 'B (state 1)', 'C (state 2)', 'D (state 3)'])
    ax.set_xlabel('Place (node)')
    ax.set_ylabel('Task state')
    ax.set_title(f'|Correlation| between task_state and place (recdays={len(recdays_used)})')
    plt.colorbar(im, ax=ax, label='|r|')

    # Add text annotations (correlation values)
    for i in range(_N_TASK_STATES):
        for j in range(21):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if corr_matrix[i, j] < 0.5 else "white",
                          fontsize=6)

    plt.tight_layout()

    return corr_matrix


def plot_task_state_place_tuning_overlap(Permutation_results,
                                         p_threshold=95):
    """Plot tuning overlap: "task_state" vs "place" populations.

    Produces a 2×2 contingency table showing the four populations:
    - tuned to task_state only
    - tuned to place only
    - tuned to both
    - tuned to neither

    This reveals whether neurons that encode task state also encode place
    (would expect high overlap if they're correlated within-task, but
    overlap should drop if state remaps across tasks).

    Parameters
    ----------
    Permutation_results : dict {mouse_recday: {neuron_idx: (F_real_dict, F_perm_dict)}}
        Output of `run_glm_analysis(..., compute_cpd=True)`.
    p_threshold : float, default 95
        Percentile of F_perm for "tuned" classification (default: p < 0.05).

    Returns
    -------
    contingency : ndarray (2 × 2)
        Counts: [[neither, place_only], [task_state_only, both]]
    """
    tuned_ts = []
    tuned_place = []

    for mr, neuron_dict in Permutation_results.items():
        for neuron, (F_real, F_perm) in neuron_dict.items():
            f_ts = F_real.get('task_state')
            f_pl = F_real.get('place')
            f_ts_null = F_perm.get('task_state')
            f_pl_null = F_perm.get('place')

            if f_ts is not None and f_ts_null is not None:
                tuned_ts.append(f_ts > np.percentile(f_ts_null, p_threshold))
            else:
                tuned_ts.append(False)

            if f_pl is not None and f_pl_null is not None:
                tuned_place.append(f_pl > np.percentile(f_pl_null, p_threshold))
            else:
                tuned_place.append(False)

    tuned_ts = np.array(tuned_ts)
    tuned_place = np.array(tuned_place)

    # Contingency table
    neither = np.sum(~tuned_ts & ~tuned_place)
    place_only = np.sum(~tuned_ts & tuned_place)
    ts_only = np.sum(tuned_ts & ~tuned_place)
    both = np.sum(tuned_ts & tuned_place)
    contingency = np.array([[neither, place_only], [ts_only, both]])

    # Plot as a heatmap-style table
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(contingency, cmap='YlOrRd', aspect='auto')

    # Add counts and percentages to each cell
    for i in range(2):
        for j in range(2):
            count = contingency[i, j]
            pct = 100 * count / (np.sum(contingency) + 1e-10)
            text = ax.text(j, i, f'{count}\n({pct:.1f}%)',
                          ha="center", va="center", color="black" if count < np.sum(contingency) / 2 else "white",
                          fontsize=12, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Not tuned to place', 'Tuned to place'], fontsize=11)
    ax.set_yticklabels(['Not tuned to task_state', 'Tuned to task_state'], fontsize=11)
    ax.set_ylabel('Task state tuning', fontsize=12, fontweight='bold')
    ax.set_xlabel('Place tuning', fontsize=12, fontweight='bold')
    ax.set_title(f'Tuning overlap: task_state vs place (p < {100-p_threshold:.1f}%, n={np.sum(contingency)} neurons)',
                fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='# neurons')
    plt.tight_layout()

    return contingency


def plot_full_model_r2(CPD_results):
    """Histogram of per-neuron full-model R² (from `compute_cpd=True` fits).

    Anchors the magnitude of CPD values: shows how much firing-rate variance the
    full GLM explains at all. Per-mouse panels + pooled.

    Returns
    -------
    r2_by_mouse : dict {mouse: np.array of per-neuron R²}
    """
    from collections import defaultdict
    r2_by_mouse = defaultdict(list)
    for mr, neuron_dict in CPD_results.items():
        mouse = mr.split('_')[0]
        for cpd in neuron_dict.values():
            r2 = cpd.get('__r2_full__', np.nan)
            if np.isfinite(r2):
                r2_by_mouse[mouse].append(r2)
    r2_by_mouse = {m: np.array(v) for m, v in sorted(r2_by_mouse.items())}

    pooled = np.concatenate(list(r2_by_mouse.values())) if r2_by_mouse else np.array([])

    n_mice = len(r2_by_mouse)
    fig, axes = plt.subplots(1, n_mice + 1, figsize=(4 * (n_mice + 1), 4), squeeze=False)
    axes = axes[0]

    for i, (mouse, r2) in enumerate(r2_by_mouse.items()):
        ax = axes[i]
        ax.hist(r2, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
        ax.axvline(np.median(r2), color='red', linewidth=2,
                   label=f'median = {np.median(r2):.3f}')
        ax.set_xlabel('Full-model R²')
        if i == 0:
            ax.set_ylabel('Number of neurons')
        ax.set_title(f'Mouse {mouse}\n(n={len(r2)})')
        ax.legend(fontsize=8)

    # Pooled
    ax = axes[-1]
    ax.hist(pooled, bins=40, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.axvline(np.median(pooled), color='red', linewidth=2,
               label=f'median = {np.median(pooled):.3f}')
    ax.set_xlabel('Full-model R²')
    ax.set_title(f'All mice pooled\n(n={len(pooled)})')
    ax.legend(fontsize=8)

    plt.suptitle('Full-model R² per neuron (how much variance the GLM explains)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return r2_by_mouse


# ============================================================================
# Decile-edge sanity check (pre-fit diagnostic)
# ============================================================================

# Maps canonical regressor names → per-session array keys in prepare_session_data
_DECILE_PREP_KEYS = {
    'time_from_reward':     'time_from_reward',
    'time_to_reward':       'time_to_reward',
    'distance_from_reward': 'dist_from_reward',
    'distance_to_reward':   'dist_to_reward',
    'speed':                'Speed',
    'acceleration':         'Acc',
}


def plot_decile_distributions(mouse_recdays, data_dic,
                              downsample_factor=10,
                              regressors=None,
                              n_bins=10, outlier_pct=1,
                              max_recdays=None):
    """Histogram the pooled regressor values that feed `compute_decile_edges`,
    with the decile edges and outlier-clip percentiles overlaid.

    Pools the SAME values that `run_glm_analysis` uses internally (after the
    Locs<=21 node filter and after downsampling), per regressor across all
    included recdays. Lets you sanity-check whether each variable's bin
    boundaries land where you'd expect (e.g. that time_from_reward isn't so
    spiky at 0 that the first few deciles collapse).

    Parameters
    ----------
    mouse_recdays, data_dic : as in `run_glm_analysis`.
    downsample_factor : int
        Match what you pass to `run_glm_analysis`.
    regressors : list of str or None
        Subset of the 6 deciled regressors to inspect. Accepts aliases
        (`time_since_reward` etc.). None → all 6.
    n_bins, outlier_pct : as in `compute_decile_edges`.
    max_recdays : int or None
        Cap on recdays to pool (None = all). Use 3–5 for a fast pass.

    Returns
    -------
    pooled : dict {canonical_name: np.ndarray of finite pooled values}
    """
    if regressors is None:
        regressors = ['time_from_reward', 'time_to_reward',
                      'distance_from_reward', 'distance_to_reward',
                      'speed', 'acceleration']
    # Resolve aliases and validate
    canonical = [_REGRESSOR_NAME_ALIASES.get(r, r) for r in regressors]
    invalid = [r for r in canonical if r not in _DECILE_PREP_KEYS]
    if invalid:
        raise ValueError(f"plot_decile_distributions only supports deciled regressors. "
                         f"Unknown / non-deciled: {invalid}. "
                         f"Valid: {list(_DECILE_PREP_KEYS.keys())}")

    recdays_used = mouse_recdays[:max_recdays] if max_recdays else list(mouse_recdays)
    pooled = {r: [] for r in canonical}

    for mr in recdays_used:
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        for s in sessions:
            prep = prepare_session_data(data_dic[mr][s])
            prep = truncate_all_arrays(prep)
            prep = downsample_session_data(prep, downsample_factor)
            nf = prep['Locs'] <= 21
            for r in canonical:
                vals = prep[_DECILE_PREP_KEYS[r]][nf]
                pooled[r].append(vals)
    pooled = {r: np.concatenate(v) if v else np.array([])
              for r, v in pooled.items()}

    # Filter to finite values for plotting / edge computation
    pooled_finite = {r: v[np.isfinite(v)] for r, v in pooled.items()}

    n = len(canonical)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, r in zip(axes, canonical):
        vals = pooled_finite[r]
        if len(vals) < 2 * n_bins:
            ax.set_title(f'{_display(r)} (insufficient data n={len(vals)})')
            ax.axis('off')
            continue

        lo, hi = np.percentile(vals, [outlier_pct, 100 - outlier_pct])
        edges = compute_decile_edges(vals, n_bins=n_bins, outlier_pct=outlier_pct)
        # Inner edges only (compute_decile_edges replaces the outer two with +-inf)
        inner = edges[1:-1]

        # Plot histogram restricted to ~[lo, hi] for readability + a small margin
        margin = 0.02 * (hi - lo if hi > lo else 1.0)
        plot_lo, plot_hi = lo - margin, hi + margin
        ax.hist(vals[(vals >= plot_lo) & (vals <= plot_hi)],
                bins=80, color='steelblue', alpha=0.7, edgecolor='black',
                linewidth=0.3)

        for e in inner:
            ax.axvline(e, color='red', lw=1.0, alpha=0.7)
        ax.axvline(lo, color='gray', ls='--', lw=1.0,
                   label=f'{outlier_pct}th / {100 - outlier_pct}th pct')
        ax.axvline(hi, color='gray', ls='--', lw=1.0)

        clip_frac = float(np.mean((vals < lo) | (vals > hi)))
        ax.set_title(f'{_display(r)}\nn={len(vals)}  clipped={100 * clip_frac:.1f}%')
        ax.set_xlabel(_display(r))
        ax.set_ylabel('count')
        ax.set_xlim(plot_lo, plot_hi)
        ax.legend(fontsize=8, loc='upper right')

    for ax in axes[n:]:
        ax.axis('off')

    plt.suptitle(f'Decile sanity check: pooled values + bin edges '
                 f'(downsample={downsample_factor}, recdays={len(recdays_used)})',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    return pooled_finite


def check_task_state_place_collinearity(mouse_recdays, data_dic,
                                        downsample_factor=10,
                                        max_recdays=None):
    """Check that task_state (A/B/C/D) and place (1-21) decorrelate across tasks.

    Pools task state and place location the same way `run_glm_analysis` does
    (via `prepare_session_data`, `truncate_all_arrays`, `downsample_session_data`,
    `get_sessions_for_glm`), then computes the absolute correlation between
    every task_state one-hot column (4) and every place one-hot column (21).

    Returns the max |correlation| and prints a summary. If decorrelation is good,
    this should be substantially less than 1.0 (task remaps across the 6 unique
    tasks/recday pooled in each fit, so state identity is independent of place
    once you control for task).

    Parameters
    ----------
    mouse_recdays, data_dic : as in `run_glm_analysis`.
    downsample_factor : int
        Match what you pass to `run_glm_analysis`.
    max_recdays : int or None
        Cap on recdays to pool (None = all). Use 3–5 for a quick pass.
    """
    recdays_used = mouse_recdays[:max_recdays] if max_recdays else list(mouse_recdays)
    all_state = []
    all_locs  = []

    for mr in recdays_used:
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        for s in sessions:
            prep = prepare_session_data(data_dic[mr][s])
            prep = truncate_all_arrays(prep)
            prep = downsample_session_data(prep, downsample_factor)
            nf = prep['Locs'] <= 21
            all_state.append(prep['State'][nf])
            all_locs.append(prep['Locs'][nf])

    state_pooled = np.concatenate(all_state).astype(int)
    locs_pooled  = np.concatenate(all_locs).astype(int)

    # One-hot encode
    state_onehot = (state_pooled[:, None] == np.arange(_N_TASK_STATES)).astype(float)
    place_onehot = (locs_pooled[:, None] == np.arange(1, 22)).astype(float)

    # Compute correlations (column-wise between the two matrices)
    from scipy.stats import pearsonr
    max_corr = 0.0
    for i in range(state_onehot.shape[1]):
        for j in range(place_onehot.shape[1]):
            r, _ = pearsonr(state_onehot[:, i], place_onehot[:, j])
            max_corr = max(max_corr, abs(r))

    print(f"Task-state vs place collinearity check (recdays={len(recdays_used)}, "
          f"downsample={downsample_factor}):")
    print(f"  Max |correlation| across state×place one-hot columns: {max_corr:.4f}")
    print(f"  → Decorrelation: {'GOOD' if max_corr < 0.3 else 'MODERATE' if max_corr < 0.5 else 'POOR'}")
    return max_corr


def verify_state_labeling(mouse_recdays, data_dic, max_recdays=None,
                          verbose=True):
    """Assert the `State` array matches the `Trial_times` ground truth.

    Ground truth (see `CCGP_STATE_PAIRS.md` §2.1): `Trial_times[r, s]` is the
    time the animal is AT goal s, so state s occupies
    `[Trial_times[r, s], Trial_times[r, s+1])`. For every leg this checks that
    the modal value of `State` over that window equals s.

    This exists because `compute_task_state_arrays` silently mislabeled 72.9%
    of legs (a per-trial cyclic rotation caused by a running counter that
    advanced across skipped duplicate boundaries). Nothing downstream noticed:
    CPD merely fell toward null and the state-vs-place collinearity check read
    "GOOD" precisely *because* the labels were scrambled. Run this before
    trusting any `task_state` result.

    Returns (n_correct, n_total). `n_correct == n_total` is the pass condition.
    """
    recdays_used = mouse_recdays[:max_recdays] if max_recdays else list(mouse_recdays)
    n_ok = 0
    n_tot = 0
    bad_examples = []

    for mr in recdays_used:
        for sess in sorted(data_dic[mr].keys()):
            sd = data_dic[mr][sess]
            tt_raw = sd.get('Trial_times')
            if tt_raw is None:
                continue
            tt = np.asarray(tt_raw).astype(int)
            if tt.ndim != 2 or tt.shape[1] < 2:
                continue
            num_states = tt.shape[1] - 1
            state, _, _, _, _ = compute_task_state_arrays(tt)
            for r in range(tt.shape[0]):
                for s in range(num_states):
                    a, b = int(tt[r, s]), int(tt[r, s + 1])
                    if b <= a or b > len(state):
                        continue
                    n_tot += 1
                    modal = np.bincount(state[a:b], minlength=num_states).argmax()
                    if modal == s:
                        n_ok += 1
                    elif len(bad_examples) < 5:
                        bad_examples.append((mr, sess, r, s, int(modal)))

    if verbose:
        pct = 100.0 * n_ok / n_tot if n_tot else float('nan')
        print(f"State-labeling check (recdays={len(recdays_used)}): "
              f"{n_ok}/{n_tot} legs correct ({pct:.1f}%)")
        if n_ok == n_tot:
            print("  → PASS: State matches Trial_times ground truth")
        else:
            print("  → FAIL: task_state results are NOT trustworthy")
            for mr, sess, r, s, got in bad_examples:
                print(f"     {mr} sess{sess} trial{r}: true state {s}, got {got}")
    return n_ok, n_tot


def check_since_A_task_state_collinearity(mouse_recdays, data_dic,
                                          downsample_factor=10,
                                          max_recdays=None):
    """Check collinearity between time_since_A and task_state (A/B/C/D).

    Pools time_since_A and task_state the same way `run_glm_analysis` does
    (via `prepare_session_data`, `truncate_all_arrays`, `downsample_session_data`,
    `get_sessions_for_glm`), bins time_since_A into deciles, then computes the
    absolute correlation between every task_state one-hot column (4) and every
    time_since_A decile column (10).

    Expected to be SUBSTANTIAL: within a loop, elapsed-time-since-A rises
    monotonically with state index (0→3), so collinearity is intentional and
    expected. This diagnostic measures it quantitatively so the CPD can be
    interpreted in light of the known confound. Unlike task_state-vs-place
    (which should decorrelate across tasks), this correlation is *not* a design
    flaw — it's the core assumption that makes time_since_A distinct from
    task_state within the same time window.

    Returns the max |correlation| and prints a summary.

    Parameters
    ----------
    mouse_recdays, data_dic : as in `run_glm_analysis`.
    downsample_factor : int
        Match what you pass to `run_glm_analysis`.
    max_recdays : int or None
        Cap on recdays to pool (None = all). Use 3–5 for a quick pass.
    """
    recdays_used = mouse_recdays[:max_recdays] if max_recdays else list(mouse_recdays)
    all_state = []
    all_tsa   = []

    for mr in recdays_used:
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        for s in sessions:
            prep = prepare_session_data(data_dic[mr][s])
            prep = truncate_all_arrays(prep)
            prep = downsample_session_data(prep, downsample_factor)
            nf = prep['Locs'] <= 21
            all_state.append(prep['State'][nf])
            all_tsa.append(prep['time_since_A'][nf])

    state_pooled = np.concatenate(all_state).astype(int)
    tsa_pooled   = np.concatenate(all_tsa).astype(int)

    # One-hot encode task_state
    state_onehot = (state_pooled[:, None] == np.arange(_N_TASK_STATES)).astype(float)

    # Decile-bin time_since_A
    tsa_edges = compute_decile_edges(tsa_pooled)
    if tsa_edges is None:
        print(f"Time-since-A vs task-state collinearity: insufficient data")
        return np.nan
    tsa_onehot = apply_onehot(tsa_pooled, tsa_edges)

    # Compute correlations (column-wise between the two matrices)
    from scipy.stats import pearsonr
    max_corr = 0.0
    for i in range(state_onehot.shape[1]):
        for j in range(tsa_onehot.shape[1]):
            r, _ = pearsonr(state_onehot[:, i], tsa_onehot[:, j])
            max_corr = max(max_corr, abs(r))

    print(f"Time-since-A vs task-state collinearity check (recdays={len(recdays_used)}, "
          f"downsample={downsample_factor}):")
    print(f"  Max |correlation| across state×time_since_A columns: {max_corr:.4f}")
    print(f"  (Expected to be substantial: time-since-A is monotonic with state within a loop)")
    return max_corr


def _pool_poke_covariates(mouse_recdays, data_dic, downsample_factor=10,
                          max_recdays=None):
    """Pool the arrays the poke diagnostics need, exactly as run_glm_analysis does."""
    recdays_used = mouse_recdays[:max_recdays] if max_recdays else list(mouse_recdays)
    keys = ['poke_rewarded', 'poke_unrewarded', 'time_from_reward', 'Locs']
    pooled = {k: [] for k in keys}
    for mr in recdays_used:
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        for s in sessions:
            prep = prepare_session_data(data_dic[mr][s])
            prep = truncate_all_arrays(prep)
            prep = downsample_session_data(prep, downsample_factor)
            nf = prep['Locs'] <= 21
            for k in keys:
                pooled[k].append(prep[k][nf])
    return {k: np.concatenate(v) for k, v in pooled.items()}, len(recdays_used)


def check_poke_reward_collinearity(mouse_recdays, data_dic, downsample_factor=10,
                                   max_recdays=None):
    """Quantify how far `poke_rewarded` duplicates `time_from_reward`.

    THE critical diagnostic for this analysis. A rewarded poke's entry bin IS the
    reward bin (96.8% identical across the LEC dataset), so the in-port indicator
    necessarily overlaps the early time-since-reward bins. This does not make the
    regressor useless — it makes its CPD an arbitrary split of shared variance,
    which is the same in-sample failure documented in
    TIME_VS_PROGRESS_DISSOCIATION.md §2. Read the number, then let
    `run_poke_duration_split` do the actual adjudication.

    Prints, per time_from_reward decile, the fraction of its samples that fall
    inside a rewarded poke. A decile at ~1.0 is fully inside the consumption
    window and its beta cannot be attributed to timing rather than consumption.

    Returns (max_abs_corr, occupancy_per_decile).
    """
    from scipy.stats import pearsonr

    pooled, n_rec = _pool_poke_covariates(mouse_recdays, data_dic,
                                          downsample_factor, max_recdays)
    pkr, tf = pooled['poke_rewarded'], pooled['time_from_reward']

    edges = compute_decile_edges(tf)
    if edges is None:
        print("check_poke_reward_collinearity: insufficient data")
        return np.nan, np.array([])
    TF = apply_onehot(tf, edges)

    corrs = np.array([abs(pearsonr(pkr, TF[:, j])[0]) for j in range(TF.shape[1])])
    occ = np.array([pkr[TF[:, j] > 0].mean() if np.any(TF[:, j] > 0) else np.nan
                    for j in range(TF.shape[1])])

    print(f"Rewarded-poke vs time-from-reward collinearity "
          f"(recdays={n_rec}, downsample={downsample_factor}):")
    print(f"  overall in-port fraction: rewarded={pkr.mean():.3f} "
          f"unrewarded={pooled['poke_unrewarded'].mean():.3f}")
    print(f"  max |r| against any time_from_reward decile: {corrs.max():.4f} "
          f"(decile {int(corrs.argmax())})")
    print("  decile | median t (bins) | frac inside a rewarded poke | |r|")
    for j in range(TF.shape[1]):
        sel = TF[:, j] > 0
        med = np.median(tf[sel]) if np.any(sel) else np.nan
        bar = '#' * int(round(20 * (occ[j] if np.isfinite(occ[j]) else 0)))
        print(f"    {j:>4d} | {med:>15.0f} | {occ[j]:>10.3f} {bar:<20s} | {corrs[j]:.3f}")
    print("  → deciles near 1.0 are fully inside the consumption window; their")
    print("    time-coding beta is NOT separable from consumption in-sample.")
    return float(corrs.max()), occ


def check_poke_place_collinearity(mouse_recdays, data_dic, downsample_factor=10,
                                  max_recdays=None):
    """Poke indicators vs the 21 place one-hots.

    A poke at port p implies Locs == p (ports are nodes 1–9; 10–21 are corridors),
    so poke is nested inside place. The animal is at a tower ~88% of the time,
    so the variance that separates them is the at-tower-but-not-poking residual —
    which this reports directly.

    Returns (max_abs_corr_rewarded, max_abs_corr_unrewarded).
    """
    from scipy.stats import pearsonr

    pooled, n_rec = _pool_poke_covariates(mouse_recdays, data_dic,
                                          downsample_factor, max_recdays)
    pkr, pku, locs = pooled['poke_rewarded'], pooled['poke_unrewarded'], pooled['Locs'].astype(int)
    place = (locs[:, None] == np.arange(1, 22)).astype(float)

    mr_ = max(abs(pearsonr(pkr, place[:, j])[0]) for j in range(21))
    mu_ = max(abs(pearsonr(pku, place[:, j])[0]) for j in range(21))

    at_tower = locs <= 9
    poking = (pkr > 0) | (pku > 0)
    print(f"Poke vs place collinearity (recdays={n_rec}, downsample={downsample_factor}):")
    print(f"  time at tower nodes (1–9): {at_tower.mean():.3f}")
    print(f"  of tower time, fraction poking: {poking[at_tower].mean():.3f} "
          f"→ at-tower-not-poking residual = {1 - poking[at_tower].mean():.3f}")
    print(f"  max |r| vs any place column: rewarded={mr_:.4f} unrewarded={mu_:.4f}")
    print(f"  poke samples landing off a tower node: "
          f"rewarded={np.mean(locs[pkr > 0] > 9):.4f} "
          f"unrewarded={np.mean(locs[pku > 0] > 9):.4f}")
    print("  → Expect ~0 for rewarded and a few % for unrewarded. Measured "
          "per-bout\n    agreement is 0.985 (rewarded) vs 0.679 (unrewarded): the brief "
          "~175 ms\n    unrewarded pokes happen while the tracked centroid is still moving, "
          "so\n    the tracker bins them to an adjacent edge. This is NOT a clock offset —"
          "\n    Trial_times → Task[state] matches Locs at 0.998 with a best lag of 0.")
    return float(mr_), float(mu_)


# ============================================================================
# Poke-duration dissociation: time cell vs consumption cell
# ============================================================================
#
# The GLM cannot settle this (see check_poke_reward_collinearity). This can.
#
# A rewarded poke starts at the reward and ends when the animal withdraws, and
# withdrawal time varies 3.5x across pokes (p10 1.10 s, median 2.58 s, p90
# 3.83 s). That spread is the only thing in the data that separates the two
# hypotheses:
#
#   TIME cell        peak at a FIXED latency after reward, whatever the animal
#                    does -> peak latency is flat across duration terciles
#                    (slope 0), and survives on pokes already terminated.
#   CONSUMPTION cell activity tracks time IN PORT and stops at withdrawal
#                    -> peak latency grows with poke duration (slope ~1) and
#                    collapses on short pokes.
#
# Peak latency is read from a smoothed trial-averaged PSTH, and terciles are
# count-matched by subsampling to the smallest tercile so a tercile difference
# cannot be a power difference (the convention used by
# time_vs_progress_dissociation.plot_peak_latency_vs_duration).

def build_poke_psth(mouse_recdays, data_dic, window_bins=(-20, 240),
                    max_recdays=None, bin_size_ms=25,
                    max_duration_bins=_POKE_MAX_DURATION_BINS):
    """Peri-reward spike matrices for every rewarded poke, per recday.

    Uses RAW 25 ms bins (no downsampling) — the whole analysis is about
    within-bout timing, so the decimation used for the GLM would throw away
    exactly the resolution that matters.

    Returns {mouse_recday: dict(psth=(n_pokes, n_neurons, n_lags) float32,
                                duration=(n_pokes,) int, lags=(n_lags,) int)}
    where lag 0 is the reward bin (== the poke entry bin).
    """
    recdays_used = mouse_recdays[:max_recdays] if max_recdays else list(mouse_recdays)
    lo, hi = window_bins
    lags = np.arange(lo, hi)
    out = {}

    for mr in tqdm(recdays_used, desc='poke PSTH'):
        sessions, _ = get_sessions_for_glm(data_dic[mr])
        chunks, durs = [], []
        for s in sessions:
            sd = data_dic[mr][s]
            pokes = sd.get('Pokes')
            if pokes is None:
                continue
            pokes = np.asarray(pokes)
            if pokes.size == 0:
                continue
            FR = sd['Neuron_raw']
            n_bins = FR.shape[1]
            rew = pokes[pokes[:, 3] == 1]
            dur = rew[:, 1] - rew[:, 0] + 1
            rew = rew[(dur > 0) & (dur <= max_duration_bins)]
            dur = dur[(dur > 0) & (dur <= max_duration_bins)]
            for (entry, _exit, _p, _r, _st), d in zip(rew, dur):
                a, b = int(entry) + lo, int(entry) + hi
                if a < 0 or b > n_bins:
                    continue          # keep only fully-covered windows
                chunks.append(FR[:, a:b].astype(np.float32))
                durs.append(int(d))
        if chunks:
            out[mr] = {'psth': np.stack(chunks), 'duration': np.asarray(durs),
                       'lags': lags, 'bin_size_ms': bin_size_ms}
    return out


def _smooth(x, sigma_bins):
    return gaussian_filter1d(x, sigma=sigma_bins, axis=-1, mode='nearest')


def run_poke_duration_split(psth_dic, n_terciles=3, smooth_sigma_bins=6,
                            min_pokes_per_tercile=8, seed=0,
                            peak_search_from_bin=2, min_modulation_z=3.0):
    """Per-neuron response OFFSET latency within each poke-duration tercile.

    Terciles are count-matched (subsampled to the smallest) so a tercile
    difference cannot be a power difference. The headline statistic is the OLS
    slope of **offset latency** on tercile median duration, both in seconds:

        slope ~ 0  -> fixed-latency TIME cell
        slope ~ 1  -> withdrawal-locked CONSUMPTION cell

    OFFSET, not peak, is the primary statistic. The synthetic control caught
    this: a consumption cell is a BOXCAR over the bout, and a plateau has no
    well-defined argmax — the planted consumption cell returned peak slope
    +0.20 (arbitrary, peak at 0.93 s) while the planted withdrawal BUMP
    returned +0.96. Offset latency (last crossing of half-max above baseline)
    is well defined for a plateau and a bump alike, so it scores both at ~1 and
    the fixed-latency time cell at ~0. `peak_slope` is still reported, as
    secondary colour only.

    Neurons whose peri-reward modulation does not reach `min_modulation_z`
    baseline SDs are DROPPED rather than assigned a slope: an unmodulated cell
    has no offset to measure, and the planted noise cell otherwise returns a
    confident-looking -0.77. (Same lesson as the `r2_floor` gate in
    time_vs_progress_dissociation.py.)

    Returns a list of per-neuron dicts.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for mr, d in psth_dic.items():
        psth, dur, lags = d['psth'], d['duration'], d['lags']
        dt = d['bin_size_ms'] / 1000.0
        if len(dur) < n_terciles * min_pokes_per_tercile:
            continue
        edges = np.quantile(dur, np.linspace(0, 1, n_terciles + 1))
        groups = []
        for k in range(n_terciles):
            lo_, hi_ = edges[k], edges[k + 1]
            sel = np.where((dur >= lo_) & (dur <= hi_))[0] if k == n_terciles - 1 \
                else np.where((dur >= lo_) & (dur < hi_))[0]
            groups.append(sel)
        if min(len(gp) for gp in groups) < min_pokes_per_tercile:
            continue
        nmin = min(len(gp) for gp in groups)
        groups = [rng.choice(gp, nmin, replace=False) for gp in groups]

        n_neurons = psth.shape[1]
        means = np.stack([_smooth(psth[gp].mean(0), smooth_sigma_bins) for gp in groups])
        tercile_dur = np.array([np.median(dur[gp]) for gp in groups]) * dt

        pre = lags < 0
        post = lags >= peak_search_from_bin
        lags_post_s = lags[post] * dt

        for n in range(n_neurons):
            curves = means[:, n, :]
            if not np.all(np.isfinite(curves)):
                continue
            base = curves[:, pre].mean(axis=1, keepdims=True) if pre.any() else 0.0
            noise = curves[:, pre].std() if pre.any() else 0.0
            bs = curves - base
            post_bs = bs[:, post]
            amp = post_bs.max(axis=1)
            if noise <= 0 or amp.min() <= min_modulation_z * noise:
                continue                       # unmodulated: no offset to measure

            offsets, peaks = [], []
            for k in range(n_terciles):
                c = post_bs[k]
                i_pk = int(np.argmax(c))
                peaks.append(lags_post_s[i_pk])
                half = 0.5 * c[i_pk]
                above = np.where(c[i_pk:] >= half)[0]
                offsets.append(lags_post_s[i_pk + above[-1]] if above.size
                               else lags_post_s[i_pk])
            offsets = np.asarray(offsets)
            peaks = np.asarray(peaks)

            rows.append({
                'mouse_recday': mr, 'mouse': str(mr).split('_')[0], 'neuron': n,
                'slope': float(np.polyfit(tercile_dur, offsets, 1)[0]),
                'peak_slope': float(np.polyfit(tercile_dur, peaks, 1)[0]),
                'offsets_s': offsets, 'peaks_s': peaks,
                'tercile_durations_s': tercile_dur,
                'offset_mean_s': float(offsets.mean()),
                'peak_mean_s': float(peaks.mean()),
                'modulation_z': float(amp.min() / noise),
                'n_pokes_per_tercile': int(nmin),
                'mean_rate': float(psth[:, n, :].mean()),
            })
    return rows


def poke_short_bout_contrast(psth_dic, short_max_s=2.0, probe_window_s=(2.0, 3.0),
                             baseline_window_s=(-0.5, 0.0), smooth_sigma_bins=6,
                             min_pokes=8):
    """The sharpest single contrast: on pokes the animal has ALREADY left, is
    there still a response in the probe window?

    Restricts to rewarded pokes shorter than `short_max_s`, so by 2–3 s the
    animal is out of the port. A time cell keeps its peak there; a consumption
    cell cannot. Returns per-neuron dicts with the baseline-subtracted probe
    response on short vs long pokes.
    """
    rows = []
    for mr, d in psth_dic.items():
        psth, dur, lags = d['psth'], d['duration'], d['lags']
        dt = d['bin_size_ms'] / 1000.0
        dur_s = dur * dt
        short = np.where(dur_s <= short_max_s)[0]
        long_ = np.where(dur_s > short_max_s)[0]
        if len(short) < min_pokes or len(long_) < min_pokes:
            continue
        t = lags * dt
        probe = (t >= probe_window_s[0]) & (t < probe_window_s[1])
        base = (t >= baseline_window_s[0]) & (t < baseline_window_s[1])
        if not probe.any() or not base.any():
            continue
        ms = _smooth(psth[short].mean(0), smooth_sigma_bins)
        ml = _smooth(psth[long_].mean(0), smooth_sigma_bins)
        for n in range(psth.shape[1]):
            rows.append({
                'mouse_recday': mr, 'mouse': str(mr).split('_')[0], 'neuron': n,
                'probe_short': float(ms[n, probe].mean() - ms[n, base].mean()),
                'probe_long': float(ml[n, probe].mean() - ml[n, base].mean()),
                'n_short': int(len(short)), 'n_long': int(len(long_)),
            })
    return rows


def plot_poke_duration_split(psth_dic, split_rows, neuron_key=None,
                             smooth_sigma_bins=6, seed=0):
    """Population figure for the duration split.

    Panel 1: offset-slope histogram with the two hypothesis landmarks
             (0 = fixed-latency time cell, 1 = withdrawal-locked consumption).
    Panel 2: per-tercile response OFFSET vs tercile duration, mean +/- SEM,
             against the slope-0 and slope-1 reference lines.
    Panel 3: example neuron's three count-matched PSTHs (highest |slope| by
             default), with each tercile's median withdrawal time marked.
    """
    if not split_rows:
        print("plot_poke_duration_split: no neurons passed the tercile criteria")
        return None
    slopes = np.array([r['slope'] for r in split_rows])
    peaks = np.stack([r['offsets_s'] for r in split_rows])
    durs = np.stack([r['tercile_durations_s'] for r in split_rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.hist(slopes, bins=40, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axvline(0, color='k', ls='--', lw=1.5, label='0 = time cell')
    ax.axvline(1, color='crimson', ls='--', lw=1.5, label='1 = consumption')
    ax.axvline(np.median(slopes), color='orange', lw=2,
               label=f'median {np.median(slopes):+.2f}')
    ax.set_xlabel('d(response offset) / d(poke duration)')
    ax.set_ylabel('neurons')
    ax.set_title(f'Offset-latency slope (n={len(slopes)} modulated cells)')
    ax.legend(fontsize=7)

    ax = axes[1]
    md, mp = durs.mean(0), peaks.mean(0)
    sp = peaks.std(0) / max(1, np.sqrt(len(peaks)))
    ax.errorbar(md, mp, yerr=sp, fmt='o-', color='steelblue', capsize=4,
                label='observed')
    ax.plot(md, [mp[0]] * len(md), 'k--', lw=1, label='time cell (flat)')
    ax.plot(md, md, ls='--', color='crimson', lw=1, label='consumption (offset=exit)')
    ax.set_xlabel('poke duration (s, tercile median)')
    ax.set_ylabel('response offset (s)')
    ax.set_title('Response offset vs poke duration')
    ax.legend(fontsize=7)

    ax = axes[2]
    if neuron_key is None:
        best = int(np.argmax(np.abs(slopes)))
        neuron_key = (split_rows[best]['mouse_recday'], split_rows[best]['neuron'])
    else:
        best = next((i for i, r in enumerate(split_rows)
                     if (r['mouse_recday'], r['neuron']) == tuple(neuron_key)), None)
        if best is None:
            raise KeyError(f"{neuron_key} is not in split_rows")
    mr, n = neuron_key
    d = psth_dic[mr]
    psth, dur, lags = d['psth'], d['duration'], d['lags']
    dt = d['bin_size_ms'] / 1000.0
    edges = np.quantile(dur, np.linspace(0, 1, 4))
    rng = np.random.default_rng(seed)
    groups = []
    for k in range(3):
        sel = np.where((dur >= edges[k]) & (dur <= edges[k + 1]))[0] if k == 2 \
            else np.where((dur >= edges[k]) & (dur < edges[k + 1]))[0]
        groups.append(sel)
    nmin = min(len(gp) for gp in groups)
    cmap = plt.get_cmap('viridis')
    for k, gp in enumerate(groups):
        gp = rng.choice(gp, nmin, replace=False)
        curve = _smooth(psth[gp][:, n, :].mean(0), smooth_sigma_bins) / dt
        col = cmap(k / 2)
        med_exit = np.median(dur[gp]) * dt
        ax.plot(lags * dt, curve, color=col, lw=1.6,
                label=f'{np.median(dur[gp]) * dt:.1f} s bouts')
        ax.axvline(med_exit, color=col, ls=':', lw=1.2)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('time from reward (s)')
    ax.set_ylabel('firing rate (spk/s)')
    ax.set_title(f'{mr} n{n}  offset-slope={slopes[best]:+.2f}\n'
                 f'(dotted = median withdrawal)')
    ax.legend(fontsize=7)

    plt.suptitle('Poke-duration dissociation: fixed-latency timing vs consumption',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()
    return fig


# ----------------------------------------------------------------------------
# Synthetic positive control (MANDATORY gate)
# ----------------------------------------------------------------------------

_SYNTH_POKE_CELLS = ['consumption', 'time', 'withdrawal', 'noise']


def make_synthetic_poke_data(data_dic, mouse_recdays, peak_s=2.5, width_s=0.4,
                             rate_hi=0.6, rate_lo=0.02, seed=0):
    """Plant four cells with known ground truth into REAL covariate structure.

    Returns a shallow copy of `data_dic` whose `Neuron_raw` is replaced by four
    synthetic neurons per session, driven by that session's real poke table and
    real Trial_times:

      0 consumption : fires while inside a rewarded poke, stops at withdrawal
      1 time        : Gaussian bump at `peak_s` after reward, whatever the animal does
      2 withdrawal  : Gaussian bump locked to poke EXIT
      3 noise       : constant rate

    THE GATE: `run_poke_duration_split` must return slope ~1 for the consumption
    cell and slope ~0 for the time cell, and the GLM must send CPD to
    `poke_rewarded` vs `time_from_reward` accordingly. Because a rewarded poke's
    onset IS the reward onset, there is no other way to know the pipeline can
    tell these two hypotheses apart — the real data cannot reveal it. See
    `SYNTHETIC_CONTROLS` practice notes in CCGP_STATE_PAIRS.md §7.
    """
    rng = np.random.default_rng(seed)
    bin_s = 0.025
    peak_b = peak_s / bin_s
    width_b = width_s / bin_s
    synth = {}

    for mr in mouse_recdays:
        if mr not in data_dic:
            continue
        synth[mr] = {}
        for s, sd in data_dic[mr].items():
            sd = dict(sd)
            FR = sd['Neuron_raw']
            n_bins = FR.shape[1]
            pokes = sd.get('Pokes')
            tt = sd.get('Trial_times')

            pk_r, _pk_u = compute_poke_arrays(pokes, n_bins)

            # time-cell drive: bump at peak_s after EVERY reward boundary
            time_drive = np.zeros(n_bins)
            wd_drive = np.zeros(n_bins)
            if tt is not None and np.asarray(tt).size:
                for t0 in np.unique(np.asarray(tt).astype(int).ravel()):
                    c = int(t0 + peak_b)
                    a, b = max(0, c - int(4 * width_b)), min(n_bins, c + int(4 * width_b))
                    if b > a:
                        time_drive[a:b] += np.exp(
                            -0.5 * ((np.arange(a, b) - c) / width_b) ** 2)
            if pokes is not None and np.asarray(pokes).size:
                p = np.asarray(pokes)
                for entry, exit_, _pt, r, _st in p[p[:, 3] == 1]:
                    if exit_ - entry + 1 > _POKE_MAX_DURATION_BINS:
                        continue
                    c = int(exit_)
                    a, b = max(0, c - int(4 * width_b)), min(n_bins, c + int(4 * width_b))
                    if b > a:
                        wd_drive[a:b] += np.exp(
                            -0.5 * ((np.arange(a, b) - c) / width_b) ** 2)

            drives = np.stack([
                pk_r,
                np.clip(time_drive, 0, 1),
                np.clip(wd_drive, 0, 1),
                np.zeros(n_bins),
            ])
            lam = rate_lo + (rate_hi - rate_lo) * drives
            sd['Neuron_raw'] = rng.poisson(lam).astype(float)
            sd['num_neurons'] = len(_SYNTH_POKE_CELLS)
            synth[mr][s] = sd
    return synth


def run_synthetic_poke_controls(data_dic, mouse_recdays, max_recdays=3, **kw):
    """Run the synthetic cells through the REAL duration-split pipeline.

    Prints the recovered slope per planted cell type and returns
    {cell_type: median_slope}. Gate: consumption > 0.5, time < 0.35, and
    consumption must exceed time.
    """
    mrs = list(mouse_recdays)[:max_recdays]
    synth = make_synthetic_poke_data(data_dic, mrs, **kw)
    psth = build_poke_psth(mrs, synth)
    rows = run_poke_duration_split(psth)

    out = {}
    print("Synthetic poke controls — recovered OFFSET-latency slope")
    print("  (expected: consumption ~1, time ~0, withdrawal ~1, noise DROPPED)")
    for i, name in enumerate(_SYNTH_POKE_CELLS):
        sl = np.array([r['slope'] for r in rows if r['neuron'] == i])
        pk = np.array([r['peak_slope'] for r in rows if r['neuron'] == i])
        off = np.array([r['offset_mean_s'] for r in rows if r['neuron'] == i])
        out[name] = float(np.median(sl)) if sl.size else np.nan
        if sl.size:
            print(f"    {name:12s} offset-slope={out[name]:+.3f}  "
                  f"peak-slope={np.median(pk):+.3f}  "
                  f"mean offset={off.mean():.2f} s  (n={sl.size} recdays)")
        else:
            print(f"    {name:12s} dropped by the modulation gate "
                  f"(correct for 'noise')")

    ok = (out.get('consumption', np.nan) > 0.5
          and out.get('time', np.nan) < 0.35
          and out.get('consumption', -9) > out.get('time', 9)
          and np.isnan(out.get('noise', np.nan)))
    print(f"  → GATE {'PASS' if ok else 'FAIL'}: the split "
          f"{'separates' if ok else 'does NOT separate'} consumption from timing"
          f"{'' if np.isnan(out.get('noise', np.nan)) else '; NOISE CELL NOT DROPPED'}")
    return out, ok


# ============================================================================
# Save / load + light publication style
# ============================================================================
#
# Lightweight section-level wrappers for caching fits and exporting figures.
# Convention: every artifact lives under
#     {save_dir}/{section_name}__{key}.pkl    (result dicts)
#     {save_dir}/{section_name}__{name}.pdf   (figures)
# Persistent dir, section-prefixed filenames; re-running a fit overwrites the
# pickle, and `run_or_load_glm` short-circuits the fit when all expected
# pickles already exist.
#
# Figure styling follows the .claude/skills/gridmaze-plotter SKILL: Arial 8pt,
# top/right spines hidden, Type-42 fonts for editable PDFs. Colors of the
# existing plotting functions are left alone (per user pick).


def apply_gridmaze_style():
    """Idempotent rcParams setter applying the gridmaze-plotter publication
    style: Arial 8pt + hidden top/right spines + Type-42 PDF fonts. Call once
    at notebook top (before any plotting) so every subsequent figure picks it up.

    Does NOT change color settings — the existing plot functions' steelblue /
    crimson / darkorange / etc. references stay as-is.
    """
    import matplotlib as mpl

    rc = {
        # Fonts — Arial primary with a portable fallback for headless boxes.
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'DejaVu Sans'],
        'font.size':          8,
        'figure.titlesize':   8,
        'axes.titlesize':     8,
        'axes.labelsize':     8,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':    8,

        # Axes furniture
        'axes.linewidth':     0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,

        # Editable PDFs (embed Type-42; required for Illustrator round-trips).
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    }
    mpl.rcParams.update(rc)


# ----- low-level pickle / fs helpers -------------------------------------

def _ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)


def _save_pickle(obj, path):
    import os, pickle
    _ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def _artifact_path(save_dir, section_name, key, ext):
    import os
    return os.path.join(save_dir, f'{section_name}__{key}.{ext}')


# ----- section-level save -----------------------------------------------

# Matches a mouse_recday key like 'ah08_20250613_20250615'. Used to tell a cached
# artifact that can be filtered by recday from one that pools across them.
_RECDAY_RE = __import__('re').compile(r'^[a-z]{2}\d{2}_\d{8}_\d{8}$')


# Result keys that `save_section` / `load_glm_results` understand for fitted
# artifacts. Extra entries in `results` are saved verbatim under their key.
_GLM_RESULT_KEYS = ('glm_results', 'permutation_results', 'cpd_results',
                    'cv_results',
                    'neuron_scales', 'tuned_dict', 'mouse_tuning_concat')


def save_section(section_name, save_dir, *, results=None,
                 fig_names=None, close_after=False):
    """Save one section's outputs in one call.

    Parameters
    ----------
    section_name : str
        Filename prefix and cache key (e.g. 'baseline', 'extended_cpd').
    save_dir : str
        Output directory. Created if missing.
    results : dict {str: obj}, optional
        Each value is pickled to `{save_dir}/{section_name}__{key}.pkl`.
        Use the canonical keys when applicable so `run_or_load_glm` /
        `load_glm_results` can pick them up:
        {_GLM_RESULT_KEYS}.
    fig_names : list of str, optional
        Names for currently-open figures in `plt.get_fignums()` order. Saved
        as `{save_dir}/{section_name}__{name}.pdf`. Missing/extra names
        fall back to `fig1`, `fig2`, ... `None` → use auto-names for all.
    close_after : bool, default False
        Close saved figures after saving (memory hygiene for long notebooks;
        breaks inline display so leave False in Jupyter).

    Returns
    -------
    written : list of str
        Absolute paths of every file written.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    _ensure_dir(save_dir)
    written = []

    # ---- pickle each result dict --------------------------------------
    if results:
        for key, obj in results.items():
            path = _artifact_path(save_dir, section_name, key, 'pkl')
            _save_pickle(obj, path)
            written.append(path)

    # ---- save every currently-open figure as PDF ----------------------
    fignums = plt.get_fignums()
    if fig_names is None:
        fig_names = []
    save_rc = {
        'savefig.bbox':       None,
        'savefig.pad_inches': 0.0,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    }
    with mpl.rc_context(save_rc):
        for i, num in enumerate(fignums):
            name = fig_names[i] if i < len(fig_names) and fig_names[i] else f'fig{i+1}'
            path = _artifact_path(save_dir, section_name, name, 'pdf')
            fig = plt.figure(num)
            fig.savefig(path, bbox_inches=None)
            written.append(path)
            if close_after:
                plt.close(fig)

    print(f'save_section({section_name!r}): wrote {len(written)} file(s) to {save_dir}')
    return written


# ----- run-or-load (skip-if-exists fit wrapper) -------------------------

def run_or_load_glm(mouse_recdays, data_dic, save_dir, section_name,
                    *, force_refit=False, **kwargs):
    """Skip-if-exists wrapper around `run_glm_analysis`.

    If `{save_dir}/{section_name}__glm_results.pkl` (plus permutation_results
    and, when `compute_cpd=True`, cpd_results) already exist and
    `force_refit=False`, loads them and returns the same tuple shape
    `run_glm_analysis(**kwargs)` would. Otherwise fits, pickles, returns.

    Use the SAME `section_name` for both `run_or_load_glm` and `save_section`
    so the cache key is consistent.
    """
    import os
    compute_cpd = bool(kwargs.get('compute_cpd', False))
    return_scales = bool(kwargs.get('return_scales', False))
    cross_validate = bool(kwargs.get('cross_validate', False))

    # Order must mirror run_glm_analysis's return tuple exactly.
    needed = ['glm_results', 'permutation_results']
    if compute_cpd:
        needed.append('cpd_results')
    if return_scales:
        needed.append('neuron_scales')
    if cross_validate:
        needed.append('cv_results')

    paths = {k: _artifact_path(save_dir, section_name, k, 'pkl') for k in needed}

    if not force_refit and all(os.path.exists(p) for p in paths.values()):
        print(f'run_or_load_glm({section_name!r}): loading cached results from {save_dir}')
        stale = _stale_or_excluded(section_name)
        loaded = tuple(_drop_excluded_recdays(_load_pickle(paths[k]), stale)
                       for k in needed)
        missing = [r for r in mouse_recdays
                   if r in stale or r not in loaded[0]] if isinstance(loaded[0], dict) else []
        if missing:
            # The cache predates the current data for these recdays. Say so loudly: a
            # silent gap between the recdays asked for and the recdays returned is how a
            # stale fit gets into a figure.
            print(f'  WARNING: cache has no usable fit for {len(missing)} requested '
                  f'recday(s): {missing}. Re-run with force_refit=True to include them.')
        return loaded

    print(f'run_or_load_glm({section_name!r}): no cache (or force_refit) — fitting')
    result = run_glm_analysis(mouse_recdays, data_dic, **kwargs)

    _ensure_dir(save_dir)
    for k in needed:
        idx = needed.index(k)
        _save_pickle(result[idx], paths[k])
    print(f'run_or_load_glm({section_name!r}): saved {len(needed)} pickle(s) to {save_dir}')
    return result


# ----- manual loader ----------------------------------------------------

def _stale_or_excluded(section_name=None):
    """Recdays that must not come back out of a cached fit.

    `section_name` scopes the STALE list: a section produced by the W1 production refit
    postdates the data fix, so its recdays are good and must not be dropped. Without this
    the check matches on recday name alone and silently removes a freshly refitted recday
    from the new results -- which it did, costing `ly05_20250618_20250619` from a 25-recday
    fit. `EXCLUDE_RECDAYS` is unconditional and always applies.

    The union of two different problems: `EXCLUDE_RECDAYS` (the data itself is bad) and
    `STALE_CACHE_RECDAYS` (the data has been fixed, but these pickles were fitted before
    the fix). Both make a cached result wrong; only the first makes the recday unusable.
    """
    registry = _recday_registry()
    if registry is None:
        return set()
    excluded = set(registry.EXCLUDE_RECDAYS)
    # STALE entries flag PRE-REFIT pickles. A section stamped with the production
    # configuration postdates the data fix, so its recdays are good.
    if not (hasattr(registry, 'is_post_refit_section')
            and registry.is_post_refit_section(section_name)):
        excluded |= set(registry.STALE_CACHE_RECDAYS)
    return excluded


def _is_recday_keyed(obj):
    """True if `obj` is a dict keyed by mouse_recday, i.e. filterable by recday."""
    return isinstance(obj, dict) and any(
        _RECDAY_RE.match(str(k)) for k in obj)


def _drop_excluded_recdays(obj, excluded, verbose=True):
    """Strip `excluded` recday keys from a cached per-recday result dict.

    GLM artifacts are keyed `{mouse_recday: {session: ...}}`, so a recday found to be
    corrupt can be taken out of every aggregate figure by filtering at load time — no
    refit needed for the other recdays. Anything not shaped that way is returned as-is;
    `load_glm_results` warns separately about those, because an artifact that pools
    neurons across recdays (`*__mouse_tuning_concat`, keyed by MOUSE) still contains the
    excluded recday's neurons and cannot be repaired by filtering.
    """
    if not isinstance(obj, dict) or not excluded:
        return obj
    hits = [k for k in obj if str(k) in excluded]
    if not hits:
        return obj
    if verbose:
        print(f"  dropping {len(hits)} recday(s) from cached results as excluded "
              f"or stale: {[str(k) for k in hits]}")
    return {k: v for k, v in obj.items() if str(k) not in excluded}


def load_glm_results(save_dir, section_name, *, apply_exclusions=True, verbose=True):
    """Return whichever section pickles exist as a dict.

    Looks for the canonical `_GLM_RESULT_KEYS` plus any other `*.pkl` files
    whose name starts with `{section_name}__`. Returns
    `{key: unpickled_object}`; missing keys are simply omitted.

    Excluded recdays (`recday_registry.EXCLUDE_RECDAYS`) are filtered out of every
    per-recday dict, so cached fits made before a recday was found to be corrupt still
    give correct aggregates. Pass `apply_exclusions=False` to see the raw cache.
    """
    import os, glob

    excluded = _stale_or_excluded(section_name) if apply_exclusions else set()

    out = {}
    prefix = f'{section_name}__'

    # canonical keys first (stable order)
    for k in _GLM_RESULT_KEYS:
        p = _artifact_path(save_dir, section_name, k, 'pkl')
        if os.path.exists(p):
            out[k] = _drop_excluded_recdays(_load_pickle(p), excluded, verbose)

    # plus any other section pickles
    for p in sorted(glob.glob(os.path.join(save_dir, f'{prefix}*.pkl'))):
        key = os.path.basename(p)[len(prefix):-len('.pkl')]
        if key in out:
            continue
        out[key] = _drop_excluded_recdays(_load_pickle(p), excluded, verbose)

    if excluded:
        # Artifacts that pool neurons across recdays (mouse-keyed tuning concatenations,
        # geometry tables) still contain the excluded recday's neurons. Filtering cannot
        # reach them, so say which ones need regenerating rather than returning them
        # silently.
        unfilterable = [k for k, v in out.items() if not _is_recday_keyed(v)]
        if unfilterable:
            print(f'  WARNING: {len(unfilterable)} artifact(s) are not recday-keyed and '
                  f'may still pool excluded/stale recdays: {unfilterable}. '
                  f'Regenerate them from the filtered per-recday results.')

    if not out:
        print(f'load_glm_results({section_name!r}): no pickles found in {save_dir}')
    return out
