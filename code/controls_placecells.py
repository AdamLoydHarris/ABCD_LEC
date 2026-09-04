"""
Synthetic 'place cells + speed modulation' control for the recday-averaged
task-phase ring analysis.

Question it answers: the recday-averaged pooled top/diam sits modestly above the
noise floor (real ~0.16-0.34 vs noise ~0.07, ring 0.83). Is that explained by a
population of consistently-tuned PLACE CELLS with SPEED MODULATION (slow near
rewards, fast between), rather than an abstract task ring?

Logic: each session has a different ABCD reward layout, so the physical A->B->C->D
path differs. A fixed place cell therefore fires at a DIFFERENT task-phase each
session and averages OUT across sessions. But the speed profile in task-phase
coordinates is CONSISTENT (slow at the 4 rewards, fast between), so a
speed-modulated component SURVIVES pooling. This generator builds such a
population on the real 3x3 maze and feeds it through the UNCHANGED pipeline
(`analyse_recday_taskphase_ring`) so the result is directly comparable.

Run:  python code/controls_placecells.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import persistent_homology_analysis as ph          # noqa: E402
import glm_analysis_v2 as glm                       # noqa: E402


def node_xy(node):
    """Node 1..9 -> (x, y) on the 3x3 grid."""
    i = int(node) - 1
    return np.array([i % 3, i // 3], dtype=float)


def _bfs_path(graph, src, dst):
    """Shortest node path src->dst on the maze graph (list of nodes, inclusive)."""
    if src == dst:
        return [src]
    prev = {src: None}
    frontier = [src]
    while frontier:
        nxt = []
        for node in frontier:
            for nb in graph.get(node, ()):  # graph values are sets
                if nb not in prev:
                    prev[nb] = node
                    if nb == dst:
                        path = [dst]
                        while path[-1] is not None:
                            path.append(prev[path[-1]])
                        return list(reversed(path[:-1]))
                    nxt.append(nb)
        frontier = nxt
    return [src, dst]  # disconnected fallback (shouldn't happen on this maze)


def leg_positions(a, b, graph, n=90):
    """Continuous (n,2) positions along the shortest path from reward a to b."""
    pts = np.array([node_xy(x) for x in _bfs_path(graph, a, b)])
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = np.r_[0, np.cumsum(seglen)]
    if d[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    u = np.linspace(0, d[-1], n)
    return np.stack([np.interp(u, d, pts[:, 0]), np.interp(u, d, pts[:, 1])], axis=1)


def session_positions(abcd, graph, n_per_state=90):
    """(4*n_per_state, 2) task-phase positions for one A->B->C->D->A loop."""
    seq = list(abcd) + [abcd[0]]
    return np.vstack([leg_positions(a, b, graph, n_per_state)
                      for a, b in zip(seq[:-1], seq[1:])])


def speed_profile(n_bins=360, n_states=4):
    """Speed vs task-phase: 0 at each reward (leg ends), 1 mid-leg. 4-periodic."""
    per = n_bins // n_states
    frac = np.arange(per) / per
    return np.tile(np.sin(np.pi * frac), n_states)


def make_recday(kind="place_speed", n_neurons=120, n_sessions=8, n_trials=15,
                field_sigma=0.9, speed_gain_sd=1.5, noise=0.6, seed=0):
    """Build a fake recday_dict {sess: {'Neurons_norm': (N, n_trials, 360), ...}}.

    kind: 'place' (fields only), 'speed' (flat field x speed gain), or
    'place_speed' (fields x speed gain). Speed gains are mixed-sign so the
    population is diverse (PCA rank > 1).
    """
    rng = np.random.default_rng(seed)
    graph = glm._build_maze_graph()
    fields = rng.uniform([0, 0], [2, 2], size=(n_neurons, 2))
    gains = rng.normal(0, speed_gain_sd, n_neurons)
    v = speed_profile()
    recday = {}
    for s in range(n_sessions):
        abcd = rng.choice(np.arange(1, 10), size=4, replace=False)
        pos = session_positions(abcd, graph)                       # (360, 2)
        d2 = ((pos[None] - fields[:, None]) ** 2).sum(-1)          # (N, 360)
        place = np.exp(-d2 / (2 * field_sigma ** 2))
        gain = 1.0 + gains[:, None] * v[None, :]                   # (N, 360)
        if kind == "place":
            base = place
        elif kind == "speed":
            base = np.ones_like(place) * gain
        else:                                                      # place_speed
            base = place * gain
        trials = base[:, None, :] + rng.normal(0, noise, (n_neurons, n_trials, 360))
        recday[s] = {"Neurons_norm": trials,
                     "Task": tuple(int(x) for x in abcd)}
    return recday


# ---------------------------------------------------------------------------
# REAL-behaviour version: drive the synthetic cells with a real recday's actual
# positions / speed / reward-configs (not the idealized maze cartoon above).
# ---------------------------------------------------------------------------

def _warp_to_taskphase(raw_1d, trial_times_bins, n_per=90, n_states=4):
    """Reimplementation of glm_analysis_v2._raw_to_norm (mean aggregator) so it is
    available on the LEC side too. Warps a per-bin signal to (n_trials, n_per*n_states)."""
    from scipy import stats as _st
    raw = np.asarray(raw_1d, dtype=float)
    tt = np.asarray(trial_times_bins, dtype=int)
    if tt.ndim != 2 or tt.shape[1] < 2 or tt.shape[0] == 0:
        return np.zeros((0, n_per * n_states))
    bounds = np.hstack((np.concatenate(tt[:, :-1]), [tt[-1, -1]])).astype(int)
    rebinned = []
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if b > a and a >= 0 and b <= raw.shape[0]:
            seg = raw[a:b]
            if len(seg) < n_per:
                seg = np.repeat(seg, 10) / 10.0
            rebinned.append(_st.binned_statistic(np.arange(len(seg)), seg,
                            statistic="mean", bins=n_per)[0])
    n_full = (len(rebinned) // n_states) * n_states
    if n_full == 0:
        return np.zeros((0, n_per * n_states))
    return np.asarray(rebinned[:n_full]).reshape(n_full // n_states, n_per * n_states)


def _warp_to_taskphase_multi(X, trial_times_bins, n_per=90, n_states=4):
    """Vectorised warp of a (n_neurons, T) signal to (n_neurons, n_trials,
    n_per*n_states) — equivalent to `_warp_to_taskphase` per row but ~100x faster
    (equal-width bin means over all neurons at once)."""
    tt = np.asarray(trial_times_bins, dtype=int)
    T = X.shape[1]
    if tt.ndim != 2 or tt.shape[1] < 2 or tt.shape[0] == 0:
        return np.zeros((X.shape[0], 0, n_per * n_states))
    bounds = np.hstack((np.concatenate(tt[:, :-1]), [tt[-1, -1]])).astype(int)
    legs = []
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if b > a and a >= 0 and b <= T:
            seg = X[:, a:b]
            if seg.shape[1] < n_per:
                seg = np.repeat(seg, 10, axis=1) / 10.0
            edges = np.linspace(0, seg.shape[1], n_per + 1).astype(int)
            legs.append(np.stack([seg[:, edges[k]:edges[k + 1]].mean(1)
                                  for k in range(n_per)], axis=1))   # (n, n_per)
    n_full = (len(legs) // n_states) * n_states
    if n_full == 0:
        return np.zeros((X.shape[0], 0, n_per * n_states))
    arr = np.stack(legs[:n_full], axis=1)                            # (n, n_full, n_per)
    return arr.reshape(X.shape[0], n_full // n_states, n_per * n_states)


def _goalprogress_frac(trial_times_bins, T):
    """Per-bin within-leg goal progress in [0,1] (0 = just left a reward, 1 = at
    next reward), period = one leg. Non-cyclic (endpoints differ) -> arc, not ring."""
    tt = np.asarray(trial_times_bins, dtype=int)
    bounds = np.hstack((np.concatenate(tt[:, :-1]), [tt[-1, -1]])).astype(int)
    gp = np.full(T, np.nan)
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if b > a and a >= 0 and b <= T:
            gp[a:b] = (np.arange(a, b) - a) / (b - a)
    return gp


def real_behaviour_recday(recday_dict, kind, config, sessions=None, n_neurons=120,
                          noise=0.3, speed_gain_sd=1.5, seed=0):
    """Synthetic cells driven by a REAL recday's behaviour, warped to task phase.

    kind:
      'place'       — Gaussian place field in real XY space (physical place cell).
      'speed'       — flat field * real-speed gain (pure speed modulation).
      'place_speed' — place field * real-speed gain.
      'goalprog'    — tuned to real within-leg goal progress [0,1] (distance-to-
                      reward); the positive control that should fold to an arc.
    Shared cell params across sessions (that's what makes place session-inconsistent
    but goalprog/speed consistent). Returns a fake recday_dict for
    `ph.analyse_recday_taskphase_ring`.
    """
    import glm_analysis_v2 as glm
    rng = np.random.default_rng(seed)
    keys = list(recday_dict.keys()) if sessions is None else sessions
    # need real position + trial boundaries; Neurons_norm NOT required (we
    # generate synthetic firing) so this also works for a light PFC build.
    valid = [s for s in keys if recday_dict.get(s, {}).get("XY_raw") is not None
             and recday_dict[s].get("Trial_times") is not None
             and np.asarray(recday_dict[s]["Trial_times"]).shape[0] >= 3]
    if not valid:
        return None
    allxy = np.vstack([np.asarray(recday_dict[s]["XY_raw"], float) for s in valid])
    lo, hi = np.nanpercentile(allxy, [2, 98], axis=0)
    fields = rng.uniform(lo, hi, size=(n_neurons, 2))
    field_sigma = 0.1 * float(np.mean(hi - lo))
    gains = rng.normal(0, speed_gain_sd, n_neurons)
    gp_pref = rng.uniform(0, 1, n_neurons)          # preferred goal-progress
    gp_sigma = 0.12

    fake = {}
    for si, s in enumerate(valid):
        sd = recday_dict[s]
        XY = np.asarray(sd["XY_raw"], float)
        nr = sd.get("Neuron_raw")
        T = XY.shape[0] if nr is None else min(XY.shape[0], np.asarray(nr).shape[1])
        XY = XY[:T]
        tt = np.clip(np.asarray(sd["Trial_times"], float).astype(int), 0, T)
        if kind == "goalprog":
            gp = _goalprogress_frac(tt, T)
            base = np.exp(-((gp[None, :] - gp_pref[:, None]) ** 2) / (2 * gp_sigma ** 2))
            base = np.nan_to_num(base)
        else:
            speed = glm.smooth_and_calculate_scalar_derivatives(XY)[:, 2]
            spz = np.nan_to_num((speed - np.nanmean(speed)) / (np.nanstd(speed) + 1e-9))
            d2 = ((XY[None, :, :] - fields[:, None, :]) ** 2).sum(-1)
            place = np.exp(-d2 / (2 * field_sigma ** 2))
            gain = 1.0 + gains[:, None] * spz[None, :]
            base = place if kind == "place" else (
                np.ones_like(place) * gain if kind == "speed" else place * gain)
        firing = base + rng.normal(0, noise, base.shape)
        norm = _warp_to_taskphase_multi(firing, tt, config.n_bins_per_state,
                                        config.n_states)
        if norm.shape[1] >= 1:
            fake[si] = {"Neurons_norm": norm,
                        "Task": tuple(int(x) for x in np.asarray(sd.get("Task", [0, 0, 0, 0]))[:4])}
    return fake or None


def run_real_behaviour(recday_dict, config, sessions=None, seed=0, verbose=True):
    """Run the four real-behaviour variants through the recday-averaged pipeline.
    Returns a list of row dicts (model, top/diam, coverage, legSim, fold top/diam)."""
    rows = []
    for kind in ("place", "speed", "place_speed", "goalprog"):
        rd = real_behaviour_recday(recday_dict, kind, config, sessions, seed=seed)
        rows.append(_row(kind, ph.analyse_recday_taskphase_ring(rd, config)))
    if verbose:
        print(f"{'model':16}{'top/diam':>10}{'coverage':>10}{'legSim':>8}{'fold_td':>9}"
              f"   [real arc fold_td~0.39]")
        for r in rows:
            print(f"{r['model']:16}{r['top_over_diam']:>10.3f}{r['coverage']:>10.3f}"
                  f"{r['leg_similarity']:>8.2f}{r['fold_top_over_diam']:>9.3f}")
    return rows


def plot_real_behaviour_comparison(recday_dict, config, real_result, seed=0):
    """Fig: the 4 real-behaviour variants + REAL neural, each as 2D(goal progress)
    / folded / 3D. Returns (fig, rows). Cheap (~30 s) thanks to the vectorised warp."""
    import matplotlib.pyplot as plt
    variants = []
    for kind in ("place", "speed", "place_speed", "goalprog"):
        r = ph.analyse_recday_taskphase_ring(
            real_behaviour_recday(recday_dict, kind, config, seed=seed), config)
        r["mouse_recday"] = kind
        variants.append((kind, r))
    variants.append(("REAL", real_result))
    rows = [_row(n if n != "REAL" else "REAL neural", r) for n, r in variants]

    nrow = len(variants)
    fig = plt.figure(figsize=(11, 3.3 * nrow))
    for row, (name, r) in enumerate(variants):
        X, X90 = r["_X"], r["_X90"]; nb, per = X.shape[0], X90.shape[0]
        gp = np.tile(np.arange(per), nb // per)
        a1 = fig.add_subplot(nrow, 3, row * 3 + 1)
        a1.scatter(X[:, 0], X[:, 1], c=gp, cmap="hsv", s=9)
        a1.set_ylabel(f"{name}\nlegSim={r['leg_similarity']:.2f}\n"
                      f"fold_td={r['fold']['top_over_diameter']:.2f}", fontsize=7)
        a1.set_aspect("equal", "datalim"); a1.set_xticks([]); a1.set_yticks([])
        if row == 0:
            a1.set_title("2D (goal progress)", fontsize=8)
        a2 = fig.add_subplot(nrow, 3, row * 3 + 2)
        a2.plot(np.r_[X90[:, 0], X90[0, 0]], np.r_[X90[:, 1], X90[0, 1]],
                "-", color="0.75", lw=0.6)
        a2.scatter(X90[:, 0], X90[:, 1], c=np.arange(per), cmap="hsv", s=14)
        a2.set_aspect("equal", "datalim"); a2.set_xticks([]); a2.set_yticks([])
        if row == 0:
            a2.set_title("folded (goal progress)", fontsize=8)
        a3 = fig.add_subplot(nrow, 3, row * 3 + 3, projection="3d")
        ph.plot_ring_3d(r, color_by="goalprogress", which="full", ax=a3)
    fig.suptitle(f"Real-behaviour place/speed/goalprog vs REAL neural "
                 f"({real_result.get('mouse_recday','')}) — 2D + 3D", fontsize=10)
    fig.tight_layout()
    return fig, rows


def _row(model, r):
    return dict(model=model,
                top_over_diam=r["geometry"]["top_over_diameter"],
                coverage=r["coverage"],
                leg_similarity=r.get("leg_similarity", float("nan")),
                fold_top_over_diam=r.get("fold", {}).get("top_over_diameter", float("nan")))


def run(seed=0, verbose=True):
    """Run the three place/speed variants + a true-ring sanity through the
    UNCHANGED recday-averaged pipeline. Returns a list of row dicts (model,
    top/diam, coverage, leg_similarity, folded top/diam) for tabulation."""
    cfg = ph.PHConfig(coeff_fields=(2,), n_pca=6, min_neurons=12)
    rows = []
    for kind in ("place", "speed", "place_speed"):
        rows.append(_row(kind, ph.analyse_recday_taskphase_ring(
            make_recday(kind=kind, seed=seed), cfg)))
    # sanity: the true abstract-ring generator still fires through the same code
    rng = np.random.default_rng(seed)
    NB, NN = 360, 60
    phase = 2 * np.pi * np.arange(NB) / NB
    pref = rng.uniform(0, 2 * np.pi, NN)
    ring_rd = {s: {"Neurons_norm": np.stack(
        [np.exp(2 * np.cos(phase[None] - pref[:, None]))
         + rng.normal(0, 1.0, (NN, NB)) for _ in range(15)], axis=1)}
        for s in range(8)}
    rows.append(_row("TRUE ring (ctl)", ph.analyse_recday_taskphase_ring(ring_rd, cfg)))
    if verbose:
        print(f"{'model':16}{'top/diam':>10}{'coverage':>10}{'legSim':>8}"
              f"{'fold_td':>9}   [noise 0.07 | real 0.16-0.34 | ring 0.83]")
        for row in rows:
            print(f"{row['model']:16}{row['top_over_diam']:>10.3f}{row['coverage']:>10.3f}"
                  f"{row['leg_similarity']:>8.2f}{row['fold_top_over_diam']:>9.3f}")
    return rows


if __name__ == "__main__":
    run()
