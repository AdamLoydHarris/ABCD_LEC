"""
Piecewise-linear video frame clock from pinstate edges and one mouse's pyControl rsyncs.

Why not one line (measured 2026-09-18 on 389 sessions of both datasets): the camera drops
frames in bursts -- ah08/ly07 2025-06-16 s3 lose 230 frames between 500 and 900 s in many small
steps, bp04 2024-03-25 s4 loses 142 frames inside one 5 s pulse interval -- and one camera's rate
drifts (ah10 2025-06-13 s3: 59.7-60.0 fps block to block). A global line plus tolerance pairing
lost 30-70 % of the pulses in those sessions and gave clocks worse than a plain 60 fps line. The
pinstate also carries the paired box's pulses (twice as many edges as rsyncs), so pairing must
skip edges freely but may never run backwards.

Algorithm: a monotone alignment (dynamic programme) of the rsync train onto the edge train.
Each pulse is either matched to an edge or skipped (cost C_SKIP). Between consecutive matched
pulses the change in offset (edge_frame - rsync_ms * slope) is free within a jitter tolerance
that grows with the interval (rate drift), cheap when negative (frames dropped: C_DROP plus a
little per frame) and expensive when positive (frames cannot be added: C_ADD). A partner's edge
therefore costs a drop to reach and an addition to leave, and the path stays on this mouse's
train. The seed slope is the frame rate at which the histogram of all edge-minus-rsync
differences has the tallest 3-bin peak. The clock is the piecewise-linear interpolation through
the matched anchors, extrapolated at the global slope beyond them.

I/O: `fit_frame_clock(rising_frames, rsync_ms) -> dict` with anchors_ms, anchors_frames, a, b,
fps and QC numbers; `frames_at`, `ms_at_frame`, `uncertain_gaps`, `uncertain_mask`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

MAX_JUMP_FRAMES = 600            # 10 s of dropped frames is the most one step may bridge
MAX_SKIP = 5                     # consecutive pulses a path may skip
C_SKIP = 4.0                     # cost of leaving a pulse unmatched
C_DROP = 3.0                     # fixed cost of a dropped-frame step (plus 0.02 per frame)
C_ADD = 30.0                     # fixed cost of an added-frame step (plus 0.2 per frame)
JITTER_FRAMES = 1.0              # free offset change between pulses ...
JITTER_PER_S = 0.4               # ... plus this per second of interval (rate drift, 0.4 fps)
CANDIDATE_SPAN_FRAMES = 3000.0   # edges considered per pulse: within this of the seed offset
STEP_JUMP_FRAMES = 2.0           # anchor steps beyond this count as jumps in the QC
UNCERTAIN_STEP_FRAMES = 6.0      # a gap whose step exceeds this (0.1 s) is flagged uncertain
GAP_UNCERTAIN_S = 30.0           # ... as is any gap longer than this with a step > STEP_JUMP


def _peak(up: np.ndarray, rs: np.ndarray, fps: float, bin_ms: float = 20.0) -> Tuple[int, float]:
    d = (up[None, :] / fps * 1000.0 - rs[:, None]).ravel()
    edges = np.arange(d.min(), d.max() + bin_ms, bin_ms)
    h, _ = np.histogram(d, edges)
    k = int(h.argmax())
    return int(h[max(k - 1, 0):k + 2].sum()), (edges[k] + edges[k + 1]) / 2


def _seed(up: np.ndarray, rs: np.ndarray, fps_range: Tuple[float, float]) -> Tuple[float, float]:
    """(fps, offset_ms) of the best-supported constant clock."""
    grid = np.arange(fps_range[0], fps_range[1], 0.005)
    coarse = grid[int(np.argmax([_peak(up, rs, f)[0] for f in grid]))]
    fine = np.arange(coarse - 0.01, coarse + 0.01, 0.0002)
    fps0 = fine[int(np.argmax([_peak(up, rs, f)[0] for f in fine]))]
    _, offset_ms = _peak(up, rs, fps0)
    return float(fps0), float(offset_ms)


def _align(up: np.ndarray, rs: np.ndarray, slope: float, offset_frames: float,
           max_jump: float, max_skip: int) -> np.ndarray:
    """Monotone alignment; returns frame_of (NaN for skipped pulses)."""
    n = len(rs)
    O = up[None, :] - rs[:, None] * slope                      # offset of every edge for every pulse
    cands: List[np.ndarray] = []
    for i in range(n):
        j = np.where(np.abs(O[i] - offset_frames) <= CANDIDATE_SPAN_FRAMES)[0]
        cands.append(j)
    INF = np.inf
    cost = [np.full(len(c), INF) for c in cands]
    back = [np.full((len(c), 2), -1, dtype=int) for c in cands]   # (i_prev, k_prev)
    for i in range(n):
        if len(cands[i]) == 0:
            continue
        cost[i][:] = i * C_SKIP                                  # start here, earlier pulses skipped
        oi = O[i, cands[i]]
        fi = up[cands[i]]
        for ip in range(max(0, i - max_skip - 1), i):
            if len(cands[ip]) == 0 or not np.isfinite(cost[ip]).any():
                continue
            dt_s = (rs[i] - rs[ip]) / 1000.0
            tol = JITTER_FRAMES + JITTER_PER_S * dt_s
            d = oi[None, :] - O[ip, cands[ip]][:, None]        # (k_prev, k)
            mono = fi[None, :] > up[cands[ip]][:, None]
            step = np.where(np.abs(d) <= tol, 0.3 * np.abs(d) / tol,
                            np.where(d < 0, C_DROP + 0.02 * (np.abs(d) - tol), C_ADD + 0.2 * (d - tol)))
            step = np.where(mono & (np.abs(d) <= max_jump), step, INF)
            total = cost[ip][:, None] + step + (i - ip - 1) * C_SKIP
            kp = total.argmin(axis=0)
            best = total[kp, np.arange(len(cands[i]))]
            better = best < cost[i]
            cost[i][better] = best[better]
            back[i][better] = np.stack([np.full(better.sum(), ip), kp[better]], axis=1)
    # best end: any (i, k) plus skipping the pulses after i
    end_i, end_k, end_cost = -1, -1, INF
    for i in range(n):
        if len(cands[i]) == 0:
            continue
        c = cost[i] + (n - 1 - i) * C_SKIP
        k = int(c.argmin())
        if c[k] < end_cost:
            end_i, end_k, end_cost = i, k, float(c[k])
    frame_of = np.full(n, np.nan)
    i, k = end_i, end_k
    while i >= 0:
        frame_of[i] = up[cands[i][k]]
        i, k = back[i][k]
    return frame_of


def fit_frame_clock(rising_frames: np.ndarray, rsync_ms: np.ndarray,
                    fps_range: Tuple[float, float] = (58.5, 61.5),
                    max_jump_frames: float = MAX_JUMP_FRAMES, max_skip: int = MAX_SKIP) -> Dict:
    """Alignment fit; see module docstring. Raises ValueError when fewer than half the rsyncs
    pair or the global rate is not a camera's."""
    up = np.sort(np.asarray(rising_frames, dtype=float))
    rs = np.asarray(rsync_ms, dtype=float)
    n = len(rs)
    if len(up) < 5 or n < 5:
        raise ValueError(f"too few pulses (edges {len(up)}, rsyncs {n})")
    fps0, offset_ms = _seed(up, rs, fps_range)
    slope0 = fps0 / 1000.0
    frame_of = _align(up, rs, slope0, offset_ms * slope0, max_jump_frames, max_skip)
    matched = ~np.isnan(frame_of)
    if matched.sum() < max(5, n // 2):
        raise ValueError(f"only {int(matched.sum())}/{n} rsyncs pair (seed {fps0:.3f} fps)")
    m_t, m_f = rs[matched], frame_of[matched]
    keep = np.concatenate([[True], np.diff(m_f) > 0])
    m_t, m_f = m_t[keep], m_f[keep]
    # The camera rate is the seed slope (the best-supported constant clock). A least-squares
    # line through anchors that straddle dropped frames tilts towards the drops (ah08 2025-06-16
    # s3: 59.71 fps for a 59.95 camera) and would extrapolate wrongly and turn the drift of the
    # clean half into fake jumps, so steps, jumps and extrapolation all use the seed slope.
    a = slope0
    b = float(np.median(m_f - a * m_t))
    a_fit, _ = np.polyfit(m_t, m_f, 1)
    resid = m_f - (a * m_t + b)
    steps = np.diff(m_f) - np.diff(m_t) * a
    jumps = steps[np.abs(steps) > STEP_JUMP_FRAMES]
    if len(jumps) and np.abs(jumps).max() > max_jump_frames:
        raise ValueError(f"implausible frame clock: largest jump {np.abs(jumps).max():.0f} frames "
                         f"at {fps0:.2f} fps (pinstate truncated?)")
    if len(m_t) >= 3:
        loo = m_f[1:-1] - (m_f[:-2] + (m_t[1:-1] - m_t[:-2]) * (m_f[2:] - m_f[:-2]) / (m_t[2:] - m_t[:-2]))
        local_rms = float(np.sqrt(np.mean(loo ** 2)))
    else:
        local_rms = float("nan")
    um = ~matched
    run_s, run_len = _longest_run(um)
    last_anchor = int(np.where(matched)[0][-1])
    clock = dict(anchors_ms=m_t, anchors_frames=m_f, a=float(a), b=b, fps=float(fps0))
    gaps = uncertain_gaps(clock)
    clock.update(fps_global_fit=float(a_fit * 1000.0),
                 n_rsync=int(n), n_matched=int(len(m_t)), n_edges=int(len(up)),
                 n_edges_unmatched=int(len(up) - len(m_t)),
                 resid_rms_frames=float(np.sqrt(np.mean(resid ** 2))),
                 local_resid_rms_frames=local_rms,
                 n_jumps=int(len(jumps)), max_jump_frames=float(np.abs(jumps).max()) if len(jumps) else 0.0,
                 total_drop_frames=float(-steps[steps < -STEP_JUMP_FRAMES].sum()) if len(steps) else 0.0,
                 total_add_frames=float(steps[steps > STEP_JUMP_FRAMES].sum()) if len(steps) else 0.0,
                 unmatched_run_max=int(run_len),
                 unmatched_run_max_s=float((rs[min(run_s + run_len, n - 1)] - rs[run_s]) / 1000.0) if run_len else 0.0,
                 tail_unmatched_s=float((rs[-1] - rs[last_anchor]) / 1000.0),
                 n_uncertain_gaps=int(len(gaps)),
                 uncertain_s=float(sum(t1 - t0 for t0, t1, _ in gaps) / 1000.0),
                 first_rsync_frame=float(m_f[0] if matched[0] else a * rs[0] + b),
                 seed_fps=float(fps0))
    return clock


def _longest_run(mask: np.ndarray) -> Tuple[int, int]:
    """(start, length) of the longest run of True."""
    best_s, best_l, s = 0, 0, None
    for i, m in enumerate(np.append(mask, False)):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if i - s > best_l:
                best_s, best_l = s, i - s
            s = None
    return best_s, best_l


def uncertain_gaps(clock: Dict) -> List[Tuple[float, float, float]]:
    """Anchor gaps inside which the clock is interpolated across dropped frames whose position is
    unknown: step beyond UNCERTAIN_STEP_FRAMES, or a gap longer than GAP_UNCERTAIN_S with a step
    beyond STEP_JUMP_FRAMES. Returns [(t0_ms, t1_ms, step_frames), ...]."""
    m_t, m_f, a = clock['anchors_ms'], clock['anchors_frames'], clock['a']
    out = []
    for t0, t1, f0, f1 in zip(m_t[:-1], m_t[1:], m_f[:-1], m_f[1:]):
        step = (f1 - f0) - (t1 - t0) * a
        if abs(step) > UNCERTAIN_STEP_FRAMES or ((t1 - t0) > GAP_UNCERTAIN_S * 1000.0 and abs(step) > STEP_JUMP_FRAMES):
            out.append((float(t0), float(t1), float(step)))
    return out


def uncertain_mask(clock: Dict, t_ms) -> np.ndarray:
    """True for times inside an uncertain gap (see `uncertain_gaps`)."""
    t = np.asarray(t_ms, dtype=float)
    mask = np.zeros(len(t), dtype=bool)
    for t0, t1, _ in uncertain_gaps(clock):
        mask |= (t > t0) & (t < t1)
    return mask


def frames_at(clock: Dict, t_ms) -> np.ndarray:
    """pyControl time (ms) -> fractional frame through the anchors; global slope beyond them."""
    t = np.asarray(t_ms, dtype=float)
    m_t, m_f, a = clock['anchors_ms'], clock['anchors_frames'], clock['a']
    f = np.interp(t, m_t, m_f)
    lo, hi = t < m_t[0], t > m_t[-1]
    f[lo] = m_f[0] + (t[lo] - m_t[0]) * a
    f[hi] = m_f[-1] + (t[hi] - m_t[-1]) * a
    return f


def ms_at_frame(clock: Dict, frame: float) -> float:
    """Inverse of `frames_at` for one frame index."""
    m_t, m_f, a = clock['anchors_ms'], clock['anchors_frames'], clock['a']
    if frame <= m_f[0]:
        return float(m_t[0] + (frame - m_f[0]) / a)
    if frame >= m_f[-1]:
        return float(m_t[-1] + (frame - m_f[-1]) / a)
    return float(np.interp(frame, m_f, m_t))
