"""
Rave visualiser — the animal's motion toward reward as a techno-set light show.

Pick a session + neuron + trial and render the animal's real 2D path through the
maze as a glowing, reverby animation: fading neon trails, a marked maze with the
reward towers, a comet head that pulses + fires shockwave rings to the chosen
neuron, a synced firing-rate panel, and a subtle whole-scene strobe.

This is a *fun* visualiser, not analysis. Data comes from the same aligned pickle
the rest of the repo uses (``data_dic_for_yaren.pkl``); maze geometry reuses
``LOC_TO_GRID`` from :mod:`spatial_ratemaps`.

Typical use (see ``rave_visualiser.ipynb``)::

    import rave_visualiser as rv
    sess = rv.load_session('ah08_20250613_20250615', 0)
    anim, fig = rv.render_rave(sess, neuron=42, trial=2, palette='ice_uv')
    from IPython.display import HTML
    HTML(anim.to_html5_video())                      # inline
    rv.save_outputs(anim, 'rave_output/ah08_s0_n42_t2', fps=30)   # mp4 + gif
    rv.render_long_exposure(sess, neuron=42, trial=2)            # still PNG
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.collections import LineCollection
from matplotlib.patches import RegularPolygon, Rectangle
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from spatial_ratemaps import LOC_TO_GRID

# ─── paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DEFAULT_PKL = os.path.join(_REPO, "data", "processed_data", "data_dic_lec.pkl")
CACHE_DIR = os.path.join(_REPO, "data", "rave_cache")
OUTPUT_DIR = os.path.join(_HERE, "rave_output")

SAMPLE_HZ = 40.0  # Neuron_raw / XY are 40 Hz (25 ms bins)

# ─── maze look (ideal-grid units: adjacent towers are 1.0 apart) ──────────────
HEX_R = 0.20          # tower hexagon radius
CORR_W = 0.16         # corridor band width
PAD_SECONDS = 3.0     # context padding before/after the trial
CONTEXT_DIM = 0.45    # glow multiplier for the padding (vs 1.0 in-trial)
FLASH_LIFE = 14       # frames a full-screen reward flash lasts

# task-state colours A/B/C/D (match the repo's EVENT_COLOURS in lda_state_analysis_*)
STATE_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]
STATE_LABELS = ["A", "B", "C", "D"]

# module-level cache so the ~3.5 GB pickle loads only once per kernel
_DATA_DIC = None


# ─── palettes ─────────────────────────────────────────────────────────────────
# each palette: bg, maze_line, reward, comet colours + a trail colourmap
# (either a registered cmap name or a list of colours to ramp through).
PALETTES = {
    "ice_uv": dict(
        bg="#03010c", maze_line="#3a6ff7", reward="#bfe9ff", comet="#eaf6ff",
        trail=["#04004a", "#1f4fd6", "#7b5cff", "#c9b8ff", "#f2f8ff"],
    ),
    "neon_noir": dict(
        bg="#000000", maze_line="#00ffd5", reward="#ff2bd6", comet="#ffffff",
        trail=["#00204a", "#00e5ff", "#9b5cff", "#ff00e5", "#aaff00"],
    ),
    "plasma_fire": dict(
        bg="#0a0004", maze_line="#ff7b00", reward="#ffd000", comet="#fff3cc",
        trail=["#1b0033", "#7a0a8a", "#b5179e", "#ff7b00", "#ffe600"],
    ),
    "rainbow": dict(
        bg="#000000", maze_line="#777777", reward="#ffffff", comet="#ffffff",
        trail="hsv",
    ),
}


def _trail_cmap(palette):
    spec = PALETTES[palette]["trail"]
    if isinstance(spec, str):
        return plt.cm.get_cmap(spec)
    return LinearSegmentedColormap.from_list(f"rave_{palette}", spec)


# ─── data loading ─────────────────────────────────────────────────────────────
def _load_data_dic(pkl_path=DEFAULT_PKL):
    global _DATA_DIC
    if _DATA_DIC is None:
        print(f"Loading {os.path.basename(pkl_path)} (one-off, ~3.5 GB)…", flush=True)
        with open(pkl_path, "rb") as f:
            _DATA_DIC = pickle.load(f)
    return _DATA_DIC


def list_sessions(pkl_path=DEFAULT_PKL):
    """Print available mouse_recdays and their sessions (with #trials / #neurons)."""
    dd = _load_data_dic(pkl_path)
    for mr in dd:
        sess_ids = sorted(dd[mr].keys())
        print(f"\n{mr}  ({len(sess_ids)} sessions)")
        for s in sess_ids:
            v = dd[mr][s]
            nn = v.get("Neuron_raw")
            tt = v.get("Trial_times")
            nneur = 0 if nn is None else nn.shape[0]
            ntr = 0 if tt is None else len(tt)
            print(f"   session {s}: {nneur:3d} neurons, {ntr:2d} trials, task={v.get('Task')}")


def _ideal_node(n):
    """Tower n's position on a perfect square grid (towers 1.0 apart)."""
    r, c = LOC_TO_GRID[n]
    return np.array([c / 2.0, (4 - r) / 2.0])


def _tower_edges(node_xy):
    """Edges between towers that neighbour on the 3×3 lattice (share a corridor)."""
    edges = []
    ids = list(node_xy)
    for i, a in enumerate(ids):
        ra, ca = LOC_TO_GRID[a]
        for b in ids[i + 1:]:
            rb, cb = LOC_TO_GRID[b]
            if (ra == rb and abs(ca - cb) == 2) or (ca == cb and abs(ra - rb) == 2):
                edges.append((node_xy[a], node_xy[b]))
    return edges


def _maze_geometry(XY, Locs):
    """Square the maze: fit an affine from observed tower centroids onto a perfect
    grid. Returns ``(node_xy, edges, transform)`` where ``node_xy`` is the *ideal*
    grid (when squared) and ``transform`` (3×2) maps observed XY → that grid, or
    ``None`` if too few towers (then ``node_xy`` is the raw centroids).
    """
    obs = {}
    for n in range(1, 10):  # towers 1–9
        mask = Locs == n
        if mask.sum() >= 3:
            obs[n] = XY[mask].mean(axis=0)

    M = None
    if len(obs) >= 3:
        O = np.array([obs[n] for n in obs])
        G = np.array([_ideal_node(n) for n in obs])
        A = np.column_stack([O, np.ones(len(O))])          # (k,3) [x,y,1]
        M, *_ = np.linalg.lstsq(A, G, rcond=None)          # (3,2): [x,y,1]@M → ideal
        if not np.all(np.isfinite(M)):
            M = None

    if M is not None:
        node_xy = {n: _ideal_node(n) for n in obs}         # perfectly square
    else:
        node_xy = obs                                      # fallback: raw centroids
    return node_xy, _tower_edges(node_xy), M


def _apply_transform(XY, M):
    """Map (T,2) points through a (3,2) affine; identity if M is None."""
    if M is None:
        return XY
    return np.column_stack([XY, np.ones(len(XY))]) @ M


def load_session(mouse_recday, session_idx, pkl_path=DEFAULT_PKL, use_cache=True):
    """Load + align one session's arrays, ready for rendering.

    Returns a dict with truncated-aligned ``XY`` (T,2 smoothed), ``Locs`` (T,),
    ``FR`` (n_neurons, T), integer ``Trial_times``, ``node_xy`` (tower→xy),
    ``edges``, plus ``Task``, ``n_neurons``, ``n_trials``.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{mouse_recday}_{session_idx}.npz")
    if use_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        XY, Locs, FR, TT = z["XY"], z["Locs"], z["FR"], z["Trial_times"]
        task = z["Task"].item()
    else:
        dd = _load_data_dic(pkl_path)
        sess = dd[mouse_recday][session_idx]
        FR = np.asarray(sess["Neuron_raw"], dtype=float)
        Locs = np.asarray(sess["Locs_raw"]).ravel().astype(int)
        XY = np.asarray(sess["XY_raw"], dtype=float)
        if XY.ndim == 1:
            XY = XY.reshape(-1, 2)
        TT = np.asarray(sess["Trial_times"], dtype=float)
        task = str(sess.get("Task", session_idx))
        # align everything to the shortest length (repo convention)
        T = min(FR.shape[1], len(Locs), len(XY))
        FR, Locs, XY = FR[:, :T], Locs[:T], XY[:T]
        if use_cache:
            np.savez_compressed(cache, XY=XY, Locs=Locs, FR=FR,
                                Trial_times=TT, Task=np.array(task, dtype=object))

    # light smoothing of position for buttery motion
    XY = np.column_stack([gaussian_filter1d(XY[:, 0], 2.0),
                          gaussian_filter1d(XY[:, 1], 2.0)])
    # square the maze: warp the path onto a perfect grid (node_xy is then ideal)
    node_xy, edges, M = _maze_geometry(XY, Locs)
    XY = _apply_transform(XY, M)
    return dict(XY=XY, Locs=Locs, FR=FR, Trial_times=TT.astype(int),
                node_xy=node_xy, edges=edges, transform=M, Task=task,
                n_neurons=FR.shape[0], n_trials=len(TT),
                label=f"{mouse_recday}_s{session_idx}")


def get_trial(session, trial_idx, pad_seconds=PAD_SECONDS):
    """Slice one trial (± context padding) and locate the reward towers + reach times.

    Trial_times row = [A_start, B_start, C_start, D_start, trial_end];
    state X spans [col_X, col_{X+1}). The reward tower for each state is the
    modal tower occupied just before its closing boundary.

    ``pad_seconds`` of context is added before/after the trial, except the
    before-pad on the session's first trial and the after-pad on its last trial.
    ``trial_rel`` returns the true-trial bounds within the padded slice so the
    renderer can dim the padding.
    """
    XY, Locs, FR = session["XY"], session["Locs"], session["FR"]
    node_xy = session["node_xy"]
    T = len(Locs)
    row = session["Trial_times"][trial_idx]
    a_start, t_end = int(row[0]), int(min(row[-1], T))
    a_start = max(0, min(a_start, t_end - 1))

    pad = int(round(pad_seconds * SAMPLE_HZ))
    pad_before = pad if trial_idx > 0 else 0
    pad_after = pad if trial_idx < session["n_trials"] - 1 else 0
    start = max(0, a_start - pad_before)
    end = min(T, t_end + pad_after)

    xy = XY[start:end]
    fr = FR[:, start:end]
    locs = Locs[start:end]

    rewards = []  # (relative_frame, node_id, xy)
    for b in row[1:]:                       # B_start, C_start, D_start, trial_end
        b = int(min(b, T))
        if b <= a_start:
            continue
        w0 = max(0, b - 12)                 # ~0.3 s window before the boundary
        win = Locs[w0:b]
        towers = win[(win >= 1) & (win <= 9)]
        if len(towers):
            node = int(np.bincount(towers).argmax())
        else:                               # fall back to nearest known tower
            p = XY[min(b, T - 1)]
            node = min(node_xy, key=lambda n: np.hypot(*(node_xy[n] - p))) if node_xy else None
        if node in node_xy:
            rewards.append((b - start, node, node_xy[node]))
    a_rel = max(0, a_start - start)
    e_rel = min(t_end - start, end - start - 1)        # clamp to last valid index
    return dict(xy=xy, fr=fr, locs=locs, n_frames=end - start, rewards=rewards,
                trial_rel=(a_rel, max(a_rel, e_rel)), bounds=(start, end))


# ─── signal prep ──────────────────────────────────────────────────────────────
def _resample(arr, m, axis=0):
    n = arr.shape[axis]
    if n == m:
        return arr
    src = np.linspace(0, 1, n)
    dst = np.linspace(0, 1, m)
    kind = "cubic" if n >= 4 else ("linear" if n >= 2 else "nearest")
    return interp1d(src, arr, axis=axis, kind=kind, fill_value="extrapolate")(dst)


def _norm01(x):
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _prep_playback(trial, energy_raw, fps, speedup, extra=None, max_frames=450):
    """Resample trajectory + a raw 1-D energy signal to M playback frames.

    ``energy_raw`` (length n_frames) drives the comet glow / shockwaves / strobe —
    a single neuron's firing (neuron mode) or overall population activity (pop mode).
    ``extra`` maps name → array (n_frames, ...) to also resample into the result
    (``'state'`` uses nearest-neighbour; everything else uses the default interp).
    """
    n = trial["n_frames"]
    duration = n / SAMPLE_HZ
    m = int(np.clip(round(duration / max(speedup, 1e-3) * fps), 24, max_frames))

    xy = _resample(trial["xy"], m, axis=0)
    fr_raw = gaussian_filter1d(np.asarray(energy_raw, dtype=float), 2.0)
    fr = _resample(_norm01(fr_raw), m)
    fr = np.clip(fr, 0, 1)

    extras = {}
    if extra:
        idx_near = np.round(np.linspace(0, n - 1, m)).astype(int)
        for name, arr in extra.items():
            arr = np.asarray(arr)
            extras[name] = arr[idx_near] if name == "state" else _resample(arr, m, axis=0)

    def to_pb(rel):                         # raw frame → playback frame
        return int(round(rel / max(n - 1, 1) * (m - 1)))

    # reward reach-frames mapped into playback time
    rewards = [(to_pb(rel), node, xyp) for (rel, node, xyp) in trial["rewards"]]

    # context dimming: 1.0 inside the true trial, CONTEXT_DIM in the ±pad context
    a_rel, e_rel = trial.get("trial_rel", (0, n - 1))
    trial_lo = int(np.clip(to_pb(a_rel), 0, m - 1))
    trial_hi = int(np.clip(to_pb(e_rel), 0, m - 1))
    dim = np.full(m, CONTEXT_DIM)
    dim[trial_lo:trial_hi + 1] = 1.0

    # shockwaves: spawn where fr crosses up through a high threshold ("beat drop")
    thr = 0.72
    crossings = np.where((fr[1:] >= thr) & (fr[:-1] < thr))[0] + 1
    shock = [(int(c), xy[int(c)]) for c in crossings]

    return dict(xy=xy, fr=fr, m=m, fps=fps, rewards=rewards, shock=shock,
                trial_lo=trial_lo, trial_hi=trial_hi, dim=dim, **extras)


def _ring_alpha(age, life):
    return max(0.0, 1.0 - age / life)


def _hex(ax, center, radius, **kw):
    ax.add_patch(RegularPolygon(center, numVertices=6, radius=radius,
                                orientation=np.pi / 6, **kw))


def _draw_maze(ax, node_xy, edges, reward_nodes, target_node, pal, phase=0.0):
    """Square maze: wide neon corridor bands + glowing hexagon towers, with the
    current target tower brightened and wrapped in a pulsing halo."""
    mz, rw, cm = pal["maze_line"], pal["reward"], pal["comet"]

    # corridors as filled neon bands (axis-aligned because the maze is squared)
    for (a, b) in edges:
        horiz = abs(a[0] - b[0]) >= abs(a[1] - b[1])
        for wmul, al in [(1.9, 0.10), (1.0, 0.5)]:
            w = CORR_W * wmul
            if horiz:
                x0, ln = min(a[0], b[0]), abs(a[0] - b[0])
                rect = Rectangle((x0, a[1] - w / 2), ln, w)
            else:
                y0, ln = min(a[1], b[1]), abs(a[1] - b[1])
                rect = Rectangle((a[0] - w / 2, y0), w, ln)
            rect.set(facecolor=mz, edgecolor="none", alpha=al, zorder=1)
            ax.add_patch(rect)

    # hexagon towers (reward towers tinted with the reward colour)
    for n, p in node_xy.items():
        is_rew = n in reward_nodes
        col = rw if is_rew else mz
        _hex(ax, p, HEX_R * 1.45, facecolor=col, edgecolor="none", alpha=0.13, zorder=2)
        _hex(ax, p, HEX_R, facecolor=col, edgecolor="none",
             alpha=0.55 if is_rew else 0.30, zorder=2)
        _hex(ax, p, HEX_R, facecolor="none", edgecolor=col, lw=1.4, alpha=0.85, zorder=3)

    # current target: brighter hex + breathing halo
    if target_node in node_xy:
        p = node_xy[target_node]
        pulse = 0.5 + 0.5 * np.sin(phase)
        _hex(ax, p, HEX_R, facecolor=rw, edgecolor="none", alpha=0.8, zorder=3)
        _hex(ax, p, HEX_R * 1.15, facecolor="none", edgecolor=cm, lw=1.8, alpha=0.9, zorder=4)
        _hex(ax, p, HEX_R * (1.5 + 0.3 * pulse), facecolor="none", edgecolor=cm,
             lw=2.2, alpha=0.35 + 0.45 * pulse, zorder=4)


def _maze_limits(xy, node_xy):
    """Padded equal-aspect limits around a trajectory + the towers."""
    pts = [xy] + ([np.array(list(node_xy.values()))] if node_xy else [])
    allp = np.vstack(pts)
    pad = 0.12 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 1.0)
    xlim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ylim = (allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    return xlim, ylim, xlim[1] - xlim[0], ylim[1] - ylim[0]


def _draw_maze_frame(ax, i, S):
    """Draw the full maze scene for playback frame ``i`` (shared by the neuron and
    population renders). ``S`` holds the precomputed per-render arrays. Returns the
    population/firing ``glow`` for the caller's side panel."""
    pal, cmap, pb, dim, xy = S["pal"], S["cmap"], S["pb"], S["dim"], S["xy"]
    xlim, ylim, m, trail_len = S["xlim"], S["ylim"], S["m"], S["trail_len"]
    SHOCK_LIFE, BURST_LIFE = 22, 30

    ax.clear()
    glow = S["fr"][i] * dim[i]
    base = np.array(to_rgb(pal["bg"])); tint = np.array(to_rgb(pal["maze_line"]))
    ax.set_facecolor(tuple(np.clip(base + 0.07 * glow * (tint - base), 0, 1)))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")

    _draw_maze(ax, S["node_xy"], S["edges"], S["reward_nodes"], S["target_at"](i),
               pal, phase=i * 0.3)

    # reward reach burst (expanding ring at the tower)
    for (rf, _, p) in pb["rewards"]:
        age = i - rf
        if 0 <= age < BURST_LIFE:
            a = _ring_alpha(age, BURST_LIFE)
            ax.scatter(*p, s=300 + age * 320, facecolors="none",
                       edgecolors=pal["reward"], linewidths=2.5 * a, alpha=a, zorder=6)

    # continuous glowing trail ribbon (newest brightest, dimmed in the ±pad context)
    k0 = max(0, i - trail_len)
    P = xy[k0:i + 1]
    if len(P) > 1:
        segs = np.stack([P[:-1], P[1:]], axis=1)
        fidx = np.arange(k0, i)
        cols = cmap(fidx / max(m - 1, 1))
        ages = np.linspace(0, 1, len(segs))               # oldest → newest
        seg_a = (ages ** 2) * dim[fidx]
        for lw, am in [(9, 0.12), (4, 0.25), (1.8, 1.0)]:
            c = cols.copy(); c[:, 3] = seg_a * am
            ax.add_collection(LineCollection(segs, colors=c, linewidths=lw,
                                             capstyle="round", zorder=5))

    # shockwave rings on beat-drop frames
    for (sf, p) in pb["shock"]:
        age = i - sf
        if 0 <= age < SHOCK_LIFE:
            a = _ring_alpha(age, SHOCK_LIFE) * dim[sf]
            ax.scatter(*p, s=120 + age * 260, facecolors="none",
                       edgecolors=pal["comet"], linewidths=2.2 * a, alpha=0.8 * a, zorder=6)

    # comet head: size + brightness pulse with the energy signal
    head = xy[i]
    hs = 70 + 520 * glow
    for s, al in [(hs * 4, (0.10 + 0.20 * glow) * dim[i]),
                  (hs * 2, 0.25 * dim[i]), (hs, 0.95 * dim[i])]:
        ax.scatter(*head, s=s, color=pal["comet"], alpha=al, edgecolors="none", zorder=7)

    # full-screen colour flash when a reward triggers
    for k, (rf, _, _) in enumerate(S["rsorted"]):
        age = i - rf
        if 0 <= age < FLASH_LIFE:
            fa = 0.30 * (1 - age / FLASH_LIFE)
            ax.add_patch(Rectangle((xlim[0], ylim[0]), S["fw"], S["fh"],
                                   facecolor=S["flash_cols"][k], edgecolor="none",
                                   alpha=fa, zorder=8))

    ax.text(0.015, 0.975, S["hud"], transform=ax.transAxes, color="white", alpha=0.55,
            fontsize=8, va="top", family="monospace", zorder=10)
    return glow


def _build_scene(session, pb, palette, trail_len, hud):
    """Assemble the shared per-render context ``S`` used by :func:`_draw_maze_frame`."""
    pal = PALETTES[palette]
    node_xy, edges = session["node_xy"], session["edges"]
    rsorted = sorted(pb["rewards"], key=lambda r: r[0])
    rfr = np.array([r[0] for r in rsorted], dtype=int)
    rnd = [r[1] for r in rsorted]

    def target_at(i):
        k = int(np.searchsorted(rfr, i, side="left"))
        return rnd[k] if k < len(rnd) else None

    xlim, ylim, fw, fh = _maze_limits(pb["xy"], node_xy)
    return dict(pal=pal, cmap=_trail_cmap(palette), pb=pb, dim=pb["dim"], xy=pb["xy"],
                fr=pb["fr"], m=pb["m"], trail_len=trail_len, node_xy=node_xy, edges=edges,
                reward_nodes={node for (_, node, _) in pb["rewards"]}, target_at=target_at,
                rsorted=rsorted, flash_cols=plt.cm.hsv(np.linspace(0, 0.85, max(len(rsorted), 1))),
                xlim=xlim, ylim=ylim, fw=fw, fh=fh, hud=hud)


# ─── main render ──────────────────────────────────────────────────────────────
def render_rave(session, neuron, trial, palette="ice_uv", fps=30, speedup=3.0,
                trail_len=45, figscale=1.0):
    """Build the rave animation. Returns ``(anim, fig)``.

    Parameters
    ----------
    session : dict from :func:`load_session`
    neuron  : int neuron index
    trial   : int trial index
    palette : one of :data:`PALETTES`
    fps     : playback frames per second
    speedup : real-time multiplier (3 → 3× faster than life)
    trail_len : comet trail length in playback frames
    """
    pal = PALETTES[palette]
    bg = pal["bg"]
    tr = get_trial(session, trial)
    pb = _prep_playback(tr, tr["fr"][neuron], fps, speedup)
    fr, m = pb["fr"], pb["m"]
    S = _build_scene(session, pb, palette, trail_len,
                     hud=f"{session['label']}  •  neuron {neuron}  •  trial {trial}")
    fr_line_x = np.arange(m)

    fig = plt.figure(figsize=(7 * figscale, 8.4 * figscale), facecolor=bg)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.06,
                  left=0.02, right=0.98, top=0.97, bottom=0.07)
    ax = fig.add_subplot(gs[0]); axf = fig.add_subplot(gs[1])

    def update(i):
        glow = _draw_maze_frame(ax, i, S)

        # firing-rate panel with scrolling playhead + context shading
        axf.clear()
        axf.set_facecolor(bg)
        axf.plot(fr_line_x, fr, color=pal["maze_line"], lw=1.0, alpha=0.4)
        axf.fill_between(fr_line_x[:i + 1], fr[:i + 1], color=pal["comet"], alpha=0.18)
        axf.plot(fr_line_x[:i + 1], fr[:i + 1], color=pal["comet"], lw=1.4, alpha=0.9)
        if pb["trial_lo"] > 0:
            axf.axvspan(0, pb["trial_lo"], color="white", alpha=0.06)
        if pb["trial_hi"] < m - 1:
            axf.axvspan(pb["trial_hi"], m - 1, color="white", alpha=0.06)
        axf.axvline(i, color="white", lw=1.0, alpha=0.7)
        axf.scatter(i, fr[i], s=40 + 160 * glow, color=pal["comet"], zorder=5,
                    edgecolors="none")
        axf.set_xlim(0, m - 1); axf.set_ylim(-0.05, 1.1)
        axf.set_xticks([]); axf.set_yticks([])
        for sp in axf.spines.values():
            sp.set_visible(False)
        axf.text(0.01, 0.85, f"firing rate · neuron {neuron}", transform=axf.transAxes,
                 color="white", alpha=0.45, fontsize=7, family="monospace", va="top")
        return []

    anim = animation.FuncAnimation(fig, update, frames=m, interval=1000 / fps,
                                   blit=False)
    return anim, fig


# ─── population (PCA) version ─────────────────────────────────────────────────
def _session_pca(session, n_smooth=3, n_components=10):
    """Fit PCA on the whole session's z-scored, smoothed population activity and
    cache (on the session dict, keyed by ``n_smooth``) the projection, a per-frame
    population-energy signal, and per-timepoint task-state labels."""
    cache = session.setdefault("_pop_cache", {})
    if n_smooth in cache:
        return cache[n_smooth]

    from sklearn.decomposition import PCA                      # lazy (heavy import)
    FR = session["FR"].astype(float)                           # (n_neurons, T)
    mu = FR.mean(axis=1, keepdims=True)
    sig = FR.std(axis=1, keepdims=True)
    sig = np.where(sig == 0, 1.0, sig)
    Zs = gaussian_filter1d((FR - mu) / sig, n_smooth, axis=1)
    k = max(2, int(min(n_components, FR.shape[0] - 1, FR.shape[1] - 1)))
    pca = PCA(n_components=k)
    scores = pca.fit_transform(Zs.T)                           # (T, k)
    pop_energy = gaussian_filter1d(FR.mean(axis=0), n_smooth)  # (T,) overall activity

    from glm_analysis import compute_task_state_arrays         # lazy
    state_full = compute_task_state_arrays(session["Trial_times"])[0]
    state = np.zeros(FR.shape[1], dtype=int)
    L = min(len(state_full), FR.shape[1])
    state[:L] = state_full[:L]
    if L < FR.shape[1] and len(state_full):
        state[L:] = state_full[-1]

    out = dict(scores=scores, pop_energy=pop_energy, state=state,
               evr=pca.explained_variance_ratio_)
    cache[n_smooth] = out
    return out


def render_rave_pop(session, trial, palette="ice_uv", fps=30, speedup=3.0,
                    trail_len=45, n_smooth=3, pc_x=0, pc_y=1, figscale=1.0):
    """Population version: maze (left) beside a 2-D PCA of the population state
    (right). PCA is fit on the whole session and the single trial's window is
    projected; the trajectory is coloured by task state A/B/C/D and the maze comet
    is driven by overall population activity. Returns ``(anim, fig)``."""
    pal = PALETTES[palette]
    bg = pal["bg"]
    tr = get_trial(session, trial)
    start, end = tr["bounds"]
    P = _session_pca(session, n_smooth=n_smooth)
    proj = P["scores"][start:end][:, [pc_x, pc_y]]             # (n_frames, 2)
    pb = _prep_playback(tr, P["pop_energy"][start:end], fps, speedup,
                        extra={"scores": proj, "state": P["state"][start:end]})
    m, dim = pb["m"], pb["dim"]
    sc = pb["scores"]                                          # (m, 2)
    st = pb["state"].astype(int)                               # (m,)

    S = _build_scene(session, pb, palette, trail_len,
                     hud=f"{session['label']}  •  population  •  trial {trial}")

    # PCA-panel limits + state centroids + kept variance
    ppad = 0.10 * max(np.ptp(sc[:, 0]), np.ptp(sc[:, 1]), 1e-6)
    pxlim = (sc[:, 0].min() - ppad, sc[:, 0].max() + ppad)
    pylim = (sc[:, 1].min() - ppad, sc[:, 1].max() + ppad)
    centroids = {s: sc[st == s].mean(0) for s in range(4) if np.any(st == s)}
    evr = P["evr"]
    var2 = float(evr[pc_x] + evr[pc_y]) if len(evr) > max(pc_x, pc_y) else 0.0

    fig = plt.figure(figsize=(13 * figscale, 6.8 * figscale), facecolor=bg)
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.04,
                  left=0.01, right=0.99, top=0.96, bottom=0.04)
    ax = fig.add_subplot(gs[0]); axp = fig.add_subplot(gs[1])

    def update(i):
        glow = _draw_maze_frame(ax, i, S)

        axp.clear(); axp.set_facecolor(bg)
        axp.set_xlim(*pxlim); axp.set_ylim(*pylim)
        axp.set_aspect("equal"); axp.axis("off")

        # faint ghost of the whole trial trajectory + labelled state centroids
        axp.plot(sc[:, 0], sc[:, 1], color="white", alpha=0.10, lw=0.8, zorder=1)
        for s, c in centroids.items():
            axp.scatter(*c, s=260, color=STATE_COLOURS[s], alpha=0.18,
                        edgecolors="none", zorder=2)
            axp.text(*c, STATE_LABELS[s], color="white", alpha=0.7, fontsize=10,
                     ha="center", va="center", family="monospace", zorder=3)

        # state-coloured trail ribbon up to i (newest brightest, dimmed in context)
        k0 = max(0, i - trail_len)
        pts = sc[k0:i + 1]
        if len(pts) > 1:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            fidx = np.arange(k0, i)
            cols = np.array([to_rgb(STATE_COLOURS[st[j]]) for j in fidx])
            cols = np.column_stack([cols, np.ones(len(cols))])
            ages = np.linspace(0, 1, len(segs))
            seg_a = (ages ** 2) * dim[fidx]
            for lw, am in [(7, 0.18), (3, 0.4), (1.5, 1.0)]:
                c = cols.copy(); c[:, 3] = seg_a * am
                axp.add_collection(LineCollection(segs, colors=c, linewidths=lw,
                                                  capstyle="round", zorder=4))

        # reward ring at the population position when a reward fires
        for (rf, _, _) in S["rsorted"]:
            age = i - rf
            if 0 <= age < 26:
                a = _ring_alpha(age, 26)
                axp.scatter(*sc[min(rf, m - 1)], s=120 + age * 240, facecolors="none",
                            edgecolors=pal["reward"], linewidths=2.2 * a,
                            alpha=0.8 * a, zorder=5)

        # moving comet sized by population activity
        for s, al in [(70 + 360 * glow, 0.5), (35 + 180 * glow, 0.95)]:
            axp.scatter(*sc[i], s=s, color=pal["comet"], alpha=al,
                        edgecolors="none", zorder=6)

        # matching full-screen flash on the population panel
        for k, (rf, _, _) in enumerate(S["rsorted"]):
            age = i - rf
            if 0 <= age < FLASH_LIFE:
                fa = 0.30 * (1 - age / FLASH_LIFE)
                axp.add_patch(Rectangle((pxlim[0], pylim[0]), pxlim[1] - pxlim[0],
                                        pylim[1] - pylim[0], facecolor=S["flash_cols"][k],
                                        edgecolor="none", alpha=fa, zorder=7))

        axp.text(0.02, 0.975,
                 f"population PCA · PC{pc_x + 1} vs PC{pc_y + 1}  ({var2:.0%} var)",
                 transform=axp.transAxes, color="white", alpha=0.5, fontsize=8,
                 va="top", family="monospace", zorder=10)
        return []

    anim = animation.FuncAnimation(fig, update, frames=m, interval=1000 / fps, blit=False)
    return anim, fig


# ─── long exposure still ──────────────────────────────────────────────────────
def render_long_exposure(session, neuron, trial, palette="ice_uv", speedup=3.0,
                         fps=30, save_path=None, dpi=140, show=True):
    """Single 'light-painting' frame: the whole trajectory blended at once."""
    pal = PALETTES[palette]
    cmap = _trail_cmap(palette)
    bg = pal["bg"]
    tr = get_trial(session, trial)
    pb = _prep_playback(tr, neuron, fps, speedup)
    xy, fr, m = pb["xy"], pb["fr"], pb["m"]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=bg)
    ax.set_facecolor(bg)
    node_xy, edges = session["node_xy"], session["edges"]
    allp = np.vstack([xy] + ([np.array(list(node_xy.values()))] if node_xy else []))
    pad = 0.12 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 1.0)
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.axis("off")

    # squared maze: wide corridors + hex towers (reward towers tinted)
    reward_nodes = {node for (_, node, _) in pb["rewards"]}
    _draw_maze(ax, node_xy, edges, reward_nodes, target_node=None, pal=pal)

    # the whole path as one glowing continuous ribbon, brightness ∝ firing rate
    if len(xy) > 1:
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        cols = cmap(np.linspace(0, 1, len(segs)))
        cols[:, 3] = 0.05 + 0.45 * fr[:-1]
        for lw, am in [(7, 0.15), (3, 0.3), (1.2, 1.0)]:
            c = cols.copy(); c[:, 3] = cols[:, 3] * am
            ax.add_collection(LineCollection(segs, colors=c, linewidths=lw,
                                             capstyle="round", zorder=5))

    for (_, node, p) in pb["rewards"]:
        ax.scatter(*p, s=90, marker="*", color="white", alpha=0.95, zorder=6)

    ax.set_title(f"{session['label']} · neuron {neuron} · trial {trial}",
                 color="white", alpha=0.6, fontsize=9, family="monospace")
    fig.tight_layout()

    if save_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(
            OUTPUT_DIR, f"{session['label']}_n{neuron}_t{trial}_{palette}_still.png")
    fig.savefig(save_path, dpi=dpi, facecolor=bg)
    print(f"saved still → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


# ─── saving ───────────────────────────────────────────────────────────────────
def save_outputs(anim, basepath, fps=30, dpi=120, mp4=True, gif=True):
    """Write the animation to ``basepath.mp4`` (ffmpeg) and ``basepath.gif`` (pillow)."""
    if not os.path.isabs(basepath):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        basepath = os.path.join(OUTPUT_DIR, basepath)
    os.makedirs(os.path.dirname(basepath), exist_ok=True)
    paths = {}
    bgc = anim._fig.get_facecolor()
    if mp4:
        p = basepath + ".mp4"
        anim.save(p, writer="ffmpeg", fps=fps, dpi=dpi,
                  savefig_kwargs={"facecolor": bgc})
        paths["mp4"] = p; print(f"saved mp4 → {p}")
    if gif:
        p = basepath + ".gif"
        anim.save(p, writer="pillow", fps=min(fps, 20),
                  savefig_kwargs={"facecolor": bgc})
        paths["gif"] = p; print(f"saved gif → {p}")
    return paths
