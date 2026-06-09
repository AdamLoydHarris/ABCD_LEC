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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from spatial_ratemaps import LOC_TO_GRID

# ─── paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DEFAULT_PKL = os.path.join(_REPO, "data", "processed_data", "data_dic_for_yaren.pkl")
CACHE_DIR = os.path.join(_REPO, "data", "rave_cache")
OUTPUT_DIR = os.path.join(_HERE, "rave_output")

SAMPLE_HZ = 40.0  # Neuron_raw / XY are 40 Hz (25 ms bins)

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


def _maze_geometry(XY, Locs):
    """Tower centroids in the trajectory's own XY frame + neon edge list."""
    node_xy = {}
    for n in range(1, 10):  # towers 1–9
        mask = Locs == n
        if mask.sum() >= 3:
            node_xy[n] = XY[mask].mean(axis=0)
    # connect towers that are neighbours on the 3×3 tower lattice (share a corridor)
    edges = []
    towers = {n: LOC_TO_GRID[n] for n in node_xy}
    ids = list(towers)
    for i, a in enumerate(ids):
        ra, ca = towers[a]
        for b in ids[i + 1:]:
            rb, cb = towers[b]
            if (ra == rb and abs(ca - cb) == 2) or (ca == cb and abs(ra - rb) == 2):
                edges.append((node_xy[a], node_xy[b]))
    return node_xy, edges


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
    node_xy, edges = _maze_geometry(XY, Locs)
    return dict(XY=XY, Locs=Locs, FR=FR, Trial_times=TT.astype(int),
                node_xy=node_xy, edges=edges, Task=task,
                n_neurons=FR.shape[0], n_trials=len(TT),
                label=f"{mouse_recday}_s{session_idx}")


def get_trial(session, trial_idx):
    """Slice one trial and locate the four reward towers + reach times.

    Trial_times row = [A_start, B_start, C_start, D_start, trial_end];
    state X spans [col_X, col_{X+1}). The reward tower for each state is the
    modal tower occupied just before its closing boundary.
    """
    XY, Locs, FR = session["XY"], session["Locs"], session["FR"]
    node_xy = session["node_xy"]
    T = len(Locs)
    row = session["Trial_times"][trial_idx]
    start, end = int(row[0]), int(min(row[-1], T))
    start = max(0, min(start, end - 1))

    xy = XY[start:end]
    fr = FR[:, start:end]
    locs = Locs[start:end]

    rewards = []  # (relative_frame, node_id, xy)
    for b in row[1:]:                       # B_start, C_start, D_start, trial_end
        b = int(min(b, T))
        if b <= start:
            continue
        w0 = max(start, b - 12)             # ~0.3 s window before the boundary
        win = Locs[w0:b]
        towers = win[(win >= 1) & (win <= 9)]
        if len(towers):
            node = int(np.bincount(towers).argmax())
        else:                               # fall back to nearest known tower
            p = XY[min(b, T - 1)]
            node = min(node_xy, key=lambda n: np.hypot(*(node_xy[n] - p))) if node_xy else None
        if node in node_xy:
            rewards.append((b - start, node, node_xy[node]))
    return dict(xy=xy, fr=fr, locs=locs, n_frames=end - start, rewards=rewards)


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


def _prep_playback(trial, neuron, fps, speedup, max_frames=450):
    """Resample trajectory + neuron firing to M playback frames; return everything
    the per-frame renderer needs (positions, fr_norm, reward/shockwave frames)."""
    n = trial["n_frames"]
    duration = n / SAMPLE_HZ
    m = int(np.clip(round(duration / max(speedup, 1e-3) * fps), 24, max_frames))

    xy = _resample(trial["xy"], m, axis=0)
    fr_raw = gaussian_filter1d(trial["fr"][neuron].astype(float), 2.0)
    fr = _resample(_norm01(fr_raw), m)
    fr = np.clip(fr, 0, 1)

    # reward bursts: boundary frame mapped into playback time
    rewards = [(int(round(rel / max(n - 1, 1) * (m - 1))), node, xyp)
               for (rel, node, xyp) in trial["rewards"]]

    # shockwaves: spawn where fr crosses up through a high threshold ("beat drop")
    thr = 0.72
    crossings = np.where((fr[1:] >= thr) & (fr[:-1] < thr))[0] + 1
    shock = [(int(c), xy[int(c)]) for c in crossings]

    return dict(xy=xy, fr=fr, m=m, fps=fps, rewards=rewards, shock=shock)


def _ring_alpha(age, life):
    return max(0.0, 1.0 - age / life)


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
    cmap = _trail_cmap(palette)
    bg = pal["bg"]
    tr = get_trial(session, trial)
    pb = _prep_playback(tr, neuron, fps, speedup)
    xy, fr, m = pb["xy"], pb["fr"], pb["m"]

    node_xy, edges = session["node_xy"], session["edges"]
    # padded equal-aspect limits around the whole trajectory + towers
    pts = [xy] + ([np.array(list(node_xy.values()))] if node_xy else [])
    allp = np.vstack(pts)
    pad = 0.10 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 1.0)
    xlim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ylim = (allp[:, 1].min() - pad, allp[:, 1].max() + pad)

    fig = plt.figure(figsize=(7 * figscale, 8.4 * figscale), facecolor=bg)
    gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.06,
                  left=0.02, right=0.98, top=0.97, bottom=0.07)
    ax = fig.add_subplot(gs[0]); axf = fig.add_subplot(gs[1])

    fr_line_x = np.arange(m)
    SHOCK_LIFE, BURST_LIFE = 22, 30
    segs = [np.array([a, b]) for (a, b) in edges]              # for LineCollection
    node_pts = np.array(list(node_xy.values())) if node_xy else np.empty((0, 2))
    rew_pts = np.array([p for (_, _, p) in pb["rewards"]]) if pb["rewards"] \
        else np.empty((0, 2))

    def draw_maze():
        # static maze redrawn each frame (axis is cleared) but vectorised:
        # one LineCollection per glow layer, one scatter for all nodes.
        for lw, al in [(11, 0.05), (6, 0.10), (3, 0.22), (1.4, 0.85)]:
            ax.add_collection(LineCollection(segs, colors=pal["maze_line"], linewidths=lw,
                                             alpha=al, capstyle="round", zorder=1))
        if len(node_pts):
            ax.scatter(node_pts[:, 0], node_pts[:, 1], s=60, color=pal["maze_line"],
                       alpha=0.35, edgecolors="none", zorder=2)

    def update(i):
        ax.clear(); axf.clear()
        glow = fr[i]
        # whole-scene strobe: maze bg lightens subtly with firing
        base = np.array(to_rgb(bg))
        tint = np.array(to_rgb(pal["maze_line"]))
        ax.set_facecolor(tuple(np.clip(base + 0.07 * glow * (tint - base), 0, 1)))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.axis("off")

        draw_maze()

        # reward towers: shared pulsing glow (one scatter per layer for all towers)
        if len(rew_pts):
            puls = 0.45 + 0.25 * np.sin(i * 0.25)
            for s, al in [(900, 0.05 * puls), (430, 0.12 * puls), (190, 0.5 * puls)]:
                ax.scatter(rew_pts[:, 0], rew_pts[:, 1], s=s, color=pal["reward"],
                           alpha=al, edgecolors="none", zorder=2)
            ax.scatter(rew_pts[:, 0], rew_pts[:, 1], s=70, marker="*",
                       color="white", alpha=0.9, zorder=3)
            # expanding burst when each tower is reached
            for (rf, _, p) in pb["rewards"]:
                age = i - rf
                if 0 <= age < BURST_LIFE:
                    a = _ring_alpha(age, BURST_LIFE)
                    ax.scatter(*p, s=300 + age * 320, facecolors="none",
                               edgecolors=pal["reward"], linewidths=2.5 * a,
                               alpha=a, zorder=4)

        # comet trail with bloom (newest brightest), colour ramps over time
        k0 = max(0, i - trail_len)
        seg = xy[k0:i + 1]
        if len(seg) > 1:
            ages = np.linspace(0, 1, len(seg))            # 0 oldest → 1 newest
            cols = cmap((k0 + np.arange(len(seg))) / max(m - 1, 1))
            cols[:, 3] = ages ** 2
            for size, am in [(26, 0.10), (12, 0.22), (5, 0.95)]:
                c = cols.copy(); c[:, 3] = cols[:, 3] * am
                ax.scatter(seg[:, 0], seg[:, 1], s=size, c=c,
                           edgecolors="none", zorder=5)

        # shockwave rings on beat-drop frames
        for (sf, p) in pb["shock"]:
            age = i - sf
            if 0 <= age < SHOCK_LIFE:
                a = _ring_alpha(age, SHOCK_LIFE)
                ax.scatter(*p, s=120 + age * 260, facecolors="none",
                           edgecolors=pal["comet"], linewidths=2.2 * a,
                           alpha=0.8 * a, zorder=6)

        # comet head: size + brightness pulse with firing rate
        head = xy[i]
        hs = 70 + 520 * glow
        for s, al in [(hs * 4, 0.10 + 0.20 * glow), (hs * 2, 0.25),
                      (hs, 0.95)]:
            ax.scatter(*head, s=s, color=pal["comet"], alpha=al,
                       edgecolors="none", zorder=7)

        ax.text(0.015, 0.975, f"{session['label']}  •  neuron {neuron}  •  trial {trial}",
                transform=ax.transAxes, color="white", alpha=0.55, fontsize=8,
                va="top", family="monospace")

        # firing-rate panel with scrolling playhead
        axf.set_facecolor(bg)
        axf.plot(fr_line_x, fr, color=pal["maze_line"], lw=1.0, alpha=0.4)
        axf.fill_between(fr_line_x[:i + 1], fr[:i + 1], color=pal["comet"], alpha=0.18)
        axf.plot(fr_line_x[:i + 1], fr[:i + 1], color=pal["comet"], lw=1.4, alpha=0.9)
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
    allp = np.vstack([xy] + ([np.array(list(session["node_xy"].values()))]
                             if session["node_xy"] else []))
    pad = 0.10 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 1.0)
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.axis("off")

    for (a, b) in session["edges"]:
        for lw, al in [(9, 0.05), (4, 0.12), (1.4, 0.5)]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=pal["maze_line"], lw=lw,
                    alpha=al, solid_capstyle="round")

    # additive low-alpha blend of every point, brightness ∝ firing rate
    cols = cmap(np.linspace(0, 1, m))
    cols[:, 3] = 0.06 + 0.5 * fr
    for size, am in [(60, 0.25), (24, 0.5), (8, 1.0)]:
        c = cols.copy(); c[:, 3] = cols[:, 3] * am
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c=c, edgecolors="none")

    for (_, node, p) in pb["rewards"]:
        for s, al in [(1400, 0.06), (650, 0.14), (260, 0.5)]:
            ax.scatter(*p, s=s, color=pal["reward"], alpha=al, edgecolors="none")
        ax.scatter(*p, s=90, marker="*", color="white", alpha=0.95)

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
