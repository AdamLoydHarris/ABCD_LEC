"""
Maze geometry from nose pokes (Stage B2 of the mEC port): node centroids in camera pixels.

At the moment a mouse pokes port n it is, by definition, at node n. So the tracked
`head_back` position at every poke entry gives, per session and port, a sample of that
node's pixel position; the median over pokes and then over sessions is the node centroid,
with no corners, masks or homography. This is the method of Mohamady's
`Preprocessing_Social/S6_Tracking_calibration.ipynb`, restated for this repo's inputs:

    poke entries      data/processed_data/trialtimes_raw_mingyutest/pokes_{recday}_{s}.npy
                      (entry_bin on the 25 ms grid from the first A_on -- extract_pokes.py)
    frame of first A_on   pinstate first rising edge + (A_on - rsync[0]) x 60 fps, as
                      sleap_alignment_copilot does it
    head_back pixels  the raw SLEAP h5 (data/raw/sleap/{mouse}/maze/)
    mask labels       data/preprocessed/sleap/maze/{mouse}/*_ROIs.csv (sleap_to_rois.py),
                      the thing this is checked against

One calibration per camera epoch (bp01 has two: masks early/late), taken from
`sleap_to_rois.geometry`. Node radius = 0.292 x median nearest-neighbour pitch (the
social notebook's value, reverse-engineered from hand-drawn ROI areas); bridges at the
midpoints of the twelve grid edges, named `edge_a-b` in the order `build_data_dic.MAZE_EDGES`
uses, so the labels are drop-in for `Locs_raw`.

Checks written per epoch (`checks.json`), all measured, none assumed:
1. closed loop -- fraction of calibration pokes whose own entry frame assigns to the poked port
2. occupancy -- node / bridge / none fractions over every frame (social rig: ~76 / 24)
3. adjacency -- fraction of consecutive distinct node labels that are grid neighbours
4. agreement with the mask-based head_back label, frame-wise, where both are labelled
5. distance from each poke-derived centroid to the mask centroid of the same node

Usage::

    python poke_calibration.py --dataset mec --mouse bp04 [--write]
    python poke_calibration.py --dataset mec                 # every mouse, every epoch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
import sleap_alignment_copilot as cp                                  # noqa: E402
import sleap_to_rois as s2r                                           # noqa: E402
from build_data_dic import MAZE_EDGES                                 # noqa: E402

POKES_PATH = rr.REPO_ROOT / "data" / "processed_data" / "trialtimes_raw_mingyutest"
OUT_ROOT = rr.REPO_ROOT / "data" / "processed_data" / "poke_calibration"

MIN_POKES_PER_PORT = 20        # per session, as in the social notebook
NODE_RADIUS_FRACTION = 0.292   # of the median nearest-neighbour node pitch
BRIDGE_MAX_FRACTION = 1.5      # x node radius: beyond this from every bridge midpoint = off maze
FRAMES_PER_BIN = 25 * cp.VIDEO_FPS / 1000.0   # 1.5 frames per 25 ms bin at 60 fps
GRID = {n: ((n - 1) // 3, (n - 1) % 3) for n in range(1, 10)}       # port -> (row, col), row-major


# ---------------------------------------------------------------------------
# per-session inputs
# ---------------------------------------------------------------------------

def head_back_pixels(h5: Path) -> np.ndarray:
    with h5py.File(h5, "r") as f:
        nodes = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
        i = nodes.index("head_back")
        tr = f["tracks"][0, :, i, :]          # (2, T)
    return tr.T                                # (T, 2)


def frame_clock(mouse: str, recday: str, s: int, row: pd.Series) -> tuple[dict | None, str]:
    """Per-session frame clock `frame = a * t_ms + b` from the whole pinstate train against
    this mouse's pyControl rsyncs (`sleap_alignment_copilot.fit_frame_clock`). The first-edge
    + nominal-fps rule is not used here: the maze-2 camera runs at 59.948 fps and drifts ~1 s
    per session under it. Returns (clock dict with 'first_A_on_ms' added, pinstate name)."""
    d = str(row["_date"])
    stamp = str(int(float(row["Tracking"]))).zfill(6)
    dt = datetime.strptime(f"{d} {stamp}", "%Y%m%d %H%M%S")
    pin = cp.find_closest_pinstate_file(mouse, dt, max_diff_minutes=5)
    if pin is None:
        return None, "no pinstate"
    pinstate = cp.load_pinstate(pin[0])
    pc = cp.parse_pycontrol_file(rr.pycontrol_file(recday, s))
    a_on = cp.find_first_A_on_timestamp(pc)
    if a_on is None:
        return None, "no A_on"
    try:
        clock = cp.fit_frame_clock(cp.rising_edges(pinstate), cp.all_rsync_timestamps(pc))
    except ValueError as exc:
        return None, f"clock fit failed: {exc}"
    clock["first_A_on_ms"] = float(a_on)
    return clock, pin[0].name


def session_inputs(mouse: str, recday: str, s: int, verbose: bool) -> dict | None:
    row = rr.task_session_rows(recday).iloc[s]
    pokes_file = POKES_PATH / f"pokes_{recday}_{s}.npy"
    if not pokes_file.exists():
        return None
    pokes = np.load(pokes_file)
    if len(pokes) == 0:
        return None
    d = str(row["_date"])
    stamp = str(int(float(row["Tracking"]))).zfill(6)
    day = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    h5s = sorted((rr.REPO_ROOT / "data" / "raw" / "sleap" / mouse / "maze").glob(f"*_{mouse}_{day}-{stamp}.analysis.h5"))
    if not h5s:
        near = cp.find_closest_sleap_datetime(mouse, f"{day}-{stamp}", 2)
        h5s = sorted((rr.REPO_ROOT / "data" / "raw" / "sleap" / mouse / "maze").glob(f"*_{mouse}_{near}.analysis.h5")) if near else []
    if not h5s:
        return None
    stem = re.search(rf"({mouse}_\d{{4}}-\d{{2}}-\d{{2}}-\d{{6}})\.analysis\.h5$", h5s[0].name).group(1)
    rois_csv = cp.SLEAP_PATH / mouse / f"{stem}.analysis_ROIs.csv"
    clock, note = frame_clock(mouse, recday, s, row)
    if clock is None:
        if verbose:
            print(f"  {recday} s{s}: skipped ({note})")
        return None
    xy = head_back_pixels(h5s[0])
    # poke entry time on the pyControl clock, then its frame through the fitted clock
    entry_ms = clock["first_A_on_ms"] + pokes[:, 0] * 25.0
    entry_frames = np.rint(cp.frames_at(clock, entry_ms)).astype(int)
    ok = (entry_frames >= 0) & (entry_frames < len(xy))
    mask_labels = pd.read_csv(rois_csv)["head_back"].to_numpy(dtype=object) if rois_csv.exists() else None
    return dict(recday=recday, session=s, date=date(int(d[:4]), int(d[4:6]), int(d[6:])), h5=h5s[0].name,
                xy=xy, fps=clock["fps"], ports=pokes[ok, 2], entry_frames=entry_frames[ok],
                n_pokes_dropped=int((~ok).sum()), mask_labels=mask_labels)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def node_centroids(sessions: list[dict]) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    rows = []
    for sess in sessions:
        pts = sess["xy"][sess["entry_frames"]]
        for port in range(1, 10):
            sel = (sess["ports"] == port) & ~np.isnan(pts).any(axis=1)
            if sel.sum() >= MIN_POKES_PER_PORT:
                med = np.median(pts[sel], axis=0)
                rows.append(dict(recday=sess["recday"], session=sess["session"], port=port, n=int(sel.sum()),
                                 x=float(med[0]), y=float(med[1]),
                                 scatter_px=float(np.median(np.linalg.norm(pts[sel] - med, axis=1)))))
    table = pd.DataFrame(rows)
    centroids = {int(p): g[["x", "y"]].median().to_numpy() for p, g in table.groupby("port")}
    return centroids, table


def build_geometry(centroids: dict[int, np.ndarray]) -> dict:
    nodes = {n: centroids[n] for n in range(1, 10) if n in centroids}
    pts = np.array([nodes[n] for n in sorted(nodes)])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    pitch = float(np.median(d.min(axis=1)))
    radius = NODE_RADIUS_FRACTION * pitch
    bridges = {}
    for edge in MAZE_EDGES:
        a, b = (int(x) for x in edge.split("-"))
        if a in nodes and b in nodes:
            bridges[edge] = (nodes[a] + nodes[b]) / 2
    return dict(nodes=nodes, bridges=bridges, pitch_px=pitch, node_radius_px=radius,
                missing_nodes=[n for n in range(1, 10) if n not in nodes])


def assign(xy: np.ndarray, geom: dict) -> np.ndarray:
    """Per-frame label: 'node_k' within the node radius, else nearest 'edge_a-b' within
    BRIDGE_MAX_FRACTION x radius, else 'None' (off maze or untracked)."""
    labels = np.full(len(xy), "None", dtype=object)
    ok = ~np.isnan(xy).any(axis=1)
    if not ok.any():
        return labels
    node_ids = sorted(geom["nodes"])
    node_pts = np.array([geom["nodes"][n] for n in node_ids])
    dn = np.linalg.norm(xy[ok][:, None] - node_pts[None], axis=2)
    nearest = dn.argmin(axis=1)
    in_node = dn[np.arange(len(nearest)), nearest] <= geom["node_radius_px"]
    out = np.full(ok.sum(), "None", dtype=object)
    out[in_node] = [f"node_{node_ids[i]}" for i in nearest[in_node]]
    if geom["bridges"]:
        edge_ids = list(geom["bridges"])
        edge_pts = np.array([geom["bridges"][e] for e in edge_ids])
        de = np.linalg.norm(xy[ok][:, None] - edge_pts[None], axis=2)
        near_e = de.argmin(axis=1)
        on_bridge = (~in_node) & (de[np.arange(len(near_e)), near_e] <= BRIDGE_MAX_FRACTION * geom["node_radius_px"])
        out[on_bridge] = [f"edge_{edge_ids[i]}" for i in near_e[on_bridge]]
    labels[ok] = out
    return labels


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def mask_centroids(mask_dir: Path) -> dict[int, np.ndarray]:
    out = {}
    for n in range(1, 10):
        f = mask_dir / f"node_{n}.tif"
        if f.exists():
            ys, xs = np.nonzero(tifffile.imread(f))
            out[n] = np.array([xs.mean(), ys.mean()])
    return out


def run_checks(sessions: list[dict], geom: dict, mask_dir: Path) -> dict:
    closed_hits = closed_total = 0
    occ = {"node": 0, "edge": 0, "None": 0}
    adj_ok = adj_total = 0
    agree = both = node_agree = node_both = 0
    per_session = []
    grid_edges = {tuple(sorted(int(x) for x in e.split("-"))) for e in MAZE_EDGES}
    poke_dist = []
    for sess in sessions:
        lab = assign(sess["xy"], geom)
        # 1. closed loop, plus the distance of each poke's head_back from its port's centroid
        at_pokes = lab[sess["entry_frames"]]
        want = np.array([f"node_{p}" for p in sess["ports"]], dtype=object)
        closed_hits += int((at_pokes == want).sum())
        closed_total += len(want)
        pts = sess["xy"][sess["entry_frames"]]
        for p in np.unique(sess["ports"]):
            if int(p) in geom["nodes"]:
                sel = sess["ports"] == p
                poke_dist.extend(np.linalg.norm(pts[sel] - geom["nodes"][int(p)], axis=1).tolist())
        # 2. occupancy
        kinds = np.where(np.char.startswith(lab.astype(str), "node"), "node",
                         np.where(np.char.startswith(lab.astype(str), "edge"), "edge", "None"))
        for k in occ:
            occ[k] += int((kinds == k).sum())
        # 3. adjacency of consecutive distinct nodes
        nodes_seq = [int(l[5:]) for l in lab if str(l).startswith("node")]
        seq = [n for i, n in enumerate(nodes_seq) if i == 0 or n != nodes_seq[i - 1]]
        for a, b in zip(seq[:-1], seq[1:]):
            adj_total += 1
            adj_ok += int(tuple(sorted((a, b))) in grid_edges)
        # 4. agreement with masks
        sess_agree = np.nan
        if sess["mask_labels"] is not None and len(sess["mask_labels"]) == len(lab):
            m = sess["mask_labels"].astype(str)
            l = lab.astype(str)
            b = (m != "None") & (l != "None")
            both += int(b.sum())
            agree += int((m[b] == l[b]).sum())
            nb = b & np.char.startswith(m, "node") & np.char.startswith(l, "node")
            node_both += int(nb.sum())
            node_agree += int((m[nb] == l[nb]).sum())
            sess_agree = float((m[b] == l[b]).mean()) if b.any() else np.nan
        per_session.append(dict(recday=sess["recday"], session=sess["session"], n_pokes=len(want),
                                closed_loop=float((at_pokes == want).mean()) if len(want) else np.nan,
                                mask_agreement=sess_agree, none_frac=float((kinds == "None").mean())))
    total = sum(occ.values()) or 1
    mc = mask_centroids(mask_dir)
    dist = {n: float(np.linalg.norm(geom["nodes"][n] - mc[n])) for n in geom["nodes"] if n in mc}
    pd_arr = np.array(poke_dist)
    pd_arr = pd_arr[~np.isnan(pd_arr)]
    poke_q = {f"p{q}": float(np.percentile(pd_arr, q)) for q in (50, 75, 90, 95)} if len(pd_arr) else {}
    return dict(closed_loop=closed_hits / max(closed_total, 1), n_calibration_pokes=closed_total,
                poke_to_centroid_px=poke_q, poke_nan_frac=float(1 - len(pd_arr) / max(len(poke_dist), 1)),
                occupancy={k: v / total for k, v in occ.items()}, n_frames=total,
                adjacency_ok=adj_ok / max(adj_total, 1), n_transitions=adj_total,
                mask_agreement=agree / max(both, 1), n_frames_both_labelled=both,
                mask_agreement_nodes_only=node_agree / max(node_both, 1),
                centroid_vs_mask_px=dist, centroid_vs_mask_px_median=float(np.median(list(dist.values()))) if dist else None,
                per_session=per_session)


def plot(sessions, geom, table, mask_dir, out: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.6))
    cmap = plt.get_cmap("tab10")
    for sess in sessions:
        pts = sess["xy"][sess["entry_frames"]]
        for port in range(1, 10):
            sel = sess["ports"] == port
            ax.scatter(pts[sel, 0], pts[sel, 1], s=3, color=cmap(port - 1), alpha=0.25, linewidths=0)
    for n, c in geom["nodes"].items():
        ax.add_patch(plt.Circle(c, geom["node_radius_px"], fill=False, color=cmap(n - 1)))
        ax.text(*c, str(n), ha="center", va="center", fontsize=9, weight="bold")
    for n, c in mask_centroids(mask_dir).items():
        ax.plot(*c, "kx", ms=7)
    for e, m in geom["bridges"].items():
        ax.plot(*m, ".", color="0.5", ms=4)
    ax.set_xlim(0, 1280); ax.set_ylim(1024, 0); ax.set_aspect("equal")
    ax.set_title(f"{out.parent.name}: poke-derived nodes (circles) vs mask centroids (x); "
                 f"pitch {geom['pitch_px']:.0f} px, radius {geom['node_radius_px']:.0f} px", fontsize=8)
    fig.savefig(out, dpi=110, bbox_inches=None)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def epochs_for(mouse: str) -> dict[str, list[tuple[str, int]]]:
    """{mask-set name: [(recday, session), ...]} over merged recdays of this mouse."""
    out: dict[str, list] = {}
    for recday in rr.all_recdays("merged"):
        if not recday.startswith(mouse):
            continue
        rows = rr.task_session_rows(recday)
        for s, row in rows.iterrows():
            d = str(row["_date"])
            mask_name, _, _ = s2r.geometry(mouse, date(int(d[:4]), int(d[4:6]), int(d[6:])))
            out.setdefault(mask_name, []).append((recday, s))
    return out


def write_json(path: Path, obj) -> None:
    if path.exists():
        n = 1
        while (alt := path.with_name(f"{path.stem}.v{n}{path.suffix}")).exists():
            n += 1
        path = alt
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"  -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--mouse", nargs="*", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    verbose = not args.quiet

    summary = []
    for mouse in args.mouse or rr.EPHYS_MICE:
        for mask_name, pairs in epochs_for(mouse).items():
            sessions = [x for x in (session_inputs(mouse, rd, s, verbose) for rd, s in pairs) if x]
            if not sessions:
                print(f"{mouse}/{mask_name}: no usable sessions")
                continue
            centroids, table = node_centroids(sessions)
            if len(centroids) < 4:
                print(f"{mouse}/{mask_name}: only {len(centroids)} ports with >= {MIN_POKES_PER_PORT} pokes; skipped")
                continue
            geom = build_geometry(centroids)
            mask_dir = s2r.MASKS_ROOT / mask_name
            checks = run_checks(sessions, geom, mask_dir)
            line = (f"{mouse}/{mask_name}: {len(sessions)} sessions, {checks['n_calibration_pokes']} pokes, "
                    f"nodes {sorted(geom['nodes'])} (missing {geom['missing_nodes']}), pitch {geom['pitch_px']:.1f} px, "
                    f"radius {geom['node_radius_px']:.1f} px | closed loop {checks['closed_loop']:.3f} | "
                    f"occupancy node {checks['occupancy']['node']:.2f} edge {checks['occupancy']['edge']:.2f} "
                    f"none {checks['occupancy']['None']:.2f} | adjacency {checks['adjacency_ok']:.3f} "
                    f"(n={checks['n_transitions']}) | mask agreement {checks['mask_agreement']:.3f} "
                    f"(nodes only {checks['mask_agreement_nodes_only']:.3f}, n={checks['n_frames_both_labelled']}) | "
                    f"centroid-vs-mask median {checks['centroid_vs_mask_px_median']} px")
            print(line)
            if verbose:
                print("   per node centroid-vs-mask px:", {k: round(v, 1) for k, v in checks["centroid_vs_mask_px"].items()})
                print("   per node scatter px (median over sessions):",
                      table.groupby("port")["scatter_px"].median().round(1).to_dict())
                print("   poke head_back distance to its port centroid, px:",
                      {k: round(v, 1) for k, v in checks["poke_to_centroid_px"].items()},
                      f"| NaN at poke frame {checks['poke_nan_frac']:.3f}")
                ps = pd.DataFrame(checks["per_session"])
                print("   per session closed loop / mask agreement / none:")
                print("   " + ps.round(3).to_string(index=False).replace("\n", "\n   "))
            summary.append(dict(mouse=mouse, epoch=mask_name, **{k: v for k, v in checks.items() if k not in ("per_session", "centroid_vs_mask_px", "occupancy")},
                                **{f"occ_{k}": v for k, v in checks["occupancy"].items()}, pitch_px=geom["pitch_px"], radius_px=geom["node_radius_px"]))
            if args.write:
                out = OUT_ROOT / mouse / mask_name
                out.mkdir(parents=True, exist_ok=True)
                write_json(out / "maze_nodes.json", dict(mouse=mouse, epoch=mask_name, nodes={str(k): v for k, v in geom["nodes"].items()},
                                                          bridges={k: v for k, v in geom["bridges"].items()}, pitch_px=geom["pitch_px"],
                                                          node_radius_px=geom["node_radius_px"], bridge_max_px=BRIDGE_MAX_FRACTION * geom["node_radius_px"],
                                                          missing_nodes=geom["missing_nodes"], sessions=[(x["recday"], x["session"], x["h5"]) for x in sessions],
                                                          min_pokes_per_port=MIN_POKES_PER_PORT, node_radius_fraction=NODE_RADIUS_FRACTION,
                                                          source="poke_calibration.py, method of Preprocessing_Social/S6_Tracking_calibration.ipynb"))
                write_json(out / "checks.json", checks)
                p = out / "per_session_ports.csv"
                if not p.exists():
                    table.to_csv(p, index=False)
                plot(sessions, geom, table, mask_dir, out / "node_map.png")
    if summary:
        pd.set_option("display.width", 250)
        print("\n" + pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
