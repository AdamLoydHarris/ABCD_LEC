"""Goal-progress sequence heat maps across tasks, split by brain region.

The classic figure — neurons x normalised goal-progress bins, rank-ordered by
peak, that ordering then carried to other tasks — but with the population split
by the anatomy from ``code/histology_refit/``.  The question it answers: does
the goal-progress code hold its phase across a task switch, and does that differ
between ENTl-deep, SUB/ProS, ENTm and CA1?

Prior art is ``LEC_sploratory_analysis.ipynb`` cells 84-85 (6 sessions, all mice
pooled, no region split).  Two things are different here.

**The repeated-task confound.**  In 19 of 25 recdays *session 3 re-runs session
0's task*.  A naive ``sessions[:4]`` therefore puts the same task in panels 1
and 4, which produces a strong diagonal in panel 4 for a completely trivial
reason.  Rather than dropping those sessions, panel 4 is **labelled** as the
within-task repeat and used as a **ceiling**: it says how much order survives
when nothing remapped, which is the scale panels 2-3 must be read against.
``panel_sessions`` matches tasks **by value**, never by position.

**Panel 1 is circular, by construction.**  The sort order is defined by panel
1's argmax and then applied to panel 1 itself, so its diagonal is guaranteed
even for pure noise.  That is deliberate (it matches the existing figure), and
``run_synthetic_controls`` plants a pure-noise population to demonstrate it:
noise yields a clean panel-1 diagonal and nothing anywhere else.  **Only panels
2-4 carry evidence.**

Everything upstream already exists: ``Smoothed_norm`` is (n, 360) and already
circularly smoothed (``smooth_circ``, sigma=10 with wrap-around, in
``preprocessing/build_data_dic.py``), 360 = 4 states x 90 bins.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
OUT_DIR = REPO_ROOT / "data" / "processed_data"
FIGURE_DIR = REPO_ROOT / "data" / "figures" / "gp_region_heatmaps"
CURVES_PATH = OUT_DIR / "gp_curves.pkl"

N_STATES = 4
N_BINS = 90                      # normalised bins per state; 4 x 90 = 360
PANEL_LABELS = ("Task A\n(sort defined here)", "Task B\n(novel)",
                "Task C\n(novel)", "Task A again\n(within-task ceiling)")

#: Region rows, in plotting order.  Mirrors region_assignment.REGION_GROUPS.
REGION_ORDER = ("ENTl-sup", "ENTl-deep", "ENTm", "SUB/ProS", "CA1/HPF",
                "fibre/other")

VECTOR_FORMATS = ("pdf", "svg")
RASTER_FORMATS = ("png",)


def _save(fig, stem: str, *, dpi: int = 200) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in VECTOR_FORMATS + RASTER_FORMATS:
        p = FIGURE_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        paths.append(p)
        print(f"saved {p}")
    return paths


# --------------------------------------------------------------------------
# panel selection
# --------------------------------------------------------------------------

def load_tasks() -> dict:
    with open(OUT_DIR / "tasks_dic.pkl", "rb") as f:
        return pickle.load(f)


def panel_sessions(recday: str, tasks: dict | None = None,
                   available: set | None = None) -> list[int] | None:
    """``[A, novel B, novel C, A-repeat]`` session indices, or None.

    Tasks are compared **by their goal sequence**, never by session position --
    positional pairing is exactly what produced the repeated-task confound.
    Returns None when the recday never re-runs its first task, so there is no
    ceiling panel to be had.

    ``available`` restricts the choice to sessions that actually have curves.
    Two recdays (ah08_20250620_20250623, ly05_20250618_20250619) have a session
    3 in ``tasks_dic`` with no ``Smoothed_norm`` behind it, and it is precisely
    their repeat session -- without this filter they are silently dropped
    despite re-running task A later on.
    """
    tasks = tasks if tasks is not None else load_tasks()
    if recday not in tasks:
        return None
    seq = {s: tuple(int(x) for x in tasks[recday][s]) for s in sorted(tasks[recday])}
    order = [s for s in sorted(seq) if available is None or s in available]
    if not order:
        return None
    first = order[0]
    target = seq[first]

    repeat = next((s for s in order[1:] if seq[s] == target), None)
    novel: list[int] = []
    for s in order[1:]:
        if seq[s] != target and all(seq[s] != seq[n] for n in novel):
            novel.append(s)
        if len(novel) == 2:
            break
    if repeat is None or len(novel) != 2:
        return None
    return [first, novel[0], novel[1], repeat]


def _available(recday: str, curves: dict | None) -> set | None:
    return set(curves[recday]) if curves and recday in curves else None


def qualifying_recdays(tasks: dict | None = None,
                       curves: dict | None = None) -> list[str]:
    tasks = tasks if tasks is not None else load_tasks()
    return [rd for rd in sorted(tasks)
            if panel_sessions(rd, tasks, _available(rd, curves)) is not None]


# --------------------------------------------------------------------------
# the (n, 90) goal-progress curves
# --------------------------------------------------------------------------

def build_gp_curves(rebuild: bool = False, verbose: bool = True) -> dict:
    """Extract ``Smoothed_norm`` -> (n, 90) per (recday, session), cached.

    ``data_dic_lec.pkl`` is ~3.8 GB and this needs only a few MB of it, so the
    curves are cached and every later run is instant.
    """
    if CURVES_PATH.exists() and not rebuild:
        return load_gp_curves()

    import sys
    sys.path.insert(0, str(CODE_DIR))
    import glm_analysis_v2 as glm

    data_dic = glm.load_data_dic(verbose=verbose)
    out: dict[str, dict[int, np.ndarray]] = {}
    for recday in sorted(data_dic):
        sessions = {}
        for sess, sd in data_dic[recday].items():
            sm = sd.get("Smoothed_norm") if isinstance(sd, dict) else None
            if sm is None:
                continue
            sm = np.asarray(sm, float)
            if sm.ndim != 2 or sm.shape[1] != N_STATES * N_BINS:
                raise ValueError(
                    f"{recday} s{sess}: Smoothed_norm is {sm.shape}, "
                    f"expected (n, {N_STATES * N_BINS})")
            # mean over the four states -> goal progress within a leg
            sessions[int(sess)] = sm.reshape(len(sm), N_STATES, N_BINS) \
                                    .mean(axis=1).astype(np.float32)
        if sessions:
            out[recday] = sessions
        if verbose:
            print(f"  {recday}: {len(sessions)} sessions", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CURVES_PATH, "wb") as f:
        pickle.dump(out, f)
    if verbose:
        print(f"saved {CURVES_PATH} ({len(out)} recdays)")
    return out


def load_gp_curves() -> dict:
    with open(CURVES_PATH, "rb") as f:
        return pickle.load(f)


def load_regions() -> dict:
    with open(OUT_DIR / "unit_regions.pkl", "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

def stack_panels(curves: dict | None = None, regions: dict | None = None,
                 tasks: dict | None = None, *, verbose: bool = True):
    """Pool neurons across qualifying recdays.

    Returns ``(panels, meta)`` where ``panels`` is a (4, N, 90) array and
    ``meta`` is an N-row frame with ``mouse``, ``recday``, ``group``.  Row *i*
    of every panel is the same neuron.
    """
    curves = curves if curves is not None else load_gp_curves()
    regions = regions if regions is not None else load_regions()
    tasks = tasks if tasks is not None else load_tasks()

    blocks, meta_rows, used = [], [], []
    for recday in sorted(curves):
        sess = panel_sessions(recday, tasks, _available(recday, curves))
        if sess is None or recday not in regions:
            continue
        if any(s not in curves[recday] for s in sess):
            continue
        reg = regions[recday]
        mats = [curves[recday][s] for s in sess]
        n = mats[0].shape[0]
        if any(m.shape[0] != n for m in mats):
            raise ValueError(f"{recday}: sessions disagree on neuron count")
        if n != len(reg):
            raise ValueError(
                f"{recday}: {n} neurons in curves but {len(reg)} region rows -- "
                f"the positional join is broken")
        blocks.append(np.stack(mats))                    # (4, n, 90)
        meta_rows.append(reg[["mouse", "group"]].assign(recday=recday))
        used.append(recday)

    if not blocks:
        raise RuntimeError("no qualifying recdays with curves")
    panels = np.concatenate(blocks, axis=1)
    meta = pd.concat(meta_rows, ignore_index=True)
    if verbose:
        print(f"{len(used)} recdays, {panels.shape[1]} unit-recordings")
    panels.flags.writeable = False
    return panels, meta


# --------------------------------------------------------------------------
# normalisation, sorting, and the diagonality statistic
# --------------------------------------------------------------------------

def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    """z-score each neuron across its 90 bins; flat rows become zeros."""
    mat = np.asarray(mat, float)
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    out = np.divide(mat - mu, sd, out=np.zeros_like(mat), where=sd > 0)
    return np.nan_to_num(out)


def sorted_panels(panels: np.ndarray, rows: np.ndarray | None = None):
    """z-score every panel, sort by panel 1's peak, return (4, n, 90) + order.

    The sort is defined on panel 1 and applied to panel 1 as well, so panel 1's
    diagonal is circular -- see the module docstring.
    """
    sel = panels if rows is None else panels[:, rows, :]
    z = np.stack([_zscore_rows(sel[k]) for k in range(sel.shape[0])])
    order = np.argsort(np.argmax(z[0], axis=1), kind="stable")
    return z[:, order, :], order


def _circ_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Jammalamadaka circular-circular correlation between two angle sets."""
    if len(a) < 3:
        return np.nan
    am = np.angle(np.mean(np.exp(1j * a)))
    bm = np.angle(np.mean(np.exp(1j * b)))
    sa, sb = np.sin(a - am), np.sin(b - bm)
    den = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    return float(np.sum(sa * sb) / den) if den > 0 else np.nan


def diagonality(panels: np.ndarray, rows: np.ndarray | None = None, *,
                n_shuffle: int = 1000, seed: int = 0) -> pd.DataFrame:
    """How well each panel's peak ordering matches panel 1's.

    Circular correlation between per-neuron peak bins in panel 1 and panel k,
    against a null that permutes neurons within the same set.  Panel 1 scores
    1.0 by construction; panel 4 (the within-task repeat) is the ceiling that
    panels 2-3 are read against.
    """
    sel = panels if rows is None else panels[:, rows, :]
    z = np.stack([_zscore_rows(sel[k]) for k in range(sel.shape[0])])
    peaks = np.argmax(z, axis=2) * 2 * np.pi / z.shape[2]      # (4, n)
    rng = np.random.default_rng(seed)

    out = []
    for k in range(z.shape[0]):
        r = _circ_corr(peaks[0], peaks[k])
        null = np.array([_circ_corr(peaks[0], rng.permutation(peaks[k]))
                         for _ in range(n_shuffle)])
        null = null[np.isfinite(null)]
        p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)) \
            if len(null) and np.isfinite(r) else np.nan
        zsc = float((r - null.mean()) / null.std()) if len(null) and null.std() > 0 else np.nan
        out.append({"panel": k + 1, "label": PANEL_LABELS[k].replace("\n", " "),
                    "n": int(z.shape[1]), "circ_r": r, "z_vs_null": zsc, "p": p})
    return pd.DataFrame(out)


#: A ceiling this weak means the region does not reproduce its own ordering
#: when the SAME task is re-run, so the novel-task panels have nothing to be
#: read against and the ratio is withheld.
MIN_CEILING_R = 0.25


def summary(panels: np.ndarray, meta: pd.DataFrame, *,
            regions=REGION_ORDER, include_all: bool = True,
            n_shuffle: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per region: novel-task diagonality, the within-task ceiling, and the ratio.

    ``retention`` is mean(novel r) / ceiling r -- how much of the order that
    survives a *repeat* also survives a *task switch*.  It is withheld when the
    ceiling itself is at floor, because a region that cannot reproduce its own
    ordering within a task tells you nothing about remapping.
    """
    names = list(regions) + (["ALL"] if include_all else [])
    rows = []
    for name in names:
        ix = (np.arange(len(meta)) if name == "ALL"
              else np.flatnonzero((meta["group"] == name).values))
        if len(ix) < 10:
            continue
        d = diagonality(panels, ix, n_shuffle=n_shuffle, seed=seed)
        r = d.set_index("panel")["circ_r"]
        z = d.set_index("panel")["z_vs_null"]
        sub = meta.iloc[ix]
        share = sub["mouse"].value_counts(normalize=True)
        ceiling = float(r[4])
        novel = float(np.mean([r[2], r[3]]))
        rows.append({
            "region": name, "n": int(len(ix)),
            "n_mice": int(sub["mouse"].nunique()),
            "top_mouse": share.idxmax(), "top_share": round(float(share.max()), 3),
            "taskB_r": round(float(r[2]), 3), "taskC_r": round(float(r[3]), 3),
            "novel_r": round(novel, 3), "ceiling_r": round(ceiling, 3),
            "novel_z": round(float(np.mean([z[2], z[3]])), 2),
            "ceiling_z": round(float(z[4]), 2),
            "retention": (round(novel / ceiling, 3)
                          if ceiling >= MIN_CEILING_R else np.nan),
            "note": ("" if ceiling >= MIN_CEILING_R
                     else "ceiling at floor - tuning not reproducible within task"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def plot_gp_region_grid(panels: np.ndarray, meta: pd.DataFrame, *,
                        regions=REGION_ORDER, include_all: bool = True,
                        vlim: float = 2.0, cmap: str = "coolwarm",
                        save: bool | str = False):
    """Rows = brain region, columns = the 4 task panels."""
    rows = [(r, np.flatnonzero((meta["group"] == r).values)) for r in regions]
    rows = [(r, ix) for r, ix in rows if len(ix) >= 10]
    if include_all:
        rows.append(("ALL", np.arange(len(meta))))

    fig, axes = plt.subplots(len(rows), 4,
                             figsize=(13, 2.45 * len(rows)), squeeze=False)
    for i, (name, ix) in enumerate(rows):
        z, _ = sorted_panels(panels, ix)
        sub = meta.iloc[ix]
        share = sub["mouse"].value_counts(normalize=True)
        for k in range(4):
            ax = axes[i][k]
            ax.imshow(z[k], aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim,
                      interpolation="nearest",
                      extent=(0, 100, len(ix), 0))
            if i == 0:
                ax.set_title(PANEL_LABELS[k], fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("goal progress (%)", fontsize=8)
            else:
                ax.set_xticklabels([])
            if k == 0:
                ax.set_ylabel(f"{name}\nn={len(ix)}", fontsize=8.5)
                if name != "ALL":
                    ax.text(-0.34, 0.02, f"{share.idxmax()} {share.max()*100:.0f}%",
                            transform=ax.transAxes, fontsize=6, style="italic",
                            color=("#B00020" if share.max() > 0.6 else "#666"))
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle("Goal-progress tuning across tasks, by brain region\n"
                 "neurons rank-ordered by peak bin in Task A, same order in all "
                 "four panels", fontsize=11, y=1.005)
    fig.text(0.5, -0.012 / max(1, len(rows) / 6),
             "Panel 1 defines the sort and is therefore diagonal by construction — even for "
             "pure noise (see the synthetic gate); read evidence from panels 2–4 only.  "
             "Panel 4 re-runs Task A, so it is a within-task ceiling, not a fourth task.  "
             "Red mouse-share labels mark rows that are effectively one animal.  "
             "Units are unit-recordings pooled over recdays, not unique neurons.",
             ha="center", fontsize=6.5, style="italic")
    fig.tight_layout()
    if save:
        _save(fig, save if isinstance(save, str) else "gp_region_heatmaps")
    return fig


# --------------------------------------------------------------------------
# synthetic gate
# --------------------------------------------------------------------------

def _synth(kind: str, n: int = 400, *, noise: float = 0.4, seed: int = 0):
    """Plant a population with known cross-task behaviour, shaped like the real
    input: (4, n, 90), panel 4 sharing panel 1's task."""
    rng = np.random.default_rng(seed)
    x = np.arange(N_BINS) * 2 * np.pi / N_BINS
    phase_A = rng.uniform(0, 2 * np.pi, n)

    def bump(ph):
        return np.exp(2.0 * np.cos(x[None, :] - ph[:, None]))

    if kind == "abstract":          # same phase in every task
        phases = [phase_A] * 4
    elif kind == "remap":           # new phase per task; panel 4 == panel 1's task
        phases = [phase_A, rng.uniform(0, 2 * np.pi, n),
                  rng.uniform(0, 2 * np.pi, n), phase_A]
    elif kind == "noise":           # no tuning at all
        phases = None
    else:
        raise ValueError(kind)

    if phases is None:
        out = rng.normal(0, 1, (4, n, N_BINS))
    else:
        out = np.stack([bump(p) + rng.normal(0, noise, (n, N_BINS))
                        for p in phases])
    return out


def run_synthetic_controls(*, n_shuffle: int = 300, verbose: bool = True) -> pd.DataFrame:
    """Gate the pipeline in both directions before any real figure is believed.

    Expectations:
      abstract -> every panel diagonal;
      remap    -> panels 1 and 4 only (which is what makes 4 a valid ceiling);
      noise    -> panel 1 ONLY, demonstrating that panel 1 is not evidence.
    """
    rows, ok = [], True
    for kind in ("abstract", "remap", "noise"):
        d = diagonality(_synth(kind), n_shuffle=n_shuffle)
        r = d.set_index("panel")["circ_r"]
        rows.append(d.assign(case=kind))
        if kind == "abstract":
            good = bool((r[[2, 3, 4]] > 0.5).all())
        elif kind == "remap":
            good = bool(r[4] > 0.5 and abs(r[2]) < 0.25 and abs(r[3]) < 0.25)
        else:
            good = bool(abs(r[2]) < 0.25 and abs(r[3]) < 0.25 and abs(r[4]) < 0.25)
        ok &= good
        if verbose:
            print(f"  {kind:9s} r2={r[2]:+.3f} r3={r[3]:+.3f} r4={r[4]:+.3f}"
                  f"   {'PASS' if good else 'FAIL'}")

    out = pd.concat(rows, ignore_index=True)
    out.attrs["passed"] = bool(ok)
    if verbose:
        print("\nSYNTHETIC GATE:", "PASS" if ok else "FAIL")
    return out


def plot_synthetic_controls(save: bool | str = False, *, n: int = 300):
    """The gate as a picture -- notably the noise row, whose panel 1 is diagonal."""
    cases = ("abstract", "remap", "noise")
    titles = {"abstract": "abstract progress (phase fixed across tasks)",
              "remap": "remapping (new phase per task)",
              "noise": "pure noise (no tuning at all)"}
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.6), squeeze=False)
    for i, kind in enumerate(cases):
        z, _ = sorted_panels(_synth(kind, n=n))
        for k in range(4):
            ax = axes[i][k]
            ax.imshow(z[k], aspect="auto", cmap="coolwarm", vmin=-2, vmax=2,
                      interpolation="nearest", extent=(0, 100, n, 0))
            if i == 0:
                ax.set_title(PANEL_LABELS[k], fontsize=8.5)
            if k == 0:
                ax.set_ylabel(titles[kind], fontsize=8)
            else:
                ax.set_yticklabels([])
            if i < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("goal progress (%)", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Synthetic gate — what the figure looks like when the answer is known",
                 fontsize=11, y=1.005)
    fig.text(0.5, -0.02,
             "The bottom row is the point: pure noise still gives a clean diagonal in panel 1, "
             "because panel 1 defines the sort. Panel 1 is never evidence.",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    if save:
        _save(fig, save if isinstance(save, str) else "gp_synthetic_gate")
    return fig


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def verify(verbose: bool = True) -> pd.DataFrame:
    """Panel identity + join gates, per recday."""
    tasks, regions = load_tasks(), load_regions()
    curves = load_gp_curves()
    rows, ok = [], True
    for recday in sorted(tasks):
        sess = panel_sessions(recday, tasks, _available(recday, curves))
        row = {"recday": recday, "qualifies": sess is not None,
               "sessions": str(sess) if sess else "-"}
        if sess is not None:
            seq = {s: tuple(int(x) for x in tasks[recday][s]) for s in sess}
            row["panel4_is_panel1"] = seq[sess[3]] == seq[sess[0]]
            row["panels_23_novel"] = (seq[sess[1]] != seq[sess[0]]
                                      and seq[sess[2]] != seq[sess[0]]
                                      and seq[sess[1]] != seq[sess[2]])
            have = recday in curves and all(s in curves[recday] for s in sess)
            row["curves_present"] = have
            row["n_curve"] = curves[recday][sess[0]].shape[0] if have else -1
            row["n_region"] = len(regions[recday]) if recday in regions else -1
            row["join_ok"] = row["n_curve"] == row["n_region"] > 0
            ok &= bool(row["panel4_is_panel1"] and row["panels_23_novel"]
                       and row["curves_present"] and row["join_ok"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["passed"] = bool(ok)
    if verbose:
        q = df[df["qualifies"]]
        print(df.to_string(index=False))
        print(f"\n{len(q)}/{len(df)} recdays qualify;  GATES:",
              "PASS" if ok else "FAIL")
    return df
