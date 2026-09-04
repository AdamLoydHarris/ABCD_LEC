"""Assign brain regions to recording channels and to post-QC single units.

Turns the corrected probe fits into the thing the analyses actually need: for
every recorded channel, and every good unit, which structure it sits in.

Usage::

    import region_assignment as ra
    ch = ra.channel_regions('ah08', ra.blocks_for('ah08')[0])   # per channel
    un = ra.unit_regions('ah08', ra.blocks_for('ah08')[0])      # per QC unit
    arrays = ra.build_unit_region_arrays()                      # for downstream

**The channel-id trap, and why this module exists.**
``cluster_reports/unit list.csv`` gives each unit a ``max_on_channel_id`` like
``'CH123'``.  That is a channel **name**, not a row index into
``channel_positions.npy``.  The retained channels are a *subset* of CH0-CH383
and the dropped ones are **interior**, not at the tail -- ah08 is missing CH98,
CH120, CH153, CH278, CH286, CH324, CH348; ly07 misses CH3, CH108, CH111, CH140,
CH165, CH360 -- so every channel after the first gap is shifted.  The correct
map is ``channel_ids`` from
``sorting_analyzer/recording_info/recording_attributes.json``.

The superseded ``find_LEC_units`` indexed the number directly.  Measured against
the correct lookup over 2839 QC units, that put **47.6% on a different shank**
(shanks sit in different structures) and **72.3% at a different depth**, and
crashed on 12.  Validation against an independent peak-channel computation from
the analyzer templates: **90.4%** exact agreement for the name lookup versus
**9.3%** for 0-based arithmetic and **1.8%** for 1-based.  ``validate_mapping``
ships that check so it cannot silently regress.

Note ``channel_map.npy`` cannot reveal this: it is ``[0..n-1]`` by construction.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import probe_refit as pr

EPHYS_ROOT = pr.REPO_ROOT / "data" / "preprocessed" / "ephys"
OUT_DIR = pr.REPO_ROOT / "data" / "processed_data"
FIGURE_DIR = pr.REPO_ROOT / "data" / "figures" / "region_census"

#: Mice with ephys.  ah09 has histology but no recordings.
EPHYS_MICE = ("ah08", "ah10", "ly05", "ly06", "ly07")

#: Coarse groups, in plotting order.  Keep ENTl split by depth -- superficial
#: (2/3) and deep (5/6a) entorhinal layers are different populations, and the
#: whole point of the histology work was to stop lumping them.
REGION_GROUPS = {
    "ENTl-sup": ("ENTl1", "ENTl2", "ENTl3"),
    "ENTl-deep": ("ENTl5", "ENTl6a", "ENTl6b"),
    "ENTm": None,          # prefix match
    "SUB/ProS": ("SUB", "ProS"),
    "CA1/HPF": ("CA1", "CA2", "CA3", "DG", "HPF"),
    "VIS": None,           # prefix match
}
GROUP_COLOURS = {
    "ENTl-deep": "#1f77b4", "ENTl-sup": "#7fb8e0", "ENTm": "#ff7f0e",
    "SUB/ProS": "#9467bd", "CA1/HPF": "#2ca02c", "VIS": "#8c8c8c",
    "fibre/other": "#d9d9d9", "none": "#f0f0f0",
}


def group_of(acronym) -> str:
    """Coarse group for an Allen acronym."""
    if not isinstance(acronym, str) or not acronym:
        return "none"
    for group, members in REGION_GROUPS.items():
        if members is None:
            if acronym.startswith(group):
                return group
        elif acronym in members:
            return group
    if acronym.startswith("ENTl"):
        return "ENTl-deep" if acronym[4:5] in "56" else "ENTl-sup"
    if acronym.startswith("VIS"):
        return "VIS"
    return "fibre/other"


# --------------------------------------------------------------------------
# blocks and the channel-name map
# --------------------------------------------------------------------------


def blocks_for(mouse: str) -> list[str]:
    """Sorted recording-block directory names for a mouse."""
    return sorted(os.path.basename(p)
                  for p in glob.glob(str(EPHYS_ROOT / mouse / "*_preprocessed")))


def _ks(mouse: str, block: str) -> Path:
    return EPHYS_ROOT / mouse / block / "kilosort_output"


def channel_name_map(mouse: str, block: str) -> dict[str, int]:
    """``'CH123' -> row index`` into ``channel_positions.npy``.

    Read from the recording's own ``channel_ids``; never computed from the
    digits, for the reason in the module docstring.
    """
    p = _ks(mouse, block) / "sorting_analyzer/recording_info/recording_attributes.json"
    ids = json.loads(p.read_text())["channel_ids"]
    return {name: i for i, name in enumerate(ids)}


def validate_mapping(mouse: str, block: str) -> dict:
    """Check the name map against a peak channel computed from the templates.

    Independent of the CSV, so a regression in either shows up as a fall in
    agreement.  Expect ~90%; the residual is the peak metric definition
    (|amplitude| vs peak-to-trough), not the mapping.
    """
    ks = _ks(mouse, block)
    n2r = channel_name_map(mouse, block)
    ul = pd.read_csv(ks / "cluster_reports/unit list.csv", sep="\t")
    tpl = np.load(ks / "sorting_analyzer/extensions/templates/average.npy")
    peak = np.abs(tpl).max(axis=1).argmax(axis=1)
    rows = ul["max_on_channel_id"].map(n2r)
    resolved = rows.notna()
    n = min(len(peak), len(rows))
    agree = float((rows[:n][resolved[:n]].astype(int).values
                   == peak[:n][resolved[:n].values]).mean())
    naive = ul["max_on_channel_id"].astype(str).str.extract(r"(\d+)").astype(int)[0]
    naive_ok = naive < len(n2r)
    naive_agree = float((naive[:n][naive_ok[:n]].values
                         == peak[:n][naive_ok[:n].values]).mean())
    return {"mouse": mouse, "block": block, "n_units": int(len(ul)),
            "resolved": int(resolved.sum()), "agree_name_map": agree,
            "agree_naive_arithmetic": naive_agree,
            "n_channels": len(n2r),
            "missing_channels": [i for i in range(384) if f"CH{i}" not in n2r]}


# --------------------------------------------------------------------------
# channel- and unit-level assignment
# --------------------------------------------------------------------------


def _contact_lookup(mouse: str, data: dict | None = None) -> pd.DataFrame:
    """Projected probe contacts for a subject, indexed by (x, y) in um."""
    fit = pr.load_fit(mouse)
    plane, params = pr.split_fit(fit)
    data = data if data is not None else pr.load_volumes(mouse)
    df = pr.project_probe(plane, params, data)
    df = df.assign(_x=df["probe_coords.x"].round(1),
                   _y=df["probe_coords.y"].round(1))
    return df.set_index(["_x", "_y"])


def channel_regions(mouse: str, block: str, *, data: dict | None = None,
                    contacts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Region per **recorded channel** (the real 377-383, not 384 contacts)."""
    ks = _ks(mouse, block)
    pos = np.load(ks / "sorter_output/channel_positions.npy")
    ids = json.loads((ks / "sorting_analyzer/recording_info/recording_attributes.json")
                     .read_text())["channel_ids"]
    contacts = contacts if contacts is not None else _contact_lookup(mouse, data)

    rows = []
    for row, (name, (x, y)) in enumerate(zip(ids, pos)):
        key = (round(float(x), 1), round(float(y), 1))
        hit = contacts.loc[key] if key in contacts.index else None
        if hit is not None and isinstance(hit, pd.DataFrame):
            hit = hit.iloc[0]
        acro = hit["structure.acronym"] if hit is not None else np.nan
        rows.append({
            "mouse": mouse, "block": block, "row": row, "channel": name,
            "x_um": float(x), "y_um": float(y),
            "shank": int(pr.shank_id(x)),
            "in_recorded_bank": bool(y <= pr.RECORDED_BANK_MAX_UM),
            "ap_i": float(hit["downsample_coords.i"]) if hit is not None else np.nan,
            "acronym": acro, "group": group_of(acro),
        })
    return pd.DataFrame(rows)


def unit_regions(mouse: str, block: str, *, data: dict | None = None,
                 contacts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Region per QC single unit, **ordered exactly as ``QC_single_units.npy``**.

    That ordering is what `Neuron_raw` rows follow, so the result drops straight
    into the existing analysis code.
    """
    ks = _ks(mouse, block)
    qc = np.load(ks / "QC_single_units.npy")
    ul = pd.read_csv(ks / "cluster_reports/unit list.csv", sep="\t").set_index("unit_id")
    n2r = channel_name_map(mouse, block)
    ch = channel_regions(mouse, block, data=data, contacts=contacts).set_index("row")

    rows = []
    for order, uid in enumerate(qc):
        name = ul["max_on_channel_id"].get(uid, None)
        row = n2r.get(name) if isinstance(name, str) else None
        rec = ch.loc[row] if row is not None and row in ch.index else None
        rows.append({
            "mouse": mouse, "block": block, "order": order, "unit_id": int(uid),
            "max_channel": name, "channel_row": row,
            "shank": int(rec["shank"]) if rec is not None else -1,
            "y_um": float(rec["y_um"]) if rec is not None else np.nan,
            "ap_i": float(rec["ap_i"]) if rec is not None else np.nan,
            "acronym": rec["acronym"] if rec is not None else np.nan,
            "group": rec["group"] if rec is not None else "unresolved",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# cohort build
# --------------------------------------------------------------------------


def block_to_recday(mouse: str, block: str) -> str:
    """``ah08`` + ``2025-06-13_2025-06-15_preprocessed`` -> ``ah08_20250613_20250615``."""
    d = re.findall(r"(\d{4})-(\d{2})-(\d{2})", block)
    return f"{mouse}_" + "_".join("".join(x) for x in d)


def build_cohort(mice=EPHYS_MICE, *, verbose: bool = True):
    """Channel and unit tables for every block, plus the mapping-validation table."""
    chans, units, checks = [], [], []
    for mouse in mice:
        data = pr.load_volumes(mouse)
        contacts = _contact_lookup(mouse, data)
        for block in blocks_for(mouse):
            checks.append(validate_mapping(mouse, block))
            c = channel_regions(mouse, block, data=data, contacts=contacts)
            u = unit_regions(mouse, block, data=data, contacts=contacts)
            c["recday"] = u["recday"] = block_to_recday(mouse, block)
            chans.append(c); units.append(u)
            if verbose:
                print(f"  {mouse} {block[:21]}: {len(c)} channels, {len(u)} units",
                      flush=True)
        pr.clear_volume_cache(mouse)
    return (pd.concat(chans, ignore_index=True),
            pd.concat(units, ignore_index=True),
            pd.DataFrame(checks))


def build_unit_region_arrays(units: pd.DataFrame) -> dict:
    """``{mouse_recday: DataFrame}`` in `QC_single_units` order, for downstream.

    Aligned to `Neuron_raw` rows so it can index the existing GLM inputs.
    """
    return {rd: g.sort_values("order").reset_index(drop=True)
            for rd, g in units.groupby("recday")}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _order_groups(cols) -> list:
    """Groups in a stable plotting order."""
    pref = list(REGION_GROUPS) + ["fibre/other", "none", "unresolved"]
    return [g for g in pref if g in cols] + [g for g in cols if g not in pref]


def _stacked(ax, frac: pd.DataFrame, title: str, counts=None):
    groups = _order_groups(frac.columns)
    bottom = np.zeros(len(frac))
    for g in groups:
        ax.bar(frac.index, frac[g], bottom=bottom, label=g,
               color=GROUP_COLOURS.get(g, "#bbbbbb"), edgecolor="white", lw=0.6)
        bottom += frac[g].values
    ax.set_ylim(0, 1); ax.set_ylabel("fraction")
    ax.set_title(title, fontsize=11)
    if counts is not None:
        for i, m in enumerate(frac.index):
            ax.text(i, 1.02, f"n={counts[m]}", ha="center", fontsize=8, color="#444")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_region_census(chans: pd.DataFrame, units: pd.DataFrame,
                       save: bool | str = False):
    """Channel- and unit-level composition per mouse, and cohort totals."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9),
                             gridspec_kw={"width_ratios": [3, 1]})
    for row, (df, lbl) in enumerate(((chans, "recording channels"),
                                     (units, "QC single units"))):
        frac = pd.crosstab(df["mouse"], df["group"], normalize="index")
        _stacked(axes[row][0], frac, f"{lbl} — per mouse",
                 counts=df.groupby("mouse").size())
        tot = df["group"].value_counts(normalize=True).to_frame().T
        tot.index = ["cohort"]
        _stacked(axes[row][1], tot, f"{lbl} — total (n={len(df)})")
        axes[row][1].legend(fontsize=8, loc="center left",
                            bbox_to_anchor=(1.05, 0.5), frameon=False)
    fig.suptitle("Brain-region assignment of recording channels and single units",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        p = FIGURE_DIR / (save if isinstance(save, str) else "region_census.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"saved {p}")
    return fig


def plot_yield_by_region(chans: pd.DataFrame, units: pd.DataFrame,
                         save: bool | str = False):
    """Units per channel by region -- sampling is not the same as yield.

    Channel composition says what the probe *sampled*; unit composition says
    what it *recorded*.  They differ systematically, so the n available per
    region in any analysis is not proportional to the channels in it.
    """
    import matplotlib.pyplot as plt

    c = chans["group"].value_counts()
    u = units["group"].value_counts()
    groups = [g for g in _order_groups(set(c.index) | set(u.index)) if c.get(g, 0) > 0]
    cf = np.array([c.get(g, 0) / len(chans) for g in groups])
    uf = np.array([u.get(g, 0) / len(units) for g in groups])
    yld = np.array([u.get(g, 0) / c.get(g, np.nan) for g in groups])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(groups))
    ax[0].bar(x - 0.2, cf, 0.4, label="channels (sampled)", color="#9ecae1")
    ax[0].bar(x + 0.2, uf, 0.4, label="units (recorded)", color="#3182bd")
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups, rotation=30, ha="right")
    ax[0].set_ylabel("fraction of total"); ax[0].legend(fontsize=9, frameon=False)
    ax[0].set_title("composition: sampled vs recorded", fontsize=11)

    ax[1].bar(x, yld, color=[GROUP_COLOURS.get(g, "#bbb") for g in groups])
    ax[1].axhline(len(units) / len(chans), ls="--", c="#B00020", lw=1,
                  label=f"cohort mean {len(units)/len(chans):.2f}")
    ax[1].set_xticks(x); ax[1].set_xticklabels(groups, rotation=30, ha="right")
    ax[1].set_ylabel("units per channel"); ax[1].legend(fontsize=9, frameon=False)
    ax[1].set_title("unit yield per channel by region", fontsize=11)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        p = FIGURE_DIR / (save if isinstance(save, str) else "yield_by_region.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"saved {p}")
    return fig


def plot_per_shank(chans: pd.DataFrame, units: pd.DataFrame,
                   save: bool | str = False):
    """Composition per shank, ordered by physical AP position.

    Shank *numbers* carry the contact-face convention (see PROBE_REFIT.md 7g);
    the AP ordering and the anatomy are convention-independent.
    """
    import matplotlib.pyplot as plt

    mice = sorted(chans["mouse"].unique())
    fig, axes = plt.subplots(2, len(mice), figsize=(3.1 * len(mice), 7.5),
                             sharey=True)
    for col, m in enumerate(mice):
        for row, (df, lbl) in enumerate(((chans, "channels"), (units, "units"))):
            d = df[df["mouse"] == m]
            order = (d.groupby("shank")["ap_i"].mean().sort_values().index.tolist())
            frac = pd.crosstab(d["shank"], d["group"], normalize="index").reindex(order)
            frac.index = [f"{s}?" for s in order]
            _stacked(axes[row][col], frac, f"{m} — {lbl}" if row == 0 else lbl)
            axes[row][col].set_xlabel("shank (anterior → posterior)", fontsize=8)
            if col:
                axes[row][col].set_ylabel("")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=9, frameon=False)
    fig.suptitle("Per-shank composition, ordered by AP position "
                 "(shank numbers pending the surgical record)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        p = FIGURE_DIR / (save if isinstance(save, str) else "per_shank_census.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"saved {p}")
    return fig


def implant_table(chans: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """ENTl yield per mouse against implant geometry, for planning future cohorts."""
    rows = []
    for m in sorted(chans["mouse"].unique()):
        fit = pr.load_fit(m)
        plane, params = pr.split_fit(fit)
        ang = pr.trajectory_angles(plane["v_axis"])
        u = np.asarray(plane["u_axis"], float)
        entry = np.asarray(plane["surface_coord"], float)
        c = chans[chans["mouse"] == m]; un = units[units["mouse"] == m]
        rows.append({
            "mouse": m,
            "ENTl_frac_channels": float(c["group"].str.startswith("ENTl").mean()),
            "ENTl_frac_units": float(un["group"].str.startswith("ENTl").mean()),
            "array_dev_from_AP_deg": float(np.degrees(np.arccos(min(abs(u[0]), 1.0)))),
            "lateral_deg": ang["lateral_deg"], "ap_deg": ang["ap_deg"],
            "entry_i": entry[0], "entry_j": entry[1], "entry_k": entry[2],
            "depth_um": float(params["probe_depth"]),
        })
    return pd.DataFrame(rows)


def plot_implant_relationship(tab: pd.DataFrame, save: bool | str = False):
    """ENTl yield vs implant geometry.  n = 5 mice: DESCRIPTIVE, not inferential.

    Plotted with every mouse labelled and rho quoted as a summary only.  With
    five points a correlation carries essentially no evidential weight -- this
    is a lookup for planning the next implant, not a result.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    xs = [("array_dev_from_AP_deg", "shank-array deviation from A-P (deg)"),
          ("lateral_deg", "trajectory lateral angle (deg)"),
          ("ap_deg", "trajectory AP angle (deg)"),
          ("entry_k", "entry medio-lateral (voxel k)"),
          ("entry_i", "entry antero-posterior (voxel i)"),
          ("depth_um", "probe depth (um)")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, lbl) in zip(axes.ravel(), xs):
        for y, c, mk in (("ENTl_frac_channels", "#3182bd", "o"),
                         ("ENTl_frac_units", "#e6550d", "s")):
            ax.scatter(tab[col], tab[y], c=c, marker=mk, s=70,
                       label=y.replace("ENTl_frac_", ""), zorder=3)
            r = spearmanr(tab[col], tab[y]).statistic
            ax.plot([], [], " ", label=f"  rho={r:+.2f}")
        for _, r in tab.iterrows():
            ax.annotate(r["mouse"], (r[col], r["ENTl_frac_channels"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(lbl, fontsize=9); ax.set_ylabel("ENTl fraction", fontsize=9)
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("ENTl yield vs implant geometry — n = 5 mice, DESCRIPTIVE ONLY "
                 "(rho quoted as a summary, not evidence)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        p = FIGURE_DIR / (save if isinstance(save, str) else "implant_relationship.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"saved {p}")
    return fig


def plot_counts(df: pd.DataFrame, *, label: str = "units",
                annotate: bool = True, save: bool | str = False):
    """Absolute counts: grouped bars per mouse, plus the cohort total.

    Complements `plot_region_census`, which shows fractions.  Fractions answer
    "what is this mouse made of"; counts answer "how many cells do I actually
    have in region X", which is the number that decides whether a per-region
    analysis is powered at all.
    """
    import matplotlib.pyplot as plt

    tab = pd.crosstab(df["mouse"], df["group"])
    groups = [g for g in _order_groups(tab.columns) if tab[g].sum() > 0]
    tab = tab[groups]
    mice = list(tab.index)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                             gridspec_kw={"width_ratios": [len(mice), 1.15]})
    width = 0.8 / len(groups)
    base = np.arange(len(mice))
    for i, g in enumerate(groups):
        pos = base + (i - (len(groups) - 1) / 2) * width
        bars = axes[0].bar(pos, tab[g].values, width, label=g,
                           color=GROUP_COLOURS.get(g, "#bbbbbb"),
                           edgecolor="white", lw=0.5)
        if annotate:
            for b, v in zip(bars, tab[g].values):
                if v:
                    axes[0].text(b.get_x() + b.get_width() / 2, v, str(int(v)),
                                 ha="center", va="bottom", fontsize=6.5,
                                 rotation=90)
    axes[0].set_xticks(base)
    axes[0].set_xticklabels([f"{m}\n(n={tab.loc[m].sum()})" for m in mice])
    axes[0].set_ylabel(f"number of {label}")
    axes[0].set_title(f"{label} per region, per mouse", fontsize=11)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)

    tot = tab.sum(axis=0)
    bars = axes[1].bar(np.arange(len(groups)), tot.values,
                       color=[GROUP_COLOURS.get(g, "#bbbbbb") for g in groups],
                       edgecolor="white", lw=0.5)
    if annotate:
        for b, v in zip(bars, tot.values):
            axes[1].text(b.get_x() + b.get_width() / 2, v, str(int(v)),
                         ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(np.arange(len(groups)))
    axes[1].set_xticklabels(groups, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel(f"number of {label}")
    axes[1].set_title(f"total across mice (n={int(tot.sum())})", fontsize=11)

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"{label.capitalize()} per brain region — absolute counts", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        p = FIGURE_DIR / (save if isinstance(save, str) else f"{label}_counts.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"saved {p}")
    return fig
