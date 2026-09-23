"""
Stage A of the mEC integration: copy the cohort-7 behaviour inputs into the LEC raw layout.

The mEC tree (`Taskspace_abstraction_mEC`) has sorted spikes for bp01, bp02, bp04 but its
`data/raw/{pycontrol,pinstate,sleap}` are empty or bp01-only. The complete inputs live in the
Google-Drive backup. This script copies, for the **sorted dates only**, into the mEC tree:

    data/raw/pycontrol/{mouse}-YYYY-MM-DD-HHMMSS.{txt,tsv}
    data/raw/pinstate/{mouse}_pinstate_YYYY-MM-DD-HHMMSS.csv
    data/raw/sleap/{mouse}/maze/<model>.<idx>_{mouse}_YYYY-MM-DD-HHMMSS.analysis.{h5,csv}
    data/metadata/MetaData-{mouse}.csv

which is the layout `recday_registry`, `extract_pokes` and `sleap_alignment_copilot` read, and
writes corrected copies of the mEC sbatch files whose script name does not exist there.

Nothing is deleted or overwritten (user's rule, 2026-09-18). A destination that already exists
with identical bytes is skipped; one that exists with different bytes is written as
`<name>.conflictN` and flagged. Every action goes in the manifest CSV.

Usage::

    python mec_consolidate_inputs.py            # dry run: report what would be copied
    python mec_consolidate_inputs.py --write    # copy + write manifest

I/O: reads the backup and the mEC sorted-block directory names; writes ~600 small files, ~190 h5.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

MEC_ROOT = Path("/ceph/behrens/adam_harris/Taskspace_abstraction_mEC")
BACKUP = Path("/ceph/behrens/adam_harris/google_drive_backup/behaviour_videos")
SLEAP_SRC = BACKUP / "cohort7_SLEAP" / "ABCD_july_2025"
OPENFIELD_PYCONTROL = BACKUP / "cohort7_openfield" / "pycontrol"
# Big open field (7x7 rig): single-instance SLEAP from the July-2025 run, pinstate beside the videos
OPENFIELD_SLEAP_SRC = BACKUP / "cohort7_SLEAP" / "bigmaze_july_2025"
OPENFIELD_VIDEO = BACKUP / "cohort7_openfield" / "video"
SHEETS_SRC = MEC_ROOT / "data" / "cohort7"
MANIFEST = MEC_ROOT / "data" / "metadata" / "consolidation_manifest.csv"

# Backup folder per mouse: bp01+bp02 and bp03+bp04 were run as pairs into one folder each.
BACKUP_FOLDER = {"bp01": "BP01_BP02", "bp02": "BP01_BP02", "bp04": "BP03_BP04"}
MICE = ("bp01", "bp02", "bp04")

# The 28 top-level / bp02 / bp04 sbatch files call this name, which does not exist in the mEC tree.
SBATCH_BAD = "code/merge_preprocess_kilosort_spikeinterface.py"
SBATCH_GOOD = "code/merge_preprocess_kilosort_spikeinterface_mEC.py"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sorted_dates(mouse: str) -> set[str]:
    """Dates with a sorted block, from the block directory names (never hard-coded)."""
    dates = set()
    for block in (MEC_ROOT / "data" / "preprocessed" / "ephys" / mouse).glob("*_preprocessed"):
        dates.update(re.findall(r"\d{4}-\d{2}-\d{2}", block.name))
    return dates


def date_of(name: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})-\d{6}", name)
    return m.group(1) if m else None


def plan_copies(mouse: str) -> list[tuple[Path, Path, str]]:
    """(source, destination, family) for every file of this mouse on a sorted date."""
    dates = sorted_dates(mouse)
    folder = BACKUP / BACKUP_FOLDER[mouse]
    out: list[tuple[Path, Path, str]] = []

    # pyControl ABCD task files: bp02-2024-03-24-105246.txt (already lower-case)
    for src in sorted(folder.glob(f"{mouse}-*.txt")):
        if date_of(src.name) in dates:
            out.append((src, MEC_ROOT / "data/raw/pycontrol" / src.name, "pycontrol"))
    # pyControl open-field .tsv (pyControl 2 format), kept so the metadata Behaviour stamps resolve
    for src in sorted(OPENFIELD_PYCONTROL.glob(f"{mouse}-*.tsv")):
        if date_of(src.name) in dates:
            out.append((src, MEC_ROOT / "data/raw/pycontrol" / src.name, "pycontrol_tsv"))
    # pinstate: BP02_pinstate_2024-03-24-105214.csv -> bp02_pinstate_...
    for src in sorted(folder.glob(f"{mouse.upper()}_pinstate_*.csv")):
        if date_of(src.name) in dates:
            out.append((src, MEC_ROOT / "data/raw/pinstate" / src.name.replace(mouse.upper(), mouse, 1),
                        "pinstate"))
    # SLEAP July-2025 single-instance outputs: labels_test_mec_bp02.027_bp02_2024-03-24-105214.analysis.h5
    for src in sorted(SLEAP_SRC.glob(f"*_{mouse}_*.analysis.*")):
        if src.suffix in (".h5", ".csv") and date_of(src.name) in dates:
            out.append((src, MEC_ROOT / "data/raw/sleap" / mouse / "maze" / src.name, "sleap"))
    # Big open field / object sessions (Stage E1): h5 from the July-2025 run, pinstate from the
    # video folder (one per video, stamps match the sheet's Tracking column)
    for src in sorted(OPENFIELD_SLEAP_SRC.glob(f"*_{mouse}_*.analysis.h5")):
        if date_of(src.name) in dates:
            out.append((src, MEC_ROOT / "data/raw/sleap" / mouse / "openfield" / src.name, "openfield_sleap"))
    for src in sorted(OPENFIELD_VIDEO.glob(f"{mouse}_pinstate_*.csv")):
        if date_of(src.name) in dates:
            out.append((src, MEC_ROOT / "data/raw/pinstate" / src.name, "openfield_pinstate"))
    # metadata sheet
    sheet = SHEETS_SRC / f"MetaData.xlsx - {mouse}.csv"
    if sheet.exists():
        out.append((sheet, MEC_ROOT / "data/metadata" / f"MetaData-{mouse}.csv", "metadata"))
    return out


def plan_sbatch_fixes() -> list[tuple[Path, Path]]:
    fixes = []
    for src in sorted((MEC_ROOT / "sbatch_files").rglob("*.sbatch")):
        if "fixed" in src.parts:
            continue
        if SBATCH_BAD in src.read_text() and SBATCH_GOOD not in src.read_text():
            fixes.append((src, MEC_ROOT / "sbatch_files" / "fixed" / src.relative_to(MEC_ROOT / "sbatch_files")))
    return fixes


def copy_no_overwrite(src: Path, dst: Path, write: bool) -> tuple[str, Path]:
    """Copy unless an identical file is already there. Never replace a differing one."""
    if dst.exists():
        if md5(src) == md5(dst):
            return "skip_identical", dst
        n = 1
        while (alt := dst.with_name(dst.name + f".conflict{n}")).exists():
            n += 1
        if write:
            alt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, alt)
        return "conflict_written_beside", alt
    if write:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return "copied", dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help="perform the copies; default is a dry run")
    args = ap.parse_args()

    rows = []
    per_day: dict[tuple[str, str, str], int] = defaultdict(int)
    for mouse in MICE:
        for src, dst, family in plan_copies(mouse):
            action, final = copy_no_overwrite(src, dst, args.write)
            rows.append(dict(mouse=mouse, family=family, date=date_of(src.name) or "",
                             source=str(src), destination=str(final), action=action,
                             bytes=src.stat().st_size, md5=md5(src)))
            if family in ("pycontrol", "pinstate") or (family == "sleap" and src.suffix == ".h5"):
                per_day[(mouse, date_of(src.name) or "", family)] += 1

    for src, dst in plan_sbatch_fixes():
        text = src.read_text().replace(SBATCH_BAD, SBATCH_GOOD)
        action = "skip_identical" if dst.exists() and dst.read_text() == text else "written"
        if args.write and action == "written":
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
        rows.append(dict(mouse="", family="sbatch_fix", date="", source=str(src), destination=str(dst),
                         action=action, bytes=len(text), md5=hashlib.md5(text.encode()).hexdigest()))

    manifest = pd.DataFrame(rows)
    mode = "WRITE" if args.write else "DRY RUN"
    print(f"[{mode}] {len(manifest)} planned actions")
    print(manifest.groupby(["mouse", "family", "action"]).size().to_string())
    print("\nper sorted day: pyControl txt / pinstate / SLEAP h5")
    table = pd.Series(per_day).unstack("family" if False else 2).fillna(0).astype(int)
    print(table.to_string())
    short = table[(table.get("pycontrol", 0) < 4) | (table.get("pinstate", 0) < 4) | (table.get("sleap", 0) < 4)]
    if len(short):
        print(f"\n{len(short)} mouse-days have fewer than 4 of some family -- check these against the sheet")
    # every SLEAP h5 should have exactly one pinstate with the same stamp
    stamps = defaultdict(set)
    for r in rows:
        m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", Path(r["source"]).name)
        if m and r["family"] in ("pinstate", "sleap") and (r["family"] == "pinstate" or r["source"].endswith(".h5")):
            stamps[(r["mouse"], m.group(1))].add(r["family"])
    unpaired = sorted(k for k, v in stamps.items() if v != {"pinstate", "sleap"})
    print(f"\nSLEAP h5 <-> pinstate stamp pairing: {len(stamps) - len(unpaired)} paired, {len(unpaired)} unpaired")
    for k in unpaired[:20]:
        print("  unpaired", k, stamps[k])

    if args.write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        if MANIFEST.exists():
            n = 1
            while (alt := MANIFEST.with_name(MANIFEST.stem + f".v{n}.csv")).exists():
                n += 1
            manifest.to_csv(alt, index=False)
            print(f"\nmanifest exists; wrote {alt}")
        else:
            manifest.to_csv(MANIFEST, index=False)
            print(f"\nmanifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
