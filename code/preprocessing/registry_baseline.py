"""Snapshot every `recday_registry` return, for the equivalence control around a registry edit.

Written 2026-09-22 for the multi-date-recday change (6-day July blocks). Run once before the
edit and once after; `--compare` must report 0 differences.

    python code/preprocessing/registry_baseline.py --write  baseline_lec.json   # TSA_DATASET unset
    TSA_DATASET=mec python code/preprocessing/registry_baseline.py --write baseline_mec.json
    python code/preprocessing/registry_baseline.py --compare baseline_lec.json

I/O: writes/reads one JSON keyed by recday; exit 1 on any difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import recday_registry as rr


def _jsonable(x):
    """Paths, tuples and numpy scalars down to plain JSON types."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (tuple, list)):
        return [_jsonable(i) for i in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def snapshot() -> dict:
    """Every per-recday and per-session registry return, for the active dataset."""
    out = {"dataset": rr.DATASET, "mice": list(rr.EPHYS_MICE), "recdays": {}}
    for mouse in rr.EPHYS_MICE:
        out.setdefault("blocks", {})[mouse] = list(rr.blocks_for(mouse))
    for kind in (None, "merged", "single"):
        out[f"all_recdays_{kind}"] = list(rr.all_recdays(kind))
    for recday in rr.all_recdays():
        rec = {}
        for fn in ("recday_dates", "recday_kind", "recday_to_block", "kilosort_dir",
                   "qc_unit_count", "metadata_tasks"):
            try:
                rec[fn] = _jsonable(getattr(rr, fn)(recday))
            except Exception as e:                      # a raise is part of the contract
                rec[fn] = f"{type(e).__name__}: {e}"
        try:
            rows = rr.task_session_rows(recday)
            rec["n_task_sessions"] = int(len(rows))
            rec["sessions"] = {
                str(s): {
                    "session_type": _safe(rr.session_type, recday, s),
                    "pycontrol_file": _safe(rr.pycontrol_file, recday, s),
                }
                for s in rows.index
            }
        except Exception as e:
            rec["task_session_rows"] = f"{type(e).__name__}: {e}"
        out["recdays"][recday] = rec
    return out


def _safe(fn, *a):
    try:
        return _jsonable(fn(*a))
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def compare(path: Path) -> int:
    """Report every key whose value moved. Returns the number of differences."""
    old = json.loads(path.read_text())
    new = snapshot()
    diffs = []

    def walk(a, b, trail):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a:
                    diffs.append((trail + [k], "<missing>", b[k]))
                elif k not in b:
                    diffs.append((trail + [k], a[k], "<missing>"))
                else:
                    walk(a[k], b[k], trail + [k])
        elif a != b:
            diffs.append((trail, a, b))

    walk(old, new, [])
    for t, a, b in diffs[:40]:
        print(f"  DIFF {'/'.join(map(str, t))}:\n    was {a!r}\n    now {b!r}")
    if len(diffs) > 40:
        print(f"  ... and {len(diffs) - 40} more")
    print(f"{path.name}: {len(diffs)} difference(s) over "
          f"{len(old['recdays'])} recdays, dataset {old['dataset']}")
    return len(diffs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", metavar="PATH")
    p.add_argument("--compare", metavar="PATH")
    args = p.parse_args()
    if args.write:
        out = Path(args.write)
        if out.exists():
            sys.exit(f"{out} exists; baselines are never overwritten")
        snap = snapshot()
        out.write_text(json.dumps(snap, indent=1, sort_keys=True))
        print(f"wrote {out}: dataset {snap['dataset']}, {len(snap['recdays'])} recdays")
    elif args.compare:
        sys.exit(1 if compare(Path(args.compare)) else 0)
    else:
        p.error("one of --write / --compare")
