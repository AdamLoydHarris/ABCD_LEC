"""
Frames for object marking (Stage G1): a per-pixel median over the session and the first frame.

For every object session (and one open-field session per arena, for the arena corners) this
decodes N frames spread over the video and writes the per-pixel median: the animal, which
moves, averages away; the object, which does not, stays sharp. Object positions are recorded
nowhere else (sheets, pyControl and SLEAP skeletons all lack them), so these frames are the
source the click notebook works from.

Outputs (never overwritten):
    data/processed_data/object_frames/{mouse}_{YYYY-MM-DD-HHMMSS}_{kind}_median.png
    data/processed_data/object_frames/{mouse}_{YYYY-MM-DD-HHMMSS}_{kind}_first.png
    data/processed_data/object_frames/index_{dataset}.csv     (recday, session, mouse, stamp, kind, video, n_frames)

Videos are read in place: mEC from the backup `cohort7_openfield/video/`, LEC from
`videos/implanted/{mouse}/`. Run with the `histology` env (cv2).

Usage::
    python object_frames.py --dataset mec [--n 200] [--write]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--dataset" in sys.argv:
    os.environ["TSA_DATASET"] = sys.argv[sys.argv.index("--dataset") + 1]

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import recday_registry as rr                                          # noqa: E402
from openfield_to_xy import openfield_sessions                        # noqa: E402

OUT = rr.REPO_ROOT / "data" / "processed_data" / "object_frames"
VIDEO_ROOTS = {
    "mec": [Path("/ceph/behrens/adam_harris/google_drive_backup/behaviour_videos/cohort7_openfield/video")],
    "lec": [rr.REPO_ROOT / "videos" / "implanted"],
}


def find_video(mouse: str, stamp: str) -> Path | None:
    """`{mouse}_{stamp}.mp4` under the dataset's video roots; exact stamp first, else the unique
    video of that mouse within 2 min (the sheet's Tracking stamps are hand-typed; ly07 2025-06-24
    reads 133853 for a video stamped 133835)."""
    import re
    from datetime import datetime
    for root in VIDEO_ROOTS[rr.DATASET]:
        for cand in (root / f"{mouse}_{stamp}.mp4", root / mouse / f"{mouse}_{stamp}.mp4",
                     root / f"{mouse.upper()}_{stamp}.mp4"):
            if cand.exists():
                return cand
    want = datetime.strptime(stamp, "%Y-%m-%d-%H%M%S")
    near = []
    for root in VIDEO_ROOTS[rr.DATASET]:
        for d in (root, root / mouse):
            for p in d.glob(f"{mouse}_{stamp[:10]}-*.mp4") if d.exists() else []:
                m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{6})", p.name)
                if m and abs((datetime.strptime(m.group(1), "%Y-%m-%d-%H%M%S") - want).total_seconds()) <= 120:
                    near.append(p)
    return near[0] if len(near) == 1 else None


def median_frame(video: Path, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {video}")
    idx = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8), first, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=rr.DATASET, choices=sorted(rr.DATASETS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--kinds", nargs="*", default=["object", "openfield"])
    args = ap.parse_args()
    import cv2

    rows, seen_arena_of = [], set()
    for recday, s, mouse, stamp, kind in openfield_sessions():
        if kind not in args.kinds:
            continue
        if kind == "openfield":
            # one open-field frame per arena is enough for the corners
            arena = "bigfield" if rr.DATASET == "mec" else ("box2" if mouse in ("ah08", "ly06") else "box1")
            if arena in seen_arena_of:
                continue
            seen_arena_of.add(arena)
        video = find_video(mouse, stamp)
        rec = dict(recday=recday, session=s, mouse=mouse, stamp=stamp, kind=kind,
                   video=str(video) if video else None)
        if video is None:
            rec["status"] = "no video"
            rows.append(rec)
            print(f"{recday} s{s} {kind} {stamp}: NO VIDEO")
            continue
        med_path = OUT / f"{mouse}_{stamp}_{kind}_median.png"
        first_path = OUT / f"{mouse}_{stamp}_{kind}_first.png"
        if med_path.exists():
            rec.update(status="exists_skipped")
            rows.append(rec)
            print(f"exists          {recday} s{s} {kind} {stamp}")
            continue
        if args.write:
            med, first, total = median_frame(video, args.n)
            OUT.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(med_path), med)
            cv2.imwrite(str(first_path), first)
            rec.update(status="written", n_frames=total, median_png=med_path.name)
        else:
            cap = cv2.VideoCapture(str(video))
            rec.update(status="dry_run", n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        rows.append(rec)
        print(f"{rec['status']:15s} {recday} s{s} {kind:9s} {stamp} frames={rec.get('n_frames')} {video.name}")

    table = pd.DataFrame(rows)
    if args.write and len(table):
        idx = OUT / f"index_{rr.DATASET}.csv"
        n = 0
        while (p := idx.with_name(idx.stem + (f".v{n}" if n else "") + ".csv")).exists():
            n += 1
        table.to_csv(p, index=False)
        print(f"index -> {p}")


if __name__ == "__main__":
    main()
