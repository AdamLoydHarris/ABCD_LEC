# Tracking QC and cleaning v2 (SLEAP, both datasets, all arenas)

Dated paragraphs; nothing rewritten. Plan: `docs/handoff/TRACKING_QC_PLAN.md` (2026-09-20, superseded
in its ordering by the decisions of 2026-09-23 recorded here). Code: `tracking_clean.py`,
`tracking_clean_synthetics.py`, `tracking_qc.py`, `sleap_to_rois.py`, `openfield_to_xy.py`,
`sleap_alignment_copilot.py`, `../glm_analysis_v3.py` (+ mirror), `../glm_v3_synthetics.py`.

## 2026-09-23 -- scope, decisions, snapshot

Decisions (user, after a grill on measured numbers): (D1) fix the mEC maze-2 geometry first with its
own rebuild; (D2) run the QC on every session of both datasets BEFORE freezing the v2 constants;
(D3) the GLM V3 NaN path lands inside v3 as a default-on branch that only triggers on NaN (no GLM
result exists on a NaN-carrying pickle); (D4) gaps bridged by a straight line up to 2 s everywhere,
plus a per-frame provenance flag file; (D5) GLM bins with NaN speed are dropped from the fit, not
zero-coded; (D6) `Locs_raw` from the cleaned track; (D7) the 5-frame rolling mean stays; (D8) the
confidence gate is conditional; (D9) straight-line bridging, maze-graph bridging deferred. Nothing
writes under `data/raw/`; every output is a new sibling directory or a new pickle name.

Snapshot before the first edit: commits d406dd1 (mEC integration 2026-09-17..20) and 6035467
(working state), tag `pre-tracking-qc-2026-09-23`; sha256 manifests
`data/processed_data/checksums_2026-09-23_pre_tracking_qc.txt` in both trees (LEC 8577 lines,
mEC 5082).

## 2026-09-23 -- Stage 0: the mEC maze-2 affine was degenerate

`MAZE2_MEC` in `sleap_to_rois.py` listed the four corner pixels in Z order (node 1, 3, 7, 9) against
the cyclic `PHYSICAL_PTS`. No affine realises that pairing, so the least-squares fit collapsed x.

| corner set | det(A) | corner residual | head_back x range in the CSVs and in `data/processed/` |
|---|---|---|---|
| MAZE1_EARLY / LATE (bp01) | -3.1e-3 / -2.9e-3 | 0.38 / 0.41 cm | -3 to 49 cm |
| MAZE2_MEC Z order (bp02, bp04; 69 sessions) | -7e-5 | 23 cm | 22.8 to 24.2 cm |
| MAZE2_MEC cyclic (fix) | -3.8e-3 | 0.52 cm | 47 to 50 cm |

Node-mask centroids under the cyclic order land at the same cm positions as under maze 1 and both
LEC geometries (node 1 at (5, 5), node 3 at (42, 5), node 9 at (42, 42) cm). `HD_raw` was computed
in the collapsed frame, so 87-89 % of bp02/bp04 bins sat within 10 deg of +-90 (bp01: 15-19 %).
`Locs_raw` (raw pixels + masks) was never affected, which is why every validator passed.

Fix: `--geometry {v1,v2}` (v1 default reproduces `maze/`; v2 writes `maze_geom2/`), an affine guard
(`affine()` raises above 2 cm residual or below |det| 1e-4 unless the v1 path passes `check=False`),
QC columns `affine_resid_cm`, `hb_x_span_cm`, `hb_y_span_cm`. Then
`sleap_alignment_copilot.py --dataset mec --metadata --sync matched --sessions all --tracking-clean
geom2 --out-tag v2` -> mEC `data/processed_v2/` (212 sessions, same clock fits: 4768 uncertain bins
as before). Gates, all measured:

| gate | result |
|---|---|
| bp01 CSVs `maze_geom2/` vs `maze/` | 99/99 files byte-identical |
| ROI CSVs, all 102 sessions | 102/102 byte-identical |
| bp02 / bp04 head_back span, `maze_geom2/` | median x 47.5 / 50.1 cm, y 47.1 / 50.3 cm; 8 of 69 under 40 cm (the all-NaN 114 s bp02 03-27 14:17, a 25 s fragment, six sessions at 39-40 cm) |
| HD within 10 deg of +-90, `maze_geom2/` | bp02 0.871 -> 0.134, bp04 0.866 -> 0.159, bp01 0.192 -> 0.192 |
| copilot arrays `processed_v2/` vs `processed/` | bp01 XY/HD/Locs 100 % byte-identical; every Locs array of every mouse identical; bp02/bp04 XY x-span median 47.4 / 49.9 cm; HD +-90 fraction 0.160 / 0.154 |

`validate_alignment.py --dataset mec --neural v2 --tracking v2`: 216 session rows, all checks pass
(`alignment_qc_mec_nv2_tv2.csv`). Against the previous report: `loc_at_goal` and `poke_loc`
identical in every scored session (the 48 non-identical rows are NaN-vs-NaN arena sessions);
speed-rate correlation median per mouse bp01 0.184 -> 0.184, bp02 0.114 -> 0.167, bp04 0.075 ->
0.088, i.e. speed from a two-dimensional track correlates better with the population rate.

Pickle: `build_data_dic.py --dataset mec --neural v2 --tracking v2 --tag v2` -> `data_dic_mec_v2.pkl`
(sbatch job 3660377, pending at the time of writing). Until it lands, `XY_raw` and `HD_raw` in
`data_dic_mec.pkl` are unusable for bp02 and bp04.

## 2026-09-23 -- T1: the QC report (`tracking_qc.py`)

One row per raw SLEAP h5 (maze: `sleap_to_rois.session_files`, 461 LEC + 102 mEC, training days
included; arenas: every h5 in the arena directory, 20 LEC box + 12 mEC big field), head_back in cm
through the session's geometry (maze affine under `--geometry v2`; `openfield_to_xy.session_calibration`
for the arenas, cv2), keyed by SLEAP model (`labels_path` in the h5). Outputs
`data/processed_data/tracking_qc_{dataset}_{maze,arena}.csv` in each tree (`.vN` versions; the
final set carries `jump45cm_*` and the flags `degenerate_geometry` < 0.10 span / `low_span` < 0.60).
`--gate` runs the planted synthetic h5 through the same function: 10/10 checks (spike count exact,
gap lengths exact, low-score fractions exact, bridged / left-NaN counts exact, moving mask 0.99
agreement with the planted speed profile, v1 moving-frame drop within 5 % of truth).

### What v1 (`drop_jumps`, median + 3 MAD of the first bodypart) does, per model

Medians over sessions unless stated; "moving" = smooth-only speed > 5 cm/s.

| dataset / model | sessions | v1 threshold px | v1 drop of moving frames, median | p90 | sessions > 20 % |
|---|---|---|---|---|---|
| mEC labels_test_mec (bp01) | 33 | 13.0 | 1.7 % | -- | 13 |
| mEC labels_test_mec_bp02 | 44 | 4.5 | 62 % | -- | 25 |
| mEC labels_test_mec_bp04 | 25 | 7.7 | 12 % | -- | 11 |
| LEC ah08_implanted | 72 | 1.0 | 77 % | 99 % | 70 |
| LEC ah10_ly05_implanted | 144 | 5.1 | 32 % | 99 % | 76 |
| LEC ly06_ly07_implanted | 145 | 3.3 | 68 % | 78 % | 99 |
| LEC ABCDE_implanted (July) | 99 | 3.1 | 70 % | 98 % | 63 |
| LEC box (labels.v005, 4 MAD) | 20 | 12.8 | 0.4 % | -- | 0 |
| mEC big field (5 models, 4 MAD) | 12 | 9.9-12.2 | 0.3-1.5 % | -- | 0 |

The LEC maze CSVs never had v1 applied (raw affine, see below); the LEC rows say what v1 would do.
The mEC CSVs and `data_dic_mec.pkl` do carry it: 49 of 102 sessions lose more than 20 % of their
moving frames, and the plan's damage table rested on one dead-pinstate session. On the arenas the
mouse moves most of the time, the MAD does not collapse, and v1 is harmless.

### Scores and the conditional gate, per model

`pt<0.5` = head_back point score below 0.5 on tracked frames; `inst<0.5` = instance score; `low` =
either; `off3` = fraction of low-score frames more than 3 cm from the chord between the flanking
good frames; `gated` = fraction the conditional gate removes; `long` = fraction of low-score frames
in runs longer than 15 frames.

| model | pt<0.5 | inst<0.5 median (max) | low frames | off3 | gated | long |
|---|---|---|---|---|---|---|
| mEC bp01 | 0.1 % | 0.9 % (10.7 %) | ~1 % | 4.6 % | 9.7 % | 38 % |
| mEC bp02 | 0.0 % | 0.6 % (100 %, the all-NaN session) | 0.6 % | 8.0 % | 7.5 % | 27 % |
| mEC bp04 | 0.0 % | 1.0 % (3.3 %) | 1.0 % | 8.2 % | 8.4 % | 32 % |
| LEC ah08 | 0.0 % | 0.4 % (4.5 %) | 0.4 % | 14.3 % | 14.8 % | 29 % |
| LEC ah10_ly05 | 0.0 % | 0.1 % (2.7 %) | 0.1 % | 2.7 % | 5.3 % | 38 % |
| LEC ly06_ly07 | 0.0 % | 0.3 % (3.5 %) | 0.3 % | 6.9 % | 8.5 % | 40 % |
| LEC ABCDE | 0.0 % | 0.3 % (2.6 %) | 0.3 % | 5.1 % | 7.3 % | 34 % |
| LEC box | 0.1 % | 1.1 % (2.3 %) | 1.1 % | 8.1 % | 8.5 % | 44 % |
| mEC big field | 0.0 % | 0.1-2.4 % | 0.1-2.4 % | 13-23 % | 12-24 % | 0-32 % |

Point scores are essentially never low; the instance score carries the information. A hard gate
would have removed 100 % of low-score frames; 75-95 % of them sit where the trajectory says the
mouse was. That is why the gate is conditional (D8).

### Motion and the skeleton-shift artefact

| model | disp p99 (cm/s) | disp p99.9 | jumps > 4.5 cm in one frame | of which land within 1 cm of another node's previous position | R1 removals / session |
|---|---|---|---|---|---|
| mEC bp01 | 39 | 290 | 0.14 % of frames | 88 % | 247 |
| mEC bp02 | 40 | 362 | 0.45 % | 95 % | 699 |
| mEC bp04 | 41 | 368 | 0.45 % | 94 % | 706 |
| LEC ah08 | 31 | 304 | 0.17 % | 87 % | 323 |
| LEC ah10_ly05 | 32 | 39 | 0.01 % | 100 % | 20 |
| LEC ly06_ly07 | 35 | 130 | 0.07 % | 92 % | 93 |
| LEC ABCDE | 34 | 75 | 0.03 % | 90 % | 52 |
| LEC box | 31 | 169 | 0.05 % | 64 % | 57 |
| mEC big field | 66-126 | 164-692 | 0.04-0.42 % | 23-55 % | 194-1202 |

(jumps measured in cm on every arena; in the big field, at ~8 px/cm, only a quarter to a half of
the large jumps are skeleton shifts, the rest are fast motion and coarser jitter.)

Found while diagnosing R1: on the mEC maze-2 model 0.6 % of frames, and 0.04-0.16 % elsewhere,
head_back jumps 90-105 px (5-6 cm, 300-400 cm/s) in one frame with a point score of 1.00 and lands
within 11 px of where body_back was on the previous frame, while head_back-body_back stays at its
normal 100 px and head_back-head_mid at 20 px: the whole skeleton slides one segment backwards for
1-6 frames (R1 run length p50 2, p90 4, max 6). Not a whole-frame shift (body_mid does not move with
head_back), not real motion (a mouse does not reach 3 m/s), not low confidence. Body-length checks
are blind to it by construction. R1 catches the jump in; R3 (Hampel 11) removes events up to 5
frames; R1 alone would re-accept after 4 frames (5.4 cm / 1.67 cm per frame). `v2_r1_run_gt5`
counts the longer events: 0 per session (median) on mEC, 4-24 per model on LEC in total. This
is where most of v2's removed moving frames come from, so the plan's "v2 < 2 % of moving frames"
is the wrong criterion; the right one is that the removed frames are these events, which the
`|raw - cleaned|` at R1 frames shows: p50 3.2-3.5 cm, p90 5.4-6.1 cm on bp04 (53-64 % > 3 cm).

### What v2 does (dry run on every session)

| model | moving frames removed, median (p90) | R1 / R2 / R3 / gate removals per session | left NaN, median (max) | speed p90 ref -> v2 (cm/s) | path v1 -> v2 (ratio to smooth-only) |
|---|---|---|---|---|---|
| mEC bp01 | 1.0 % | 247 / 3 / 14 / 75 | 0 (--) | 23.5 -> 23.1 | 0.992 -> 0.984 |
| mEC bp02 | 2.5 % | 699 / 0 / 8 / 31 | 0 | 22.5 -> 22.4 | 0.959 -> 0.979 |
| mEC bp04 | 2.3 % | 706 / 1 / 7 / 40 | 0 | 22.4 -> 22.2 | 0.986 -> 0.976 |
| LEC ah08 | 2.0 % (3.4 %) | 323 / 1 / 14 / 45 | 0 (0.9 %) | 13.4 -> 13.2 | 0.923 -> 0.964 |
| LEC ah10_ly05 | 0.1 % (0.3 %) | 20 / 0 / 0 / 4 | 0 (7.7 %) | 17.4 -> 17.4 | 0.992 -> 0.998 |
| LEC ly06_ly07 | 0.4 % (0.7 %) | 93 / 0 / 3 / 17 | 0 (2.8 %) | 19.5 -> 19.4 | 0.979 -> 0.991 |
| LEC ABCDE | 0.3 % (1.6 %) | 52 / 1 / 3 / 14 | 0 (0.3 %) | 17.9 -> 17.9 | 0.973 -> 0.992 |
| LEC box | 0.3 % | 57 / -- / 3 / 42 | 0 (0) | 15.6 -> 15.5 | 0.999 -> 0.990 |
| mEC big field | 0.4-1.7 % | 194-1202 / -- / 2-30 / 39-506 | 0 (0.6 %) | 41-48 -> 41-48 | 0.996-1.027 -> 0.983-0.998 |

Bridged frames beyond the kept frames' p99 distance from an edge centreline (`bridged_off_maze_frac`,
node centres at 5 / 23.5 / 42 cm from the mask centroids): median 0 on every model, 3.8 % in the
worst session. Body length head_back-body_back: median 5.2-5.9 cm on the maze, 5.5 cm in the box,
6.2-7.0 cm in the big field; implausible (> 2x or < 0.4x the median) 0.0-0.8 %. Out of arena
(> 5 cm): 0.0 % everywhere, one session at 0.1 %.

### Flags

Counts over the 460 readable LEC maze sessions (238 ABCD, 36 ABCDE, 187 training days) and 102 mEC
(final tables `tracking_qc_lec_maze.v1.csv`, `tracking_qc_mec_maze.v1.csv`, `tracking_qc_{lec,mec}_arena.v2.csv`):
`false_drop` 308 / 49, `long_gap` (a raw gap > 2 s) 24 / 16, `coverage` (< 7 of 9 nodes visited in a
task session) 14 / 9, `low_span` (span < 60 % of the maze) 10 / 4, `low_conf` (> 2 % of frames with
instance score < 0.5) 6 / 10, `short` (< 3000 frames) 6 / 2, `degenerate_geometry` (span < 10 %)
1 / 1 (the 597-frame ly06 2025-06-20 12:56 fragment; the all-NaN bp02 session), `whole_session_nan`
0 / 1 (bp02 2024-03-27 14:17, 114 s, no track at all). Under the corrected maze-2 corners no
session with a real track trips `degenerate_geometry`; the Z-order affine would have tripped it on
all 69. The `coverage` sessions are the no-trial sessions of the
registry (ly05 06-20 s3, 06-23 s6, bp01 03-24, 03-25, 04-03, bp02 03-26 and their like): the mouse
sat at 1-6 nodes for a whole session, and v1 then deletes 98-100 % of its few moving frames. The
truncated h5 ly06 2025-06-19 15:40 is `h5_unreadable` and is skipped everywhere, never repaired.
Arena flags: LEC `short` (the 41 s ah10 2025-06-25 09:52 fragment), `low_conf` (ly05 06-25 09:13,
2.3 %); mEC `long_gap` (bp01 03-21 18:14, 7.5 s; bp01 03-25 15:32, 6.6 s), `low_conf` (bp04 03-25
16:36, 2.4 %).

### Constants frozen from the tables (D2)

| constant | value | what the tables say |
|---|---|---|
| point_min, instance_min | 0.5 | low-score frames are 0.1-2.4 % of frames per model; 75-95 % of them are on the trajectory, so the threshold is not where the leverage is |
| v_max_cm_s | 100 | real head_back motion p99 31-45 cm/s on the maze, 66-77 in the big field; the artefact sits at 300-400 cm/s; the cap fires on 0.03-1.6 % of frames and 74-91 % of its large hits are skeleton shifts |
| hampel_window / hampel_thr_cm | 11 / 3 | R3 removes 0-30 frames per session; it exists for events R1 re-accepts |
| spike_thr_cm | 1 | R2 removes 0-3 frames per session; kept, harmless |
| lowscore_thr_cm, chord fractions, long run, path fraction | 1.5 cm, 0.25 / 0.75, 15 frames, 0.9 | gate removes 5-24 % of low-score frames per model; the synthetic shows it keeps curved running paths and removes static mislocalisations |
| max_gap_frames | 120 (2 s) | frames left NaN: median 0 on every model, worst session 7.7 % (a 24 s SLEAP dropout) |
| smooth_window | 5 | kept (D7) |

## 2026-09-23 -- cleaning v2 (`tracking_clean.py`)

Per bodypart, on the cm track, in this order: the conditional gate (low-score runs compared with the
chord between their flanking good frames: short runs off by more than 1.5 cm + 0.25 x chord, long
runs off by more than 1.5 cm + 0.75 x chord or with a path shorter than 0.9 x chord - 1.5 cm, are
removed; the gate runs first so a rejected run is never R1's reference); R1 velocity against the
previous valid frame (100 cm/s x elapsed frames); R2 out-and-back spike (d_in, d_out > 1 cm, net <
half the smaller); R3 Hampel (11 frames, 3 cm); straight bridge of interior NaN runs <= 120 frames
(`np.interp` per run, so a longer run is left whole); centred 5-frame mean with the pre-smoothing
NaN mask re-imposed. Flags, int8 per frame per bodypart: 0 kept, 2 / 3 kept with a low point /
instance score, `1r` removed for reason r and bridged, `2r` removed and left NaN, r in {1 raw NaN,
4 R1, 5 R2, 6 R3, 7 low score off trajectory}. `--clean v2` in `sleap_to_rois.py` (maze, `maze_v2/`,
implies `--geometry v2`, head direction and ROI labels from the cleaned track inverted through the
affine) and `openfield_to_xy.py` (`openfield_v2/`); both write `*_track_flags.csv` beside the CSVs.

Synthetic controls (`tracking_clean_synthetics.py --quick`, through a generated SLEAP-style h5):
29/29, of which control 9 pushes the planted h5, converted to LEC maze-1 pixels, through
`sleap_to_rois.process_one` under `--clean v2` against the real masks: four files written including
the flag file, coordinates equal to `clean_v2` applied directly (2e-14 cm), flags identical, spike
frames labelled from the true position (96 % agree with the truth's ROI, 22 % with the spike's),
every frame of the 200-frame gap `None`, bridged dropout frames all labelled. The others, planted: 50 spikes 30 cm away (all removed, bridged within 0.79 cm of truth; the 1 cm bound
allows the synthetic's instantaneous wall reversals), a 300-frame 40 cm/s run (0 frames lost; v1
deletes 299/300 of it), dropouts of 10 / 100 / 200 frames (bridged whole / bridged whole / left NaN
whole), 50 low-score frames on a curved running path and 50 while still (all kept, flag 2), 10
instance-low frames (kept, flag 3), a static 20-frame low-score run 15 cm off the path while running
and one 6 cm off while still (both removed: 11 + 9 and 17 + 3 frames by the gate + R1), a hard gate
(thresholds 0) removing the on-trajectory frames too, NaN-run lengths preserved through the re-mask
(a plain rolling mean shrinks them), flag round trip, an unplanted node with 0 rule removals, no
unplanted head_back frame removed, missing `point_scores` raising, mismatched score shapes raising.
Two design changes came out of these controls: the gate moved from a Hampel test to the chord / path
test (a 20-frame run's median IS the run) and from after R1 to before it (R1 re-accepted rejected
runs and ate the true frames on the way back).

Equivalence: `sleap_to_rois.py --dataset mec --clean v1 --geometry v1 --compare-existing`: 102/102
CSV triples byte-identical to `maze/`. `--dataset lec --clean none --compare-existing`: 460/460
readable sessions have coordinates equal to the raw affine to 5.7e-14 cm (float printing) with an
identical NaN pattern, confirming the LEC provenance claim of the plan; the ROI files differ in
0.43 % of cells (median; max 5.7 %), all within 0.7 px of a mask edge: the notebook iterated the
mask TIFFs in filesystem order, which is not reproducible (node-first helps ah08 and hurts ly05).
Not worth reproducing; v2 labels ROIs from the cleaned track. Measured cost of D6 on 12 sessions:
0.2-1.8 % of bins change label, node-to-node swaps 0.000-0.095 %.

Writes (2026-09-23): `sleap_to_rois.py --dataset mec --clean v2 --write` -> `maze_v2/`, 102
sessions, 4 files each (`sleap_to_rois_qc_mec_v2.csv`); `openfield_to_xy.py --dataset {lec,mec}
--clean v2 --write` -> `openfield_v2/`, 19 LEC box + 11 mEC big-field sessions
(`openfield_to_xy_qc_{lec,mec}_v2.csv`): per session 55-1775 frames removed by the rules, 100-3508
bridged, 0 frames left NaN except the two bp01 big-field sessions with 7 s SLEAP dropouts (666 and
672 frames); head-direction NaN <= 0.6 %. `sleap_to_rois.py --dataset lec --clean v2 --write` ->
`maze_v2/`, 460 of 461 sessions written with flag files (`sleap_to_rois_qc_lec_v2.csv`); the h5 of
ly06 2025-06-19 15:40 could not be opened. **Resolved the same day (user):** the video is a normal
22 min 11 s (79 851 frames, 258 MB; pinstate 79 850 rows) and the h5 in
`google_drive_backup/behaviour_videos/cohort10/ABCD/SLEAP/implanted/` is intact (7 002 055 bytes,
opens, scores present); the repo's copy was a byte-identical 74 % prefix of it (5 177 344 bytes, a
transfer that stopped early on 2025-08-04; the other 480 LEC h5 copies match the backup's sizes).
The truncated copy was renamed `*.analysis.h5.truncated_2026-09-23`, the intact file copied in
(sha256 equal to the backup's) and the session written: 79 851 frames, 88 frames removed by the
rules, 163 bridged, 0 left NaN, so `maze_v2/` holds all 461. Its copilot v3 arrays (the recday
re-run `--animals ly06 --ephys_folder 2025-06-18`) and the LEC pickle rebuild followed on
2026-09-28, because the first `data_dic_lec_v3.pkl` (job 3660650, 2026-09-23) was built before
those arrays existed and carries "no XY_raw" for ly06_20250618_20250619 s6. LEC head_back under
v2: 0.7-5.5 % of frames unlabelled ('None'), affine residual 0.39-0.55 cm, spans 46-50 cm.

## 2026-09-23 -- GLM V3 NaN path (`glm_analysis_v3.py`, both trees, parity OK)

Measured before the change: one NaN `XY_raw` sample -> 25 NaN speed bins and 27 acceleration bins
(gaussian sigma 3, truncate 4); `compute_distance_to_rewards` is a scalar accumulator, NaN from the
first NaN speed to the end of the session; `_DOWNSAMPLE_AGG` averaged Speed/Acc with `np.mean`;
`np.nan_to_num` on the design matrix then coded all of it as the reference bin. Live today on the
frame-clock mask of 15 LEC v2 sessions (49 runs, median 8.2 s, max 30 s) and 5 mEC sessions. Change:
(a) `smooth_and_calculate_scalar_derivatives` interpolates across NaN for the smoothing only and sets
speed / acceleration back to NaN inside each gap +- 13 bins, positions inside the gap; (b) the
distance accumulator takes `nan_to_num(speed)`; (c) rows with NaN speed leave the fit
(`nf &= isfinite(Speed)`, counted per session); (d) Speed / Acc block aggregation is `nanmean`. All
four are no-ops without NaN. Gates: `glm_v3_synthetics.py` control 12 (planted 3 s gap: local NaN,
finite distances, bit-identical arrays before the gap and beyond the reach, exact row drop) added
beside the existing control 1 (v3 == v2 on a real recday, max|diff| 0). Job 3660306 (full suite,
36 min, ah10_20250616_20250617): **controls 1 and 1b PASS**, i.e. the engine with the NaN path is
bit-identical to v2 on a NaN-free recday for the production matched-13 design and the decile arm;
controls 3-11 pass; control 12 failed on two of my own checks (distance TO the next reward before
the gap legitimately depends on speed inside the gap for legs ending after it; the row count came
from `bin_occupancy`, which exists only for overridden regressors) while its engine checks passed
(speed / acceleration NaN inside the gap + 13 bins and finite outside, 146 NaN bins, distances
finite everywhere, Speed / Acc / distance-from bit-identical before the gap, speed bit-identical
beyond the reach, 14 rows to drop). Fixed (forward-looking arrays compared up to the last reward
before the gap; `cv_results['n_rows']`), resubmitted as job 3660409 `--only-gap`: **control 12
PASS**, 13/13 controls, results in `glm_v3_synthetics.v1.pkl` (the script now writes `.vN`
versions instead of replacing its results pickle). `check_mirror_parity.py`: OK.

## 2026-09-23 -- copilot (`sleap_alignment_copilot.py`)

`--tracking-clean {v1,geom2,v2}` rebinds the two SLEAP roots (`maze/` + `openfield/`, `maze_geom2/`
+ `openfield/`, `maze_v2/` + `openfield_v2/`); anything but v1 needs `--out-tag`; v2 needs `--sync
matched` and sets `PROPAGATE_TRACK_NAN`: a 25 ms bin whose bracketing frames are NaN is written NaN
in `XY_raw` (`untracked_bins`), counted as `n_bins_track_nan` in the sync QC. HD ('nearest') carries
NaN by itself; Locs arrive as 'None' -> 0 from the v2 ROI file.

Stage 3, mEC (2026-09-23): `--dataset mec --metadata --sync matched --sessions all --tracking-clean
v2 --out-tag v3` -> mEC `data/processed_v3/`, 212 sessions, the same clock fits as v2 (4768
frame-clock uncertain bins). Track NaN propagated into `XY_raw`: 16 274 bins (407 s) in 36 of 212
sessions, at most 2394 bins in one (bp04 2024-03-27 19:06, the 60 s SLEAP dropout), against the
4768 bins the frame clock already blanked. Pickle build `--neural v2 --tracking v3 --tag v3` -> job
3660499; validator and `tracking_impact.py --old v2 --new v3`: see below.

Stage 3, LEC (2026-09-23): the same command with `--dataset lec` -> LEC `data/processed_v3/`, 208
sessions (v2 had 210: ah10_20250618_20250619 s5 is the salvaged dead-pinstate session, re-staged
below; ly06_20250618_20250619 s6 was the truncated-copy h5, repaired and re-run on 2026-09-28). Frame-clock uncertain
bins 15 612; track NaN propagated 1658 bins (41 s) in 7 sessions, at most 465 bins (11.6 s,
ly05_20250613_20250615 s4). `restage_salvaged_v3.py --write`: ah10 s5 rebuilt from the `maze_v2/`
CSVs with the salvage's arithmetic (first A_on frame 2556, 50 295 bins, 0 NaN), sidecar
`RESTAGED_V3_ah10_20250618_20250619_5.json`. Pickle build `--neural v2 --tracking v3 --tag v3
--diff-against data_dic_lec_v2.pkl` -> `data_dic_lec_v3.pkl` (job 3660650, EXIT 0, 25 June recdays;
rebuilt 2026-09-28 with the repaired ly06 session, see below). mEC: `data_dic_mec_v2.pkl` (job
3660377) and `data_dic_mec_v3.pkl` (job 3660499), both EXIT 0, in the mEC tree;
`load_data_dic(dataset='mec', version='v3', recday_kind=...)`.

## 2026-09-23 -- mEC v3 arrays: validator and impact

`validate_alignment.py --dataset mec --neural v2 --tracking v3`: 216 session rows, all checks pass
(`alignment_qc_mec_nv2_tv3.csv`). `tracking_impact.py --dataset mec --old v2 --new v3` (212
sessions, `tracking_impact_mec_v2_vs_v3.csv`; v2 = geometry fix with v1 cleaning, v3 = cleaning
v2), medians over sessions per mouse, with the worst session in brackets:

| | bp01 (72) | bp02 (88) | bp04 (52) |
|---|---|---|---|
| position change, p90 over moving bins (cm) | 0.00 (14.9) | 0.08 (12.1) | 0.06 (3.6) |
| bins moved by > 1 cm | 0.5 % (6.7 %) | 1.5 % (11.8 %) | 1.2 % (4.5 %) |
| path length v3 / v2 | 0.990 (0.966-1.39) | 0.998 (0.979-1.94) | 0.992 (0.972-1.16) |
| `XY_raw` NaN, v3 | 0 (3.6 %) | 0 (0.6 %) | 0 (17.5 %) |
| longest NaN run, v3 | -- (19.5 s) | -- (7.7 s) | -- (59.9 s) |
| `Locs_raw` agreement v3 vs v2, tracked bins | 0.997 | 0.994 | 0.993 |
| `Locs_raw` untracked bins, v2 -> v3 | 1.9 % -> 1.8 % | 1.0 % -> 0.7 % | 2.7 % -> 2.3 % |
| `HD_raw` difference p90 (deg) | 0.0 (16.3) | 0.4 (24.0) | 0.2 (7.7) |

The large tails are the fix working, not damage: the sessions where the path grows by 8-94 % and
a tenth of the bins move by more than 1 cm are the sessions where v1 had deleted most of the
locomotion (bp02 2024-04-03 s1 / merged s6, 2024-03-25 s3, 2024-04-03 s2), so v2 (geom2) had
their running bridged by straight lines; v3 restores it. Across the 97 task sessions joined to the
QC, the v1 moving-frame drop fraction correlates 0.65 with the fraction of bins moved > 1 cm and
0.61 with the path ratio; the 41 sessions where v1 dropped > 50 % move 3.4 % of bins (path +4.8 %,
position change p90 1.3 cm during running), the 46 where it dropped < 5 % move 1.0 % (path -1.1 %,
p90 0.0 cm). The 17.5 % NaN session is bp04
2024-03-27 19:06 with its 60 s SLEAP dropout, blank now instead of bridged. Locs agreement
0.993-0.997 over bins tracked in both is the D6 cost measured in advance (0.2-1.8 % of bins); v3
leaves fewer bins untracked than v2 because bridged frames gain a label. (Corrected 2026-09-28:
the first impact run counted the NaN objects that pandas makes of the ROI files' 'None' as tracked
labels, which inflated agreement to 0.99 and hid the untracked fraction;
`tracking_impact_*_v2_vs_v3.v1.csv` are the corrected tables.) The 22 arena sessions: bins moved
> 1 cm median 0.2 % (max 1.5 %), NaN max 0.6 %.

## 2026-09-23 -- LEC v3 arrays: validator and impact

`validate_alignment.py --dataset lec --neural v2 --tracking v3`: 211 session rows, all checks pass
(`alignment_qc_lec_nv2_tv3.csv`). Against the v2 report: `loc_at_goal` mean +0.0001 (min +0.000,
max +0.011; no session worse by more than 0.01, the D6 gate), `poke_loc` within +-0.002;
speed-rate correlation mean +0.004 (medians ah08 0.058 -> 0.074, ah10 0.249 -> 0.273, ly05 0.069 ->
0.062, ly06 -0.042 -> -0.031, ly07 -0.071 -> -0.089; 29 sessions lower by more than 0.01, the
statistic is not gated and is negative for two mice in both versions). `tracking_impact.py
--dataset lec --old v2 --new v3` (209 sessions; v2 = raw affine with no cleaning, v3 = cleaning
v2), medians over sessions per mouse, worst session in brackets:

| | ah08 (42) | ah10 (42) | ly05 (42) | ly06 (41) | ly07 (42) |
|---|---|---|---|---|---|
| position change, p50 / p90 over moving bins (cm) | 0.06 / 0.14 (0.69) | 0.06 / 0.11 (0.12) | 0.06 / 0.11 (0.12) | 0.06 / 0.12 (0.13) | 0.06 / 0.12 (0.13) |
| bins moved by > 1 cm | 0.9 % (2.1 %) | 0.06 % (0.4 %) | 0.02 % (0.5 %) | 0.1 % (0.5 %) | 0.3 % (0.6 %) |
| speed p90, v2 -> v3 (cm/s) | 14.1 -> 14.1 | 18.6 -> 18.6 | 15.0 -> 15.0 | 17.9 -> 17.9 | 19.9 -> 19.9 |
| path length v3 / v2 | 0.80 (0.64-0.99) | 0.92 (0.87-1.00) | 0.90 (0.67-0.99) | 0.89 (0.82-1.00) | 0.88 (0.81-0.99) |
| `XY_raw` NaN, v3 (max) | 9.4 % | 4.5 % | 1.1 % | 0.9 % | 4.8 % |
| `Locs_raw` agreement v3 vs v2, tracked bins | 0.992 | 0.996 | 0.998 | 0.991 | 0.995 |
| `Locs_raw` untracked bins, v2 -> v3 | 0.7 % -> 0.6 % | 0.6 % -> 0.5 % | 0.6 % -> 0.5 % | 4.2 % -> 4.0 % | 5.5 % -> 5.4 % |
| `HD_raw` difference p90 (deg) | 5.8 | 5.7 | 4.6 | 5.8 | 5.7 |

Reading: the LEC v2 track was raw pixels through the affine, so v3 differs from it by the 5-frame
mean everywhere (0.06 cm per bin, 5-6 deg of head direction) and by the spikes it removed (path
8-20 % shorter, most for the ah08 model that has the most skeleton shifts); the GLM-style smoothed
speed does not move. The NaN maxima are the frame-clock mask already present in v2 (ah08 06-16/17
s3 9.4 %). 209 sessions on 2026-09-23; 210 after the ly06 repair (impact `.v1.csv`, 2026-09-28).

## 2026-09-23 -- nothing pre-existing changed (mEC tree)

`sha256sum -c` against the pre-work manifest of the mEC tree after every mEC stage above: 5081 of
5082 listed files unchanged; the one "failure" is the manifest's own `.partial` name, which the
listing caught while it was being written and which no longer exists.

LEC tree, after Stage 3 (2026-09-23): 8575 of 8577 listed files unchanged. The `.partial` artefact
again, and **one real change: `data/processed_data/glm_v3_synthetics.pkl`**, the results pickle of
the GLM V3 synthetic controls, which the pre-existing `glm_v3_synthetics.py` writes with a plain
`pickle.dump` to a fixed name and which the full-suite job 3660306 therefore replaced (the same
eleven controls re-run with the same outcomes, plus control 12). Not a data file, regenerable, but
an overwrite against the rule of this work; the script now writes `.vN` versions (fix below) so the
`--only-gap` job does not do it again. Nothing under `data/raw/`, `data/preprocessed/sleap/maze/`,
`openfield/`, `processed/`, `processed_v2/` or any pickle changed.

## 2026-09-28 -- pickle diffs, the ly06 re-run, and ah10's ABCDE week

`check_data_dic_v2.py --dataset mec --recday-kind merged --against data_dic_mec.pkl` for v2 and
v3 (jobs 3665697 / 3665698): 12 merged recdays, 108 sessions each, `Neuron_raw` identical in
108/108, `Trial_times` identical in 97/97, no session gained or lost `XY_raw`; per-session median
tracking shift against the original pickle bp01 0.00 cm (unchanged), bp02 18.1 cm, bp04 18.2 cm
(p95 19.6-20.2 cm), i.e. the collapsed x restored; `XY_raw` NaN max 2.0 % (v2), 17.4 % (v3, the
60 s dropout session). `data_dic_diff_mec_v{2,3}.csv`.

The ly06 recday re-run (`--animals ly06 --ephys_folder 2025-06-18`) wrote s6 (50 710 bins, 260/260
pulses anchored, 0 track-NaN bins) and reported XY/HD of the other seven sessions identical but
their Locs as "conflicts". Measured: the arrays are string-identical; pandas reads the ROI files'
'None' as NaN (a default NA token), so Locs arrays hold NaN objects and `np.array_equal` calls
NaN != NaN. `save_new` now compares NaN-aware (`_same_array`, numeric `equal_nan` and a string
comparison for object arrays); the seven `*.conflict1.npy` were renamed
`*.spurious_nan_compare_2026-09-28.npy`. The first `data_dic_lec_v3.pkl` (job 3660650, no
tracking for ly06 s6) and its three companions were renamed `*.missing_ly06s6_2026-09-23.pkl` and
the build resubmitted (job 3665699).

ah10's ABCDE week. The sheet lists 12 ABCDE sessions; 6 had their SLEAP h5 in the repo. Copied
from the backup on 2026-09-28 (28 files, every sha256 equal to its source, 3 csv already present
and identical): the 7 `ABCDE_implanted.0{36,37,40,43,47,51,52}` h5 (+csv) that were never
transferred, and the 18 July sessions the user tracked and uploaded on 2026-09-25 with a new
model `ah10_18July_ABCDE.slp` (4 h5 + csv, 72 732-75 914 frames; 4 pinstates, rows = frames - 1;
5 pyControl files incl. an aborted 16:34:10). The sheet has four ah10 sessions on 18 July: ABCD
12:46, ABCDE 14:02, ABCD 15:19, ABCDE 16:36. QC (`tracking_qc_lec_maze.v2.csv`, ah10 only, 101
rows), the new model beside the July model on the same mouse:

| model | sessions | inst < 0.5 median (max) | low-score frames off > 3 cm | gated | disp p99 / p99.9 (cm/s) | jumps > 4.5 cm | v1 drop of moving frames | v2 removed | R1 / session | body (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| ABCDE_implanted (ah10) | 25 | 0.3 % (2.6 %) | 3.8 % | 7.0 % | 34 / 70 | 0.0 % | 0.6 % | 0.1 % | 40 | 5.9 |
| ah10_18July_ABCDE | 4 | 0.2 % (0.4 %) | 6.5 % | 17 % | 32 / 57 | 0.0 % | 16 % | 0.1 % | 26 | 5.7 |

Within range on every column; the gate touches 24 low-score frames per session. Two of the four
18 July sessions carry `false_drop` (v1 would delete 56 % and 22 % of their moving frames).
Cleaning v2 wrote the 11 new sessions into `maze_v2/` (22-100 frames removed by the rules per
session, 36-249 bridged, 0 left NaN). Copilot: the July blocks are six-day sorts
(`2025-07-14_..._2025-07-19_preprocessed`), which `sleap_alignment_copilot.py` could not parse
(two dates at most); it now takes any number of dates and names outputs with all of them, as the
registry, `build_data_dic` and `validate_alignment` already do
(`ah10_20250714_20250715_20250716_20250717_20250718_20250719`); one- and two-date names are
unchanged: re-running the ly06 June recday with the patched script reported all 24 arrays
identical and 0 conflicts. `--animals ah10 --ephys_folder 2025-07-14_2025-07-15 --tracking-clean
v2 --out-tag v3`: 24 sessions aligned into `processed_v3/ah10/20250714_..._20250719/`
(`sync_qc_lec.v3.csv`), every pulse anchored in every session (220-287 of 220-287), 0 uncertain
bins, 0 track-NaN bins, camera 59.991-59.996 fps, at most 5 dropped frames per session; the four
18 July sessions are s16-s19. The July recdays have no neural extraction, so this prepares their
tracking (`maze_v2/`, `processed_v3/`) without putting the ABCDE week into a pickle; that is
`extract_neuron_raw.py --sync fitted` on the July blocks, not requested here.

`validate_alignment.py --dataset lec --neural v2 --tracking v3` on the arrays as they now stand
(`alignment_qc_lec_nv2_tv3.v1.csv`): 223 session rows, all checks pass; the repaired ly06 s6
scores `loc_at_goal` 1.000 on 80 goal checks and `poke_loc` 0.908, every pulse anchored. The
July rows are blank ('-', not 'ok'): no spikes or trial times exist for them yet, and the
validator refuses a vacuous pass.

`data_dic_lec_v3.pkl` rebuilt (job 3665699, EXIT 0, 25 June recdays, 3.82 GB): ly06_20250618_20250619
s6 now carries XY/HD/Locs; the only sessions without `Locs_raw` are the 20 object-box sessions,
as before, and the only session without `XY_raw` is ah10_20250624_20250625 s9 (as in v2).
Checked 2026-09-28: that is the object session of 2025-06-25 15:18:23, a real session with its
own 227 MB video, pinstate and pyControl file in the backup and the repo, and it has simply never
been tracked with SLEAP (the object-vector model's outputs for ah10 that day are the 09:52 fragment
and the 09:53 session, which is the sheet's OTHER object session and is in the pickle). The plan's
"registry stamp override" for it was wrong; it needed tracking, as the 18 July sessions did.

Tracked by the user and uploaded on 2026-09-28 (`implanted/ah10_objvec/`, model
`ah10_obj_25june.slp`): 55 590 frames against 55 589 pinstate rows, scores present, box corners
and both objects (blackcurves, vertring) already clicked on 2026-09-18. Copied into
`data/raw/sleap/ah10/objects/` (h5 + csv, sha256 equal). `openfield_to_xy.py --clean v1` ->
`openfield/` (so the v1 tree is complete too) and `--clean v2` -> `openfield_v2/` (107 frames
removed, 390 bridged, 0 left NaN, box frame, x -1 to 51 cm, y -2 to 50 cm). Arena QC
(`tracking_qc_lec_arena.v3.csv`, 21 rows): instance < 0.5 on 1.5 % of frames, 3 % of low-score
frames > 3 cm off the trajectory, displacement p99 27 cm/s, no jump > 4.5 cm, v2 removes 0.3 %
of moving frames, out of arena 0 %, span 0.99 of the box, body 5.25 cm, no flag. Copilot
(`--animals ah10 --ephys_folder 2025-06-24`): 178/178 pulses anchored, 36 117 bins, 0 uncertain,
`XY_raw`/`HD_raw` for ah10_20250624_20250625 s9 written (object sessions carry no Locs). The LEC v3
pickles of 10:17 were renamed `*.missing_ah10s9_2026-09-28.pkl` and rebuilt (job 3665733, EXIT 0,
11:36): the build log reports no session without `XY_raw`, so every one of the 211 LEC sessions
with a video has tracking in `data_dic_lec_v3.pkl` (v2 had 210). Validator on the v3 arrays
(`alignment_qc_lec_nv2_tv3.v2.csv`): 223 rows, all checks pass. Per-session diff against v2
(job 3665734, `data_dic_diff_lec_v3.v1.csv`): 211 sessions in both, 211 with `XY_raw` (one gained:
ah10_20250624_20250625 s9), `Neuron_raw` identical 211/211, `Trial_times` identical 191/191,
per-mouse median tracking shift 0.01-0.04 cm.

## Not established (2026-09-28)

- The ABCDE week is in no pickle: no July neural extraction exists, and the five July Kilosort
  resumes of 2026-09-23/24 (`resumeks_*`, the user's jobs) all failed after 10-24 GPU hours at
  `kilosort/swarmsplitter.py:96` (`np.NaN`, removed in NumPy 2; the `maze_ephys` env now has NumPy
  2.0.2 while the June sorts ran the same Kilosort 4.0.18 under NumPy 1.x on 2025-07-31). The
  `kilosort_output` folders hold no spikes. Fix options put to the user: a pinned `numpy<2` env
  with Kilosort 4.0.18, the one-line patch, or `maze_ephys_si104` (Kilosort 4.1.7).
- The implementation is uncommitted on top of `pre-tracking-qc-2026-09-23`.
Per-session diff against v2 (`check_data_dic_v2.py --dataset lec --version v3 --against
data_dic_lec_v2.pkl`, job 3665715, `data_dic_diff_lec_v3.csv`): 25 recdays, 211 sessions in both,
`Neuron_raw` identical in 211/211, `Trial_times` identical in 191/191, 210 sessions with `XY_raw`
in both and none gained or lost, `XY_raw` NaN max 9.4 % (the frame-clock mask); per-mouse median
tracking shift 0.01-0.04 cm (p95 0.11-0.14 cm), the 5-frame mean. The one exception is the
repaired ly06_20250618_20250619 s6: median 0.66 cm, p95 16.3 cm, last minute 1.7 cm, and 50 710
bins against v2's 50 660. Its v1/v2 arrays had been built from the notebook's csv export, which
lists tracked frames only; the 75 untracked frames of that video were dropped by row, so the v2
track drifted by up to 1.25 s towards the end of the session. v3 is built from the intact h5,
frame-indexed. No other session was csv-derived (`--clean none --compare-existing` found every
other `maze/` CSV equal to its h5 frame for frame).

LEC manifest re-check after everything above (`sha256sum -c`, 8577 lines): 8575 unchanged and the
same two exceptions as on 2026-09-23, the manifest's own `.partial` name and the
`glm_v3_synthetics.pkl` replaced by the pre-existing script on 2026-09-23; nothing else. The
manifest does not cover `data/raw/`; the 2026-09-28 copies into it were verified by sha256 against
their backup sources at copy time, and the one replaced raw file keeps its old bytes as
`*.analysis.h5.truncated_2026-09-23`.

## Not established (2026-09-23)

- `check_data_dic_v2.py --against` per-session diffs of the three new pickles: submitted
  2026-09-28, see the dated paragraph when it lands.
- The ABCDE week (July 2025) is registered (six-day merged recdays) but has no neural extraction
  and no copilot arrays; its tracking is prepared (QC, `maze_v2/`) and the ah10 July recday aligned
  on 2026-09-28; the pickle inclusion waits for `extract_neuron_raw.py --sync fitted` on the July
  blocks, which was not requested.
- Skeleton-shift events longer than 5 frames are partially re-accepted by R1 + R3 (`v2_r1_run_gt5`);
  an explicit rule was not built because no event longer than 6 frames was seen in 6 sessions.
- The big-field R1 removals (0.2-1.1 % of frames) were not checked frame by frame the way the maze
  ones were.
- Whether any downstream result changes: no GLM has been run on a v2 or v3 pickle.

## Commands

```
# snapshot (done): git tag pre-tracking-qc-2026-09-23; sha256 manifests in both trees
python sleap_to_rois.py --dataset mec --geometry v2 --write                       # maze_geom2/
python sleap_alignment_copilot.py --dataset mec --metadata --sync matched --sessions all --tracking-clean geom2 --out-tag v2
sbatch sbatch_files/mec_stage.sbatch build_data_dic.py --dataset mec --neural v2 --tracking v2 --tag v2
python validate_alignment.py --dataset mec --neural v2 --tracking v2
python tracking_qc.py --gate; python tracking_clean_synthetics.py --quick
python tracking_qc.py --dataset {mec,lec} --arena maze --write                    # maze_ephys env
python tracking_qc.py --dataset {mec,lec} --arena arena --write                   # histology env (cv2)
python sleap_to_rois.py --dataset mec --clean v1 --geometry v1 --compare-existing
python sleap_to_rois.py --dataset lec --clean none --compare-existing
sbatch sbatch_files/mec_stage.sbatch ../glm_v3_synthetics.py                      # controls 1, 1b, 12
```
