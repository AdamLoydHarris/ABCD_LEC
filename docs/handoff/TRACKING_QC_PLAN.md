# Plan: SLEAP tracking QC and a corrected cleaning (both datasets), rebuild the pickles

**Status: superseded (2026-09-23).** Implementation started 2026-09-23 with a revised order and four
changed decisions (maze-2 geometry hotfix first; QC on every session before freezing constants;
conditional confidence gate; `Locs_raw` from the cleaned track); the record of what was measured and
built is `code/preprocessing/TRACKING_QC.md`. The context below stands, except that the LEC maze
CSVs are the raw affine (see the correction paragraph) and the mEC maze-2 affine was degenerate.

**Original status: handoff, not started (user, 2026-09-20).** Written after the measurements below;
nothing in it has been implemented. Scope and constants were agreed with the user (full scope,
2 s gap limit, 100 cm/s cap). Until it runs, `XY_raw`/`HD_raw` in every pickle carry the v1
cleaning described under Context, and any pickle with NaN positions (LEC v2, mEC) will break GLM
V3's speed, acceleration and distance regressors downstream of the first gap (T4).

## Context

Question from the user (2026-09-20): what QC exists for the SLEAP tracking itself — noise, jumps,
low confidence? Measured answer: the cleaning in `code/preprocessing/sleap_to_rois.py`
(maze sessions) and `openfield_to_xy.py` (open field), inherited numerically from the original
mEC notebook:

**Correction (planning agent, 2026-09-20, measured on one maze session per LEC mouse): the LEC
maze CSVs in `data/preprocessed/sleap/maze/` are the raw SLEAP pixels through the maze affine with
NO cleaning at all** — max difference from `to_cm(raw)` 7e-15 cm, NaN pattern identical to the
h5, head direction = arctan2 of those columns. `sleap_to_rois.py` has never run on LEC
(`geometry()` raises for `lec`; `LEC_MAZE1/2` are defined but unused). So the table below and the
MAD damage apply to **mEC only** (`data_dic_mec.pkl`); LEC v1 and v2 `XY_raw` carry raw
single-frame spikes, no smoothing, with SLEAP-missing frames (0.0–0.3 % of cells) linearly bridged
by the copilot's `sample_at_frames`, and `HD_raw` keeps those NaN (0.6 %). Neither dataset has a
confidence gate or a per-session report. Mapping for the LEC branch of `geometry()`: ah08, ly06 →
`LEC_MAZE2` / `masks/lEC_maze2`; ah10, ly05, ly07 → `LEC_MAZE1` / `masks/lEC_maze1`.

mEC cleaning today (`--clean v1`):

| check | today |
|---|---|
| jumps | `drop_jumps`: NaN the landing frame of any frame-to-frame displacement > median + 3 MAD (4 MAD open field) of the displacement distribution; threshold from the FIRST bodypart reused for all |
| gaps | `interpolate_smooth`: linear interpolation, no length limit |
| noise | centred 5-frame rolling mean |
| confidence | not read; the h5 has `point_scores` (nodes × T), `instance_scores` (T), `tracking_scores` |
| body plan, out-of-arena, speed plausibility, per-session report | none (indirect checks only: location at goal pokes, mask agreement, wall straightness) |

**The jump rule is mobility-dependent and, on maze sessions, wrong.** Mice are mostly still on the
maze, so median + 3 MAD collapses to 0.3–1.1 px/frame and the rule deletes the locomotion:

| session | frames dropped | of frames moving > 5 cm/s | of still frames | running speed p90, current vs smooth-only | position error p90 (max) during running |
|---|---|---|---|---|---|
| ly05 maze 2025-07-14 11:46 | 24 % | **97 %** | 15 % | 12.2 vs 16.2 cm/s | 1.4 cm (4.1) |
| bp01 maze 2024-03-23 12:43 | 32 % | 68 % | 18 % | 21.6 vs 23.5 | 0.7 cm (3.5) |
| ah08 maze 2025-07-17 11:20 | 12 % | 58 % | 6 % | 19.1 vs 20.6 | 0.5 cm (11.3) |
| ah08 object 2025-06-24 13:38 | 0.3 % | – | – | unaffected | unaffected |

Path length falls 6–8 % against a smooth-only track. Low confidence is rare (< 0.1 % of head_back
points below 0.5; 0.4–1.2 % of frames with instance score < 0.5) and displacement at those points
is 3–10× the median, so a confidence gate is cheap and right. Body length head_back→body_back is
stable (p1–p99 54–121 px, never > 2× median): no swaps. Longest NaN run after today's cleaning
2.8 s, all from the false drops. `Locs_raw` is unaffected (ROI labels come from the raw pixels);
`XY_raw`, `HD_raw` and everything derived from them are affected, including GLM V3's `speed` and
`acceleration` (`smooth_and_calculate_scalar_derivatives`: gaussian σ = 3 bins on `XY_raw`, then
`np.gradient`).

**User decisions (2026-09-20):** full scope — QC stage + corrected cleaning + rebuild both pickles;
gaps interpolated up to **2 s**, longer left NaN; speed cap **100 cm/s**.

**Rules:** nothing deleted or overwritten (new dirs / versioned names); LEC v1 and the current v2
outputs stay byte-identical; `--clean v1` (default) reproduces today's CSVs exactly; every new piece
gated by synthetic controls that accept AND reject; heavy compute under sbatch
(`sbatch_files/mec_stage.sbatch`, maze_ephys env); numbers with semantics in an .md.

## Approach

### T1. `code/preprocessing/tracking_qc.py` (new) — the standing per-session report
Reads the raw h5 (`sleap_to_rois.load_tracks` plus the three score datasets) for every maze and
open-field session of a dataset (session lists: `sleap_to_rois.session_files`,
`openfield_to_xy.openfield_sessions`/`find_h5`); cm geometry from `sleap_to_rois.geometry` +
`affine` (maze) or `openfield_to_xy.session_calibration` (open field). Per session, per bodypart
where it matters (head_back always):
- confidence: fraction of point scores < 0.5 and < 0.3 per bodypart, fraction of frames with
  instance score < 0.5; raw NaN fraction;
- motion: displacement p50 / p90 / p99 / max in px and cm/s; speed p50 / p90 raw vs cleaned;
- what the **current** rule does: fraction dropped, split into moving (> 5 cm/s) and still frames;
  gap statistics after cleaning (n, longest, count > 30 frames, > 120 frames);
- what the **v2** rule does (T2), same statistics; longest run left NaN;
- stuck plateaus (< 0.3 px/frame for ≥ 120 frames): count, longest, mean score during them;
- body-length consistency: head_back–body_back p1 / p99, fraction > 2× or < 0.4× median;
- out-of-arena: fraction of frames > 5 cm outside the 47 cm maze / 50 cm box / 120 cm field;
- flags: `false_drop` (current rule removes > 20 % of moving frames), `low_conf` (> 2 % of frames
  with instance score < 0.5), `long_gap` (any gap > 2 s), `body_len` (> 1 % implausible),
  `out_of_arena` (> 1 %).
Outputs: `data/processed_data/tracking_qc_{dataset}[.vN].csv`, one figure per mouse (sessions ×
metrics), and one example figure per dataset of raw vs current vs v2 tracks over a running bout.
**Gate** (`--gate`): a synthetic track (random walk with still periods) with planted single-frame
spikes to far points, planted dropouts of known lengths and planted low-score points; the QC must
count exactly the planted spikes and gap lengths, and report the current rule's moving-frame drop
fraction within 5 % of the planted truth. ~1 min per session; both datasets under sbatch.

### T2. Cleaning v2 — `--clean v2` on `sleap_to_rois.py` and `openfield_to_xy.py`
Default `--clean v1` is byte-identical to today. v2 writes to sibling dirs
`data/preprocessed/sleap/maze_clean2/` and `openfield_clean2/` (same file names), nothing
overwritten. In `sleap_to_rois`:
- `load_tracks` returns the scores as well (`point_scores`, `instance_scores`);
- `confidence_gate(df, scores)`: point score < 0.5 → NaN that point; instance score < 0.5 → NaN
  the frame (all points);
- `drop_jumps_v2(df, nodes, px_per_cm, fps)`: per bodypart, in cm: (a) speed cap — a frame whose
  displacement from the previous VALID frame exceeds 100 cm/s × Δt (Δt counts the skipped frames)
  is NaN; (b) out-and-back spike — d_in and d_out both > 100 cm/s × Δt while the two-frame net
  displacement < half of d_in → the middle frame is NaN; the threshold is physical, so it never
  moves with the mouse's mobility (the failure of v1);
- `interpolate_smooth_v2(df, nodes, max_gap_frames=120)`: linear interpolation only across gaps ≤
  120 frames (2 s, user), longer gaps stay NaN; the centred 5-frame mean is applied on the
  interpolated track with `min_periods=1`, and frames adjacent to a NaN run are smoothed over the
  available neighbours only (as today), so no NaN spreads;
- `head_direction` unchanged (NaN where any of its four points is NaN); ROI labels unchanged.
`openfield_to_xy` calls the same functions with the same constants (the 4-MAD multiplier is
gone in v2). Both scripts record `clean_version` and the constants in their QC CSVs.
**Gates:** (i) equivalence — `--clean v1` reproduces every existing cleaned CSV byte-for-byte
(sample of 6 sessions interactively, whole cohort under sbatch, `cmp`); (ii) synthetic — planted
spikes all removed, a planted 40 cm/s run loses 0 frames, planted dropouts ≤ 2 s interpolated and
> 2 s left NaN, planted low-score points removed; (iii) rejection — v1 on the same synthetic must
delete most of the planted run (the known failure), otherwise the synthetic is not testing the
right thing.

### T3. Thread the new tracks through
- `sleap_alignment_copilot.py`: `--clean {v1,v2}` (default v1) selects `SLEAP_PATH` /
  `SLEAP_PATH_OPENFIELD` = `maze_clean2` / `openfield_clean2`; NaN positions pass through
  `sample_at_frames` (NaN-aware) into `XY_raw` / `HD_raw`; `Locs_raw` unchanged. Runs:
  mEC `--sync matched --sessions all --clean v2 --out-tag v2` → `data/processed_v2/`;
  LEC `--sync matched --sessions all --clean v2 --out-tag v3` → `data/processed_v3/`
  (~13 s per session: mEC 202 + LEC 206 ≈ 45 min each, sbatch).
- `build_data_dic.py`: mEC `--neural v2 --tracking v2 --tag v2` → `data_dic_mec_v2.pkl`
  (mEC tree); LEC `--neural v2 --tracking v3 --tag v3 --diff-against data_dic_lec_v2.pkl` →
  `data_dic_lec_v3.pkl`. `load_data_dic(dataset='mec', version='v2')`, `version='v3'` for LEC.
- Validators on both (`validate_alignment --tracking v2|v3`); `check_data_dic_v2.py --against`
  for the per-session diff.

### T4. Downstream NaN handling (required, not optional) and the impact report
`XY_raw` will hold NaN runs (> 2 s gaps), and **already does** in `data_dic_lec_v2.pkl` and
`data_dic_mec.pkl` for the frame-clock uncertain bins written on 2026-09-18 (14 LEC sessions,
up to 9.4 % of a session). Read on 2026-09-20: GLM V3's `smooth_and_calculate_scalar_derivatives`
runs `gaussian_filter1d` on `XY_raw` (NaN spreads across the kernel → speed/acceleration NaN in
and around every gap), `compute_distance_to_rewards` builds a cumulative sum of speed (one NaN
makes every later value NaN → `distance_from/to_reward` garbage for the rest of the session), and
the design matrix is then `np.nan_to_num`'d (line 1948), silently zeroing all of it. So any pickle
with NaN positions currently breaks four regressors downstream of the first gap. Fix in
`glm_analysis_v3.py` (both trees, parity): (a) in `smooth_and_calculate_scalar_derivatives`,
when NaN is present, interpolate linearly across NaN for the smoothing only, then set speed and
acceleration back to NaN inside the gaps (± the kernel's 3σ); (b) `compute_distance_to_rewards`
accrues `np.nan_to_num(speed)` (no distance during untracked bins, documented); (c) the regressor
NaNs are then handled where the design matrix is built — bins with NaN speed are masked out of the
fit rather than zeroed (check how `nan_to_num` at 1948 interacts with the existing row selection;
if masking rows is not possible without changing results for NaN-free sessions, zero them but
count and report them). The branch only triggers on NaN, so NaN-free sessions are bit-identical
(gate: `max|diff| = 0` on one LEC recday against the current engine). Synthetic: a track with a
planted 3 s gap must give finite speed and distances outside the gap, NaN inside, and unchanged
values far from it. Impact report (`tracking_qc.py --impact`): per session, `XY_raw` new vs old —
position difference p50 / p90 during moving frames, speed p90 and path length old vs new,
fraction of bins NaN; per mouse summary; written into `TRACKING_QC.md`. A GLM V3 re-run on the
new speed regressor is Stage D and the user's call.

## Design notes from the planning agent (2026-09-20), to take into implementation

- **Shared module `code/preprocessing/tracking_clean.py`** used by both scripts, the QC and the
  synthetics: `load_scores(h5)`, `clean_v1` (thin wrapper, no numeric change), `clean_v2(raw_px,
  nodes, scores, to_cm, fps, cfg)` returning the cleaned cm track plus a per-rule per-bodypart
  report, `gap_runs`, `interpolate_limited`, `smooth_remask`, the rule functions, and one `V2`
  dict of constants copied into every QC row. **User constants override the agent's:**
  `v_max_cm_s = 100`, `max_gap_frames = 120` (2 s); keep `point_min = instance_min = 0.5`,
  `spike_thr_cm = 1.0`, Hampel window 11 / 3 cm, smoothing window 5.
- **v2 order: cm first**, so every threshold is physical and geometry- and mobility-independent
  (for the affine this commutes with interpolation and the 5-frame mean exactly; for the bigfield
  homography + undistortion the difference is sub-mm over ≤ 30 frames — pin both with a control).
- **Rules:** R1 velocity gate against the previous VALID frame (`|p_i − p_j| > v_max (i−j) dt`);
  R2 out-and-back single-frame spike (`d_in, d_out > 1 cm`, net two-frame move < 0.5 cm); R3
  Hampel residual (`|p_i − median(p_{i−5..i+5})| > 3 cm`, NaN-aware, `min_periods 6`) for 2–5
  frame excursions and spikes next to a gap. Known limit, to be stated: R1 re-accepts an excursion
  once `D/(n dt) < v_max` and then eats true frames on the return; the QC column
  `v2_r1_run_gt5` counts such runs. Freeze `v_max` only after the QC's `disp_cm_p999` on gated
  frames is seen below it.
- **Interpolation:** per NaN run with `np.interp`, only runs ≤ 120 frames — not
  `pandas.interpolate(limit=)`, which fills the first N frames of a longer gap. Leading/trailing
  NaN stay NaN (v1 filled them with `limit_direction='both'`).
- **Smoothing:** `rolling(5, center, min_periods=1)` then re-impose the pre-smoothing NaN mask
  (pandas otherwise shrinks every gap by 2 frames a side); the 2 frames beside a gap are averaged
  over 3–4 samples — accepted and documented.
- **Modes:** `--clean {v1,v2,none}`; `none` is the LEC provenance mode (raw affine, write
  refused) and `--compare-existing` checks bytes / max-abs-diff / NaN pattern against the file in
  place, exit 1 on any difference. Cohort equivalence: mEC 102 files `--clean v1`, LEC 461 maze
  files `--clean none`, 31 open-field files `--clean v1`.
- **Copilot:** `--tracking-clean {v1,v2}` with `set_tracking_clean()` rebinding the two SLEAP
  roots and setting `PROPAGATE_TRACK_NAN`; in `_save_matched`, bins whose bracketing frames are
  NaN become NaN in `XY_raw` (HD via nearest already carries NaN; Locs untouched); refuse
  `--tracking-clean v2` without `--out-tag`. The salvaged ah10_20250618_20250619 s5 (dead
  pinstate) has to be re-staged from the `maze_v2` CSV with
  `salvage_tracking_offset.build_tracking_arrays` at the f0 in its json, or copied again and flagged.
- **Synthetic controls (`tracking_clean_synthetics.py --quick`, 8):** QC counts exact; v1
  equivalence (mEC bytes, LEC `none` max diff 0); v2 positive (50 spikes + 10 short excursions
  removed and re-interpolated to < 0.5 cm, a 40 cm/s 300-frame run loses 0 frames, dropouts
  ≤ 120 frames bridged and longer ones left NaN with exact lengths, low-score points/frames gone);
  rejection in both directions (v1 deletes > 50 % of the run AND v1 removes all spikes on a
  still-only track); smoothing edge (gap lengths preserved; fails with the re-mask off);
  commutation; copilot NaN propagation; guards (missing `point_scores` raises, no writing into
  `maze/`, `--tracking-clean v2` needs `--out-tag`). The generator writes a real h5 (tracks,
  scores, node_names, track_occupancy) so the synthetics enter through `process_one`.
- **Impact (`tracking_impact.py`):** per session position difference p50/p90 on moving bins,
  speed p90 and path length old/new, NaN fractions, HD NaN, `locs_agree` (≥ 0.999 expected; LEC
  v1 ROI files have empty cells where the h5 is NaN vs `"None"`), salvaged flag; then the
  validators and `check_data_dic_v2.py --against`.
- **GLM V3 NaN behaviour — two readings, to be measured, not argued:** the agent finds
  `apply_onehot` zeroes non-finite rows (= reference bin) and `compute_decile_edges` ignores
  them, so a gap costs ~46 bins of speed regressor (Gaussian spread 12 bins a side + 1) and is
  otherwise benign; my reading of `compute_distance_to_rewards` is that its cumulative sum turns
  NaN after the first NaN speed and stays NaN for the rest of the session. Settle it with the
  planted-gap synthetic in T4 before deciding whether the engine needs a change.
- Timings: ~5 s per h5 for cleaning + CSVs; QC ~4 s/session (LEC 481 sessions incl. training
  days ~40 min, mEC 114 ~10 min); copilot ~13 s/session (206 + 202 ≈ 45 min each).

## Files

| file | change |
|---|---|
| `code/preprocessing/tracking_clean.py` (new) | shared cleaning rules, constants, `load_scores` |
| `code/preprocessing/tracking_clean_synthetics.py` (new) | the 8 controls |
| `code/preprocessing/tracking_impact.py` (new) | old-vs-new impact report |
| `code/preprocessing/tracking_qc.py` (new) | T1 report, gate, impact |
| `code/preprocessing/sleap_to_rois.py` | LEC branch in `geometry()`; `--clean {v1,v2,none}`, `--compare-existing`; output root by mode (`maze_v2/`) |
| `code/preprocessing/openfield_to_xy.py` | same flags; `openfield_v2/` |
| `code/preprocessing/sleap_alignment_copilot.py` | `--tracking-clean`, `set_tracking_clean`, NaN mask in `_save_matched` |
| `code/glm_analysis_v3.py` (+ `mFC_data/code/` mirror, parity) | NaN-aware speed / distance path (T4); bit-identical for NaN-free sessions, gated |
| `code/preprocessing/TRACKING_QC.md` (new), `docs/handoff/MEC_INTEGRATION.md`, `docs/handoff/README.md` | write-up and pointers |

## Reuse
`sleap_to_rois.{load_tracks, affine, to_cm, head_direction, roi_labels, geometry, session_files}`,
`openfield_to_xy.{openfield_sessions, find_h5, session_calibration}`, the copilot's
`sample_at_frames` / `set_output_tag` / `--out-tag`, `build_data_dic --tracking/--tag`,
`check_data_dic_v2.py`, `validate_alignment.py`, `mec_stage.sbatch`.

## Commands, in order
1. `tracking_qc.py --gate`; `sleap_to_rois.py --clean v2 --gate` (synthetics + v1 equivalence on 6 sessions).
2. sbatch: `sleap_to_rois.py --dataset {mec,lec} --clean v1 --write --equivalence` (whole-cohort byte check, nothing written) then `--clean v2 --write`; `openfield_to_xy.py --dataset {mec,lec} --clean v2 --write` (histology env).
3. sbatch: `tracking_qc.py --dataset {mec,lec} --write`.
4. sbatch: copilot mEC `--clean v2 --out-tag v2`, LEC `--clean v2 --out-tag v3`.
5. sbatch: builds (`data_dic_mec_v2.pkl`, `data_dic_lec_v3.pkl`), validators, `check_data_dic_v2.py --against`.
6. `tracking_qc.py --impact`, `TRACKING_QC.md`, docs pointers.

## Verification
- v1 equivalence: 0 bytes differ over every existing cleaned CSV of both datasets.
- Synthetics: T1 counts exact; T2 accept (spikes gone, run intact, gaps ≤ 2 s bridged, > 2 s NaN,
  low-score points gone) and reject (v1 deletes the planted run).
- Cohort numbers expected: current-rule moving-frame drop 50–97 % on maze sessions → v2 < 2 %;
  gaps > 2 s rare (count them); out-of-arena and body-length flags ~0.
- Validators pass on the new tracking (`loc_at_goal` unchanged by construction: `Locs_raw` from
  raw pixels); v1 `data_dic_lec.pkl` and v2 pickles untouched (checksums).
- Impact report written; GLM NaN handling verified by its synthetic if touched.
