# mEC integration — survey and plan

**Written 2026-09-18.** Every number below was measured on disk, not assumed. Sibling tree:
`/ceph/behrens/adam_harris/Taskspace_abstraction_mEC` (not a git repo). Behaviour backup:
`/ceph/behrens/adam_harris/google_drive_backup/behaviour_videos/`. Mac repos uploaded 2026-09-17 to
`Taskspace_abstraction_mEC/mEC_local_upload/{mEC_analysis,mEC_preprocessing}`.

The question this answers: how far did the mEC (cohort 7: bp01, bp02, bp04) preprocessing get, how
many neurons are there, and what does it take to bring that data into the LEC `data_dic` schema so
both regions run through the same analysis code.

---

## 0. Decisions taken (user, 2026-09-17)

| decision | choice |
|---|---|
| mice in scope | all three; bp02 yield is probably too low for anything serious |
| recday unit | **2-day merged sorts**, matching LEC |
| where it lives | **this repo, separate pickle** `data/processed_data/data_dic_mec.pkl`; fold into one dataset later (possibly with PFC) |
| anatomy | **later.** No DiI in the bp mice. `BP3` histology is **bp03**, not bp04. bp01's mEC tissue was lost at extraction |
| naming | LEC registry convention `bp01_20240321_20240323` (YYYYMMDD). Old DDMMYYYY artefacts are superseded |

---

## 1. What exists on the mEC side

### 1.1 Recordings

OpenEphys binary, `Record Node 101`, **Neuropixels 1.0** single shank (384 ch, 20 µm pitch, y 0–3820 µm;
`neuropixels_1_params/probe_layout.tsv`). Several mice recorded into one folder; mouse identity is the
probe serial (`data/metadata/serial_numbers_subjects.csv`: 22175704781 bp01, 22175704251 bp02,
22175704721 bp04). Probe slot is not stable across recordings. Rsync TTLs are on every probe stream
(`events/Neuropix-PXI-100.Probe?-AP/TTL`, line 1, 484 rising edges in a 20 min session).

| mouse | AP recordings | recording days | date range | sorted days |
|---|---|---|---|---|
| bp01 | 311 | 34 | 2024-03-15 → 04-25 | 8 |
| bp02 | 418 | 42 | 2024-03-15 → 05-03 | 10 |
| bp04 | 103 | 11 | 2024-03-15 → 03-28 | 6 |

Raw: 26 TB in `data/raw/ephys/` (465 folders). Unsorted blocks include bp01's 5-state task
(2024-04-04 → 04-08, `ME_mastertask`) and ABCDE block (04-09 → 04-30, `ABCDE_v1`).

### 1.2 Sorting — done, with the LEC pipeline

`code/merge_preprocess_kilosort_spikeinterface_mEC.py` differs from the LEC script only in paths, the
probe params dir (`neuropixels_1_params/` vs `dense/`) and `amplitude_median_thres` (40 vs 20). Same
Kilosort 4.0.18, `Th_universal=9, Th_learned=8, do_CAR=False`, same IBL destriping, same
`get_single_units` gate. 36 jobs ran 2025-08-10 → 08-20, all `All done!`, logs in
`code/preprocessing/slurm_outputs/`. Output layout is byte-for-byte what `recday_registry.py` reads:
`data/preprocessed/ephys/{mouse}/{d1}_{d2}_preprocessed/kilosort_output/QC_single_units.npy` +
`recording_sessions_in_concat.csv`.

**QC single units (the `Neuron_raw` rows):**

| mouse | 2-day merged recdays | units (merged) | single days | units (single) | KS `good` (single) |
|---|---|---|---|---|---|
| bp01 | 4 | **180** (48, 48, 44, 40) | 8 | 392 | 648 |
| bp02 | 5 | **17** (3, 4, 2, 4, 4) | 10 | 66 | 292 |
| bp04 | 3 | **54** (15, 18, 21) | 6 | 148 | 359 |
| total | 12 | **251** | 24 | 606 | 1299 |

LEC reference: 2851 unit-recordings over 25 two-day recdays, ~114 per recday. mEC per merged recday:
bp01 ~45, bp04 ~18, bp02 ~3.

The amplitude threshold is not the lever. Re-scoring the stored `quality_metrics.htsv` with the LEC
value of 20: bp01 +0 (180), bp02 +5 (22), bp04 +4 (58) on the merged blocks. Relaxing
`presence_ratio` to 0.8 as well: 183 / 27 / 66. The merged-vs-single halving is the presence-ratio
requirement across two days, which is the same requirement LEC lives with.

Merged blocks: bp01 `2024-03-21_2024-03-23`, `03-24_03-25`, `03-28_03-31`, `04-02_04-03`;
bp02 `03-19_03-21`, `03-24_03-25`, `03-26_03-27`, `03-28_03-31`, `04-02_04-03`;
bp04 `03-21_03-23`, `03-24_03-25`, `03-27_03-28`.

**Hygiene:** 28 of 36 files in `sbatch_files/` call `code/merge_preprocess_kilosort_spikeinterface.py`,
which does not exist in that tree (only `_mEC.py` does). They ran under an older filename; resubmitting
them fails instantly. No phy curation on any KS4 output.

### 1.3 Alignment — never reached the spikes

`mEC_aligned/` (2025-09-13, 208 files) is **bp01 only** and contains **no neural data**: `HD_raw`,
`XY_raw`, `Locs_raw` (string labels, object dtype), `Trial_times` `(n_trials, 4)`, `Tasks`,
`Tasks_abstract`, `Wake_sleep`. It was built by `ephys_alignment.ipynb` cell 31 against the **old
Kilosort 3** session boundaries in `data/cohort7/Sorted/bp01/*_combined_all`, and the alignment call is
literally `align_to_first_A(..., ephys_mat=None)`. Missing sessions: `24032024_25032024` awake 3 and 7,
`02042024_03042024` awake 8.

### 1.4 Behaviour and tracking — where it actually lives

In the mEC tree: bp01 only (`mEC_preprocessed/all/`: 214 pyControl, 210 pinstate, 245 SLEAP CSVs).
`data/raw/sleap/*` empty for all three mice. No `MetaData.xlsx - bp02.csv` anywhere.

In the backup, **complete for every sorted date of every mouse:**

| | pyControl `.txt` | pinstate `.csv` | video `.mp4` | where |
|---|---|---|---|---|
| bp01 | 509 | 360 | 358 | `behaviour_videos/BP01_BP02/` |
| bp02 | 507 | 359 | 357 | `behaviour_videos/BP01_BP02/` |
| bp04 | 367 | 318 | 261 | `behaviour_videos/BP03_BP04/` |

Per sorted day: 4 pyControl + 4 pinstate + 4 videos (5–7 on a few days). Task
`METraining_Task_v1` (ABCD, pyControl 1.8.1; state ids `A_on..D_on`), open field as pyControl-2 `.tsv`.
bp01 and bp02 were run **simultaneously in paired boxes** (identical video timestamps, e.g.
`BP01_2024-03-24-105214` / `BP02_2024-03-24-105214`); bp03 and bp04 likewise.

**SLEAP is already run**, single-instance, 8 nodes incl. `head_back`, 60 fps (frame count equals
pinstate line count, e.g. 74439):
`behaviour_videos/cohort7_SLEAP/ABCD_july_2025/` — models `labels_test_mec` (bp01, 67 h5),
`labels_test_mec_bp02` (73), `labels_test_mec_bp04` (46). These cover every sorted date. The older
`cohort7_SLEAP/ABCD/labels.v006.*` files are 102-track multi-instance outputs; do not use them.

Metadata sheets (`data/cohort7/MetaData.xlsx - {bp01,bp04}.csv`) carry the same 12 core columns the LEC
registry joins on — `Behaviour`, `Tracking`, `Ephys` timestamps per session are filled for every sorted
date. bp01 is **maze 1**, bp04 is **maze 2**; bp02's maze is not recorded anywhere.

Camera calibration: `sleap_preprocess_mEC.ipynb` defines pixel corners for bp01 early (≤ 2024-03-25),
bp01 late (≥ 03-28; the 26th "is all over the place"), and bp04 (`maze_2_pixel_pts_mEC`). Mask sets
`code/masks/{mEC_maze1_early,mEC_maze1_late,mEC_maze2}/` (21 TIFFs each). **Nothing for bp02:** no
corners, no masks. No merged recday straddles the bp01 camera switch.

### 1.5 The old bp01 pipeline (superseded, kept for cross-check)

Kilosort 3 (MATLAB) → `preprocessing_pipeline_functions.bin_spikes` per recording → rsync realignment
→ `mEC_local_upload/mEC_analysis/Neurons_rsync_realignment/binnedFR_bp01_{recday}_{n}.npy` (75 files)
→ `mEC_task_correlations.ipynb` cell 9 cuts to sessions. Yield **155 neurons** over 4 bp01 recdays
(40, 50, 26, 39). Mohamady's Dec-2024 flat export of the same is `data/Intermediate_objects/`
(bp01, 2-day pilot 03-24/03-25 as the working day list). The upload supplies `processing.py` and
`mBaseFunctions.py`, so that notebook is runnable again; `preprocessing_pipeline_functions.py` has a
syntax error in `align_to_first_A` (`if ephys_mat != None` without a colon).

### 1.6 Histology

`data/raw/histology/BP{1,2,3}/` raw serial-2P StitchIt stacks (owner `rcini`, 48 GB), unregistered.
BP1 = bp01 (mEC portion lost at extraction), BP2 = bp02, **BP3 = bp03** (not in the ephys cohort).
No bp04 brain in the tree. No DiI. Deferred.

---

## 2. The LEC pipeline, stage by stage, and what blocks each on mEC

| stage | LEC script | mEC status | blocker |
|---|---|---|---|
| ephys metadata | `code/preprocessing/file_merging.ipynb` → `data/metadata/{mouse}_ephys_metadata.csv` | **done** | — |
| sort + merge | `merge_preprocess_kilosort_spikeinterface.py` | **done** | — |
| session sheet | `data/metadata/MetaData-{mouse}.csv` | bp01, bp04 exist | **bp02 sheet must be generated** (§3 A3) |
| rsync trains | `data/processed_data/ephys_rsync_mingyutest/` (cached, provenance unknown) | absent | derive from OpenEphys TTL (`ttl_path` column) |
| spikes → `Neuron_raw` | `extract_neuron_raw.py` | absent | repo root, `EPHYS_MICE`, rsync source |
| pyControl → trialtimes, Task, pokes | `extract_pokes.py` | absent | repo root, `MICE`, pyControl files not in tree |
| SLEAP h5 → ROI CSVs | `sleap_preprocess_lEC.ipynb` cell 13 (has bp01/bp04 branches already) | bp01 done on old h5s; bp02/bp04 none | bp02 corners + masks; h5s not in tree |
| ROI CSVs → `Locs/XY/HD_raw` | `sleap_alignment_copilot.py` | absent | repo root, `MICE`, pinstate not in tree, box pairing |
| assemble | `build_data_dic.py` | absent | `MAZE_EDGES` order vs mEC mask names, `NUM_STATES=4` fine (ABCD only) |
| validate | `validate_alignment.py`, `recday_registry` guards | absent | none once above exist |
| anatomy | `histology_refit/`, `region_assignment.py` | deferred | NP1 geometry, no DiI |

Hard-coded in the LEC scripts and needing a dataset switch: repo root (`extract_pokes.py:30`,
`sleap_alignment_copilot.py:29`, merge script ×5, every sbatch), mouse list (`recday_registry.py:52`,
`extract_pokes.py:54`, `sleap_alignment_copilot.py:51`, `region_assignment.py:50`), `split_recday`'s
two-dates assumption (fine for merged blocks), `VIDEO_FPS=60` (matches), `BIN_MS=25` (matches),
`MAZE_EDGES` tuple in `build_data_dic.py:55` (must match the mEC mask edge names, which use the same
`edge_a-b` convention).

---

## 3. Plan

### A. Consolidate inputs into the mEC tree (no compute)

A1. Mirror the LEC raw layout: copy from the backup into
`Taskspace_abstraction_mEC/data/raw/{pycontrol,pinstate,sleap/{mouse}/maze}/` for the sorted dates
only (bp01 + bp02 from `BP01_BP02/`, bp04 from `BP03_BP04/`; h5 + csv from
`cohort7_SLEAP/ABCD_july_2025/`). Keep the backup untouched. Roughly 200 pyControl, 200 pinstate, 190
h5. Lower-case the `BP0x_` prefixes to match pyControl.

A2. Copy the two metadata sheets to `data/metadata/MetaData-{bp01,bp04}.csv` (LEC filename
convention) and normalise dates.

A3. **Generate `MetaData-bp02.csv`** from bp01's rows: same dates, same ephys folders (bp02's serial
appears in the same recordings), `Tracking` from the bp02 pinstate/video timestamps, `Behaviour` from
the bp02 pyControl filenames, `Structure` from each pyControl file's `active_poke` line (the registry
already treats pyControl as ground truth; `recday_registry.pycontrol_active_poke`). Write a QC listing
every row's source.

A4. Fix the 28 broken sbatch filenames (or delete them; the sorts exist).

### B. Tracking → ROI CSVs

B1. bp01 and bp04: run the existing registration cell (`sleap_preprocess_lEC.ipynb` cell 13 already
has both branches) on the July-2025 single-instance h5s. Decide whether the existing 245 bp01 CSVs
(from the older h5s) are kept or regenerated; regenerate, so all three mice come from one SLEAP model
generation.

B2. bp02: **pick 4 maze corners from a first frame** (`mEC_local_upload/mEC_preprocessing/get_first_frame.ipynb`)
and **draw a 21-ROI mask set** `masks/mEC_bp02/` — or replace masks with a geometric assignment in cm
space after the homography (nodes on a 3×3 grid over 47 × 47 cm). The geometric route would serve
every mouse and remove the per-camera mask sets; the mask route matches what LEC did. Choose before B1.

### C. Port the LEC preprocessing with a dataset switch (this repo, `code/preprocessing/`)

C1. `recday_registry.py`: add a `DATASET` selector (env var or module constant) that sets
`REPO_ROOT`, `EPHYS_MICE`, sheet path pattern. Default `lec`; every existing number unchanged.
Equivalence control: with `DATASET=lec`, `blocks_for`, `block_to_recday`, `qc_unit_count` return
byte-identical results for all 25 LEC recdays.

C2. New `extract_rsync_from_ttl.py`: read `ttl_path` from `recording_sessions_in_concat.csv`, take the
stream that belongs to the mouse, rising edges only, write
`data/processed_data/ephys_rsync/{mouse}/rsync_timestamps_{date}_{time}.npy` in the shape
`extract_neuron_raw.first_rsync_sample` expects. Assert pulse count against the pyControl `rsync`
list per session (484 vs pyControl's count for 2024-03-24_10-52-32 is the first check). Watch the
TTL-sample vs continuous-first-sample offset (`466833` vs `603172` in that recording) — the LEC bug
doc notes this offset is 0 on some probes and 3000–9000 samples on others.

C3. `extract_neuron_raw.py`, `extract_pokes.py`, `sleap_alignment_copilot.py`: read root and mouse
list from the registry instead of module constants. Box pairing for pinstate: `(bp01, bp02)`,
`(bp03, bp04)` — verify whether paired pinstates carry two rsync trains as LEC's do (LEC pairs did;
the 484-pulse count above suggests mEC may not).

C4. `build_data_dic.py --dataset mec` → `data/processed_data/data_dic_mec.pkl`, plus
`norm_neurons_dic_mec.pkl` etc. Both registry guards run first (unit count against
`QC_single_units.npy` of the date-matched block; `Task` against pyControl). Expected: 12 recdays,
251 units, 8–9 ABCD sessions per recday, the open-field/objectvector session per recday present as
neural data with `num_trials=0` (never renumber sessions).

C5. `validate_alignment.py --dataset mec` → `alignment_qc_mec.csv`; the `loc_at_goal` threshold of
0.60 was set from LEC's aligned/misaligned separation and should be re-measured, not assumed.

C6. `glm_analysis_v3.load_data_dic(dataset='mec')` (new option, default unchanged) and `--dataset`
on the runners. Mirror to `mFC_data/code/`, `check_mirror_parity.py` exit 0.

### D. Synthetic gate and first look

Run the mEC `data_dic` through `elasticnet_v5_synthetics`-style planted-cell controls via the same
door as the data, then the cheapest real analysis (spatial ratemaps or GLM V3 on bp01) to see whether
the units look like anything. Report per mouse; bp01 is 72% of the units, so any pooled mEC claim is a
one-mouse claim until bp04 is in.

### E. Deferred

Registration (no DiI; NP1 single-shank geometry in `probe_refit.py`), sorting the unsorted bp01 blocks
(each day ≈ 3–18 h GPU + 250–570 GB cache; ceph at 97%), disk cleanup (12.8 TB of spikeinterface
`binary/` caches, half redundant now that merged is the unit; 9.7 TB old concatenated `.dat` in
`data/cohort7/Raw/`), folding `data_dic_mec` into one dataset with LEC and PFC.

---

## 4. Open questions

1. bp02 masks and corners are manual work for 17 merged units. Do it, or geometric ROIs for all mice?
2. Which analysis is first once `data_dic_mec.pkl` exists — GLM V3, El-Gaby's notebooks via the flat
   exporter, or ratemaps? Determines whether pokes and HD are needed in the first cut.
3. Unsorted bp01 blocks (5-state task, ABCDE): in scope for this integration or a separate job?
4. Disk: which caches may be deleted, and when.
5. Were bp01/bp02 and bp03/bp04 on a shared camera sync line (pinstate with 2× pulses) or separate?

---

## 5. Reference: QC re-score on the merged blocks (2026-09-18)

`isi_violations_ratio<0.1, amplitude_cutoff<0.1, firing_rate>0.1, presence_ratio>0.9, sd_ratio<3`,
then `|amplitude_median| > 40` (shipped) vs `> 20` (LEC), vs `> 20` with `presence_ratio>0.8`.

| block | clusters | amp>40 | amp>20 | amp>20, pr>0.8 | median amp |
|---|---|---|---|---|---|
| bp01 03-21_03-23 | 366 | 48 | 48 | 48 | 43.4 |
| bp01 03-24_03-25 | 362 | 48 | 48 | 48 | 43.3 |
| bp01 03-28_03-31 | 487 | 44 | 44 | 45 | 65.1 |
| bp01 04-02_04-03 | 361 | 40 | 40 | 42 | 43.5 |
| bp02 03-19_03-21 | 277 | 3 | 5 | 5 | 32.4 |
| bp02 03-24_03-25 | 342 | 4 | 4 | 5 | 33.1 |
| bp02 03-26_03-27 | 298 | 2 | 3 | 4 | 32.2 |
| bp02 03-28_03-31 | 310 | 4 | 6 | 8 | 36.1 |
| bp02 04-02_04-03 | 303 | 4 | 4 | 5 | 29.4 |
| bp04 03-21_03-23 | 356 | 15 | 17 | 18 | 48.9 |
| bp04 03-24_03-25 | 356 | 18 | 19 | 21 | 52.4 |
| bp04 03-27_03-28 | 370 | 21 | 22 | 27 | 54.3 |

---

## 6. Implementation log, 2026-09-18

Everything below was run today; numbers are measured. Nothing was deleted or overwritten;
superseded outputs were renamed in place (`data/processed_*_2026-09-18/`).

**Decisions added by the user today:** both recday kinds (merged default, single available);
bp02 gets tracking only if it comes free; first consumer GLM V3 on bp01; never delete, copy or
rename copies; bp02's own metadata sheet was uploaded (`data/cohort7/MetaData.xlsx - bp02.csv`);
the poke-alignment method from the social videos should be used for maze geometry.

### 6.1 Stage A -- inputs consolidated (`code/preprocessing/mec_consolidate_inputs.py`)
410 files copied into the LEC raw layout under the mEC tree, manifest at
`data/metadata/consolidation_manifest.csv`. Every sorted day of every mouse has 4 pyControl + 4
pinstate + 4 SLEAP h5 (a few days 5-7); 102 of 103 SLEAP/pinstate stamps pair, the one
pinstate without a video (bp01 2024-03-23-160016) is an aborted recording with no sheet row.
Corrected copies of the 28 broken sbatch files are in `sbatch_files/fixed/`.

### 6.2 Registry switch (`code/recday_registry.py`)
`TSA_DATASET=mec` (set by `--dataset` before import in every stage script) selects root, mice
and `QC_GATE`. New `recday_dates` (1 or 2 dates), `recday_kind`, `all_recdays(kind)`,
`.tsv` fallback and a 60 s hand-typed-stamp tolerance in `pycontrol_file`,
`metadata_stamp_discrepancies`. **LEC equivalence: every registry return identical for all 25
recdays and 211 sessions** (baseline JSON captured before the edit). `check_mirror_parity.py`
exit 0. mEC: 12 merged + 24 single recdays, 251 / 606 QC units, 108 task sessions per kind, 0
unresolved pyControl files; 5 sheet stamps were 1-20 s off the filename and resolve by the 60 s
rule; every hand-typed `Structure` agrees with pyControl's `active_poke`.

### 6.3 Stage B1 -- SLEAP h5 -> ROI CSVs (`code/preprocessing/sleap_to_rois.py`)
Headless copy of the notebook cell, same numerics (first-bodypart MAD threshold reused for
all, affine to 47 cm, masks on raw pixels). 102 sessions written to
`data/preprocessed/sleap/maze/{mouse}/`, QC at `data/preprocessed/sleap/sleap_to_rois_qc_mec.csv`.
head_back unlabelled 3.5 % (bp01), 4.1 % (bp02), 9.5 % (bp04); two videos with no labels are
aborted recordings absent from the sheets.

### 6.4 Rsync from TTL (`code/preprocessing/extract_rsync_from_ttl.py`)
**The mEC PXI sync line carries both paired boxes' rsync trains** (bp01+bp02, bp03+bp04): an
ABCD recording has ~2x the edges pyControl sent (484 vs 247), and on 2024-03-24_10-52-32 the
484 edges are 243 bp01 + 241 bp02 + 2 coincident + 2 neither. `Rsync_aligner` cannot match
that; a pairwise-difference histogram (20 ms bins, 3-bin neighbourhood because the ~12 ms
drift straddles bin edges) picks each mouse's train out. 107/107 task recordings matched,
pulses paired min 0.967 median 0.992, residual rms median 0.039 ms max 0.243 ms, first
pyControl rsync observed in every recording. Per-box clock slopes: bp01 -7 ppm, bp02 -15 ppm,
bp04 -11 ppm. Written for every recording in `data/processed_data/ephys_rsync/{mouse}/`
(element 0 = ephys sample of pyControl rsync[0]); QC `ephys_rsync_qc.csv`.

### 6.5 Trial-times, Task_data, pokes (`code/preprocessing/extract_pokes.py`)
`--write-trialtimes` creates the files LEC's untracked writer made. Its conventions were
recovered by dry-checking against all 191 LEC sessions until **382/382 arrays were identical
and 0 would be created**: no file when the sheet does not call the session ABCD and there is
no A_on (20 Object sessions); an empty `(0,)` float64 trialtimes when the sheet says ABCD but
no trial completed (12 sessions, 3 with no A_on at all); float64 trialtimes and int64
Task_data otherwise. `parse_pycontrol` now also reads pyControl-2 `.tsv`. LEC `--validate-only`
unchanged: 191 checked, 182 exact, 9 empty, 0 failures. mEC: 194 trialtimes + Task_data and
194 poke files (97 ABCD sessions x 2 kinds), 22 open-field sessions skipped, validation 170
exact / 24 empty / 0 failures. The 12 empty sessions completed at most one trial.

### 6.6 Spike binning (`extract_neuron_raw.py --dataset mec --write`, sbatch job 3644105)
Dry run on `bp01_20240324_20240325`: 48 units, 9 task + 12 sleep sessions, every session
anchored, the open-field session running to the end of its recording. A first submission
(3644076) died on the missing output directory; fixed. `sbatch_files/mec_stage.sbatch` is the
runner for these stages.

### 6.7 Tracking alignment (`sleap_alignment_copilot.py --dataset mec --metadata --sync matched`)
Three lessons, each costing a rename:
1. `--dataset` in argv[1] made the script miss `--metadata` and run the retired ephys-time
   path for all mice (552 files) -> `data/processed_LEGACY_ephystime_path_2026-09-18/`. Mode
   flags are now found anywhere in argv and the legacy path refuses `--dataset`.
2. The LEC first-edge + 60 fps method ran clean (540 files, 180 sessions), and
   `validate_alignment` passed all 194 sessions (loc_at_goal bp01 1.00, bp02 0.93-0.99, bp04
   0.86-0.98). But the poke calibration (6.8) showed a quarter of bp02/bp04 pokes a whole
   node from their port. **The maze-2 camera runs at 59.948 fps** (13 sessions, 59.946-59.951);
   under 60 fps that is 61-66 frames, ~1 s, of drift by the end of a 20 min session. The
   maze-1 camera runs at 59.995 fps (5-10 frames). -> `data/processed_firstedge_60fps_2026-09-18/`.
3. A single fitted line is not enough either: videos drop frames (bp04 2024-03-24 s4 loses 16
   frames at 14.6 min; two other sessions 3-5). `fit_frame_clock` now anchors every matched
   rsync pulse and interpolates between anchors, with a shift search for the segment after a
   drop; `frames_at` maps pyControl time to frames. Smoke outputs of the two intermediate
   versions are `processed_matched_linear_smoke_` and `processed_matched_piecewise_smoke_`.
The LEC method is untouched as the default (`--sync first_edge`). `sync_qc_mec.csv` records
fps, anchored fraction and jumps per session.

### 6.8 Poke calibration (`code/preprocessing/poke_calibration.py`, method of the social S6 notebook)
Node centroids from `head_back` at poke entries, radius 0.292 x pitch, bridges at edge
midpoints named as `build_data_dic.MAZE_EDGES`. Results on the fitted clock:

| mouse / epoch | sessions | pokes | pitch px | closed loop | frame-wise agreement with masks (nodes only) | centroid vs mask centroid, median px |
|---|---|---|---|---|---|---|
| bp01 early | 13 | 5602 | 335 | 0.996 | 0.971 (1.000) | 10.4 |
| bp01 late | 15 | 5847 | 343 | 0.996 | 0.973 (1.000) | 8.4 |
| bp02 on maze-2 masks | 34 | 14051 | 301 | 0.80 (first-edge clock) | 0.981 (1.000) | 8.1 |
| bp04 | 21 | 8609 | 306 | 0.987 | 0.948 (1.000) | 9.5 |

Adjacency of consecutive node visits is 1.000 in every epoch (6k-15k transitions). **bp02
shares bp04's camera geometry**: its frame-wise labels against the maze-2 masks agree 98 %
(nodes 100 %) over 2.5 M frames and its centroids sit 8 px from the mask centroids. bp04's
masks and corners are therefore valid for bp02 and are what `sleap_to_rois.py` used.

### 6.9 Still to run / open
- Finish the matched-clock tracking for all mice, re-run `validate_alignment` and the poke
  calibration (`--write`), then `build_data_dic.py --dataset mec` under sbatch,
  `load_data_dic(dataset='mec', recday_kind='merged')`, and GLM V3 on bp01.
- Open-field / object sessions: spikes and pyControl kept (neural-only sessions, as LEC's
  Object sessions); no tracking yet. Big-maze SLEAP outputs exist in the backup for all three
  mice; an open-field XY/HD stage on the same fitted clock is proposed (user to confirm).
- bp01 2024-03-23 s6 (5 min, 1 trial) has a pulseless pinstate: no tracking, spikes kept.
- `glm_analysis_v3.load_data_dic(dataset=, recday_kind=)` added in both trees; a data_dic
  holding both kinds now requires the caller to choose. `_RECDAY_RE` accepts single-day names.

## 7. Implementation log, 2026-09-18, second pass: open field, objects, tight clocks on both datasets

User decisions this pass: add open-field/object tracking (mEC sorted dates: 5 open-field + 6
object sessions; LEC: 20 object sessions), align tracking/neural/rsync as tightly as possible on
**both** datasets, mark object positions by a click notebook (identity labels, any number of
objects), LEC corrected data as `data_dic_lec_v2.pkl` beside v1, mEC open field is a 120 cm square
placed above the peg maze (pegs not visible), LEC open field is a square box on the maze under the
same camera.

### 7.1 LEC alignment defects found (all measured; none visible to the v1 validators)
| | |
|---|---|
| camera fps | ah08/ly06 box 59.95 (drift ~1 s per session under 60 fps); ah10/ly05/ly07 59.99-60.00 |
| first pinstate edge = this mouse's rsync[0] | false in 40/108 sessions (paired boxes share the camera line) |
| legacy rsync caches (`ephys_rsync_mingyutest/`) | first pulse is the partner's for ah08/ly06 in all 76 sessions (-61 to -72 ms); probe-dependent reference offsets vs the raw TTL (ah08 ~+4800 samples = 160 ms; ly05 -3352, -6447, -9465 samples across one day's sessions); `ly05_20250620_20250623`'s 8 trains match neither ly05's nor ly06's pyControl, while the raw TTL matches ly05's own at 246/251, 0.04 ms |
| consequence | v1 `Neuron_raw` origins: ah08 shifted 2-4 bins (fitted vs legacy lag +4 bins on 06-13/15), ly05 06-20/23 by an unknown amount (lags hit the +-40-bin search limit) |

### 7.2 Rsync trains from raw TTL, both datasets (`extract_rsync_from_ttl.py`, now also storing the fit)
`rsync_fit_{rec}.json` per recording: `ephys_sample = a * t_ms + b`. mEC 107/107 task recordings;
LEC **211/211**, first pyControl rsync observed in every one, pulses matched median 0.993 (one
recording at 0.55: ah10 2025-06-23_17-35-04, 50 TTL edges for 91 rsyncs), residual median 0.045 ms.
Sheet stamp typo handled: bp02 2024-03-28 Ephys `14-30-34` -> folder `14-30-43`
(`recday_registry.match_ephys_stamp`, 60 s rule, unique candidate).

### 7.3 Fitted-clock spike binning (`extract_neuron_raw.py --sync fitted --out neuron_raw_v2`)
Every spike mapped to pyControl ms through the recording's fitted line, binned on the 25 ms grid;
origin first-A_on bin or rsync[0] (no trials); sleep-box sessions stay legacy. `--sync legacy`
reproduces every existing array byte-for-byte (checked on all 36 mEC recdays and the LEC dry runs).
`compare_with_legacy` reports the population-rate lag between the two versions. **mEC v2: 450
arrays, lag 0 bins (max 1) vs today's TTL-anchored legacy.** **LEC v2 (job 3644210): 509 arrays;
lag new minus legacy per mouse, median over task sessions: ah08 +2, ah10 0, ly05 -8, ly06 -2,
ly07 0 bins** (extremes reach the +-40-bin search limit in low-rate sessions and are noise);
`--sync legacy` reproduced every existing LEC array byte-for-byte
(`neuron_raw_fitted_vs_legacy_lec.v2.csv`).

### 7.4 Tracking on the fitted frame clock
mEC (`--sync matched`, 180 sessions): fps bp01 59.995, bp02/bp04 59.949; 99.6 % of pulses anchored
at the median; 110 sessions have dropped-frame jumps (median 2 frames; bp04 2024-03-25 s0 and
2024-03-27 s0 lose 151 and 85 frames, followed piecewise). Superseded outputs renamed:
`data/processed_firstedge_60fps_2026-09-18/`, `processed_matched_linear_smoke_`,
`processed_matched_piecewise_smoke_`. LEC (`--out-tag v2` -> `data/processed_v2/`, task sessions):
187 sessions on the first pass. Poke calibration on the fitted clock: closed loop bp01 0.998,
bp02 0.998, bp04 0.982; outputs in `data/processed_data/poke_calibration/{mouse}/{epoch}/`.
**Superseded the same evening** by the alignment frame clock and the re-runs of 7.8 and 7.11
(these first-pass outputs are kept as `processed_globalline_2026-09-18/` and
`processed_v2_globalline_2026-09-18/`).

### 7.5 Open field (Stage E)
- Inputs: 12 big-maze SLEAP h5 + 12 pinstate copied (`mec_consolidate_inputs.py` `openfield`
  families; manifest `.v1`). All open-field/object pyControl `.tsv` (rsync only) were already in.
- `arena_calibration.py`: mEC big field from the 13-peg registration, `cv2.calibrateCamera` with
  fx=fy, fixed principal point, k1/k2 only: **reprojection 1.31 px, per-peg error <= 0.27 cm,
  k1 = -0.20** (strong barrel). Scale is the peg plane until the clicked corners set the
  120 cm plane factor. LEC boxes: the maze affine (reprojection 6.8 / 8.9 px on the 4 maze
  corners), distortion to be judged by the straight-wall test. JSONs in `data/metadata/`.
- `openfield_to_xy.py` (h5 -> cm CSVs) and `sleap_alignment_copilot.py --sessions openfield`
  (anchor at rsync[0], no `Locs_raw`) are written; they run once the arena corners are clicked.
- `recday_registry.session_type()`: `ABCD | ABCDE | object | openfield`.

### 7.6 Object marking (Stage G)
`object_frames.py`: per-pixel median of 200 frames + first frame, **7 mEC (1 open-field + 6
object) and 20 LEC sessions written** to `data/processed_data/object_frames/` in each tree; the
bottle is sharp, the mouse averaged away. `object_click.ipynb` (ipympl): arena corners, then
objects with identity labels and optional radius, appended to `object_positions_{dataset}.csv`.
`object_checks.py`: corners -> calibration (plane factor for mEC), objects -> cm, occupancy hole,
straight walls, label consistency, contact sheet. Clicks done the same day; results in 7.9-7.11.

### 7.7 Validators and loaders
`validate_alignment.py --neural v2 --tracking v2`: new `rsync_ok` (matched >= 0.9 and first pulse
observed; a hard failure) and `speed_lag_bins` (speed x population-rate cross-correlation peak;
broad and noisy, report only). mEC v2: all checks pass, rsync matched >= 0.97 everywhere.
`build_data_dic.py --neural --tracking --tag`, new session keys `session_type`, `arena`,
`objects` (v2 / mEC only); `glm_analysis_v3.load_data_dic(version=)` in both trees, parity OK.
LEC `build_data_dic --check` (default sources) built 25 recdays with both guards passing;
**job 3644214: "diff vs disk: 0 difference(s) across 0 recday(s)"** -- the refactored builder
reproduces `data_dic_lec.pkl` exactly. `validate_alignment --dataset lec` (v1): all 212 session
rows pass once the rsync gate counts the TTL pulses as denominator for a recording stopped before
pyControl (ah10 2025-06-23 17-35-04: 50 of 91 pulses, all matched).

### 7.8 Frame clock, second version (`code/preprocessing/frame_clock.py`)
The v2 validator on LEC (`--neural v2 --tracking v2`) dropped `loc_at_goal` for two sessions
(ah08 06-16/17 s3: 0.94 -> 0.75; ly07 06-16/17 s3: 1.00 -> 0.50) and their sync QC showed why:
the global-line fitter had anchored 172/246 and 175/248 pulses with 28-51-frame residuals. A
window-by-window offset trace of the pinstate showed the cause -- **the cameras drop frames in
bursts**: on 2025-06-16 17:25 both cameras lose ~230 frames between 500 and 900 s in many small
steps (both boxes' pulses, so it is the acquisition PC); bp04 2024-03-25 s4 loses 142 frames
inside one 5 s pulse interval; bp04 2024-03-27 s0 82 frames; and ah10 2025-06-13 s3 runs
59.7-60.0 fps block to block. Drops are always negative steps; frames are never added. The
global line plus tolerance pairing lost 30-70 % of the pulses there and produced clocks worse
than plain 60 fps. 77 of 187 LEC and 28 of 202 mEC sessions had a linear residual > 1 frame.

Replacement: a monotone alignment (dynamic programme) of the rsync train onto the edge train.
A pulse is matched or skipped (cost 4); between matched pulses the offset change is free within
1 + 0.4/s frames, cheap when negative (dropped frames: 3 + 0.02/frame) and expensive when
positive (30 + 0.2/frame), so a partner's edge costs a drop to reach and an addition to leave.
The camera rate is the seed (best-supported constant clock), not a least-squares slope that
drops would tilt. On the six diagnosed sessions it anchors 245-255 of 246-255 pulses with
local (leave-one-out) residuals 0.4-1.1 frames, total drops 5-163 frames, all negative. Bins
inside an anchor gap that hides more than 6 dropped frames are written untracked (NaN / Locs 0;
`n_bins_uncertain` in the sync QC), and a session with fewer than 90 % of pulses anchored is
not written. `sleap_alignment_copilot.fit_frame_clock` now points at it; the old fitter is
kept as `fit_frame_clock_globalline`. Regression survey (`frame_clock_survey.py`, jobs 3644443
/ 3644444, `data/processed_data/frame_clock_survey_{lec,mec}.csv`): see 7.11.

### 7.9 Open field, results
mEC (`openfield_to_xy.py --dataset mec --write`, `sleap_alignment_copilot.py --dataset mec
--metadata --sync matched --sessions openfield`): 11 sessions (5 open-field + 6 object; 22
recday-session pairs with the single-day recdays), walls straight after the k1/k2 + corner
homography (RMS 0.2-0.9 cm; bp02 03-26 right wall 3.7 cm, cable on the bottle; bp04 03-25 left
1.4 cm), frame clock 337-377 of 337-377 pulses, XY/HD written under
`data/processed/{mouse}/{recday}/` (no Locs). Validation figure:
`Taskspace_abstraction_mEC/data/processed_data/arena_validation_mec_bigfield.png`.

LEC: the user asked for **corners per session** ("subtle shifts in the positioning of the small
maze arena across sessions"). Measured on the 20 sessions: the box centre moves 0.9-1.9 cm
between sessions; its mean side under the maze affine is 49.9-50.2 cm (a 50 cm box), but every
session is a **4 % trapezoid** (bottom 51.2 cm, top 49.1 cm, left 49.7, right 50.3; angle error
1.3-2.5 deg). A k1 scan (`lec_box_distortion.py`: k1 in [-0.45, 0.15], maze affine refitted on
undistorted corners, scored by wall straightness and corner squareness over all sessions) puts
the optimum at k1 = 0 for both cameras, the corner spread unchanged: camera tilt, not lens
distortion. So each session gets its **own homography from its four clicked corners to the
50 cm square** (`arena_calibration.apply_arena_corners_homography`, `LEC_BOX_SIDE_CM`);
`XY_raw` and objects of that session are in the box frame (origin image-top-left, x right,
y down) and `arena.corners_maze_cm` keeps the maze-frame corners. Walls under the homography:
0.5-1.6 cm RMS, the same as under the affine, so for the 50 cm box that is the floor of the
occupancy-edge method (gate 2 cm for LEC, 1 cm for mEC). Figures (2026-09-20,
`arena_validation_figure.py --dataset lec`): `data/processed_data/arena_validation_lec_openfield_box{1,2}.png`
-- one session's frame with its corners, the frame warped into the 50 cm box, every session's
corners in the maze frame (the shift and the trapezoid), the per-session numbers, and head_back
occupancy per session with the square and the objects. Note the box frame is image-oriented
(origin at the image's top-left corner): box 2's camera views the box the other way up, so its
objects sit at y ~ 38 cm where box 1's sit at y ~ 10 cm; within a mouse this is consistent, across
boxes it is not a room frame. 19 of 20 sessions have tracks
(ah10 2025-06-25 151823 has no SLEAP h5); the affine-frame CSVs were moved to
`data/preprocessed/sleap/openfield_mazeaffine_2026-09-18/`.

### 7.10 Objects, results (`object_click.ipynb` + `object_checks.py --write`)
mEC: 6 bottle positions, one per object session, `object_positions_mec_cm.v1.csv`, 0 problems
(bottles sit in occupancy holes with a ring of investigation). LEC: 40 clicks over 20 sessions:
**four physical objects per recday, two per session** (day 1: `object_1_pyramid` +
`object_2_vertsquare`, day 2: `object_3_blackcurves` + `object_4_vertring`); the second session
of each day moves both by ~37 cm (corner swap) on all five mice; positions sit 10-13 cm or
37-39 cm from the walls in the box frame; 0 problems (`object_checks_lec.v1.json`,
`object_positions_lec_cm.v1.csv` with `x_cm,y_cm` box frame and `x_maze_cm,y_maze_cm`,
`arena_corners_lec_cm.v2.csv`). The per-box corner clicks made before the per-session decision
are kept as `arena_corners_lec.perbox_2026-09-18.csv`; the objects table written with a clobbered
`frame` column as `object_positions_lec_cm.v1.bad_framecol_2026-09-18.csv`.

### 7.11 Frame-clock survey and re-runs
`frame_clock_survey.py` scored both fitters on every session of the matched-clock QC (LEC 187,
mEC 202; `data/processed_data/frame_clock_survey_{lec,mec}.csv`):

| | LEC (187) | mEC (202) |
|---|---|---|
| new fitter failures | 0 | 0 |
| pulses anchored, min / median | 0.981 / 1.000 | 0.988 / 1.000 |
| local (leave-one-out) residual, median / max, frames | 0.43 / 1.35 | 0.42 / 2.37 |
| sessions dropping > 60 frames (1 s) | 5 | 4 |
| sessions with uncertain bins; total uncertain time | 14; 382 s | 10; 119 s |
| old vs new clock, sessions differing by > 10 frames anywhere | 10 | 4 |
| old vs new on sessions where the old fit was clean (resid < 1, >= 0.97 anchored): max diff | <= 4 frames (110 sessions) | <= 4.4 frames (174 sessions) |
| legacy first-edge 60 fps vs new: max diff per session, median / 95 % | 15 / 90 frames | 55 / 67 frames |
| camera fps (seed) | ah08, ly06 59.949; others 59.994-59.995 | bp01 59.995; bp02, bp04 59.949 |

The <= 4-frame differences on clean sessions are single dropped frames placed at the pulse
(new) instead of interpolated across it (old). 73 LEC / 90 mEC sessions show one or two +2-3
frame anchor steps (a pulse whose edge lands a couple of frames late; the next step cancels
it); left as they are. ah08 2025-06-20/23 s3, refused by the old fitter as "truncated
pinstate", is an ordinary 30-frame burst and is now tracked.

Re-runs with the new fitter: mEC `data/processed/` -> renamed `processed_globalline_2026-09-18/`,
LEC `data/processed_v2/` -> `processed_v2_globalline_2026-09-18/`, then
`sleap_alignment_copilot.py --dataset {mec,lec} --metadata --sync matched --sessions all
[--out-tag v2]` (jobs 3644523 mEC, 3644524 LEC) and `poke_calibration.py --dataset mec --write`
(job 3644522). Results: **mEC 202 sessions** (180 task + 22 open-field/object recday-sessions),
every one >= 0.988 of pulses anchored, 10 sessions carry uncertain bins (2.0 min in all);
**LEC v2 206 sessions** (187 task incl. ah08 06-20/23 s3, + 19 object), >= 0.981 anchored, 14
sessions with uncertain bins (6.4 min in all). One LEC object session (ah10 2025-06-24 13:59)
was first refused because the copilot's nearest-pyControl rule picked an aborted 13:58:58 file
with one rsync; under `--sync matched` the copilot now takes the file from the registry's
sheet mapping (the one trialtimes and spikes use) and falls back to nearest only if that is
missing; re-run for that folder (`--animals ah10 --ephys_folder 2025-06-24`): 190/190 pulses.
Poke calibration on the new clock: closed loop 0.997-0.998, mask agreement 0.969-0.980
(`maze_nodes.v1.json`). `validate_alignment`: mEC (`--neural v2`) 216 session rows all pass;
LEC (`--neural v2 --tracking v2`) 211 rows all pass, `loc_at_goal` v2 minus v1 mean +0.002
over 211 sessions with no session worse by more than 0.05 (untracked bins, e.g. inside a
frame-clock gap, are now counted separately from misses; reports are versioned, never
overwritten).

Builds (`build_data_dic.py`, sbatch): the mEC pickle lands in the **mEC tree**,
`Taskspace_abstraction_mEC/data/processed_data/data_dic_mec.pkl` (1.15 GB; with
`norm_neurons_dic_mec.pkl`, `session_inds_dic_mec.pkl`, `tasks_dic_mec.pkl`), because
`PROCESSED_DATA` follows the dataset root; `glm_analysis_v3.load_data_dic(dataset='mec',
recday_kind='merged'|'single')` finds it there (36 recdays in the file: 12 merged / 251 units,
24 single-day / 606 units; 108 sessions each, 106 with `XY_raw`, NaN fraction <= 2 %; session
types 97 ABCD + 5 openfield + 6 object; both registry guards pass, 194 sessions checked against
pyControl; final build job 3644580, load-checks 3644784/3644785 via `check_data_dic_v2.py`).
The LEC pickle stays in the LEC tree as `data_dic_lec_v2.pkl` (3.82 GB, job 3644586, with
`norm_neurons_dic_v2.pkl`, `session_inds_dic_v2.pkl`, `tasks_dic_v2.pkl`; guards pass on 25
recdays / 191 pyControl sessions; only ah10 06-24/25 s9 has no `XY_raw`). Superseded builds are
kept as `data_dic_mec.incomplete_tracking_2026-09-18.pkl` (job 3644571, before the two
tracking fixes below) and `data_dic_lec_v2.pre_salvage_copy_2026-09-18.pkl` (job 3644581).
`load_data_dic(version='v2')` reads the LEC v2 pickle. Per-session v1 -> v2 diff
(`check_data_dic_v2.py`, job 3644791, `data_dic_diff_lec_v2.csv`):

| | |
|---|---|
| sessions | v1 212, v2 211; only in v1: `ah08_20250624_20250625` s10 (no sheet row; a stray legacy array) |
| `Trial_times` | identical in all 191 task sessions |
| `Neuron_raw` | differs in all 211 (fitted clock); population-rate lag v2 minus v1, median per mouse: ah08 +2, ah10 0, ly05 -8, ly06 -2, ly07 0 bins (the +-40 extremes are the search limit in sparse sessions) |
| sessions gaining `XY_raw` | 22 (19 object + 3 no-trial); none lose it |
| tracking shift v2 vs v1, per-session median (cm), median over sessions | ah08 0.36, ah10 0.18, ly05 0.04, ly06 0.67, ly07 0.10; last minute of a session: ah08 0.58, ly06 1.29; 95th percentile within a session: ah08 10.9, ly06 12.4 cm (the 59.95 fps boxes), others <= 1.9 |
| `XY_raw` NaN fraction | median 0, max 0.094 (ah08 06-16/17 s3, uncertain bins) |
| session types | 191 ABCD, 20 object; keys added: `session_type`, `arena`, `objects` |

Sessions without tracking after the full re-runs, and what was done (2026-09-18, evening):

| session | sheet | why untracked | outcome |
|---|---|---|---|
| bp02 2024-03-27 14:38 (bp02_20240326_20240327 s5, 30 trials) | ABCD | nearest pyControl to the tracking stamp was an aborted 14:37:44 start | registry mapping -> tracked |
| bp01 03-24 14:18 (s3), 03-25 13:36 (s7), 04-03 14:39 (s8), bp02 03-26 11:53 (s1); ly05 06-20 15:50 (s3), 06-23 15:31 (s6), ly07 06-25 13:53 (s8) | ABCD, 0 trials started (`trialtimes` shape `(0,)`) | no A_on to anchor on; **the LEC three were untracked in v1 too** | under `--sync matched` bin 0 = rsync[0], as `extract_neuron_raw` already did for their spikes -> tracked (XY length within 1-2 % of the `Neuron_raw` bins: the video ends a little before or after the recording) |
| bp01 2024-03-23 12:43 (0 trials) | ABCD | pinstate has 0 pulses (camera sync line dead for that video) | untracked; the partner's pinstate is another camera's frame clock, so no fallback |
| ah10 2025-06-19 14:35 (ah10_20250618_20250619 s5, 20 trials) | ABCD | pinstate 0 pulses; **salvaged in v1 on 2026-09-06** by poke/track agreement (`salvage_staging/`) | the v1 arrays are copied unchanged into `processed_v2/` with `SALVAGED_FROM_V1_*.json` beside them (60 fps arithmetic, not the fitted clock; loc_at_goal 1.000 on 116 checks) so v2 does not lose the session; the first v2 build without it is `data_dic_lec_v2.pre_salvage_copy_2026-09-18.pkl` |
| bp02 2024-03-19 10:32 (24 trials), ah10 2025-06-25 15:18 (object) | ABCD / object | no SLEAP h5 | untracked until SLEAP is run off-cluster (user) |

Object checks with tracking (`object_checks.py --write`, `.v3`/`.v4`): an occupancy *hole*
does not exist -- head_back occupancy within the object radius is 0.5-4.4x uniform (mice
investigate the objects), so the position test is now the centring of the investigation ring
(centroid of the points within 8 cm of the click): mEC median 1.0 cm, max 2.0 cm over 6
bottles; LEC median 1.3 cm, max 3.7 cm over 36 objects (two flagged at 3.1 and 3.7 cm: ly05
06-24 s4 pyramid, ly07 06-25 s9 vertring; contact sheet `object_contact_sheet_lec.v4.png` for
the eye). Walls > gate: bp02 03-26 s4 right 3.5 cm (cable), bp04 03-25 s8 left 1.7 cm, ly07
06-24 s4 bottom 3.0 cm -- reported, not corrected.

## 7.12 Histology registration of the bp brains (2026-09-19, registration only)
Plan `docs/handoff/MEC_REGISTRATION_PLAN.md`, log and numbers `code/histology_refit/MEC_REGISTRATION.md`.
Identity correction: the histology folder `BP3` is **bp04** (Adam; brainsaw run `AK30_ML__BP1_BP2__BP4`),
so all three ephys mice have a brain; §0 and §1.6 above are superseded on this point. All probes
left hemisphere, NP1.0, no DiI. BP1 (bp01) has lost about a third of its LEFT hemisphere along the
whole forebrain, not only mEC. brainreg with the LEC recipe on the CPU partition (220 GB), QC made
dataset-aware and runnable without a probe fit, and a raw-ephys brain-surface detector
(`code/preprocessing/probe_surface_from_raw.py`) that finds the top 200-600 um of the bank outside
the brain in many recordings, against the sorter's "no channel out". Probe track deferred.

## 8. Sleep in the sorts, what limits the mEC yield, and the LEC July week (2026-09-22)

Three questions were asked of this document: does the mEC data include sleep sessions, would a
sleep-free re-sort yield more neurons, and should the ABCD/ABCDE days be preprocessed. All numbers
below were measured today. Tool: `code/preprocessing/rescore_qc_epochs.py`, which recomputes the
four interval-dependent gate criteria (`num_spikes`, `firing_rate`, `presence_ratio`,
`isi_violations_ratio`) from `sorting_analyzer/sorting/spikes.npy` over any subset of the
concatenated recordings and re-applies the shipped six-criterion gate, holding the waveform
criteria at their stored values. Every run prints a control first.

### 8.1 Sleep is the majority of every sort, in both datasets

Neither merge script filters by session type: the only selection is by date
(`merge_preprocess_kilosort_spikeinterface_mEC.py:49`, unconditional concat loop `:255-268`).
Classification is the pipeline's own rule -- a concat row is task/openfield/object iff a metadata
task row names its exact `HH-MM-SS`, everything else is sleep box (`extract_neuron_raw.session_map`,
`:111-136`) -- and for mEC it is corroborated against the separate
`data/cohort7/MetaData.xlsx - {mouse}_sleep.csv` sheets, which claim every such row but three
sub-2-minute aborted fragments.

| | mEC, 12 merged recdays | LEC, 25 merged recdays |
|---|---|---|
| concatenated | 81.98 h | 171.51 h |
| ABCD task | 32.35 h (0.395) | 65.66 h (0.383) |
| sleep box | **44.02 h (0.537)** | **100.72 h (0.587)** |
| open field / object | 5.58 h (0.068) | 5.09 h (0.030) |

Per merged block the sleep fraction is 0.42-0.61 (mEC) and 0.55-0.62 (LEC). Downstream this is
already visible: 234 of 450 mEC `Neuron_raw` arrays and 289 of 509 LEC ones are `_sb_`.

### 8.2 A sleep-free re-sort would not help: +9 units, and that is an upper bound

Re-scoring the 12 mEC merged blocks over task time only. **Control: scoring the full concat returns
251 units, exactly the shipped count, on 12 of 12 blocks** (and 607 of 607 on the five LEC blocks
of 8.4).

| scored over | units passing the gate | clusters passing `isi<0.1` | median ISI ratio over 12 blocks |
|---|---|---|---|
| full concat (shipped) | **251** | 419 | 1.200 |
| task time only | **260** | 460 | 1.052 |
| sleep time only | -- | 448 | 1.375 |

Sleep epochs are dirtier (1.375 against 1.052) but both sit about 10x above the 0.1 gate, so
dropping 53.7% of the recording moves the median cluster from 1.20 to 1.05 and almost nothing
crosses. The +9 is an upper bound on the real thing, because it holds the sorting fixed: an actual
sleep-free re-sort halves what Kilosort learns templates from and cuts the continuous drift trace
KS4's registration follows between task sessions, which span the same wall-clock either way.

**What does limit the yield is over-merging.** Per merged block, from every `quality_metrics.htsv`
in both trees:

| | clusters/block | median FR | median spikes/cluster | median ISI ratio | pass gate | fail *only* ISI |
|---|---|---|---|---|---|---|
| mEC bp01 | 394 | 3.32 Hz | 82 086 | 0.94 | 180 | 437 |
| mEC bp02 | 306 | 2.42 Hz | 59 941 | 1.28 | 17 | 367 |
| mEC bp04 | 361 | 4.40 Hz | 106 376 | 1.32 | 54 | 475 |
| LEC ah08 | 843 | 0.79 Hz | 19 914 | 0.34 | 501 | 1173 |
| LEC ah10 | 702 | 1.47 Hz | 37 325 | 0.34 | 772 | 1888 |
| LEC ly05 | 552 | 1.52 Hz | 38 273 | 0.34 | 299 | 979 |
| LEC ly06 | 523 | 1.75 Hz | 43 400 | 0.39 | 356 | 1442 |
| LEC ly07 | 511 | 2.21 Hz | 55 519 | 0.37 | 424 | 1520 |

mEC returns roughly half the clusters at 2-4x the firing rate on recordings of the same length, and
the median mEC cluster has an ISI violation ratio near 1.0 -- the Hill estimate saying it is
essentially fully contaminated. 1279 of 4188 mEC clusters (30.5%) pass every criterion except ISI.
Leave-one-out on the gate: dropping `presence_ratio` buys +68, dropping `isi_violations_ratio` buys
+1279; relaxing ISI to 0.2 gives 370 units and to 0.5 gives 705. The levers are therefore the KS4
settings (`nblocks=1`, i.e. rigid whole-probe drift registration across a 3.8 mm NP1.0 shank over
two days; `dminx`; `Th_learned`) and curation -- there is no phy curation on any KS4 output -- plus
bp02's median `|amplitude_median|` of 32 uV against its 40 uV gate. **Not done, by decision.**

### 8.3 Correction to §1.2

The claim at `:69` that "the merged-vs-single halving is the presence-ratio requirement across two
days" is wrong twice. There is no halving: 251 units over **12** merged blocks is 20.9/block against
606 over **24** single-day blocks at 25.2/block, a 17% per-block drop. And presence ratio is not the
mechanism: re-scoring the five LEC blocks of 8.4 one day at a time, the presence-ratio pass count is
flat (ah08 674 over both days against 679 over day 1) while the gate count still drops.

### 8.4 The cost of a longer concat, measured

Re-scoring five LEC 2-day blocks on one day's time at a time isolates the scoring cost of extra
days, with the sort held fixed. Control exact on all five.

| block | shipped | scored on both days | day 1 | day 2 | mean of single days |
|---|---|---|---|---|---|
| ah08 20250616_20250617 | 164 | 164 | 151 | 194 | 172.5 |
| ah10 20250618_20250619 | 162 | 162 | 172 | 174 | 173.0 |
| ly05 20250616_20250617 | 116 | 116 | 119 | 121 | 120.0 |
| ly06 20250618_20250619 | 76 | 76 | 74 | 80 | 77.0 |
| ly07 20250620_20250623 | 89 | 89 | 95 | 99 | 97.0 |
| **total** | **607** | **607** | **611** | **668** | **639.5 (-5.1%)** |

About 5% per extra day at two days. Nothing here constrains six days: drift over six days is a
different regime, and a re-sort changes the sort, not only the scoring.

### 8.5 The LEC July 2025 ABCD/ABCDE week, submitted as 6-day sorts

`data/metadata/MetaData-{mouse}.csv` for July is identical across all five LEC mice: 2025-07-13 has
4 ABCD and no ABCDE; **2025-07-14 through 07-19 have 2 ABCD + 2 ABCDE each**, plus 5 unlabelled
sleep rows per day. 80 ABCD + 60 ABCDE sessions over 5 mice, every row carrying an `Ephys` stamp,
`Include` blank. pyControl confirms 58 `ABCDE_v1` + 78 `METraining_Task_v1` files for 07-13→19, and
the `ABCDE_v1` state set (`A_on`..`E_on`, `prime_A`..`prime_E`) is identical in the LEC and mEC
trees, so the two datasets ran the same task. SLEAP is already run: 208 `ABCDE_implanted.*` h5 under
`data/raw/sleap/{mouse}/maze/`. `ah10 2025-07-18` has ephys but 0 SLEAP h5 and 0 pyControl.

User decision (2026-09-22): sort the six ABCDE days as **one 6-day concat per mouse**, all five
mice, so the interleaved ABCD and ABCDE sessions share a unit set (12 + 12 sessions per block).

| mouse | AP recordings | recording hours | raw AP |
|---|---|---|---|
| ah08 | 53 | 20.46 | 1697.4 GB |
| ah10 | 53 | 20.50 | 1700.2 GB |
| ly05 | 55 | 21.24 | 1762.0 GB |
| ly06 | 54 | 20.85 | 1729.6 GB |
| ly07 | 53 | 20.47 | 1697.6 GB |
| total | 268 | 103.5 | 8.59 TB |

`{mouse}_ephys_metadata.csv` already covers July, so no metadata stage was needed. The sorter needed
no change: `dates = sys.argv[2:-2]` (`:456`) and `dates_string = "_".join(dates)` (`:467-471`) take
any number of dates. `recday_registry` did: `recday_dates` capped at 2 dates, now
`MAX_RECDAY_DATES = 6`, and `recday_kind` returns `single` for one date and `merged` for any more.
**Equivalence control: every registry return identical for all 25 LEC and 36 mEC recdays**, against
baselines captured before the edit (`code/preprocessing/registry_baseline.py`,
`data/metadata/registry_baselines/baseline_{lec,mec}_2026-09-22.json`, 0 differences both ways);
`check_mirror_parity.py` exit 0. The PFC copy `mFC_data/code/recday_registry.py` predates the
dataset switch and is deliberately outside the byte-identical mirror set; it was not touched.

Job spec and why, from `sacct` on the 46 completed June/August sorts: partition `gpu`
(`PreemptMode=OFF`, `MaxTime` 10 days) rather than `a100` or `gpu_lowp`, which are both
`PreemptMode=REQUEUE` -- KS4 has no checkpointing, so a preemption at hour 40 loses the run;
`--gres gpu:1` (KS4 uses one device, the old `gpu:2` was wasted); `--time 5-00:00:00` (2-day blocks
took 10.5-23.2 h for 6.1-7.3 h of recording, this is 20.5 h); `--mem 450G`. The memory figure is
headroom, not a projection: **MaxRSS does not scale with concat length** -- regressed on recording
hours the slope is negative (r^2 = 0.33), because mEC 1-day blocks (2.65-3.98 h) peak at 410-972 GB
while LEC 2-day blocks (6.13-7.27 h) peak at 229-813 GB with no relation to duration inside the LEC
set. Reported MaxRSS also exceeds the node's physical RAM in places (972 GB on the 760 GB node), so
it is counting evictable memmapped pages of the destriped binary, which is why 142 of 143 jobs
completed while requesting 256 G.

Submitted 2026-09-22 (second attempt; see the incident below): jobs **3654420 ah08, 3654421 ah10,
3654422 ly05, 3654387 ly06, 3654388 ly07**,
`sbatch_files/merge_{mouse}_2025-07-14_to_2025-07-19.sbatch`. Recipe otherwise identical to June
(KS4 4.0.18, `Th_universal=9`, `Th_learned=8`, `do_CAR=False`, IBL destriping,
`amplitude_median_thres=20`) so the blocks are comparable. Output ~8.59 TB of `binary/` plus
~0.28 TB of `kilosort_output/` against 28 T free; the 25 June blocks are untouched and all five
July output paths were confirmed clear before submission. **Results pending.** The open risk is
KS4's `nblocks=1` rigid drift registration across six days of a chronic NP2.0: if units split or
smear rather than survive, `nblocks=5` is the first thing to try.

**Incident, 2026-09-22, and two traps worth keeping.** The first submission
(3654055-3654061) died in 2.5 minutes on
`FileNotFoundError: data/raw/ephys/2025-07-19_14-51-59`. That recording had been drag-dropped by
mistake into the previous recording's folder, so it sat at
`data/raw/ephys/2025-07-19_13-59-26/2025-07-19_14-51-59/`. It is a real session: ABCD with
behaviour `145203` for ah10, a sleep row for ah08 and ly07, and it holds only ProbeA/B/C, which is
why just three mice reference it. Moved back to `data/raw/ephys/2025-07-19_14-51-59`; all three
`continuous.dat` sizes match their metadata `dat_size` to the byte (ProbeC 29066830848 ah08,
ProbeB 29070160896 ah10, ProbeA 29069939712 ly07), the parent `13-59-26` kept its own intact
`Record Node 104`, and spikeinterface opens the recovered folder at 21.03 min / 384 ch. A full
sweep now finds **0 metadata rows without a raw folder** for any mouse on any date.

1. The sbatch template printed `All done!` unconditionally, so a python traceback still left SLURM
   recording **COMPLETED**. The five July sbatch files now check the exit code and `exit $rc`.
   For the older files, the job state is not evidence -- read the `.err`.
2. `merge_preprocess_kilosort_spikeinterface.py:477-489` never overwrites: if the output path
   exists it appends `_2`. But `recday_registry.blocks_for` globs `*_preprocessed`, so a
   `..._preprocessed_2` block is **invisible to the registry**. The three crashed jobs had already
   created their output directory (holding only `recording_sessions_in_concat.csv`), so those
   were renamed to `..._preprocessed.failed_missing_rawfolder_2026-09-22` -- outside the glob,
   nothing deleted -- before resubmitting. Check the target path is absent, not merely that the
   script "won't overwrite", before any re-run.

Still to do once the sorts land: rsync from TTL, `extract_pokes` (already 5-state capable --
`STATE_NAMES` includes `E_on` at `:56-58` and the state-id map is read per file from the pyControl
header at `:75-77`, which matters because `ABCDE_v1` and `METraining_Task_v1` assign different ids
to the same state; `trialtimes` becomes `(n_trials, 6)`), tracking on the fitted frame clock,
`build_data_dic` with `NUM_STATES = 4` at `:67` made per-session from `session_type()`, and a
`data_dic_lec_v3.pkl` carrying a byte-identical June control against v2.

Not in scope and unchanged: mEC ABCDE (156 sessions, bp01+bp02 only, bp04 left the rig 2024-03-28;
no maze SLEAP exists past 2024-04-08 though video and pinstate do); the mEC 2024-04-04→04-08 block,
which pyControl calls `ME_mastertask` and is a **third task**, not ABCDE -- it carries `A2_on`,
`A_on_first`, `prime_A_first`, `tone2`, `tone3` on top of A-E and needs its own trial parser; and
LEC June 09-12 (4 dates, 5.23 TB, unsorted).

## 9. Paths

| what | path |
|---|---|
| mEC sorted blocks | `Taskspace_abstraction_mEC/data/preprocessed/ephys/{mouse}/{d1}_{d2}_preprocessed/` |
| mEC ephys metadata | `Taskspace_abstraction_mEC/data/metadata/{mouse}_ephys_metadata.csv` |
| mEC session sheets | `Taskspace_abstraction_mEC/data/cohort7/MetaData.xlsx - {bp01,bp04}.csv` |
| behaviour + pinstate + video | `google_drive_backup/behaviour_videos/{BP01_BP02,BP03_BP04}/` |
| SLEAP h5 (use these) | `google_drive_backup/behaviour_videos/cohort7_SLEAP/ABCD_july_2025/` |
| masks | `Taskspace_abstraction_mEC/code/masks/{mEC_maze1_early,mEC_maze1_late,mEC_maze2}/` |
| old bp01 pipeline + data | `Taskspace_abstraction_mEC/mEC_local_upload/mEC_analysis/` |
| old preprocessing functions | `Taskspace_abstraction_mEC/mEC_local_upload/mEC_preprocessing/*.py` |
| LEC stage scripts | `code/preprocessing/{extract_neuron_raw,extract_pokes,sleap_alignment_copilot,build_data_dic,validate_alignment}.py`, `code/recday_registry.py` |
| LEC data facts | `docs/handoff/README.md`, `docs/BUG_ly05_recday_mismatch.md` |
