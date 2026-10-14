# Event-locked kernels (Tsao 2018 / Kanter 2025) in LEC vs PFC — handoff

**Written 2026-10-02, 17:05 UTC.** For a session that picks this analysis up with no prior context.
The approved plan is `~/.claude/plans/glittery-wandering-wadler.md`; everything needed is restated
here. Read `.claude/CLAUDE.md` (repo rules) and `docs/handoff/SPIKE_CLOCK_V4.md` (the spike clock)
first.

## 1. The question

Do LEC and PFC neurons ramp with a spread of time constants locked to task events, as Tsao et al.
2018 found in rat LEC? Events: maze entry (recording start), the first A, every reward A (= trial
start) and every reward; in rest, rest-box entry. Compare PFC (El-Gaby mFC deposit) with the LEC
sub-regions and MEC. Then read the fits as a state-space model (SSM): ramps as a bank of leaky
event traces (a Laplace transform of the past), peaked lag cells as its readout, and Mamba's
input-dependent step as a clock that can stretch into goal progress or reset at events.

References in the repo root:

- Tsao 2018: `s41586-018-0459-6.pdf` and the supplement `41586_2018_459_MOESM1_ESM.pdf`.
- Kanter 2025 (Science 388 eadr0927): `science.adr0927-8.pdf` and `science.adr0927_sm.pdf`.
- The user's note: `SSMs, Mamba and music boxes.pdf` (2026-09-24).

Three things carry time and are kept apart throughout:

| quantity | definition | measured by |
|---|---|---|
| locked clock (Tsao) | a kernel of time since an event that repeats across repetitions of the event | predicting held-out sessions |
| drift (Kanter) | slow change unique to each session | within-session readouts minus the locked part |
| boundary shift (Kanter) | abrupt population acceleration at an event, carried by a different ensemble each time | trajectory acceleration, boundary-response decoding |

The existing population results are all leave-one-session-out, so they measure the locked clock
only:

- LDA, seconds r: 0.54 in LEC against 0.26 in PFC.
- MLP (`nn_time_decoder`): r 0.69-0.81 on two ah08 recdays.

## 2. Decisions taken by the user (2026-10-01 and 02)

| topic | decision |
|---|---|
| kernel shape | shape-free event kernels are primary, classified ramp up / ramp down / bump / other; Tsao's stepwise GLM is a side table |
| kinematics | speed and \|accel\| pass through an exponential bank; event-locked only if the event block beats it held-out |
| why \|accel\| | a leaky filter of signed acceleration equals (v - leaky v) / tau, a linear mix of speed and the speed bank, so it adds nothing; signed instantaneous acceleration stays in the nuisance |
| rest | rest-entry kernels vs maze-entry kernels per unit; intrinsic timescale in rest vs wake tau; offline co-activity ordered by wake tau / lag |
| SSM tests | clock (seconds / distance / progress, alpha); reset vs superposition; readout width vs lag; tau spectrum per region |
| session start | recording start, from full sessions re-extracted on the v4 clock; the first A is the swap anchor |
| sleep staging | spike-based proxy in both datasets: SWS-like = dense population DOWN states, Kanter's 25 ms rule |
| objects | the 20 LEC object sessions are in (novel objects, no food; the day's second session swaps the corners) |
| extensions | day-scale ramps, mEC descriptive, poke-exit anchor (LEC), preceding-task memory in rest |
| LEC data | `data_dic_lec_v4.pkl` content (v4 spikes + v3 tracking) |
| MEC | ENTm units on the LEC probes now; the bp cohort after its re-sorts; both descriptive |
| amplitudes | cheap version: KS amplitudes per unit per recording, for the day-scale ramps only |
| rest < 2 min | OpenEphys misclicks; excluded |
| probe-flagged sessions | the 4 v4 sessions with residual probe failures stay in; no exclusion arm |
| El-Gaby's 20% rule | diagnostic only (section 4) |
| PFC XY | converted to cm at load from a transform table; the deposit is never modified |
| partition | `gpu_lowp` for CPU-only jobs while `cpu` is degraded (no `--gres`) |

## 3. Process rules for this analysis

- **Milestones, with a stop after each.** Report after each milestone and wait for the user's
  go-ahead before the next:
  - M1: WP0 data preparation + WP1 synthetic gate.
  - M2: Stage 1 + reset test + Kanter part 1.
  - M3: Stage 2 tau + Tsao / Kanter single-unit tables.
  - M4: rest.
  - M5: SSM tests.
  - M6: extensions (day clock, objects, poke exit, bp mEC).
- **Commit and push** (user, 2026-10-02) at the checkpoint "WP0 smoke tests pass + quick
  synthetics run end to end", on `main`, `git push origin main`. No Co-Authored-By trailer or
  "Generated with" footer (CLAUDE.md). Scope (user chose this after being shown the dependency
  chain):
  - **This analysis:**
    - `code/event_kernels.py`, `code/event_kernels_synthetics.py`, `code/population_drift.py`,
      `code/population_drift_synthetics.py`, and their four `mFC_data/code/` mirrors;
    - `code/event_kernels_checks.py`, `code/preprocessing/extract_full_sessions.py`,
      `mFC_data/code/pfc_session_index.py`, `sbatch_files/event_kernels.sbatch`;
    - this file, `docs/handoff/SPIKE_CLOCK_V2_MISALIGNMENT.md`, and `code/EVENT_KERNELS.md` once
      written.
  - **v4 spike clock:**
    - `code/preprocessing/extract_rsync_from_ttl.py`, `extract_neuron_raw.py`,
      `validate_alignment.py`, `spike_clock_synthetics.py`, `spike_clock_v4_gates.py`;
    - `docs/handoff/SPIKE_CLOCK_V4.md`, `docs/BUG_ly05_recday_mismatch.md`.
  - **Tracking-v3 chain**, which the v4 pickle and this analysis read:
    - `code/preprocessing/sleap_alignment_copilot.py`, `sleap_to_rois.py`, `openfield_to_xy.py`;
    - `tracking_clean.py`, `tracking_clean_synthetics.py`, `tracking_qc.py`, `tracking_impact.py`,
      `restage_salvaged_v3.py`, `code/preprocessing/TRACKING_QC.md`,
      `docs/handoff/TRACKING_QC_PLAN.md`;
    - `code/glm_analysis_v3.py` and `code/glm_v3_synthetics.py` plus their `mFC_data/code/`
      copies (NaN path, control 12).
  - **Docs:** `docs/handoff/README.md`, `docs/handoff/MEC_INTEGRATION.md`,
    `docs/handoff/TODO_HANDOFF.md` (README links to it).
  - **`code/check_mirror_parity.py`: stage ONLY this analysis's four BYTE_PAIRS lines.** The working
    copy also registers the kNN decoder, whose files are untracked and out of scope; committing
    those lines would leave the committed parity check pointing at missing files. Build the
    staged version with a patch and `git apply --cached` (`git add -p` is interactive and
    unavailable).
  - **Leave out:**
    - `code/merge_preprocess_kilosort_spikeinterface.py`: it mixes the v4 sorter guard with an
      import of the untracked `numpy2_compat.py`;
    - the kNN, lagged-maps, LFP-anatomy, Kilosort-resume and El-Gaby `REPRODUCTION.md` / `RUN.md`
      edits;
    - the `code/brainreg_probe` submodule.
- Repo rules that bind here:
  - compute through sbatch, never the interactive cgroup;
  - nothing is overwritten (outputs are versioned `.vN`);
  - shared modules stay byte-identical, so run `python code/check_mirror_parity.py` (exit 0)
    before any submission (`event_kernels.sbatch` refuses on drift);
  - every analysis ships synthetic controls through the real data door;
  - mice are the replicates;
  - dated `.md` entries.

## 4. Data facts this analysis relies on (measured 2026-10-01/02)

**Spike clock.**

- v2/v3 spikes were 0.94-29.2 s late in 42 of 211 LEC task sessions; v4 fixes this
  (`SPIKE_CLOCK_V4.md`).
- `data_dic_lec_v4.pkl`: 25 recdays, 191 ABCD sessions.
- Four probe failures remain in sessions v4 did not change. They are kept:
  - speed lag 0.6-0.9 s off in ah08_20250620 s0, ly06_20250616 s1 and ly06_20250624 s4;
  - reward peak at the window edge in ly05_20250616 s4.
- ah08's speed reference sits at +0.25 to +0.40 s against -0.30 s elsewhere; unexplained. Report
  reward-kernel latency per mouse.
- v1 (`data_dic_lec.pkl`) is not fixed: ly05_20250620_20250623 is off by 2.6-38.6 s in all 8
  sessions.

**LEC sessions.**

- `Neuron_raw` runs from the first A to the last completed trial, in 25 ms bins. The full sessions
  re-extracted here start at recording start.
- The pre-A window ran 7.2-461 s in the five ah08 recdays (the first five done). It was fully
  tracked: pre-A tracked fraction 1.000 in every session so far.

**LEC day structure.**

- ~38 min rest, then 4-5 x (20 min task, ~20 min rest).
- Task-to-rest gap median 4.7 min, rest-to-task gap median 31.2 min.
- One KS4 sort per recday: the same units, in the same rows, in every task and rest block.
- 279 rest arrays (`neuron_raw_v4/*_sb_*`, origin 0). The 12 under 2 min are misclicks.

**PFC deposit.**

- 25 recdays, 7 mice, 1252 units; no pokes, no HD.
- `Awake_Sleep/binned_FR_dic_<rd>_<i>` holds every recording of the recday, task shards included,
  as whole recordings. Each wake `Neuron_raw` is a unique exact slice of one task shard (194/194).
- `All_session` tokens agree in 167/194 sessions. Four recdays have synthetic tokens, so no wall
  clock there: ab03_29082023, me08_10092021, me10_14122021, me10_20122021.
- XY is in pixels. Per-recday affine onto the LEC maze frame from pooled node centroids:

  | cohort | mice | fitted px/cm |
  |---|---|---|
  | 2021 | ah03, ah04, me08, me10, me11 | 19.8-20.0 (El-Gaby measured 20.2 / 20.5) |
  | 2023 | ab03, ah07 | 16.0-16.1, a different camera, which his measurement file predates |

- Session centroids can sit 3.7-4.7 cm off their recday fit (the camera moved between days)
  while the scale stays within 3%; only scale matters for speed.
- **me08_12092021_13092021's XY is unusable** (recday residual 27 cm; scale 0.7-12 px/cm by
  session). Its 7 sessions load as untracked (`xy_ok` False).
- PFC probes: reward peak median 0.00 s (184 testable sessions, 3 outside [-2, 4] s). The
  population leads speed by a median 0.50 s (IQR -0.69 to +0.10 s, 98 testable sessions).
- Running speed median 10.5 cm/s per session.
- No pre-A tracking in the deposit, so PFC pre-A rows carry no kinematics (`kin_missing`). Make no
  first-minute kinematic claims in PFC.
- **El-Gaby's 20% rule** (2024 Methods, curation rule 2): a unit is discarded if its rate in 3 or
  more sessions falls below 20% of its peak session.
  - Applied to LEC v4 it would discard 245 of 2851 unit-recordings (8.6%; ah08 8.7%, ah10 7.8%,
    ly05 11.4%, ly06 9.8%, ly07 5.4%).
  - Use it as a diagnostic only: LEC keeps all units, and every LEC vs PFC contrast is shown both
    ways.
  - Characterise the 245 units (day ramp, task-specific silence, amplitude loss).
  - PFC day-scale results are a lower bound.

**mEC.** `data_dic_mec_v4.pkl` holds 34/36 blocks; the bp02 2024-03-19 re-sorts are pending. It
waits for M6.

## 5. Code map

| file | role |
|---|---|
| `code/preprocessing/extract_full_sessions.py` | WP0, LEC. Pre-A / post-last-A spike segments on the v4 clock (`fitted_clock` from `ephys_rsync_v4`), with `first_A_bin` a multiple of 10. Gate: the core slice equals `neuron_raw_v4` byte for byte. Pre-A tracking re-sampled through the copilot's own matched-clock path (`--sync matched --sessions all --tracking-clean v2`, `_save_matched` patched to capture inputs). Gate: the overlap equals `processed_v3`. Also writes a recording index and cheap unit stability. One set of tables per recday. |
| `mFC_data/code/pfc_session_index.py` | WP0, PFC. Shard index by exact search, recording index from El-Gaby's `All_session` tokens, per-recday pixel-to-cm affine with `xy_ok`, speed and reward probes (reusing `validate_alignment.speed_probe` / `reward_probe`). |
| `code/event_kernels.py` (+ mirror) | Loaders: `load_recday(rd, 'lec' / 'pfc')` from per-session files, never the 3.8 GB pickle. Features at 25 ms, blocks of 250 ms, `build_design`, `stage1`, `reset_test`, `stage2_unit`. |
| `code/event_kernels_synthetics.py` (+ mirror) | WP1 gate, G1-G11: planted units on real behaviour (table in its docstring). |
| `code/population_drift.py` (+ mirror) | Kanter measures: `distance_vs_lag`, `knn_epochs` (within-session vs held-out session), `event_acceleration`, `synchrony_onsets`, `subsample`. |
| `code/population_drift_synthetics.py` (+ mirror) | P1-P4: noise, locked clock, OU drift, boundary ensembles. |
| `code/event_kernels_checks.py` | LEC only. The loader's core window vs `data_dic_lec_v4.pkl`; shared primitives (block counts, speed, signed accel, time since reward) vs `glm_analysis_v3` on the same sessions. |
| `sbatch_files/event_kernels.sbatch` | Runs one script from either tree on `gpu_lowp` (8 CPU, 32 GB, 4 h); refuses on mirror drift. |

Design choices inside `event_kernels.py`, each with its reason:

- **Rows** are 250 ms blocks. Spike counts are summed; lag one-hots are the fraction of the block
  in each lag bin, so a row straddling an event is coded exactly (V3's block mean of the sawtooth
  mis-coded one row per leg).
- **Lag bins:** `EDGES_S` per event (entry 0-25 min, A 0-140 s, reward 0-60 s), plus an
  **overflow** column (legs > 60 s hold 22.4% of samples) and a **pre-event** column (before an
  event's first occurrence).
- **Reference coding:** the most-occupied column is dropped per block, and so is any column
  populated in fewer than 2 sessions.
- **Kinematics.**
  - Speed and signed-acceleration deciles at block level, as V3.
  - A bank of causal exponential filters at 25 ms, tau 0.5, 2, 8, 32, 128 and 512 s. Every 2
    octaves, because neighbours at 2 per octave correlate 0.985.
  - The bank is steady-state initialised (the session mean), because a zero start is itself an
    entry ramp.
  - Fast (tau <= 8 s) and slow groups.
  - Untracked blocks are coded **missing** (kinematic columns 0 + a `kin_missing` indicator),
    not dropped. That keeps the PFC pre-A rows and the entry kernel's first seconds.
  - `kin='strict'` (added 2026-10-02 after the full G5 leak; default `'full'` unchanged, pinned
    by `code/event_kernels_pin.py`). It adds three things:
    - decile one-hots of each fast-bank trace in `kin_fast`;
    - `kin_missing` split into before-first-A, after-last-trial and gap indicators;
    - A / reward / poke-exit columns zeroed on untracked rows.

    Without it, PFC's untracked segments let the event blocks absorb kinematics. The diagnosis
    is in `code/EVENT_KERNELS.md`.
- **Place:** fractional occupancy of codes 0-21; unknown = 0 is its own category.
- **Stage 1:**
  - both X and FR centred within session, so per-session offsets are exact;
  - `glm_cv.cv_scores`, leave one session out, Freedman-Lane `n_perm=100`;
  - tested groups `ev_*`, `kin_fast`, `kin_slow`, `place`;
  - joints `ev_A+reward` and `ev_entry+kin_slow`;
  - per-fold rank recorded.
  - A unit is **locked** to a block when p < 0.05 **and** cpd > 0. Added 2026-10-02: at
    n_perm 20 the p floor (1/21) is also reached by units whose cpd is <= 0, and those were half
    of the quick-run speed-unit "leaks".
- **Reset test:** uncentred, with an intercept; groups `ev_entry` and a `day` term. `day` is
  linear in hours since the calendar day's first recording, one slope per day, plus a day-2
  offset, and is not reference-coded. `entry_beats_day = rss(entry only) < rss(day only)`.
  Needed because a day-long trend gives the same within-session slope in every session and fits
  an entry kernel under centring.
  - 2026-10-02: the first version used hat functions of hours. They cannot extrapolate to a
    held-out first or last session. On a planted day trend the day model's held-out RSS was 7095
    against 5803 for the entry model, so G7 read 0.67.
  - Sessions without a wall clock make the test return `skipped`, rather than fit a day term of
    zeros.
- **Stage 2** (`stage2_unit`):
  - the offset is the training fold's full Poisson fit (glum, alpha 1e-4, session intercepts)
    minus the event block;
  - candidate shapes: ramp exp(-dt/tau), and gamma bumps (peak x k in 2/4/8/16);
  - tau / peak chosen by training-fold deviance, a 1-D profile, and scored on the held-out
    session with only its intercept refit;
  - class from the fraction of the shape-free kernel's held-out gain each shape recovers
    (>= 0.5).
  - `dev_null` drops the block from the full fit without refitting. Use it for shape fractions,
    not as a significance test (Stage 1 is the test).
- **Loader vs plan:** the LEC loader reads the per-session files that the v4 pickle was built
  from, which is far lighter. `event_kernels_checks.py` proves it equals the pickle.

## 6. Outputs (all gitignored)

| path | content |
|---|---|
| `data/processed_data/neuron_raw_v4_full/Neuron_raw_{pre,post}_<rd>_<s>.npy` | spike segments before the first A and after the last trial |
| `data/processed_v3_full/<mouse>/<d1>_<d2>/{XY,HD,Locs}_raw_pre_<rd>_<s>.npy` | pre-A tracking; HD / Locs absent for object sessions |
| `data/processed_data/full_sessions_v4/full_sessions_<rd>.csv` | one row per task session: bins, clocks, wall time, `first_A_bin`, `n_pre` / `n_core` / `n_post`, first poke, tracking gate |
| `data/processed_data/full_sessions_v4/recording_index_<rd>.csv` | every recording: kind, wall start, duration, gap, preceding task, stage, misclick, KS dshift |
| `data/processed_data/full_sessions_v4/unit_stability_<rd>.csv` | unit x recording: spikes, KS amplitude median, last/first-quarter ratio, slope per 10 min |
| `data/processed_data/full_sessions_v4/gates_<rd>.csv` | spike gate per session |
| `mFC_data/data/processed/pfc_{session_index,recording_index,xy_to_cm}.v2.csv` | PFC index; the loaders take the highest `.vN` (v0 and v1 are superseded drafts) |
| `data/event_kernels/synthetics/`, `data/event_kernels/checks/` | synthetic and check pickles / CSVs, versioned |

## 7. Status

### 2026-10-14: user decisions after the leaky test; M3 part 2 approved

- **Commit and push** the leaky step (this entry's commit).
- **M3 part 2 next**, as planned on 2026-10-02: Tsao's stepwise side table with the Kanter GLM
  variant, and Kanter part 2. Plan and rationale: `docs/handoff/EVENT_KERNELS_WRITEUP.md`,
  Part 2.
- **The ramp test is in.** Is M2's held-out first-minute decoding the slow ramps' early slope, or
  a fast component locked to bin 0?
- **Tsao's trial time is time since the last reward.** A leg ends in one reward, as a lap did.
  Time since A stays as an extra candidate, because Stage 1 locked 316 LEC units to reward A
  against 1380 to every reward.
- **The which-clock tests are outstanding, not run.** Reframed in §8.
- **The clock origin,** from the user:
  - the first sync pulse marks the task start, when the white noise comes on;
  - the mouse is placed in the maze up to about a minute before, while the rig is set up. That
    time is not recorded.
  - Measured in LEC, bin 0 is the ephys recording start. The task start comes a median 4.6 s
    later, range 1.5-43 s, over 191 ABCD sessions (`pycontrol_start_bin` in the full-session
    tables).

### 2026-10-13: entry clock vs leaky reward count (M5's test brought forward), stopped for review

M3 part 1 is committed and pushed as 90c10d0. This step is uncommitted. Details are in
`code/EVENT_KERNELS.md`, entry 2026-10-13; the notebook is `code/event_kernels_leaky_count.ipynb`
and the figures are in `data/figures/event_kernels/M5_leaky/`.

- **What was built.**
  - `stage2_unit(extra=, joint=)` plus `leaky_bank` / `leaky_trace`. Defaults unchanged, pinned by
    `--gate-stage2`.
  - The `leaky_count` planted kind and `leaky_specs`.
  - Runner modes `--leaky`, `--gate-leaky` and `--leaky --merge`; the merge checks equivalence with
    Stage 2.
  - F19-F21.
- **Gates:** every check passes in both datasets. The leaky gate calls planted counts and ramps
  correctly in 100% at tau <= 600 s. Stage 1 calls 100% of planted counts entry-locked.
- **Result.**
  - LEC: the ramp wins in 79% of 972 entry-locked units. It adds unique gain beyond the count in
    76% (median 0.30 of the kernel gain); the count adds none (43%, median 0).
  - PFC: the ramp wins in 60%, with weaker unique gains (51% vs 46%).
  - So the slow entry signal is time since recording start, not integrated reward experience.
- **Decisions to put to the user:**
  1. Commit this step?
  2. Next, either M3 part 2 (Tsao's stepwise table, Kanter part 2) or more clock tests. Which
     clock: placement vs recording start; leaky counts of trials or distance.

### 2026-10-12, night: M3 part 1 (Stage 2) complete, stopped for review

Details: `code/EVENT_KERNELS.md`, the entries "runner V3" and "M3 part 1".

- Notebook: `code/event_kernels_M3_stage2.ipynb`.
- Figures F15-F18: `data/figures/event_kernels/M3/`.
- M2 is committed and pushed as d1b81b2.

- **Runner V3** (user-approved):
  - `build_design(first_A_pre=False)` for the first-A anchor (engine option, pinned);
  - a held-out rank criterion;
  - Stage 2, compare and gate modes.
  - Gates pass in both datasets: Stage 1 19/19; Stage 2 within 1.5e-13 relative of M1, classes
    17/17.
  - Real-data check, LEC >= 5: the entry anchor is unchanged (3e-15, no call changes). First-A
    changes sit only on the 8 artefact recdays, plus the A + reward joint, which now carries the
    pre-A offset as under anchor entry.
- **Stage 2 at >= 5 trials:** LEC 2936 + 750 null units; PFC 1413 + 647. Every job `EXIT=0`.
- **Correction to M2.**
  - Entry kernels are mostly ramps with tau a median 320 s, not first-minute transients. Their
    largest |beta| falls in the first minute because the kernel is relative to a late reference
    bin.
  - 85% of entry-locked LEC units are ramps or ramp-like bumps (54% in PFC, 27% `other`).
  - Reward kernels: bumps at about 1 s and recovery ramps with tau about 2 s.
  - Reward A: 31-44% of passers gain nothing held out.
  - No regional difference in tau.
  - The drift write-up's "transient" reading carries a dated caution.
- **Decisions to put to the user:**
  1. Commit the M3 part-1 code and documents?
  2. M3 part 2: Tsao's stepwise table and Kanter part 2. Kanter part 2 could include the
     decoding test of whether held-out first-minute decoding is just the ramps' early slope.
  3. Whether to move the M5 test (entry clock vs leaky reward count) forward. Rising entry ramps
     with tau of minutes are what a leaky reward count predicts.

### 2026-10-12, evening: M2 complete, stopped for the M3 go-ahead

Full write-ups:

- `code/EVENT_KERNELS.md`, 2026-10-12 entry: Stage 1, the reset test, the runner gates, the rank
  analysis;
- `code/POPULATION_DRIFT.md`: Kanter part 1.

Figures F1-F14 are in `data/figures/event_kernels/M2/` and `data/figures/population_drift/M2/`.
Notebooks, all `RUN_MODE = 'load'`: `code/event_kernels_M2_comparison.ipynb`,
`code/LEC_event_kernels.ipynb`, `mFC_data/code/PFC_event_kernels.ipynb`.

- **Results in brief.**
  - Every event block exceeds the null rate (0.050) in every mouse. Units locked, mean of mice:

    | block | LEC, 5 mice | PFC, 7 mice |
    |---|---|---|
    | every reward | 0.51 | 0.61 |
    | entry | 0.29 | 0.34 |
    | reward A | 0.12 | 0.16 |

  - The entry kernel is mostly a first-minute transient: 62-78% of entry-locked LEC units peak
    there. Held-out-session decoding of the 1-min epoch likewise sits almost entirely in the
    first minute (0.25 vs 0.07, chance 0.056). Over the rest of the session, change is drift.
  - The entry kernel beats a day-long trend in 82% (LEC) and 72% (PFC) of entry-locked units.
  - PFC event calls compete with the fast speed bank more than LEC's do.
  - No regional contrast is established. ENTl-deep > SUB/ProS for reward locking is a candidate
    (+0.13, p 0.023, 4 mice, two near zero).
- **Runs.** LEC 24/25 recdays at >= 10 trials (ly05_20250618_20250619 has 1 session), 25/25 at
  >= 5; PFC 24/25 (me08_12092021_13092021 has no tracked position). Every log reads `EXIT=0`.
  Both runner gates pass exactly (max |diff| = 0 against the M1 pickles).
- **Incidents, all resolved and documented in the entry:**
  - two jobs refused on a transient mirror drift, after a preemption requeue re-ran the parity
    check;
  - the me08 crash: a new skip rule, and the PFC arm rerun on the revised runner;
  - a mis-specified rank smoke check, replaced by a null-space analysis.
- **Decisions to put to the user before M3:**
  1. Commit the M2 code and documents? Nothing since 949b99f is committed. The M2 lines of
     `check_mirror_parity.py` are 3 `BYTE_PAIRS`; the file carries no other uncommitted edits
     now.
  2. The next runner revision:
     - a tolerance in `rank_report` (exact-bytes false alarms);
     - dropping the redundant `ev_first_A:pre` column. That is an engine design change, so a
       default-preserving option and a pin.
  3. M3 inputs:
     - Stage 2 on which arm, >= 10 or >= 5 (ly05 and ah08 are power-limited at 10);
     - a cross-validated sort for the F4 heatmaps before reading the peak-time spread as tiling;
     - whether to exclude the 2-3-session recdays whose median held-out r2 is negative.

### 2026-10-12: M2 go-ahead and the decisions behind it

The user approved M2 (plan `~/.claude/plans/please-read-this-handoff-crystalline-pinwheel.md`)
and settled the three questions left open at 20:45 on 2026-10-02.

| topic | decision |
|---|---|
| kinematic design | `kin='strict'` for every M2 fit |
| trial threshold | 10 in both datasets is primary; LEC also runs at 5 as a sensitivity arm. `MIN_TRIALS` stays 10 and the runners take `--min-trials`. |
| commit | the post-57eac39 work, this entry included, committed and pushed |
| stop point | through the M2 report; stop early only if the runner gate or the real-data smoke fails |
| partition | `cpu_backfill`. Supersedes the `gpu_lowp` row of section 2: since 2026-10-09 `gpu_lowp` warns that CPU-only jobs will be blocked. |

What the threshold changes:

| | at 10 | at 5 |
|---|---|---|
| LEC analysis sessions | 136 | 164 |
| ly05 analysis sessions | 13 | 25 |
| LEC recdays with fewer than 2 / 3 sessions | 1 / 2 | 0 / 0 |
| PFC analysis sessions | 176 | 187, not run |

- At 10, ly05_20250618_20250619 has 1 session and is skipped (109 units). ah08_20250624_20250625
  has 2 and runs with 2 folds, flagged.
- The gate recday ah10_20250616_20250617 has 8 sessions at either threshold, so the M1 gates cover
  both arms.

Checked before starting:

- mirror parity exits 0;
- no loader input changed after the M1 report;
- `glm_analysis_v3`, `glm_cv` and `anatomy_split` are identical to 57eac39;
- the docs written since do not change M2: `BAD_RECDAY_ly05_20250620_20250623.md` is a v1-only
  fault, the gaps in `TRACKING_GAPS_PROVENANCE.md` are the `gap` segment `strict` already codes,
  and `V4_MIGRATION.md` concerns other analyses.

M2 adds `run_event_kernels.py`, `run_population_drift.py` and `event_kernels_plots.py` (all
mirrored), `sbatch_files/submit_event_kernels.sh`, three notebooks and figures F1-F14.

### 20:45 UTC, 2026-10-02: M1 complete, stopped for the user's go-ahead

- Full gates under `--kin strict` pass every row in both datasets (jobs 3675492/93, `EXIT=0`).
  - PFC tau 2 s speed units: 0/8 locked (5/8 under the default).
  - LEC: 2/8, Fisher p 0.24 against noise.
  - Every remaining hit has a noise-sized cpd (<= +0.0007).
  - Planted-unit recovery is unchanged.
- The M1 report is the first dated entry of `code/EVENT_KERNELS.md`.
- **Decisions put to the user before M2:**
  1. Run M2 with `kin='strict'` (recommended) or the default.
  2. MIN_TRIALS = 10 leaves ly05_20250618_20250619 with 1 analysis session, so no held-out fit
     is possible. ah08_20250624_20250625 keeps 2.
  3. Whether to commit the post-57eac39 work: the `strict` option, the gate fixes, the pin and
     diagnostic scripts, the checks guard, and these documents.
- **Uncommitted since 57eac39:** `code/event_kernels.py` and `code/event_kernels_synthetics.py`
  (+ mirrors), `code/event_kernels_checks.py`, `code/event_kernels_pin.py`,
  `code/event_kernels_leak_diag.py`, `code/EVENT_KERNELS.md` and this file.

### 19:40 UTC, 2026-10-02: full gates; the G5 leak; `kin='strict'`

- The data-path checks pass (job 3675174):
  - the loader equals the v4 pickle in 211/211 sessions;
  - the blocks equal V3's in 177/177;
  - time since reward agrees to 1.1e-13 s.
- Full population-drift gates pass in both datasets (jobs 3675172/73).
- Full event-kernel gates, linear bank (jobs 3675170/71): every row passes as first written.
- The gate itself was faulty. G5 and G9 did not count the A+reward joint, which G3/G4 count as
  a hit, and G5 pooled fast and slow speed units. Fixed in `gates()`, one hit rule everywhere.
  Re-scored (`gates_rescored_*.csv`), PFC's tau 2 s speed units are locked in 5/8 (Fisher
  p 0.002 vs noise): a real fail.
- `code/event_kernels_leak_diag.py` locates the leak. It comes mostly from PFC's untracked
  segments before the first A and after the last trial, which one missing indicator could not
  separate but the A / reward pre-event and overflow columns could. Curvature from the linear
  bank is a smaller path.
- `kin='strict'` removes it in the diagnostic: the joint's gain is <= 0 on tracked rows for
  every unit, and near 0 on untracked rows.
- **Running:** full gates under `--kin strict` (jobs 3675492/93, about 90 min); quick
  3675490/91.
- The quick runs of an earlier deciles-only option (`_kin-full_nl` pickles) are superseded. Its
  full runs were cancelled.

**M1 still to do** (both done at 20:45): read the full `strict` gates; M1 report to the user.

### 17:45 UTC, 2026-10-02: WP0 complete, quick gates pass, checkpoint commit

**Done and measured.**

| step | result |
|---|---|
| WP0 LEC, all 25 recdays (jobs 3675058-61 and the ah08 job) | 211 task sessions. Spike segments: 211/211 byte-identical to `neuron_raw_v4` on the core window. Pre-A tracking: 210/211 equal to `processed_v3` on the overlap. |
| the one tracking exception | ah10_20250618_20250619 s5 (video 2025-06-19 14:35:22). Its pinstate is empty (0 rising edges). SLEAP did run: `restage_salvaged_v3.py` built its v3 arrays from a poke-based offset (first rsync frame 2244, 539 pokes, agreement 0.994). The copilot's matched clock has no pulses there, so the 7.5 s before the first A stays untracked and is coded `kin_missing`. Left as is. |
| LEC pre-A window | median 21.0 s, IQR 12.0-41.6, range 1.8-472 (211 sessions); tracked fraction of the window median 1.000, min 0.221 |
| analysis sessions | 136 ABCD sessions with >= 10 trials; first poke a median 9.6 s after recording start |
| quick `population_drift` synthetics (jobs 3675068 / 69) | P1-P4 pass in both datasets (LEC / PFC): noise kNN at chance (within 0.03 / 0.04); locked clock 17x / 16x chance across sessions; drift 13x / 12x chance within and 0.75x / 0.88x across; boundary z 6.7 / 7.5 at lag 0-1 |
| quick `event_kernels` synthetics, pass 3 (jobs 3675136 / 37) | every G1-G11 row passes in both datasets, `EXIT=0` |

Pass 3, n_perm 20. LEC is ah10_20250616_20250617 and PFC is me11_05122021_06122021. Values are the
fraction of planted units flagged (locked = p < 0.05 and cpd > 0).

| gate | LEC | PFC |
|---|---|---|
| G1 noise, any event block (30 units) | 0.00 | 0.03 |
| G2-G4 planted event units recovered (every kind) | 1.0 | 1.0 |
| G5 speed units, any event: with the bank / without kinematics (8 units) | 0.00 / 0.50 | 0.12 / 0.50 |
| G6 OU drift on `ev_entry` (4 units) | 0.00 | 0.00 |
| G7 day beats entry in trend units / entry beats day in ramps | 1.0 / 1.0 | 1.0 / 1.0 |
| G9 place cells, any event: with / without the place block (8 units) | 0.00 / 0.50 | 0.00 / 0.88 |
| G10 Stage 2: ramp class, tau within 2x, bump class | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| G11 local Freedman-Lane vs `glm_cv`, max abs dp | 0 | 0 |

**Fixes between the three quick passes.** All are in the first, uncommitted version of the
modules, so there is no version bump.

1. Pass 1 failed G5: speed units read as locked in 0.50 (LEC) and 0.25 (PFC). Half the hits had
   cpd <= 0 at the p floor. Fix: the locked rule (p < 0.05 and cpd > 0) and 4x more speed units.
2. Pass 1 failed G7 in LEC: day beat entry in 0.67 of trend units. Fix: the linear per-day day
   basis (section 5). The reset test now skips sessions without a wall clock.
3. Pass 2 failed G9 in PFC: 1 of 2 place units was flagged by `ev_A`, with cpd +0.0001 at the p
   floor. That is the same as the noise hits; Fisher p against noise was 0.24.
   - The fixed cut-offs (<= 0.10 per block, noise + 0.10) fail a pure-null process 16-29% of the
     time at these n.
   - Fix: the chance rows are now exact one-sided tests at 0.01.
     - G1 and G6 are binomial against the permutation test's exact null rate (1/21 at n_perm 20,
       5/101 at n_perm 100).
     - G5 and G9 are Fisher tests against the noise units.
   - Place units went to 8 (4 per node).
   - A `place=False` arm is reported as G9's positive control. It shows that G9 can fail.

**Committed and pushed** as `57eac39` (origin/main, 2026-10-02), with the section 3 scope. The
push also published the user's two unpushed 2026-09-23 snapshot commits (`d406dd1`, `6035467`).

**M1 still to do, in order.**

1. Full synthetics, both datasets (n_perm 100), for `event_kernels` and `population_drift`.
2. `code/event_kernels_checks.py` (64 GB). Every column must be equal; time since reward max
   |diff| 0.
3. M1 report: the first dated entry of `code/EVENT_KERNELS.md` (what was done, measured, NOT
   established, commands), then a chat summary. **Stop.**

**Deviations from the approved plan, to state in the M1 report.**

- **Identity check:** a full CPD identity with V3 was replaced by the primitives identity in
  `event_kernels_checks.py`. V3's design choices differ on purpose (mode place coding, NaN-row
  drop, transition mask, core-only rows), so a CPD identity would have tested a V3
  re-implementation, not this pipeline.
- **PFC XY gate redesigned:** El-Gaby's px/cm fits only the 2021 cohort. The gate became one
  pooled affine per recday plus a session scale check.
- **Missing kinematics are coded, not dropped.**
- **Stage 2 picks tau** by training-fold profile likelihood, not inner leave-one-session-out (an
  8x7 inner loop was too slow). The evaluation stays on held-out sessions.
- **Gate statistics:** a unit counts as locked only when p < 0.05 and cpd > 0, and the chance rows
  are exact tests (fixes 1 and 3 above).
- **Day term:** linear per day, not hat functions (fix 2, section 5).

### 17:05 UTC, 2026-10-02 (history)

**Done and measured.**

| step | result |
|---|---|
| LEC extraction smoke, ah10_20250620_20250623 (restart sessions) | spikes 9/9 byte-identical to v4; tracking 9/9 exact; pre-A median 21 s (5.2-63.5) |
| LEC extraction, the five ah08 recdays | spikes 42/42 exact; tracking 42/42 exact, object sessions included |
| PFC index v2, all 25 recdays (job 3675056) | slices 194/194; `xy_ok` 187/194 (only me08_12092021's 7 sessions are dropped); probes as in section 4 |
| `event_kernels` smoke on ah08_20250618_20250619 (3 sessions, 20 units, n_perm 0) | loader, design (14,838 rows x 95 columns), Stage 1, reset test and Stage 2 run in 23 s |
| mirror parity | OK with the four new pairs |

- Smoke caveat: held-out r2 is negative at that size (two training sessions, 95 columns). Watch
  r2_cv with 6-8 sessions.
- A harmless `Mean of empty slice` warning comes from all-NaN speed blocks.

**Running** (logs `sbatch_files/slurm_outputs/mec-fullsess_<mouse>-<id>.out`): LEC extraction for
ah10 (3675058), ly05 (3675059), ly06 (3675060) and ly07 (3675061), 5 recdays each, about 1 min per
recday. The first all-recday job (3675025) died on ah08_20250624 with `KeyError: 'rois'` (object
sessions have no ROI file); that is fixed.

## 8. Open questions and risks

- **Which clock (outstanding, 2026-10-14).**
  - Placement vs recording start cannot be tested, because placement is not recorded.
  - It matters little for ramps. Moving an exponential ramp's origin by d multiplies its
    amplitude by exp(-d/tau) and leaves tau unchanged.
  - Testable, in LEC only: task start vs recording start, a median 4.6 s apart. It matters only for
    a transient in the first seconds.
  - Still untested: leaky counts of trials (A) and of distance run, against the entry ramp.
- **Day-scale drift passing as an entry ramp.** The reset test is the defence; G7 must pass on
  synthetics before any entry-ramp claim.
- **Speed-bank start-up.** G8 compares zero vs steady-state initialisation.
- **Freedman-Lane shifts** are uniform and include near-identity draws. G11 reports power and FPR
  with a 2 min minimum. Add a default-preserving `min_shift` option to `glm_cv` only if verdicts
  change; it would need a pinning control and a mirror.
- **tau is selection-conditioned:** the WP1 recovery envelope is computed under the same Stage 1
  selection as the real data.
- **Kernel at the block-mean lag:** Stage 2 evaluates kernels at the block's mean lag, an
  approximation for tau <= 0.5 s.
- **No reset test on four PFC recdays:** they have no wall clock (synthetic tokens).
- **Region / rate confounds:** use `anatomy_split` (`rate_match`, within-recday permutation,
  50 um margin). The primary contrast is ENTl-deep vs SUB/ProS (3 mice). ENTl-sup = ah08 and
  ENTm = ly07 are single-mouse.

## 9. Commands

```
# mirror check (must exit 0)
python code/check_mirror_parity.py
# WP0, LEC (per mouse; never overwrites; tables versioned)
sbatch --partition=gpu_lowp --job-name=fullsess_<mouse> --time=0-02:00 --mem=32G \
    sbatch_files/mec_stage.sbatch extract_full_sessions.py --recdays <rd ...> --write
# WP0, PFC
sbatch --partition=gpu_lowp --job-name=pfcidx --time=0-02:00 --mem=48G \
    sbatch_files/mec_stage.sbatch ../../mFC_data/code/pfc_session_index.py --write
# WP1 synthetic gates
sbatch --job-name=ek_syn_lec_quick sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec --quick
sbatch --job-name=ek_syn_pfc_quick sbatch_files/event_kernels.sbatch mFC_data/code/event_kernels_synthetics.py --dataset pfc --quick
sbatch --job-name=pd_syn_lec_quick sbatch_files/event_kernels.sbatch code/population_drift_synthetics.py --dataset lec --quick
sbatch --job-name=pd_syn_pfc_quick sbatch_files/event_kernels.sbatch mFC_data/code/population_drift_synthetics.py --dataset pfc --quick
# the conservative kinematic design: pin the default first, then gate it
python code/event_kernels_pin.py --against 57eac39            # exit 0 = default bit-identical
sbatch --job-name=ek_syn_lec_st sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec --kin strict
sbatch --job-name=ek_syn_pfc_st sbatch_files/event_kernels.sbatch mFC_data/code/event_kernels_synthetics.py --dataset pfc --kin strict
# where a leak comes from: the joint's held-out gain by row type, both designs
sbatch --job-name=ek_diag_pfc --time=0-01:00 sbatch_files/event_kernels.sbatch code/event_kernels_leak_diag.py pfc me11_05122021_06122021
# data-path checks (LEC, loads the v4 pickle)
sbatch --job-name=ek_checks --mem=64G sbatch_files/event_kernels.sbatch code/event_kernels_checks.py
```

Read logs to their `EXIT=` line. `sbatch_files/slurm_outputs/event_kernels-<name>-<id>.out`.

## 10. Pointers

- `docs/handoff/SPIKE_CLOCK_V4.md`: the v4 clock and the run order for new ephys.
- `docs/handoff/SPIKE_CLOCK_V2_MISALIGNMENT.md`: the original finding; its root-cause section is
  superseded.
- `docs/handoff/README.md`: anatomy split, the inference rules, the rate confound.
- `docs/handoff/TODO_HANDOFF.md`: the rest of the TODO list.
- Memory (`~/.claude/projects/.../memory/`): `event-kernels-plan.md`, `pfc-deposit-facts.md`,
  `oe-counter-restart-spike-clock.md`, `cluster-gpu-lowp-for-cpu-jobs.md`.
