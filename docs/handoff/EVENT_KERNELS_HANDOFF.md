# Event-locked kernels (Tsao 2018, Kanter 2025) in LEC vs PFC — handoff

**Rewritten 2026-10-16 as a current-state handoff,** after M4 part 1 (commit f2a1872). It is for a
session or a person who picks this analysis up with no prior context.

- **Read in this order:**
  1. `.claude/CLAUDE.md`, the repo rules;
  2. §0 here;
  3. the section you need.
- **The dated status log** this file carried from 2026-10-02 to 2026-10-16 is unchanged in
  `docs/handoff/EVENT_KERNELS_LOG.md`.
- **New status entries go in §7.**

---

## 0. Start here

**Where it stands.** M1 to M4 part 1 are done, committed and pushed on `main`. Each milestone stopped
for the user's review before the next.

| milestone | date | commit | what it established |
|---|---|---|---|
| M1: data path and synthetic gate | 2026-10-02 | 57eac39 | full sessions on the v4 clock, exact to the bytes; the PFC index; every gate passes under `kin='strict'` |
| M2: Stage 1, reset test, Kanter part 1 | 2026-10-12 | d1b81b2 | event-locked units are common; entry calls are not day trends; population state drifts within sessions |
| runner V3 and M3 part 1: Stage 2 | 2026-10-12 | 90c10d0 | entry kernels are ramps of minutes, median tau 320 s; reward kernels last 1-2 s |
| entry clock vs leaky reward count | 2026-10-13 | 9b88256 | in LEC the entry signal follows elapsed time, not accumulated reward |
| M3 part 2: the papers' recipes, the ramp test, Kanter part 2 | 2026-10-14 | e5c94bf | first-minute decoding is the slow ramps' early slope; reward ensembles repeat across loops |
| M4 part 1: rest | 2026-10-16 | f2a1872 | LEC's rate falls after rest entry; rest-entry locking is unrelated to maze entry; the fast timescale is a unit property |

- **Results, all of them with figures:** https://claude.ai/artifact/557jgXBgUcq5jMarZvJBrp, a
  private page. Republish it to the same URL after every milestone; read the URL first.
- **Numbers and commands:** dated entries in `code/EVENT_KERNELS.md` and `code/POPULATION_DRIFT.md`.
- **Plan and reasoning, M1 to M3 part 2:** `docs/handoff/EVENT_KERNELS_WRITEUP.md`.

**Next, in the user's order (2026-10-15):**

1. **The PFC end-of-recording check, §8.1.** About half of PFC's entry calls rest on the last minute
   of the recording.
2. **Decide with the user what to do about the PFC `Trial_times` defect, §8.2.** It touches every
   PFC reward and A kernel in 8 sessions.
3. **Then, by the user's choice:**
   - the outstanding follow-ups, §8.4-8.6;
   - M4 part 2: offline co-activity, and preceding-task memory in rest;
   - M5, the SSM tests;
   - M6, the extensions.

**Before trusting any PFC number, read §8.1 and §8.2.**

**The rules that bite here** (§3 has them all):

- **Compute on the cluster.** Anything that loads a recday, fits or merges goes through sbatch. The
  interactive node is a shared 64 GB cgroup, and it can be slow enough that imports alone time out.
- **Registered files are mirrored.** They live byte-identical in `code/` and `mFC_data/code/`, and
  `code/check_mirror_parity.py` must exit 0 before any submission.
  - Write both copies in one step.
  - Never edit a registered file while jobs of that pipeline are queued or running. A requeued job
    re-runs the parity check, and shards stamp the sha1 of the files they ran.
- **Gates come first.** Run the quick gate, then the full gate, then the recday jobs held with
  `afterok`. If a must-hold row fails, stop, find the cause in the gate's own pickles, and report
  it.
- **Mice are the replicates.** A per-mouse value is the recday mean, then the mouse mean.
- **Stop and report after each milestone.** Commit only when the user asks, with no Co-Authored-By
  or "Generated with" lines.

---

## 1. The question

Do LEC and PFC neurons keep time from task events, as Tsao et al. 2018 found in rat LEC, and does
the population drift as Kanter et al. 2025 found? The event-locked kernels are read as the state of
a state-space model (SSM): ramps are leaky traces of events, a Laplace transform of the past, and
peaked lag cells are its readout. Mamba's input-dependent step is a clock that can stretch into goal
progress or reset at events. The working hypothesis is that LEC is a fixed clock and mFC tracks
progress.

- **Events:**
  - maze entry, which here means recording start;
  - the first A;
  - every reward A, which is a trial start;
  - every reward;
  - in rest, rest-box entry.
- **References, in the repo root:**
  - Tsao 2018: `s41586-018-0459-6.pdf` and the supplement `41586_2018_459_MOESM1_ESM.pdf`;
  - Kanter 2025, Science 388 eadr0927: `science.adr0927-8.pdf` and `science.adr0927_sm.pdf`;
  - the user's note `SSMs, Mamba and music boxes.pdf`, also an artifact:
    https://claude.ai/artifact/7uTiRZUTdihgeoGT8ML8vk.

Three things carry time, and the analysis keeps them apart:

| quantity | what it is | measured by |
|---|---|---|
| locked clock, Tsao | a kernel of time since an event that repeats across repetitions of the event | prediction of held-out sessions |
| drift, Kanter | slow change unique to each session | within-session readouts, beyond the locked part |
| boundary shift, Kanter | an abrupt population change at an event | trajectory acceleration; event ensembles |

---

## 2. Decisions taken by the user

| date | decision |
|---|---|
| 2026-10-01/02 | Shape-free event kernels are primary, classed ramp up / ramp down / bump / other; Tsao's stepwise GLM is a side table |
| 2026-10-01/02 | Speed and \|accel\| pass through a causal exponential bank, and an event block counts only if it beats the bank held out. Signed instantaneous acceleration stays in the nuisance |
| 2026-10-01/02 | Rest: rest-entry kernels against maze-entry kernels per unit; intrinsic timescale in rest against wake tau; offline co-activity ordered by wake tau. Sleep staging is a spike-based proxy in both datasets, Kanter's 25 ms rule |
| 2026-10-01/02 | SSM tests: the clock (seconds / distance / progress); reset vs superposition; readout width vs lag; the tau spectrum per region |
| 2026-10-01/02 | Session start is recording start, from full sessions re-extracted on the v4 clock; the first A is the swap anchor |
| 2026-10-01/02 | The 20 LEC object sessions are in; mEC is descriptive (ENTm on the LEC probes, the bp cohort later); amplitudes cheaply, per unit per recording, for day-scale ramps only |
| 2026-10-01/02 | LEC data = `data_dic_lec_v4.pkl` content; rests under 2 min are misclicks and excluded; the 4 v4 sessions with probe failures stay in; El-Gaby's 20% rule is a diagnostic only; PFC XY is converted to cm at load, the deposit is never modified |
| 2026-10-12 | `kin='strict'` for every fit. Primary arm >= 10 trials in both datasets, plus LEC at >= 5 as a sensitivity arm; M3 and later at >= 5 in both. Runner V3 before Stage 2 |
| 2026-10-13 | Run M5's leaky-count test before M3 part 2 |
| 2026-10-14 | Tsao's trial time is time since the last reward, because a leg ends in one reward as a lap did; time since A is an extra candidate. Include the ramp test |
| 2026-10-14 | The which-clock tests and three M3 part 2 follow-ups stay outstanding (§8). M4 runs in two parts; the sleep proxy is checked against LFP on a subset of LEC rest blocks |
| 2026-10-14 | Clock origin, from the user: the first sync pulse is the task start, when the white noise comes on. The mouse is in the maze up to about a minute before; that time is not recorded |
| 2026-10-15 | Fix M4 part 1 first, then the PFC end-of-recording check. Continuous sleep terms only, with no binary SWS flag. Add a population-rate term to the rest design |
| 2026-10-16 | The population rate divides each unit's counts. G6 and G7 get more units and segments. Commit and push M4 part 1 |

---

## 3. Process rules

**Milestones, each with a stop.** After each, report, update the write-ups and the results page, and
wait for the go-ahead:

- M1: data preparation, WP0, and the synthetic gate, WP1;
- M2: Stage 1, the reset test, Kanter part 1;
- M3: Stage 2, Tsao's and Kanter's tables;
- M4: rest, in two parts;
- M5: the SSM tests;
- M6: the extensions: day clock, objects, poke exit, bp mEC.

**Repo rules that bind here**, from `.claude/CLAUDE.md`:

- Compute through sbatch, never the interactive cgroup. A detached child dies with the session, and
  `EXIT 137` with no traceback is the cgroup.
- Nothing is overwritten. Merges are versioned `.vN`, and `load_latest` reads the highest.
- An engine change goes in a new version with an equivalence control. A new option inside the
  current version is fine when its default reproduces every existing number and a synthetic control
  pins that default: `code/event_kernels_pin.py --against HEAD`.
- Every analysis ships synthetic controls that enter through the same door as the data. A rejection
  test must also be shown to accept something.
- Mice are the replicates. LEC regions come from `anatomy_split`, with rate matching, within-recday
  label shuffles and a 50 µm boundary margin.
- `.md` deliverables are dated entries: what was done, measured, NOT established, and the commands.
  Never rewrite an earlier entry; add a dated correction.

**Operational rules this analysis learned:**

- **Mirrors.** Write both copies of a registered file in one step, e.g. edit then `cp` in the same
  command, and run the parity check. A requeued `cpu_backfill` job re-runs the check and refuses on
  drift.
- **Stamps.** Shards stamp the sha1 of the runner and the modules they import. A merge refuses
  mismatched stamps, so never edit those files while jobs of that arm can still start.
- **Waiting.** Wait by SLURM job id, `until [ -z "$(squeue -j <ids> -h)" ]`. Never `pgrep -f`: it
  matches its own command line.
- **The scratchpad** under `/tmp` is the node's own disk. Job outputs written there vanish.
- **Notebooks** run through nbclient have two timeouts, the SLURM walltime and the per-cell
  `--ExecutePreprocessor.timeout`; set both. Read every log to its `EXIT=` line.
- **Gate chain:**
  1. quick gate;
  2. full gate, `--dependency=afterok:<quick>`;
  3. recday jobs, `submit_event_kernels.sh --after <full>`.

  A failed gate leaves the held jobs pending forever (`DependencyNeverSatisfied`); cancel them.
- **When a gate row fails, read the gate pickle before touching a threshold:** the failing units'
  rates, the planted truth, the design columns. Every M4 cause was found that way.

---

## 4. Data facts

**Spike clock and LEC data.**

- v2/v3 spikes were 0.94-29.2 s late in 42 of 211 LEC task sessions; v4 fixes this
  (`docs/handoff/SPIKE_CLOCK_V4.md`). v1 (`data_dic_lec.pkl`) is not fixed: ly05_20250620_20250623
  is off by 2.6-38.6 s in all 8 sessions.
- `data_dic_lec_v4.pkl`: 25 recdays, 5 mice, 2851 units, 66-186 per recday, median 108. 211 task
  sessions: 191 ABCD, 20 object. One KS4 sort per recday: the same units, in the same rows, in every
  task and rest block.
- Four probe failures remain in sessions v4 did not change, and they are kept:
  - speed lag 0.6-0.9 s off in ah08_20250620 s0, ly06_20250616 s1 and ly06_20250624 s4;
  - reward peak at the window edge in ly05_20250616 s4.
- ah08's speed reference sits at +0.25 to +0.40 s, against -0.30 s elsewhere; unexplained. Report
  reward-kernel latency per mouse.
- The full sessions start at recording start: `Neuron_raw_{pre,post}` add the segments before the
  first A and after the last trial to `neuron_raw_v4`'s core. Pre-A tracking covers at least 90% of
  the pre-A window in 202 of 210 sessions; the lowest is 22%.

**The clock origin.** Every "entry" result is time since bin 0.

- **LEC:** bin 0 is the ephys recording start.
  - The task starts at the first sync pulse, when the white noise comes on: a median 4.6 s later,
    range 1.5-43 s over 191 ABCD sessions (`pycontrol_start_bin`).
  - The first A comes a median 23.5 s after bin 0.
  - Video starts with the recording, and the mouse is in view from the first frame.
  - The mouse is put in the maze up to about a minute before the task starts, and that time is not
    recorded.
- **PFC:** bin 0 is the start of El-Gaby's whole-recording shard, and the first A comes a median
  18.0 s later. The task start is not indexed.
- **For an exponential ramp the origin cannot be identified.** Moving it by d multiplies the amplitude
  by exp(-d/tau) and leaves tau unchanged.

**ABCD timing, LEC:** legs from reward to reward, median 9.3 s, mean 14.4 s, n = 13064; loops from A
to A, median 43.6 s, mean 57.5 s, n = 3266. Legs lengthen within a session, rho about -0.3.

**Day structure.**

- About 38 min of rest, then 4-5 cycles of a 20-min task and about 20 min of rest.
- Gaps: task to rest a median 4.7 min, rest to task a median 31.2 min.
- LEC rest: 279 recordings, 12 of them misclicks under 2 min. Before the first task, a median 36 min
  over 55 blocks; between tasks, 20.5 min over 212.
- PFC rest: 239 shards, 36.5 and 20.3 min. The 4 recdays without wall-clock tokens have no usable
  rest.
- Rest has no tracking or video. LEC has raw LFP; PFC has none.

**PFC deposit** (`docs/handoff/`, memory `pfc-deposit-facts`).

- 25 recdays, 7 mice, 1252 units; no pokes, no HD.
- `Awake_Sleep/binned_FR_dic_<rd>_<i>` holds every recording of a recday as whole recordings, task
  shards included. Each wake `Neuron_raw` is a unique exact slice of one task shard (194/194).
- Wall clock: tokens agree in 167/194 sessions. Four recdays have synthetic tokens and so no wall
  clock: ab03_29082023, me08_10092021, me10_14122021, me10_20122021. No reset test there, and no
  rest.
- XY is in pixels, with a per-recday affine onto the LEC maze frame:
  - 2021 cohort, ah03, ah04, me08, me10 and me11: 19.8-20.0 px/cm;
  - 2023 cohort, ab03 and ah07: 16.0-16.1 px/cm.
- **me08_12092021_13092021's XY is unusable,** so its 7 sessions load untracked. It is skipped
  wherever tracking is needed.
- There is no pre-A tracking in PFC, so make no first-minute kinematic claims there.
- El-Gaby's 20% rule was already applied to PFC, which selects against slow change across sessions.
  On LEC it flags 149 of 2851 units, or 245 by M1's count, which is not reconciled. Every LEC vs PFC
  contrast is shown both ways.
- **Two defects matter here:** the post-task end of each recording (§8.1) and shared `Trial_times`
  arrays (§8.2).

**Sessions analysed.**

- ABCD sessions with >= 10 trials: 135 LEC sessions over 24 recdays, 171 PFC over 24.
- >= 5 trials: 164 LEC sessions over 25 recdays, 180 PFC over 24.
- LEC ly05_20250618_20250619 has 1 session at 10 trials.

**mEC.** `data_dic_mec_v4.pkl` holds all 36 blocks since the bp02 re-sorts of 2026-10-05. It waits
for M6.

---

## 5. Code map

Everything is in `code/`. A `+ mirror` file is byte-identical in `mFC_data/code/` and registered in
`code/check_mirror_parity.py`.

| file | role |
|---|---|
| `preprocessing/extract_full_sessions.py` | WP0, LEC: pre-A / post-last-trial spike segments on the v4 clock, pre-A tracking, recording index, unit stability. Gates: the core equals `neuron_raw_v4` byte for byte; the tracking overlap equals `processed_v3` |
| `mFC_data/code/pfc_session_index.py` | WP0, PFC: shard index by exact search, recording index from El-Gaby's tokens, pixel-to-cm affine with `xy_ok`, speed and reward probes |
| `event_kernels.py` (+ mirror) | **The engine.** Loaders `load_recday(rd, 'lec'/'pfc', with_rest=)`; 25 ms features; 250 ms design `build_design`; `stage1`; `reset_test`; `stage2_unit`, which takes `extra=` and `joint=` families; `leaky_bank`. Defaults pinned by `event_kernels_pin.py` |
| `event_kernels_synthetics.py` (+ mirror) | M1 gate G1-G11 and the leaky kinds; planted units on real behaviour; table in its docstring |
| `run_event_kernels.py` (+ mirror) | Runner: Stage 1 and the reset test per recday and anchor; `--stage2`, `--leaky`; `--merge`; gates `--gate`, `--gate-stage2`, `--gate-leaky`; `--compare` of two merges |
| `event_kernels_pin.py` | Pins the default design bit-identical to a commit, and checks what `kin='strict'` and `first_A_pre=False` change |
| `event_kernels_checks.py`, `event_kernels_leak_diag.py` | LEC data-path checks against the v4 pickle and GLM V3; the PFC leak diagnosis of M1 |
| `population_drift.py` (+ mirror) | Kanter's measures: distance vs lag, kNN epochs, acceleration; part 2's boundary ensembles, PV correlation, trial-number decoding, coding axes, Fano and its removal |
| `population_drift_synthetics.py` (+ mirror), `run_population_drift.py` (+ mirror) | Gates P1-P15; the runner, with `--part2` and `--gate-part2` |
| `unit_timescales.py` (+ mirror) | M3 part 2: Tsao's stepwise Poisson GLM with MATLAB's `stepwiseglm` rules, Newton IRLS; `exp_fit`; Kanter's GLM and ACF tau; `heldout_mu`, the Stage 2 full model held out, with the ramp test's divisors |
| `unit_timescales_synthetics.py` (+ mirror), `run_unit_timescales.py` (+ mirror) | Gates T1-T8, K1-K5c; the runner, with equivalence checks at merge |
| `rest_kernels.py` (+ mirror) | M4: the DOWN-state proxy, calibrated on task data; signal-loss QC; `relative_pop_rate`, the median divisor; `build_rest_design`; `state_masks`; low- and high-DOWN windows; Murray's tau; drift segments |
| `rest_kernels_synthetics.py` (+ mirror), `run_rest_kernels.py` (+ mirror) | Gates G1-G8 and G4b; the runner, with two half fits per recday and no-state and shared-ramp controls in `--gate` |
| `rest_lfp_check.py` | LEC only, not mirrored: the proxy against LFP delta, `--pick`, per block, `--summarise` |
| `event_kernels_plots.py` (+ mirror) | Every figure and table: F1-F37, from the merged pickles |
| notebooks, load mode | `event_kernels_M2_comparison.ipynb`, `LEC_event_kernels.ipynb`, `mFC_data/code/PFC_event_kernels.ipynb`, `event_kernels_M3_stage2.ipynb`, `event_kernels_leaky_count.ipynb`, `event_kernels_M3_part2.ipynb`, `event_kernels_M4_rest.ipynb` |
| `sbatch_files/event_kernels.sbatch`, `sbatch_files/submit_event_kernels.sh` | One script from either tree on `cpu_backfill`, refusing on mirror drift; one job per recday for each runner: `stage1`, `drift`, `stage2`, `leaky`, `timescales`, `drift2`, `rest` |

**Design choices in the engine,** each with its reason:

- **Rows** are 250 ms blocks. Spike counts are summed. Lag one-hots hold the fraction of the block in
  each lag bin, so a row that straddles an event is coded exactly.
- **Lag bins:** `EDGES_S` per event: entry 0-25 min, A 0-140 s, reward 0-60 s, rest entry 0-40 min.
  Each block also has an overflow column and a pre-event column. Reference coding drops the
  most-occupied column, and any column populated in fewer than 2 sessions.
- **Kinematics:**
  - speed and signed-acceleration deciles;
  - a causal exponential bank with tau 0.5, 2, 8, 32, 128 and 512 s, every 2 octaves, because
    neighbours at 2 per octave correlate 0.985;
  - the bank is steady-state initialised, because a zero start is itself an entry ramp;
  - untracked blocks are coded missing, not dropped.
  - `kin='strict'` adds decile codes of the fast bank, splits `kin_missing` by segment, and masks
    fast events on untracked rows. M1 showed the default leaked fast speed into PFC's event blocks.
- **Stage 1:**
  - X and FR are centred within session;
  - `glm_cv.cv_scores`, leave one session out, Freedman-Lane with 100 permutations;
  - a unit is locked when p < 0.05 **and** cpd > 0. The exact null rate is 5/101.
- **Reset test:** uncentred, the event kernel against a linear trend in hours since the calendar
  day's first recording, one slope per day.
- **Stage 2:**
  - Poisson fits with glum, the offset being the training fold's full model minus the event block;
  - a ramp exp(-t/tau) and gamma bumps;
  - tau and peak chosen on the training sessions and scored held out;
  - a class needs at least half of the shape-free kernel's held-out gain.

**Design choices in rest, M4:**

- **The proxy.** The summed 25 ms population count, against a threshold at the 1st percentile of the
  recday's own task bins. DOWN states are runs of at least 100 ms at or below it; `down_frac` is
  their fraction of each 10 s window.
- **Calibration.** It always comes from the recday's real task data, in the gate too, because planted
  Poisson task data have no lulls.
- **State terms:** `down_frac` deciles, with 0 as its own bin.
- **Division.** Units are split even and odd, and each half's counts are divided by the other half's
  relative population rate. That rate is the median, over units of at least 0.3 Hz, of each unit's
  30 s count over its own mean, floored at 0.2. A recday needs 10 such units per half. So
  rest-entry locking is relative to the rest of the population.
- **Rest splits:** low-DOWN windows, `down_frac` at most the recday's task cut, against high-DOWN
  windows, at least 0.1 and above the cut.

---

## 6. Outputs

Every output is gitignored.

| path | content |
|---|---|
| `data/processed_data/neuron_raw_v4_full/`, `data/processed_v3_full/` | spike segments and pre-A tracking outside the core |
| `data/processed_data/full_sessions_v4/{full_sessions,recording_index,unit_stability,gates}_<rd>.csv` | session and recording tables; `stage`, misclicks, `pycontrol_start_bin` |
| `mFC_data/data/processed/pfc_{session_index,recording_index,xy_to_cm}.v2.csv` | the PFC index |
| `data/event_kernels/synthetics/`, `checks/` | M1 gate and check pickles |
| `data/event_kernels/stage1/stage1_<ds>_kin-strict_mt{10,5}_np100[.vN].pkl` + `_units`, `_reset`, `_summary` CSVs | Stage 1. LEC at >= 5 trials is `.v1`, runner V3 |
| `data/event_kernels/stage2/stage2_<ds>_kin-strict_mt5_np100.pkl` + CSVs | Stage 2 shapes and tau |
| `data/event_kernels/leaky/leaky_<ds>_kin-strict_mt5_np100.pkl` | entry clock against a leaky reward count |
| `data/event_kernels/timescales/timescales_<ds>_kin-strict_mt5_np100.pkl` | Tsao, Kanter single-unit, the ramp test |
| `data/population_drift/drift_<ds>_mt{10,5}.pkl`; `data/population_drift/part2/part2_<ds>_mt5.pkl` | Kanter parts 1 and 2 |
| `data/event_kernels/rest/rest_<ds>_mt5_np100.pkl` + CSVs; `rest/runner_gate/`; `rest/lfp_check/` | M4 part 1 |
| each pipeline's `shards/` and `runner_gate/` folders | per-recday shards, and gate CSVs and pickles. Superseded shards sit in `shards_superseded/` |
| `data/figures/event_kernels/{M2,M3,M5_leaky,M3_part2,M4}/`, `data/figures/population_drift/{M2,M3_part2}/` | F1-F37, PDF + PNG |

---

## 7. Status

### 2026-10-16: M4 part 1 committed and pushed as f2a1872

- User: commit and push. The rest code in both trees, F31-F37, the notebook and the write-ups are in.
- The 2026-10-15 PFC end-of-recording note went in with it.
- **Next:** the PFC end-of-recording check, §8.1.

### Results so far, in numbers

Fractions are means over mice, each mouse the mean of its recdays; ranges are over mice. LEC has 5
mice and PFC 7.

| result | LEC | PFC | source |
|---|---|---|---|
| units locked to every reward / entry / reward A, >= 10 trials | 0.51 / 0.29 / 0.12 | 0.61 / 0.34 / 0.16 | M2, F1 |
| entry kernel beats a day trend, of entry-locked units | 0.82 | 0.72 | M2, F6 |
| entry kernels that are ramps or ramp-like; median tau | 0.85; 320 s | 0.54; 320 s, plus 0.27 `other` | M3, F15-F16 |
| reward kernels: bump peak; ramp tau | 1.0 s; 2.0 s | 1.0 s; 1.4 s | M3 |
| entry ramp beats a leaky reward count, of entry-locked units | 0.79 of 972 | 0.60 of 414 | leaky, F19 |
| first-minute held-out decoding removed by slow ramps of tau >= 240 s / by the whole kernel | 0.55 / 0.62 | 0.37 / 0.49 | M3.2, F30 |
| Tsao's recipe: day time / entry time | 0.175 / 0.166; Tsao BW12 0.182 / 0.062 | 0.190 / 0.174 | M3.2, F22 |
| reward ensembles across loops, PV correlation: modulated / other units | 0.33-0.42 / 0.20 | 0.50-0.53 / 0.18 | M3.2, F27 |
| leg vs session coding axes, 20-unit draws; stabilities | 71.7°; 7-10° | 69.8°; 7-10° | M3.2, F28 |
| population rate after rest entry, minutes 15-20 over 0-2 | 0.79-0.84 | 0.74-1.31 | M4, F36 |
| units locked to rest entry, relative to the population | 0.29 | 0.13, 5 mice | M4, F32 |
| fast timescale, rest vs task: Spearman across units; median ratio | 0.47; 0.66 | 0.50; 0.91 | M4, F34 |

**Predictions that did not hold,** for orientation:

- In sample, entry time is not selected over day time.
- Tsao's exponential tau is not of the order of 320 s: it has no baseline.
- The maze-entry clock does not restart, unit by unit, at rest entry.
- Neither intrinsic timescale tracks the event kernels' tau.

The full list, with what held, is on the results page and in the dated entries.

### Status log

Earlier entries, 2026-10-02 to 2026-10-16: `docs/handoff/EVENT_KERNELS_LOG.md`. Add new dated entries
above this line, newest first.

---

## 8. Open questions and risks

### 8.1 PFC entry calls lean on the end of the recording. Next.

- At >= 5 trials, 184 of 414 PFC entry-locked units put their largest kernel weight in the
  1200-1500 s bin, 178 of them negative. In LEC it is 16 of 972.
- In PFC, 88% of that bin's rows come after the last trial; in LEC 59%. The PFC task ends a median
  1153 s after bin 0, and the recording 1219 s.
- These units are 119 of PFC's 130 Stage 2 `other` entry units.
- **To do:**
  - add a default-preserving option to the runner that drops the rows after the last trial, pinned
    by `event_kernels_pin.py`;
  - rerun Stage 1 at the entry anchor in both datasets, at >= 5 trials, 50 jobs;
  - report how many entry calls survive, then PFC's Stage 2 classes and its reset and leaky tests.
- Every PFC entry number, M2 to M3 part 2, waits on this. `code/EVENT_KERNELS.md`, 2026-10-15.

### 8.2 PFC `Trial_times` defect: found by the spatial work, 2026-10-14

- **The defect.** Seven single-day PFC sessions share one 37-trial array in
  `mFC_data/data/Trial_times/`:
  - ah04 05-12, 07-12, 08-12 and 09-12 session 3;
  - me10 09-12 and 14-12 session 3;
  - me10 15-12 session 2.

  me10 10-12 session 4 holds its first 35 rows. The mouse is off the B, C and D goals at 66-99% of
  those trial times. `mFC_data/code/diag_pfc_arrivals.py` lists the sessions; `docs/handoff/README.md`
  has the detail.
- **Why it matters here.** `event_kernels.py` reads these files for PFC: `_load_pfc`,
  `trialtimes_{recday}_{s}.npy`. So the reward and A events of these 8 sessions are wrong in every
  PFC Stage 1, Stage 2, leaky, timescale and boundary result that includes them.
  - The entry clock and the core window do not use them; the core comes from the shard index.
  - Each session's trial count does, through `n_trials = len(trial_times)`, and so does the >= 5 and
    >= 10 trial selection.
- **To decide with the user:** exclude the 8 sessions and re-merge the PFC arms, or recover the
  correct trial times. The impact on PFC numbers has not been measured; most PFC recdays are
  unaffected.

### 8.3 Rest, M4 part 1

- **Rest-entry locking is relative to the population.**
  - A unit whose state dependence differs from the population's locks on average, because state
    changes come at block-specific times that average into a time course.
  - State-locked units lock more often: 0.33 against 0.24 in LEC.
  - The gate plants only shared and SWS-specific state changes. A gate row for unit-specific state
    dependence is the obvious next control.
  - 40% of LEC calls do no better than a linear day trend.
- **The rest-maze comparison is not like for like.** Maze kernels are absolute. The LEC population's
  own rest-entry decline, 16-21% over 20 min, is absorbed by design.
- **Sleep staging is not validated.** Kanter's theta/delta < 0.8 holds in 82-100% of every LEC
  block's windows, and rest delta never reaches the awake threshold. The DOWN fraction tracks delta
  between tasks only, Spearman 0.60-0.86, and there is no REM.
- **The low-DOWN class is loose where a recday's cut is high.** Cuts run 0.013-0.68 in LEC: ah08 at
  0.13-0.26, and ly06_20250613 at 0.68 with proxy AUC 0.755.
- **PFC's DOWN fraction** is high from the first minute in some blocks. In 20-60 units, joint silences
  are not necessarily sleep.
- **No rest-entry fit for 6 small PFC recdays.** ah03 and me08 drop out of that part.
- **Drift in rest against task is confounded.** Rest mixes quiet wake, drowsiness and REM, and a 60 s
  task segment carries about one loop.

### 8.4 Which clock: outstanding, user 2026-10-14

- Placement against recording start cannot be tested, because placement is not recorded. It matters
  little for ramps, whose tau does not depend on the origin.
- Testable in LEC only: task start against recording start, a median 4.6 s apart. It matters only for
  a transient in the first seconds.
- Untested: leaky counts of trials, A, and of distance run, against the entry ramp.

### 8.5 Follow-ups from M3 part 2: outstanding, user 2026-10-14

1. What carries the 40% (LEC) and 51% (PFC) of M2's first-minute held-out decoding left after
   division? Size-matched non-locked units decode minute 1 at 0.164 and 0.223, against chance 0.056.
   Sub-threshold entry kernels are the candidate.
2. SUB/ProS against ENTl-deep. In all four LEC mice, SUB/ProS has the longer residual ACF tau, the
   higher residual Fano and the more day-time units. Check against probe depth and sorting quality
   before reading it as anatomy.
3. Reward ensembles repeat across loops: PV correlation 0.33-0.53 against 0.18-0.20. Test with place
   and kinematics regressed out of the boundary windows, per reward type.

### 8.6 Method caveats that stand

- **Per-unit power at real effect sizes.** Planted units modulate the rate about 3x, while real
  held-out cpd is about 0.005. Per-unit calls are noisy; the robust evidence is in fractions and
  asymmetries.
- **tau.** It is selection-conditioned, and tau of 30 min or more reads as "slower than the
  session". Kernels are evaluated at the block-mean lag, an approximation for tau <= 0.5 s.
- **Freedman-Lane shifts** are uniform. A 2-min minimum shift changed no verdict in G11, so
  `glm_cv` stays unchanged.
- **The stepwise fit against MATLAB.** It copies `stepwiseglm`'s rules, but a run against MATLAB
  itself was planned and no result is recorded.
- **LEC vs PFC.** No contrast has been tested at mouse level, with 5 and 7 mice. PFC's event calls
  compete more with kinematics, and the PFC Fano excess is not rate-matched.

---

## 9. Commands

Logs are at `sbatch_files/slurm_outputs/event_kernels-<name>-<id>.out`; read each to its `EXIT=` line.
`<t>` is `code` for LEC or `mFC_data/code` for PFC.

```
python code/check_mirror_parity.py                                   # exit 0 before any submission
python code/event_kernels_pin.py --against HEAD                      # engine defaults unchanged

# M1: WP0 and the synthetic gate
sbatch --partition=gpu_lowp --job-name=fullsess_<mouse> --time=0-02:00 --mem=32G \
    sbatch_files/mec_stage.sbatch extract_full_sessions.py --recdays <rd ...> --write
sbatch --partition=gpu_lowp --job-name=pfcidx --time=0-02:00 --mem=48G \
    sbatch_files/mec_stage.sbatch ../../mFC_data/code/pfc_session_index.py --write
sbatch --job-name=ek_syn_lec sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec --kin strict

# Stage 1 and Stage 2, the leaky test (gate first, then hold the recday jobs on it)
sbatch --time=0-01:30 --job-name=ek_rgate_lec sbatch_files/event_kernels.sbatch <t>/run_event_kernels.py --gate --dataset lec
bash sbatch_files/submit_event_kernels.sh --dataset lec --min-trials 5 --after <gate>
python <t>/run_event_kernels.py --dataset lec --min-trials 5 --merge
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner stage2 --min-trials 5
python <t>/run_event_kernels.py --dataset lec --stage2 --merge --min-trials 5
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner leaky --min-trials 5 --after <gate-leaky>

# Kanter parts 1 and 2; Tsao and Kanter single-unit, the ramp test
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner drift
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner drift2 --min-trials 5 --after <gate-part2>
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner timescales --min-trials 5 --after <gate>

# M4 part 1, rest
sbatch --time=0-02:00 --job-name=rk_gate_q_lec sbatch_files/event_kernels.sbatch <t>/run_rest_kernels.py --gate --dataset lec --quick
sbatch --time=0-03:00 --job-name=rk_gate_f_lec --dependency=afterok:<quick> sbatch_files/event_kernels.sbatch <t>/run_rest_kernels.py --gate --dataset lec
bash sbatch_files/submit_event_kernels.sh --dataset lec --runner rest --min-trials 5 --after <full>
sbatch --time=0-01:00 --mem=48G sbatch_files/event_kernels.sbatch <t>/run_rest_kernels.py --dataset lec --merge --min-trials 5 --n-perm 100
python code/rest_lfp_check.py --pick                                 # then one job per block, then --summarise

# notebooks: two timeouts, SLURM walltime and the per-cell limit
sbatch --job-name=ek_nb_m4 --time=0-01:30 --mem=48G --cpus-per-task=4 sbatch_files/event_kernels.sbatch \
    -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 code/event_kernels_M4_rest.ipynb
```

Each dated entry in `code/EVENT_KERNELS.md` and `code/POPULATION_DRIFT.md` lists its exact commands.

---

## 10. Pointers

- **Results page,** private: https://claude.ai/artifact/557jgXBgUcq5jMarZvJBrp. Memory note
  `event-kernels-results-page` says how to update it.
- **Write-ups:**
  - `code/EVENT_KERNELS.md` and `code/POPULATION_DRIFT.md`, dated, newest first;
  - `docs/handoff/EVENT_KERNELS_WRITEUP.md`, the plan and results for M1 to M3 part 2;
  - `docs/handoff/EVENT_KERNELS_LOG.md`, the status log.
- **Plans** in `~/.claude/plans/`:
  - `glittery-wandering-wadler.md`, the 2026-10-02 plan for M1-M6;
  - `please-read-this-handoff-crystalline-pinwheel.md`, the M4 part 1 plan and its revision.
- **Related docs:**
  - `docs/handoff/SPIKE_CLOCK_V4.md`, the v4 clock;
  - `docs/handoff/README.md`, the anatomy split, the inference rules, the PFC `Trial_times` defect;
  - `docs/handoff/TODO_HANDOFF.md`, the wider TODO list.
- **Memory,** in `~/.claude/projects/.../memory/`: `event-kernels-plan`, `event-kernels-results-page`,
  `lec-clock-origin-task-start`, `pfc-deposit-facts`, `synthetic-controls-catch-design-errors`,
  `oe-counter-restart-spike-clock`, `cluster-gpu-lowp-for-cpu-jobs`, `waiting-on-cluster-jobs`,
  `scratchpad-is-node-local`.
