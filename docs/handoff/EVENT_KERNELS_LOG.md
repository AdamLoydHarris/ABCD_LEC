# Event-locked kernels — dated status log, 2026-10-02 to 2026-10-16

Moved here unchanged from `EVENT_KERNELS_HANDOFF.md` §7 on 2026-10-16, when the handoff was rewritten
as a current-state document. Newest entry first. Each entry records what was decided, run and found
that day; results and commands are in `code/EVENT_KERNELS.md` and `code/POPULATION_DRIFT.md`.
New status entries go in the handoff's §7, not here.

### 2026-10-16, afternoon: M4 part 1 complete, stopped for review

Details: `code/EVENT_KERNELS.md` and `code/POPULATION_DRIFT.md`, entries 2026-10-16. Notebook:
`code/event_kernels_M4_rest.ipynb`, no error outputs. Figures F31-F37: `data/figures/event_kernels/M4/`.
Uncommitted.

- **Runs.** Round 5 gates pass in both datasets. 50 recday jobs, every one `EXIT=0`; merged 25 + 25
  shards, PFC with the 4 tokenless recdays skipped. 6 small PFC recdays have no rest-entry fit, so
  ah03 and me08 drop out of it.
- **Results.**
  - The LEC population rate falls 16-21% over the first 20 min of rest in every mouse.
  - LEC mice rarely reach DOWN-rich states: high-DOWN windows are 0.4-3.9% per mouse; PFC 9-38%.
  - Relative to the population, 29% of LEC units (mice 0.14-0.35) and 13% of PFC units lock to rest
    entry. Most are slow ramps of mixed sign, median tau 1280 s; 60% beat a day trend.
  - They are unrelated to maze-entry locking: no enrichment; sign agreement 0.59 in 107 units.
  - The fast intrinsic timescale is a unit property: Spearman across rest and task 0.47-0.50, and
    in LEC it is shorter in rest, ratio 0.66. The slow one is not, Spearman 0.14-0.26. Neither
    tracks the event kernels' tau, Spearman -0.01 to 0.17.
  - Low-DOWN rest drifts more than task within 60 s in 5 of 5 LEC mice, but its epochs decode worse.
- **Caveats to put to the user:**
  - rest-entry locking may be unit-specific state dependence, which the gate does not plant;
  - the low-DOWN class is loose where a recday's cut is high: ah08, ly06_20250613;
  - task segments of 60 s carry the loop's period.
- **Decisions to put to the user:**
  1. Commit M4 part 1?
  2. Next is the PFC end-of-recording check (decided 2026-10-15). Also open: a gate row for
     unit-specific state dependence, and M4 part 2.

### 2026-10-16: M4 part 1 revised, gates and runs submitted

- **User decisions, 2026-10-15:**
  - fix M4 part 1 first, then the PFC end-of-recording check (§8);
  - continuous state terms only, with no binary SWS flag: the pre-set fallback, because the LFP
    check could not validate staging;
  - add a population-rate term to the rest design.
- **Why the first gates failed,** read from their pickles:
  - the gate calibrated the proxy on planted Poisson task data, which have no lulls. So the cut
    was 0.0000, and any 100 ms lull marked a window SWS: the proxy's onset fell 5-130 s before the
    planted onset in 10 of 11 LEC blocks, inside the planted quiet wake;
  - that misfit, and the quiet wake's 0.7x dip in every unit, are locked to rest entry on average.
    High-rate units reach p = 1/101: locked noise in the LEC quick gate had a median of 9.7 Hz,
    against 1.4 Hz unlocked;
  - the LFP rule failed both ways: theta/delta < 0.8 in 82-100% of every block's windows, and rest
    delta at a median 0-20% of the awake threshold;
  - Murray's 0.25 s fits failed at the planted OU units' real low rates, LEC median 0.8 Hz.
- **What changed,** in `rest_kernels.py`, `rest_kernels_synthetics.py` and
  `run_rest_kernels.py`, mirrored, and `rest_lfp_check.py`:
  - the `state` group replaces `sws`. It holds deciles of `down_frac` and of the other half's
    population rate per 10 s window. Units are split even / odd, each half is fitted with the
    other half's rate, and a recday needs at least 10 units per half;
  - the proxy is applied with the real recday's calibration, in the gate too;
  - low-DOWN windows, within the awake range, and high-DOWN windows, at least 0.1, replace
    non-SWS / SWS for the timescales and drift. Drift draws up to 3 segments per run;
  - planted OU units at a fixed 5 Hz; new reported row G4b, a ramp shared by every unit;
  - the LFP readout is the Spearman of `down_frac` with log delta.
- **Jobs.** Quick gates 3771441 (LEC) and 3771442 (PFC). Full gates 3771443 and 3771444, each
  `afterok` on its quick gate. Recday jobs LEC 3771449-3771473 and PFC 3771474-3771498, `afterok`
  on the full gates.
- **Do not edit any registered rest file until every one of these jobs has ended.** A requeued job
  re-runs the parity check, and the shards stamp the files' sha1.
- **Result, 2026-10-16 08:25: both quick gates fail (`EXIT=1`).** The full gates and the 50 recday
  jobs could then never start, and were cancelled.

  | row | LEC quick | PFC quick |
  |---|---|---|
  | G1 proxy AUC at the real unit count; planted high-DOWN recall / quiet wake high-DOWN | 1.000, pass; 0.999 / 0.008 | 1.000; 0.998 / 0.021 |
  | G3 SWS-gain units locked with the state terms | 4 of 4, fail | 4 of 4, fail |
  | G4 noise locked with / without the state terms | 2 of 92, pass / 21 of 92 | 12 of 58, fail / 39 of 58 |
  | G4b shared ramp: noise locked with / without the state terms | 79 of 92 / 86 of 92 | 54 of 58 / 57 of 58 |
  | G6 slow tau ratio at 120 s | 0.684, fail | 1.466, fail |
  | G7 drift_equal ratio, 12 segments | 1.36, fail | 0.91, pass |

  G2, G5, G6 at 30 s, Murray at 0.25 s, G7 fast and G8 pass in both.
  - The real calibration works: the planted quiet wake is now high-DOWN in 0.8-2.1% of windows.
  - **Deciles of the population rate cannot absorb a multiplicative shared change.** In the
    shared-ramp run the top rate decile holds 90% (LEC) and 94% (PFC) of its windows from the
    first 5 min, spanning 9.8-14.5 Hz per unit in LEC. The rate keeps falling inside that one
    decile, and the rest-entry kernel takes what is left. G3's SWS gain multiplies the shared
    process in the same way.
  - G6 at 120 s and G7 rest on 2 OU units per tau and on 12 segments in the quick gate. They
    moved in both directions across runs: 0.68 and 1.47; 1.36 and 0.91.
- **User decisions, 2026-10-16:** divide each unit's counts by the other half's population rate;
  and give G6 and G7 more units and segments.
- **The divisor has to be robust.** Ramping units move a summed population rate, and dividing by
  it gives every other unit the opposite ramp. Shift of each half's summed rate at rest entry from
  the planted rest-entry ramps, computed from the gate's truth and task rates:

  | gate | ramp units per half | shift |
  |---|---|---|
  | LEC quick / full | 3 / 6 of 77 | -0.6 to +3.4% |
  | PFC quick / full | 3 / 6 of 60 | -6.9 to +17.7% |

  So the divisor, `rest_kernels.relative_pop_rate`, is the median over the half's units of each
  unit's count over the centred 30 s, divided by its own mean. Units under 0.3 Hz are left out,
  because their counts are too sparse; a half needs 10 qualifying units, and the divisor is floored
  at 0.2. The planted ramps stay strong, so G4 tests the robustness. Also: 4 OU units per tau in
  the quick gate, and `MAX_SEGMENTS` 40.
- **Jobs, round 4:** quick gates 3771769 (LEC) and 3771771 (PFC); full gates 3771772 and 3771773,
  `afterok`; recday jobs LEC 3771779-3771803 and PFC 3771805-3771830, `afterok` on the full gates.
- **Round 4 result, 09:42: both quick gates fail; the held jobs were cancelled.**

  | row | LEC quick | PFC quick |
  |---|---|---|
  | G3 SWS-gain units locked with the state terms | 3 of 4, fail | 4 of 4, fail |
  | G4 noise, with / without the state terms | 1 of 84, pass / 7 of 84 | 1 of 50, pass / 20 of 50 |
  | G4b shared ramp, noise with / without | 3 of 84 / 77 of 84 | 8 of 50 / 49 of 50 |
  | G6 slow tau ratio at 120 s, 4 units | 0.855, pass | 1.443, fail |
  | G7 drift_equal ratio | 0.703 on 21 segments, fail | 1.017 on 25, pass |

  - The median divisor works: noise stays at chance with the state terms, and the shared ramp is
    absorbed.
  - **G3 had a coding cause.** `down_frac` is 0 in about 36% of windows. With so many ties the
    quantile edges collapse, and `np.digitize` puts everything from 0 to the next distinct edge in
    one bin: 0-0.172 in LEC and 0-0.147 in PFC. That bin held 5% (LEC) and 4% (PFC) of the
    planted SWS windows, so the SWS gain leaked into time since entry. Now 0 is its own bin, with
    deciles over the positive values.
  - **G6 and G7 are still short of data.** G7's equal-drift ratio has read 0.70-1.36 across eight
    runs. So drift takes every non-overlapping segment, capped at 120 per context, and G6 plants
    8 OU units per tau. G6 also scores rest windows clear of the planted quiet wake, as G7 does,
    because that 0.7x step adds slow structure that the planted OU does not have.
- **Jobs, round 5:** quick gates 3771856 (LEC) and 3771857 (PFC); full gates 3771858 and 3771859,
  `afterok`; recday jobs LEC 3771862-3771888 and PFC 3771889-3771913, `afterok` on the full gates.
- **Round 5 quick gates pass, 09:59, both `EXIT=0`:**

  | row | LEC quick | PFC quick |
  |---|---|---|
  | G3 SWS-gain locked, with / without the state terms | 0 of 4 / 1.00 | 1 of 4, p 0.177 / 1.00 |
  | G4 noise, with / without | 0 of 68 / 13 of 68 | 2 of 34 / 25 of 34 |
  | G4b shared ramp, noise with / without | 1 of 68 / 63 of 68 | 5 of 34 / 34 of 34 |
  | G6 slow tau ratio at 30 / 120 s | 0.911 / 1.25, 9 windows | 1.08 / 1.256, 11 windows |
  | G7 drift_equal / drift_fast | 0.92 / 1.907, 61 segments | 0.937 / 2.746, 74 segments |

  G1, G2, G5, Murray at 0.25 s and G8 pass. G6 at 120 s sits near its 1.33 limit, on 9-11 rest
  windows clear of quiet wake.
- **Round 5 full gates pass, 10:29, both `EXIT=0`; the 50 recday jobs then started:**

  | row | LEC full | PFC full |
  |---|---|---|
  | G2 rest-entry ramps locked / Stage 2 tau within 2x | 0.833 / 0.8 | 1.0 / 1.0 |
  | G3 SWS-gain locked, with / without the state terms | 0 of 8 / 1.00 | 0 of 8 / 1.00 |
  | G4 noise, with / without | 0 of 54 / 24 of 54 | 1 of 20 / 15 of 20 |
  | G4b shared ramp, noise with / without | 2 of 54 / 49 of 54 | 0 of 20 / 20 of 20 |
  | G6 slow tau ratio at 30 / 120 s | 0.941 / 0.924, 6 windows | 0.986 / 1.124, 12 windows |
  | G7 drift_equal / drift_fast | 0.935 / 2.204, 54 segments | 0.909 / 2.293, 89 segments |

  Murray's tau at 0.25 s is within 2x in every planted unit, median 0.231 and 0.210 s, and G1,
  G5 and G8 pass.

### 2026-10-15: M4 part 1 (rest) approved, in progress

- **Plan:** `~/.claude/plans/please-read-this-handoff-crystalline-pinwheel.md`. Four parts:
  - a spike-based SWS proxy, with an LFP check on 2 blocks per LEC mouse;
  - rest-entry kernels against maze entry;
  - intrinsic timescales in rest vs task vs event tau;
  - rest drift vs task drift.

  Part 2 comes later: offline co-activity and preceding-task memory.
- **Code:** `rest_kernels.py`, `rest_kernels_synthetics.py`, `run_rest_kernels.py` (+ mirrors), and
  the LEC-only `rest_lfp_check.py`.
- **The proxy needs its recday's own calibration.**
  - Real awake data have far more low-count 25 ms bins than Poisson: the LEC ah10 task 1st
    percentile is 4 spikes, against 14 for Poisson at the same rates. So the power check scores
    planted sleep with the real theta and cut.
  - PFC task shards can end in a silent 10 s window, which pushed me11's cut to 1.0. Dead windows
    are now excluded from calibration.
  - In PFC, awake windows spend more than half their time in silent runs (me11's cut is 0.56). The
    proxy then recovers no planted SWS, so PFC staging is flagged unusable. A usable recday needs
    AUC >= 0.95, recall >= 0.8 and quiet wake <= 5%.
- **2026-10-15, afternoon: every rest gate run fails; the production runs are cancelled.**
  - Runs 3769494 / 3769495 (LEC quick / full) and 3769496 / 3769497 (PFC), all `EXIT=1`. G1 now
    passes on both gate recdays: 154 LEC and 46 PFC units, AUC 1.00, recall 1.00, quiet wake 0.042.

    | row | LEC quick | LEC full | PFC quick | PFC full |
    |---|---|---|---|---|
    | G3 SWS-gain units locked with the nuisance (must be chance) | 1/4, pass | 7/8, fail | 4/4, fail | 7/8, fail |
    | G4 noise locked (chance, p > 0.01) | 26/92, fail | 8/70, p 0.022, pass | 2/58, pass | 0/36, pass |
    | G6 Murray tau within 2x at 0.25 s | 2/2 | 2/4, fail | 2/2 | 3/4, fail |
    | G7 drift_equal ratio (0.8-1.25) | 1.15 | 1.12 | 1.06 | 0.73, fail |

    G2, G5, G6 slow, G7 fast and G8 pass in all four. The cause of G3/G4 is not diagnosed.
  - The 50 production jobs (3769498-3769547) were held behind the full gates; the PFC ones could
    never start. All 50 were cancelled.
  - LFP check v2 (absolute delta threshold, the 99th percentile of the nearest task session's first
    10 min): LFP-SWS in 0.00-0.02 of windows in all 10 blocks, so the pre-set AUC rule cannot be
    applied. Spearman of `down_frac` with log delta: 0.60-0.86 in the four between-task blocks with
    variance, -0.08 to 0.45 before the first task. `data/event_kernels/rest/lfp_check/summary.csv`.
- **Results page** (user request, 2026-10-15): https://claude.ai/artifact/557jgXBgUcq5jMarZvJBrp,
  M1 to M4 part 1 with F1-F30. Captioning F4 found the PFC end-of-recording peak in §8.

### 2026-10-14, evening: M3 part 2 complete, stopped for review

Details: `code/EVENT_KERNELS.md` and `code/POPULATION_DRIFT.md`, entries 2026-10-14. Notebook:
`code/event_kernels_M3_part2.ipynb`. Figures F22-F30: `data/figures/{event_kernels,population_drift}/M3_part2/`.
Uncommitted.

- **Runs.** 100 production jobs, every one `EXIT=0`. Gates pass in both datasets after nine documented row
  revisions: five in the unit gate, four in the part-2 gate. Equivalences at merge:
  - held-out deviance = Stage 2 `dev_np` to 2.7e-14;
  - raw ramp-test kNN = the M2 drift arm, exactly.
- **The ramp test.** M2's held-out first-minute decoding is mostly the slow entry ramps' early
  slope, not a fast transient at task start.
  - LEC: dividing out held-out ramps with tau >= 240 s removes 55% of the advantage, against 62%
    for the shape-free kernel.
  - That removal stays flat across floors, as planted slow ramps do; planted 15 s transients fall
    to 33%.
  - The remaining ~40% sits in units Stage 1 did not call entry-locked.
- **Tsao's recipe** gives 17.5% day-time units in LEC (his BW12 session time: 18.2%) and 16.6%
  entry-time units (his trial time: 6.2%).
  - It is in sample: its entry-time units are 70% Stage 1 entry-locked, but it finds a third of
    them, and it calls 33-58% of planted drift time cells.
  - Reward timing goes to place.
- **Kanter's single-unit measures.** His GLM test fires for anything slow (planted drift 80-85%).
  Minute-scale variability is 14-23x Poisson.
  - Once locked structure is removed, entry-locked units are no more slowly variable than others.
  - His ACF fit is biased low by session length: 0.45x at 120 s for our 20-min sessions, 0.31x for
    his 10-min ones.
- **Kanter part 2.**
  - Rewards are boundaries, but stamped by the same ensemble every loop: PV correlation 0.33-0.53
    vs 0.18-0.20 for other units, the opposite of Kanter.
  - The leg and session axes are stable (7-14°) and near-orthogonal (70-78°), short of noise's
    81-85°.
  - The 25% most variable units carry part of the drift.
  - Loop-number decoding is no evidence of time stamps: drift alone does it.
- **Regions (LEC, 4 mice, every mouse the same sign):** SUB/ProS has longer residual ACF tau, more
  residual minute-scale variability and more day-time units than ENTl-deep.
- **Decisions to put to the user:**
  1. Commit M3 part 2? That is the code, both trees, the notebook, the write-ups and the handoff,
     plus `docs/handoff/EVENT_KERNELS_WRITEUP.md`.
  2. Next: M4 (rest), or a follow-up. Candidates:
     - what carries the 40-51% of first-minute decoding left in non-locked units;
     - the region differences;
     - the reward-ensemble contrast with Kanter.
     The which-clock tests stay outstanding (§8).
- **User decisions, 2026-10-14 evening:** commit and push M3 part 2; next is M4 (rest). The three
  follow-ups stay outstanding (§8).

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
