# Event-locked kernels in LEC vs PFC (Tsao 2018, Kanter 2025)

Per-neuron kernels of time since maze entry, first A, reward A and every reward, with their time
constants, in LEC and PFC, read as the state of an SSM trace bank. Plan, user decisions, process
rules and code map: `docs/handoff/EVENT_KERNELS_HANDOFF.md`. Population drift (Kanter) results
will go in `code/POPULATION_DRIFT.md`.

## 2026-10-02 — M1: data preparation (WP0) and the synthetic gate (WP1)

**Outcome.** The data path is exact in both datasets. With the conservative kinematic design
(`kin='strict'`), the pipeline recovers what is planted on real behaviour and does not flag what
is not. The default design does not meet that bar in PFC. No real-data kernel has been fitted
yet; that is M2.

- LEC: every task session was re-extracted from recording start on the v4 clock. All 211 spike
  segments agree byte for byte with `neuron_raw_v4` on the core window. 210/211 pre-A tracks
  agree with `processed_v3` on the overlap.
- PFC: each of the 194 deposit sessions sits in exactly one place in El-Gaby's whole-recording
  shards. XY is in cm for 187/194.
- The loader equals `data_dic_lec_v4.pkl` in every session. Its 250 ms blocks equal GLM V3's;
  time since reward agrees to 1.1e-13 s.
- Synthetic gates, full runs (n_perm 100, one real recday per dataset). The default design
  leaks fast speed tuning into the A and reward kernels in PFC: 5/8 planted units, mostly
  through the untracked segments. The new `kin='strict'` design removes the leak (0/8) and leaves
  recovery of planted event units unchanged: 1.0 for every kind except LEC's superposition
  traces, 0.83 under both designs. Under `strict`, every gate passes in both datasets.
  `strict` is the recommended M2 design; the decision is the user's.

### What was done

| file | role |
|---|---|
| `code/preprocessing/extract_full_sessions.py` | LEC WP0: pre-A and post-last-trial spike segments on the v4 clock, pre-A tracking, recording index, unit stability |
| `mFC_data/code/pfc_session_index.py` | PFC WP0: shard index, recording index, pixel-to-cm affine, behaviour probes |
| `code/event_kernels.py` (+ mirror) | loaders, 25 ms features, 250 ms design, Stage 1, reset test, Stage 2 |
| `code/event_kernels_synthetics.py` (+ mirror) | gates G1-G11 (table in its docstring) |
| `code/population_drift.py`, `code/population_drift_synthetics.py` (+ mirrors) | Kanter measures; gates P1-P4 |
| `code/event_kernels_checks.py` | loader vs the v4 pickle; shared primitives vs GLM V3 |
| `code/event_kernels_pin.py` | the default design is bit-identical to a committed version; what `kin='strict'` changes |
| `code/event_kernels_leak_diag.py` | the A+reward joint's held-out gain for planted speed units, by row type |
| `sbatch_files/event_kernels.sbatch` | runs any of these on `gpu_lowp` after the mirror parity check |

Design choices and the measurement behind each are in the handoff, section 5.

### WP0, LEC: 25 recdays

| quantity | value |
|---|---|
| task sessions | 211: 191 ABCD, 20 object |
| unit-recordings | 2851; 66-186 per recday, median 108 |
| spike gate: full-session core == `neuron_raw_v4` | 211/211 byte-identical |
| tracking gate: pre-A resample == `processed_v3` on the overlap | 210/211 equal |
| the tracking exception | ah10_20250618_20250619 s5. Its pinstate is empty (0 rising edges), so the copilot has no matched clock. The v3 arrays come from a poke-based salvage. Its 7.5 s before the first A are untracked and coded missing. |
| recording start to first A | median 21.0 s, IQR 12.0-41.6, range 1.8-472.0 (211 sessions) |
| last trial to recording end | median 35.0 s, range 0-1095.7 (211) |
| pre-A tracked fraction | median 1.000, min 0.221; 8/210 sessions below 0.9 |
| first tracked frame | the first video frame in 209/210 sessions. Video starts a median 0.0 s after recording start (max 15.3 s). The exception, ly05_20250618_20250619 s0, is continuously tracked only from 23.1 s. |
| first poke | median 9.6 s after recording start (191 ABCD sessions); first poke to first A median 12.0 s |
| analysis sessions (ABCD, >= 10 trials) | 136 over 25 recdays; 1-8 per recday, median 6 |
| El-Gaby's 20% rule, diagnostic only | would drop 245/2851 unit-recordings (8.6%): ah08 8.7%, ah10 7.8%, ly05 11.4%, ly06 9.8%, ly07 5.4% |

The mouse is in view from the first frame, so it is already in the maze when recording starts.
Recording start is the earliest anchor available, not the true entry.

### WP0, PFC: 25 recdays from the deposit

| quantity | value |
|---|---|
| recdays, sessions, units | 25, 194, 1252 |
| shard search | 194/194 sessions have exactly one exact match |
| wall clock | 167/194 sessions. Four recdays carry synthetic tokens and have no wall clock: ab03_29082023_30082023, me08_10092021_11092021, me10_14122021_15122021, me10_20122021_21122021. The reset test cannot run on them. |
| shard start to first A | median 18.0 s, IQR 8.3-34.4, range 0.2-500.7 (194) |
| XY to cm | `xy_ok` 187/194. Dropped: me08_12092021_13092021, 7 sessions (affine 1.13 px/cm, residual 27.4 cm; its XY is not a maze track) |
| fitted scale | 2021 cohort 19.57-20.10 px/cm (18 recdays), 2023 cohort 15.83-16.13 px/cm (6). El-Gaby's measured value fits only the first. |
| node residual, kept recdays | worst node per recday: median 0.78 cm, max 1.78 cm |
| reward probe | population PSTH peak a median 0.00 s from reward (194 sessions) |
| speed probe | rate-speed lag median -0.47 s, IQR -0.82 to +0.33 (187 sessions); LEC's reference is -0.30 s |
| running speed | median 10.3 cm/s (187) |

The deposit's pixel files are untouched. The transform is applied at load from
`mFC_data/data/processed/pfc_xy_to_cm.v2.csv`.

### Data-path checks (`event_kernels_checks.py`, job 3675174)

| check | sessions equal |
|---|---|
| loader core counts == `Neuron_raw` in `data_dic_lec_v4.pkl` | 211/211 |
| XY, NaN-equal | 211/211 |
| Locs codes | 191/191 (object sessions have none) |
| `Trial_times` | 182/182 (sessions with trials) |
| 250 ms block counts == V3 `downsample_session_data(10, 'bin')` | 177/177 (ABCD sessions with >= 2 trials) |
| block speed and signed acceleration, 2 blocks inside the core's edges | 177/177 each |
| time since reward, max abs difference | 1.1e-13 s over 794,897 blocks |

No session the loader builds is missing from the pickle. The script fails if an analysis session
is missing, because a missing session would otherwise read as a pass.

### WP1, synthetic gate

Planted Poisson units on one real recday per dataset: LEC ah10_20250616_20250617, PFC
me11_05122021_06122021, 8 analysis sessions each. The spikes are replaced; timings, tracking,
place, pokes and the wall clock are real and enter through the same loader dicts. Quick runs use
n_perm 20 and 80 units. Full runs use n_perm 100 and 160 (LEC) or 158 (PFC) units. A unit counts
as locked to a block when the Freedman-Lane p < 0.05 and the held-out cpd > 0.

**Quick passes.** Three were needed. Each fix is in the first, uncommitted version of the
modules, so there is no version bump.

| pass | failed | cause | fix |
|---|---|---|---|
| 1 | G5: speed units locked in 0.50 (LEC) and 0.25 (PFC) | half the hits had cpd <= 0 at the p floor (1/21) | the locked rule needs cpd > 0; 4x more speed units |
| 1 | G7, LEC: day beat entry in 0.67 of trend units | hat functions of hours cannot extrapolate to a held-out first or last session (held-out RSS 7095 for the day model vs 5803 for entry) | day term linear per calendar day plus a day-2 offset, not reference-coded |
| 2 | G9, PFC: 1 of 2 place units flagged by `ev_A` | chance: cpd +0.0001 at the p floor, like the noise hits; Fisher p vs noise 0.24. Fixed cut-offs fail a pure-null process 16-29% of the time at these n. | chance rows became exact one-sided tests at 0.01; 8 place units; a no-place arm as G9's positive control |
| 3 | none | | |

**Full runs, linear bank** (`kin='full'`, jobs 3675170/71, n_perm 100). Values are the fraction of
planted units flagged; n is per dataset.

| gate | n | LEC | PFC |
|---|---|---|---|
| G1 noise, any event block or the A+reward joint | 60 | 0.07 | 0.10 |
| G2 entry ramps, tau <= 30 min, on `ev_entry` | 16 | 1.0 | 1.0 |
| G3 A ramps | 6 | 1.0 | 1.0 |
| G4 reward ramps, bumps, fixed-width bumps, progress and distance clocks | 6, 6, 4, 2, 2 | 1.0 each | 1.0 each |
| G4 reward superposition traces | 6 | 0.83 | 1.0 |
| G5 speed units tau 2 s, any event or the joint | 8 | 0.38 (Fisher p 0.031 vs noise) | **0.62 (p 0.002): fails** |
| G5 speed units tau 128 s | 8 | 0.00 | 0.00 |
| G5 speed units without kinematics | 16 | 0.62 | 0.50 |
| G6 OU drift on `ev_entry` | 8 | 0.00 | 0.00 |
| G7 day beats entry in trend units / entry beats day in ramps | 6 / 16 | 1.0 / 1.0 | 1.0 / 1.0 |
| G8 speed units on `ev_entry`, zero-init vs steady bank | 16 | 0.19 vs 0.06 | 0.06 vs 0.00 |
| G9 place cells, any event: with / without the place block | 16 | 0.06 / 0.50 | 0.12 / 0.69 |
| G10 Stage 2: ramp class and sign, tau within 2x | 32 | 0.91, 0.91 | 0.91, 0.94 |
| G10 Stage 2: bump class | 6 | 1.0 | 1.0 |
| G11 p < 0.05 with shifts >= 0 vs >= 2 min: entry ramps / OU / noise | 20 / 8 / 60 | 0.90 vs 0.90, 0 vs 0, 0.05 vs 0.05 | 0.90 vs 0.90, 0 vs 0, 0.07 vs 0.05 |

**The G5 failure, and why the gate first passed it.** The run passed every row as first written.
Two faults in the gate hid a leak.

- G3/G4 count a unit as A- or reward-locked if `ev_A`, `ev_reward` **or** the A+reward joint
  passes. G5 and G9 did not count the joint.
- G5 pooled fast and slow speed units, which halves its sensitivity. Fast speed history tracks
  reward timing; slow speed history does not.

Re-scored with one hit rule and per tau, PFC's tau 2 s speed units are locked in 5/8. All five
sit at the p floor (1/101), through the joint, with joint cpd +0.0004 to +0.0015. That is 5-10%
of what their `kin_fast` block explains (cpd 0.011-0.017). LEC shows 3/8 through `ev_A` or
`ev_reward` alone, with cpd near +0.0002. The re-scored tables are
`data/event_kernels/synthetics/gates_rescored_<ds>_<rd>.csv`.

**Where the leak comes from.** `event_kernels_leak_diag.py` rebuilds the full PFC synthetic and
splits the joint's held-out SSE gain by row type. Gains are summed over leave-one-session-out
folds for the 8 tau 2 s units; positive means the joint helps.

| rows (PFC, n blocks) | linear bank, one missing indicator | `strict` |
|---|---|---|
| tracked (34,385) | -1.7 to +15.4 | -40 to -14 |
| untracked before the first A (290) | -7.9 to +15.5 | -0.8 to +0.3 |
| untracked after the last trial (1,941) | -22.7 to +21.9 | -2.3 to +4.1 |

- PFC has no tracking outside the core. One `kin_missing` indicator cannot tell the segment
  before the first A from the one after the last trial. The A and reward blocks can: their
  pre-event and overflow columns mark those segments. So any speed-driven rate difference
  between the segments became an "event" gain.
- The same coefficients then spill into tracked rows through shared lag bins. With the segments
  absorbed, the tracked-row gain turns negative for every unit, which is pure overfitting.
- Curvature is the candidate second path. The bank enters linearly while the planted rate is
  exp(0.5 z). LEC's 3/8 must come from tracked rows, since this recday has no untracked pre-A
  rows. The full `strict` run tests it; the quick runs cannot, because the quick linear-bank run
  already had 0/4.

**The fix: `build_design(kin='strict')`.** It is a new option; the default `'full'` is unchanged.

| change | why |
|---|---|
| decile one-hots of each fast-bank trace (speed and \|accel\|, tau <= 8 s) inside `kin_fast`: 54 columns | tracked-row curvature |
| `kin_missing` split into before-first-A, after-last-trial and gap indicators | the segments' means go to the nuisance |
| A, reward and poke-exit columns zeroed on untracked rows | no fast-event claim rests on rows without kinematics |

The entry clock is not masked, because its first seconds are the pre-A rows. The plan already
rules out first-minute kinematic claims in PFC.

`python code/event_kernels_pin.py --against 57eac39` checks both sides:

- the default design is bit-identical to the committed version in three configurations;
- `strict` differs only in those three ways, and its split indicator sums to the default one.

**Full runs, `strict`** (jobs 3675492/93, n_perm 100, same planted units): every row passes in
both datasets.

| gate | n | LEC | PFC |
|---|---|---|---|
| G1 noise, any event block or the A+reward joint | 60 | 0.10 | 0.07 |
| G2 entry ramps, tau <= 30 min, on `ev_entry` | 16 | 1.0 | 1.0 |
| G3 A ramps | 6 | 1.0 | 1.0 |
| G4 reward ramps, bumps, fixed-width bumps, progress and distance clocks | 6, 6, 4, 2, 2 | 1.0 each | 1.0 each |
| G4 reward superposition traces | 6 | 0.83 | 1.0 |
| G5 speed units tau 2 s, any event or the joint | 8 | 0.25 (Fisher p 0.24) | 0.00 |
| G5 speed units tau 128 s | 8 | 0.00 | 0.00 |
| G5 speed units without kinematics | 16 | 0.62 | 0.50 |
| G6 OU drift on `ev_entry` | 8 | 0.00 | 0.00 |
| G7 day beats entry in trend units / entry beats day in ramps | 6 / 16 | 1.0 / 1.0 | 1.0 / 1.0 |
| G8 speed units on `ev_entry`, zero-init vs steady bank | 16 | 0.19 vs 0.06 | 0.19 vs 0.00 |
| G9 place cells, any event: with / without the place block | 16 | 0.06 / 0.50 | 0.19 (Fisher p 0.16) / 0.69 |
| G10 Stage 2: ramp class and sign, tau within 2x | 32 | 0.88, 0.91 | 0.94, 0.97 |
| G10 Stage 2: bump class | 6 | 1.0 | 1.0 |
| G11 p < 0.05 with shifts >= 0 vs >= 2 min: entry ramps / OU / noise | 20 / 8 / 60 | 0.90 vs 0.90, 0 vs 0, 0.07 vs 0.05 | 0.90 vs 0.90, 0 vs 0, 0.05 vs 0.03 |

- **What remains is noise-sized.** Every remaining hit has a cpd no larger than the noise
  units' (up to +0.0004 for noise):
  - LEC's two tau 2 s speed units: +0.0001 to +0.0007;
  - PFC's three place cells: <= +0.0001;
  - two PFC OU drift units (+0.0004 each through `ev_reward`; one also +0.0018 through the
    joint, as under `full`). 2/8 against 4/60 noise is chance.
- **G8 now shows the start-up artefact the steady-state bank prevents.** With a zero-initialised
  bank, 3/16 speed units read as entry-locked in both datasets. With the steady-state bank it is
  1/16 (LEC) and 0/16 (PFC).
- **Cost.**
  - The design grows from 94 to 149 (LEC) and 147 (PFC) columns. The minimum fold rank is 147
    of 149 (LEC) and 147 of 147 (PFC).
  - Planted-unit recovery is unchanged.
  - On real PFC data, masking removes the A / reward kernels from the untracked segments: in
    me11_05122021_06122021 that is 2,247 of 36,632 blocks (6%).

**Recovery envelope** (full runs, linear bank).

- **Entry ramps.**
  - Detected at every tau from 30 s to 30 min: 16/16 in both datasets.
  - At tau 2 h, the falling ramp is detected 2/2 but the rising one 0/2. The rising version runs
    at a third of the base rate, so it carries fewer spikes.
  - Stage 2 tau is within 0.94-1.33x of truth up to 8 min.
  - At 30 min the median is 0.71x (LEC) and 1.72x (PFC), range 0.50-2.43x. At 2 h it is 0.43x
    and 1.01x, range 0.21-1.01x. A 20 min session cannot pin a slower tau, so tau >= 30 min
    should be read as "slower than the session", not as a number.
- **A and reward ramps.** tau within 1.0-1.41x of truth, all classed ramp-down.
- **Bump peaks.** Planted 1, 3 and 8 s are recovered as 1.00, 2.83 (the grid point nearest 3)
  and 8.00 s. Bumps are classed as bumps in 10/10, fixed-width ones included.
- **Shape errors.** Entry ramps are right in 17/20; 2 are classed bump and 1 none.
- **Superposition reads as an entry ramp.** A leaky reward count is also flagged by `ev_entry`:
  - LEC: 2/2 at tau 30 s and 120 s, 1/2 at 8 s;
  - PFC: 2/2 at all three taus.

  It rises over the session as rewards accumulate. An entry-ramp call can therefore be a leaky
  reward count. M5's reset-vs-superposition test is what separates them.
- **Minimum shift.** A 2 min minimum Freedman-Lane shift changes no verdict (G11), so no
  `min_shift` option goes into `glm_cv`.
- **Under `strict`** (same planted units):
  - entry ramps are classed right in 16/20 (LEC) and 18/20 (PFC);
  - tau up to 8 min stays within 0.71-1.41x of truth for entry, A and reward ramps;
  - tau 30 min and 2 h stay unresolved (0.13-3.43x);
  - superposition traces are still flagged by `ev_entry`, in 5/6 (LEC) and 6/6 (PFC).

**Population drift** (full, jobs 3675172/73; P-gates in the docstring of
`population_drift_synthetics.py`): every row passes in both datasets.

| gate | LEC | PFC |
|---|---|---|
| P1 noise kNN accuracy within / across sessions (limit: chance + 0.05) | 0.047 / 0.067 | 0.037 / 0.053 |
| P1 noise distance change from lag 1 to lag 10 | +0.0001 | -0.0004 |
| P2 locked clock, across-session accuracy / chance | 17.6 | 16.1 |
| P3 OU drift, within / across, each over chance | 13.7 / 0.92 | 12.7 / 0.80 |
| P3 OU drift, distance change from lag 1 to lag 10 | +0.21 | +0.21 |
| P4 boundary ensembles, z at lag 0 or 1 / largest abs z of the other populations | 8.9 / 0.95 | 9.5 / 0.71 |

### Deviations from the approved plan

1. **Identity check.** The planned CPD identity with V3 became the primitives identity above.
   V3's design differs on purpose (mode place coding, NaN-row drop, transition mask, core-only
   rows). A CPD identity would have tested a re-implementation of V3, not this pipeline.
2. **PFC XY gate.** El-Gaby's px/cm fits only the 2021 cohort. The gate became one pooled
   affine per recday plus a 5% session scale check where >= 6 nodes were visited.
3. **Missing kinematics are coded, not dropped.** This keeps the PFC pre-A rows and the entry
   kernel's first seconds.
4. **Stage 2 picks tau** by a training-fold profile, not an inner leave-one-session-out; an 8 x 7
   inner loop was too slow. Evaluation stays on held-out sessions.
5. **Gate statistics.** The locked rule and the exact chance tests (quick passes above).
6. **Day term.** Linear per day, not hat functions.
7. **Kinematic design `kin='strict'`** (decile fast bank, per-segment missing indicators, fast
   events masked where untracked). It was added after the full gate exposed the PFC leak. It is
   an option; the approved default is unchanged.
8. **Deferred to M4.** The SWS proxy per rest block was not computed in WP0. The rest synthetics
   (rest-entry ramps confounded with SWS onset, OU units with a known intrinsic timescale) were
   not built. Both are only used in M4 and will be built and gated with it.

### What was NOT established

- Any real-data result: kernels, time constants, reset test, drift.
- The envelope above comes from one recday per dataset with 8 analysis sessions. Recdays with
  fewer sessions are outside it. ly05_20250618_20250619 has 1 analysis session, so no held-out
  fit is possible there. ah08_20250624_20250625 has 2.
- True maze entry. The mouse is in the maze before recording starts, so entry-kernel lags are
  left-censored by an unknown amount.
- ah08's speed reference (+0.25 to +0.40 s against -0.30 s in the other mice): physiology or a
  residual offset is still open. Reward-kernel latencies will be reported per mouse.
- PFC's day-scale ramps will be a lower bound. El-Gaby's 20% rule pre-filtered the deposit and
  cannot be undone.
- `strict` on real data. Its decile bank corrects curvature only to decile resolution. A neuron
  with stronger nonlinear speed-history tuning than the planted exp(0.5 z) could still leak; the
  per-unit kinematic cpd should be reported next to each event call.
- Whether a slow "entry" ramp is an entry clock or a leaky reward count. Superposition traces
  with tau >= 30 s are flagged by `ev_entry` under both designs; M5's test is still needed.

### Commands

```
python code/check_mirror_parity.py                       # must exit 0
# WP0, LEC (per mouse; tables versioned, never overwritten)
sbatch --partition=gpu_lowp --job-name=fullsess_<mouse> --time=0-02:00 --mem=32G \
    sbatch_files/mec_stage.sbatch extract_full_sessions.py --recdays <rd ...> --write
# WP0, PFC
sbatch --partition=gpu_lowp --job-name=pfcidx --time=0-02:00 --mem=48G \
    sbatch_files/mec_stage.sbatch ../../mFC_data/code/pfc_session_index.py --write
# data-path checks (loads the v4 pickle)
sbatch --job-name=ek_checks --mem=64G sbatch_files/event_kernels.sbatch code/event_kernels_checks.py
# WP1, quick then full
sbatch --job-name=ek_syn_lec_q3 sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec --quick
sbatch --job-name=ek_syn_pfc_q3 sbatch_files/event_kernels.sbatch mFC_data/code/event_kernels_synthetics.py --dataset pfc --quick
sbatch --job-name=ek_syn_lec sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec
sbatch --job-name=ek_syn_pfc sbatch_files/event_kernels.sbatch mFC_data/code/event_kernels_synthetics.py --dataset pfc
sbatch --job-name=pd_syn_lec sbatch_files/event_kernels.sbatch code/population_drift_synthetics.py --dataset lec
sbatch --job-name=pd_syn_pfc sbatch_files/event_kernels.sbatch mFC_data/code/population_drift_synthetics.py --dataset pfc
# the conservative kinematic design: pin the default first, then gate it
python code/event_kernels_pin.py --against 57eac39            # exit 0 = default bit-identical
sbatch --job-name=ek_syn_lec_st sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec --kin strict
sbatch --job-name=ek_syn_pfc_st sbatch_files/event_kernels.sbatch mFC_data/code/event_kernels_synthetics.py --dataset pfc --kin strict
# where a leak comes from: the joint's held-out gain by row type, both designs
sbatch --job-name=ek_diag_pfc --time=0-01:00 sbatch_files/event_kernels.sbatch code/event_kernels_leak_diag.py pfc me11_05122021_06122021
```

Logs: `sbatch_files/slurm_outputs/event_kernels-<name>-<id>.out`, read to the `EXIT=` line.
Outputs: `data/event_kernels/synthetics/` and `data/event_kernels/checks/`, versioned `.vN`.
