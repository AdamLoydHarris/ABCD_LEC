# Event-locked time in LEC and PFC: where the analysis stands, and the plan for M3 part 2

**2026-10-14.** Written when M3 part 2 was approved. Part 1 is what had been done and found, M1 to
the leaky-count test; Part 2 is what M3 part 2 does and why. The execution steps are in the
handoff (`EVENT_KERNELS_HANDOFF.md`). Results are logged as dated entries in
`code/EVENT_KERNELS.md` and `code/POPULATION_DRIFT.md`; this file is not rewritten when they
change. A later state of the analysis gets a new dated section below.

Every number comes from those two logs and the handoff, except the ABCD timing and clock-offset
numbers, measured on 2026-10-14 from the LEC full-session tables
(`data/processed_data/full_sessions_v4/`) and trial times (`trialtimes_raw_mingyutest/`).

---

# Part 1. What has been done so far

## 1.1 The question

- **Tsao et al. 2018, LEC:**
  - many cells ramp up or down, locked to the start of a trial or of the session;
  - in the BW12 task, 18.2% of LEC cells encoded session time and 6.2% trial time;
  - time constants run from minutes to over 2 h;
  - in a repetitive lap task the balance flips: 35.7% trial time, 5.7% session time.
- **Kanter et al. 2025, LEC:**
  - population activity drifts continuously over minutes;
  - event boundaries shift it abruptly, carried by a different ensemble each time;
  - trial time and session time run on near-orthogonal axes;
  - the single-cell substrate is minute-scale variability, mostly not a monotonic ramp.
- **The user's note, "SSMs, Mamba and music boxes":**
  - ramps are the state of a diagonal state-space model: leaky traces of events, a Laplace
    transform of the past;
  - peaked time cells are its readout;
  - Mamba's input-dependent step is a clock that can stretch into goal progress or reset at
    events.

  Working hypothesis: LEC is a fixed clock and mFC tracks progress.
- **Three things carry time, and the analysis keeps them apart:**

| | what it is | how it is measured |
|---|---|---|
| locked clock, Tsao | a kernel of time since an event that repeats across repetitions of that event | prediction of held-out sessions |
| drift, Kanter | slow change unique to each session | within-session readouts, beyond the locked part |
| boundary shift, Kanter | an abrupt population change at an event | trajectory acceleration; event ensembles |

## 1.2 Data

| | LEC | PFC |
|---|---|---|
| source | our recordings, v4 spike clock | El-Gaby 2024 deposit |
| mice, recdays | 5, 25 | 7, 25 |
| units | 2851; 66-186 per recday, median 108 | 1252 |
| task sessions | 211: 191 ABCD, 20 object | 194 ABCD |
| sorting | one sort per recday over 2 days and 18-24 recordings | whole-recording shards |
| behaviour | XY in cm, pokes, head direction | XY converted from pixels to cm, usable in 187/194 sessions; no pokes |
| curation | all units kept | El-Gaby's 20% rule already applied, so slow across-session change was selected against |

- **A day:** about 38 min of rest, then 4-5 task sessions of about 20 min, each followed by rest.
- **Sessions analysed:** ABCD sessions with >= 10 trials in M2's primary arm. M2's sensitivity
  arm, M3 and the leaky test use >= 5 trials: 164 LEC sessions over 25 recdays and 180 PFC
  sessions over 24 recdays.
- **ABCD timing in LEC:**
  - legs, reward to reward: median 9.3 s, mean 14.4 s, n = 13064;
  - loops, A to A: median 43.6 s, mean 57.5 s, n = 3266.

**The clock origin.** Every "entry" result is time since bin 0.

- **LEC:**
  - Bin 0 is the ephys recording start.
  - The task starts a median 4.6 s later, range 1.5-43 s, over 191 ABCD sessions. Task start is
    the first sync pulse, when the white noise comes on.
  - The first A comes a median 23.5 s after bin 0.
- **Placement is not recorded** (user, 2026-10-14). The mouse goes into the maze up
  to about a minute before the task start, while the rig is set up. Video starts with the
  recording, and the mouse is in view from the first frame.
- **PFC:**
  - Bin 0 is the start of El-Gaby's whole-recording shard.
  - The first A comes a median 18.0 s later.
  - Only the recording start is indexed; the task start is not.
- So "time since entry" below means time since the recording start. That is a few seconds before
  the task start, and up to a minute after placement.

## 1.3 Methods in brief

- **Rows:** 250 ms blocks of each full session, from bin 0 to the end of the recording.
- **Design:**
  - place: the 22 maze location codes;
  - kinematics: speed and acceleration deciles, plus a bank of causal exponential filters of
    speed and |acceleration| with tau 0.5-512 s;
  - event blocks: time since entry, since reward A and since every reward, coded shape-free as
    lag bins.
- **`kin='strict'`,** a user decision, 2026-10-12. It adds three things:
  - decile codes of the fast bank;
  - missing kinematics split by segment;
  - fast events masked where tracking is missing.

  M1 showed that the default design leaked fast speed into the PFC event blocks.
- **Stage 1:**
  - leave one session out;
  - cpd = the held-out variance a block adds;
  - a Freedman-Lane permutation null, 100 permutations;
  - a unit is **locked** when p < 0.05 and cpd > 0. The exact null rate is 5/101 = 0.050.
- **Reset test:** the entry kernel against a linear hours-since-day-start term, on held-out
  sessions, uncentred.
- **Stage 2:** for each locked unit, Poisson fits of a ramp exp(-t/tau) and a gamma bump to the
  event block.
  - tau and peak are chosen on the training sessions and scored on the held-out session;
  - the shape class comes from the fraction of the shape-free kernel's held-out gain that each
    shape recovers.
- **Kanter part 1:**
  - soft-normalised population vectors;
  - cosine distance between 1-min epochs as a function of lag;
  - kNN decoding of the 1-min epoch, within a session and on a held-out session;
  - trajectory acceleration at events.
- **Gates:**
  - Every analysis first ran on synthetic units planted on one real recday per dataset.
  - Only the spikes are replaced. Timings, tracking, place, pokes and wall clock stay real.
  - The planted units go through the same loaders and code as the data.
- **Inference:**
  - Mice are the replicates. A per-mouse value is the recday mean, then the mouse mean.
  - LEC regions come from `anatomy_split`. Units within 50 µm of a boundary are excluded, and
    labels are shuffled within recday.

## 1.4 M1, 2026-10-02: data preparation and the synthetic gate

- **The data path is exact:**
  - all 211 LEC spike segments equal `neuron_raw_v4` byte for byte;
  - 210/211 pre-A tracks equal `processed_v3`;
  - the loader equals `data_dic_lec_v4.pkl` in every session;
  - each of the 194 PFC sessions sits in exactly one place in the deposit's shards.
- **The default design leaked, and `strict` fixed it.**
  - Planted speed-history units, tau 2 s, read as A- or reward-locked in 5/8 PFC units. The
    leak ran through the untracked segments.
  - Under `strict` it is 0/8, and recovery of planted event units is unchanged.
- **Under `strict`, every gate passes in both datasets.** Selected rows:

| gate | LEC | PFC |
|---|---|---|
| noise flagged by any event block, n = 60 | 0.10 | 0.07 |
| entry ramps, tau <= 30 min, n = 16 | 1.0 | 1.0 |
| A ramps; reward ramps, bumps and clocks | 1.0 | 1.0 |
| OU drift flagged as entry-locked, n = 8 | 0.00 | 0.00 |
| day trend beats entry, in trend units / entry beats day, in ramps | 1.0 / 1.0 | 1.0 / 1.0 |
| Stage 2 ramp class, n = 32 | 0.88 | 0.94 |

- **Recovery envelope:**
  - Stage 2 recovers tau within 0.71-1.41x of truth up to 8 min.
  - At 30 min and 2 h the recovery spans 0.13-3.43x. A 20-min session cannot pin a slower tau.
- **A known confound:** a leaky count of rewards is also flagged as entry-locked, 5/6 in LEC and
  6/6 in PFC, because it rises as rewards accumulate. Section 1.8 tests it.

## 1.5 M2, 2026-10-12: Stage 1, the reset test and Kanter part 1

Fraction of units locked at >= 10 trials, mean of mice, with the range over mice:

| block | LEC, 5 mice | PFC, 7 mice |
|---|---|---|
| every reward | 0.508 (0.271-0.653) | 0.610 (0.313-0.815) |
| entry | 0.288 (0.167-0.426) | 0.344 (0.115-0.613) |
| first A, the alternative anchor | 0.271 | 0.244 |
| reward A | 0.117 (0.100-0.152) | 0.158 (0.084-0.304) |
| fast speed bank | 0.434 | 0.618 |
| place | 0.795 | 0.735 |

- **At >= 5 trials:**
  - entry: 0.342 in LEC, 0.322 in PFC;
  - every reward: 0.587 and 0.609;
  - reward A: 0.109 and 0.174.
- **Effects per unit are small.** The median held-out cpd of locked units is about 0.005 for entry
  and 0.004 for reward. For place it is 0.028 in LEC.
- **Kinematics compete more in PFC.** The event block beats the fast speed bank in 68% of LEC
  entry- and reward-locked units, against 55-58% in PFC.
- **Entry calls are not day-long trends.** On held-out sessions, the entry kernel beats a linear
  hours-since-day-start term in:
  - 82% of LEC entry-locked units, per mouse 67-95%;
  - 72% of PFC entry-locked units, per mouse 42-94%.
- **Regions:** no difference is established.
  - ENTl-deep is more reward-locked than SUB/ProS by +0.13, rate-matched, 4 mice, p 0.023.
  - Two of the four mice sit near zero, so it is a candidate.
- **Population, Kanter part 1,** at a matched 20 units:
  - within a session, the 1-min epoch decodes at 2.9x chance in LEC and 2.3x in PFC;
  - on a held-out session it decodes at 1.5x in both, almost only in the first minute: 0.25 in
    LEC and 0.26 in PFC there, against 0.07 in minutes 2-17 and a chance of 0.056;
  - population distance grows about 27% from a 1-min to a 10-min lag;
  - rewards shift the trajectory: LEC about 1 s after the reward, PFC at the reward. The first A
    does not.
- M2 read the first-minute decoding, and the entry kernels' early peak, as a transient. M3
  corrected that, in 1.7.

## 1.6 Runner V3, 2026-10-12

- **An artefact in the first-A anchor was removed.**
  - One column duplicated the A block's pre-A column.
  - When a recday's only untracked pre-A rows were held out, that column took an arbitrary
    share of the pre-A offset. 8 LEC recdays were affected.
- **The rank check now flags only real leaks:** null-space directions that reach held-out rows
  through a tested column. No shard has one.
- **Entry-anchor results are unchanged:** cpd agrees to 3e-15, and every locked call is identical.

## 1.7 M3 part 1, 2026-10-12: kernel shapes and time constants

- **Entry kernels are minutes-scale ramps, and most rise.**
  - 85% of entry-locked LEC units are ramps or ramp-like broad bumps. In PFC it is 54%, with 27%
    fitting neither shape.
  - Ramp tau: median 320 s in both datasets. The middle half is 193-905 s in LEC (n = 567) and
    160-640 s in PFC (n = 158). 17% are slower than the session.
  - Two thirds of LEC entry ramps rise over the session, and 59% in PFC.
  - So M2's "transient" reading was wrong. A 5-min ramp differs most from a late reference bin
    in the first minute, which is where its largest kernel weight falls.
- **Reward kernels are brief.**
  - Bumps peak a median 1.0 s after the reward.
  - Ramps have tau 2.0 s in LEC and 1.4 s in PFC. About 80% rise: the post-reward dip
    recovering. Most reward-locked units are suppressed right after the reward.
- **Reward-A calls are the weakest.** For 31% of LEC and 44% of PFC reward-A passers, the
  shape-free kernel gains nothing on held-out sessions.
- **Tau does not differ by region.** LEC entry-ramp medians are 273-593 s across five regions;
  reward medians are 1.6-2 s.

## 1.8 The leaky-count test, 2026-10-13

A leaky count of rewards rises over a session much as an entry ramp does. The test fits both to
each entry-locked unit, chooses each one's tau on the training sessions, and compares them on
held-out sessions.

| units | n | count beats ramp | ramp adds unique gain | count adds unique gain |
|---|---|---|---|---|
| LEC entry-locked | 972 | 0.21 | 0.76; median 0.30 of the kernel's gain | 0.43; median -0.003 |
| PFC entry-locked | 414 | 0.40 | 0.51; median 0.05 | 0.46; median 0.002 |

- **The gate separates the two perfectly at tau <= 600 s on real behaviour,** although ramp and
  count correlate at |r| 0.88-0.99 there. Stage 1 alone cannot: it calls 100% of planted counts
  entry-locked.
- **Per mouse,** the count wins in 0.12-0.29 of LEC units and 0.24-0.62 of PFC units.
- **Reading:** in LEC, the slow entry signal is a clock, not accumulated reward. In PFC, about
  half the units carry no unique information from either.

## 1.9 What is established, and what is not

**Established:**

1. **Event-locked units are common in both regions.** Every reward is the strongest anchor, then
   entry, then reward A. Each block exceeds the null in every mouse.
2. **Entry-locked units mostly ramp over minutes,** tau about 5 min, and most rise.
   - They are locked to the session's start, not the day.
   - In LEC they follow time, not accumulated reward.
3. **Reward kernels** are a dip with about 2 s of recovery, plus bumps at about 1 s.
4. **Population state drifts within sessions.** Across sessions, only the first minute is shared.
   Rewards shift the trajectory.

**Not established:**

- **Any LEC vs PFC difference.**
  - None was tested, and with 5 and 7 mice the power is low.
  - PFC event calls compete more with kinematics.
- **Regional differences:** candidates only, each resting on one or two mice.
- **What the first-minute population decoding is.** It could be the ramps' early slope, or an
  early transient, such as a response to the white noise at task start. This is the M3 part 2
  ramp test.
- **Which clock.**
  - Placement is not recorded.
  - Task start and recording start differ by a median 4.6 s.
  - Leaky counts of trials or of distance were not tested.
- **Time constants beyond the session:** 17% of entry ramps, tau >= 30 min.
- **Per-unit power at real effect sizes.** Planted units modulate the rate about 3x, while real
  held-out cpd is about 0.005.
- **Rest, the remaining SSM tests and the extensions:** M4-M6.

## 1.10 Outstanding, not run

- **Which clock.** The user's clarification of 2026-10-14 changes this test.
  - Placement vs recording start cannot be tested, because placement is not recorded.
  - For an exponential ramp the origin cannot be identified anyway. Moving the origin by d
    multiplies the amplitude by exp(-d/tau) and leaves tau unchanged. So the time constants and
    shapes stand, whichever origin is true.
  - What can be tested, in LEC only, is task start vs recording start, a median 4.6 s apart. It
    matters only for a transient in the first seconds.
  - Leaky counts of trials or of distance remain untested.
- **Later milestones:** M4, rest; M5, the SSM tests, of which the leaky test was brought forward;
  M6, extensions.

---

# Part 2. M3 part 2: what will be done, and why

## 2.1 Purpose

- **Put our results on the papers' own scales.**
  - Tsao's fractions and time constants come from an in-sample stepwise GLM. Kanter's
    single-unit timescales come from a GLM, autocorrelation and Fano factors.
  - Running their recipes on our units makes the numbers comparable.
  - Setting them beside our held-out calls shows what the in-sample recipes add or miss.
- **Answer the population questions M2 and M3 left open:**
  - What carries the first-minute decoding?
  - Are reward boundaries stamped by a different ensemble each time?
  - Do within-leg time and session drift use orthogonal axes?
  - Is drift carried by a few highly variable units?

## 2.2 How the papers map onto the ABCD day

| paper | ours |
|---|---|
| Tsao session time, 1.3 h including intertrial periods | `day_time`: hours since the calendar day's first recording, rest included. It resets on day 2 |
| Tsao BW12 trial time, a 250 s box exposure | `entry_time`: seconds since recording start |
| Tsao figure-eight trial time, one lap ending in one reward | `leg_time`: seconds since the last reward. A leg ends in one reward, as a lap did |
| no analog in either paper | `loop_time`: seconds since the last A, kept as an extra candidate. Stage 1 locked 316 LEC units to reward A, against 1380 to every reward |
| Tsao wall colour | `task`: the session's reward configuration, categorical |
| Tsao position | `place`: the 22 location codes, one categorical term |
| Kanter session, 10 min of foraging | each task session, from bin 0 to the end |
| Kanter boundary, the reward | arrival at a reward, per reward type A-D across loops. Each type sits at one place within a session, as Kanter's single reward did |
| Kanter within-trial time, the 6 s before the reward | the 6 s before each reward arrival, in legs >= 8 s. Pooled over the four rewards, and per reward type |

- **Sessions:** ABCD with >= 5 trials, as in Stage 2 and the leaky test. `kin='strict'` wherever
  a design is built.
- **Skips:** the usual rules, fewer than 2 sessions or no tracking at all. That skips PFC
  me08_12092021_13092021.
- **No wall clock:** four PFC recdays, so they lose the `day_time` term.

## 2.3 Where the paradigms differ, and what the plan does about it

| difference | consequence in the plan |
|---|---|
| Both papers' lap tasks give one reward per lap. ABCD gives four per loop | the leg is the trial: `leg_time`, a boundary at every reward, the leg axis. Legs >= 8 s keep the 6 s window clear of the previous reward; they are 60% of legs |
| Kanter's laps all end on the same stretch of the central stem, so his within-trial time is largely also place. The four ABCD legs run different paths | the leg axis is computed pooled over rewards, giving a time-to-reward axis largely free of place, and per reward type, Kanter's place-confounded analog |
| Rewards come every ~9 s and a loop takes a median 44 s, both inside Kanter's 1-90 s autocorrelation fit. Foraging has no period | the raw autocorrelation carries these periods, so its exponential fit is biased. The residual autocorrelation, after the held-out model, is the intrinsic-timescale measure. Both are reported, and the raw curves are shown |
| Behaviour changes within an ABCD session: legs lengthen, rho about -0.3. Foraging is closer to stationary | Fano is computed on raw counts, as Kanter did, and on counts divided by the held-out prediction |
| Kanter's boundary bin is the time of the population shift. M2 measured ours on a 1 s grid not aligned to the reward: LEC shifts about 1 s after the reward, PFC at the reward | the shift bin is where the reward-aligned population acceleration peaks, per dataset. The full preferred-bin histogram is shown |
| Both papers start the clock at placement. Ours starts at recording start, a median 4.6 s before the task start and up to a minute after the unrecorded placement | the entry clock stays time since recording start. Ramp time constants do not depend on the origin. The ramp test separates a fast early component, which could be a response to the white noise, from the slow ramps' early slope |
| Tsao's BW12 session held 12 trials in 1.3 h. Our day holds 4-5 sessions over about 5 h with rest, and a recday spans 2 days | `day_time` resets each calendar day |
| Tsao fitted in sample | his table stays a side table. T7 measures how often it calls drift a time cell, and F22 cross-tabulates it against held-out Stage 1 |
| Mice average 71-158 LEC units and 14-96 PFC units per recday, and angles between axes depend on dimension | LEC is compared with PFC only at matched size, using M2's draws, with the stability control and P15 |
| El-Gaby's 20% rule pre-filtered the PFC units | wherever LEC is set against PFC, LEC is shown with and without the rule |
| Rats in both papers, mice here; tetrodes for Tsao | no change. Comparisons with the papers are of fractions and time constants, not absolute rates |

## 2.4 The analyses

### A. Tsao's stepwise table

**What Tsao did.**

- Per cell, a Poisson GLM fitted with MATLAB's `stepwiseglm` on 500 ms rates, unsmoothed.
- BW12 variables: wall colour, position, trial time and session time, one predictor each. Session
  time counted the intertrial periods. The figure-eight model used trial type, trial time and
  session time.
- A term entered if it raised adjusted R² by 0.01 and left if it contributed less than 0.005.
  The fit was in sample.
- A cell was selective for the terms in its final model.
  - BW12: 18.2% session or mixed session time, 6.2% trial or mixed trial time.
  - Figure-eight: 5.7% and 35.7%.
  - Objects: 34.0% and 7.7%.
- Time-selective cells were then fit with f = a e^(bx): session-time cells on the rate across all
  trial periods, trial-time cells on the rate averaged across trials. Tsao reports b; we report
  tau = 1/|b| in seconds.

**What we do.**

- **Rows:** 500 ms bins over each session's core, first A to last trial end. Tsao fitted only
  trial periods.
- **Candidate terms:** `day_time`, `entry_time`, `leg_time` and `loop_time` are linear, one
  predictor each; `task` and `place` are categorical, one term each.
  - Place codes with < 20 bins merge into unknown. A rare code with no spikes has no finite
    estimate.
- **The same thresholds,** in sample.
- **The code is Python, statsmodels.** It copies the rules of `stepwiseglm` that the paper does
  not state:
  - start from the constant model;
  - main effects only;
  - a categorical term enters or leaves whole;
  - the criterion is adjusted R² of the fitted mean, 1 - (n-1)/(n-p) SSE/SST.
- **Checked against MATLAB itself.** `matlab/R2018a` is on the cluster.
  - If the Statistics Toolbox licence is there, `stepwiseglm` runs on 40 units, 20 planted and
    20 real.
  - It must select the same terms, with adjusted R² within 1e-6.
  - A missing licence is reported.
- **Exponential fits as Tsao:**
  - `entry_time` cells on the rate averaged across sessions;
  - `day_time` cells on all core bins;
  - `leg_time` and `loop_time` cells on the rate averaged across legs or loops, truncated to the
    shortest complete one.
- All units in both datasets, cross-tabulated against Stage 1's held-out calls for the same
  units.

**What it tells us.**

- Our fractions on Tsao's scale.
- How many in-sample time cells survive a held-out test.
- Whether his method gives our entry ramps the same time constants as Stage 2.
- What his recipe does with reward coding. `leg_time` is linear, so it catches the slow recovery
  after a reward, not the ~1 s bumps.

**Gate.**

- Planted on the gate recday's real behaviour, through the real runner.
- Planted kinds come from `event_kernels_synthetics.make_synthetic`, which is not edited. Two
  local kinds, task offsets and pure exponentials in rate, are made in the gate module.

| row | planted | must hold |
|---|---|---|
| T1 | entry ramps, tau 60-300 s | `entry_time` selected in >= 0.8 |
| T2 | day trends | `day_time` in >= 0.8 |
| T3 | reward ramps, tau 8 s | `leg_time` or `place` in >= 0.8. `leg_time` alone is reported, since Tsao found lap time and position covary |
| T3b | A ramps, tau 30 s | reported: which of `loop_time`, `leg_time` and `place` enters |
| T3c | reward bumps, peak 1 s | reported: how many the linear recipe catches |
| T4 | place cells | `place` in >= 0.8; any time term <= 0.1 |
| T5 | task offsets | `task` in >= 0.8 |
| T6 | noise | any term <= 0.05. This is the control for T1-T5 |
| T7 | OU drift, tau 30-300 s | reported: how often the in-sample recipe calls drift a time cell |
| T8 | pure exponentials in rate, tau 300 and 600 s | the exponential fit recovers tau within 1.25x |

### B. Kanter's single-unit timescales

**What Kanter did,** on 10-min foraging sessions in 0.5 s bins:

- **GLM:** a Poisson GLM, MATLAB `glmfit`, on unsmoothed counts, with elapsed time and the mean
  population rate as predictors. Units with p < 0.05 for elapsed time were fit with
  f = a e^(bx), tau = 1/b.
- **Autocorrelation:** of the unsmoothed counts, positive lags only, divided by its maximum,
  with an exponential fit over 1-90 s.
- **Fano:** variance over mean of the rate smoothed with a 30 s Gaussian. It was divided by the
  mean over 500 synthetic Poisson neurons, then logged, so 0 means Poisson-like variability.
  LEC was compared with MEC and CA1.

**What we do:**

- Per unit and session, the full session from bin 0.
- **The GLM as Kanter,** except that the unit itself is left out of the population rate.
- **Autocorrelation and Fano twice:**
  - on raw counts, as Kanter did;
  - on held-out residuals.
- **The residual model** is the Stage 2 full model: place, kinematics, and the entry, reward-A and
  reward kernels, fitted on the other sessions.
  - It removes locked and behavioural structure and leaves session-unique change.
  - Its held-out deviance must equal Stage 2's `dev_np` to relative 1e-9. That proves it is the
    same model.
- **Why the residual versions are needed here.** Our task repeats every ~9 s and ~44 s, inside the
  1-90 s fit range, and behaviour changes within a session. Kanter's foraging had neither.

**What it tells us:**

- whether units carry minute-scale variability beyond Poisson once task structure is removed;
- their intrinsic timescales;
- how Kanter's elapsed-time units relate to our held-out entry-locked units.

**Gate:**

| row | planted | must hold |
|---|---|---|
| K1 | entry ramps | GLM elapsed-time p < 0.05 in >= 0.8 of sessions |
| K2 | noise | GLM false-positive rate <= 0.1; median log Fano within 0.1 of 0 |
| K3 | OU drift, tau 30 and 120 s | autocorrelation tau within 2x; tau 300 s reported for its session-length bias; log Fano > 0 |
| K4 | entry ramps vs OU drift | the residual autocorrelation tail, mean over 30-90 s, falls >= 50% for ramps and stays within 25% for drift. Each is the other's control |

### C. The ramp test

**The question.** Held-out decoding of the 1-min epoch is 0.25 in minute 1 and 0.07 after it.
Which of these produces that?

- the slow entry ramps' early slope;
- a fast component locked to bin 0, such as a response to the white noise at task start, a
  median 4.6 s after bin 0.

**Method.** M2's held-out epoch kNN, with the same settings and the >= 5-trial sessions, on four
inputs:

1. raw counts;
2. each Stage 1 entry-locked unit divided by its held-out shape-free entry kernel. This removes
   everything locked to bin 0;
3. each divided by its held-out slow ramp only, tau >= 60 s, profiled on the training sessions
   as Stage 2 does;
4. the entry-locked units alone, against size-matched non-locked units.

**Reading:**

- If input 3 removes the first-minute advantage as fully as input 2, the decoding is the ramps'
  slope.
- If only input 2 removes it, there is a fast locked component beyond the ramps.
- Within-session accuracy is reported for every input. It is the internal control: removing a
  locked kernel should leave drift-driven within-session decoding largely intact.

**Gate:**

| row | planted | must hold |
|---|---|---|
| K5 | slow ramps, tau drawn from the real Stage 2 spread, 190-900 s | held-out decoding above chance raw, and within 0.05 of chance after input 2 and after input 3 |
| K5b | an early transient: an entry ramp with tau 15 s | the first-minute advantage falls to chance after input 2 and survives input 3. K5 and K5b control each other |
| K5c | OU drift; every unit divided, the worst case for noise added by division | no held-out advantage either way; within-session accuracy changes < 10% |

### D. Kanter's population measures

**D1. Boundary responses at rewards.**

What Kanter did, on the figure-eight:

- windows from -3 to +4 s around the reward, in 1 s bins;
- each neuron's preferred time: the bin with the largest absolute change in trial-averaged rate
  from the preceding bin, chance 1/6;
- more neurons preferred the shift time than chance;
- neurons modulated up or down at the shift had a lower mean population-vector correlation across
  trials than random 25-neuron sets preferring other times, p 0.03 and 0.006. So each event got
  a distinct time stamp;
- trial number was decoded by kNN.

What we do:

- The same windows around each reward arrival, per reward type across loops.
- The shift bin is where the reward-aligned population acceleration peaks, per dataset. M2
  predicts 1-2 s in LEC and 0-1 s in PFC.
- Each job computes every bin, and the merge picks the shift bin.
- Loop number is decoded as Kanter decoded trial number.

| row | planted | must hold |
|---|---|---|
| P5 | a fixed boundary ensemble: the same 20% of units at every reward, mixed amplitudes | shift bin preferred >= 3x chance; PV correlation of positive units >= 0.3 |
| P6 | the existing M1 population: a different 20% at each reward | PV correlation of positive units <= 0.1. P5 and P6 control each other |
| P7 | noise | shift bin preferred within chance +/- 0.05 |
| P8 | OU drift, no boundaries | reported: whether drift alone decodes loop number below the shuffle. If it does, that measure is not specific to boundary ensembles |
| P9 | noise | loop-number error >= 0.9x the shuffle |

**D2. Orthogonal coding axes.**

What Kanter did:

- **Session axis:** PCA to 50% of the variance, then LDA on 1-min epochs of 10 s bins.
- **Trial axis:** the same on the 6 s before the reward, in 1 s bins.
- **Each axis** is the loadings of the PC with the largest weight in the first discriminant.
- **Offset** = arccos|L1 . L2|. It was about 90°, against a leave-one-out stability of about 15°.
- **Multiplexing:** units above the 75th percentile of both axes' |loadings|.

What we do:

- The session axis exactly as Kanter.
- The leg axis on the 6 s before each reward arrival, legs >= 8 s:
  - pooled over rewards;
  - per reward type.
- The loop axis, secondary: 1 s bins since A, truncated to the fastest complete loop.
- Size-matched draws wherever LEC is set against PFC.

| row | planted | must hold |
|---|---|---|
| P10 | leg-locked trajectories plus independent drift | leg vs session offset >= 60°, and above both stabilities |
| P11 | leg trajectory and drift on one shared unit axis | offset <= 30° |
| P15 | noise | P10's criterion fails, so orthogonality cannot be claimed by chance in high dimensions |

**D3. Fano, and removing the most variable units.**

- **What Kanter did:** removing the top 25% of units by Fano cut the distance travelled more than
  removing a random 25%.
- **What we do:**
  - drift growth = the distance at 5-10 min lags minus the distance at 1 min, from M2's
    distance-vs-lag;
  - compared with the top 25% removed, against a random 25% removed, 50 draws.

| row | planted | must hold |
|---|---|---|
| P12 | drift in 25% of units | removing the top 25% cuts drift growth >= 50% more than random removal |
| P13 | drift in all units | removing the top 25% is within 20% of random removal |
| P14 | noise | median log Fano within 0.1 of 0 |

## 2.5 What M2 and M3 part 1 predict

The report will state which of these held.

| analysis | prediction | basis |
|---|---|---|
| Tsao time terms | `entry_time` is selected more often than `day_time` | the reset test: entry beat the day term in 82% of LEC and 72% of PFC entry-locked units |
| Tsao trial terms | `leg_time` is selected more often than `loop_time` | Stage 1: 1380 reward-locked vs 316 reward-A-locked LEC units |
| Tsao vs Stage 1 | the recipe catches the strongest locked units and adds drifting ones | its 0.01 adjusted-R² floor, against a median held-out cpd of about 0.005; and it is in sample |
| Tsao on reward coding | `leg_time` catches the slow recovery after a reward, not the ~1 s bumps | Stage 2: reward kernels are ~1 s bumps and ~2 s recovery from a dip |
| exponential tau | of the order of Stage 2's entry tau, median 320 s | Stage 2 |
| residual autocorrelation | the long-lag tail falls in entry-locked units, not in others | entry kernels are locked; the remaining drift is session-unique, M2 |
| boundary bin | preferred above chance, with more negative than positive units | M2 acceleration at rewards; most reward-locked units are suppressed after the reward |
| loop-number decoding | below the shuffle from drift alone | within-session epoch decoding at 2.9x chance, M2 |
| leg vs session axes | close to orthogonal, above the stability angles | goal-progress coding is shared across sessions while drift is session-unique |
| ramp test | the first-minute advantage falls toward 0.07 under input 2. Whether input 3 matches it is the open question | the M2 caution; the white noise at task start |

## 2.6 What M3 part 2 cannot establish

- **The Tsao table is in sample.** A selected term is not evidence of a locked clock; Stage 1
  stays the test.
- **Kanter's measures are descriptive per session.** Contrasts between regions or datasets are at
  mouse level, with 5 and 7 mice. A difference stays a candidate unless it is large and
  consistent across mice.
- **Placement as the clock's origin** cannot be tested.
- **Rest and sleep,** including Kanter's drift in REM, are M4.

---

# 2026-10-14, evening: what M3 part 2 found

The full numbers, gates and commands are in `code/EVENT_KERNELS.md` and
`code/POPULATION_DRIFT.md` (entries 2026-10-14). The figures are F22-F30, in
`code/event_kernels_M3_part2.ipynb`.

**The ramp test.** M2's held-out first-minute decoding is mostly the slow entry ramps' early
slope, not a fast response at task start.

| first-minute held-out advantage removed | shape-free | ramp >= 60 s | >= 120 s | >= 240 s |
|---|---|---|---|---|
| LEC, real | 0.62 | 0.58 | 0.58 | 0.55 |
| LEC, planted 15 s transients | 0.98 | 0.66 | 0.55 | 0.33 |
| PFC, real | 0.49 | 0.40 | 0.38 | 0.37 |
| PFC, planted 15 s transients | 0.60 | 0.40 | 0.35 | 0.15 |

Planted slow ramps are removed fully, about 1.0, at every floor. The rest of the real advantage,
40% in LEC and 51% in PFC, sits in units Stage 1 did not call entry-locked.

**Tsao's recipe.**

- 17.5% of LEC units select his session-time term (day time); he reports 18.2%. 16.6% select his
  trial-time term (entry time); he reports 6.2%.
- In sample, its entry-time units are 70% Stage 1 entry-locked, against a 34% base rate. But it
  finds a third of the entry-locked units, and calls 33-58% of planted drift units time cells.
- Reward timing goes to place.
- Its exponential tau, a median 2125 s, is not comparable with Stage 2's 320 s, because the fit
  has no baseline.

**Kanter's measures.**

- His GLM test of elapsed time fires for anything slow, 80-85% of planted drift. It reads 76% of
  real unit-sessions.
- Minute-scale variability is 14x Poisson in LEC and 23x in PFC. The PFC excess is not
  rate-matched.
- Once the held-out model removes locked structure, entry-locked units are no more slowly
  variable than the rest.
- His autocorrelation fit is biased low by session length: 0.45x of tau at 120 s in our 20-min
  sessions, 0.31x in his 10-min ones.
- Rewards are boundaries, but stamped by the same ensemble every loop: PV correlation 0.33-0.53
  for modulated units, 0.18-0.20 for others. That is the reverse of Kanter's figure-eight result.
- The within-leg and session axes are stable (7-14°) and near-orthogonal (70-78°). Noise axes read
  81-85° but are unstable (50-68°).
- Loop-number decoding is no evidence of time stamps, since drift alone does it.

**Regions, LEC, 4 mice, each the same sign.** SUB/ProS shows more slow, session-unique variability
than ENTl-deep: residual ACF tau about 1.5x, residual log Fano +0.20, day-time selection +0.13.

**Predictions in §2.5.**

- Held:
  - leg time over loop time;
  - the recipe's high precision and low recall;
  - reward timing as slow recovery and place;
  - the residual tail falling only in entry-locked units;
  - the boundary preference;
  - drift alone decoding loop number;
  - stable, near-orthogonal axes.
- Did not hold:
  - entry time selected over day time: in sample, the day term also takes drift;
  - exponential tau near 320 s: the no-baseline fit.
- Partly held: the ramp test's fall toward 0.07.
