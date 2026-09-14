# W3 — Generalising state cells and coherent remapping, by region

**Status: COMPLETE (2026-09-02).** Results and methods are in `code/ANATOMY_SPLIT.md` §W3;
the notebook is `code/LEC_anatomy_state_remapping.ipynb`. The rest of this document is the
plan as approved, kept as the record of intent — read §W3 of `ANATOMY_SPLIT.md` for what was
actually found.

## Outcome in three lines

- **Generalising state cells: a null.** 12 cells in 2851 unit-recordings. Cross-task rate
  0.257 against a chance of ¼ — **recdays p=0.34, mice p=0.31**, only 14/23 recdays above
  their own shuffle — against an X-vs-X′ ceiling of 0.81. The estimator works; the constancy
  is not there. Every region at chance; rate matching moves nothing; true for all 6
  reference-task choices.
- **Coherent remapping: positive.** On the **dual (two-comparison, chance 1/16)** criterion,
  0.109 against a shuffle of 0.063: **recdays p=1×10⁻⁹, mice p=0.002, 24/25 recdays beat
  their own shuffle**. Holds on both the reference-anchored and reference-free metrics, and
  survives restriction to pairs that started ≥90° apart (22/23 recdays, mouse p=0.001) — so
  it is not co-place-tuned pairs.
- **The coherence is not anatomical.** Same-region minus cross-region = +0.010 with only
  **10/19 recdays and 3/4 mice positive**, label-shuffle p=0.088, and the largest mouse runs
  the other way. Coherence modules recover the region labels at exactly chance (ARI 0.000,
  1/25 recdays at p<0.05). ENTl-deep and SUB/ProS rotate *together*.

Four decisions taken in session that depart from the plan below:

1. **Coherence is the dual, two-comparison criterion** (X→Y *and* X→Z), so chance is
   **1/16**, not 1/4. The synthetic control pins this: an independently-rotating population
   returns 0.068 against an analytic 0.0625.
2. **The recday is the aggregation atom**, and both recday (n≈25) and mouse (n=5) levels are
   plotted with significance at each. Pooling a mouse's pairs would weight each recday by
   `n_pairs ~ n²`.
3. **X-vs-X′ is its own polar histogram**, never an overlay — it peaks so hard at 0 that
   overlaying it flattens the real distribution.
4. The leg-duration / time-cell confound (§5) was **not run**; and a sixth figure was added —
   coherence against initial pairwise tuning distance — because it is what rules out shared
   place tuning as the source of the coherence.

Depends only on W0 — this workstream reads `Neuron_raw` / `Trial_times` directly and **never
touches the GLM**, so it ran while W1's refit was on the cluster.

---

## 1. What is being asked, and why the framing matters

Two questions that are really one analysis seen from both ends.

**"Do we have A cells, B cells, C cells, D cells?"** means **generalising state cells**: cells
that keep the same task-state preference *across tasks*. Not the within-task state-tuning
threshold — that is a precondition, not the result.

This distinction is forced by the task. Each recday has **6 unique tasks, and `A` is a different
physical port in every one of them**. So state labels are purely ordinal, and a within-task
preference alone is just "fires on one leg" with an arbitrary letter attached. What makes the
letters mean anything is *constancy across ports*: a generalising A cell fires on the first leg
of the sequence in every task, even though "first leg" is a different place each time.

**"Coherent remapping"** asks whether, even when cells do remap, the population rotates as a
rigid body — whether the *pairwise* angle between two cells is preserved across tasks.

Both reduce to a **remapping angle** on the 360-bin task-space rate map (1 bin = 1°, one state
= 90°). Generalising = angle ≈ 0. Coherent = pairwise angle constant.

## 2. Run order: gate first

**The feasibility of half this workstream depends on one unmeasured number** — the fraction of
neurons passing the state-tuning gate. Measured per-region unit counts, and what survives
(recdays / mice with ≥10 neurons of one region recorded simultaneously, the floor for pairwise
coherence):

| assumed pass rate | ENTl-deep | SUB/ProS | ENTm | CA1/HPF | primary contrast |
|---|---|---|---|---|---|
| no gate | 25 / 5 | 20 / 4 | 9 / 3 | 10 / 2 | — |
| 40% | 17 / 4 | 15 / 3 | 5 / 1 | 5 / 2 | **feasible** |
| 25% | 9 / 3 | **1 / 1** | 2 / 1 | 0 / 0 | **collapses** |

So Stage A decides whether Stage C exists. Stage B has no minimum-n floor and is built either
way.

---

## Stage A — the gate and the feasibility table (blocking)

**Gate definition: the El-Gaby peak-z t-test alone**
(`identify_state_tuned_neurons_raw`, `code/elasticnet_regression_v3.py:238`), at parity with the
reference work. Per neuron: peak firing per state per trial → z-score across states within
trial → mean across trials → preferred state = argmax → one-sample t-test of that state's
per-trial z against 0, p < 0.05.

**Deliverables:**

1. **Pass rate per region**, reported next to the firing-rate table from W0. This is a result in
   its own right — it is the "do we have state cells here" number — *and* the selection
   statistic for everything downstream.
2. **Post-gate feasibility table.** Any contrast that cannot clear ≥10 neurons per region in ≥2
   mice is **dropped, with the table showing why published**.

**The gate is rate-dependent and that is not fixed.** Its power scales with firing rate, and
rate is confounded with region (SUB/ProS 5.86 Hz vs ENTl-sup 1.07 Hz). This was a deliberate
decision — keep parity with the published method and report the pass rate as the selection
statistic it is. It does not remove the confound. Everything downstream inherits it.

**A prior worth stating before you start.** The production GLM ranks `task_state` **last of 16
in LEC** (−0.00015 even after bias correction) and negative in PFC. The two measures differ —
peak firing per state versus a 3-column step function competing against place — so this is not
decisive. But if the gate also comes back weak, **W3 is a null result and should be reported as
one**, not pursued into ever-finer splits.

---

## Stage B — generalising state cells, per neuron (always feasible)

Per-neuron rotation vs the reference task, joined to anatomy. Generalising = |angle| < 45° on
every task comparison (45° = half a state, so a cell that moved to a neighbouring state cannot
pass).

**Two angle metrics, both run:**

- **(i) Correlation-maximising circular shift** — shift task 1's map against task 2's, take the
  shift maximising Pearson r. A 30-bin shift ⇒ 30°. Uses the whole curve shape.
- **(ii) Peak-bin difference** — circular difference of argmax bins. Peak at bin 10 in task 1
  and bin 100 in task 2 ⇒ 90°. Uses only the mode.

They disagree informatively: (i) is sensitive to overall profile and multi-peaked cells, (ii) is
shape-robust but noisy for broad tuning. **Report both per region plus their agreement rate** —
a region where they diverge has broad or bimodal tuning, which is itself a finding.

**Headline number: fraction of generalising cells per region.**

**Level 3 — which letter?** Over generalising cells only, per region: the distribution across
A/B/C/D, tested for non-uniformity (χ², circular resultant). Because `A` is a different port in
every task, a *generalising* letter preference is substantive in a way a within-task one is not.
But whether an A- or D-peak is *interpretable* depends on whether those states are behaviourally
privileged (session/trial start, tone). **Establish that and record it**; if they are not
privileged, report distribution shape only, not "more A cells".

### Already built and tested — do not rewrite

`code/anatomy_split.py` (W0) provides all the angle machinery, synthetic-tested:

| function | notes |
|---|---|
| `remapping_angle(a, b, method='xcorr'\|'peak')` | returns **signed degrees in (−180, 180]**. Verified: 0/±30/±45/±90/±180 recovered exactly by both methods |
| `pairwise_angles(curves, method)` | all-pairs, batched FFT. **7 h → 2.5 s** for the cohort |
| `circular_xcorr(a, b)` | FFT; matches the O(n²) loop to 7×10⁻¹⁶ |
| `is_generalising(angles, threshold_deg=45)` | all comparisons must pass |

Two properties to know:

- **Antisymmetry holds mod 360, not exactly.** A half-turn is its own inverse, so an exact 180°
  pair returns −180 in both directions. Harmless (both thresholds fail, circular statistics
  handle it) — but do not assert `M == -M.T`.
- `pairwise_angles` was validated on the discrimination this analysis depends on: a **rigidly
  rotated** population preserves every pairwise angle to 0.00e+00; an **independently rotated**
  one changes them by a median of 87° (≈ chance).

---

## Stage C — coherence, only for contrasts Stage A clears

**Primary metric: the pairwise angle measured *within* each task, checked for constancy across
tasks.** For neurons *i,j*, cross-correlate their two 360-bin maps within task *t* ⇒ Δ_ij(t).
The pair is coherent if Δ_ij(t) is preserved — 90° apart in task 1, 90° in task 2, 90° in task
3. Statistic: circular spread of Δ_ij across tasks, thresholded at 45°.

**This is not what the existing code computes.** `_relative_pairs`
(`code/remapping_rotation_analysis.py:264`) derives the pair angle from each neuron's rotation
**vs one reference task**. Algebraically they agree when each neuron's phase is well defined —
r_i(t) − r_j(t) = Δ_ij(t) − Δ_ij(1) — but the direct version drops the reference anchoring, so
one bad reference task cannot corrupt every pair at once. **Direct version primary,
`_relative_pairs` as parity; where they disagree, the reference task is the suspect.** Cell 337
of `code/LEC_sploratory_analysis_with_glm_and_population.ipynb` is the existing prototype.

**Three framings, in order:**

1. **Per-neuron rotations by region** — this is Stage B's measure, read a second way: fraction
   near 0 = generalising cells; the spread and structure of the rest = remapping.
2. **Within-region coherence** — `coherent_prop` restricted to within-region pairs.
3. **Cross-region module test** — within-ENTl-deep vs within-SUB/ProS vs ENTl×SUB pairs. Each
   region internally coherent but mutually incoherent ⇒ two independently rotating modules.
   Plus: do the incoherence-matrix `cluster_labels` align with `group`? **This is the most
   likely place in the whole project for a genuinely new claim.**

**Statistical care specific to pairs:** `n_pairs ~ n²`, so pair counts differ enormously between
regions and the noise floors do not match. **Subsample to equal pair counts**, and make the
permutation null **resample neurons, not pairs**.

---

## 3. Code changes required (all outstanding — the file is unmodified)

`code/remapping_rotation_analysis.py`:

- **Export the join keys.** `analyse_recday` returns `n_included` (a count) and nothing
  joinable. Verified: `included_idx` currently appears in that file only as a *parameter name*
  inside helpers. Add `out['included_idx']`, `out['valid_neur_idx']` (for `cluster_labels`), and
  the `(i, j)` pair index array. **Nothing downstream can reach anatomy without this.** Additive
  only.
- **Generalise `pairwise_tuning_angles_ref`** (line 240) from the reference task to *all* tasks,
  using `anatomy_split.pairwise_angles`. The existing Python shift loop will not finish: at ~80
  neurons × 6 tasks that is ~3.4M pairwise correlations per recday.
- Add the **peak-bin metric**; set `rotation_step_single = 1` for 1° resolution.

`code/ccgp_state_pairs.py`:

- Add **`neuron_subset`** to `build_task_state_matrices` (line 175). Currently absent.

## 4. Population cross-check — CCGP

State-pair decoders are the population form of the same question: a decoder trained on some
tasks and tested on a held-out task generalises only if state identity is abstract across tasks.

Run per region **twice**:

- **Matched n ≈ 20 — for contrasts.** Decoder accuracy scales with n and per-recday region
  counts differ two-fold. Feasibility at n=20: ENTl-deep in ah10/ly05/ly06/ly07, SUB/ProS in
  ah10/ly06/ly07 — **the primary contrast survives at 3 mice**.
- **Full n — for sensitivity.** Every region at its actual count, to see whether there is *any*
  signal above its own null. **A within-region existence claim against that region's own null,
  never a cross-region comparison** — full-n ENTl-deep (80 units) beating full-n SUB/ProS (41)
  is expected from n alone. Print n on every point.

`geometry__ccgp_join.pkl` already pins `n_neurons_used = 40` for every recday, so neuron-count
matching is this pipeline's existing convention. Note its `region` column is the *dataset* label
(`LEC`), not anatomy.

## 5. The confound to measure, not assume away

**Leg durations differ, and that manufactures state cells then selectively destroys them.**

A fixed-latency time-since-reward cell fires only on legs longer than its latency, so it passes
the Level-1 peak-per-state t-test as a "state cell". Worse for Stage B: because `raw_to_norm`
warps each leg to 90 bins, such a cell sits at a **different normalised phase in every leg**,
and its apparent rotation changes when tower distances change between tasks. So it is
**selectively removed at Level 2** and will depress the generalising fraction in whichever
region is richest in time cells.

Deliverables: cross-tabulate Level-1 state-tuned × reward-time-tuned per region (using the W1
`time_from_reward` results), and report leg-duration spread per recday. If the regions with the
lowest generalising fraction are the regions richest in time cells, **the finding is reported as
confounded**.

## 6. Reliability floor

`compute_splithalf_remapping_angles` (`code/splithalf_ratemap_consistency.py:400`) gives the
within-session noise floor for a rotation estimate, per region. It is rate-dependent in the same
direction as the Level-1 gate, so it bounds how much of any regional difference could be
measurement noise. Cheap, existing code, and partly answers what the gate leaves open.

## 7. Verification — synthetic controls, run through the real pipeline

Blocking, per repo practice:

- **Remapping angle**: a curve rotated by a known 30 bins returns 30° from the xcorr metric;
  peaks moved bin 10 → bin 100 return 90° from the peak-bin metric. *(Already passing in W0.)*
- **Pairwise coherence**: a rigidly-rotated synthetic population gives ~100% coherent pairs; an
  independently-rotating one sits at ~1/num_states. Direct and `_relative_pairs` metrics agree
  on both.
- **Generalising-state detection**: a cell with a fixed ordinal-state preference across tasks is
  classified generalising; one with a fixed *place* preference is not.
- **The leg-duration confound**: a synthetic fixed-latency time-since-reward cell must **pass
  Level 1 and fail Level 2**. If it does not, the cascade is not measuring what it claims.
- Region contrasts report n mice, per-mouse points, and a within-recday permutation null, and
  are preceded by per-region panels.

## 8. Deliverables

- `code/LEC_anatomy_state_remapping.ipynb` — Stages A, B, C in one notebook. **Do not split
  Stage B and Stage C across notebooks:** they share one remapping run, and Stage C's framing 1
  *is* Stage B's measure.
- Figures under `data/figures/anatomy_split/`.
- A section appended to `code/ANATOMY_SPLIT.md` in that document's register.
