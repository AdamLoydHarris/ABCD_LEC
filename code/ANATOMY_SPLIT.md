# Splitting the LEC dataset by brain region

Methods record for the anatomy split. Written in the register of
[PROBE_REFIT.md](histology_refit/PROBE_REFIT.md): what was done, what it showed, and what
each result does *not* license. One section per workstream; sections appear as they land.

The recorded probe fits say the "LEC" bank spans ENTl (superficial and deep), ENTm,
SUB/ProS and CA1/HPF, in proportions that differ wildly between the five mice. Per-unit
labels are in `data/processed_data/unit_regions.pkl`, one row per QC unit in `Neuron_raw`
row order, for all 25 recdays.

---

## W0 — Foundation

Status: **complete**. Code: [w0_gates.py](w0_gates.py), [anatomy_split.py](anatomy_split.py).
Cached results: `data/processed_data/w0_gates.pkl`.

### W0.1 Gates

| # | Gate | Result |
|---|---|---|
| 1 | `len(unit_regions[rd]) == Neuron_raw.shape[0]` | **PASS 25/25** |
| 2 | HD column identity: `earL2earR − back2mid ≈ ±90°` | **PASS**, 187 sessions |
| 3 | Depth ordering consistent within each mouse | **PASS 4/5 mice** (ly07 below) |
| 4 | Unit quality per region, published first | done — and it is a problem, see below |
| 5 | Positive control: spatial tuning higher in CA1/HPF and SUB/ProS | **PASS on the graded measure; FAILS on the one the plan specified** |

`load_data_dic` also re-ran both registry validators: 25 recdays, 0 mismatched; 191 sessions
checked against their own pyControl `active_poke`, 0 mismatched.

#### Gate 2 — the head-direction columns are what the docstring claims

Across 187 sessions with a `(T, 2)` `HD_raw`, the signed circular difference
`earL2earR_deg − back2mid_deg` has median |difference| **84.8°**, and **98.3%** of samples
sit within 25° of ±90°. So column 0 is `back2mid_deg` as documented, and the fix in W0.2 is
taking the right column. 25 sessions have no HD at all.

#### Gate 3 — depth ordering, and one boundary that is not resolvable

Median `y_um` per region orders consistently with the insertion trajectory in every mouse.
Two apparent exceptions are not failures:

- **ah10** has ENTl-deep and fibre/other tied at exactly 300 µm in one recday. A tie is not
  a reversal.
- **ly07** genuinely reverses ENTl-deep against ENTm between recdays, but their medians are
  separated by only **15–75 µm**. The local deformation field runs 8.06–12.95 µm/voxel and
  boundary uncertainty exceeds that, so these two groups are **not separable in ly07**.

That second point has teeth: ly07 is the ENTm-rich mouse (179 of 284 ENTm units), so it is
the animal that would supply any ENTm claim. **Do not contrast ENTm against ENTl-deep in
ly07 without the 50 µm boundary-margin filter.** The primary contrast is unaffected —
ENTl-deep (135–210 µm) and SUB/ProS (390–442 µm) are separated by more than 200 µm.

#### Gate 4 — region is confounded with firing rate, and badly

Median firing rate, pooled (from `clustering__meta.pkl`, joined by region):

| group | n | median rate (Hz) | median sd | median r²_full |
|---|---|---|---|---|
| ENTl-sup | 675 | **1.07** | 0.172 | 0.010 |
| ENTl-deep | 933 | **1.91** | 0.227 | 0.018 |
| ENTm | 276 | 2.28 | 0.242 | 0.010 |
| SUB/ProS | 505 | **5.86** | 0.391 | 0.021 |
| CA1/HPF | 213 | 1.93 | 0.253 | 0.018 |

SUB/ProS fires **3× faster than ENTl-deep** and **5.5× faster than ENTl-sup**, and `r²_full`
tracks it. Every statistic whose power scales with rate — the state-tuning t-test, the
permutation significance threshold, the reliability of a rotation estimate — inherits this.

**The direction is not consistent across mice**, which matters more than the pooled ratio:

| mouse | ENTl-deep | SUB/ProS | ratio |
|---|---|---|---|
| ah10 | 1.95 | 4.57 | 2.35 |
| ly05 | 1.35 | 8.75 | 6.48 |
| ly06 | **8.43** | **5.61** | **0.67** |
| ly07 | 3.07 | 10.35 | 3.38 |

ly06 has the ratio **inverted**. So the rate confound cannot be described as a fixed
regional property and cannot be reasoned away — it has to be handled per mouse, which is
what `anatomy_split.rate_match` does (stratified on log rate, within recday).

#### Gate 5 — the positive control passes, but not as the plan specified it

The plan asked for spatial tuning to be higher in CA1/HPF and SUB/ProS than in ENTl-sup,
within ah10 and ly05, and said a failure blocks everything.

Measured as the **fraction of units tuned to place**, it fails: CA1/HPF is the *lowest*
group (0.728) and SUB/ProS (0.849) barely exceeds ENTl-sup (0.824). But that statistic is
**saturated** — 80.0% of all 2651 units are place-tuned, with a range across regions of only
70.3–85.6%. A binary permutation F-test at p<0.05 over ~48,000 time bins passes almost
everything; it has no dynamic range to discriminate regions with.

Measured as the **place CPD** — graded, no ceiling — the control passes, and passes in the
direction predicted:

| contrast | difference | 95% CI (mice) | n mice | per-mouse |
|---|---|---|---|---|
| SUB/ProS − ENTl-deep | +0.0028 | +0.0010, +0.0046 | 4 | positive in **all four** |
| CA1/HPF − ENTl-deep | +0.0021 | +0.0011, +0.0031 | 2 | positive in both |
| CA1/HPF − ENTl-sup | +0.0048 | +0.0035, +0.0062 | 2 | positive in both |
| SUB/ProS − ENTl-sup | +0.0048 | −0.0003, +0.0080 | 3 | ly06 ≈ 0 |

Median place CPD by region: ENTl-sup 0.0037 < ENTl-deep 0.0058 < ENTm 0.0064 <
CA1/HPF 0.0077 < SUB/ProS 0.0082.

**Conclusion: the anatomy map is validated and the pipeline is not blocked.** The gate is
amended to use place CPD; the binary tuned fraction is not a usable discriminator anywhere
in this project and should not be used for a regional comparison.

### W0.2 The head-direction fix

`HD_raw` is `(T, 2)`. `prepare_session_data` did `HD_raw.flatten()`, interleaving the two
columns into a `2T` vector; `truncate_all_arrays` then cut every array to the shortest, `T`.
So the GLM received `back2mid[0:T/2]` at even indices interleaved with `earL2earR[0:T/2]` at
odd indices — **half the session, mixed with a ~90°-rotated copy of itself, and offset from
every other regressor from the second sample on**. Verified: `T == n_locs` in 187/187
sessions and the flattened length is exactly `2T` in all of them; median session is 48,242
bins (20.1 min). Every HD result fitted before this change is noise.

Fixed by `_extract_head_direction`, which takes column 0 with a shape guard, a 1-D
passthrough, NaN padding when short and truncation when long. Mirrored verbatim into
`mFC_data/code/glm_analysis_v2.py` (inert — PFC has `HD_raw=None`) per repo convention.

Synthetic control: on a ramp with a known +90° second column, the fix returns the first
column exactly (r = 1.000) where the old code returned a vector correlating r = 0.756 with
it and differing by up to 179.3°.

### W0.3 `anatomy_split.py`

The join and the inference design, in one place. Four notebooks each writing their own
`groupby('group')` is how the mouse-vs-recday confound gets dropped from one of them.

Two guards are worth naming, because the length gate alone does not cover the second:

- `join_regions` refuses to join a result whose length disagrees with `unit_regions`, and
  **drops rather than truncates** — truncation would silently keep the first N neurons of a
  misaligned array. It correctly refuses `ly05_20250618_20250619` (91 cached rows vs 109
  units).
- `assert_glm_keys_contiguous` catches a subtler failure. `compute_tuning_arrays` writes row
  *k* for the *k*-th key of `sorted(GLM_results[rd])`, so the row index is a **position in
  the sorted key list, not a neuron id**. A cache missing one neuron would shift every
  subsequent row, and would pass the length gate whenever the count happened to match.
  Checked: all 24 cached recdays are contiguous `0..n-1` except the known-stale ly05.

**`tuned_dict` is ternary `{-1, 0, +1}`, not boolean** — both the source plan and the
approved plan describe it as bool. +1 is significant with positive/ramp-up direction, −1
significant with negative/ramp-down, 0 not significant. **Taking a mean cancels +1 against
−1**; the tuned fraction is `mean(x != 0)`.

Synthetic controls, all passing:

- `remapping_angle` returns the true rotation for 0/±30/±45/±90/±180°, in both the
  correlation-shift and peak-bin methods; a 30-bin shift gives 30°, and peaks moved from bin
  10 to bin 100 give 90° — the two examples this analysis is specified by.
- `circular_xcorr` (FFT) matches the O(n²) shift loop it replaces to 7×10⁻¹⁶ and picks the
  same argmax.
- `pairwise_angles` matches the single-pair function exactly; a **rigidly rotated population
  preserves every pairwise angle to 0.00e+00**, while an independently rotated one changes
  them by a median of 87° (≈ chance). This is the discrimination the coherence analysis
  depends on.
- Antisymmetry holds **mod 360, not exactly**: a half-turn is its own inverse, so an exact
  180° pair returns −180 in both directions. Harmless downstream (both signs fail an
  |angle| threshold, and circular statistics handle it) but do not assert `M == -M.T`.

Performance: batching the pairwise cross-correlations took the coherence metric from
**~7 hours to ~2.5 s** for the whole cohort (307 µs/pair one-at-a-time, dominated by
per-call overhead, vs 16 ms for all 3160 pairs of an 80-neuron recday).

### W0.4 Which cached fits are stale — the ly05 question

The four pre-existing ly05 recdays have **unchanged task sequences** between the pre- and
post-fix rebuilds, and no non-ly05 recday changed either. What changed is the neural side:

| recday | cached fit | units on disk | status |
|---|---|---|---|
| ly05_20250613_20250615 | 117 | 117 | counts and tasks agree |
| ly05_20250616_20250617 | 116 | 116 | counts and tasks agree |
| ly05_20250618_20250619 | **91** | **109** | **definitively stale** — the cache holds the 06-20/23 block's spikes |
| ly05_20250620_20250623 | *absent* | 91 | **never fitted** |
| ly05_20250624_20250625 | 74 | 74 | counts and tasks agree |

Only `ly05_20250618_20250619` was ever quarantined, and only its **PH** outputs (18 files);
the GLM caches still contain its bad fit.

**All five ly05 recdays are refit in W1.** Two are provably wrong; the other three agree on
both observables but their caches predate the re-extraction pass, so keeping them would give
ly05's contribution two different provenances inside one analysis. Since W1 refits all 25
recdays anyway, this costs nothing.

---

## W1 — Cross-validated GLM engine

Status: **engine built and validated; refit pending the smoke-test timing.**
Code: [glm_cv.py](glm_cv.py), wired into `run_glm_analysis` via `cross_validate=True`.

### W1.1 What the old fits actually did

`run_glm_analysis` fits `np.linalg.lstsq(X, frs)` over **all** samples and computes RSS, R²,
CPD and the nested F on the same samples it fit ([glm_analysis_v2.py:1591](glm_analysis_v2.py#L1591)).
There is no train/test split anywhere in lines 1144–1680. Significance comes from a
permutation test — the firing trace is circularly shifted 100 times, refit in-sample, and a
neuron is called tuned when `F_real > percentile(F_perm, 95)`
([:1751](glm_analysis_v2.py#L1751)).

Because the null passes through identical in-sample machinery, in-sample optimism largely
cancels **for the significance call**. Two things do not cancel:

1. **CPD magnitudes are in-sample and inflated.** A richer or better-placed basis
   mechanically lowers RSS and wins — this is `glm_cv_cpd.py`'s stated reason for existing.
2. **A circular-shift null is permissive for slowly-varying regressors.** A rolled trace
   keeps its autocorrelation, and the animal occupies each maze node for long runs, so place
   alignment survives shifting more than it should.

Point 2 is measured, not hypothetical: **80.0% of all 2651 units are place-tuned, with a
range across brain regions of only 70.3–85.6%** (W0.1 gate 5). The binary measure is
saturated and cannot discriminate regions.

### W1.2 The cross-validated replacement

Leave-one-session-out. Sessions are the right fold unit because they are different tasks, so
LOSO tests **across-task generalisation** — the scientific question — and it matches the
convention already used by `run_cross_validated_regression_v3`.

Both models are fit on the training sessions and scored on the held-out session, with RSS
pooled across folds before forming a ratio (more stable than averaging per-fold CPDs):

```
cpd_cv[g] = (RSS_reduced_heldout − RSS_full_heldout) / RSS_reduced_heldout
r2_cv     = 1 − RSS_full_heldout / TSS_heldout
```

A held-out CPD **can be negative** — dropping a group can improve held-out fit when the group
was only fitting noise. That is signal, and is not clipped.

The in-sample quantities are still computed and returned unchanged, so the two sit side by
side on the same fit. That is also what makes the refit gate checkable: an old cache can be
compared against the new in-sample numbers while the CV numbers carry the science.

**Null.** Permutations shift each neuron's trace **within** each session
(`roll_within_sessions`). Shifting the pooled trace across session boundaries would wrap one
task's firing onto another task's regressors — a different and less conservative null.

**Speed.** The design matrix is shared across neurons, so the pseudo-inverse of each
(fold, model) is formed once and reused for every neuron and every permutation. Each fit
becomes a matmul. This makes the cross-validated path *cheaper per fit* than the existing
in-sample loop, which re-decomposes the same `X` inside `lstsq` for every neuron and every
regressor group.

**Rank.** `check_rank` is reported for every design. The default
`parameterization='all_bins'` is rank-deficient by 8 (each one-hot block sums to 1 per row,
so the blocks are collinear through the implicit intercept) and `lstsq` silently returns the
minimum-norm solution — RSS, R² and CPD stay valid, individual betas do not. The refit uses
`'reference_coded'` (full rank). This matters because `_beta_direction` reads betas to set
the **sign** in `tuned_dict`.

**Poisson** (`cv_scores_poisson`) uses held-out deviance instead of RSS. It cannot share a
pseudo-inverse — IRLS is fit per neuron per model per fold — so it is orders of magnitude
slower and is offered without permutations, as a check that the linear conclusions survive
the link function.

**Coding is chosen per section, not globally.** `reference_coded` cannot encode a
single-column regressor — dropping the reference bin from `poke_rewarded` or
`poke_unrewarded` leaves zero columns, and `_resolve_regressor_groups` raises rather than
silently producing an empty block. So `w1_refit.choose_parameterization` tries reference
coding and falls back.

**The poke columns do not themselves cause any rank deficiency**, and it is worth being
precise about this because the two facts are independent. The pokes are why a section
*cannot use* reference coding; `all_bins` is *separately* rank-deficient because its
multi-column one-hot blocks each sum to 1 per row, so every block contains the all-ones
vector and k blocks give k−1 dependencies. Measured on a synthetic design of the same shape:

| design | cols | rank | deficiency |
|---|---|---|---|
| 8 one-hot blocks | 91 | 84 | 7 |
| the same + 2 poke columns | 93 | 86 | **7 (unchanged)** |

The pokes added 2 columns and 2 rank — they are full-rank contributors, being sparse binary
indicators that do not sum to 1.

**The deficiency is also data-dependent, not the fixed 8 the module docstring quotes.** Blocks
whose invalid rows are zeroed — `head_direction` (`HD_onehot[~np.isfinite(hd_all)] = 0`) and
`goal_progress_distance` — no longer sum to 1 on those rows, and so stop being collinear:

| design | deficiency |
|---|---|
| 8 blocks + HD, all rows valid | 8 |
| 8 blocks + HD, 8% of rows zeroed | **7** |

`glm_cv.check_rank` therefore measures and reports the rank of every fitted design rather
than assuming a number, and the measured values are logged per recday in the refit.

| section | regressors | columns | coding |
|---|---|---|---|
| `distance_gp_state_filtered` | 11 | 131 | reference_coded (full rank) |
| `all_regressors` | 16 | 173 | **all_bins** (has pokes) |
| `distance_gp_filtered` | 9 | 119 | reference_coded (full rank) |
| `pokes_filtered` | 11 | 129 | **all_bins** (has pokes) |
| `since_A_filtered` | 12 | 146 | reference_coded (full rank) |

In the two `all_bins` sections the design is rank-deficient by 8 and betas are minimum-norm.
RSS, R², CPD and **every cross-validated quantity** — which is what the regional comparisons
use — are unaffected. The cost is confined to `tuned_dict` **signs** in those two sections;
the tuned/not-tuned call still holds.

### W1.3 Synthetic controls (all passing)

Run through the real functions, per repo practice:

| control | result |
|---|---|
| A real regressor vs a pure-noise regressor | held-out CPD **+0.72** vs **−0.002** |
| **Pure-noise firing, held-out** | median CPD **−0.003**, 3–5% of neurons > 0 |
| **Pure-noise firing, in-sample (same data)** | median CPD **+0.002** — the inflation, visible directly |
| Fold integrity | no sample is ever in both train and test |
| `roll_within_sessions` | every session's values stay inside that session |
| Null calibration under H₀ | fraction p < 0.05 = **0.025 / 0.050** for the two groups |
| `check_rank` | correctly flags a deficient design |

The noise rows are the ones that matter: **the same data gives a positive in-sample CPD and a
negative held-out CPD.** That is the bias this workstream exists to remove.

---

## W3 — Generalising state cells and coherent remapping

Status: **complete.** Code: [w3_remapping.py](w3_remapping.py), [w3_figures.py](w3_figures.py),
[w3_synthetics.py](w3_synthetics.py). Notebook:
[LEC_anatomy_state_remapping.ipynb](LEC_anatomy_state_remapping.ipynb). Cached:
`data/processed_data/w3_remapping.pkl`, `w3_extras.pkl`. Figures:
`data/figures/anatomy_split/` (png + pdf).

Headline: **cells remap almost completely, they remap coherently, and the coherence is not
anatomical.**

### W3.0 What the estimator actually is, and what "coherent" means

**Coherence is a TWO-comparison criterion, so its chance level is 1/16.** A pair counts as
coherent only if its relative rotation survives X→Y *and* X→Z — El-Gaby's dual criterion.
One comparison landing within 45° is a one-in-four coin flip given the quantisation below,
far too easy to satisfy by accident to support a claim about rigid rotation, so the criterion
is squared. Reported over every unordered pair of a recday's comparisons (recdays have 3–6,
hence 3–15 comparison-pairs); the strict first-two-only parity number agrees to within 0.01.

#### The quantisation

The task-phase spectrum is dominated by harmonic 4 (`TASKPHASE_PERIODICITY.md`: power
fraction .098 against .057 at h=1, the largest harmonic in 84% of sessions). A curve
dominated by h=4 has a circular cross-correlation with **four near-equal peaks**, so the
argmax is a state-identity vote with a noise tie-break, not a continuous angle. Three
consequences run through everything below:

1. **Chance is 1/4**, which is what the El-Gaby port already assumed.
2. **The null must preserve h=4.** A uniform circle does not, and would be far too easy to
   beat. The null used throughout is the **cell-identity shuffle** — neuron *i*'s reference
   curve against neuron *j*'s comparison curve, within recday, same curves, correspondence
   broken.
3. It is measurable, and it was measured: mean distance to the nearest multiple of 90° is
   **9.0–14.0°** by region against **22.5° for uniform**, and the shuffle reproduces the
   same quantisation. The quantisation is a property of the curves, not of any cross-task
   relationship.

Two bands frame every angle panel. The **ceiling** is X-vs-X′ — the same physical task run
twice, present in every recday — so whatever it returns is measurement noise. The **floor**
is the cell-identity shuffle.

### W3.1 The gate (Stage A)

El-Gaby peak-z t-test at parity, included if state-tuned in ≥ half of a recday's tasks:
**1755 of 2851 unit-recordings**, 14.5–85.8% per recday.

**The spread is a mouse effect, not a region effect.** ENTl-deep alone runs 33.0% in ly05
and 84.4% in ly07. Within a mouse the ordering is consistent — ENTl-sup lowest everywhere —
which is what the rate confound predicts. Pass rate (%) by region × mouse:

| group | ah08 | ah10 | ly05 | ly06 | ly07 |
|---|---|---|---|---|---|
| ENTl-sup | 30.6 | 44.4 | 20.4 | 23.1 | — |
| ENTl-deep | 34.8 | 83.3 | 33.0 | 36.8 | 84.4 |
| ENTm | — | 63.6 | 58.8 | 40.8 | 52.5 |
| SUB/ProS | — | 85.8 | 45.7 | 33.6 | 66.0 |
| CA1/HPF | — | 78.8 | 38.2 | — | — |

**The primary contrast survives**: ENTl-deep vs SUB/ProS clears ≥10 gated units per region
in **13 recdays across 4 mice**, better than the plan's projection. Stage C exists.

### W3.2 Generalising state cells — a null result (Stage B)

**12 generalising cells in the entire cohort** (|angle| < 45° in *every* task comparison).
Per-comparison the cross-task rate is **0.257 against a chance of 0.25**, and it is a null at
both levels: **recdays (n=23) p = 0.34, mice (n=5) p = 0.31**, with only **14 of 23 recdays**
above their own shuffle (paired p = 0.20) — a coin flip.

This is not a measurement failure. The **X-vs-X′ ceiling is 0.814** — run the same task
twice and the estimator recovers the tuning 81% of the time.

Robustness:

| check | result |
|---|---|
| Rate matching, ENTl-deep vs SUB/ProS | 0.233 → 0.231 and 0.237 → 0.227. Nothing moves. |
| Both angle metrics | xcorr and peak agree on **99.0%** of cells |
| Reference task rotated over all 6 anchors | gap spans **−0.020 to +0.023** — straddles zero throughout |
| Every region | at chance; the null is uniform across the bank |

**Level 3 ("which letter?") is not answerable** and is reported as such: 12 cells over 5
regions and 4 states. The question is downstream of a generalising population that does not
exist.

This is the third independent measure to agree: the production GLM ranks `task_state` last
of 16 in LEC, `CCGP_STATE_PAIRS.md` reports CCGP 0.426 against a 0.500 null, and now the
rotation estimate sits on its own shuffle.

### W3.3 Coherent remapping — positive (Stage C)

On the dual criterion, pairs hold their relative task-space angle well above the shuffle
floor: **0.109 against a shuffle of 0.063**, chance 1/16 = 0.0625.

| level | n | mean | vs chance 1/16 | beats own shuffle |
|---|---|---|---|---|
| recday | 25 | 0.109 | **p = 1×10⁻⁹** | **24 / 25**, paired p = 1×10⁻⁷ |
| mouse | 5 | 0.109 | **p = 0.002** | 5 / 5 |

Per mouse: 0.087–0.131 against own floors of 0.059–0.070.

Three things make it trustworthy:

- **the shuffle recovers the analytic chance level** (0.063 against 0.0625), so the
  1/16 reasoning and the empirical null agree;
- the reference-anchored metric (`r_j − r_i`) and the reference-free direct metric
  (`Δ_ij(t) − Δ_ij(ref)`) agree to within 0.04 per mouse; and
- averaging over all comparison-pairs and El-Gaby's strict first-two parity agree to within
  0.01 (e.g. ah10 0.1314 vs 0.1330).

On the single-comparison rate (chance 1/4) the same result reads 0.286–0.350 against
0.246–0.254, and the gap is positive for **all 6 reference-task choices** (+0.065 to +0.097).

**The control that decides whether this is real.** A pair whose cells already peak at the
same task-space phase is close to trivially coherent — and two cells with the same *place
field* sit exactly there, then move together for reasons that have nothing to do with a
task-space rotation. Coherence by initial pairwise separation:

| initial separation | recdays | dual real | shuffle | gap | > own null | vs 1/16 (recday / mouse) |
|---|---|---|---|---|---|---|
| 0–45° | 24 | 0.130 | 0.062 | +0.068 | 24 / 24 | p=3×10⁻¹⁰ / p=0.001 |
| 45–90° | 23 | 0.107 | 0.063 | +0.044 | 21 / 23 | p=5×10⁻⁷ / p=0.006 |
| 90–135° | 24 | 0.096 | 0.061 | +0.035 | 22 / 24 | p=4×10⁻⁷ / p=0.006 |
| **135–180°** | **23** | **0.096** | **0.058** | **+0.038** | **22 / 23** | **p=1×10⁻⁷ / p=0.001** |

It declines but does not vanish, and stays clear of the 1/16 chance line at every separation.
The far bin — the one no shared place field can explain — is significant at **both** levels.
The coherence is not an artefact of co-tuned or co-place-tuned pairs.

### W3.4 The coherence is not anatomical

Same-region versus cross-region coherence, matched on **pair count** (`n_pairs ~ n²`, so a
region with twice the neurons brings four times the pairs) **and on initial tuning distance**
(W3.3 just showed coherence depends on it, and same-region neurons plausibly have more
similar tuning — unmatched, the comparison returns "same-region" for free and is measuring
tuning similarity, not anatomy):

- **within − cross = +0.010** on a base of ~0.12, against a within-recday label shuffle of
  0.000 ± 0.006. **p_perm = 0.088**, 19 recdays, 4 mice.
- At the **recday** level a t-test against zero gives p = 0.04 — but only **10 of 19 recdays
  are positive**, a coin flip, so that p is carried by a few large days rather than a
  consistent effect.
- At the **mouse** level p = 0.06 with 3 of 4 positive, and the negative one is **ah10**,
  which contributes by far the most data (−0.004, against ly05/ly06/ly07 at
  +0.017/+0.016/+0.013).

The stronger form agrees. Letting the coherence structure define its own modules
(agglomerative clustering on the incoherence matrix, *k* by maximum silhouette) and asking
whether those modules recover the region labels gives **ARI 0.000 against a null of 0.000,
with 1 of 25 recdays at p<0.05** where chance is 1.25. The clusters are real — silhouettes
0.16–0.56 — but they do not follow the anatomy.

**ENTl-deep and SUB/ProS rotate together, not as independent modules.**

### W3.4b The reliability floor

Within-session split-half rotation angle, per region — the measurement noise floor that
bounds how much of any regional difference in W3.2 could be noise:

| group | n | median &#124;Δ&#124; | within 45° |
|---|---|---|---|
| ENTl-deep | 3761 | 3° | 86% |
| SUB/ProS | 2216 | 3° | 87% |
| CA1/HPF | 850 | 3° | 85% |
| ENTm | 931 | 6° | 73% |
| ENTl-sup | 845 | 9° | 67% |

The floor is small against the 45° threshold everywhere, and worst in ENTl-sup — the
lowest-rate group, as the confound predicts. It is **not** overlaid on the figure-2 dials:
being that tight, it sets the radial limit and flattens the real distribution. Reported as a
number instead, which is the useful form.

### W3.5 Synthetic controls (all passing)

Through the real pipeline: synthetics enter as `Neuron_raw` and `Trial_times` and are binned,
warped, smoothed, gated and rotated by the production code, never handed a ready-made curve.

| # | control | result |
|---|---|---|
| 1 | known rotations recovered | 0/±30/±45/±90/±180 exact, both metrics |
| 2 | rigid vs independent population | DUAL 1.000 vs **0.068 against an analytic 1/16 = 0.0625** — this is what pins the chance level |
| 3 | ordinal-state vs place cell | generalising 1.00 vs 0.00, gate kept both |
| 4 | **the pair → region join** | within 1.00, cross 0.00, label-shuffled gap −0.05 |
| 5 | **the cell-identity shuffle** | 1.00 quantised to multiples of 90; per-comparison 0.25 = chance |
| 6 | one-pass tuning build | matches `rr.build_session_tuning` exactly |
| 7 | **ragged task counts** | both a 5- and a 6-comparison recday score 1.00 |

Control 7 caught a live bug in this workstream's own code. Recdays do not all have 6 usable
tasks (they range 4–7, giving 3–6 comparisons), so concatenating the per-recday angle
matrices pads the short recday's last column with NaN — and because both "generalising" and
"coherent" require *every* comparison to be within threshold, a NaN column silently condemned
every neuron and every pair of that recday. **A whole animal-day would have read as fully
remapping.** Both flags are now decided per recday, before the concat.

### W3.6 What this does not license

- **Nothing here is about ENTl-sup or ENTm as regions.** They are ah08 (611/682 units) and
  ly07 (179/284) wearing region labels.
- **The gate is rate-dependent**, and varies more between mice than between regions. The
  per-region *null* is robust to rate matching; a per-region *difference* would not have been.
- **W3.4 is a bounded negative, not proof of no effect.** With 4 mice and a null SD of 0.006,
  effects below about 0.012 are not resolvable.
- **The dual X-vs-X′ ceiling exists in only 9 recdays / 3 mice.** It needs two exact task
  repeats; 8 recdays have none. The data has **4–7 unique tasks per recday**, not the nominal
  "6 unique + 2 repeats" — worth knowing before any analysis budgets for repeats.
- **W3 §5 (the leg-duration / time-cell confound) was deliberately not run.** A fixed-latency
  time cell passes the gate then fails generalisation by construction. It cannot manufacture
  the W3.3 coherence result, but it is one unseparated contributor to the W3.2 null.

### W3.7 Why the earlier prototype found nothing at distal tuning distances

The prototype in `LEC_sploratory_analysis_with_glm_and_population.ipynb` (cells 341/347/350)
ran this analysis and reported the two distal bins as **non-significant** — 90–135° p = 0.992
and 135–180° p = 0.978, both with *negative* t against a flat 1/16. That result was chased
down and it is a **yardstick error, not a different finding**.

Three candidate causes were tested by reproducing the prototype's structure inside the W3
pipeline. Two were ruled out:

| candidate | effect |
|---|---|
| **no state-tuning gate** (`use_only_lec = False` → all neurons) | negligible; the pattern holds gated and ungated |
| **3-task triplets** (`day1 = sessions[:3]`, `day2 = sessions[3:6]`, n = 44 "days") | negligible; same pattern with one 6-task reference, n = 25 |
| **the metric** — direct `Δ_ij(Y) − Δ_ij(X)` vs ref-anchored `r_j − r_i` | **this is it** |

| metric | 0–45° | 45–90° | 90–135° | 135–180° |
|---|---|---|---|---|
| direct (prototype) | 0.227 *** | 0.086 *** | 0.061 **n.s.** | 0.071 **n.s.** |
| ref-anchored (W3) | 0.129 *** | 0.111 *** | 0.096 *** | 0.094 *** |

**The direct metric's chance level is not 1/16 when you bin by initial distance.** The metric
is `Δ_ij(Y) − Δ_ij(X)` and the bin is `|Δ_ij(X)|` — the pair's own angle is in both. A
**pair-identity shuffle** (keep `Δ_ij(X)`, compare against a *random other pair's* `Δ(Y)`,
`Δ(Z)`; destroys coherence, preserves each task's marginal angle distribution) measures the
true chance per bin:

| bin | direct: real | direct: **own chance** | ratio | ref-anchored | flat 1/16 |
|---|---|---|---|---|---|
| 0–45° | 0.227 | **0.087** | 2.62× | 0.131 | 0.0625 |
| 45–90° | 0.083 | **0.053** | 1.57× | 0.116 | 0.0625 |
| 90–135° | 0.061 | **0.047** | 1.29× | 0.097 | 0.0625 |
| 135–180° | 0.077 | **0.058** | 1.33× | 0.100 | 0.0625 |

So the prototype compared 0.061 against 0.0625 and read "below chance", when that bin's
actual chance is 0.047 — it was **1.29× above** chance. **Both metrics agree on the science
once each is measured against its own null.**

The mechanism is quantitative, not hand-waving. Pair angles are not uniform on the circle —
they pile up near 0 (measured marginal **0.299** in the 0–45 bin against 0.25 for uniform),
so a pair at 0° is more likely to meet another pair at 0° by accident. Predicting each bin's
chance as `1/16 × (marginal ⁄ 0.25)²` gives 0.089 / 0.057 / 0.054 / 0.053 against the
measured 0.087 / 0.053 / 0.047 / 0.058.

**W3 therefore uses the ref-anchored metric for anything conditioned on initial tuning
distance** (figures 6 and 6b), and keeps the direct metric as the unconditioned parity check
in figure 3, where no such conditioning occurs and the two agree (0.109 vs 0.115).

One further fact worth recording: the two metrics are algebraically identical only when every
per-neuron rotation is a clean roll, and in this data **that holds for just 2.3% of pairs**.
They are not interchangeable in general — only in aggregate.

Reusable diagnostic: `w3_figures.pair_identity_null_by_distance(results)`.

#### W3.7b The direct metric plotted honestly — `fig7_direct_vs_own_chance`

`data/figures/anatomy_split/fig7_direct_vs_own_chance.{png,pdf}` shows the direct metric
against its **own** bin-dependent chance, globally and per region. The dotted line is the flat
1/16 the prototype tested against; the dashed line is what chance actually is.

**Globally the direct metric is significant at every distance**, including the distal bins,
with **5/5 mice positive in all four bins**:

| bin | real | own chance | ratio | recdays > own null | recday p | mouse p |
|---|---|---|---|---|---|---|
| 0–45° | 0.208 | 0.097 | 2.15 | 24/24 | 6×10⁻⁸ | 0.031 |
| 45–90° | 0.081 | 0.056 | 1.48 | 21/23 | 2×10⁻⁶ | 0.031 |
| 90–135° | 0.063 | 0.049 | 1.28 | 22/24 | 7×10⁻⁶ | 0.031 |
| 135–180° | 0.067 | 0.048 | 1.45 | 20/23 | 3×10⁻⁶ | 0.031 |

Note the **own-chance line falls with distance** (0.097 → 0.048) while 1/16 is flat — that
gap is the entire discrepancy with the prototype.

**Per region the direct metric runs out of power in the distal bins.** Region subsets have
few pairs per recday (the ≥25-pair floor drops recdays: ENTl-deep keeps 14 recdays in the
near bin but only 10 in the far one), and the pair-identity null adds variance on top. So
ENTl-deep and SUB/ProS are significant at 45–135° but marginal or n.s. at 135–180° on this
metric, where the ref-anchored version (W3.3, figure 6b) is clearly significant.

**Use the ref-anchored metric for anything conditioned on initial distance.** It has a fixed,
valid 1/16 chance level and does not need a per-bin empirical null, so it is both correct and
better powered. Figure 7 exists to reconcile the two analyses, not to carry the claim.

### W3.8 CCGP by region — the population form of the W3.2 null

Figure: `data/figures/anatomy_split/fig8_ccgp_by_region.{png,pdf}`. A decoder trained on some
tasks and tested on a held-out one generalises only if state identity is abstract across
tasks, so this is an independent check on W3.2 that can see structure distributed across
cells and invisible one neuron at a time.

**Every region sits at or below its own role-permutation null, and the within-task ceiling
rules out "these cells carry no state information".**

| region | acc (matched n=20) | null | within-task ceiling | recdays below null | mice |
|---|---|---|---|---|---|
| ENTl-sup | 0.443 | 0.500 | 0.695 | 4/4 | 1 |
| ENTl-deep | 0.451 | 0.500 | 0.750 | **18/20**, p=2×10⁻⁴ | 5 |
| ENTm | 0.465 | 0.498 | 0.699 | 4/5 | 1 |
| SUB/ProS | 0.450 | 0.499 | 0.749 | **14/15**, p=6×10⁻⁴ | 3 |
| CA1/HPF | 0.445 | 0.501 | 0.763 | 7/9, p=0.049 | 2 |

Full n agrees throughout (ENTl-deep 0.444 vs 0.498, 20/24 recdays below null, p=2×10⁻⁵;
SUB/ProS 0.450 vs 0.501, 17/20, p=4×10⁻⁴). Full n is a **within-region existence claim only**
— full-n ENTl-deep at 38 units beating full-n ENTm at 24 would be expected from n alone — so
mean n is printed on every bar.

The ceilings (0.70–0.79) are the load-bearing part: the same populations decode state
perfectly well **within** a task and fail **across** tasks. That is the population-level
statement of W3.2, and it matches `CCGP_STATE_PAIRS.md` (LEC 0.426 against a 0.500 null)
computed independently of anatomy.

**No region is an exception.** There is no part of the recorded bank where state identity
generalises.
