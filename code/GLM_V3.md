# GLM V3 — the reduced core model: goal-progress vs absolute time, with confounds, by region

`glm_analysis_v3.py` · `LEC_glm_v3.ipynb` · `glm_v3_synthetics.py` · `check_mirror_parity.py` ·
`w1_refit.py` (sections `core_progress_time`, `core_progress_only`) · `run_glm_batch.py`
(`--leg-cap-s`, `--tfr-scheme`, `--tfr-bins`) · `glm_plots.py` (paired / factorial / β-profile /
region-significance figures)

Linear GLM of 250 ms-binned spike counts on a **reduced** regressor set — `place`, `goal_progress`
(the time-normalised version), `speed`, `acceleration`, `time_from_reward` — leave-one-session-out
cross-validated, Freedman–Lane permutation null, CPD and Δr² per neuron, LEC split by brain region and
PFC fitted with the identical design. V3 is seeded from `glm_analysis_v2.py`, which is left untouched
so every production fit (`all_regressors__*_250ms_decile`) stays reproducible; V3 under its defaults
reproduces V2 to `max|diff| = 0` (control 1), and `diff glm_analysis_v2.py glm_analysis_v3.py` is the
complete change list. The PFC mirror write-up is [`../mFC_data/code/GLM_V3.md`](../mFC_data/code/GLM_V3.md).

> Side check, kept entirely separate so it cannot mix into these results:
> [`GLM_V3_NORMALISATION.md`](GLM_V3_NORMALISATION.md) — the firing-normalisation ladder (no
> per-session normalisation / per-session offset = production / per-session offset and gain). An
> across-session z-score is proved and measured to be a **no-op** for every statistic below.

```bash
python code/glm_v3_synthetics.py                     # 11 controls through the real pipeline; run before trusting a result
python code/check_mirror_parity.py                   # code/ vs mFC_data/code/: must print OK (the submit scripts run it)
diff code/glm_analysis_v2.py code/glm_analysis_v3.py # must show only the hunks listed in section 3
```

## 1. Why V3 exists

The production GLM (`all_regressors__full_250ms_decile`, 16 regressors, 160 columns) carries **nine
within-leg regressors that are all functions of one latent — where the animal is in the current leg**:
`goal_progress`, `goal_progress_distance`, `time_from_reward`, `time_to_reward`,
`distance_from_reward`, `distance_to_reward`, `time_since_A`, `time_to_A`, `progress_since_A`.
Unique-variance measures attribute shared variance to *none* of them, so each looks tiny:
`goal_progress` scores −0.00015 (bias-corrected Δr²) in LEC and `task_state` is last in both
datasets. The question this version answers is whether that is **dilution by collinearity** or
**absence of signal**.

It is worth being exact about what the full model is and is not. It is **not overparameterised by
count**: the ah10 recday fits 29,614 rows against 160 columns (185 samples per parameter) with a
positive held-out R². It **is collinear**, and collinearity is what a unique-variance measure punishes.
Dropping regressors does not remove the shared variance; it **reassigns it to whichever survivor is
most correlated with the dropped block**. So a reduced model's CPDs are larger *by arithmetic*. The
informative outputs are (a) *how much* larger — the shared-variance budget the full model was splitting
nine ways, read off the arm that shares rows and coding with production — and (b) how that budget
splits between the two survivors, normalised progress and absolute time.

Why a 2×2 plus a sanity arm rather than one fit: each factor isolates one thing.

| comparison | isolates | which arms |
|---|---|---|
| full-16 → core-5 | the regressor set | production vs `core_progress_time` decile/60 (same rows, same tfr coding) |
| tfr decile → tfr uniform | the coding of absolute time | within each cap |
| cap 30 → cap 60 | the row set (long legs in or out) | within each coding |
| gp-only → gp+tfr | adding an absolute-time competitor to goal progress | `core_progress_only` vs `core_progress_time` at each cap |

## 2. What V3 changes in the module

Five hunks, nothing else. `check_mirror_parity.py` asserts the two V3 copies have identical code for
every shared definition (docstrings excepted) and that the two frozen V2 copies differ only in their
two pre-existing, allowlisted places (`KNOWN_V2_DRIFT`: a folded cast in
`compute_transition_filter_mask`, and `run_or_load_glm` never collecting `cv_results` in the mFC copy —
both fixed in V3).

1. **`GLM_VERSION = 'v3'`**, stamped into `CV_results[recday]['engine']`.
2. **`compute_decile_edges(..., fixed_range=None)`**. With `fixed_range=(lo, hi)` and
   `scheme='uniform'`, `n_bins` equal-width edges over exactly `[lo, hi]` in the variable's own units,
   ignoring the data; outer edges ∓inf as before. Raises on `fixed_range` with `scheme='decile'`.
   Without `fixed_range` the function is byte-identical to V2.
3. **`run_glm_analysis(..., binning_overrides=None)`**, shape
   `{'time_from_reward': {'scheme': 'uniform' | 'decile', 'range_s': (0.0, 30.0) | None, 'n_bins': 10}}`,
   applied by a local `_edges_for(name)` at the edge block. A `'uniform'` override converts `range_s`
   to raw-bin units (`× 1000 / bin_size_ms`, the raw bin being 25 ms) and calls (2); a `'decile'`
   override is the V2 quantile path with a custom bin count (control 1b: identical to V2). Only the eight
   continuous regressors can be overridden; overriding a regressor not in the design raises.
   **Out-of-range rows.** `time_from_reward` is a sawtooth, and a 250 ms window that straddles a reward
   averages the previous leg's tail (large) with this leg's start (~0). When the previous leg was longer
   than the cap that mean exceeds the range. These rows are leg-*start* rows, so they are NaN-ed: under
   reference coding a zero row *is* the first bin, which is where they belong, rather than the sparse
   top bin that `edges[-1] = +inf` would otherwise give them. They are counted
   (`bin_occupancy[...]['n_out_of_range']`) and reported in the log.
4. **`n_cols_override`** on `_resolve_regressor_groups` and `compute_tuning_arrays`, applied before the
   reference-coding decrement, so a non-default bin count keeps the column map consistent everywhere
   (control 10). `run_glm_analysis` derives it from `binning_overrides`.
5. **`CV_results[recday]['bin_occupancy'][name]`** for every overridden regressor: `scheme`, `edges`
   (raw units) and `edges_s` (seconds, time regressors), `rows` and `frac_rows` per bin,
   `legs_reaching` (kept legs longer than the bin's lower edge) and `n_legs`, `n_out_of_range`,
   `n_rows`. Printed as a table in the job log. Plus `CV_results[recday]['n_rows']`.

Not changed: `glm_cv.py` (the CV engine and the three nulls), `prepare_session_data`,
`downsample_session_data`, mixed reference coding, the in-sample permutation loop, `run_or_load_glm`,
`load_glm_results`.

## 3. The reduced designs

**Regressors** (fixed across the 2×2): `place` (21 nodes), `goal_progress` (10 equal-width phase
bins), `speed` (10 decile bins), `acceleration` (10 decile bins), `time_from_reward` (10 bins, coding
is the factor). Mixed reference coding: intercept + 20 + 9 + 9 + 9 + 9 = **57 columns**, full rank in
every recday. The gp-only arm drops `time_from_reward`: **48 columns**.

**Why each exclusion.** `task_state` (A/B/C/D leg identity) ranks last in both datasets and would
compete with `goal_progress` for the loop-position variance being consolidated. `goal_progress_distance`,
`distance_*`, `time_to_reward`, `time_since_A`, `time_to_A`, `progress_since_A` are the collinear
family whose dilution is the question. `head_direction` and the pokes are unavailable in PFC; dropping
them is what makes the design identical across datasets — at a cost recorded in caveat 3.

**Joint block.** `progress_or_time` drops `goal_progress` and `time_from_reward` together: the honest
"within-leg task structure, however coded" number to read beside the split.

**The 2×2.**

| | tfr decile | tfr uniform |
|---|---|---|
| cap 30 s | `core_progress_time__matched_250ms_decile_cap30s_tfrD10b` | `…_cap30s_tfrU10b` (3 s bins) |
| cap 60 s | `…_cap60s_tfrD10b` — production rows and coding | `…_cap60s_tfrU10b` (6 s bins) |

*Uniform*: `binning_overrides={'time_from_reward': {'scheme': 'uniform', 'range_s': (0, cap), 'n_bins': 10}}`
— the bin range is the leg cap, so no sample can exceed it except the straddling rows of section 2.3.
*Decile*: the V2 coding, per-recday 1–99 %-trimmed quantile edges.

**The gp-only arm.** `core_progress_only__matched_250ms_decile_cap{30,60}s`: `place`, `goal_progress`,
`speed`, `acceleration`; no joint block. Read with caveat 9.

**Section names and CLI.** `w1_refit.section_name(base, width_ms, scheme, regset, extra)` with
`extra = arm_extra(cfg, ...)` → `cap30s_tfrU10b` / `cap30s_tfrD10b` / `cap30s`. The leading `decile`
token is the global `--scheme` (speed, acceleration); the trailing token is the arm.
`run_glm_batch.py <recday> --section core_progress_time --regset matched --leg-cap-s 30
--tfr-scheme uniform --tfr-bins 10`, then `--merge` with the same flags. Both datasets use
`--regset matched` so the names coincide. `recday_registry._POST_REFIT_SECTION` accepts the trailing
token; without that the stale-cache list would silently drop `ly05_20250618_20250619` from every V3 fit
at load time (control 9).

**Config stamp** (per shard, checked at merge): the V2 fields plus, for V3 arms only, `engine`,
`leg_cap_s`, `tfr_scheme`, `tfr_bins`, `tfr_range_s` — V2 stamps are byte-identical to before, so a
later V2 shard still merges with the ones on disk.

## 4. Structural facts

### 4.1 Units and the tail

`time_from_reward` is produced in **raw 25 ms bin units** (`compute_task_state_arrays`) and
mean-aggregated to 250 ms; 30 s = 1200 units, and the fixed edges are built in those units. The tail
after the final reward of a session (tfr unbounded, `time_to_reward` = 0) is excluded whenever a leg cap
is set, because `compute_transition_filter_mask` starts from an all-False mask and marks only segments
between consecutive trial boundaries.

### 4.2 Leg caps

| cutoff (LEC) | legs dropped | samples dropped |
|---|---|---|
| 20 s | 16.2 % | 50.1 % |
| **30 s** | 8.4 % | 37.9 % |
| 40 s | 5.2 % | 30.7 % |
| **60 s** | 2.6 % | 22.4 % |

The 30 s arms therefore sit on **80.2 %** of the 60 s-cap rows in LEC (90.6 % in PFC). A cap is a
behavioural selection, not a neutral one: it removes the slowest legs.

### 4.3 Occupancy, measured from the trial times before any fit (all 25 recdays, uniform 3 s bins over 0–30 s)

| | LEC | PFC |
|---|---|---|
| leg duration median / p90 / p99 (s) | 9.2 / 25.6 / 98.4 | 7.5 / 17.2 / 49.8 |
| rows with tfr > 30 s under the 60 s cap | 5.1 % | 2.3 % |
| in-range rows in bin 0–3 s / bin 27–30 s (60 s cap) | 26.4 % / 1.7 % | 32.6 % / 0.8 % |
| in-range rows in bin 27–30 s (**30 s cap**) | **0.2 %** | 0.1 % |
| legs (≤ 60 s) reaching 9 s / 18 s / 27 s | 50 % / 16 % / 7 % | 38 % / 9 % / 3 % |

With the range tied to the cap, the top 3 s bin at cap 30 holds ~35 rows per LEC recday: its β is
noise that still costs a parameter. **At cap 60 the top 6 s bins can be entirely empty in a recday**
— the smoke test on `ah10_20250616_20250617` found the 48–54 s and 54–60 s bins with no rows, so the
design was rank-deficient by 2 (`CV_results[recday]['rank']`). RSS, R², CPD, Δr² and the
Freedman–Lane null are unaffected (the pseudo-inverse gives the minimum-norm solution and a dead column
contributes nothing), and the β of an empty bin is 0 by construction; but the uniform/60 arm's
`time_from_reward` block is, in such recdays, effectively an 8- or 9-bin block. The notebook's arm
table reports the rank per arm and the occupancy table flags every bin under 0.5 % of rows. Every fit
stores its own occupancy (section 2, item 5); the per-arm tables from the fits are in section 9.1.

### 4.4 Decile edges differ per recday

Under decile coding "bin 7" is a different number of seconds in every recday and in each dataset
(on the two synthetic-control recdays the interior edges were 1.3, 2.4, 3.5, 4.6, 5.9, 7.2, 8.8, 10.8,
13.7 s in LEC and 1.1, 2.0, 2.9, 3.8, 4.9, 6.0, 7.4, 9.1, 12.0 s in PFC). The notebooks tabulate the
spread of each edge across recdays; for cross-recday and cross-dataset comparison this is the
argument for the uniform coding, and it is why the uniform arm is the primary regional arm despite
its occupancy problem.

### 4.5 Boundary-straddling rows — present in every arm and in production

`time_from_reward` is a sawtooth, and the 250 ms aggregation takes the block **mean**. A window that
straddles a reward therefore averages the previous leg's tail (large) with this leg's start (~0), so
roughly one row per leg — ~1.5 % of rows — carries a tfr value that belongs to neither leg and lands
in a middle bin under either coding. This is inherited from V2 (the production fit has it too) and is
left as is, because changing the aggregation would break the same-coding comparison with production.
The V3 out-of-range guard (section 2, item 3) only fires when the previous leg was longer than about
twice the cap, so that the mean exceeds the range; on the two synthetic-control recdays at cap 30 this
happened to **zero** rows, and the fits report the count per recday. A principled fix — aggregating the
sawtooth variables with the block median — is a V4 item, with a legacy flag so the V3 numbers stay
reproducible.

## 5. How to read the figures

- **Aggregation chain**, identical to the production notebooks (`glm_plots.per_mouse_stat`): per
  neuron, RSS and TSS summed over held-out folds and the ratio formed once; per recday the median over
  neurons; per mouse the mean over its recdays; the bar is the mean over mice, the points are mice.
  Recdays of one mouse are the same probe re-sorted — repeated measures, never pooled as neurons.
- **Raw CPD is the user's primary statistic** (ΔRSS / RSS_reduced); Δr² (ΔRSS / TSS, a common
  denominator) and the bias-corrected variants (observed − Freedman–Lane null centre) are shown
  beside it. In the reduced designs every block has 9 columns except `place` (20), so the parameter
  penalty is more even than in the 16-regressor fit and the correction moves less; zero is still not
  the reference, the null centre is.
- **Significance** is the Freedman–Lane null on CPD (and on Δr², both stored): "g adds nothing beyond
  the *other regressors in this design*". A reduced design has a weaker H0 than the full one for the
  same neurons, so its fractions are higher partly by construction; compare, never read alone.
- **Regions**: every region on its own before any contrast; the primary contrast (ENTl-deep vs
  SUB/ProS, the only one replicated in 3 mice) with a mouse-resampling bootstrap, a within-recday label
  permutation, **rate matching** and a 50 µm boundary margin. ENTl-sup is 90 % ah08 and ENTm 63 %
  ly07: single-mouse claims wearing a region label.
- **Which figure isolates which factor**: `plot_paired_fits` (full-16 vs core-5; decile vs uniform;
  30 vs 60; gp-only vs gp+tfr), `plot_factorial_grid` (the 2×2 for one regressor),
  `plot_tfr_beta_profile` (where along time the tfr-tuned neurons load, over the occupancy),
  `plot_regressor_ranking_by_region(..., extra={'PFC': ...})` (LEC regions beside PFC).

## 6. Caveats — read before interpreting a single bar

1. **Bigger CPDs are arithmetic, not evidence** (section 1). Read the full-16 → core-5 shift on the
   decile/60 arm, which shares rows and coding with production, and the `r2_cv` panel beside it: the
   reduced model is a *worse* model with *bigger* CPDs, because `head_direction` (2nd after
   correction in LEC) and `poke_unrewarded` (4th) are gone.
2. **gp vs tfr in this model is the time-vs-progress dissociation.** Within a leg of duration D,
   `time_from_reward = goal_progress × D` exactly; the two are identified only through leg-duration
   variability (p90/p10 ≈ 6× in LEC). Their unique variances answer "at fixed phase, does firing track
   elapsed seconds?" (tfr) and "at fixed elapsed time, does firing track phase?" (gp). The
   out-of-distribution version of the same test is `time_vs_progress_dissociation.py` (fit on short
   legs, predict long legs); read the two together.
3. **With the pokes out of the model, reward consumption lands in tfr's first bin.** `poke_rewarded`
   is collinear with early `time_from_reward` by construction (the animal is in the port for the first
   seconds of every leg). A large tfr CPD is therefore partly consumption. The β profile
   (`plot_tfr_beta_profile`) shows whether the first bin dominates; synthetic control 5 shows what a
   pure consumption cell looks like there. PFC has no poke tables, so a `poke_rewarded` nuisance would
   break the matched design — this is the price of the identical design.
4. **The two codings fail in opposite ways, which is why both are run.** Uniform bins are populated
   by different legs — only legs longer than *t* reach the bin at *t* — so late bins are sparse and act
   as a **long-leg indicator**, and long legs are slow, disengaged, wrong-path and early-session (leg
   duration drifts within session, ρ ≈ −0.3). Decile bins are equally populated but their edges differ
   per recday, so a decile coefficient is not comparable across recdays or datasets. The occupancy
   tables and the edge-spread column are the evidence; a bin under 0.5 % of rows is flagged.
5. **The Freedman–Lane H0 depends on the design** (section 5). Never quote a reduced-design fraction
   without the full-design fraction for the same neurons beside it.
6. **Cap effects are partly row-set effects.** The 30 s arms use 80 % (LEC) / 91 % (PFC) of the 60 s
   rows, and the removed rows are the slowest legs.

   *A prior this work refuted.* The plan expected the reduced model to fit **worse** out of sample
   (it drops `head_direction` and `poke_unrewarded`, 2nd and 4th after correction). Measured on the
   24 same-row LEC recdays: mouse-chain median `r2_cv` **0.0421 (full-16) → 0.0430 (core-5,
   decile/60)**, and 60 % of neurons generalise *better* under the reduced design. Cross-validation
   charges the 16-regressor model a held-out parameter penalty for ~100 columns that mostly fit
   noise, and that penalty outweighs what the dropped regressors added. So "bigger CPD" here comes
   with "no worse a model", not with a worse one — which makes caveat 1 (the CPD growth is
   reassignment of shared variance) the only reading, not the charitable one.
7. **Region is confounded with firing rate** (SUB/ProS 5.86 Hz vs ENTl-deep 1.91 Hz; ly06 inverts
   it) and significance is rate-dependent. A regional gap in frac_sig is descriptive until it survives
   `anatomy_split.rate_match`.
8. **Cross-dataset comparison is licensed here** because the design is identical in both datasets —
   but PFC's legs are shorter (median 7.5 vs 9.2 s), so under uniform coding the same bin holds a
   different share of rows in the two datasets; and the PFC column is a whole dataset beside LEC
   sub-regions.
9. **In the gp-only arm `goal_progress` means "within-leg structure of any kind".** With no other
   within-leg regressor, gp absorbs elapsed time, distance, consumption and leg-type effects alike — a
   pure fixed-latency cell scores as a progress cell there (control 11). The gp-only arm answers "is
   there within-leg structure at all, beyond place/speed/acceleration?", which is the eyeballed signal;
   it does not say the structure is phase-locked. Read the three gp CPDs at a cap together: gp-only,
   gp+tfr(decile), gp+tfr(uniform).

## 7. Synthetic controls (`glm_v3_synthetics.py`)

Controls 3–8 and 11 replace `Neuron_raw` in one real recday (`ah10_20250616_20250617` in LEC;
`ab03_01092023_02092023` in PFC) with Poisson cells driven by that recday's real `Locs_raw` and
`Trial_times`, and go through `run_glm_analysis` with the V3 production flags at cap 30 — the repo
rule that a synthetic enters at the same door the data does. Cells: a **latency cell** (15 Hz for
6–9 s after every reward), a **phase cell** (15 Hz at goal progress 0.4–0.6 whatever the duration),
a **consumption cell** (15 Hz for 0–1.5 s after reward), a **place cell** (15 Hz at the most-visited
node), and 60 **noise cells** at a constant 2 Hz.

Run 2026-09-07, `n_perm = 100`, 60 noise cells, both trees. Numbers are LEC / PFC.

| # | control | outcome |
|---|---|---|
| 1 | v3 defaults == v2, production matched-13 design, cap 60 (every artefact, `max\|diff\|`) | PASS, **`max\|diff\| = 0.0` / `0.0`** over betas, in-sample F and permutation F, CPD, ΔR², and every cross-validated key (RSS, TSS, CPD, Δr², p-values, null centres and 95th percentiles). The first run reported 15.6 / 11.4: that was `cv_results['elapsed_s']`, the wall-clock time, which the comparer now ignores. |
| 1b | v3 decile arm == v2 with the same five regressors | PASS, **`0.0` / `0.0`**; the decile arm's occupancy is 9.7–10.8 % per bin. |
| 2 | fixed-range edges: centres → bins 0..9 at both caps; edges ignore the data; identical from both trees; decile + fixed_range raises | PASS. Edges at cap 30: 3, 6, …, 27 s; at cap 60: 6, 12, …, 54 s. |
| 3 | latency cell → tfr significant, tfr ≫ gp, peak in the 6–9 s bin / overlapping decile | PASS. Uniform: Δr²_tfr 0.586 / 0.582 against Δr²_gp −0.0002 / +0.0001, β peak in bin 2 (6–9 s). Decile: 0.527 / 0.570, peak in decile 6 (7.2–8.8 s / 6.0–7.4 s). p = 0.0099 (the floor at 100 permutations). |
| 4 | phase cell → gp significant, gp ≫ tfr, both codings | PASS. Δr²_gp 0.599 / 0.608 (uniform), 0.595 / 0.602 (decile); Δr²_tfr within ±0.0003 of zero. |
| 5 | consumption cell → tfr significant with the first bin dominant | PASS. Every β for bins 1–9 negative relative to bin 0 (uniform −0.54…−0.58 / −0.69…−0.85). Under decile coding β₁ is less negative than the rest (−2.6 / −1.4 vs ≈ −3.0): the 0–1.5 s field spills into the second decile (edges 1.3 s / 1.1 s). |
| 6 | place cell → place only | PASS. Δr²_place 0.612 / 0.525; gp and tfr within ±0.0006. |
| 7 | noise cells: fraction p < 0.05 per regressor, both codings | PASS. Pooled over the five regressors: 0.040 / 0.043 (uniform), 0.047 / 0.033 (decile); per regressor 0.017–0.100 (60 cells, so ±1 cell is 0.017); mean(observed − null centre) within ±5×10⁻⁵ for every regressor. The reduced design's Freedman–Lane null is calibrated. |
| 8 | fit occupancy == trial-times measurement; out-of-range rows < 1 % | PASS. max |Δ frac_rows| 0.0048 / 0.0049 (the 250 ms aggregation blurs the edges by that much); `legs_reaching` exact; 0 out-of-range rows in either recday at cap 30 (section 4.5). Uniform rows per bin at cap 30, LEC: 25.9, 25.1, 19.9, 13.4, 7.6, 4.3, 2.3, 1.1, 0.4, **0.04** %. |
| 9 | six V3 names + two production names pass `is_post_refit_section`; `_stale_or_excluded` drops nothing for them and drops ly05 for the pre-refit name | PASS (both trees). |
| 10 | `--tfr-bins 6`: 53 columns, full rank, tfr in the last 5 columns, `compute_tuning_arrays` accepts the override | PASS. Rank 53/53, tfr columns 48–52, tuning array (64, 5). |
| 11 | gp-only: phase cell gp-significant; latency cell gp-significant too and loses most of that gp variance once tfr is added | PASS. Latency cell: Δr²_gp **0.0315 / 0.0436 in gp-only**, −0.0002 / +0.0001 with tfr in the model; p = 0.0099 in both designs. The gp-only arm cannot tell a progress cell from a latency cell. |

Cost of one synthetic fit (64 neurons, ~27k rows, 57 columns, 100 Freedman–Lane permutations, 3–4
BLAS threads on a shared login node): 330–370 s. The Freedman–Lane loop scales with neurons, so a
150-neuron LEC recday is ~2–3× that before the in-sample permutation loop.

## 8. Runtime (2026-09-07, `cpu` partition, 8 cores and 64 GB (LEC) / 32 GB (PFC) per job, 100 Freedman–Lane permutations)

| | min / median / max per job | jobs |
|---|---|---|
| LEC (6 arms × 25 recdays) | 0.9 / **4.2** / 14.2 min | 150 |
| PFC (6 arms × 25 recdays) | 0.3 / **2.3** / 9.6 min | 150 |

300 jobs, **81 min wall clock** (08:42 → 10:03) at 15–20 concurrent slots, **0 failures**. The
longest LEC recdays (~150 neurons, 29k rows) spend ~10 min in the Freedman–Lane loop and ~1 min
loading the 3.8 GB pickle (the OS cache made the load 48 s rather than 4 min). Peak RSS 9.3 GB. The
gp-only arm is the cheapest (5 models × 48 columns). For comparison the production 16-regressor LEC fit
took ~40 min per recday.

## 9. Results — LEC (25 recdays, 2851 neurons, 5 mice; fits of 2026-09-07)

Every number below is the mouse chain (recday median over neurons → mouse mean → mean over mice) unless
it says "per neuron". Figures: `data/figures/glm_v3/`, produced by `LEC_glm_v3.ipynb`.

### 9.1 Occupancy from the fits (`time_from_reward`, mean % of design rows per bin over recdays)

| arm | bins 0 → 9 | recdays with an empty bin | rows beyond the range → bin 0 | decile-edge spread across recdays (max − min, s) |
|---|---|---|---|---|
| decile/30 | 10.6, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.9, 9.8, 10.8 | 0/25 | — | 0.8 … 7.7 (edge 1 → edge 9) |
| uniform/30 (3 s) | **28.3, 25.3, 17.6, 11.4, 7.2, 4.6, 2.9, 1.7, 0.8, 0.24** | 0/25 | 47 of 423,760 | 0 |
| decile/60 | 10.6, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 9.8, 10.8 | 0/25 | — | 1.2 … 18.4 |
| uniform/60 (6 s) | **45.6, 25.8, 12.2, 6.7, 4.1, 2.6, 1.6, 0.95, 0.47, 0.12** | **4/25** | 13 of 528,026 | 0 |

The decile edges move by up to 7.7 s (cap 30) and 18.4 s (cap 60) between recdays: under decile coding
the same bin is not the same seconds. Under uniform coding the top bin holds 0.24 % (cap 30) and 0.12 %
(cap 60) of rows and is empty in four recdays at cap 60. Rows per recday: 16,699 (cap 30) and 22,395
(cap 60), medians. Every arm is full rank in every recday except uniform/60 (21/25).

### 9.2 Model fit (`r2_cv`, mouse chain)

| full-16 (production) | decile/30 | uniform/30 | gp-only/30 | decile/60 | uniform/60 | gp-only/60 |
|---|---|---|---|---|---|---|
| 0.0413 | 0.0431 | 0.0417 | 0.0408 | 0.0426 | 0.0400 | 0.0404 |

On the 24 same-row recdays the full-16 fit gives 0.0421 against 0.0430 for decile/60; per neuron the
reduced design generalises better in 60 % of cells (caveat 6). The gp-only arms lose almost nothing
against gp+tfr (0.0408 vs 0.0417–0.0431 at cap 30), and uniform/60 is the worst of the six — its
absolute-time block is the poorest basis (9.5).

### 9.3 Ranking per arm — raw CPD (the user's primary), then bias-corrected CPD, then frac_sig (Freedman–Lane on CPD; the Δr² p-values agree to ±0.001 everywhere)

**Raw CPD**

| regressor | full-16 | decile/30 | uniform/30 | gp-only/30 | decile/60 | uniform/60 | gp-only/60 |
|---|---|---|---|---|---|---|---|
| place | 0.01341 | 0.01526 | 0.01528 | 0.01560 | 0.01521 | 0.01590 | 0.01594 |
| goal_progress | −0.00061 | −0.00043 | **+0.00047** | **+0.00171** | −0.00032 | **+0.00093** | **+0.00128** |
| speed | 0.00173 | 0.00182 | 0.00185 | 0.00206 | 0.00224 | 0.00250 | 0.00251 |
| acceleration | −0.00010 | −0.00004 | −0.00002 | +0.00003 | +0.00017 | +0.00023 | +0.00023 |
| time_from_reward | −0.00037 | **+0.00057** | −0.00039 | — | **+0.00070** | −0.00102 | — |
| progress_or_time (joint) | — | 0.00257 | 0.00159 | — | 0.00236 | 0.00058 | — |

**Bias-corrected CPD** (observed − Freedman–Lane null centre)

| regressor | full-16 | decile/30 | uniform/30 | gp-only/30 | decile/60 | uniform/60 | gp-only/60 |
|---|---|---|---|---|---|---|---|
| place | 0.01596 | 0.01857 | 0.01850 | 0.01887 | 0.01766 | 0.01832 | 0.01842 |
| goal_progress | 0.00031 | 0.00075 | 0.00171 | 0.00294 | 0.00062 | 0.00190 | 0.00223 |
| speed | 0.00244 | 0.00279 | 0.00282 | 0.00303 | 0.00294 | 0.00318 | 0.00323 |
| acceleration | 0.00051 | 0.00079 | 0.00082 | 0.00087 | 0.00077 | 0.00083 | 0.00084 |
| time_from_reward | 0.00073 | 0.00194 | 0.00123 | — | 0.00191 | 0.00038 | — |

**Fraction of neurons significant** (p < 0.05, Freedman–Lane, CPD)

| regressor | full-16 | decile/30 | uniform/30 | gp-only/30 | decile/60 | uniform/60 | gp-only/60 |
|---|---|---|---|---|---|---|---|
| place | 0.817 | 0.791 | 0.795 | 0.796 | 0.818 | 0.818 | 0.816 |
| goal_progress | 0.309 | 0.405 | 0.542 | **0.654** | 0.419 | 0.600 | **0.644** |
| speed | 0.671 | 0.657 | 0.662 | 0.667 | 0.702 | 0.728 | 0.727 |
| acceleration | 0.427 | 0.457 | 0.461 | 0.471 | 0.515 | 0.518 | 0.519 |
| time_from_reward | 0.437 | 0.575 | 0.455 | — | 0.584 | 0.280 | — |

Two things stand out before any interpretation. First, the *sign* of `goal_progress` and
`time_from_reward` swaps with the tfr coding (9.5). Second, the raw CPDs of both are within
±0.001 of zero in every arm while place sits at 0.015 and speed at 0.002: whatever within-leg
structure there is, its unique share is small once the two ways of coding it compete.

### 9.4 The regressor-set effect: full-16 → core-5 on the 24 same-row recdays (decile/60)

| | place | goal_progress | speed | acceleration | time_from_reward |
|---|---|---|---|---|---|
| raw CPD | 0.01372 → 0.01552 | −0.00060 → −0.00033 | 0.00169 → 0.00217 | −0.00012 → +0.00016 | −0.00037 → +0.00068 |
| corrected CPD | 0.01622 → 0.01797 (×1.1) | 0.00032 → 0.00061 (×1.9) | 0.00240 → 0.00288 (×1.2) | 0.00048 → 0.00076 (×1.6) | 0.00073 → 0.00188 (**×2.6**) |
| frac_sig (CPD) | 0.821 → 0.818 | 0.314 → 0.417 | 0.669 → 0.698 | 0.423 → 0.513 | 0.439 → 0.582 |
| per neuron, CPD reduced ≥ full | 60 % | 57 % | 60 % | 68 % | 68 % |

So consolidating nine within-leg regressors into two roughly **doubles** goal progress's and
**nearly triples** absolute time's corrected unique variance — and both remain an order of magnitude
below place (0.0006 and 0.0019 against 0.018). The shared-variance budget the full model was
splitting nine ways is real but small. `ah10_20250618_20250619` is excluded from this comparison:
its production fit (2 Sep) predates the salvage of its session 5 (6 Sep) and has 5 folds where the
V3 arms have 6.

### 9.5 Coding and cap effects

**The coding decides which survivor gets the shared variance.** Under decile tfr, absolute time wins
(tfr +0.0006/+0.0007, gp −0.0004/−0.0003 at cap 30/60); under uniform tfr, phase wins (gp
+0.0005/+0.0009, tfr −0.0004/−0.0010). The joint `progress_or_time` block explains less under uniform
coding (0.0016 vs 0.0026 at cap 30; 0.0006 vs 0.0024 at cap 60): a 10-bin equal-width basis with
60–70 % of its rows in the first two bins is a poorer basis for absolute time than ten quantile bins,
so the within-leg variance the tfr block cannot carry flows to `goal_progress`. Read together with
caveat 4: decile is the better *fit*, uniform the more *interpretable* axis, and neither is neutral
about the gp-vs-tfr split. The corrected values are less coding-dependent (tfr 0.0019/0.0012 at cap
30; gp 0.0008/0.0017) because the correction removes the parameter penalty that the sparse uniform
bins pay.

**The cap matters little for the decile arms** (gp −0.0004 → −0.0003, tfr +0.0006 → +0.0007) and more
for uniform (tfr −0.0004 → −0.0010, gp +0.0005 → +0.0009): at cap 60 the 6 s bins concentrate 71 % of
rows in bins 0–1 and empty the top bins in four recdays, so the uniform tfr block degrades further.

### 9.6 The gp-only sanity arm and the gp decomposition

| cap | gp CPD, gp-only | → gp+tfr decile | → gp+tfr uniform | frac_sig gp | per neuron gp-only ≥ gp+tfr uniform |
|---|---|---|---|---|---|
| 30 s | **+0.00171** (frac_sig 0.654) | −0.00043 (0.405) | +0.00047 (0.542) | | 67 % |
| 60 s | **+0.00128** (0.644) | −0.00032 (0.419) | +0.00093 (0.600) | | 62 % |

The eyeballed signal is there: with `goal_progress` as the only within-leg regressor, 65 % of LEC
neurons carry within-leg structure beyond place, speed and acceleration, and its unique CPD (0.0017)
is comparable to speed's. Adding one absolute-time regressor removes most or all of it — the gp CPD
falls to about zero (decile) or a quarter (uniform). By caveat 9 and control 11 this is exactly what
a population of *latency* cells would do too, so the gp-only number is "within-leg structure of any
kind"; the split between phase and seconds is made in the gp+tfr arms, and there it depends on how
seconds are coded (9.5).

### 9.7 LEC by region (raw CPD, mouse chain; `n` mice per region 1–5, see `docs/handoff/README.md` §2)

**Place** is largest in SUB/ProS (0.031) and CA1/HPF (0.025–0.028), then ENTl-deep (0.022), ENTm
(0.018–0.020), and smallest in ENTl-sup (0.010), in every arm. **Speed** follows the same order
(SUB/ProS 0.004–0.005, ENTl-deep 0.003, ENTl-sup ≈ 0). **Goal progress** in the gp-only arm:
ENTl-deep 0.0035, SUB/ProS 0.0026, ENTm 0.0013, CA1/HPF 0.0011, ENTl-sup 0.0008 (cap 30).
**Time from reward** (decile arms): ENTl-deep 0.0015–0.0019 and ENTm 0.0012–0.0014 against SUB/ProS
0.0004, CA1/HPF −0.0006 and ENTl-sup −0.0002–+0.0003; in the uniform arms every region is near or
below zero except ENTl-deep at cap 30 (+0.0004).

Significant fraction (Freedman–Lane on CPD), primary arm uniform/30: place 0.63 (ENTl-sup) to 0.85
(ENTl-deep); goal progress 0.47 (ENTl-sup) to 0.59 (SUB/ProS); time from reward 0.32 (ENTl-sup),
0.54 (ENTl-deep), 0.50 (ENTm), 0.43 (SUB/ProS), 0.39 (CA1/HPF). ENTl-sup is lowest on everything and
fires at 1.07 Hz — caveat 7.

**The primary contrast, ENTl-deep − SUB/ProS** (raw CPD, mouse-resampling bootstrap; 4 mice
contribute — ah10, ly05, ly06, ly07 — ly05 with few SUB/ProS units per recday):

| arm | regressor | diff [95 % CI] | p (within-recday permutation) | rate-matched diff [CI] | boundary-excluded diff [CI] | per-mouse diffs |
|---|---|---|---|---|---|---|
| decile/30 | time_from_reward | +0.0012 [0.0004, 0.0017] | 0.005 | +0.0013 [−0.0007, 0.0031] | +0.0013 [−0.0006, 0.0022] | ah10 +0.0015, ly05 0.0000, ly06 +0.0013, ly07 +0.0019 |
| uniform/30 | time_from_reward | +0.0009 [0.0007, 0.0012] | 0.003 | **+0.0014 [0.0006, 0.0021]** | **+0.0008 [0.0001, 0.0014]** | +0.0011, +0.0006, +0.0007, +0.0012 |
| decile/60 | time_from_reward | +0.0014 [0.0008, 0.0019] | 0.0005 | +0.0014 [−0.0006, 0.0034] | **+0.0017 [0.0006, 0.0023]** | +0.0019, +0.0006, +0.0011, +0.0020 |
| uniform/60 | time_from_reward | +0.0009 [0.0001, 0.0022] | 0.0005 | **+0.0005 [0.0002, 0.0010]** | +0.0008 [−0.0001, 0.0021] | +0.0002, +0.0029, +0.0007, 0.0000 |
| all four | goal_progress | −0.0001 to +0.0006, every CI spans 0 | 0.14–0.71 | spans 0 | spans 0 | mixed signs |
| gp-only/30, /60 | goal_progress | +0.0007 / +0.0002, CI spans 0 | 0.20 / 0.67 | spans 0 | spans 0 | mixed signs |

So: **ENTl-deep carries more unique time-from-reward variance than SUB/ProS in all four arms**, with
every mouse's difference ≥ 0 in every arm; the difference survives rate matching in the two uniform
arms (and keeps its sign, with a CI spanning zero, in the two decile arms), and survives the 50 µm
boundary margin in three of four. It is a small effect (+0.001 CPD on a place CPD of 0.02–0.03). By the
§5 rules this is a **candidate regional result, not yet a finding**: the two robustness checks each
fail in one or two arms, and 4 mice is the ceiling. There is **no regional difference in goal
progress** in any arm, including the gp-only arm where its CPD is largest.

### 9.8 PFC (identical designs, 25 recdays, 1252 neurons) and LEC beside PFC

Full detail in [`../mFC_data/code/GLM_V3.md`](../mFC_data/code/GLM_V3.md). The comparison is licensed:
both datasets carry the same five regressor groups, the same coding and the same caps, and the PFC
production fit shares rows with the PFC decile/60 arm in all 25 recdays.

**Raw CPD, mouse chain, LEC | PFC**

| regressor | decile/30 | uniform/30 | gp-only/30 | decile/60 | uniform/60 | gp-only/60 |
|---|---|---|---|---|---|---|
| place | 0.0153 \| 0.0074 | 0.0153 \| 0.0073 | 0.0156 \| 0.0072 | 0.0152 \| 0.0080 | 0.0159 \| 0.0078 | 0.0159 \| 0.0078 |
| goal_progress | −0.0004 \| **0.0020** | 0.0005 \| **0.0041** | 0.0017 \| **0.0087** | −0.0003 \| 0.0016 | 0.0009 \| **0.0065** | 0.0013 \| **0.0078** |
| speed | 0.0018 \| 0.0066 | 0.0019 \| 0.0067 | 0.0021 \| 0.0069 | 0.0022 \| 0.0065 | 0.0025 \| 0.0066 | 0.0025 \| 0.0068 |
| acceleration | 0.0000 \| 0.0022 | 0.0000 \| 0.0022 | 0.0000 \| 0.0023 | 0.0002 \| 0.0023 | 0.0002 \| 0.0024 | 0.0002 \| 0.0024 |
| time_from_reward | 0.0006 \| **0.0020** | −0.0004 \| 0.0006 | — | 0.0007 \| **0.0029** | −0.0010 \| −0.0006 | — |
| progress_or_time (joint) | 0.0026 \| **0.0116** | 0.0016 \| 0.0101 | — | 0.0024 \| **0.0111** | 0.0006 \| 0.0077 | — |

Three differences between the datasets, all robust across arms:

1. **PFC carries four to five times LEC's within-leg structure.** The joint block is 0.011–0.012 in PFC
   against 0.0024–0.0026 in LEC (decile arms), and in the gp-only arm `goal_progress` is PFC's single
   largest regressor (0.0087, frac_sig 0.83) — larger than place (0.0072) — where in LEC it is a tenth
   of place. In PFC *both* survivors stay positive under decile coding (gp 0.0020, tfr 0.0020 at cap 30),
   so the within-leg variance is not just a coding artefact there; in LEC the two fight over a budget
   close to zero.
2. **Place and speed are the other way round.** Place is 0.015 in LEC and 0.007 in PFC; speed 0.002 in
   LEC and 0.007 in PFC; acceleration ≈ 0 in LEC and 0.002 in PFC.
3. **The coding effect has the same direction in both**: uniform tfr coding moves the shared variance
   from `time_from_reward` to `goal_progress` (PFC decile/30 → uniform/30: gp 0.0020 → 0.0041, tfr
   0.0020 → 0.0006), and the joint block explains less under uniform coding. In PFC's uniform/60 arm the
   6 s bins leave the top bins empty in **16 of 25 recdays** (PFC legs are shorter: median 7.5 s), so
   that arm is not interpretable for tfr in PFC.

**Regressor-set effect in PFC** (full-13 → core-5 decile/60, 25 same-row recdays): `r2_cv` 0.0411 →
0.0508, with **80 %** of neurons generalising better under the reduced design — the parameter-penalty
argument of caveat 6 is even stronger here. Corrected CPD: goal_progress 0.0008 → 0.0024 (×3.2),
time_from_reward 0.0025 → 0.0040 (×1.6), speed ×1.1, acceleration ×1.2 — and **place 0.0114 → 0.0101
(×0.9)**, the one survivor whose unique variance falls when the within-leg family leaves, which says
some of what place explained in the full model was standing in for the dropped regressors.

**gp decomposition in PFC:** gp CPD 0.0087 (gp-only) → 0.0020 (with tfr decile) → 0.0041 (with tfr
uniform) at cap 30; 0.0078 → 0.0016 → 0.0065 at cap 60; frac_sig 0.83 → 0.65 → 0.75. Per neuron, gp-only
≥ gp+tfr uniform in 73 % / 63 %. Absolute time takes roughly three quarters of PFC's within-leg budget
when it is coded by deciles and a third when coded by seconds — again the coding, not the data, decides
the split, and the honest statement is the joint block.

**LEC regions beside PFC** (`plot_regressor_ranking_by_region(..., extra={'PFC': ...})`, notebook §9):
PFC's goal-progress CPD in the gp-only arm (0.0087) exceeds every LEC region's (ENTl-deep 0.0035 is the
highest); its place CPD (0.0072) sits below every LEC region except ENTl-sup (0.010 → PFC is below
ENTl-sup too — 0.0072 < 0.0098); its time-from-reward CPD under decile coding (0.0020) is above ENTl-deep's
(0.0015) and every other region's.

### 9.9 Where the tfr-tuned neurons load (`plot_tfr_beta_profile`; unit-norm reference-coded β, tfr-significant neurons)

| arm | LEC n | LEC peak in bin 0 / last bin | LEC bin-0 dominant | PFC n | PFC peak in bin 0 / last bin | PFC bin-0 dominant |
|---|---|---|---|---|---|---|
| decile/30 | 1615 | 11 % / 19 % | 11 % | 902 | 13 % / 22 % | 13 % |
| uniform/30 | 1299 | 21 % / 21 % | 21 % | 699 | 14 % / 22 % | 13 % |
| decile/60 | 1656 | 14 % / 21 % | 14 % | 914 | 13 % / 26 % | 13 % |
| uniform/60 | 823 | 22 % / 18 % | 19 % | 425 | 25 % / 10 % | 8 % |

Two populations bracket the profile. **11–21 % of tfr-significant neurons are first-bin dominant** (every
later β negative) — the consumption signature of caveat 3 and control 5, so a fifth of the "time" cells
under uniform coding are reward-consumption cells. **18–26 % peak in the last bin** — the sparsest bin,
reached by the fewest legs, i.e. the long-leg indicator of caveat 4 (under decile coding the last decile
spans everything beyond ~12–14 s). The middle bins (3–9 s, where a latency cell like control 3 would sit)
hold the smallest shares. Neither population is "elapsed time" in the sense the regressor name suggests.

### 9.10 The goal-progress β profile (`plot_beta_profile(..., 'goal_progress')`, notebook §8.1)

Same construction as 9.9 over the ten equal-width phase bins; bin 0 — the **first tenth of the leg, i.e.
the reward-consumption period** — is the reference, so every β is "relative to consumption". Occupancy is
equal by construction — the cap excludes whole legs, so every kept leg puts a tenth of its samples in each
bin — and measured so: 9.9–10.1 % per bin at the raw rate, 9.5–10.4 % per 250 ms design row (LEC ah10 and
PFC ab03, either cap). The small tilt (bin 0 high, bin 9 low) has two named causes: the block-mode
aggregation of the phase bin breaks ties toward the lower index, and in PFC the tracking filter removes
1.5–2.3 % of rows in the last three bins against 0.6–0.9 % early (tracking drops near the destination
port; in LEC it removes nothing at 250 ms). So no phase bin is sparse, and the lower panel shows where the
gp-significant neurons' |β| peaks fall instead.
Figures `lec_gp_beta_profile_*.pdf` (by region) and `pfc_gp_beta_profile_*.pdf`, all six arms.

**How to read the reference bin.** Under reference coding bin 0 has no design column: its level is
absorbed into the intercept, and each β_k is the difference in expected firing between bin k and bin 0
with every other regressor held fixed. So every neuron sits at **exactly 0 in bin 0 by construction** (its
s.e.m. band vanishes), a curve above zero means "fires more in that phase than in the first tenth of the
leg", and only the *shape* across bins 1–9 is data — the offset is not, and a different reference bin
would draw a different-looking curve from identical information. The normalisation is per neuron: each
neuron's β vector is divided by its own max|β|, and the curve is the mean of those unit profiles across
the region's neurons. Because the reference bin is the consumption period, every `*_meancentred.pdf`
variant (`plot_beta_profile(..., center='mean')`) subtracts each neuron's mean over the ten bins first,
so the baseline is the neuron's own phase-average and bin 0 is no longer privileged; the shapes are
identical. The **peak** below is the bin of highest firing per neuron, *bin 0 included* (the signed
argmax): a neuron whose every β is negative fires most during consumption and peaks in bin 0. (An earlier
version of this table used the argmax of |β|, which cannot select bin 0 and mis-filed such neurons at
their most negative bin; the late-peak shares it produced were too high.)

| arm, region | n | mean β, bins 0.1–0.3 | 0.4–0.7 | 0.8–0.9 | peak in bin 0 (consumption) / 0.1–0.3 / 0.4–0.7 / 0.8–0.9 (%) |
|---|---|---|---|---|---|
| gp-only/30, ENTl-sup | 372 | 0.23 | 0.19 | 0.10 | 21 / 38 / 27 / 13 |
| gp-only/30, ENTl-deep | 664 | 0.27 | 0.36 | 0.30 | 16 / 30 / **35** / 19 |
| gp-only/30, ENTm | 170 | 0.38 | 0.31 | 0.18 | 14 / 44 / 31 / 11 |
| gp-only/30, SUB/ProS | 381 | 0.34 | 0.35 | 0.26 | 17 / 39 / 33 / 11 |
| gp-only/30, CA1/HPF | 159 | 0.54 | 0.53 | 0.45 | **9** / 44 / 38 / 8 |
| decile/30 (tfr in model), ENTl-sup | 228 | 0.10 | 0.01 | **−0.14** | **23** / 36 / 22 / 20 |
| decile/30, ENTl-deep | 425 | 0.18 | 0.13 | −0.03 | 20 / 41 / 21 / 18 |
| decile/30, ENTm | 81 | 0.24 | 0.11 | −0.05 | 19 / 42 / 21 / 19 |
| decile/30, SUB/ProS | 259 | 0.37 | 0.30 | 0.12 | 13 / 49 / 24 / 15 |
| decile/30, CA1/HPF | 88 | 0.65 | 0.54 | 0.33 | **2** / 61 / 30 / 7 |
| gp-only/30, **PFC** | 1077 | 0.19 | **0.37** | 0.29 | 14 / 21 / **43** / 22 |
| decile/30, **PFC** | 836 | 0.12 | 0.16 | −0.01 | 18 / 28 / 31 / 24 |

Three readings, each with its caveat:

1. **In the gp-only arm every LEC region sits above the consumption bin for the whole leg**, CA1/HPF
   highest (a flat plateau at ~0.55), ENTl-deep and SUB/ProS at ~0.3–0.35, ENTl-sup lowest and falling
   towards the leg's end; 9–21 % of neurons nevertheless fire most *during* consumption. Because the
   reference is the consumption period, a positive plateau is as much "lower during consumption than
   during the leg" as "tuned to progress" — caveat 3 seen from the other side. PFC's profile is different
   in shape, not just level: a smooth **ramp to a mid-leg peak** (0.37 at 0.4–0.7, 43 % of neurons
   peaking there against 27–38 % in any LEC region) and a decline towards the next reward.
2. **With absolute time in the model, the early-leg component leaves goal progress** (it is what
   `time_from_reward` codes). What remains in the entorhinal regions is small and *falls* through the leg
   (ENTl-sup mean β −0.14 in the last two bins, negative at 0.9 for 60 % of its neurons), while SUB/ProS
   and CA1/HPF keep a positive, early-peaked profile (49 % and 61 % peak in bins 0.1–0.3, only 2 % of CA1
   neurons peak during consumption). The late-leg fall in ENTl is the approach to the *next* reward, which
   no regressor in the design codes directly (`time_to_reward` and `distance_to_reward` were dropped) — a
   candidate prospective component, and equally a candidate artefact of the tfr sawtooth's last bin (9.9).
3. **The uniform arm sits between the two**, as 9.5 predicts: with a poorer tfr basis, more of the early-leg
   variance stays with goal progress.

None of this identifies phase coding as such: the profile is what goal progress explains *given the other
regressors*, and its shape moves with the competitor set exactly as 9.5 says the CPDs do.

### 9.11 Where the goal-progress cells peak, and how wide they are — W5

[`GP_TUNING_WIDTH.md`](GP_TUNING_WIDTH.md) (2026-09-08) takes the gp-significant cells of the
uniform/30 arm and asks what the β profiles above look like per neuron rather than averaged: sorted
β heatmaps per arm, and the relation between where a cell peaks in the leg and how wide its tuning
is, read against the V that **time** coding produces once time is divided by a variable leg duration.

Headline: in both datasets ρ(|peak − 0.5|, width) is ≈ −0.52 against −0.48 for planted time cells and
**+0.01 for planted phase cells**, and ρ(peak, width) is positive (LEC +0.43, PFC +0.30) where a
retrospective-only planted population gives +0.45/+0.29 and a prospective-only one −0.41/−0.29. So
the within-leg structure this model's `goal_progress` block scores looks like a population anchored
to the **last reward**, not like phase tuning. It does not settle the question: stratifying by this
model's own gp-vs-tfr split fails to abolish the relation (§5 there), which is either a limit of that
split as a per-neuron label or evidence against the simple time-cell reading.

That work reads these fits and does not refit; it is standalone code (`gp_tuning_width.py`) and does
not modify `glm_plots.py`, `pfc_glm_plots.py` or the V3 notebooks.

*Housekeeping:* `mFC_data/data/**` is set read-only after the fact (every directory under it, including
ones created the same day, ends up `r-x`; it looks like a periodic protection of the published dataset).
Regenerating a PFC figure there needs `chmod u+w` on the directory first; the merged fits under
`mFC_data/glm_outputs/` are unaffected.

## 10. What this does not license

- No claim that the full model was "wrong". Its unique-variance numbers are correct for the question
  it asks; the reduced model asks a different one.
- No pooling across arms. Each arm is one model; the comparisons between them are the result.
- No inference about ENTl-sup or ENTm beyond one mouse each.
- Consumption is not separated from elapsed time in `time_from_reward` (caveat 3).
- Decile coefficients are not comparable across recdays or datasets (caveat 4).
- **gp-only CPD is "within-leg structure of any kind", not phase-locking** (caveat 9).
- A 30 s-cap result is a result about legs shorter than 30 s.
- The uniform/60 arm's `time_from_reward` block is not interpretable in PFC (empty top bins in 16 of
  25 recdays) and only marginally in LEC (4 of 25); use it for the cap comparison of the other four
  regressors, not for tfr.
- The ENTl-deep > SUB/ProS difference in `time_from_reward` (9.7) is a candidate, not a finding:
  +0.001 CPD, 4 mice, rate matching survived in two of four arms.
- The gp-vs-tfr *split* is a property of the coding as much as of the neurons (9.5); only the joint
  block and the gp-only arm are coding-independent statements about within-leg structure.
