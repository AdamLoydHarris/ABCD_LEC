# Anchoring regression V4 — handoff

Everything an incoming agent needs about this work. Written 2026-09-05.

> **Superseded 2026-09-06 by V5** — read [`ELASTICNET_V5.md`](ELASTICNET_V5.md) and §0 of
> [`ELGABY_FIGURE5_RECONCILIATION.md`](ELGABY_FIGURE5_RECONCILIATION.md) first. In brief: the
> deposited GitHub code was read line by line; the "leading suspect" in §6 below (phase
> segmentation) is **refuted** (the Fig 5g raster proves 3-bin phases; the deposited notebook is
> not the executed code); the "25% vs 67%" gap was a **semantics mismatch** (his any-fold rule vs
> our majority vote — 66% under his rule); the state-tuning test in §5 "matches exactly" is
> **wrong** (his code uses the preferred-phase MEAN with NaN propagation, not the peak, and is not
> leg-duration confounded); the preferred phase is his peak-of-thirds rule (30% disagreement);
> "with non-zero beta coefficients" means finite in every fold. V4 is frozen; V5
> (`elasticnet_regression_v5.py`, `*_v5.ipynb`, `elasticnet_v5_synthetics.py` with 57 controls
> incl. v5-legacy == v4) carries the corrections. PFC reproduction under his definitions:
> 447 / 246 / 79 vs the paper's 489 / 329 / 224; the 90° panel is the residual.

Companion docs, all in `code/` unless noted:
[`ELASTICNET_V4.md`](ELASTICNET_V4.md) (the method) ·
[`ELGABY_FIGURE5_RECONCILIATION.md`](ELGABY_FIGURE5_RECONCILIATION.md) (comparison with the
paper) · [`../mFC_data/code/ELASTICNET_V4.md`](../mFC_data/code/ELASTICNET_V4.md) (what differs
on the PFC dataset).

---

## 0. Read this first

**The one live scientific question.** Our fitted betas pile up at **lag 0**: 37.8% of 4,433
fitted (neuron, fold) pairs have their largest coefficient at lag 0 against a uniform 8.3%, with
44.3% inside the 30° excluded set {0,11} and **72.4% inside the 90° set**. So most PFC units in
our fits are best explained by the animal's *current position*, not by a lagged anchor, and the
non-zero-lag criterion rejects them — correctly. El-Gaby's pass rates (67% / 46%) require peak
lags spread away from 0 to a degree our fits do not show. **This is upstream of every criterion
choice** and is the thing to work on next. Leading suspect: his `Phases_raw2_*` phase
segmentation, which is not in this repo (see §6).

**Do not repeat these two dead ends** — both were tested and falsified (§8):
* the gap is *not* beta sparsity from α — lowering α makes betas 15× denser and the pass rate
  goes **down**;
* our distribution is *not* less right-shifted than the paper's — on matched quantities we
  exceed it.

**State**: nothing is committed. A PFC Poisson sweep is running in the user's notebook kernel
(8/25 recdays as of 11:16). Details in §9.

---

## 1. What the analysis is

Per neuron, leave-one-*task*-out across sessions:

1. **Regressors.** Every (location, goal-progress phase) pair is an "anchor". Anchor
   *(loc, ap)* at lag *k* is on when the animal visited `loc` during a phase-`ap` segment *k*
   phase-steps ago. 9 locations × 3 phases × 12 lags = **324 columns**, in raw 25 ms bins.
2. **Preferred-phase restriction.** Each neuron is fit only on bins of its preferred
   goal-progress phase — **both X and y** are subset.
3. **Fit.** ElasticNet (α=0.01, l1_ratio 0.5, `positive=True`) or Poisson (α=1), per neuron per
   fold, on the concatenated training sessions.
4. **Readout.** Apply betas to the held-out session, normalise actual and predicted onto
   90 bins × 4 states, reduce each state to the **mean of its 30 preferred-phase bins**,
   Pearson-correlate the two 4-vectors. That's the headline `r`, from **n = 4 points**.
5. **Selection.** State-tuned × non-zero-lag (largest betas at intermediate lags).

Source of truth for the method: `code/Figure5_Regression.ipynb` (El-Gaby's own notebook, cells
15 / 21 / 26 are the live ones) and the paper PDF at `docs/s41586-024-08145-x.pdf`.

### Three structural facts that shape everything

* **Phase is a strict 0→1→2 cycle.** 0 of 39,732 (LEC) and 0 of 59,904 (PFC) transitions
  deviate from +1 mod 3. Segment *durations* are irrelevant (they range 1–10,285 bins).
* **Therefore only 108 of the 324 columns are live in any fit**: the anchor phase at lag *k* is
  determined, `ap == (pref − k) % 3`. Verified: 402/402 per-fold beta vectors have support on
  that stripe, zero exceptions. Betas are stored/plotted as **location × lag**.
* **Therefore the prediction is exactly 0 at every non-preferred-phase bin** (measured non-zero
  fraction: 0.000), so 240 of the 360 normalised bins of any "predicted tuning curve" are zero
  by construction. Use `cv_tuning_correlations_pref`, not the full-360 one.

---

## 2. Repo layout and the rules

```
code/                                   LEC side
  elasticnet_regression_v4.py           THE module (2460 lines)
  elasticnet_v4_synthetics.py           30 synthetic controls -- the gate
  elasticnet_v4_compare.py              LEC vs PFC comparison figure
  elgaby_figure5.py                     his scoring semantics, re-scores stored betas
  elgaby_ladder.py                      one-factor-at-a-time ladder (largely superseded)
  LEC_elasticnet_regression_v4.ipynb    24 cells
  elasticnet_regression_v3.py           UNTOUCHED, kept for the equivalence control

mFC_data/code/                          PFC side (El-Gaby's dataset)
  elasticnet_regression_v4.py           BYTE-IDENTICAL copy
  elasticnet_v4_synthetics.py           BYTE-IDENTICAL copy
  PFC_elasticnet_regression_v4.ipynb    26 cells
  glm_analysis_v2.py                    has build_data_dic_from_pfc (the PFC loader)
```

**The mirror rule.** `elasticnet_regression_v4.py` and `elasticnet_v4_synthetics.py` are kept
**byte-identical** between the two trees, as `elasticnet_regression_v3.py`,
`ccgp_state_pairs.py`, `taskphase_periodicity.py` and `persistent_homology_analysis.py` already
are. Edit one, copy to the other, then:

```bash
diff code/elasticnet_regression_v4.py mFC_data/code/elasticnet_regression_v4.py   # must be empty
diff code/elasticnet_v4_synthetics.py mFC_data/code/elasticnet_v4_synthetics.py   # must be empty
```

Anything dataset-specific goes in the notebook or the loader, never the module. The one
asymmetry the module carries: **anatomy is optional** — PFC has no `unit_regions` and no
`anatomy_split`, so `build_unit_table` falls back to identity columns and groups by `mouse`.
`table.attrs['has_anatomy']` records which path ran.

**The gate.** `python elasticnet_v4_synthetics.py` → **30/30**. Run it in both trees after any
module change. This is repo practice and it has earned its place (see the user's memory note
`synthetic-controls-catch-design-errors`).

---

## 3. The module

### Config (`RegressionConfigV4`), reference-matched defaults

```python
use_poisson=False, regularize=True      # ElasticNet a=0.01, l1_ratio 0.5, positive=True
lag_direction='past'                    # 'future' = the prospective mirror
num_lags=12
restrict_to_pref_phase=True
pref_phase_source='test'                # matches his cell 21 -- LEAKAGE, warned at run time
state_tuning_min_fraction=1/3           # ">1/3 of the recorded tasks" (paper methods)
state_tuning_p_threshold=0.05           # 0.01 reproduces ED Fig 8b
state_tuning_statistic='max'            # 'mean' is duration-invariant (see the confound, S5)
nonzero_lag_zero_lags=(0, 11)           # "30 degrees or more"
nonzero_lag_min=1, nonzero_lag_max=10
nonzero_lag_zero_lags_strict=(0,1,2,9,10,11)      # "90 degrees (one state) or more"
nonzero_lag_min_strict=3, nonzero_lag_max_strict=8
poisson_link='linear'                   # his readout; 'log' = exp(Xb+b), the paper's LNP
alpha_mode='fixed'                      # 'relative' = alpha_frac * per-neuron alpha_max
state_reduce='mean'                     # both mean and max always stored
require_positive_top3=True, nz_per_fold=True, drop_untracked_bins=True
```

**`pref_phase_source='test'` is leakage and is the default.** The held-out session's phase
tuning chooses which bins the held-out score averages over. It is default only for comparability
with the published numbers. `'train'` is one flag away and gives an unbiased score. Every run
prints a warning and `run_config.json` records it.

### What every run produces

Per recday per direction, in `{estimator}_v4_{direction}_{stamp}/`:

* `{recday}_{direction}_arrays.npz` — ~50 keys: `cv_coeffs`, `cv_intercepts`,
  `cv_actual_tuning`, `cv_predicted_tuning`, `cv_predicted_nz_tuning`,
  `cv_predicted_nz_strict_tuning`, the n=4 `*_state4` vectors (mean and max),
  `corrs` / `corrs_nonzero` / `corrs_nonzero_strict` (+ `mean_*`), `pref_phases`,
  `state_tuned_fraction`, `state_tuned_per_session`, `n_nonzero_betas`, the masks, and the
  config.
* **six PDFs**: `_all` and `_nonzerolag` × {summary, `_foldbetas`, `_foldratemaps`}.
* `cross_mouse_v4_{direction}_summary.svg` — the paper's **three panels** side by side with
  n / mean / t / P / effect size and the published values inline.
* `recday_diagnostics_{direction}.csv`, `run_config.json`.

**All three lag levels are emitted every run** (none / 30° / 90°), mirroring his three saved
arrays, because the reported number is very sensitive to which is chosen. **A Poisson run emits
both links from one fit** (`corrs_altlink` etc.) — verified equal to a dedicated run at the
other setting to `max|diff| = 0`. `state_tuned_fraction` is exported so any tuning threshold can
be re-applied post hoc without a re-fit.

### Functions worth knowing

| | |
|---|---|
| `run_cross_validated_regression_v4` | one recday |
| `run_and_summarise_all_mice_v4(..., n_jobs=)` | all recdays; joblib **threading** (sklearn releases the GIL, and `data_dic` stays shared rather than pickled) |
| `three_panel_summary(table)` | the paper's three panels vs the published values |
| `regenerate_summary(run_dir)` | redraw the summary figure from exports — **a figure change never needs a re-fit** |
| `build_unit_table` / `region_summary` / `compare_directions` | per-unit table, anatomy optional |
| `plot_fold_ratemap_pages` | actual vs predicted **per fold** |
| `assert_beta_stripe` | the phase/lag invariant, asserted on every export |
| `apply_link` | identity / `exp(Xb+b)` |
| `elgaby_figure5.score_elgaby` | his per-fold-NaN semantics, re-scored from stored betas |
| `elgaby_figure5.verify_against_v4` | proves re-derived regressors match the fitted ones |

---

## 4. Everything measured (LEC and PFC)

### Dataset facts

| | LEC | PFC (mFC) |
|---|---|---|
| units / recdays | 2851 / 25 | 1252 / 25 |
| trials per session (median) | 18 | 27 |
| mean firing rate | 2.94 Hz | 6.25 Hz |
| `Neuron_raw` | integer spike counts / 25 ms bin | same (values 0–7, mean 0.096) |
| untracked locations | code **0** (`locs_to_int` maps NaN→0) — 4.1% | real NaN, exact-0 fraction 0 |
| anatomy | `unit_regions.pkl`, 6 groups | **none** |

**`mFC_data/` is El-Gaby's own published dataset** — `combined_ABCDonly_days.npy` is the file
his notebook loads, recdays `me08/me10/me11/ah03/ah04/ah07/ab03`. Any PFC discrepancy is a
failure to reproduce, not biology.

### The α problem

| | LEC | PFC |
|---|---|---|
| all-zero fits at α=0.01 | 92/151 (61%) | 67/117 (57%) |
| median per-neuron `alpha_max` | 0.00708 | 0.00871 |
| units with **zero** informative folds | — | **410/1252 (33%)** |

α=0.01 sits above the median neuron's entire regularization path. The cut is a firing-rate cut
(0.70 Hz for zeroed units vs 6.43 Hz for survivors). `alpha_mode='relative'` puts every neuron
at the same point on its own path. Fitting on rate rather than counts needs no flag — it is
exactly `elasticnet_alpha=0.00025`.

### The state-tuning leg-duration confound

`raw_to_norm` warps each state interval onto 90 bins by *averaging*, so a longer leg has lower
variance and therefore a lower **max** — the statistic the tuning test z-scores. Constant-rate
Poisson cells acquire a "preferred state": the shortest leg. 300 noise cells per row, α=0.05:

| longest:shortest leg | FPR (`max`) | prefer shortest | FPR (`mean`) |
|---|---|---|---|
| 1.0× | 0.053 | 26% (chance) | 0.083 |
| 1.5× | 0.407 | — | 0.140 |
| 2.0× | **0.95** | **94%** | 0.093 |
| 3.0× | **1.00** | **99%** | 0.100 |

Real leg ratios: LEC median **2.26×** (p90 5.4×, max 11.5×), PFC median **1.82×** (p90 2.91×,
max 25.15×). In real data **47.4% of units prefer the shortest leg** (chance 25%), correlating
with the session's duration ratio at r = 0.43.

`state_tuning_statistic='mean'` is duration-invariant. The `>1/3 of tasks` rule mitigates but
does not remove this — it drops the pure-noise state-tuned rate from ~60% to 43%.

### The n=4 correlation is intrinsically noisy

Null sampling SD of a Pearson r from 4 points is `1/√3 = 0.577`; |r| exceeds 0.5 in 50% of
pure-noise draws and 0.9 in 10%. Measured fold-to-fold spread within a *selected* neuron:
SD 0.55 (LEC) / 0.56 (PFC), range median 1.19 / 1.25, and **57% / 96% of selected neurons have
folds that straddle zero**. Averaging unbiased noisy estimates is fine; a single fold's r is not
interpretable.

### Non-zero-lag criterion on pure noise (new defaults)

30° fires on **30–33%**, strict 90° on **0.8%**, full selection 8–17%, mean r unbiased (+0.04).
The criterion is a **shape descriptor, not a statistical test** — read the selection rate against
that baseline, not against zero.

### Past vs future

Past lag *k* and future lag 12−*k* point at the same task position one loop apart; measured
column correlation is only **0.02–0.34**, so the two directions are separable. Lag 0 is *not*
shared (causal vs anti-causal accumulation within a segment; agreement 0.62), which is why they
are fit separately.

### Timings

ElasticNet **0.29 s/fit**, Poisson **1.96 s/fit**, `raw_to_norm` 15 ms. PFC 25 recdays
ElasticNet: **2.9 h** wall (from `run_config.json`). Poisson ≈ 7× that. The box has 8 cores and
231 GB and is often heavily loaded by other users (load average 40–70) — budget generously.

---

## 5. The El-Gaby reconciliation

### Settled

| question | answer |
|---|---|
| lag set | **{0,11}** — paper says "lag from anchor of **30° or more**"; 30° is one goal-progress bin |
| strict variant | **{0,1,2,9,10,11}** — "90° (one state) or more" |
| 24-lag `_beyond`? | **No** — methods say "9 × 3 × 12 … 12 lags", "trained on five (training) tasks". `limited=False` in his cells 32/38 is leftover state |
| state-tuning *test* | **ours already matched his exactly** — peak per state per trial → z-score each row → preferred state → two-sided t-test vs 0, P<0.05 |
| state-tuning *subsetting* | ">1/3 of the recorded tasks", not our OR. 1191 → **774**; 824 → **590** with betas (his 489) |
| is our distribution less right-shifted? | **No** — see below |

### The published targets (from the captions)

| variant | state tuning | none | {0,11} 30° | 90° |
|---|---|---|---|---|
| ElasticNet (Fig 5h, ED 8a) | p<0.05 | n=489 t=9.3 | n=329 t=3.9 | n=224 t=2.53 |
| ElasticNet strict tuning (ED 8b) | p<0.01 | n=349 t=8.70 | n=227 t=2.83 | n=154 t=1.94 |
| Poisson (ED 8d) | p<0.05 | n=489 t=10.7 | n=346 t=4.74 | n=229 t=2.81 |

### Ours, PFC past, reference-matched defaults, 25 recdays

```
                      panel   n  mean_r  frac_pos      t         P  effect size
            all state-tuned 589  0.1732     0.694  12.79  3.2e-33       0.527
         non-zero-lag 30deg 190  0.0792     0.611   3.47  6.5e-04       0.252
non-zero-lag 90deg (strict)  17  0.0695     0.588   0.80     0.44       0.193
                      paper 489 / 329 / 224                    0.421 / 0.215 / 0.169
```

**Effect size `t/√n` is the comparable quantity** once n differs. We **exceed the paper on all
three panels**. The left panel is a genuine reproduction on a larger pool. The lag-filtered
panels are not — see §0 and §6.

### Paper vs his code — discrepancies

| | paper | his code |
|---|---|---|
| Poisson link | "linear–nonlinear–Poisson … **logarithmic** link" | log link **at fit**; prediction is `np.sum(X*β)` — **no `exp`, no intercept** |
| Poisson `max_iter` | unstated | sklearn default **100** (ours 1000) |
| ElasticNet sign | **not mentioned** | **`positive=True`** |
| n regressors | "9 × 3 × 12 (**312**)" | 324 (the paper's arithmetic is wrong) |
| preferred phase | "in the preferred goal-progress bin" — **train or test never stated** | the **held-out** session |

Zero occurrences of `np.exp`, `.predict(` or `.intercept_` anywhere in his notebook. So the
Poisson variant's stated purpose — robustness to the linearity assumption — is **discarded at
readout**. For ElasticNet the missing intercept is harmless (Pearson is shift-invariant); for
Poisson the missing `exp` is not. And `PoissonRegressor` has no `positive` parameter, so his two
"robustness" variants differ in the sign constraint as well as the link.

### Semantics: his per-fold NaN vs our per-neuron mask

He has no neuron mask — he NaNs *that fold's* correlation when a top-3 beta lands in the
excluded set and `nanmean`s the survivors, so a neuron counts on **≥1 surviving fold** where our
majority vote needs >50%. Measured on the same betas: 287 → 510, and the effect size *falls*
(0.274 → 0.133), because admitting a neuron on one fold admits noise.

### Session selection — matches exactly

Dedup by exact task equality **is** his `non_repeat_ses_maker`. Plus his single hand-exclusion,
in `EL_GABY_EXCLUDED_SESSIONS`: `me11_05122021_06122021` session 3 (task `[7,4,3,8]` shares 3 of
4 goals with session 0's `[7,4,3,5]`; his comment: "almost identical to session 0 (mistake)").
Exact dedup cannot catch it. With it applied me11 has **6 folds**, matching his
`num_non_repeat_ses_found = 6`.

---

## 6. The current open question — lag-0 concentration

Where the largest beta sits, over 4,433 fitted (neuron, fold) pairs, PFC past:

```
lag  0  37.8%  <- excluded (30deg)      uniform would be 8.3%
lag  1  11.3%  <- excluded (90deg)
lag  2   8.1%  <- excluded (90deg)
lag  3   4.9%
lag  4   4.8%     ... flat ~4.5% floor across 3-8 ...
lag  8   4.5%
lag  9   3.9%  <- excluded (90deg)
lag 10   4.8%  <- excluded (90deg)
lag 11   6.5%  <- excluded (30deg)

in {0,11}:            44.3%  (uniform 16.7%)
in {0,1,2,9,10,11}:   72.4%  (uniform 50.0%)
```

Pass rates follow directly: 190/774 = **25%** for 30° (his 67%) and 30/774 = **3.9%** for 90°
(his 46%). Our rates are below even random placement, because the peaks are concentrated in
exactly the excluded lags.

**Interpretation**: most PFC units in our fits are best explained by the animal's current
position. The criterion is rejecting them correctly. His pass rates require a peak-lag
distribution ours does not show.

**Leading suspect**: the phase definition. He loads `Phases_raw2_*`, `States_raw_*` and
`Times_from_reward_*` from precomputed files; we re-derive phase as linear thirds of each state
interval (`compute_phase_state_raw`). If his segmentation differs, the lag axis means something
different and the whole peak-lag distribution would shift. **Not testable here** — those files
are not in the repo.

**Secondary suspect**: the α/scaling puzzle. His n=489 is *identical* under ElasticNet and
Poisson; Poisson never zeroes a coefficient, so his "(with non-zero beta coefficients)" filter
removed nobody, while at α=0.01 on integer counts ours zeroes 33% outright. Either the Fig 5h
caption reuses the Poisson n, or his `y` was scaled somewhere not visible in the notebook.

Both need the OSF intermediate objects: **https://doi.org/10.17605/OSF.IO/3D9R2**

**Suggested next step**: add the peak-lag histogram as a standing per-run diagnostic, and if the
OSF objects can be obtained, diff `Phases_raw2_*` against `compute_phase_state_raw` on one
session. That single check would settle §6.

---

## 7. Bugs found and fixed — do not reintroduce

| bug | where | fix |
|---|---|---|
| untracked bins kept as training rows | `locs_to_int` maps SLEAP NaN → **0**, and only codes >9 were NaN'd; 4.1% of LEC bins | `drop_untracked_bins` NaNs codes <1 (no-op on PFC) |
| top-3 lags decided by argsort tie-breaking | `np.argsort(c)[-3:]` on a mostly-zero solution: one positive beta at lag 5 gives `[7,6,5]` (pass) under quicksort, `[10,11,5]` (fail) under mergesort — 56% coin flip | `require_positive_top3` |
| fold-averaging mixed coordinate frames | pref phase is refit per fold and **36% of neurons change it**; `nanmean(cv_coeffs)` superimposed two stripes in 23% of matrices | `nz_per_fold`, and beta panels average only folds sharing the modal pref phase |
| `corrs_nonzero` zeroed only lag 0 while the mask excluded {0,1,11} | v3 inconsistency | one lag set for both jobs |
| all-zero folds voted "no" | made the consistency requirement a firing-rate filter | they **abstain**; `n_informative_folds` records how many carried evidence |
| `np.concatenate([])` crash | `pooled` non-empty but every entry empty (all recdays ran, none selected) | guarded; reports and returns |
| duplicate-kwarg `TypeError` | `f(..., use_poisson=False, **overrides)` in `elgaby_ladder`, and `make_config(..., alpha_mode='fixed', **kw)` in **both notebooks** — the rate-matched cell would have crashed | merge into one dict before the call |
| `_config_from` assumed a dict | `results['config']` is a dict from npz but the live object in memory | accepts both |
| figure showed one panel | the cross-mouse summary plotted only the non-zero-lag histogram | `plot_cross_mouse_summary` draws all three |
| `verify_against_v4` tolerance too tight | 1e-6 vs float32 storage error of ~1e-6 | 1e-5, with the reason recorded |

---

## 8. Things I got wrong — corrected, do not repeat

1. **"Our distribution is markedly less right-shifted."** False. The comparison was our
   `{0,1,11}` panel against his `{0,11}` panel. On matched quantities we **exceed** him
   (effect size 0.527 vs 0.421).
2. **"Beta sparsity from α explains the low pass rates."** Falsified by direct test: lowering α
   makes betas 15× denser (median 1.6 → 23.6) and the pass rate goes **down** (27.8% → 20.4%
   for 30°; 2.5% → 0.0% for 90°). The real cause is lag-0 concentration (§6).
3. **"The published figure is the 24-lag `_beyond` run."** Plausible from the notebook state
   (cells 32/38 set `limited=False`) but the paper says 12 lags explicitly. Retired.
4. **"`pref_phase_source` drove the 287 → 190 drop."** No — the NZ mask barely moved (292 →
   287). It was the `>1/3` state-tuning rule removing 90 neurons.
5. All four corrections are now written into
   [`ELGABY_FIGURE5_RECONCILIATION.md`](ELGABY_FIGURE5_RECONCILIATION.md) — §3.3 carries the
   α-sweep that falsifies the sparsity story and the peak-lag distribution that replaces it, so
   that document and this one agree.

---

## 9. Current state

**Nothing is committed.** Modified: both `elasticnet_regression_v4.py` copies, both synthetics
copies, both notebooks, both `ELASTICNET_V4.md`. Untracked: `ELGABY_FIGURE5_RECONCILIATION.md`,
`elgaby_figure5.py`, `elgaby_ladder.py`, and this file.

**Completed runs** (all under the reference-matched defaults, summaries regenerated):

```
data/figures/elasticnet_v4_past_20260904_222853        LEC past    25 recdays
data/figures/elasticnet_v4_future_20260904_222853      LEC future  25
mFC_data/data/figures/elasticnet_v4_past_20260904_222909   PFC past   25
mFC_data/data/figures/elasticnet_v4_future_20260904_222909 PFC future 25
```

Older runs from before the default change (`*_140710`, `*_154516`) are still valid under their
own `run_config.json` and were used for the offline re-scoring.

**In progress**: a PFC **Poisson** past sweep in the user's notebook kernel —
`mFC_data/data/figures/poisson_v4_past_20260905_082332`, 8/25 recdays at 11:16, ~402% CPU. It
carries the both-links arrays. It has **no `run_config.json` yet** (written at the end), so if
it dies partway, `regenerate_summary` still works but the manifest will be missing. Two empty
`poisson_v4_past_20260905_06*` directories are dead launches — safe to delete.

**Not done**: the LEC-vs-PFC comparison figure has not been regenerated since the default
change; `elgaby_ladder.py` has been re-aimed but not re-run (largely superseded — the criterion
questions it was built to answer are now settled).

---

## 10. How to run and verify

```bash
# the gate, in BOTH trees
python code/elasticnet_v4_synthetics.py                    # 30/30
cd mFC_data/code && python elasticnet_v4_synthetics.py     # 30/30

# mirror
diff code/elasticnet_regression_v4.py mFC_data/code/elasticnet_regression_v4.py

# redraw a figure without re-fitting
python -c "import sys; sys.path.insert(0,'code'); import elasticnet_regression_v4 as v4; \
           v4.regenerate_summary('<run dir>')"

# cross-dataset
python code/elasticnet_v4_compare.py --estimator elasticnet
```

Regression numbers any change must preserve, on the PFC past sweep: `{0,11}` NZ mask **292**
(287 selected), strict 90° **33**, state tuning >1/3 **774**. Control 8 of the synthetics
reproduces v3 with `max|diff| = 0` and **must pin the old defaults explicitly** — if a default
moves, add it there or the control silently stops testing equivalence.

---

## 11. Working conventions for this user

* **Synthetic controls gate everything.** One synthetic per claim, emitted in the real
  `data_dic` shape through the unmodified pipeline, stated in both directions (the artifact
  synthetic must be OFF chance when the control is disabled, and AT chance when enabled).
* **Measure, don't assert.** Every number in these docs was measured; thresholds in the
  synthetics are set from measurements, not chosen to make tests pass.
* The user reads the figures closely and will catch a mismatch between a printed number and a
  plotted one — check which quantity a figure shows before quoting it.
* They prefer being grilled on design before implementation, and they push back with good
  instincts. When they challenge a claim, **test it** — three of their challenges were right.
* Plan mode is used heavily; `AskUserQuestion` for genuine forks, `ExitPlanMode` for approval.
* Commit only when asked. Nothing here has been committed.
* Their global `CLAUDE.md` forbids `Co-Authored-By` / "Generated with Claude Code" trailers.
