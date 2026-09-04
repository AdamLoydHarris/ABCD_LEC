# Anchoring regression V4 — past and future lags, per-neuron exports

`elasticnet_regression_v4.py` · `LEC_elasticnet_regression_v4.ipynb` · `elasticnet_v4_synthetics.py`

Per-neuron regression of raw 25 ms firing onto (location × goal-progress phase × lag) anchors,
leave-one-session-out across tasks, after El-Gaby et al. 2024 Figure 5
(`Figure5_Regression.ipynb`, cells 15/21/26). V4 is seeded from `elasticnet_regression_v3.py`,
which is left untouched so the two can be diffed; V4 in legacy mode reproduces V3 exactly.

```bash
python elasticnet_v4_synthetics.py          # 28 controls — run this before trusting a result
```

## Mirroring to the mFC (PFC) dataset

`elasticnet_regression_v4.py` and `elasticnet_v4_synthetics.py` are kept **byte-identical**
between `code/` and `mFC_data/code/`, the way `elasticnet_regression_v3.py`,
`ccgp_state_pairs.py`, `taskphase_periodicity.py` and `persistent_homology_analysis.py` already
are. Edit one, copy to the other, and check:

```bash
diff code/elasticnet_regression_v4.py mFC_data/code/elasticnet_regression_v4.py   # must be empty
diff code/elasticnet_v4_synthetics.py mFC_data/code/elasticnet_v4_synthetics.py   # must be empty
```

Everything dataset-specific lives in the notebook or the loader
(`build_data_dic_from_pfc` in `mFC_data/code/glm_analysis_v2.py`). The one asymmetry the module
itself carries is that **anatomy is optional**: PFC has no `unit_regions` and no
`anatomy_split`, so `build_unit_table` falls back to identity columns (`recday`, `mouse`,
`order`) and `region_summary` / `compare_directions` group by `mouse`. Every regression column
is identical either way, and `table.attrs['has_anatomy']` records which path ran.

The PFC dataset is **El-Gaby's own published data** — see
[`../mFC_data/code/ELASTICNET_V4.md`](../mFC_data/code/ELASTICNET_V4.md) for what differs there
and why each caveat below becomes a statement about the paper. Cross-dataset comparison:
[`elasticnet_v4_compare.py`](elasticnet_v4_compare.py).

## What V4 adds

1. **A future lag direction.** The reference is retrospective only: lag *k* means "this
   (location, phase) anchor was visited *k* phase-steps ago". `lag_direction='future'` builds
   the prospective mirror by reversing the location/phase sequences through the *identical*
   bump loop and reversing back, so both directions inherit the same seeding/roll/wipe
   behaviour. Same trick as `elasticnet_regression_v2.generate_regressors_from_norm`.
2. **Per-neuron exports.** One `.npz` per recday per direction with the fitted betas
   (averaged **and** per fold), the 360-bin actual/predicted tuning curves, and the n=4
   preferred-phase state vectors the headline *r* is computed from; plus **six paged PDFs**
   per recday per direction, in `_all` and `_nonzerolag` variants:

   | file | what it shows |
   |---|---|
   | `*_all.pdf` | summary page per neuron: betas, 360-bin curves, n=4 readout |
   | `*_all_foldbetas.pdf` | the beta matrix **per fold**, each collapsed in its own frame |
   | `*_all_foldratemaps.pdf` | actual vs predicted **per fold** (`plot_fold_ratemap_pages`) |

   The per-fold rate maps matter because **each fold holds out a different session, i.e. a
   different task**, so the actual tuning curve genuinely differs between folds — the summary
   page's fold-averaged actual blends four different tuning curves, which is precisely what the
   model is being asked to predict. Each fold also has its own preferred phase, so the shading
   and the n=4 readout are computed per fold.

   **Run folders are named `{estimator}_v4_{direction}_{stamp}`** — `poisson_…`,
   `elasticnet_…`, `linear_…` — via `run_dir_name(config)`, because the estimator is the
   setting that most changes the numbers. (The old name was hard-coded `elasticnet_v4_*` and
   was written even for Poisson runs; `resolve_latest` still matches those, so check
   `run_config.json` if an old folder looks surprising.)

   **Every run folder gets a `run_config.json`** next to the arrays and figures
   (`write_run_manifest`): the full config, the sessions actually used as folds per recday, the
   requested-vs-completed recday lists, the El-Gaby exclusions, `n_jobs`, wall time, module
   sha + git commit + dirty flag, and library versions. The config is *also* embedded in every
   `.npz`, but that needs Python to read, does not exist if a run produced no npz, and does not
   record the session selection — which is what decides the folds.
   `elasticnet_v4_compare.check_comparable` reads these manifests and warns when the two runs
   being compared differ in estimator, lag count, alpha, or the selection settings.
3. **The region join.** `build_unit_table` / `region_summary` / `compare_directions` tie every
   fitted unit to `unit_regions.pkl`, carrying mean firing rate along.
4. **Diagnostics for four measured problems in the inherited method** (below). Each is exposed
   as a config flag whose default matches the reference, so nothing changes silently.

## One correlation per fold, then averaged

`corrs[ni, fold]`, `corrs_nonzero[ni, fold]` and `cv_tuning_correlations[ni, fold]` are computed
per neuron **per fold**; `mean_corrs = np.nanmean(corrs, axis=1)` collapses to one value per
neuron, skipping folds that produced no fit.

**A single fold's r is not interpretable on its own.** It is a Pearson r over 4 points, whose
null sampling distribution has SD `1/√3 = 0.577` — |r| exceeds 0.5 in 50% of draws from pure
noise and 0.9 in 10%. Measured fold-to-fold spread within a *selected* neuron:

```
                          LEC ah08    PFC
range of r across folds     1.19      1.25    (median; max 1.75 / 1.89)
SD across folds             0.55      0.56    <- against a null SD of 0.577
folds straddling zero       57%       96%     of selected neurons
```

Indistinguishable from sampling noise, as it should be. Averaging unbiased noisy estimates is
legitimate and the population test is over neuron means — but read the per-fold panels for
**shape agreement**, never for the individual r.

## Structural facts that shape everything else

Goal-progress phase advances as a strict 0→1→2 cycle — **0 of 39,732** phase transitions across
every session on disk deviates from +1 mod 3, and segment *durations* are irrelevant to this
(they range 1–10,285 bins). So the anchor phase at lag *k* is **determined** by the current
phase:

```
ap == (pref − lag) % 3        (past)          ap == (pref + lag) % 3        (future)
```

Fits are restricted to preferred-phase rows — **both X and y**
([elasticnet_regression_v3.py:406-408](elasticnet_regression_v3.py#L406-L408), matching the
reference's cell 21) — so only 9 locations × 12 lags = **108 of the 324** columns can be
non-zero in any fit. Verified by refitting every neuron × fold on `ah08_20250613_20250615`:
**402/402** per-fold beta vectors have support on the stripe, zero exceptions.

Two consequences:

* **The 27×12 beta image is 2/3 structural zeros.** All three phase rows *are* used, at
  different lags — the live cells form a diagonal stripe. V4 stores and plots betas as
  **location × lag** (`_collapse_betas`); the phase axis carries no independent information.
* **The prediction is exactly zero at every non-preferred-phase bin** (measured non-zero
  fraction off preferred phase: **0.000**). `raw_to_norm`'s 90-bin grid coincides with the
  phase thirds, so **240 of the 360 bins** of any "predicted tuning curve" are zero by
  construction. The full-360 Pearson (`cv_tuning_correlations`) is inflated by that and is
  largely a phase-tuning correlation; use **`cv_tuning_correlations_pref`**, restricted to
  preferred-phase bins. The V3 polar plots hid this by Gaussian-smoothing with σ=10 bins.

**Preferred phase is refit per fold, and 54/151 neurons (36%) change it across folds.**
`np.nanmean(cv_coeffs, axis=1)` therefore superimposes two different coordinate frames for
those neurons: **23% of fold-averaged matrices** have more than one phase live at some lag.
V4 applies the non-zero-lag criterion **per fold** (`nz_per_fold=True`, as the reference does),
and averages betas only over folds sharing the modal preferred phase.

## Four measured problems in the inherited method

Every one has a config flag; every default matches the reference unless stated.

### 1. `state_tuning_statistic='max'` is confounded by leg duration — the serious one

`raw_to_norm` warps each state interval onto 90 bins by *averaging*, so a longer leg puts more
raw bins into each normalised bin, lowering its variance and therefore its **max** — the
statistic the tuning test z-scores across states. Constant-rate Poisson cells with no tuning
acquire a "preferred state": the shortest leg. 300 pure-noise cells per row, α = 0.05:

| longest:shortest leg | FPR (`max`) | prefer shortest leg | FPR (`mean`) |
|---|---|---|---|
| 1.0× | 0.053 | 26% (chance) | 0.083 |
| 1.5× | 0.407 | — | 0.140 |
| 2.0× | **0.95** (0.73 at 2 Hz) | **94%** | 0.093 |
| 3.0× | **1.000** | **99%** | 0.100 |
| 5.0× | **1.000** | — | 0.100 |

**The median within-session longest:shortest mean leg duration in this dataset is 2.26×**
(p90 5.4×, max 11.5×, n=164 sessions). At that inequality the filter passes essentially
everything and selects on leg geometry. Since
`selected = nonzero_lag & state_tuned & mean_corr.notna()`, this contaminates any
region-by-selection-rate table.

**It is present in the real data.** Across 31 sessions (1,860 units), **47.4% of real units
"prefer" the shortest leg** against a chance of 25%, and the per-session fraction correlates
with that session's leg-duration ratio at **r = 0.43**. Real neurons do carry genuine tuning —
the effect is about half the pure-noise strength — but a large share of the preferred-state
assignment is leg geometry.

`state_tuning_statistic='mean'` is duration-invariant and stays near nominal at every ratio.
The default stays `'max'` to match the reference. **Both masks are computed on every run**
(`state_tuned_mask`, `state_tuned_mask_alt`) so the comparison costs nothing, and
`state_duration_ratio` / `frac_pref_state_is_shortest` are reported per recday.

Note the test is also circular by construction — it picks the preferred state as the argmax
across states and then t-tests that state on the same data — but on *equal* legs that is
empirically close to nominal (0.045–0.055), so duration is the dominant problem, not the
selection.

### 2. `elasticnet_alpha=0.01` on raw counts zeroes 61% of neurons

`Neuron_raw` is spike **counts** per 25 ms bin (mean 0.0735 = 2.94 Hz). ElasticNet
(α=0.01, l1_ratio=0.5, positive) returns an all-zero beta vector for **92/151** units on
`ah08_20250613_20250615`; only **32/151** have ≥3 non-zero betas. Median per-neuron `alpha_max`
is **0.0071** — the paper's fixed α sits above the median neuron's entire regularization path.

```
alpha=0.01     -> all-zero for 90/151 (60%)      # the paper's stated value
alpha=0.001    -> all-zero for  7/151  (5%)
alpha=0.00025  -> all-zero for  0/151  (0%)      # == fitting rate (spikes/s) at alpha=0.01
```

The cut is a **firing-rate cut**: 0.70 Hz mean for zeroed units vs 6.43 Hz for survivors. So a
region difference in selection rate can be a region difference in firing rate.
`region_summary` reports mean rate per region and the selection rate *within firing-rate
quartiles*; a real anatomical effect should survive that.

`alpha_mode='relative'` puts every neuron at the same point on its own path
(`alpha = alpha_frac × alpha_max`). Fitting on rate instead of counts needs no flag — it is
exactly `alpha_mode='fixed', elasticnet_alpha=0.00025`. Poisson (the reference's *executed*
default) never zeroes a coefficient, but costs 1.96 s/fit against ElasticNet's 0.29 s.

### 3. The top-3-lag test was decided by argsort tie-breaking

With most betas exactly 0, `np.argsort(c)[-3:]` returns arbitrary tied-zero indices. One
positive beta at lag 5 gives lags `[7,6,5]` (**passes**) under quicksort and `[10,11,5]`
(**fails**) under mergesort/stable/heapsort — a 56% coin flip. At the default α this affects
27 of the 59 neurons that survive at all. `require_positive_top3=True` takes the top-3 among
strictly-positive betas: one beta at lag 5 → `(True, [5])` deterministically, and lag-5 plus a
large lag-0 beta → `(False, [5,0])` correctly rejected. The summary prints the mask size with
the guard flipped, so the gap is visible.

A fold whose beta vector is all-zero **abstains** from the per-fold vote rather than voting no
— otherwise the consistency requirement would silently become another firing-rate filter.
`n_informative_folds` records how many folds actually carried evidence.

### 4. Untracked bins were kept as training rows

`build_data_dic.locs_to_int` maps SLEAP `nan` → integer **0**, and V3 only NaN'd codes > 9, so
`keep = ~np.isnan(Ltr)` retained them — **4.1%** of bins in the session checked. The reference
drops them. `drop_untracked_bins=True` (V4 default) NaNs codes < 1 as well.

### And what the non-zero-lag criterion is not

It is a **shape descriptor of the beta vector, not a statistical test**. On pure Poisson noise
it fires at **22–23%** in both directions. What *is* unbiased is the correlation, which is what
the headline t-test runs on — measured on pure noise across 3 seeds × 30 cells × 2 directions:

```
r        past   mean=+0.020  t=+0.54  p=0.59      future  mean=+0.025  t=+0.60  p=0.55
r_nz     past   mean=+0.027  t=+0.72  p=0.47      future  mean=+0.002  t=+0.05  p=0.96
r_max    past   mean=+0.005  t=+0.15  p=0.88      future  mean=−0.005  t=−0.13  p=0.90
past vs future: p=0.93 / 0.67 / 0.84
```

No directional bias. But a region difference in *selection rate* must be read against a ~22%
noise baseline, not against zero.

## Past versus future

The worry is that with `num_lags = 12 = 4 states × 3 phases`, past lag *k* and future lag 12−*k*
point at the same task position one full loop apart, making the comparison vacuous. Measured
median column correlation across the 9 locations:

```
k= 1 → 0.09    k= 4 → 0.22    k= 7 → 0.34    k=10 → 0.11
k= 2 → 0.17    k= 5 → −0.06   k= 8 → 0.03    k=11 → 0.03
k= 3 → 0.10    k= 6 → 0.31    k= 9 → 0.02
```

Trial-to-trial route variability separates them — the comparison is identifiable. Note the
reference's own model already contains a degenerate near-future column (its lag 11 is one step
*before* the current position, one loop earlier), which is why its strict criterion excludes
lags {9,10,11} as well as {0,1,2}. A proper future model re-anchors that to the *same* loop.

**Lag 0 is not shared** between the directions: forwards it accumulates the locations visited so
far in the current phase-segment, backwards the ones still to come (agreement 0.62). That is
why the two are fit separately rather than jointly. Everything at lag ≥ 1 refers to complete
segments and is directly comparable.

`compare_directions` reports
`pro_index = (r_future − r_past) / (|r_future| + |r_past|)` per unit and per region.

## Divergences from the reference, deliberate

| | V4 | Reference |
|---|---|---|
| preferred phase | training folds (`pref_phase_source='train'`) | the **held-out** session (cell 21, `phase_peaks[ses_ind_actual]`) — leakage. `'test'` reproduces it |
| `corrs_nonzero` zeroed lags | {11, 0, 1} | {0, 11}, and the correlation NaN'd when a top-3 beta lands in the excluded set |
| non-zero-lag window | lags 2–10 | two variants: not-in-{0,11} and not-in-{0,1,2,9,10,11} |
| beta plots | location × lag, per fold and frame-matched average | none — `Figure5_Regression.ipynb` has 0 `imshow`/`matshow`/`heatmap` calls; its only 3 `savefig`s are the correlation histograms |
| state tuning | OR across all folds (fixed neuron set) | same lineage; note the fold it includes is the held-out one |

`corrs` / `corrs_nonzero` are unchanged in kind: still the Pearson *r* over the n=4 per-state
**means** taken over preferred-phase bins, matching the reference's `Actual_norm_means`.
`state_reduce='max'` is available and both reductions are always stored.

## Synthetic controls

`python elasticnet_v4_synthetics.py` — a `data_dic`-shaped recday enters at the same door the
data does, and the pipeline builds the regressors, picks the preferred phase, fits,
cross-validates and applies the criterion.

| # | control |
|---|---|
| 1 | the bump loop matches an independent segment-based reference at lags 1..11 (agreement 1.0000; lag 0 differs, as it must) |
| 2 | `future(X) == reverse(past(reverse(X)))` exactly |
| 3 | a cell anchored 5 steps in the **past** peaks at past lag 5 and not at the mirrored future lag |
| 4 | the future-anchored mirror image |
| 5 | a pure place cell lands at lag 0 and is **rejected** — and is **detected** once lag 0 is admitted, so the gate is not vacuous |
| 6 | Poisson noise: *r* at chance; the non-zero-lag false-positive rate is reported, not assumed zero |
| 7 | every per-fold beta lies on the (pref ∓ lag) stripe |
| 8 | V4 with legacy flags reproduces V3 exactly (max &#124;diff&#124; = 0 on `cv_coeffs`) |

Control 8 was also run on **real** data (`ah08_20250613_20250615`, 5 sessions, 151 neurons):
`cv_coeffs`, `corrs`, `corrs_nonzero`, `mean_corrs`, `cv_tuning_correlations`,
`state_tuned_mask` and the non-zero-lag mask all match V3 with `max|diff| = 0`. The legacy
flag set is:

```python
RegressionConfigV4(lag_direction='past', pref_phase_source='train',
                   restrict_to_pref_phase=True, drop_untracked_bins=False,
                   nonzero_lag_zero_lags=(0,), require_positive_top3=False,
                   nz_per_fold=False, state_reduce='mean',
                   state_tuning_statistic='max', alpha_mode='fixed')
```

## Runtime

ElasticNet 0.29 s/fit at 77.5k preferred-phase rows × 324; Poisson 1.96 s; `raw_to_norm` 15 ms.
≈8–11 min/recday serial → ≈4 h per direction for 25 recdays, ≈45 min at `n_jobs=6`
(joblib **threading**: sklearn's coordinate descent releases the GIL and `data_dic` stays
shared rather than being pickled to workers). Poisson is ~7× that.
