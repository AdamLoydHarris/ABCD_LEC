# Encoding bias in the time-vs-goal-progress comparison (LEC & PFC)

Companion documentation for the encoding-bias package (identical copies in `code/`
and `mFC_data/code/`):

| module | role |
|--------|------|
| `glm_cv_cpd.py` | shared design-matrix + **cross-validated CPD** core; fair ("matched") encoding schemes |
| `encoding_bias_simulation.py` | **ground-truth** simulation; misattribution matrix |
| `tuning_peak_binning.py` | the PI's data-adaptive bins, **de-circularised** |
| `time_vs_progress_report.py` | aggregator: OOD verdict + CPD robustness in one report |
| `time_vs_progress_dissociation.py` | the **decisive** OOD test (see `TIME_VS_PROGRESS_DISSOCIATION.md`) |

Notebooks: `LEC_encoding_bias.ipynb` / `PFC_encoding_bias.ipynb`.

---

## 1. The concern

The headline time-vs-goal-progress (GP) comparison rested on the CPD / nested-F GLM
in `glm_analysis_v2.run_glm_analysis`. There the compared variables are encoded with
**mismatched binning**:

| variable | binning in `glm_analysis_v2` |
|----------|------------------------------|
| `time_from_reward`, `time_to_reward` | quantile **deciles** (`compute_decile_edges`) |
| `distance_from_reward`, `distance_to_reward` | quantile **deciles** |
| `goal_progress` (temporal), `goal_progress_distance` | **equal-width** bins |

Quantile (decile) binning concentrates a variable's resolution where its samples are
dense; equal-width binning spreads resolution uniformly. The two frames therefore get
**different effective flexibility**. Because time, distance and goal progress are
collinear within a leg (GP ≈ time/D), the more flexible frame can soak up variance
that genuinely belongs to the other — so a genuinely GP-tuned cell can read out as
time- or distance-tuned.

This is one of **three coupled problems**:

1. **Binning-placement asymmetry** — the stated worry. (Note the bin *counts* are
   already matched at 10 each; it is the *placement* that differs.)
2. **No cross-validation** — CPD is computed on the full in-sample fit
   (`glm_analysis_v2.py`, the `(rss_r - rss_full)/rss_r` line), so a better-placed or
   finer basis *mechanically* lowers RSS and inflates CPD. **This is the larger driver
   and re-binning alone does not fix it.**
3. **Within-leg collinearity** — no binning scheme can *adjudicate* two collinear
   regressors; only duration variability can. That is the job of the OOD test.

## 2. Why the PI's first idea needed two fixes

The PI proposed: build a per-neuron tuning heatmap for each variable, take the peak,
look at the population distribution of peaks, and place finer bins where peaks
concentrate (symmetrically for every variable). The instinct is good — treat variables
identically and put resolution where the signal is — but as stated it has two issues:

- **Circularity / double-dipping.** Deriving bin edges from the same neurons you then
  score inflates CPD exactly where you added resolution (Kriegeskorte et al. 2009).
- **Ramps have no peak.** Time/progress cells are often monotonic ramps; argmax lands
  noisily at an edge, so the peak distribution misplaces bins for the very cells of
  interest.

`tuning_peak_binning.py` keeps the good instinct and fixes both (section 4.3).

## 3. The strategy

Attack problems (1) and (2) directly, prove the effect with ground truth, and let the
OOD test (problem 3) deliver the verdict. Everything runs off one shared, pooled,
filtered, downsampled `table` per recday from
`time_vs_progress_dissociation.build_design_tables`, so every analysis sees identical
samples. All continuous variables are encoded through the **same** primitives
(`glm_analysis_v2.compute_decile_edges`, `apply_onehot`, `make_raised_cosine_basis`),
so the simulation tests the *actual* encoding used in analysis.

## 4. The analyses

### 4.1 Ground-truth simulation — does the binning actually bias attribution?
`encoding_bias_simulation.py`

- Take the real covariates for a recday (real sampling/occupancy and the real
  time↔progress↔distance collinearity).
- Synthesise neurons tuned to **exactly one** continuous variable, generated from its
  *continuous* value (so no scheme gets an unfair exact match). Tuning shapes:
  `ramp_up`, `ramp_down`, `early`, `mid`, `late`, and `bimodal` (peaks at both ends —
  the case the concern highlights). True variables: GP(t), GP(d), and all four
  time/distance variables.
- Push the synthetic firing through the GLM design + CPD under each scheme, in-sample
  and cross-validated.
- **Output — the misattribution matrix:** rows = true variable, columns = regressor
  the CPD landed on. A perfectly unbiased pipeline is **diagonal**. Off-diagonal mass
  in the `time-*`/`dist-*` columns of the `GP(t)` row is the bias. `leakage_summary`
  reduces this to a per-variable "fraction correct" and "leaked to time/distance".

Expected: `glm_onehot` shows GP→time/distance leakage; `matched_*` + CV pull the matrix
toward the diagonal.

### 4.2 Cross-validated CPD across encoding schemes — is the result robust?
`glm_cv_cpd.py`

- `fit_encoders` / `apply_encoders` rebuild the GLM design matrix (no intercept, all
  bins, 21 place + 36 HD + 8×10 continuous = 137 cols, identical to the GLM) but let
  every continuous variable share one scheme. Schemes:
  - `glm_onehot` — reproduces the current mismatched GLM (the baseline of concern);
  - `glm_rc` — the `raised_cosine` option (still asymmetric log-vs-linear spacing);
  - `matched_linear`, `matched_quantile`, `matched_rc` — **flexibility-fair**: GP and
    time/distance encoded identically.
- `cpd_cv` computes CPD on **held-out interval folds**: encoders are fit on train rows
  only (quantile edges never see the test leg), full and per-group reduced models are
  fit on train, and squared error is accumulated on held-out legs. CPD/R² come from
  error pooled across folds. A small ridge stabilises the rank-deficient design out of
  fold. Whole legs are held out together because within-leg samples are autocorrelated.
- **Read it as robustness:** if the `GP − time` contrast is stable across `glm_onehot`
  and the `matched_*` schemes, the descriptive ordering is not a binning artefact. If
  it flips, the in-sample CPD was an artefact and only the OOD test is trustworthy.

### 4.3 De-circularised peak-binning — the PI's idea, made valid
`tuning_peak_binning.py`

- Build per-neuron tuning curves on a fine (50-bin) uniform grid, z-score each, and
  take the **population density of |tuning gradient|** (where tuning *changes* fastest)
  — robust to ramps and bumps alike. (`peak_distribution` returns the argmax-of-|z|
  histogram too, as a diagnostic for the PI's original framing.)
- Place `n_bins` one-hot bins carrying **equal cumulative tuning density**: narrow
  where tuning changes fast, wide where flat (`edges_from_density`).
- **No double-dipping:** split legs into two disjoint halves; derive edges on one half,
  score held-out CPD on the other (via `cpd_cv(..., encoders=fixed)`, which fixes the
  edges and only cross-validates the betas), then swap and average. No sample informs
  both its own bins and its own score.

### 4.4 OOD dissociation — the verdict
`time_vs_progress_dissociation.py` (documented separately)

Train on short legs, predict held-out long legs (and reverse). Only duration
variability breaks the within-leg collinearity, so the signed `delta_ood`
(`population_delta_test`) is what actually adjudicates: **>0 leans elapsed time, <0
leans goal progress**. `time_vs_progress_report.build_report` runs this alongside the
CPD-by-scheme robustness and prints a single headline.

## 5. How to read the results

- **The OOD median delta is the answer.** Significant >0 → fixed-latency time; <0 →
  goal progress. Everything else is supporting evidence.
- **CPD-by-scheme is descriptive only.** Use it to check robustness, not to decide.
- **The simulation gives direction and size of the bias** for this dataset's sampling.
- **Peak-binning is one more robustness scheme**, not the adjudicator.

## 6. Caveats

- **Per-cell vs population.** Held-out CPD and OOD delta are weak per neuron; the
  trustworthy statements are population medians (and the OOD Wilcoxon).
- **Region differences.** PFC has no head direction (`has_hd=False`, handled
  automatically); distance ranges differ. Schemes and folds are otherwise identical.
- **CPD can be negative out of fold** (a group's removal *improves* held-out fit by
  chance) — reported as-is; do not clip silently when summarising.
- **Run the full recday set** (`N_RECDAYS = len(mouse_recdays)`) before interpreting;
  the notebooks default to a 3-recday fast pass.
- **Transition-duration filter (`MAX_TRANSITION_SECONDS`).** The notebooks default to
  **30 s**, matching the later `*_filtered` GLM sections
  (`baseline_filtered` / `extended_cpd_filtered` / `distance_gp_filtered`) so the CPD
  comparison is apples-to-apples with the filtered GLM headline. There is a trade-off:
  the standalone dissociation notebook uses **60 s** to maximise the leg-duration
  *range*, which is what powers the OOD test. One `table` feeds all four analyses, so
  the value applies to the OOD test too; raise it to 60 if you want maximum OOD power
  at the cost of GLM comparability. (The first verification run used 60 s.)
- **Matched schemes equalise flexibility, not collinearity.** Even a perfectly fair,
  cross-validated CPD cannot separate time from GP *within* a leg — that is precisely
  why the OOD test, not the CPD, is the verdict.
