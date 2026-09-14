# GLM Tuning-Variable Analysis

End-to-end explanation of the per-neuron GLM analysis implemented in [`code/glm_analysis_v2.py`](../code/glm_analysis_v2.py) and surfaced in the notebook [`code/LEC_glm_tuning_variables.ipynb`](../code/LEC_glm_tuning_variables.ipynb).

## 1. Overview

For each neuron, the analysis fits an OLS linear model:

```
y_t  =  X_t β  +  ε_t
```

- `y_t` is the neuron's firing rate at downsampled timepoint `t`.
- `X_t` is a row of the design matrix encoding the animal's current task / behavioural state.
- `β` is the per-neuron weight vector returned by ordinary least squares.

The analysis then asks, for each *group of columns* in `X` (one group per variable like place, head direction, goal progress), whether dropping that group meaningfully degrades the fit. If yes, the neuron is "tuned" to that variable.

Two complementary readouts:

- **F-test** — does removing the variable significantly hurt the fit? (significance / yes-no)
- **CPD (Coefficient of Partial Determination)** — *how much* unique variance does the variable explain? (effect size)

## 2. Variables and units

| Variable | Bins | Type | Notes |
|---|---|---|---|
| `place` | 21 | one-hot | Node 1-21 of the maze. |
| `head_direction` | 36 | one-hot | 10-degree bins around 360°. |
| `goal_progress` | 10 | one-hot | Equal-width fractional bins through each state (`GP_binned`). |
| `speed` | 10 | decile bins | Computed from XY trajectory. |
| `acceleration` | 10 | decile bins | Derivative of speed. |
| `time_from_reward`  *(aka `time_since_reward`)* | 10 | decile bins | Time elapsed since the *last* reward. |
| `time_to_reward` | 10 | decile bins | Time remaining until the *next* reward. |
| `distance_from_reward` *(aka `distance_since_reward`)* | 10 | decile bins | Path length since last reward. |
| `distance_to_reward` | 10 | decile bins | Path length to next reward. |

**Rename note**: `time_since_reward` / `distance_since_reward` are aliases for `time_from_reward` / `distance_from_reward`. Both spellings work everywhere `regressors_to_include` is accepted, and plots use the clearer "since reward" phrasing by default. The internal/canonical name stays `time_from_reward` because that's what `prepare_session_data` in [`code/glm_analysis.py`](../code/glm_analysis.py) returns and we don't want to fork the data prep.

## 3. Design matrix

Total columns: **21 + 36 + 10 + 10 × 6 = 127**.

**No intercept**, **all bins kept** per variable. This makes the matrix rank-deficient by 8 (each one-hot block sums to 1 per row, creating 8 independent linear dependencies between blocks when there's no intercept to absorb them). `np.linalg.lstsq` returns the **minimum-norm** solution.

### Why this parameterization?

Conventional alternative: include an intercept + drop one reference bin per variable. Both produce identical fits in terms of predictions and identical F-test statistics — the column *space* of X is the same either way. The difference is in how individual betas are interpretable:

- Reference-coded (intercept + drop one bin): `β_k` = "how much this bin's firing differs from the reference bin."
- No-intercept, keep-all-bins (this code): `β_k` = "average firing-rate contribution when this bin is active" (modulo a constant absorbed by the min-norm constraint).

The keep-all-bins choice is more interpretable for visualizing per-bin tuning curves. The trade-off is that individual betas are no longer uniquely identified — only their differences are. But the F-test and CPD analyses don't depend on individual beta identifiability, so the choice is safe.

## 4. F-test for variable significance

For each variable `X` (a *group* of columns: 10 for goal_progress, 21 for place, etc.):

1. Fit the **full model** on all 127 columns → compute residual sum of squares `RSS_full`.
2. Fit the **reduced model** with `X`'s columns deleted → compute `RSS_reduced`.
3. Compute the F-statistic:

```
       (RSS_reduced − RSS_full) / df_num
F  =  ──────────────────────────────────────
              RSS_full / df_resid
```

where `df_num` is the number of dropped columns and `df_resid = T − n_params`.

**Big F** = removing the variable hurts the fit a lot → variable matters. **Small F (≈ 1)** = removing it doesn't change much → variable doesn't matter for this neuron.

### Permutation null

Theoretical F-tables assume i.i.d. Gaussian residuals, which firing rates aren't. We use a permutation null:

1. Circularly shift the firing rate by a random offset.
2. Refit both full and reduced models on the shifted data.
3. Compute the F-statistic for this shuffle.
4. Repeat 100 times.

A neuron is classified as **significantly tuned** to a variable if its observed F exceeds the **95th percentile** of the permutation distribution (one-sided, p < 0.05).

### Direction of tuning (per-neuron sign)

If a neuron is significantly tuned, we assign a sign:

- **Ordered variables** (goal progress, speed, time, distance) — slope of a linear fit through the bin-by-bin betas. Positive slope → +1; negative → −1. Captures "fires more as variable increases" vs "decreases."
- **Unordered variables** (place, head direction) — sign of the mean beta. A coarse summary; less meaningful for these variables.

The simplified pie-chart variant (`plot_tuning_piecharts_binary`) ignores the sign and just reports tuned-vs-not. The original `plot_tuning_piecharts` splits +/−/not.

## 5. Coefficient of Partial Determination (CPD)

A complementary, sample-size-independent metric:

```
CPD_X  =  (RSS_reduced − RSS_full) / RSS_reduced
```

Bounded in [0, 1]. Interpretation: the **fraction of variance left unexplained without X that *is* explained by adding X**.

### F vs CPD

| Metric | Tells you | Depends on sample size? |
|---|---|---|
| F | Whether the variable matters | Yes — bigger sample → bigger F for same effect |
| CPD | How much it matters | No — it's a unitless effect size |

In practice you want both. F (with the permutation null) gives you the significance threshold; CPD lets you compare the *magnitude* of contributions across variables, neurons, or recording days without sample-size confounding.

### Joint CPDs

For "is *any* time information significant?", we fit a reduced model with **both** `time_from_reward` and `time_to_reward` dropped together (20 columns removed) and compute one F-stat and one CPD for the joint group, named `time_any`.

This is stricter than an OR-rule across the two single-variable F-tests because it accounts for the variables potentially being substitutable: if neurons split half-and-half between "tuned to time-since" and "tuned to time-to", the OR-rule says ~all are time-tuned. The joint test asks whether dropping *both* hurts the fit — a stronger criterion.

Configure via the `joint_drop_groups` argument:

```python
GLM_results, Permutation_results, CPD_results = run_glm_analysis(
    mouse_recdays, data_dic,
    compute_cpd=True,
    joint_drop_groups=[('time_any', ['time_since_reward', 'time_to_reward'])],
)
```

### Normalized CPD (fraction of explainable variance)

Raw CPD values are typically tiny for single neurons (~0.001–0.01) because single-neuron firing is mostly irreducible noise. To read CPD against what the model *can* explain rather than against the raw residual, normalize by the full-model R²:

```
normalized_CPD_X  =  ΔR²_X / R²_full
                  =  [(RSS_reduced − RSS_full) / TSS]  /  [(TSS − RSS_full) / TSS]
                  =  (RSS_reduced − RSS_full) / (TSS − RSS_full)
```

This is "the fraction of the model's total explainable variance that is uniquely attributable to X." It turns "time uniquely explains 0.2% of residual" into "time accounts for X% of what the model can explain."

When `compute_cpd=True`, `CPD_results[mr][neuron]` stores reserved keys to support this:
- `'__r2_full__'`  : the full-model R² (scalar)
- `'__delta_r2__'` : `{regressor: ΔRSS/TSS}` — unique R² per regressor

Plot helpers compute the normalized version on the fly: `plot_cpd_time_vs_progress(CPD_results, normalize='r2_full')`. Any code iterating over per-neuron CPD dicts must **skip keys starting with `__`** (they're not regressors).

## 6. Time vs progress comparison

This is the primary analysis question in the extended cells: **are goal-progress-tuned neurons distinct from time-tuned neurons, or are they the same population labelled differently?**

Three independent readouts:

1. **Overlap fractions** (`plot_time_vs_progress_overlap`): four bars per mouse — GP only, Time only, Both, Neither. If "Both" dominates, the two codes overlap heavily.
2. **CPD scatter** (`plot_cpd_time_vs_progress`): per-neuron `CPD_progress` vs `CPD_time`. Points along the diagonal → substitutable codes. Points clustered along one axis → dissociated codes.
3. **CPD Δ histogram**: distribution of `CPD_time − CPD_progress`. Centred at zero → balanced. Shifted → one variable carries more unique variance.

A clean separation between time and progress codes would look like: small "Both" fraction, off-diagonal scatter clusters, bimodal/shifted Δ histogram.

**Column-count fairness**: the joint `time_any` group has 20 columns vs `goal_progress`'s 10, which mildly favours time in raw CPD (more columns → more flexibility). There are two complementary column-matched controls, both in the notebook:

1. *Match down* — compare `goal_progress` (10 cols) to `time_from_reward` alone (10 cols), i.e. a single 10-col group on each side.
2. *Match up* — re-fit with `gp_n_bins=20` so `goal_progress` has 20 one-hot columns, matching the full joint `time_any` (20) and `distance_any` (20) groups. This grows the design matrix from 127 to 137 columns; the F-test, CPD, and all plots are otherwise unchanged. Pass the same `gp_n_bins` to `compute_tuning_arrays`.

```python
GLM_results, Permutation_results, CPD_results = run_glm_analysis(
    mouse_recdays, data_dic,
    gp_n_bins=20, compute_cpd=True,
    joint_drop_groups=[
        ('time_any',     ['time_since_reward', 'time_to_reward']),
        ('distance_any', ['distance_since_reward', 'distance_to_reward']),
    ],
)
tuned = compute_tuning_arrays(GLM_results, Permutation_results, gp_n_bins=20)
```

A second structural asymmetry: there are two time variables (since + to) but one progress variable, so the OR-rule overlap gets two shots at significance — keep that in mind when reading the overlap fractions.

## 7. Model fit quality (R²)

Single-neuron full-model R² is typically low (often a few percent). This is normal: most single-neuron firing variance is irreducible (Poisson-like spiking noise, unmodeled inputs). A behavioural GLM at single-timepoint resolution explains only a small fraction.

Why this matters for reading CPD: CPD is the *unique* partial variance on top of the full model. If the full model explains 8% of variance, a per-regressor CPD of 0.002 (0.2% of residual) is a meaningful slice of what's explainable, not a failure. Always read CPD against the full-model R² (see normalized CPD, §5).

`plot_full_model_r2(CPD_results)` shows the per-neuron R² distribution per mouse and pooled. Use it to anchor the CPD magnitudes before interpreting them.

## 8. Raised-cosine basis (continuous variables)

An alternative to one-hot decile bins for the continuous regressors. Each variable is encoded as smooth overlapping cosine "bumps" (Pillow et al. 2008) instead of hard bins.

**Construction** (`make_raised_cosine_basis`): place `n_basis` cosine bumps across the variable's range (1st–99th percentile). Each bump is `0.5·(1 + cos(·))`, width = 2× center spacing, so adjacent bumps cross at half-height and the basis is smooth.

**Spacing** (per `_RAISED_COSINE_SPACING`):
- **Log-stretched** for reward-relative variables (`time_*_reward`, `distance_*_reward`) — concentrates resolution near the low end of the range, i.e. near the reward event, where neural dynamics are typically fastest. Maps values through `log(x − lo + offset)` before placing centers.
- **Linear** for speed, acceleration, and goal progress — even resolution across the range.

**Why 10 basis functions?** It matches the decile one-hot column count, so the design matrix stays 127 columns and `regressor_groups`, the F-test, CPD, and all plots are unchanged — a clean drop-in for comparison. Note this means *no* parameter reduction vs one-hot; the benefit is smoothness (overlapping bumps impose a mild correlation structure → lower effective DOF), not dimensionality. If you want the overfitting-protection benefit, use fewer (6–8) basis functions.

**Scope**: place and head direction stay one-hot. Place is a discrete graph (no smooth ordering). Head direction is circular (0° = 360°) and would need a periodic/von-Mises basis — not implemented here.

**Use** (`continuous_basis='raised_cosine'`):
```python
GLM_results, Permutation_results, CPD_results = run_glm_analysis(
    mouse_recdays, data_dic,
    compute_cpd=True,
    joint_drop_groups=[('time_any', ['time_since_reward', 'time_to_reward'])],
    continuous_basis='raised_cosine',
)
```

**Interpretation**: this is a robustness check on the representation, not the decisive test. Hard-vs-smooth binning rarely flips a population effect. If time > progress survives the smooth basis, it isn't an artefact of decile binning. If it weakens, time tuning may be punctate/non-monotonic structure that 10 smooth bumps under-resolve. The column-matched comparison (§6) remains the primary fairness control.

## 9. Caveats

- **Rank deficiency**: see §3. The F-test and CPD are unaffected, but individual betas are min-norm projections — don't over-interpret single coefficients.
- **Behavioural confounds**: states (and goal progress) correlate with specific motor patterns. Tuning to "goal progress" could in principle reflect tuning to a correlated motor variable. The model attempts to dissociate by jointly including speed/acceleration, but residual confounds with sub-second behaviour may remain.
- **Residual autocorrelation**: firing rates have temporal autocorrelation that violates OLS assumptions on the residual structure. The permutation null (circular shifts) preserves this autocorrelation in both real and shuffled data so the inference is conservative, but the absolute F-values shouldn't be compared to a parametric F-table.
- **CPD column-count asymmetry**: groups with more columns have a mild raw-CPD advantage. Use column-matched comparisons for headline claims (§6).

## 10. API reference (brief)

All in [`code/glm_analysis_v2.py`](../code/glm_analysis_v2.py).

### Fitting

```python
from glm_analysis_v2 import run_glm_analysis

# Original 2-tuple return (existing behaviour)
GLM_results, Permutation_results = run_glm_analysis(mouse_recdays, data_dic)

# With CPD + joint time test (3-tuple return)
GLM_results, Permutation_results, CPD_results = run_glm_analysis(
    mouse_recdays, data_dic,
    compute_cpd=True,
    joint_drop_groups=[('time_any', ['time_since_reward', 'time_to_reward'])],
    regressors_to_include=None,         # or a subset
    continuous_basis='onehot',          # or 'raised_cosine'
)
```

### Tuning summaries & effect sizes

```python
from glm_analysis_v2 import (
    compute_tuning_arrays,
    group_tuning_by_mouse,
    plot_tuning_piecharts,           # 3-slice: +/-/not
    plot_tuning_piecharts_binary,    # 2-slice: tuned vs not
    plot_gp_overlap,                 # legacy GP × everything overlap
    plot_time_vs_progress_overlap,   # GP vs time stacked bars
    plot_cpd_time_vs_progress,       # CPD scatter, Δ histogram, mean bars
    plot_full_model_r2,              # per-neuron R² distribution
    print_tuning_summary,
)

tuned_dict = compute_tuning_arrays(GLM_results, Permutation_results)
mouse_tuning_concat = group_tuning_by_mouse(tuned_dict)

plot_tuning_piecharts_binary(mouse_tuning_concat)
plot_time_vs_progress_overlap(mouse_tuning_concat)
plot_full_model_r2(CPD_results)                                  # anchor CPD magnitudes
plot_cpd_time_vs_progress(CPD_results, normalize='none')         # raw CPD
plot_cpd_time_vs_progress(CPD_results, normalize='r2_full')      # normalized CPD
```

### Raised-cosine basis variant

```python
# Smooth basis for continuous variables (robustness check on the one-hot result)
GLM_results, Permutation_results, CPD_results = run_glm_analysis(
    mouse_recdays, data_dic,
    compute_cpd=True,
    joint_drop_groups=[('time_any', ['time_since_reward', 'time_to_reward'])],
    continuous_basis='raised_cosine',
)
```

### Column-matched 20-bin goal-progress variant

```python
# GP gets 20 one-hot cols (137-col design matrix) to match the joint
# time_any / distance_any groups (20 cols each). Pass the same gp_n_bins
# to compute_tuning_arrays.
GLM_results, Permutation_results, CPD_results = run_glm_analysis(
    mouse_recdays, data_dic,
    gp_n_bins=20, compute_cpd=True,
    joint_drop_groups=[
        ('time_any',     ['time_since_reward', 'time_to_reward']),
        ('distance_any', ['distance_since_reward', 'distance_to_reward']),
    ],
)
tuned_dict = compute_tuning_arrays(GLM_results, Permutation_results, gp_n_bins=20)
```

### Subset analyses

```python
# Drop the forward-looking variables ("no-lookahead" variant)
GLM_results, Permutation_results = run_glm_analysis(
    mouse_recdays, data_dic,
    regressors_to_include=[
        'place', 'head_direction', 'goal_progress',
        'speed', 'acceleration',
        'time_since_reward',          # alias for time_from_reward
        'distance_since_reward',      # alias for distance_from_reward
    ],
)
```

Plot helpers accept the same `regressors_to_include` for label consistency.
