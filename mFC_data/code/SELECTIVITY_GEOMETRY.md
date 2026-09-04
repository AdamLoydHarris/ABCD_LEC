# Selectivity structure and population geometry (Posani-style) — LEC / PFC

Module: `code/selectivity_geometry.py` (mirrored at `mFC_data/code/selectivity_geometry.py`)
Notebooks: `code/LEC_selectivity_geometry.ipynb`, `mFC_data/code/PFC_selectivity_geometry.ipynb`

Ports the battery from **Posani, Wang, Muscinelli, Paninski & Fusi (2026), "Rarely
categorical, highly separable representations along the cortical hierarchy",
Nature** to the ABCD maze data. Section numbers here are the ones the code
docstrings cite.

---

## 0. What the paper does, and what changes here

The paper asks whether neurons within an area cluster into functional types
(*categorical* selectivity), and ties the answer to population geometry
(dimensionality) and computation (linear separability). Across 43 cortical
regions of the IBL Brainwide Map it finds: categorical only in primary sensory
areas, clustering declining along the hierarchy, dimensionality rising, and —
once conditions the population cannot distinguish are merged — maximal
separability essentially everywhere.

Four things do not port directly. Each is a place where a naive translation
manufactures a result.

**(1) There is no α.** His selectivity is `α_n^v = Σ_t β_n^v(t)`, a sum of
time-varying coefficients over *trial time*. `glm_analysis_v2` regresses pooled
samples on one-hot decile blocks; there is no time axis to sum. See §2.

**(2) Betas here carry firing rate.** He z-scores `y` per neuron per time step.
Our GLM fits raw `Neuron_raw` counts, so β magnitude scales with rate. Without
dividing by the neuron's own SD, the α cloud clusters by firing rate rather than
tuning. See §2.

**(3) The inclusion criterion inverts.** His ΔR² ≥ 0.015 is measured against a
*PSTH null* — keep cells that vary with the variables *beyond* the trial-averaged
time course. In the maze the phase-averaged loop response *is* the signal
(goal-progress tuning), so porting his threshold verbatim would discard what we
care about. We use a full-model R² floor plus his firing-rate band, and audit
both (§6).

**(4) Two regions, not 43.** Every "vs. position in the hierarchy" panel
collapses to a LEC-vs-PFC contrast. There is no gradient to report and the
plotting code deliberately does not fit a line through two points.

Against that, the maze has something the IBL task does not: **6 tasks per recday
with index-matched neurons**, so the place↔state conjunction rotates. His own
Limitations section names this as the missing experiment ("the IBL task is
relatively simple… repeating our analysis on a dataset involving multiple tasks
would reveal more clustering"). It is what makes `state × task` a legitimate
condition axis and what enables §7.

---

## 1. Participation ratio

`participation_ratio(X, center=True)` — `(Σλ)² / Σλ²` over the covariance
eigenvalues, computed from whichever Gram matrix is smaller.

Two identities from the paper are asserted in the gate:

- **`check_pr_identity`** — the conditions space and the neural space have the
  same dimensionality (row rank = column rank; his Methods, SVD argument).
- **`pr_from_covariance_trace`** — his eq. 16, `Tr(C)²/Tr(C²)`, evaluated
  directly, must equal the Gram shortcut.

`pr_gaussian_clusters_theory` implements his eq. 25,
`PR = M·k·δ / (1 + M + k·δ)` with `δ = (1+σ²)²`. **It is an approximation, not an
identity**, and this matters at our condition counts. Measured against the exact
eq. 16 (N = 4000, k = 4):

| case | measured | eq. 25 | error |
|---|---|---|---|
| M = 2000, σ = 0 | 3.995 | 3.990 | 0.1 % |
| M = 16, k = 2000, σ = 0 | 15.851 | 15.865 | 0.1 % |
| M = 64, σ = 1 | 12.414 | 12.642 | 1.8 % |
| M = 16, σ = 1 | 9.135 | 7.758 | **18 %** |
| M = 16, σ = 0.5 | 5.019 | 4.301 | **17 %** |

Our condition counts are M = 16–48, i.e. squarely in the regime where the closed
form is wrong by ~15 %. The gate therefore checks the exact identities and the
two asymptotic limits, and only checks eq. 25 where it is valid (M = 64).

---

## 2. The selectivity vector α

`build_alpha_matrix(glm_results, neuron_scales, cpd_results, config)`

```
alpha[n, v] = sign_v · ‖β_n[cols_v]‖₂ / √|cols_v| / sd_n
```

- **RMS, not the raw L2 norm.** Blocks differ in width (place 21, goal_progress
  10, task_state 4). An unnormalised norm would make place the largest axis for
  every neuron purely by column count.
- **`sign_v`** from the existing `glm._beta_direction` — slope across bins for
  ordered regressors, sign of the mean for unordered ones.
- **`/ sd_n`** — see §0(2). Supplied by
  `run_glm_analysis(..., return_scales=True)`.

### `neuron_scales` and the `scales_only` backfill

`run_glm_analysis` gained two keyword arguments (both default off, so existing
2- and 3-tuple returns are unchanged):

- `return_scales=True` appends `{recday: {neuron: {'sd', 'mean'}}}`, recorded
  inside the neuron loop over exactly the fitted samples.
- `scales_only=True` builds the design matrix and pooled FR as usual, records
  the SDs, then skips the OLS fit and permutation tests.

The SD must come from the same code path as the fit — the same session dedup,
node filter (`Locs ≤ 21`), transition mask, downsample and degenerate-design
skip. Recomputing it outside `run_glm_analysis` is the easiest way to introduce a
silent mismatch. `scales_only` was validated to reproduce full-fit SDs **exactly**
(0.0409 / 0.3758 / 0.6954 spikes per 25 ms bin on the first PFC recday) at
**2.7 s versus 686 s** — a 254× speedup, which is what makes it practical to
backfill onto already-cached sections rather than refit them.

### Validation

`check_alpha_against_tuning` compares `sign(alpha[n,v])` with `tuned_dict[n,v]`
wherever the latter is significant. Both derive from `_beta_direction`, so they
must agree exactly. Measured on PFC `distance_gp`: **5351 / 5351 = 1.0000**.

### Known wart

`place` and `task_state` are unsigned (the sign of a mean over an unordered block
is close to arbitrary), so those axes are one-sided and skew the Gaussian null on
those dimensions. Run the clustering with and without the categorical axes; if
the verdict flips, that is the result. `variable_contribution` (§3) quantifies it.

`build_alpha_from_cpd` offers the cached CPD vector as an alternative reduction.
It is a **cross-check only**: CPD is a non-negative variance share, so the cloud
lives in the positive orthant, where a Gaussian null fits badly and can read as
categorical on its own.

---

## 3. Is the representation categorical?

`cluster_quality(A, group_ids, config)` follows his Methods steps 1–6:

1. require ≥ 50 neurons
2. k-means, k = 3…20, `n_init` restarts
3. keep the k maximising the mean silhouette → `ss_data`
4. **purity guard** — drop any cluster whose silhouette mass is > 90 % from one
   group, then restart (his guard is per session; ours is per recday)
5. draw `n_null` matched nulls, each through the **identical** sweep
6. `z = (ss_data − mean(ss_null)) / std(ss_null)`

`n_kmeans_init` is deliberately **one knob used for both the data and the null**.
Giving the data more restarts biases the null's silhouette down and the z-score
up.

Threads are pinned to 1 inside the sweep. These problems are tiny and sklearn's
OpenMP parallelism costs far more in contention than it saves — measured
**27 s → 0.74 s** for one sweep on an 8-core node. Nulls are parallelised across
processes instead.

**Plot both panels.** A sorted selectivity matrix looks convincingly blocked
whenever a few variables dominate — his VISp (categorical) and ACAd (not) look
equally structured by eye. A data panel without its null is not evidence.

---

## 4. Conditions space, M_IC, dimensionality, separability

Posani crosses 4 binary variables into M = 16. The maze has no equivalent set of
trial-constant binaries, and its obvious factors are conjoined: within a task,
state *s* always ends at tower `Task[s+1]`. The multi-task structure is what
breaks that.

| spec | M | notes |
|---|---|---|
| `state_prog_speed` | 16 | 4 × 2 × 2, the IBL analogue; best populated |
| `state_task` | 24 | substrate for the CCGP bridge (§7) |
| `state_prog_task` | 48 | richest; many cells near the 5-trial floor |
| `state_tower` | ~19 | puts place explicitly on an axis |

`build_condition_table` mirrors `ccgp_state_pairs.build_task_state_matrices` —
dedup to one session per unique task, identical neuron columns across tasks, drop
zero-variance neurons, z-score per task — and adds sub-state progress binning and
a speed axis. `check_matches_ccgp` asserts the reproduction: measured
**max_abs_diff = 0.0** at `n_progress_bins=1`. That is what makes §7 a comparison
on identical vectors rather than on two things that merely sound alike.

Trims stay at `trim_start_bins = trim_end_bins = 15`. Per `CCGP_STATE_PAIRS.md`
those trims are what take the place and tone synthetics to chance — both
confounds live in the reward windows at the ends of every leg.

**Speed** is not in `Neurons_norm`, and PFC has no `XY_norm`. It is derived in
raw time with `glm.smooth_and_calculate_scalar_derivatives` and warped with
`raw_to_norm`, the same warp used for the neural tensor, then median-split within
recday. If `XY_raw` is missing the recday is skipped rather than silently falling
back.

**Substrate caveat.** `Neurons_norm` is phase-warped, so time and progress are
conjoined *by construction* on this substrate, and every §4 number inherits that.
Dissociating them is the job of `time_vs_progress_dissociation.py`, not this
module. Do not read these numbers as evidence about that question.

`independent_conditions` implements his ED Fig. 3 merge loop (one-vs-one CV
decoding → threshold 0.666 → Bron–Kerbosch cliques → merge the largest →
iterate). Bron–Kerbosch is written locally; `networkx` is not in `maze_ephys` and
a 25-line routine does not justify a dependency.

`cv_decode` folds by **group** (trial), never by sample — samples within a loop
are strongly autocorrelated and a plain `KFold` leaks. It also **standardises on
train-fold statistics only**: liblinear silently fails to converge on
large-magnitude features, and because convergence warnings are suppressed the
symptom is a plausible near-chance accuracy on trivially separable data. This bug
cost real debugging time during development.

**N-matching is mandatory.** M_IC, PR and separability all grow with population
size, and LEC recdays carry 66–186 tracked cells against PFC's 1–117. Any
unmatched region difference is a statement about recording yield. `n_sensitivity`
reports the whole curve plus surviving recday counts at each N.

---

## 5. THE GATE — `run_synthetic_controls()`

Synthetics flow through the **real** functions unmodified. Run this before
trusting any number on real data.

| kind | must produce | why it matters |
|---|---|---|
| `uneven` | silhouette z ≈ 0 | **the gate.** Elongated but unclustered must read non-categorical |
| `uneven(rotated)` | same z as unrotated | a covariance-matched null is rotation-invariant |
| `isotropic` | z ≈ 0 | no structure to find |
| `categorical` | z ≫ 0 | the test has power |
| `junk` | z ≫ 0 **falsely** | the artefact §6 exists to catch |
| `collinear` | M_IC maximal, separability ≈ 1/3 | dimensionality ≠ separability (his Fig. 6a middle) |
| `highdim` | separability ≈ 1 | ceiling behaves |

Measured output (seed 0):

```
  family                        kind       metric   value                             aux expect  pass
   alpha                   isotropic silhouette_z -0.6400                k=19, a-div=6.96     ns  True
   alpha                      uneven silhouette_z -0.0900                 k=3, a-div=1.53     ns  True
   alpha                 categorical silhouette_z 67.4400                 k=4, a-div=2.57    sig  True
   alpha                        junk silhouette_z 19.8800                 k=3, a-div=1.53    sig  True
   alpha             uneven(rotated) silhouette_z -0.0900 null must be rotation-invariant     ns  True
geometry                     highdim separability  1.0000              PR=13.27, AD=1.000   high  True
geometry                      lowdim separability  0.5170               PR=2.07, AD=0.573    mid  True
geometry                   collinear separability  0.3000               PR=1.02, AD=0.552    low  True
geometry                   collinear         M_IC 16.0000    high M_IC + low separability   >=12  True
   maths          PR rows == PR cols           pr 16.9693                    cols=16.9693  exact  True
   maths PR == eq.16 Tr(C)^2/Tr(C^2)           pr  9.1081                     eq16=9.1081  exact  True
   maths         PR -> k as M -> inf           pr  3.9950                             k=4     ~4  True
   maths         PR -> M as k -> inf           pr 15.8510                            M=16    ~16  True
   maths               eq.25 at M=64           pr 12.4140                   theory=12.642    <5%  True

GATE PASSES
```

Two genuine bugs were caught by building the gate first:

- **Per-centroid normalisation collapsed the collinear synthetic.** Rescaling
  each centroid to a common norm projected 16 collinear points onto the two ends
  of the line, so M_IC read 8 instead of 16 and the control silently stopped
  testing what it claimed to.
- **liblinear non-convergence on unscaled features.** With warnings suppressed,
  trivially separable data decoded at ~0.6. Fixed by train-only standardisation
  inside `cv_decode`.

---

## 6. Null-calibration and inclusion audits

`audit_null_kind` runs the same real data against three nulls:

- `gaussian_cov` — mean **and full covariance** matched. Unimodal by
  construction, so a high silhouette can only come from multimodality. Correct.
- `gaussian_iso` — matches total variance but not its shape. Any elongated cloud
  beats it. Wrong.
- `shuffle_cols` — keeps the marginals, destroys correlations between variables.
  Wrong whenever the regressors are correlated.

`audit_inclusion_threshold` sweeps the full-model R² floor. His Methods warning:
keep every neuron and a spike of near-zero-selectivity "junk" cells appears that
no unimodal null can reproduce, so the region reads categorical for reasons
unrelated to functional structure; cut too hard and the centre of the
distribution is depleted, producing the same artefact from the other side. The
`junk` synthetic reproduces the first failure (z = 19.9 at 50 % zero-selectivity
cells).

Results are in the notebooks; the headline is that the choice of null moves the
LEC z-score by more than a factor of six on identical data.

---

## 7. Separability ↔ CCGP

His Discussion states that maximal separability does not imply absence of
structure, and that a representation can be abstract in the Bernardi (2020) sense
*and* shatter every dichotomy — but he never measures both. Both are measured
here on the `state_task` table, which reproduces the CCGP sampler exactly
(§4), so the numbers are commensurable.

`run_ccgp_separability_join` returns one row per recday with `sep_all`, `sep_ic`,
`pr`, `m_ic` and mean CCGP; `plot_separability_vs_ccgp` is the scatter.

---

## 8. Reproducing

```python
import selectivity_geometry as sg
sg.run_synthetic_controls()          # gate first, always

cfg = sg.SelectivityConfig(regressors=tuple(GP_DIST))
A, meta = sg.build_alpha_matrix(glm_results, neuron_scales, cpd_results, cfg)
K = meta['keep'].values
res = sg.cluster_quality(A[K], meta['recday'].values[K], cfg)
sg.plot_silhouette_null(res, region='LEC')
```

Backfilling `neuron_scales` onto a cached section (seconds, not hours):

```python
*_, scales = glm.run_glm_analysis(recdays, data_dic, return_scales=True,
                                  scales_only=True, **SAME_KWARGS_AS_THE_FIT)
```


---

## 9. Measured results (2026-07-31)

Built from the cached `distance_gp` GLM section (9 regressors, 101 columns) in both
regions. LEC: 2742 neurons / 24 recdays / 5 mice. PFC: 1252 / 25 / 7.
Inclusion (`min_r2_full = 0.02` + 0.5–50 Hz) keeps LEC 1011, PFC 630.

### Categorical? (§3)

| region | n | k | ss_data | ss_null | **z** | α-diversity (max 9) |
|---|---|---|---|---|---|---|
| LEC | 1011 | 4 | 0.2283 | 0.1694 | **13.57** | 4.25 |
| PFC | 630 | 3 | 0.2565 | 0.2299 | **2.79** | 2.29 |

The purity guard dropped nothing in either region.

### Audit 1 — the null matters enormously (§6)

| region | null | ss_null | z | verdict |
|---|---|---|---|---|
| LEC | `gaussian_cov` *(correct)* | 0.1694 | 13.57 | categorical |
| LEC | `gaussian_iso` | 0.0854 | **87.29** | categorical |
| LEC | `shuffle_cols` | 0.1831 | 12.85 | categorical |
| PFC | `gaussian_cov` *(correct)* | 0.2299 | **2.79** | marginal |
| PFC | `gaussian_iso` | 0.0899 | **79.95** | categorical |
| PFC | `shuffle_cols` | 0.2287 | 2.91 | marginal |

An isotropic null inflates LEC's z **6.4-fold** and **flips PFC's verdict outright**.
This is the same miscalibration as an isotropic/shuffle null in the persistent-homology
pipeline, measured here on real data. `shuffle_cols` happens to behave like
`gaussian_cov` in these data, which says the α variables are not strongly
correlated with each other — it is not a general licence to use it.

### Audit 2 — the inclusion threshold matters too (§6)

| min R² | LEC n | LEC z | PFC n | PFC z |
|---|---|---|---|---|
| 0.000 | 2220 | 4.36 | 1108 | 5.90 |
| 0.005 | 2183 | 3.89 | 1091 | 5.72 |
| 0.010 | 1708 | 6.22 | 943 | 5.04 |
| **0.020** | 1011 | **13.57** | 630 | **2.79** |
| 0.050 | 257 | 11.39 | 236 | 3.97 |
| 0.100 | 25 | — | 69 | 2.36 |

**LEC is categorical at every usable threshold** (z = 3.9–13.6), so the sign of that
result is robust even though its magnitude varies 3.5-fold. **PFC's verdict is
threshold-dependent** — z crosses the significance line back and forth
(5.90 → 5.04 → 2.79 → 3.97 → 2.36). PFC should be reported as *not resolved*, not
as categorical or non-categorical.

### Which variables (ED Fig. 4b)

Read `delta_z`. In **both** regions, dropping `goal_progress_distance` raises z
sharply (LEC 13.2 → 22.4; PFC 2.6 → 7.3) — it masks clustering rather than creating
it. `place` is the only variable whose removal *lowers* LEC's z (13.2 → 12.8), i.e.
the only one contributing to the verdict.

### Geometry, genuinely N-matched at N = 20 (§4, `require_full_n=True`)

LEC 23 recdays, PFC 20.

| spec | region | M_IC | PR | PR_IC | sep_all | **sep_IC** |
|---|---|---|---|---|---|---|
| state_prog_speed | LEC | 2.0 | 3.69 | 1.00 | 0.79 | **1.00** |
| | PFC | 3.0 | 4.03 | 2.29 | 0.90 | **1.00** |
| state_task | LEC | 2.0 | 5.74 | 1.85 | 0.48 | **1.00** |
| | PFC | 4.5 | 6.49 | 2.67 | 0.73 | **0.99** |
| state_prog_task | LEC | 4.0 | 6.30 | 2.54 | 0.64 | **1.00** |
| | PFC | 7.0 | 7.31 | 3.48 | 0.83 | **0.99** |
| state_tower | LEC | 2.0 | 5.41 | 1.00 | 0.49 | **1.00** |
| | PFC | 4.0 | 6.17 | 2.27 | 0.74 | **1.00** |

**Posani's headline reproduces.** `sep_IC` ≈ 1.00 in both regions at every condition
definition, with no region difference (Mann-Whitney p = 0.06–0.35), while `sep_all`
varies from 0.48 to 0.90. Once conditions the population cannot distinguish are
merged, what is left is represented at a dimensionality sufficient to shatter it.

PFC exceeds LEC on M_IC (p = 0.009, 0.033 at two specs) and on PR_IC (p = 0.011,
0.009 at two specs), consistently in the same direction at all four. With two
regions this is a contrast, not a hierarchy gradient.

`sep_IC` is also the N-robust quantity: 1.000 at every matched N from 20 to 80 in
both regions, while `sep_all` climbs with N (LEC 0.74 → 0.84, PFC 0.80 → 0.94) and
the surviving recday count falls (PFC 20 → 6).

### CCGP bridge (§7) — unresolved, needs your eye

On the `state_task` substrate with default `CCGPConfig`, cross-task state
generalisation sits **at the module's own role-permutation null** in both regions
(LEC acc 0.423, PFC 0.484; null ≈ 0.48–0.52 per pair) against a ceiling of ~0.84.
Filtering to `place_matched` rows does not change it (0.464 → 0.457), and
`ccgp_state_pairs.summarise_pairs` reports the same values — so this is what that
module produces here, not an artefact of the join.

Two caveats before anyone reads biology into it:

1. Separability is computed on an N-matched subsample while `run_ccgp_recday` uses
   the full population. Same vectors, different neuron counts (`ccgp_n_neurons` is
   reported so the mismatch is visible).
2. `acc` must be compared against `null_mean`, not against a literal 0.5 — cross-task
   decoding has no clean analytic chance level. An earlier version of this analysis
   tested against 0.5 and produced a spurious "significantly below chance".

The bridge machinery is validated (identical sampler, `max_abs_diff = 0.0`); what it
needs is the CCGP configuration you actually trust.
