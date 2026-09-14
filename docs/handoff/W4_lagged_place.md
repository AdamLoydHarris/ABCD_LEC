# W4 — Lagged spatial cells: both axes, both directions, both estimators

**Status: not started.** Depends only on W0. Read `README.md` first.

## The question

Do cells encode locations the animal was at *earlier* (a **memory** signal) or will be at
*later* (a **planning** signal), and does that differ by region?

**Direction is a first-class axis.** Every statistic is computed separately for past and future
locations. The implementation is identical, just running the other way — which is exactly why it
is dangerous. **Fix and document one sign convention (`lag > 0 = future`), assert it in code,
and gate it with a synthetic**: a flipped sign silently swaps "memory" and "planning" in every
figure and is otherwise invisible.

## Two lag axes, which are not the same thing

### 1. Phase-lag (the El-Gaby anchoring regression)

`elasticnet_regression_v3.run_cross_validated_regression_v3`. **Critical to understand before
using it:** the lag axis advances on **goal-progress phase transitions, not clock bins**. With
`num_lags=12` and 3 phases × 4 states, 12 lags = **exactly one ABCD loop**. So a "nonzero-lag"
cell is anchored to a location visited *a fraction of a loop ago* — not a place cell with a
delay in milliseconds. It is a **truncated history window, not circular**, so use a *linear*
centre-of-mass over lags 0–11.

As written it builds **past** anchors only. The future version is the same construction on the
time-reversed location/phase sequence, un-reversed afterwards — implement as
`direction={'past','future'}` on `generate_regressors_raw` (line 126), not a second copy.

Report by region:

- the binary `identify_nonzero_lag_neurons` mask, at parity with the reference;
- **`mean_corrs_nonzero − mean_corrs`**, already computed by the pipeline — a continuous,
  threshold-free measure of how much of a neuron's cross-validated prediction comes from nonzero
  lags. **Lead the regional comparison with this**; it needs no new code. The binary mask has no
  null and its threshold is rate-dependent, so it cannot carry a regional claim alone;
- lag centre-of-mass (linear);
- null: permute the 12 lag positions within each (location, phase) anchor of the fitted
  coefficient tensor, preserving anchor structure and magnitudes. Cheap, no refit.

Run **both estimators** — `use_poisson=True` (reference default) and the ElasticNet branch —
both already supported by `RegressionConfigV3`. 2 estimators × 2 directions = **4× the fit
count**; include all four in any timing estimate.

**Save joinable per-neuron pickles** to
`data/glm_outputs/LEC_anatomy/elasticnet_v3__{recday}__{estimator}__{direction}.pkl`. Only SVGs
exist today, which cannot be joined to anatomy.

### 2. Clock-time lag (new)

Per session and neuron, shift firing relative to location over clock lags of **±10 s in 100 ms
steps** (±400 bins at 25 ms, 4-bin steps ⇒ 201 lags), scoring each lag by spatial-ratemap
coherence. Reuse `spatial_ratemaps._compute_spatial_ratemap` and `LOC_TO_GRID` (line 88) and the
split-half machinery in `splithalf_ratemap_consistency.py`. This axis is naturally symmetric, so
past and future fall out of the sign.

Per neuron: the lag maximising spatial coherence, its peak-vs-lag-0 margin, and the past/future
asymmetry of the profile. Null: a large random circular shift of the whole firing trace,
recompute the profile ⇒ per-neuron p for "peak lag ≠ 0".

**The caveat that grows with the ±10 s range.** Median leg duration is **9.38 s**, so far lags
span whole legs and approach the loop period. At lags comparable to those, a pure task-phase
cell produces spurious place-lag structure, because location is itself quasi-periodic. Plot the
leg-duration and loop-period distributions **on the same axis** as the lag profile, and report
the profile restricted to within-leg lags as well. **Any peak beyond one leg duration is
reported as not separable from task-phase periodicity** unless it survives the restricted
version.

### 3. Cross-check

Do the two axes flag the same cells? Contingency of phase-lagged × time-lagged, per region and
per direction. Disagreement is informative and must be reported, not resolved by picking one.

## Verification — synthetic controls, blocking

- A cell simulated at a known clock lag is recovered at that lag; a lag-0 cell does not pass the
  null.
- **Sign convention**: a cell built to fire 1 s *after* leaving a location reports as
  **past/memory**; one built to fire 1 s *before* arrival reports as **future/planning** — in
  *both* implementations.
- The phase-lag permutation null gives ~5% false positives on a synthetic lag-0 cell.

## Deliverables

- `code/LEC_anatomy_lagged_place.ipynb`.
- Figures under `data/figures/anatomy_split/`.
- A section appended to `code/ANATOMY_SPLIT.md`.
