# The joint manifold of session time and reward time (LEC / PFC)

Module: `code/time_manifold.py` (mirrored at `mFC_data/code/time_manifold.py`)
Notebooks: `code/LEC_time_manifold.ipynb`, `mFC_data/code/PFC_time_manifold.ipynb`

Tests the hypotheses in `mohamady_time_idea/time_torus1.pdf` ("Two nested clocks,
one manifold") and `time_torus2.pdf` ("Compression over time"). We have two
population time signals — time-in-session `T` (decoded by `nn_time_decoder` under
leave-one-session-out CV) and time-since-reward `τ` (`glm_analysis_v2`
`time_from_reward`, adjudicated by `time_vs_progress_dissociation`). This module
asks what *joint geometry* they make.

---

## 0. The question, and what can actually answer it

| candidate | τ is… | β₁ |
|---|---|---|
| **sheet** (R¹×R¹) | an open arc that resets at reward | 0 |
| **cylinder** (S¹×R¹) | a phase that closes at the leg end | 1 |
| **torus** (S¹×S¹) | closed, *and* session time closes too | 2 |

Report 1's argument: a resetting variable is only circular if three things hold —
a well-defined end (variable leg durations remove it), a **rescaled** code (phase,
not clock), and a *continuous* return (a reset teleports). Absolute-seconds τ
fails all three, so the answer is a sheet.

**But the topology is not the interesting part.** Separable, sheared, warped and
conjunctive codes are *all* β₁ = 0 sheets with completely different computational
content. Only cross-condition generalisation separates them, and that is also the
cheaper test and the one that separates at a smaller population. Read §4 before §6.

---

## 1. Substrate — `build_time_tables`

Mirrors `time_vs_progress_dissociation.build_design_tables` (same pooling, same
`get_sessions_for_glm` dedup to one session per task, same `Locs ≤ 21` node filter
∩ transition mask), and adds the one variable that exists nowhere else in the repo:

```python
T_sec = np.arange(n_bins) * (BIN_SIZE_MS / 1000.0) * downsample_factor
```

`glm_analysis_v2` has no session-time regressor, and
`time_coding_analysis.calculate_time_variables` computes it as
`trial_index * 360 / 40` — trial index on the phase-warped substrate, not elapsed
seconds. `T_sec` is computed *before* the validity filter, so removing samples
never shifts the clock.

Unlike `build_design_tables`, session boundaries are **retained** (`session_id`):
with 6–9 sessions per recday the cross-session reset is the strongest test
available for the slow axis.

### Three clocks, three binnings

`FAST_AXIS_BINNINGS = ('abs', 'phase', 'loop_phase')`

- `abs` — seconds since the last reward
- `phase` — τ/D, fraction of the current leg
- `loop_phase` — position in the A→B→C→D→A cycle

The third is ours, not report 1's, and it is the only one that closes as a *task*
variable. It is also the coordinate the existing `analyse_taskphase_ring` result
lives on (that runs on `Neurons_norm`, 90 bins/state × 4).

**Always build the tensor as a set, never singly** — `build_tensor_pair`. Report 1's
single most practical finding is that the binning variable decides the answer more
than the neuron count does.

### Trims are in seconds, not bins

`CCGP_STATE_PAIRS.md` trims 15 bins, but that is the *phase-warped* grid (90 bins
per leg = one sixth of a leg). Copying "15 bins" onto this 250 ms substrate removes
3.75 s from each end of a median 8–12 s leg — over half the data, including the
entire τ ≈ 0 region the reset test anchors on. `trim_seconds = 1.0` instead,
matching the `event_window_s` that `time_vs_progress_dissociation.build_blocks`
uses for the peri-reward transient.

---

## 2. The reset test — `reset_return_curve` (the headline)

Correlation of the population vector with the just-after-reward state, as a
function of τ. Closes → comes back up; resets → decays to a plateau and stays.
The plateau sits well above zero because a tiled code shares a mean: **the
diagnostic is the shape, not the floor.** `return_curve_shape` reduces it to
`dip` and `recovery`.

`reset_jump` is provided **for a figure panel only.** Report 1 §4 measures
1.15 / 2.96 / 3.37 for the *same* sheet depending on which τ the baseline is
matched to; it returns all three and is never used as evidence.

## 3. Closure index — `closure_test`

`(ends − far) / (neighbour − far)`. Returns `ends`, `neighbour`, `far` and
`local_structure = neighbour − far` separately, because the index is undefined
when the axis has no local structure.

**It tests the ends of your analysis window, not the topology of the axis.** The
gate asserts this: `cyl_wrap` is marked `closure_index_blind` and *must* fail the
closure test while its marginal β₁ is 1, reproducing report 1's own wrapped-clock
row. Always report `tau_window`.

## 4. Factorisation — the test that matters

`additive_r2`, `factorisation`, `cross_session_transfer`, `axis_geometry`.

## 5. Topology — `marginal_topology`, `joint_topology`, `h1_stability`

kNN-geodesic metric (Euclidean saturates on a tiled code) and arclength-uniform
resampling (seconds-uniform sampling on a Weber manifold contains one enormous gap
that sets the connectivity scale and buries every real loop). Both failures look
like clean negative results. `persistent_homology_analysis` has the geodesic metric
as a `PHConfig` option (default `'euclidean'` — must be flipped) but nothing
equivalent to the arclength resampling.

**Never report a single-run β₁.** Use `h1_stability`.

## 6. Compression — `covariance_spectrum`, `phase_consistency_with_control`, `lattice_stats`

Only the three statistics that survived report 2's own nulls are implemented.
`random_nonneg_weight_null` must be run before any gridness claim.

---

## 7. THE GATE — `run_synthetic_controls()`

Ten models with known ground truth, injected into a **real** table (real leg
boundaries, real leg-duration distribution, real session structure, real
occupancy, real N) and driven through the **real** functions unmodified. Eight are
ported from report 1's `models.py`; `cyl_loop` and `sheet_3clock` exist only
because our task has a third clock.

The checks are written as **contrasts between models**, not thresholds on absolute
values, for two reasons learned by watching a threshold version pass on one recday
and fail on the next: our timescales are not report 1's, so his magnitudes are not
the target; and a gate tuned until it passes is not a gate.

```
additive index      separable, gain-modulated  >  conjunctive
transfer ratio      separable, gain-modulated  >  conjunctive
tau drift excess    conjunctive  >  separable
closure             each closed model closes in ITS OWN binning; no sheet closes
topology            each closed model's H1 clears 2x the largest H1 any sheet
                    produces under the same binning
```

Measured, `TimeManifoldConfig()` defaults, three real LEC templates:

```
model         truth       clos abs/ph/loop   addIdx  drift  transR   H1 abs/ph/loop
sheet_orth    sheet      -0.19/-0.14/  nan    +1.01     +3   +0.99   0.0/0.0/2.7
sheet_weber   sheet      -0.12/-0.19/-1.73    +1.01    +21   +0.90   0.0/0.0/2.8
conjunctive   sheet      +0.00/-0.32/  nan    +0.61    +64   -0.30   0.0/0.0/1.7
gain_mod      sheet      -0.18/-0.26/-2.17    +0.99    +27   +0.96   0.0/0.0/2.3
ramps         sheet      -0.69/-0.78/-0.47    +2.22     -1   +0.96   0.0/0.0/2.9
cyl_phase     cylinder   -0.34/+0.99/  nan    +0.93     -5   -0.23   0.1/7.0*/2.0
cyl_wrap      cylinder     nan/-0.34/  nan    +1.02    +17   -4.40   5.2*/0.2/2.3
cyl_loop      cylinder   -0.27/+0.50/+0.98    +0.95     +6     nan   0.0/0.0/3.4
torus         torus      -0.48/+1.03/  nan    +0.90    -23   -0.10   0.0/6.1/3.0
sheet_3clock  sheet      -0.29/-0.02/+0.47    +0.97    +21   +1.07   0.1/0.0/6.0

GATE PASSES (31/31 checks)
```

Seed sweep — 5 seeds × 3 real LEC templates, **15/15 pass, 0 failed checks**. The
gate is not seed-lucky; each earlier version that was failed on one seed or one
recday, and each failure is written up in §8.

---

## 8. What building the gate discovered

Six things, each of which would have produced a plausible-looking wrong answer on
real data. All were caught by synthetics driven through the real pipeline.

### 8.1 Leg duration drifts within a session — the transfer test must be range-matched

**The most consequential one.** Leg duration *shrinks* over a session (Spearman
−0.376 / −0.299 / −0.223 against session time on the three LEC recdays tested, all
p < 1e-4): early legs run to a p90 of ~33 s, late ones to ~20 s. So the marginal
distribution of τ **itself** differs between the early and late thirds, and an
unmatched early→late split asks the decoder to extrapolate — reporting a
behavioural drift as a failure of the neural code.

Measured on synthetics whose true answer is "transfers":

| model | unmatched | range-matched | ceiling (matched) |
|---|---|---|---|
| sheet_orth | +0.19 | **+0.82** | +0.96 |
| sheet_weber | +0.02 | **+0.85** | +0.95 |
| conjunctive | −0.62 | +0.46 | +0.93 |

Unmatched, a separable code is indistinguishable from a conjunctive one. Report 1's
simulation drew stationary intervals and never hit this. `factorisation` now
restricts train and test to a common percentile window and reports
`tau_transfer_ratio` = transfer / ceiling on that same window.

An earlier hypothesis — that a context mean-shift on the T-coding neurons was to
blame — was tested by centring each block on its own mean and **rejected**: it made
every score worse (sheet_orth +0.19 → −0.36).

### 8.2 In-sample additive R² measures SNR, not additivity

Report 1's statistic is computed on a noiseless condition-mean tensor. On ours the
residual contains all the per-cell Poisson noise, and how much that costs depends
on the code's dimensionality:

| model | true answer | in-sample additive R² |
|---|---|---|
| sheet_orth (tiled, high SNR) | 1.0 | 0.97 |
| ramps (rank 2, low SNR) | 1.0 | **0.65** |
| conjunctive | 0.47 | 0.49 |

An exactly additive rank-2 code was barely distinguishable from a conjunctive one.
`additive_r2` is now cross-validated — additive and unrestricted models both fit on
one half of each cell's samples and scored on the other, reported as their ratio,
which is 1.0 for any additive code regardless of SNR. `additive_r2_insample` is
kept, documented as the biased version.

### 8.3 A subspace drift angle needs its own null

Estimating the τ subspace from two noisy halves gives a nonzero angle even when
the true drift is 0: `sheet_weber` (true 0°) read +44°, `ramps` (true 0°) read
+89°. `axis_geometry` now returns `tau_drift_null` — the same statistic between
two *interleaved* halves of the same period, carrying the same noise but no drift
— and `tau_drift_excess`, which is the interpretable quantity.

### 8.4 `h1_stability` was not resampling anything

With `n_neurons=None` it drew `rng.choice(N, N, replace=False)` — a permutation of
all neurons. The condition tensor is invariant to neuron order, so every "run"
returned bit-identical bars and the reported spread was exactly **0.00 whatever
the data**, on both true and spurious rings. It now subsamples neurons
(`frac_neurons=0.8`) *and* bootstraps behavioural units.

### 8.5 The resampling unit has to contain the structure under test

Bootstrapping legs while testing the loop-phase axis resamples the four legs of a
loop independently, scrambling loop membership and corrupting `loop_phase` itself.
The true loop ring then read 3.55 ± 3.55, detected in 4/8 runs (bimodal — half the
draws recovered it perfectly, half destroyed it) and was indistinguishable from an
artefact. `_resample_unit_for` selects `loop_id` for the loop-phase axis and
`interval_id` otherwise.

### 8.6 The decoding substrate needs smoothing and a cross-validated penalty — and the gate did not catch it

The first real-data pass returned **negative held-out R² for every decoder,
including the within-condition ceilings** (τ −0.04 to −0.08, T −0.06 to −0.11).
That is not a result; it is a broken substrate, and `factorisation` correctly
refused to divide by it (`tau_transfer_ratio` returned NaN rather than a number).

Two causes, both invisible on the synthetics:

- **Rate.** LEC fires at 0.075 counts per 250 ms bin (≈0.3 Hz), so a single-bin
  population vector is essentially Poisson noise. The session-time ceiling climbs
  from R² = **+0.02 unsmoothed to +0.59 at 4 s** of smoothing. `nn_time_decoder`
  bins at 10 s for the same reason.
- **Penalty.** `alpha = 10` (report 1's fixed value) gives negative held-out R² at
  *every* smoothing level on this data; the same data at `alpha = 1e3` reaches
  +0.59. `ridge_alpha` is now a grid cross-validated on an inner split of the
  training fold only.

Smoothing must be **within leg** (`smooth_within_legs`). Smoothing across a reward
mixes the end of one leg with the start of the next and destroys the fast axis by
construction — the analysis would be measuring its own preprocessing.

**Why the gate missed it, and what that means.** The synthetics put 100 % of their
variance into T and τ at a 12 Hz peak, so their time-SNR is far above real LEC's,
where place, speed and everything else dominate. A gate built only from
high-SNR synthetics cannot catch a rate problem. The gate therefore validates the
*discriminations* — which statistic separates which model — and not the *power*;
the within-condition ceiling is what reports power, and it must be positive before
any transfer ratio is read. Treat a NaN ratio as "the decoder does not work here",
never as a weak effect.

### 8.7 Circular synthetics must actually tile the circle

Report 1 draws von Mises centres from `rng.uniform` with ~200 cells per axis, where
gaps are vanishingly unlikely. We have ~75, and a uniform draw leaves a visible
hole often enough to matter: across six seeds `cyl_phase` measured H₁ = 6.6, 6.3,
**2.5**, 6.9, 6.4, **4.4**. One seed in three produced a "ring" model that does not
cover the ring. Centres are now jittered-uniform. This was a defect in the
generator, not in the analysis.

### 8.8 A missing dependency was reported as a scientific verdict

The gate failed on every recday with `cyl_phase: H1 under phase clears every
sheet`, `cyl_wrap: H1 under abs clears every sheet`, `torus: H1 under phase clears
every sheet` — three statements that read as substantive claims about cylinders and
tori. None of it was real. **`ripser` was not installed in the kernel the notebook
was running under**: it lives in the `maze_ephys` env, and
`LEC_time_manifold.ipynb` was running under `maze_ephys_si104`.

Three layers of error handling turned that into a number:

1. `persistence` does `from ripser import ripser` → `ModuleNotFoundError`.
2. `h1_stability`'s per-run loop catches `(RuntimeError, ValueError)`. `ImportError`
   is neither, so it escaped.
3. `_model_stats` had a bare `except Exception` that rewrote it as
   `H1_mean=nan, stable_ring=False` — **for all ten models, sheets included**.
4. `_sheet_ceiling` then had no finite value to take a max over, returned `nan`, and
   every `H1 clears every sheet` check failed the `np.isfinite(ceil)` test.

The tell is that the whole `H1 abs/ph/loop` column read `nan/nan/nan`. **A sheet
with no ring reads 0.0, not nan.** An all-nan topology column means nothing was
computed; it never means "no ring". The failure count matched the plumbing exactly,
including `loop_powered` gating `cyl_loop` — 3 failures on the 63-loop recday, 4 on
the 99-loop ones.

Three fixes:

- `_model_stats` now absorbs only `(RuntimeError, ValueError, LinAlgError)` — the
  genuine per-run numerical failures. Anything else propagates. **An un-runnable
  dependency must crash, not score.**
- `run_synthetic_controls` raises if `run_topology` is on and no sheet produced a
  finite H₁ in any binning, naming ripser as the usual cause. "The topology never
  ran" and "the ring failed to clear the sheets" are different verdicts and must not
  print the same way.
- `h1_stability`'s empty-result return was missing `stable_ring` and
  `detection_rate`, so `_model_stats` would raise `KeyError` — *outside* the try
  wrapping the call — on the first template where every resample hit
  `geodesic graph will not connect`. It now returns the same keys as the success
  path.

Both notebooks now `import ripser` in their first cell, so a wrong kernel fails at
import rather than eleven checks later. The general lesson is the same one as §8.2
and §8.6: **a fallback that produces a NaN in place of a computation converts an
infrastructure failure into a scientific claim.** Prefer a crash anywhere the
alternative is a number that will be read as evidence.

---

## 9. Power limits, measured

### Loop-phase topology needs ~90+ loops

On the `cyl_loop` synthetic (a true ring by construction):

| recday | n_loops | detected |
|---|---|---|
| ah08_20250613_20250615 | 63 | 4/8 |
| ah08_20250616_20250617 | 73 | 7/8 |
| ah08_20250618_20250619 | 99 | 7/8 |

At 63 loops the true ring (H₁ 3.55 ± 3.55) is not separable from a sheet under the
same binning (`sheet_orth` 3.19 ± 1.45). Detection *rises* with loop count, which
is the signature of underpowering rather than of a spurious feature — report 1's
finding #2 run in reverse. `min_loops_for_loop_topology = 90`; below it the gate
reports the axis as unasserted rather than testing it.

**PFC is better powered on this axis than LEC**, which is the opposite of the
usual direction: PFC recdays carry 119–218 A→A loops against LEC's 63–99, because
PFC sessions run more trials. The gate passes 32/32 on all three PFC recdays
tested (`ab03_*`, N = 63–70 neurons) *with* the loop-phase axis asserted, while
two of three LEC recdays fall below the threshold. Any loop-phase topology
comparison between the regions must therefore be loop-matched as well as
N-matched.

### The 2-D lattice is probably undetectable here

`detectable_period_band` returns it per recday. A lattice period is resolvable only
between ~2 bin widths and ~⅓ of the axis extent. With T ∈ [0, ~1200 s] and
τ ∈ [0, ~30 s] that is a T period of ~60–400 s and a τ period of ~1–6 s. Report 2
predicts ≈19 min and ≈14 s — **both above our upper limit**. Run the test, report
the band beside it, and read a null result outside the band as uninformative
rather than as a refutation.

### PFC population size

PFC recdays carry a median of 53 cells against LEC's 151–183. Report 1's sweep puts
the separable/conjunctive separation at ~50 cells and shows small populations
manufacturing rings. Every region comparison must be N-matched
(`factorisation(..., n_neurons=...)`), or it is a statement about recording yield.

---

## 10. First real-data pass (3 LEC recdays, ah08, 2026-08-19)

Gate passed on all three before these were read. **Preliminary — three recdays of
24, one region, no nuisance controls yet.**

| statistic | abs | phase | loop_phase |
|---|---|---|---|
| reset-curve recovery | +0.15 … +0.23 | **+0.35 … +0.47** | +0.16 … +0.22 |
| closure (fast axis) | −0.35 … −0.38 | −0.02 … +0.11 | +0.45, +0.47, n/a |
| H₁ (8 resamples) | 0.9–1.4, 0–1/8 detected | **0.00, 0/8** | 2.5–4.5, 5–8/8 |

| decoder | ceiling | transfer | ratio |
|---|---|---|---|
| τ across session | +0.05 / +0.09 / −0.07 | −0.08 / −0.08 / +0.18 | — |
| **T across τ** | **+0.64 / +0.38 / +0.74** | **+0.37 / +0.61 / +0.59** | 0.58 / 1.62 / 0.80 |

Three things this says, and one it does not:

1. **The session-time axis is real, strong, and generalises across reward time.**
   Held-out R² 0.38–0.74 within condition, and it still transfers at 0.37–0.61
   when trained at low τ and tested at high τ. This extends the existing
   `nn_time_decoder` result: T is not merely decodable, it is decodable
   *independently of where in the leg you are*.
2. **The fast axis does not close in absolute time.** Closure −0.35 to −0.38, no
   H₁ ring in any resample, and the reset curve decays and stays down. On this
   evidence τ is an open arc — report 1's sheet, not a cylinder.
3. **The reward-time axis is not linearly decodable from the population at all**
   (ceiling ≈ 0). Whatever τ tuning the per-neuron GLM finds does not assemble
   into a population code a ridge decoder can read at 250 ms / 1 s smoothing.
   This is the main open problem, and it blocks the factorisation test for τ: a
   transfer ratio without a positive ceiling is meaningless, which is why the
   module returns NaN rather than a number.

**What it does not say:** nothing here is corrected for place, speed or distance
run, and the `loop_phase` column is the one axis with any ring signal (H₁ 2.5–4.5)
but sits at or below the measured power threshold on two of the three recdays.
Do not read the loop-phase numbers yet.

`additive_index` is NaN on all three because the condition-mean tensor does not
cross-validate at this firing rate (`r2_full` ≈ −0.5) — the same low-rate problem
as §8.6, at the tensor level rather than the decoder level.

## 11. Per-neuron 2-D rate maps

Notebooks: `code/LEC_time_ratemaps.ipynb`, `mFC_data/code/PFC_time_ratemaps.ipynb`.
Functions: `neuron_ratemaps`, `build_ratemaps_all`, `ratemap_autocorr`,
`plot_ratemap_triptych`, `plot_ratemap_grid`, `plot_ratemap_summary`.

Descriptive only — no reliability statistic, no null, no significance. This is for
looking at the data before the geometry statistics in §2–§6 are interpreted.

### Two variants, both axes changing together

| variant | slow axis | fast axis |
|---|---|---|
| `normalised` | fraction of session | fraction of the goal→goal transition |
| `absolute` | elapsed seconds in session | elapsed seconds since last reward |

Comparing them is §1's "bin the fast axis both ways" applied to both axes at once:
a field sharp in the normalised map and smeared in the absolute one is tracking
phase, not a clock.

`build_condition_tensor` gained `T_binning={'frac','sec'}` for this. `'sec'`
defaults its window to **(0, shortest session duration)** — sessions run
1022–1281 s, so binning to the longest would build the right-hand edge of the map
out of the long sessions only, and the tail would get noisier for a reason that
has nothing to do with the neurons.

### Rate is the binding constraint

| region | p10 | median | p90 |
|---|---|---|---|
| LEC | 0.02 Hz | **0.10 Hz** | 0.66 Hz |
| PFC | 0.10 Hz | **0.50 Hz** | 1.62 Hz |

PFC fires ~5× faster than LEC, so **PFC maps are the cleaner of the two**. On a
20×15 grid a cell holds 6–14 s of data, so a median LEC neuron contributes **under
one spike per bin** against ~10 for a median PFC neuron. Most individual LEC maps
are Poisson noise, and nothing in this section distinguishes noise from a field.

Occupancy per cell (LEC, 250 ms bins): 12×10 → 66–141 samples; **20×15 → 30–57**;
30×24 → 11–24 with 31–89 empty cells of 720. 20×15 is the default; 16×12 is the
sparse-safe fallback.

### Four panels per neuron

`plot_ratemap_panels` gives **raw | smoothed | autocorr(raw) | autocorr(smoothed)**
per row. Each map sits beside *its own* autocorrelogram, so the smoothing kernel's
contribution is visible by comparison rather than taken on trust: the smoothed map
is the only readable one at these rates, but smoothing is also what can invent a
field. At LEC rates the raw autocorrelogram is often close to a delta at zero lag
plus noise — at 0.75 spikes per bin that is the honest picture.

Smoothing is **occupancy-aware** (`_smooth_occupancy_aware`): the summed spikes
and the occupancy are smoothed with the same kernel and divided, rather than
smoothing the rate map. Smoothing a rate map directly treats an unvisited cell as
if it had a rate, so empty corners bleed structure inward.

`viridis` for the maps (occupancy-normalised non-negative scalar), `RdBu_r`
centred at zero for the autocorrelograms (a signed correlation), per the
gridmaze-colors skill.

#### The 2σ kernel ellipse — and what it does *not* mean

A Gaussian kernel of width σ has an autocorrelation that is itself Gaussian with
std √2σ, i.e. ∝ exp(−d²/4σ²). At d = 2σ that has fallen to **e⁻¹ = 0.37** of the
centre; at 3σ it is 0.105. `_kernel_ellipse` draws that 2σ contour on the smoothed
autocorrelogram: structure *inside* it is substantially the kernel.

It is an **ellipse, not a circle** — σ is in bins and the two axes have different
bin widths. Measured at the 20×15 `absolute` default: T bin 0.852 min, fast bin
1.987 s, so the semi-axes are **1.704 min × 3.974 s**. Drawing a circle, or
drawing in axes-fraction coordinates, is wrong on both axes at once.

**Outside the ellipse is not the same as real.** Smoothed homogeneous-Poisson maps
still reach |autocorr| of **0.24 median / 0.33 at p90 outside the ellipse** on this
grid. The marker bounds where the *kernel* dominates; it is not a noise floor, and
a correlation of ~0.3 beyond it is comfortably achievable by chance at these
rates. The control that the marker does not hide real signal: a synthetic hex
lattice reaches 0.69 outside it.

The gridness score `plot_ratemap_panels(gridness=True)` prints is
**uncalibrated**. Report 2 §2 found random non-negative weights on a band-pass
basis matching learned gridness (1.21 vs 1.27), so it describes the map and is not
evidence of periodicity — `random_nonneg_weight_null` is what would make it
evidence, and it is not run here.

#### Axes

Both plotters draw in real units. `_map_extent` builds the extent from the outer
**bin edges** — `imshow(extent=...)` wants edges, and the earlier version passed
first/last bin *centres*, shifting every map by half a bin on both axes.
`_lag_extent` puts the autocorrelograms in **lag** units (±16.19 min × ±27.82 s at
the 20×15 `absolute` default); they were previously drawn in pixel indices, so no
period could be read off them at all.

**y ticks and a y label go on every row** of the two leading columns; x ticks go
on the bottom row only. Ticking just the bottom row was the first attempt and it
was wrong in practice: five of six rows then had no frame of reference at all and
you had to scan to the foot of the page to learn what the vertical axis of the
panel in front of you meant.

The neuron id is drawn as free text *outside* column 0, never with `set_ylabel`.
Using `set_ylabel` for it silently overwrote the axis name — which is how the
bottom-left map came to be ticked 0/15/30 with nothing saying those were seconds
since reward, while the figure looked superficially fine.

Two further legibility fixes worth keeping: the peak-rate annotation sits in a
translucent dark box (white-on-`viridis` is unreadable wherever the map is
bright, and that text carries the absolute rate), and `_ticks` normalises `-0.0`
to `0.0` so the origin tick does not render as `-0`.

`plot_ratemap_grid` — the 48-per-page scanning view — keeps ticks on its
bottom-left panel only; it is for scanning, not for reading values off.

### Caching

The notebooks write `ratemaps_{variant}.npz` holding `maps`, `maps_raw`,
`autocorr`, `autocorr_raw`, the `meta` columns **and the full axes dict** —
including `smooth_bins`, `T_window`, `fast_window`, `T_bin`, `fast_bin`. The first
version saved only the centres and labels, which is not enough: without the bin
widths and σ the 2σ kernel ellipse cannot be placed in data coordinates at all, so
anything replotted from cache silently lost its marker. `load_ratemaps_npz` in the
notebook reconstructs the dict `build_ratemaps_all` returns.

### Verified

- **Recovers known structure.** On synthetics injected into a real table, variance
  share on the (T marginal / fast marginal / interaction): `sheet_orth`
  0.50/0.48/**0.02** (additive, as built); `conjunctive` 0.22/0.32/**0.46**. This
  also confirms the axes are not transposed.
- **Normalisation is exact.** Occupancy total equals the in-window sample count,
  and the occupancy-weighted map mean equals the neuron's in-window mean rate to
  **0.000e+00 Hz**. (Only 77 % of samples are in-window — `T_range_frac` trims the
  ragged session ends and τ is capped at the p90 leg duration — so the comparison
  must be against the in-window mean, not the overall mean.)
- **Smoothing does not manufacture real-looking structure.** Smoothed
  homogeneous-Poisson maps have a median CV of **0.22** against **0.58** for real
  LEC maps on the same grid. Note that 0.22 is not 0: smoothed noise does look
  like something, which is why raw is shown beside smoothed.
- **Autocorrelogram controls.** A synthetic hexagonal field gives a centre peak of
  1.00 with a surrounding ring; a single blob gives a centre peak and no ring; an
  all-NaN map returns zeros rather than raising.
- **Ellipse geometry.** Drawn semi-axes equal 2σ × bin width on each axis
  (1.704 min, 3.974 s) and are unequal, i.e. genuinely an ellipse in data
  coordinates.
- **Extents.** The map extent equals `T_window` × `fast_window` exactly; the lag
  extent is symmetric about zero; and the autocorrelogram's zero-lag peak lands at
  (0.000, 0.000) with value 1.00 on every neuron spot-checked.
- **Marker calibration.** Smoothed pure Poisson reaches 0.24/0.33 (median/p90)
  outside the ellipse — so the marker is not a significance boundary — while a hex
  lattice reaches 0.69 outside it, so the marker does not conceal real signal.

## 12. Reproducing

```python
import time_manifold as tm

tables = tm.build_time_tables(mouse_recdays, data_dic)
t = tables[mouse_recdays[0]]

tm.run_synthetic_controls(t)                     # gate first, always

pair = tm.build_tensor_pair(t, binnings=tm.FAST_AXIS_BINNINGS)
{b: tm.closure_test(v) for b, v in pair.items()} # both/all binnings, never one
tm.reset_return_curve(t, binning='abs')
tm.factorisation(t)                              # read tau_transfer_ratio
tm.cross_session_transfer(t)                     # the reset test for the slow axis
tm.h1_stability(t, tau_binning='abs')            # never a single-run beta1
```

## 12. Deliberately not done

- jump-magnitude ratio as evidence (not identified — report 1 §4)
- gridness or module claims without `random_nonneg_weight_null` (report 2 §2)
- β₁ from folded data without a folded null (report 2 §4)
- PR / significant-PC count as evidence about manifold dimension (report 1 §5;
  PR ranges 2.0–14.6 across models that are all 2-D manifolds)
- a hierarchy gradient through two regions
