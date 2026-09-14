# Anchoring regression V5 — El-Gaby's *code*, not the paper's text

`elasticnet_regression_v5.py` · `LEC_elasticnet_regression_v5.ipynb` · `elasticnet_v5_synthetics.py`
· `elasticnet_v5_compare.py` · `elgaby_figure5.py`

Per-neuron regression of raw 25 ms firing onto (location × goal-progress phase × lag) anchors,
leave-one-session-out across tasks, after El-Gaby et al. 2024 Figure 5. V5 is a copy of
`elasticnet_regression_v4.py`, which is left untouched; V5 under the v4 flags reproduces V4 to
`max|diff| = 0` (control 8b), and V4's own write-up, [`ELASTICNET_V4.md`](ELASTICNET_V4.md),
still describes everything V5 did not change (the future lag direction, the exports, the
structural facts, the α problem, the untracked-bin fix, past-vs-future identifiability).

```bash
python elasticnet_v5_synthetics.py          # 57 controls — run this before trusting a result
diff code/elasticnet_regression_v5.py mFC_data/code/elasticnet_regression_v5.py   # must be empty
diff code/elasticnet_v5_synthetics.py mFC_data/code/elasticnet_v5_synthetics.py   # must be empty
```

## Why V5 exists

V4 implemented the paper's *methods text*. On 2026-09-06 the deposited GitHub code
(`mFC_data/mFC_schema-main/`, `Figure2.ipynb` cells 31/46/48/56, `Figure5_Regression.ipynb`
cells 21/26/38) was read line by line and turns out to differ from the text in three places
that sit upstream of every reported number. Each V5 default follows the code, because the code
is what produced the figures, and each was checked against a published count before it was
adopted. Full derivation: [`ELGABY_FIGURE5_RECONCILIATION.md`](ELGABY_FIGURE5_RECONCILIATION.md).

| | paper text (V4) | his code (V5 default) | check |
|---|---|---|---|
| state-tuning statistic | peak per state and trial (`'max'`) | **mean over the neuron's preferred-phase bins** (`'pref_phase_mean'`), `scipy.stats.zscore` across states with **NaN propagating** (`state_tuning_nan_policy='propagate'`), t-test vs 0, p<0.05 in >1/3 of tasks | PFC: **736** vs his 738 (Fig 5b/c/f); p<0.01: 492 |
| preferred phase | argmax of raw time-weighted mean rate (`'raw_mean'`) | **peak bin within each third** of the trial- and state-averaged normalised curve (`'elgaby_peak'`, his `tuning_phase_boolean_max`) | agrees with `'raw_mean'` on 70% (PFC) / 63% (LEC) of neuron-sessions |
| "with non-zero beta coefficients" | mean r finite | finite r in **every** fold | PFC: **481** vs his 489 (Fig 5h); p<0.01: 359 vs 349 (ED 8b) |
| non-zero-lag neuron | majority of folds pass, all folds averaged | **>= 1 passing fold**, mean over passing folds | PFC 30°: 287 vs his 329 |
| sessions | `num_trials >= 5` | `num_trials > 0` (`MIN_TRIALS`, a notebook setting recorded in the manifest) | 7 PFC sessions differ |

Two things V4 got *wrong* about the reference and that the docs used to state:

* "the state-tuning test matches his exactly" — it matched the **text**. His statistic is not
  the peak, and so **is not leg-duration confounded** in the way `'max'` is (control 9: FPR 0.09
  for his statistic at a 3× leg ratio, against 0.65–1.00 for `'max'`). "The reference has the
  confound too" was wrong.
* the "25% vs 67%" non-zero-lag pass-rate gap — that compared our majority-vote mask with his
  any-fold rule. Under his rule on the same betas our rate is 66%.

Also refuted, with evidence, the handoff's "leading suspect" for the remaining gap: his
deposited `Phases_raw2` is 5 bins per state, but the published Fig 5g raster (extracted from the
PDF) has all 7 non-zero betas exactly on the 3-bin stripe, and the deposited cell 21 cannot run
on the arrays cell 18 writes — the deposited notebook is not the executed code, and the executed
pipeline used 3-bin phases like ours.

## What V5 changes in the module

1. **`identify_state_tuned_neurons_raw`** gains `statistic` / `nan_policy` / `pref_phases`
   arguments and the `'pref_phase_mean'` branch. `'max'` and `'mean'` remain; the other
   statistic is always computed as `state_tuned_mask_alt`.
2. **`session_pref_phases`, `elgaby_phase_curve`, `pref_phase_from_curve`**: his preferred-phase
   rule, computed once per session in `_prepare_session` (both rules are stored per session:
   `session_pref_phase_raw`, `session_pref_phase_elgaby`; `pref_phase_agreement` in the unit
   table). `pref_phase_source='test'` reads the held-out session's rule (his leakage);
   `'train'` pools the training sessions' curves, trial-weighted. `pref_phase_smooth_sigma`
   applies his `smooth_circular` first — whether his `Neuron_` arrays were smoothed is unknown
   (the writer cell is not in the repo), so it is a sensitivity knob.
3. **`add_fold_semantics`** derives his fold semantics post hoc from the per-fold arrays
   (`n_finite_folds`, `n_passing_folds[_strict]`, `mean_corrs_nonzero[_strict]_passing`, and
   `_altlink` variants), so a **V4 export can be re-scored** without a re-fit. `build_unit_table`
   carries both selections (`selected*` = V4 semantics, `selected*_eg` = his), and
   `three_panel_summary(table, semantics=('elgaby', 'v4'))` prints one block per link ×
   semantics with the published (n, t) for that estimator (ElasticNet: Fig 5h / ED 8a / ED 8b;
   Poisson: ED 8d). `plot_cross_mouse_summary` draws one row per (link, semantics).
4. **Reduced-beta r everywhere a non-zero-lag neuron is scored.** V4's per-recday pooled
   histogram and `summarise_recday`'s `mean_r_selected` scored non-zero-lag neurons with the
   all-betas r — a combination the reference never computes. Both now use `mean_corrs_nonzero`.
5. **Plots.** `plot_fold_polar_pages`: polar actual-vs-predicted rate maps per held-out task
   (Fig 5g style; A at the top, clockwise; his σ=10 display smoothing; non-preferred thirds
   shaded because the prediction is exactly zero there). On the `_nonzerolag` pages the solid
   predicted curve is the **reduced-beta** prediction, i.e. what that neuron's r is computed
   from. For Poisson runs the other link's predicted curves are stored
   (`cv_predicted_*_altlink`) and drawn in every page with both r values. GridMaze colours
   (`POLAR_COLORS`): Classic Blue actual, Viva Magenta predicted, Living Coral / Aspen Gold
   reduced betas, Saffron the other link, Stone shading.
6. **Bookkeeping.** `_prepare_session_with_reason` → `results['sessions_skipped']`, printed per
   recday and written to `run_config.json` (with `manifest_extra`, e.g. `MIN_TRIALS`).
   `require_positive_mean_prediction` reproduces his `nanmean(prediction) > 0` gate (identical to
   the existing std > 0 requirement for ElasticNet; drops negative-mean linear predictors for
   Poisson). Run folders are `{estimator}{tag}_v5_{direction}_{stamp}`, where `tag` is
   `_zscore` for a rescaled fit target and empty otherwise.
7. **`y_scaling`** — the only setting in V5 that is *not* read off his code. See the section
   below; the default `'none'` is the reproduction.

## Legacy flags

```python
RegressionConfigV5(state_tuning_statistic='max', state_tuning_nan_policy='omit',
                   pref_phase_method='raw_mean')                       # == V4 exactly (control 8b)
# plus lag_direction='past', pref_phase_source='train', drop_untracked_bins=False,
# nonzero_lag_zero_lags=(0,), nonzero_lag_min=2, require_positive_top3=False,
# nz_per_fold=False, state_tuning_min_fraction=0, alpha_mode='fixed'       # == V3 (control 8)
```

Controls 8 and 8b must pin **every** flag whose default has moved, or they silently stop testing
equivalence and start testing the new defaults.

## Two configs

* **science** (the LEC claim): `MIN_TRIALS = 5`, `pref_phase_source='train'` (the held-out
  session no longer chooses which bins its own score averages over), past and future lags,
  selection by anatomy.
* **reproduction** (same criterion as the paper): `MIN_TRIALS = 1`, `pref_phase_source='test'`,
  past lags, ElasticNet α = 0.01. On PFC this is the run compared to 489 / 329 / 224.

## Target scaling (`y_scaling`) — the one option here that is *not* the reference

Every run above fits raw 25 ms spike counts at a **fixed** `elasticnet_alpha = 0.01`, as the
reference does. ElasticNet drives every coefficient to zero above

    alpha_max = max|Xc.T yc| / (n · l1_ratio)

which scales with `sd(y)`. So a fixed alpha is a **firing-rate filter**. Measured two ways — one
fold of the PFC recday `ah04_01122021_02122021` (preferred-phase rows), and a full 6-fold LEC
recday `ly06_20250613_20250615` (68 units) run both ways through the pipeline:

| | raw counts | ÷ sd(y) |
|---|---|---|
| PFC fold: median `alpha_max` | 0.0074 | 0.0382 |
| PFC fold: all-zero fits at α = 0.01 | **59%** | **1%** |
| PFC fold: all-zero fits, slowest rate quartile | 97% | 3% |
| LEC recday: all-zero folds | **33.3%** | **0.5%** |
| LEC recday: all-zero folds, slowest rate quartile (median 0.68 Hz) | 80% | 2% |
| LEC recday: all-zero folds, fastest quartile (median 32 Hz) | 0% | 0% |
| LEC recday: corr(all-zero, firing rate) | −0.53 | −0.14 |

The correlation is **negative** because it is the *slow* neurons that get zeroed; z-scoring
roughly quarters it.

`y_scaling='zscore_recday'` divides each neuron's fit target by its sd computed **once over all
used sessions of the recday** — one scalar per neuron. Deliberately not per session: a
per-session z-score would divide out the genuine rate differences between the tasks of a recday,
which is part of what the fits should see. Such runs are written to `*_zscore_v5_*` directories
(`run_tag`) and carry `"y_scaling": "zscore_recday"` in `run_config.json`; the raw-count runs
remain the reproduction.

Four facts to keep straight:

1. **Only the sd division bites.** sklearn centres X and y internally, so subtracting the mean
   is a no-op. Fitting `y/s` at `(α, ρ)` is the raw problem with L1 weight `α·ρ·s` and L2 weight
   `α(1−ρ)` — i.e. `α' = αρs + α(1−ρ)`, `ρ' = αρs/α'`, and the coefficients come back `1/s`
   times. Only the **L1** term follows sd, so this is *not* a pure per-neuron alpha rescale at
   `l1_ratio < 1`. Control 13 asserts the identity.
2. **Nothing upstream or downstream moves.** His state test (z across states, then a t-test),
   both preferred-phase rules and the n=4 state-mean Pearson readout are invariant to a positive
   per-neuron affine transform, and the transform is applied to `yfit` alone. `state_tuned_*`,
   `pref_phases`, `cv_actual_tuning` and the fold set come back **bit-identical** to the raw run
   (control 12), so a raw-vs-z-scored comparison isolates the penalty.
3. **The held-out session contributes to the sd.** One scalar for the whole recday means every
   fold sees the same effective penalty, at the cost of the test session entering its own
   scaling. The Pearson readout is scale-invariant, so the only channel is *which* betas clear
   the threshold. Judged negligible; a train-only variant was not added.
4. **It admits low-spike neurons.** A 122-spike unit's `alpha_max` goes 0.0005 → 0.0175, so
   neurons that the fixed alpha silently excluded now enter the selected sets. `n_spikes_recday`
   and `y_sd_recday` are exported per neuron (and stored on raw runs too) so the selection rate
   by spike-count quintile is checkable post hoc — the notebook cell after the relative-alpha
   comparison prints it. No minimum-spike gate is applied; that would be a separate decision.

A z-scored target is negative, so `use_poisson=True` rejects any `y_scaling` other than `'none'`
at construction time (control 15) — the z-scored runs are ElasticNet only. Launch them with
`slurm_v5/submit_v5_run.sh` jobs 6/7 (PFC past/future) and 8/9 (LEC science past/future);
`elasticnet_v5_compare.py --tag zscore` resolves the tagged pair, and an untagged call now
*excludes* `_zscore` directories so it cannot silently pick one as the newest raw run.

**Verified (2026-09-08).** Synthetic controls 12–15 pass in both trees (93/93 here, 89/89 in the
mirror, where control 11 skips because `elgaby_figure5.py` is LEC-tree only). On real data
(`ly06_20250613_20250615`, science config, all 6 folds) the z-scored run's `state_tuned_mask`,
`pref_phases` and `cv_actual_tuning` are identical both to a raw run in the same process **and
to the stored 6 Sep run** `elasticnet_v5_past_20260906_175233`, so fact 2 holds on real data and
not only on synthetics. Its selection did move — the his-semantics pool goes 26 → 35 units while
mean r falls 0.103 → 0.071 and the 90° panel flips sign on 4 → 6 units — but one 68-unit recday
is far too small to read; the four full runs are what to compare.

Constant-rate noise cells make the filter concrete (control 14, fixed alpha): at 0.5 Hz and 2 Hz
**no fold produces a single beta** on raw counts, so those units can never be selected under any
criterion, while every fold does once the target is z-scored. That is the point of the variant
and its risk in one line.

## What the LEC data does under his definitions

Measured before the re-fit (all 25 recdays, 2,851 units): his state test passes **1,433** units
(p<0.01: 948) against **1,477** for `'max'` — a similar *number*, not the same units — and the
NaN propagation alone removes **14.6%** of neuron-sessions, preferentially low-rate ones (tuned
4–8 Hz vs untuned 0.5–2.5 Hz). Preferred-phase agreement between his rule and `'raw_mean'` is
**0.626**. On a 3 Hz dataset his test acts as a firing-rate filter more strongly than on PFC;
read every region contrast against `mean_rate_hz`.

30 of 212 sessions cannot be used and are now reported with reasons: 21 Object-exploration
blocks (no task) and 9 ABCD sessions in which the animal never completed a loop (behavioural,
not a pipeline fault; 4 unique tasks lost). A 31st — `ah10_20250618_20250619` session 5, 29
trials, 162 neurons, camera pinstate truncated to 24 frames — was **salvaged** on 2026-09-06 by
`code/preprocessing/salvage_tracking_offset.py`: the camera offset is recovered from behaviour
(tracked node at poke entry must equal the poked port; the plateau's lower edge plus a
per-mouse reference offset), validated leave-one-out on the 181 pinstate-backed sessions
(median error 1.4 frames, 180/181 within 15 frames; the one miss, `ly05_20250613_20250615`
session 3, is flagged unreliable by the estimator itself and, being pinstate-backed, keeps its
original alignment in the dataset). The salvaged session scores `loc_at_goal` 1.000 and is in
`data_dic_lec.pkl` (backup `data_dic_lec.pkl.PRE_salvage_ah10_20250618_20250619_s5`).

That validation also found two pre-existing pipeline issues, not yet acted on: **101 of 181
videos drop frames**, and in 89 sessions the single-anchor 60 fps model drifts by more than 15
frames by session end (max 4.9 s), so the existing tracking is misaligned late in those
sessions; and the paired-mouse pinstates carry two interleaved rsync trains (harmless except in
two sessions off by 5–9 frames). See `data/processed_data/salvage_staging/salvage_validation.csv`.

## LEC results (2026-09-06/07, ElasticNet α = 0.01, 25 recdays, 2,851 units)

Run directories under `data/figures/`: `elasticnet_v5_past_20260906_175233`,
`elasticnet_v5_future_20260906_175233` (science: `MIN_TRIALS=5`, training-session preferred
phase) and `repro_elasticnet_v5_past_20260906_175233` (reproduction: `MIN_TRIALS=1`,
test-session preferred phase). The salvaged `ah10_20250618_20250619` session 5 is in every fold
set. Under `MIN_TRIALS=1` four one-trial sessions are skipped by the fit itself (a fold needs
two completed trials) and are listed in that manifest.

| run | semantics | all state-tuned | non-zero-lag 30° | non-zero-lag 90° |
|---|---|---|---|---|
| science, past | his | 976, r +0.34, **t 33.7** | 384, r +0.10, **t 4.49** | 125, r +0.07, t 1.54 |
| science, past | v4 | 1235, r +0.30, t 29.7 | 240, r +0.09, t 4.23 | 48, r +0.18, t 3.24 |
| science, future | his | 985, r +0.42, t 39.7 | 313, r +0.08, t 3.44 | 115, r −0.03, t −0.54 |
| science, future | v4 | 1250, r +0.36, t 34.6 | 197, r +0.04, t 1.74 | 48, r +0.04, t 0.58 |
| reproduction, past | his | 887, r +0.35, t 35.0 | 392, r +0.13, **t 5.98** | 129, r +0.13, t 2.89 |
| reproduction, past | v4 | 1170, r +0.32, t 31.9 | 209, r +0.12, t 4.76 | 43, r +0.16, t 2.38 |
| PFC reproduction, past | his | 447, r +0.21, t 14.5 | 246, r +0.07, t 2.48 | 79, r +0.09, t 1.64 |
| paper (PFC) | his | 489, t 9.3 | 329, t 3.9 | 224, t 2.53 |

### LEC Poisson (science config, no positive-mean gate)

`data/figures/poisson_v5_past_20260908_104821` and `poisson_v5_future_20260908_104951`
(MIN_TRIALS=5, training-session preferred phase, no positive-mean gate), his semantics, linear
readout (the `exp(Xβ+b)` column agrees to 3–4 decimals throughout — see below):

| panel | past | future |
|---|---|---|
| all state-tuned, finite r every fold | 1569, r +0.364, t 47.2, d 1.193 | 1569, r +0.413, t 52.8, d 1.333 |
| non-zero-lag 30° | 786, r +0.178, **t 11.38, d 0.406** | 649, r +0.144, **t 8.21, d 0.322** |
| non-zero-lag 90° | 275, r −0.003, t −0.10, d −0.006 | 254, r +0.019, t 0.64, d **+0.040** |

Past exceeds future in the 30° panel (d 0.406 vs 0.322), the same ordering as every other run
here, and the 90° panel is flat in **both** directions — the cleanest null in the project, on 275
and 254 units with no shrinkage anywhere in the model.

**Two independent routes to the same conclusion.** Poisson never zeroes a coefficient, so it has
no firing-rate filter at all — a completely different mechanism from z-scoring the ElasticNet
target, but the same consequence. Both strengthen the 30° panel and neither touches the 90° one:

| LEC 30°, past | n | d |
|---|---|---|
| ElasticNet, raw counts (science) | 384 | 0.229 |
| ElasticNet, recday z-score (science) | *pending* | *pending* |
| Poisson (no shrinkage at all) | 786 | **0.406** |

| LEC 90°, past | n | d |
|---|---|---|
| ElasticNet, raw counts | 125 | +0.138 (t 1.54) |
| Poisson | 275 | **−0.006 (t −0.10)** |

So the LEC 30° anchoring effect survives — and grows — as the rate filter is removed by either
route, while the 90° effect is absent under both. The third confirmation of the link equivalence
comes free here: 0.3635 vs 0.3638 on 1,569 units.

Read with the caveats above: the all-neuron panel includes lag 0 (current position), so a
dataset with strong spatial coding scores high there whatever the lagged structure; the 30° and
90° panels are the anchoring claim. Under the same criterion as the paper, the LEC 30° panel is
as right-shifted as PFC's or more (t 5.98 on 392 units), and the 90° panel is where PFC's
reproduction fails (t 2.89 on 129 units in LEC). State-tuned by his test: 1,569 (science) /
1,463 (reproduction); pref-phase flips between folds 45% (science) / 87% (reproduction);
all-zero fits 47%. The cross-mouse summaries (`cross_mouse_v5_*_summary.svg`) carry both
semantics rows; `elasticnet_v5_compare.py` needs a PFC *future* V5 run before it can draw the
cross-dataset figure.

## Synthetic controls

| # | control |
|---|---|
| 1–7 | as V4: bump loop vs segment reference, reversal, anchored cells recovered per direction, place cell rejected/admitted, noise at chance, stripe invariant |
| 8 | V5 with the v3 legacy flags == v3 (`max|diff| = 0`) |
| 8b | V5 with the v4 legacy flags == v4 (`max|diff| = 0` on betas, correlations, masks, state vectors) |
| 9 | constant-rate cells, 3× unequal legs: `'max'` FPR saturates (0.65–1.00), his `'pref_phase_mean'` stays near nominal (0.09) under both NaN policies |
| 10 | the two preferred-phase rules agree on a broad early-third field and disagree in the documented direction on a narrow late-third burst over a higher early-third mean |
| 11 | `add_fold_semantics` == `elgaby_figure5.score_elgaby` on the same betas (same neurons counted, passing-fold means to 1e-7). Skipped, not failed, in the mirror tree, where that module does not exist |
| 11b | the `'elgaby'` pool is **per panel**, the superseded `'elgaby_everyfold'` rule is a strict subset, and the two provably separate on a neuron missing one fold. This is the check whose absence let the pool bug through |
| 11c | with no positive-mean gate a Poisson run gives identical counts under both pool rules (Poisson never zeroes a coefficient) |
| 12 | `y_scaling='zscore_recday'` leaves the state test, the preferred phases, the actual tuning curves and the fold set **bit-identical** and moves only the betas |
| 13 | `alpha_max(y/s) == alpha_max(y)/s`, and `fit(y/s; α, ρ)` == `fit(y; αρs + α(1−ρ), ...)/s` — the penalty algebra above |
| 14 | the known anchored cells are still recovered with a z-scored target and the place cell still rejected; the low-rate admission is printed |
| 15 | `use_poisson=True` + `y_scaling` raises at construction; a neuron silent across the recday scores NaN rather than dividing by zero |

## Runtime

Unchanged from V4 per fit (ElasticNet 0.29 s, Poisson 1.96 s). The per-session preferred-phase
curves add ~2 s per session. PFC reproduction (25 recdays, past, `n_jobs=6`): 2.5 h.
