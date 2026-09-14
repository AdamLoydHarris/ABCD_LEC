# Correction: the "El-Gaby semantics" pool rule was wrong, and ED Fig 8a does reproduce

**Date:** 2026-09-08. **Affects:** every panel count and t-value we have reported under
`three_panel_summary(semantics='elgaby')`, in both `ELASTICNET_V5.md` files, in
`ELGABY_FIGURE5_RECONCILIATION.md`, and in the cross-mouse summary figures of every run.
**Does not affect:** any fitted coefficient, correlation, tuning mask or preferred phase. No run
was repeated; everything below is a re-scoring of stored `*_arrays.npz`.

---

## 1. What was wrong

V5's `'elgaby'` semantics required a neuron to have **a finite r in every fold** before it could
enter *any* of the three panels (`selected_all_eg = state_tuned & finite_all_folds`).

His cell 38 has no such requirement. It calls `remove_nan` on each of the three correlation
arrays **independently**:

```python
Predicted_Actual_correlation_mean_<rd>.npy               -> remove_nan -> panel 1
Predicted_Actual_correlation_nonzero_mean_<rd>.npy       -> remove_nan -> panel 2
Predicted_Actual_correlation_nonzero_strict_mean_<rd>.npy-> remove_nan -> panel 3
```

Each of those arrays is already a `nanmean` over folds, which is NaN only if **every** fold was
NaN. So a panel's pool is:

> state-tuned **AND** at least one fold produced a finite value **for that panel**

— a per-panel rule, with no cross-panel or all-fold requirement.

**Why the wrong rule was adopted.** It reproduced the paper's headline n: 481 against his 489
(Fig 5h) and 359 against 349 (ED 8b). That agreement was a coincidence of ElasticNet sparsity.
The same rule applied to a Poisson run gives **140**, nowhere near 489, which is what exposed it.

**Why it mattered so much.** At the reference's fixed `alpha = 0.01` most folds fit all-zero, and
an all-zero fit yields a constant prediction, hence a NaN correlation. Requiring *every* fold to
be finite therefore selected the small, well-fit, high-firing-rate subset — 247 of 1252 PFC
neurons. It suppressed the non-zero-lag panels by ~4× in n and ~2× in t.

## 2. The evidence that the corrected rule is his

His notebooks now run (`mFC_data/mFC_schema-main_unblocked/`, see
`mFC_data/HANDOFF_V5_VS_ELGABY.md`) and his regenerated intermediates are on disk. Re-scoring
**our stored PFC Poisson betas** under the corrected rule, against his own run:

| PFC Poisson, past | ours, corrected | his code | paper (ED 8d / 8a) |
|---|---|---|---|
| all state-tuned | 573, r +0.256, t 15.83 | 581, r +0.272, t 17.15 | 489, t 10.70 |
| non-zero-lag 30° | 298, r +0.134, t 5.11 | 296, r +0.174, t 6.76 | 346, t 4.74 |
| non-zero-lag 90° | **73, r +0.196, t 3.14** | **79, r +0.160, t 2.67** | 229, t 2.81 |
| finite values *before* the tuning mask | 998 / 601 / 158 | 993 / 598 / 155 | — |

n agrees to 1–8 neurons and the pre-tuning counts to 3–5 (0.5%). Under the old rule the same
betas gave 140 / 86 / 19 with t 12.09 / 3.34 / 1.24.

**Checked and refuted:** his cell 26 contains `close_to_anchor_bins_90 = np.arange(12)`, which
would make the 90° exclusion cover all twelve lags. It sits inside the `else:` branch of
`if limited==True:` and therefore only applies to the 24-lag `_beyond` variant. In the 12-lag
run his sets are `[0,11]` and `[0,1,2,11,10,9]` with `Num_max=3` — identical to ours. **There is
no indexing bug**, and the earlier suspicion should not be repeated.

## 3. Every run, re-scored

Corrected numbers are in `three_panel_summary_<direction>_corrected.csv` inside each run
directory, and collected in `data/figures/panel_summary_all_runs_corrected.csv` and
`mFC_data/data/figures/panel_summary_all_runs_corrected.csv`. Each carries all three semantics
(`elgaby`, `elgaby_everyfold`, `v4`) plus the run's config, so nothing has to be looked up.
**The original files were not modified** — verified by md5 before and after
(`code/rescore_runs_corrected.py`).

His corrected pool rule, primary link, superseded value in brackets:

| run | config | all state-tuned | 30° | 90° |
|---|---|---|---|---|
| PFC EN raw, past | repro | 610, t 13.98 | 358, t 3.03 | 120, t **1.99** *(1.64)* |
| PFC EN raw, future | repro | 615, t 14.29 | 349, t 1.07 | 133, t 0.64 |
| PFC EN z-scored, past | repro | 729, t 19.65 | 421, t 4.17 | 126, t 1.55 |
| PFC EN z-scored, future | repro | 729, t 19.58 | 440, t 2.78 | 106, t 0.53 |
| PFC Poisson, past | repro + gate | 573, t 15.83 | 298, t 5.11 | 73, t **3.14** *(1.24)* |
| PFC Poisson, future | repro + gate | 537, t 13.78 | 263, t 2.19 | 50, t 0.09 |
| LEC EN, past | science | 1235, t 29.70 | 528, t 3.87 *(4.49)* | 200, t 0.80 *(1.54)* |
| LEC EN, future | science | 1250, t 34.64 | 430, t 2.33 *(3.44)* | 185, t −1.03 |
| LEC EN, past | repro | 1170, t 31.88 | 544, t 5.34 | 203, t 1.72 *(2.89)* |
| LEC Poisson, past | science, no gate | 1569, t 47.24 | 786, t 11.38 | 275, t −0.10 |
| LEC Poisson, future | science, no gate | 1569, t 52.79 | 649, t 8.21 | 254, t 0.64 |

The correction is **not uniformly favourable**: PFC's panels strengthen, LEC's 30° weakens
(t 4.49 → 3.87) and LEC's 90° weakens (1.54 → 0.80), because a larger pool admits weakly-fit
neurons and the old rule had been inflating mean r by keeping only the well-fit ones.

## 4. What this changes, and what it does not

**Retracted: "ED Fig 8a is unreproduced."** Under his own estimator (Poisson) and his own
scoring, the 90° panel reproduces in sign and significance: ours t 3.14, his t 2.67, paper
t 2.53. It rests on ~1/3 of the reported neurons (73–79 vs 224), so the *effect* reproduces while
the *n* does not. The paper's 224 is reproduced by nothing — not by his 12-lag re-run (79) and
not by his own archived 24-lag outputs (69).

**Still open: where the paper's numbers come from.** His stored `_beyond` (24-lag) outputs are
482 / 278 / 69 against the paper's 489 / 329 / 224, so panel 1 matches the 24-lag run far better
than the 12-lag one (581). A plausible reading is that the paper's panel 1 came from a 24-lag
run; the 90° n remains unexplained under either.

**NOT established: a PFC-vs-LEC difference at 90°.** This is the trap to avoid. The only strong
90° result is PFC Poisson under the reproduction config **with** his positive-mean gate, and our
only finished LEC Poisson runs are science config **without** the gate — they differ in
`min_trials`, `pref_phase_source` *and* the gate. The one **matched** pair available is
ElasticNet, reproduction config, past:

| matched comparison, 90° | n | t |
|---|---|---|
| PFC EN raw, repro, past | 120 | 1.99 |
| LEC EN raw, repro, past | 203 | 1.72 |

Indistinguishable. **No regional claim at 90° should be made** until the matched LEC Poisson run
(spec 12, submitted as job 3546968) has finished.

**Unchanged:** the 30° anchoring result in both regions, the z-scoring findings (the fixed alpha
is a firing-rate filter; the ~30% effect-size gain at 30° in both PFC directions), and the
Poisson link equivalence at α = 1 (both readouts agree to 3–4 decimals on every panel of every
run).

## 5. How this got through, and the control that now catches it

Synthetic control 11 compared our per-neuron *values* against `elgaby_figure5.score_elgaby`, an
independent re-implementation of his cell 26. Those matched to 1e-7 — and they still do. The
control never checked the **pool**, so a wrong denominator sat behind correct numerators.

Control 11 now also asserts (`code/elasticnet_v5_synthetics.py`):

- **11b** the `'elgaby'` pool equals `state_tuned & finite(that panel)` for each of the three
  panels; `'elgaby_everyfold'` is a strict subset; and the two **provably separate** on a neuron
  whose folds are complete except one (constructed by blanking a fold, since these synthetics
  otherwise fit every fold and the comparison would be vacuous);
- **11c** with no positive-mean gate a Poisson run gives *identical* counts under both rules,
  because Poisson never zeroes a coefficient — which is also why the PFC Poisson pool collapsed
  to 140 only when his gate was applied.

101/101 controls pass. The generalisable lesson: **a semantics control must pin the denominator,
not only the per-neuron statistic.**

## 6. Reproducing this

```bash
# both pool rules on any finished run
python -c "
import elasticnet_regression_v5 as v5
t = v5.regenerate_summary('<run dir>', save=False,
                          semantics=('elgaby','elgaby_everyfold','v4'))"

# re-score every run to *_corrected files, leaving originals untouched
python code/rescore_runs_corrected.py            # --dry-run to preview
```

`semantics='elgaby_everyfold'` reproduces any number reported before today, so a figure or
sentence from the old regime can still be traced rather than merely contradicted.

**Note on three runs:** 3546017, 3546018 and 3546901 were already executing when this correction
landed, so the `three_panel_summary_*.csv` **written by those runs themselves** uses the
superseded rule. Use their `*_corrected.csv` (produced by re-running the sweep after they finish),
not their in-run CSV.
