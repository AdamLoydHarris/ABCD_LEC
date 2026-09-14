# Handoff: locating the difference between our V5 reimplementation and El-Gaby's code

**Task for the receiving agent.** El-Gaby's deposited notebooks now execute end to end, so
for the first time both pipelines can be compared *at the level of intermediates*, not just
final numbers. Find and quantify where `mFC_data/code/elasticnet_regression_v5.py` diverges
from `Figure5_Regression.ipynb`, and attribute each headline discrepancy to a specific
mechanism.

Written 2026-09-08, updated 2026-09-09. Everything below was measured on this
machine, not inferred.

---

## 1. What is newly available

| Asset | Path |
|---|---|
| His notebooks, runnable, 20 marked edits, audited | `mFC_schema-main_unblocked/` |
| Why each edit exists / what we fabricated / how our run differs | `mFC_schema-main_unblocked/REPRODUCTION.md` |
| The edit ledger (generated, cannot drift) | `mFC_schema-main_unblocked/EDITS.md` |
| Pinned env (numpy 1.22.0, the paper's stack) | conda env `mfc_replication` |
| **All of his intermediates, regenerated** | `replication_run/data/Intermediate_objects/` |
| Executed notebooks with his stdout preserved | `replication_run/executed/` |
| His own archived deposit outputs (the yardstick) | `replication_run/logs/deposit_outputs/` |
| Figure 5 stats, machine-readable | `replication_run/logs/figure5_stats.{json,csv}` |

Run status: **complete** — `Figure2` (0 failed), `Basic_analysis`, `Behavioural Analysis`,
`Figure5_Regression` (Poisson) and `Figure5_Regression_elasticnet` (the α=0.01
apples-to-apples run against V5). **Running** — `Figure3_fast` (a declared variant; see
§6.7). **Pending** — `Figure5_Figure6`, `Figure7`, `Figure2_UMAP` (the last is partial by
construction). Check with `bash mFC_schema-main_unblocked/_preflight/99_status.sh`.

---

## 2. The numbers to explain

The three Figure 5 panels (cell 38), `use_tuned=True`, `use_strict=False` → `State_95` mask:

| panel | his code, Poisson | **his code, ElasticNet** | V5 (ElasticNet α=0.01) | paper | his stored (`_beyond`) |
|---|---|---|---|---|---|
| all state-tuned | 581, r=+0.272, t=17.15 | **623, r=+0.203, t=15.56** | 447–481, t≈14.5 | 489, t=9.3 | 482, t=14.47 |
| non-zero-lag 30° | 296, r=+0.174, t=6.76 | **307, r=+0.073, t=2.87** | 246–287, t≈2.48 | 329, t=3.9 | 278, t=8.41 |
| non-zero-lag 90° | 79, r=+0.160, t=2.67 | **86, r=+0.119, t=2.13** | 79–92, t≤1.8 | 224, t=2.53 | 69, t=3.77 |

Both models: 25/25 recdays, 1252 neurons, 738 state-tuned — exact matches to his figures.

**Candidate (1) is now largely ANSWERED, and it reshapes the task.** The ElasticNet run
(`Figure5_Regression_elasticnet.ipynb`, complete) moves the t-statistics onto V5's:
panel 1 17.15 → **15.56** vs ~14.5; panel 2 **6.76 → 2.87** vs ~2.48; panel 3 n 79 → **86**,
inside V5's 79–92 band. Panel 2's mean r collapses +0.174 → +0.073.

That is the readout mismatch of candidate (2) made visible: cell 26 scores with
`np.sum(regressors*coeffs)` — linear, no `exp`, no intercept — so a Poisson fit read out
linearly inflated the correlations ~2.4× in t. Under ElasticNet the readout matches the
model.

**So the remaining discrepancy is in neuron COUNTS, not effect size** — 623 vs V5's ~465 on
panel 1. Start from the finite-correlation / selection filter, not the regression.

**The headline finding to follow up.** `ELASTICNET_V5.md` concluded the 90° panel (ED Fig 8a)
"should be treated as unreproduced" because V5 got 79–92 neurons at t ≤ 1.8. His own code on
our regenerated inputs gives a **significant** result on *both* models — t = 2.67 (Poisson,
n=79) and t = 2.13 (ElasticNet, n=86) — against the paper's own t = 2.53. So the panel
reproduces in direction and significance while resting on roughly a third of the published
224 neurons. Deciding whether that vindicates ED Fig 8a or merely shows the test is fragile
at this n is a judgement call worth making explicitly.

---

## 3. Already established as EXACT matches — do not spend time here

These pin down large parts of the chain and should be treated as settled:

| quantity | his | ours |
|---|---|---|
| neurons entering Figure 5 (cell 32) | 1252 | **1252** |
| state-tuned via `State_95` | 738 (= paper's 738) | **738** |
| `me11_05122021_06122021` coeff shape | 7 folds, 46 neurons | **(46, 7, 324)** |
| Figure2 total neurons `n` (cell 61) | 2182 | **2182** |
| Figure2 state-tuned p<0.05 / p<0.01 | 1287 / 860 | **1287 / 860** |
| Figure2 cell 31 printed counts | 502, 1214, 1214 | **502, 1214, 1214** |
| Figure2 cell 18 recdays / "not made" | 80 / 4 | **80 / 4** |
| regressor columns | 324 | **324** (9 × 3 × 12) |

Also verified structurally: **100.0000 %** of non-zero fitted coefficients lie on the mod-3
stripe `anchor_phase == (pref_phase − lag) mod 3`, with a non-zero fraction of exactly
0.3333 = 12/36. That is the signature the 3-bin goal-progress array must produce, and it
confirms the `Phases_raw2_` reconstruction.

---

## 4. Candidate divergences, ranked, with how to test each

### (1) Model and regularisation — highest priority, partly answered already
His deposited default is `Poisson_regression=True` → `PoissonRegressor(alpha=1)`. V5 uses
`ElasticNet(alpha=0.01, positive=True)`. Cell 21's own comment calls α=0.01 "used in paper".
The `Figure5_Regression_elasticnet.ipynb` variant (4 marked `VARIANT-EN-*` flips) isolates
exactly this — **read its cell-38 numbers first**; they are the only ones directly comparable
to V5. Outputs land under bare `GLM_anchoring_coeffs_all_*` /
`Predicted_Actual_correlation_*` alongside the `Poisson_*` set.

### (2) The readout drops the link function — check whether V5 matches
Cell 26 reconstructs prediction as `np.sum(regressors_ses * coeffs_ses_neuron, axis=1)` —
**linear, no `exp`, no intercept** (the intercept is never saved). So a Poisson fit is scored
through a linear readout. Confirm what V5 does at prediction time. If V5 applies the link, or
includes an intercept, that alone can move panel 1.

### (3) Session selection — a measured difference exists
His: `non_repeat_ses_maker` = task-non-repeat AND `num_trials > 0`.
V5: `get_sessions_for_glm` filters `num_trials > 2` (`glm_analysis_v3.py:872-884`), while the
V5 config carries `MIN_TRIALS = 1`. **Those thresholds disagree with each other and with his
`> 0`.** Reconcile all three.
Measured consequence in our run: cell 23 reports 12 missing-file skips where his reports 5 —
our `non_repeat_ses` is a superset on recdays his `Num_trials_dic2` does not cover. Skipped
sessions cannot contribute, so the usable fold set is unchanged, but verify that for V5.

### (4) Preferred-phase source (leakage)
His cell 21 reads `tuning_phase_boolean_max_[ses_ind_actual]` — the **held-out** session — and
uses it for both fit and scoring. V5 exposes `pref_phase_source` and the reference config sets
`'test'` to match (`elasticnet_regression_v5.py:261`, warning at 2715). Confirm the reference
run actually used `'test'`, and quantify the `'train'` vs `'test'` gap on the same inputs.

### (5) `num_lags` 12 vs 24
Cells 15/21/26 write 12-lag files; cells 32/38 as deposited read the 24-lag `_beyond` files —
a real inconsistency in the deposit, which is why his stored 482/278/69 came from a `_beyond`
run. We flipped 32/38 to `limited=True` (`F5-02`, `F5-03`), so **our numbers are 12-lag**.
`num_lags = 4 states × 3 phases = 12` is exactly one ABCD loop; the Methods specify 12.
Make sure V5 is compared on the same lag set.

### (6) State-tuning definition
`State_95` here is cell 48's `np.sum(State_zmax < 0.05, axis=1) > len(State_zmax.T)/3`, where
`State_zmax` comes from Figure2 cell 46's per-trial z-score with a **pref-phase mean and NaN
propagation** (not a peak). It yields 738, matching the paper. Check V5's state test against
this exact form.

### (7) Normalisation choices in the regenerated inputs
`Location_<rd>_<s>.npy` uses `normalise`'s default **mean** binning (95.1 % exactly-integer
bins, 199 unique values) rather than `take_max=True` (100 % integer, 21 unique). Rationale in
`REPRODUCTION.md` §4. If V5 bins occupancy differently, that changes which bins are dropped
by `remove_edges`.

### (8) `Num_trials_` mixed basis
Ours uses his deposited `Num_trials_dic2` where it covers a recday (19 of 92) and
`len(trialtimes_)` elsewhere; `preflight_manifest.json` records `num_trials_source` per
recday. Only `> 0` is ever consumed, so this is probably inert — verified that the final
`non_repeat_ses` is unchanged for the one recday where the masks differ.

### Ruled out — do not chase
* **Figure2's unseeded shuffle does not affect Figure 5.** Cell 25 uses
  `random.randrange(...)` with no seed anywhere in the notebook, so `Phase_`/goal-progress
  tuning is stochastic (ours 1656 vs his 1825). But cell 38 loads `Phase_` into `phase_tuning`
  and then sets `neurons_tuned = state_tuning` — **`Phase_` is never used**. State tuning is
  deterministic, hence the exact 1287/860/738 matches.
* **Cell 21 needs no fix.** `REPLICATION_STATUS.md` §5(a) claims its single
  `concatenate_complex2` cannot work. It can: `np.hstack((np.vstack([...])))` uses *grouping*
  parens, not a tuple. Verified 1773/1773 bins, and cell 21 ran unedited over all 25 recdays.
* **5-bin vs 3-bin `Phases_raw2_`.** Settled: 3-bin. Note a 5-bin array would *not* produce
  dead regressor columns (324/324 non-zero either way) — it changes the reachable
  (anchor_phase, lag) set from 12/36 to 7–8/36. See `REPRODUCTION.md`.

---

## 5. What to diff, concretely

**His artefacts**, in `replication_run/data/Intermediate_objects/`:

```
Poisson_GLM_anchoring_coeffs_all_<rd>.npy     (n_neurons, n_folds, 324)  coefficients
GLM_anchoring_coeffs_all_<rd>.npy             same, ElasticNet variant (run in progress)
GLM_anchoring_prep_dic_regressors<rd>.npy     the design matrix, ~628 MB/recday
GLM_anchoring_prep_dic_Location<rd>.npy       per-bin node occupancy used for the NaN mask
GLM_anchoring_prep_dic_Neuron<rd>.npy         per-bin firing, ~227 MB/recday
Poisson_Predicted_Actual_correlation_<rd>.npy          (n_neurons, n_folds)
Poisson_Predicted_Actual_correlation_mean_<rd>.npy     (n_neurons,)  <- panel 1 input
Poisson_Predicted_Actual_correlation_nonzero_mean_<rd>.npy         <- panel 2 input
Poisson_Predicted_Actual_correlation_nonzero_strict_mean_<rd>.npy <- panel 3 input
tuning_phase_boolean_max_<rd>.npy    per-session preferred-phase one-hot (feeds pref_phase)
State_zmax_<rd>.npy                  state-tuning p-values
State_95<rd>.npy / State_99<rd>.npy  the tuning masks (NOTE: no separator before <rd>)
Phases_raw2_<rd>_<s>.npy             3-bin goal progress, per bin
```

**V5 artefacts.** `export_regression_outputs` (`elasticnet_regression_v5.py:1614`) writes
`<recday>_<lag_direction>_arrays.npz` plus `write_run_manifest` (:1683).
**Caveat: `mFC_data/data/figures/elasticnet_v5_past_20260906_112347/` is EMPTY** — that run's
arrays are gone. You will need to re-run V5 with export enabled into a fresh `out_dir` to get
diffable arrays.

**The single most informative comparison is the design matrix.** If his
`GLM_anchoring_prep_dic_regressors<rd>` and V5's regressors differ for the same
(recday, session), then every downstream number differs and nothing else is worth comparing
until that is resolved. Compare on one small recday first — `me10_20122021_21122021` has a
single neuron, `me10_17122021_19122021` has 6.

Suggested order: design matrix → coefficients → per-neuron correlations → panel counts.
At each stage, diff **per (recday, fold, neuron)**, not in aggregate; the aggregate can agree
while individual folds are misaligned.

---

## 6. Traps that will cost you time

1. **`me11_05122021_06122021` folds are genuinely misaligned in his code.** Cells 15/21 build
   **7** folds (confirmed: our coeffs are `(46, 7, 324)`); cell 26 hardcodes
   `num_non_repeat_ses_found = 6` and drops session 3, so for k ≥ 3 it pairs fold k's
   regressors with a *different* session's phases and trial times, and session 7 is never
   scored. **46 neurons affected.** We left this in deliberately. V5 has
   `EL_GABY_EXCLUDED_SESSIONS` — check the semantics match.
2. **Cell 26 re-reads enormous files inside its loops** — the 628 MB regressor file once per
   fold, and the coefficients file once per *neuron*. ~5 GB of I/O per recday. It looks like a
   hang; it isn't. Track progress by counting saved per-recday files, not by watching stdout —
   nbclient buffers a cell's stdout until the cell completes.
3. **`F2-06` inserted a cell at Figure2 index 31**, so every later cell is +1 versus the
   deposit's numbering. Anything keyed by cell index must account for it.
4. **`State_95<rd>.npy` has no separator** before the recday. `State_<rd>.npy` is a different
   file.
5. **Not every archived output is his.** Some carry numpy ≥ 2.2 array reprs, i.e. they came
   from recent local runs. The Figure 5 chain outputs do carry his `Python38` /
   `C:\Users\moham` provenance in stderr.
6. **Figure3 cell 36 crashes on `me10_14122021_15122021` session 5 as deposited.** That
   session has `trialtimes` (so `num_trials > 0`, passing cell 36's only gate) but no
   `Neuron_raw`, so no 360-bin `Neuron_` array can exist and the unguarded load dies with
   `FileNotFoundError`, aborting the notebook before cell 45. Cell 29 wraps the identical
   load in `try/except`; cell 36 does not. Fixed as edit `F3-03`, matching the notebook's
   own convention. Anyone re-deriving Figure3 from the raw deposit will hit this.
7. **Use the pinned env.** numpy ≥ 1.24 raises on the deposit's ragged `np.asarray`, and a
   blanket `dtype=object` patch is *incorrect* — `concatenate_complex2` is polymorphic and the
   outer call must stay numeric for `~np.isnan(...)`. See `REPRODUCTION.md` §2.

---

## 7. Commands

```bash
E=/nfs/nhome/live/aharris/.conda/envs/mfc_replication/bin/python
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data

bash mFC_schema-main_unblocked/_preflight/99_status.sh          # what is running
$E mFC_schema-main_unblocked/_preflight/96_extract_figure5_stats.py   # panel stats -> json/csv
$E mFC_schema-main_unblocked/_preflight/95_checkpoints.py Figure5_Regression.ipynb
$E mFC_schema-main_unblocked/_preflight/90_audit_diff.py       # the copy is still minimal

# rebuild the ElasticNet comparison notebook if needed
$E mFC_schema-main_unblocked/_preflight/20_apply_edits.py --variant elasticnet
$E mFC_schema-main_unblocked/_preflight/40_run_notebook.py \
     "Figure5_Regression_elasticnet.ipynb" --skip 15
```

`Input_folder` for every notebook is
`replication_run/data/Intermediate_objects/` — a writable 52 GB copy. The deposit at
`mFC_data/data/` keeps its original permissions and was verified byte-identical after the
runs (200-file md5 sample); never point a run at it, because `Intermediate_objects/` there is
3841 hardlinks into the raw-array subfolders and Figure2 cell 20 *renumbers* what it writes.

---

## 8. Deliverable requested

For each headline gap (581 vs 447–481; 296 vs 246–287; 79/t=2.67 vs 79–92/t≤1.8), name the
mechanism, show the intermediate at which the two pipelines first diverge, and state whether
the difference is a bug in V5, a faithful consequence of his code, or a property of the
deposit. Where a mechanism can be toggled, quantify it by toggling rather than by argument —
that is what the ElasticNet variant is for, and the same approach extends to
`pref_phase_source`, the `num_trials` threshold, and the 12-vs-24 lag set.
