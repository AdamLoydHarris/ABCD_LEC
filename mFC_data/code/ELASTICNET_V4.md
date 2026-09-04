# Anchoring regression V4 — PFC / mFC dataset

`elasticnet_regression_v4.py` · `PFC_elasticnet_regression_v4.ipynb` · `elasticnet_v4_synthetics.py`

Mirror of the LEC write-up at [`../../code/ELASTICNET_V4.md`](../../code/ELASTICNET_V4.md).
The analysis module is kept **byte-identical** to the LEC copy — `diff` it — so everything in
that document about the method applies here unchanged. This file records only what is different
about *this dataset*.

```bash
python elasticnet_v4_synthetics.py          # 28 controls — run before trusting a result
diff ../../code/elasticnet_regression_v4.py elasticnet_regression_v4.py   # must be empty
```

## This is El-Gaby's own dataset

`data/MetaData/combined_ABCDonly_days.npy` is the file `Figure5_Regression.ipynb` itself loads,
and the 25 recdays are `me08 / me10 / me11 / ah03 / ah04 / ah07 / ab03` — including
`me11_05122021_06122021`, the one his cell 26 hard-codes a fix for.

So running V4 here is not "the same analysis on another region": it re-runs the published
Figure 5 analysis on the published data. Each caveat below is therefore a statement about the
paper, not about our recordings.

## What differs from LEC

| | PFC | LEC |
|---|---|---|
| units / recdays | **1252 / 25** | 2851 / 25 |
| sessions → unique tasks | **194 → 149** (~6 folds/recday) | — |
| trials per session (median) | **27** | 18 |
| mean firing rate | **6.25 Hz** (p10 3.25, p90 8.53) | 2.94 Hz |
| phase transitions deviating from +1 mod 3 | **0 / 59,904** | 0 / 39,732 |
| `alpha=0.01` all-zero fits | **57%** | 60% |
| median per-neuron `alpha_max` | **0.00871** | 0.00708 |
| leg longest:shortest, median | **1.82×** (p90 2.91×, max 25.15×) | 2.26× |
| sessions with leg ratio ≥ 2× | **36%** (≥ 3×: 9%) | — |
| untracked-bin bug | **absent** | 4.1% of bins kept as rows |
| anatomy | **none at all** | `unit_regions.pkl`, 6 groups |

### The structural facts transfer exactly

**0 of 59,904** phase transitions deviate from +1 mod 3, so goal-progress phase is as strict a
0→1→2 cycle here as in LEC. Everything that follows from it holds: only
`num_locations × num_lags` = 108 of the 324 columns are live in any fit, the fitted betas lie on
the `(pref ∓ lag) % 3` stripe, and the prediction is **exactly zero** at every
non-preferred-phase bin. `assert_beta_stripe` runs on every export and confirms it per fold.

### The α problem transfers essentially unchanged

Despite PFC firing more than twice as fast (6.25 Hz median session rate; 3.79 Hz on
`ah04_01122021_02122021`), `alpha=0.01` still zeroes **57%** of neurons, against 60% in LEC.
The rates cancel: `alpha_max = max|Xᵀy| / (n · l1_ratio)` scales as 1/n, and PFC sessions carry
more rows. Measured on `ah04_01122021_02122021` (117 neurons, 6 sessions, 210,446 valid rows):

```
alpha_max per neuron: min 0.00027  p10 0.00183  median 0.00871  p90 0.03342

alpha=0.01     -> all-zero for 67/117 neurons (57%)     # the paper's stated value
alpha=0.001    -> all-zero for  5/117  (4%)
alpha=0.00025  -> all-zero for  0/117  (0%)
```

This is the paper's stated α, on the paper's own data, sitting above the median neuron's entire
regularization path. Note El-Gaby's *executed* default is the Poisson branch (α=1, L2 only,
no sparsification), which never zeroes a coefficient — so the reproduction target and the
"0.01 used in paper" comment are two different paths. Run both.

### The leg-duration confound is milder but present

The state-tuning filter's false-positive rate on constant-rate Poisson cells is ~5% at a 1×
longest:shortest leg ratio, 0.41 at 1.5×, ~0.95 at 2× and ~1.00 at 3×. PFC's **median is
1.82×**, sitting between the 1.5× and 2× rows, with **36% of sessions at or past 2×** and 9%
past 3×. Milder than LEC's 2.26× median, but not clean.

`state_tuning_statistic='mean'` is duration-invariant; both masks are computed on every run
(`state_tuned_mask`, `state_tuned_mask_alt`), and `state_duration_ratio` /
`frac_pref_state_is_shortest` are reported per recday.

### The untracked-bin fix is a no-op here

`drop_untracked_bins` targets LEC's `build_data_dic.locs_to_int`, which maps SLEAP `nan` to
integer **0** so those bins survived the `~isnan(loc)` training filter. PFC `Location_raw`
carries real NaN (median 1.6% of bins, max 8.6%) and an exact-0 fraction of **0**, so
`_prepare_session` drops them either way. One less divergence from the reference.

## Session selection now matches the reference exactly

Two steps, in `PFC_elasticnet_regression_v4.ipynb`:

1. **Dedup by exact task equality.** This *is* El-Gaby's `non_repeat_ses_maker` — both compare
   reward sequences with `np.array_equal`.
2. **`EL_GABY_EXCLUDED_SESSIONS`**, his one hand-exclusion (verified: the only
   `mouse_recday ==` special case in the notebook):

```
me11_05122021_06122021   session 0: [7, 4, 3, 5]
                         session 3: [7, 4, 3, 8]   <- 3 of 4 goals shared
```

His comment is *"task in session 3 was almost identical to session 0 (mistake)"*. Step 1 cannot
catch it — the rows are not exactly equal — so that recday's "held-out task" would not be
genuinely novel. With the exclusion applied, me11 has **6 folds**, matching his hard-coded
`num_non_repeat_ses_found = 6`.

The exclusion is applied in the notebook's selection cell rather than inside the fit, so it
stays visible in `used_sessions`.

## No anatomy

PFC has no `unit_regions` and no `anatomy_split` — `remapping_rotation_analysis.py` states
outright that anatomy is an LEC-only concern. `build_unit_table` detects this and falls back to
identity columns (`recday`, `mouse`, `order` = the `Neuron_raw` row index, `unit_id`), and
`region_summary` / `compare_directions` group by `mouse`. Every regression column is identical
to the LEC path — verified column by column. `table.attrs['has_anatomy']` says which path ran.

`compare_directions` (past vs future) needs no anatomy at all, only a grouping column, so the
prospective-vs-retrospective analysis is unaffected.

## Recdays to watch

Three are too small for per-recday statistics: `me10_20122021_21122021` has **1 neuron**,
`me10_17122021_19122021` has 6, and `ah03_12082021_13082021` / `ah03_18082021_19082021` have
15–16. The cross-dataset comparison
([`../../code/elasticnet_v4_compare.py`](../../code/elasticnet_v4_compare.py)) gates on
`min_neurons=2`, matching `plot_gp_any_tuning_pfc_vs_lec.py`.

## Runtime

1252 units, ~6 folds each. ElasticNet ≈ **20 min per direction** at `n_jobs=6`; Poisson ≈ **2 h
per direction** (1.96 s/fit against 0.29 s).
