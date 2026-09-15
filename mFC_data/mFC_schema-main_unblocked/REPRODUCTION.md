# Reproducing El-Gaby's deposited notebooks: what we did, and how our run differs

**Status: live document.** Figure2 is complete; the downstream notebooks are running and
their sections are filled in as they finish. Last updated 2026-09-08.

Companion files: [`EDITS.md`](EDITS.md) is the generated ledger of every code edit;
[`RUN.md`](RUN.md) is the operating manual. This document is the narrative and the
results.

---

## 1. What the problem actually was

The OSF deposit ships the **raw** per-session arrays and the notebooks, and essentially
none of the derived intermediates the notebooks consume. `mFC_schema-main/` therefore
cannot execute as downloaded. The original handoff
(`mFC_schema-main/REPLICATION_STATUS.md`) enumerated five blockers; working through them
turned up a different picture in several places.

Our goal was deliberately narrow: **a minimally-edited copy that executes**, not a
reimplementation. The reimplementation already exists
(`mFC_data/code/`, `code/elasticnet_regression_v5.py`). Every edit is marked in the
notebook, listed in `EDITS.md`, and enforced by an audit that fails if the copy differs
from the deposit in any undeclared way.

**Final tally: 19 edits across 8 notebooks.** 8 are path repoints, 5 fix undefined names,
2 flip a flag, 1 widens a cohort, 1 reorders a dependency, 1 changes a cell type, 1 adds a
new cell — and **exactly one changes a scientific quantity** (`F2-03`).

---

## 2. The environment: why pinning numpy was mandatory, not stylistic

The deposited helpers build ragged arrays with bare `np.asarray`, which numpy ≥ 1.24
raises on. The obvious patch — add `dtype=object` — is **wrong**, and this is the single
most important technical finding of the exercise.

`concatenate_complex2` is **polymorphic**. In the idiom
`concatenate_complex2(concatenate_complex2(phases))` the inner call receives ragged 1-D
arrays and needs `dtype=object`; the outer call receives per-bin **scalars** and must
return a numeric dtype, because the very next operation is `~np.isnan(phases_conc)`.
Measured on numpy 2.0.2:

```
outer, natural            -> int64      np.isnan: OK
outer, forced dtype=object -> object     np.isnan: TypeError: ufunc 'isnan' not supported
partition on equal-length slices: natural (3,10) int64   vs forced (3,10) object
```

So a blanket `dtype=object` would have broken every per-bin mask in Figure2 cells 23/25
and Figure5_Regression cells 15/21/26, and silently changed `partition`'s dtype on
equal-length trial slices. We built conda env `mfc_replication` instead: **python 3.9,
numpy 1.22.0, scipy 1.10.1, scikit-learn 1.3.2, pandas 2.0.3, matplotlib 3.7.3,
seaborn 0.13.2, statsmodels 0.14.0, pingouin 0.5.4, umap-learn 0.5.3, numba 0.56.4** —
the paper's stated stack. That bought **zero code edits** for the largest blocker.

The env is gated by three assertions before any notebook runs, the middle one of which is
what distinguishes a correct environment from a `dtype=object` patch:

```
gate 1  ragged partition -> object, VisibleDeprecationWarning   (matches his stored output)
gate 2  equal-length partition -> int64 (3,10)                  <- a dtype=object patch FAILS
gate 3  double concatenate_complex2 then np.isnan                OK
```

A scan of all 8 notebooks for every numpy 1.20 → 2.0 removal (`np.float`/`int`/`bool`/
`object`/`str`, `alltrue`, `NaN`, `Inf`, `in1d`, `product`, `round_`, `trapz`,
`row_stack`, `mat`) and for `DataFrame.append`, `sns.distplot`, `normed=`, `interp2d`,
`iteritems`, `.ix[]` returned **zero hits**. The entire incompatibility surface was ragged
`np.asarray`.

---

## 3. Corrections to the original handoff

Four claims in `REPLICATION_STATUS.md` did not survive checking. They are recorded because
each one would have sent the next reader down a wrong path.

**(a) §5(a) is wrong — `Figure5_Regression` cell 21 runs as deposited and must not be
"fixed".** The doc reads its single `concatenate_complex2` as leaving one entry per
(trial, state), so the per-bin mask could never match, and calls this "unfixable without
deviating from the deposited code". But the expression is
`np.hstack((np.vstack([...])))` — those are *grouping* parens, not a tuple. `np.hstack`
on the `(n_trials, 4)` object array unpacks rows into a `(4·n_trials,)` object array of
per-bin 1-D arrays, and the single `concatenate_complex2` then descends to per-bin
scalars. Reproduced exactly: 1773/1773 bins, correct trial-major/state-minor order, and it
runs even on numpy 2.0.2 because the final list is scalars, not ragged arrays. Adding a
second concatenation would *break* it. **Blocker E is retired.**

**(b) §2 overstates the path problem.** Each notebook has exactly *one*
`Input_folder`/`Output_folder` assignment pair, not two. Four carried Windows paths
(`Figure2_UMAP`, `Figure3`, `Figure5_Figure6`, `Figure7`); `Figure2`,
`Figure5_Regression` and `Behavioural Analysis` were already local — but their
`Output_folder` lacked a trailing slash, so `plt.savefig(Output_folder+'x.svg')` wrote
`…/ephysx.svg` as a sibling file rather than into the directory.

**(c) §3(i) overstates what the deposited dicts can recover — and understates it in one
place.** Recovering session bookkeeping via `session_dic`/`Variable_dic` resolves only 3
of 25 recdays (they are keyed by *single-day* names), and mice `ab03`/`ah07` — 6 of the
25 recdays, the entire 2023 cohort — are absent from both dicts. **But `Num_trials_dic2`
is genuinely usable**: it is nested `[recday][session]`, holds his real per-session trial
counts, and covers 19 of the 92 recdays we build. A flat lookup makes it look empty, which
is presumably why it was written off.

**(d) Three families the doc credits to `Figure2.ipynb` are written by no notebook at
all**: `Neuron_<rd>_<s>.npy` and `Location_<rd>_<s>.npy` (the 360-bin normalised arrays,
read by six notebooks) and `State_95<rd>.npy`. See §4.

**A blocker the doc missed:** Figure2 cell 16 builds `speed_dic` over
`['combined_ABCDonly']` only, while cell 18 iterates
`['combined_ABCDonly','3_task_all']`. Without widening it, `speed_dic` misses all 55
single-day recdays, `distances = speed_dic[rd][ses]` auto-vivifies, `distances[start:end]`
raises `TypeError: unhashable type: 'slice'`, cell 18's bare `except` swallows it, and
**all six phase/state/time dicts stay silently empty** for that cohort. That is edit
`F2-02`; his own stored cell-18 output iterating 80 recdays with only 4 failures is what
proves he had the wider coverage.

---

## 4. What we had to fabricate, and on what evidence

Nothing here is a guess where the deposit offered an alternative.

### Session bookkeeping (`_preflight/10_make_bookkeeping.py`, 644 arrays, 92 recdays)

The author confirmed by email that `awake_session_*` / `awake_session_behaviour_*` "aren't
needed — it's just to count the sessions which you can get from the other arrays", and
tracing every use site bears that out: Figure2/3/5 consume them only through `len()`.
**Figure7 is the exception** — cells 20/24/28 do
`np.where(All_sessions == timestamp)[0][0]` and use the result to index
`binned_FR_dic_<rd>_<i>`, so those values must be real and mutually consistent.

The rule that makes the reconstruction determinate: every bookkeeping array is indexed by
*`Task_data_` row*, and a combined recday is the contiguous concatenation of its two single
days. **Verified for all 29 combined recdays**, and the prefix split predicts *exactly* the
session indices missing from his own cell-20 stderr (`ah07_27082023_28082023` idx 7,
`me08_12092021_13092021` idx 2, `me10_14122021_15122021` idx 5 and 6). Deposited single-day
`trialtimes_` files are also byte-identical (md5) to the corresponding prefix/suffix slice
of the combined day's, for 24 of 25 recdays.

Timestamps are **real, from the MetaData CSVs**, not placeholders — and the CSVs cover all
mice including `ab03`/`ah07`, which is why this route works where the joblib dicts do not.
Excel had silently coerced some `HH-MM-SS` values into dates; the transform is invertible
(`14/09/1956` → `14-09-56`, i.e. `D/M/YYYY` → `DD-MM-YY`) and was validated against the
sibling `Session_time` column. A strong independent check: the CSV `Structure` column
reproduces `Task_data_` **row for row** (`1-4-5-7` = `[1 4 5 7]`), which is what
establishes that CSV row order *is* `Task_data_` row order.

### The 360-bin orphans (`F2-06`, 439 + 439 files)

`Neuron_<rd>_<s>.npy` and `Location_<rd>_<s>.npy` are read by Figure2 c31, Figure3
c20/23/29/32/36/42/59/80, Figure5_Figure6 c17/22/47/50/89, Figure5_Regression c15,
Figure7 c52/59 and Figure2_UMAP c9 — and written by **nothing**. Several of those reads sit
outside any `try` and hard-crash; others sit inside one and silently yield empty dicts,
which is worse.

They are produced by a new cell inserted immediately before Figure2 cell 31, deliberately
calling **that notebook's own `raw_to_norm`** (cell 6 defines it three times; the last
wins) so the normalisation is identical to cell 46's by construction rather than by
argument. The recipe follows `Basic_analysis.ipynb` cell 21, the author's own worked
example. Output shapes `(n_neurons, n_trials, 360)` and `(n_trials, 360)`, verified against
every consumer's reshape and against cell 31's `np.split(...,10)` and `np.split(...,3)`.

One judgement call is recorded rather than buried: `Location_` uses `normalise`'s default
**mean** binning, giving 95.1 % exactly-integer bins and 199 unique values on
`me11_05122021_06122021_0`, versus 100 % / 21 unique for `take_max=True`. Mean is preferred
because `take_max` would systematically prefer edge IDs 10–21 over node IDs 1–9 in any bin
spanning both — a real bias — whereas mean leaves pure-node bins intact and makes mixed
bins fractional, hence excluded by the `==` comparisons downstream. Gated at
`frac_integer > 0.90`; `take_max=True` is the sensitivity variant.

### `State_95` / `State_99` (`_preflight/30_bridge_state_aliases.py`, 160 files)

Consumers build the name as `'State_' + tuning_percentile + recday`, with **no separator**
(`State_95ah04_01122021_02122021.npy`), which is a different file from the
`State_ah04_...npy` that cell 63 writes. The mapping is evidenced, not guessed: cell 56
lines 82–85 set `Tuned_dic2['State']['95'] = Tuned_dic['State_zmax_bool']` and
`['99'] = Tuned_dic['State_zmax_bool_strict']`, and cell 63 writes `State_` and
`State_strict_` from exactly those. The structural invariant `State_99 ⊆ State_95` (both
thresholds of one p-matrix) is asserted and holds on all 80 recdays.

### Data safety

`Intermediate_objects/` is 218 unique objects plus **3841 hardlinks** into the six
type-subfolders, and every notebook writes with `np.save(Input_folder + ...)`. Figure2 cell
20 saves `Neuron_raw_`/`Location_raw_`/`XY_raw_`/`trialtimes_` under *single-day* names —
names hardlinked to the deposited raw arrays — and it does not merely overwrite them, it
**renumbers** them (`me08_12092021` has session indices `[0,1,3]`; cell 20 would write
contiguous `0,1,2`). We therefore work from a full 52 GB physical copy, verified
inode-independent by probe, and cell 20 is converted to a `raw` cell (`F2-04`) so Run-All
skips it while its source stays readable. The deposit's own permissions are untouched, and
a 200-file md5 sample taken before the run still verifies afterwards.

---

## 5. Results, and every way our run differs

### Figure2 — complete (29 cells, 1 skipped by design, 0 failed, 13,734 s)

Scored against El-Gaby's **own stored cell outputs**, which are the right yardstick here:
the anchoring intermediates were never deposited, so agreement with *his numbers* is
evidence our regenerated inputs match his, in a way agreement with the printed figures
cannot be.

| deposit cell | quantity | his | ours | |
|---|---|---|---|---|
| 18 | recdays / "not made" | 80 / 4 | 80 / 4 | ✅ |
| 23 | recdays | 36 | 36 | ✅ |
| 25 | recdays | 36 | 36 | ✅ |
| 31 | printed counts | 502, 1214, 1214 | 502, 1214, 1214 | ✅ |
| 40, 48 | recdays | 36 | 36 | ✅ |
| 53 | recdays | 84 | 84 | ✅ |
| 56 | recdays | 25 | 25 | ✅ |
| 61 | total neurons `n` | 2182 | 2182 | ✅ |
| 61 | state-tuned, p<0.05 | 1287 | 1287 | ✅ |
| 61 | state-tuned, p<0.01 | 860 | 860 | ✅ |
| 65 | neurons | 1252 | 1252 | ✅ |
| 61 | goal-progress, p<0.05 | 1825 | 1656 | ⚠ stochastic |
| 61 | goal-progress + state | 1162 | 1091 | ⚠ stochastic |
| 61 | goal-progress, p<0.01 | 1701 | 1459 | ⚠ stochastic |

**Every deterministic quantity reproduces exactly.** The three divergences are all
goal-progress tuning, and the mechanism is verified rather than assumed:

* `Figure2.ipynb` sets **no RNG seed anywhere** — zero occurrences of `seed(`,
  `default_rng`, `RandomState` — and cell 25 draws its 100 circular shifts with
  `shift = random.randrange(max_roll - min_roll) + min_roll`. So
  `GLM_dic2['percentile_neuron_betas']`, and hence `Tuned_dic2['Phase']` which cell 56
  forms as `np.logical_and(percentile > thr, phase_bool_ttest)`, is **stochastic**. It
  cannot be reproduced exactly without seeding, and seeding would be a code edit beyond
  the minimal set.
* State tuning is **deterministic** by contrast: cell 48 derives *both*
  `State_zmax_bool` (`State_zmax < 0.05`) and `State_zmax_bool_strict`
  (`State_zmax < 0.01`) from the same real-data p-value matrix, with no shuffle. Hence
  1287 and 860 agree to the neuron.

That split — deterministic exact, stochastic close — is itself the strongest available
evidence that the regenerated inputs are right.

Two further differences, both benign and both documented:

* **cell 23 reports 12 missing-file skips where his reports 5.** His 5 are all one session
  (`me10_14122021_15122021` session 5, printed once per fold). Our `non_repeat_ses` is a
  superset of his on recdays `Num_trials_dic2` does not cover: we select one or two extra
  sessions that have `trialtimes` but no `Neuron_raw`. They fail to load and are skipped,
  so the *usable* fold set is unchanged — both runs report 36 recdays and zero "betas not
  calculated". His `Num_trials` for those recdays is not recoverable from the deposit.
* **cell 46 iterates 36 recdays where his stored output shows 11.** The cell iterates
  `['3_task','combined_ABCDonly']` = 11 + 25 = 36 by construction, so 36 is what the
  deposited code does. His output covering only the 11 `3_task` names indicates that cell
  was last run partially in his kernel. Ours is the faithful execution.

A third, cosmetic: `Phase_<rd>.npy` exists for all 25 `combined_ABCDonly` recdays but not
for the 11 `3_task` names, because cell 56 populates `Tuned_dic['Phase']` over
`combined_ABCDonly` only. Every consumer reads it over `combined_ABCDonly`, so those 11
are unreachable.

### `Basic_analysis` — complete, exact

11 cells, 0 failed. All data values identical (`(46, 46178)`, `(46179,)`). The only
textual differences are object memory addresses and a `shape=(N,)` in the archived array
repr — a numpy ≥ 2.2 feature, which incidentally shows *those* archived outputs came from
a recent run on this machine rather than from his numpy 1.22 kernel. Needed only the path
repoint (`BA-01`).

### `Behavioural Analysis (Figure 1)` — complete, exact

6 cells, 0 failed. The 124-number behavioural scoring matrix in cell 16 is **byte-identical**
to the stored output. Needed only the path repoint (`BH-01`).

### `Figure3` — exports complete; final three cells blocked by a deposit defect

Run as the `Figure3_fast.ipynb` variant (see below). **Both of Figure3's file outputs are
complete**, which is what the rest of the pipeline needs:

| output | cell | status |
|---|---|---|
| `Xneuron_correlations_<day_type>_<measure>_<rd>.npy` | 45 | **182 files** — `3_task` 11, `3_task_all` 55, `combined_ABCDonly` 25, x2 measures. All 25 `combined_ABCDonly_Angles_*` present, shape (63, 63, 7), 100 % finite |
| `sigma_goalprogress.npy` | 54 | **written**, value 11.508 |

Cell 45 is the sole producer of the `..._Angles_*` files Figure7 cell 59 needs, and
**El-Gaby never ran it** (`execution_count: null`). Cell 54 only reached its `np.save`
because of edit `F3-02`; the deposited version raises `NameError` on `mean_hist_fine` four
lines earlier.

**Cells 86, 87 and 89 cannot produce meaningful output, and this is a defect in the
deposit rather than a consequence of our reconstruction.** Cell 80 line 42 reads

```python
ephys_=np.load(Input_folder+'Neuron_'+mouse_recday+'_'+str(ses_ind)+'.npy')
```

where **`ses_ind` is not defined in cell 80** — it leaks from cell 59's loop, so it holds
whatever value that loop last left. Cell 80's bare `except Exception as e: print(e)` then
swallows the consequences: it ran 750 s, printed all 55 recdays, hit `FileNotFoundError`
on 3 of them, and populated `module_shuff_dic` for none. That cascades to cell 86
("attempt to get argmin of an empty sequence"), cell 87
(`array=defaultdict(...)`), and cell 89, which dies on
`unsupported operand type(s) for +: 'collections.defaultdict' and 'float'`.

We did **not** fix it. The correct `ses_ind` is unknowable from the deposit — cell 80 gives
no indication whether it meant session 0, a loop over sessions, or the last session — so any
choice would be fabrication rather than reconstruction. All three affected cells are
plot-only (2 SVGs and a Wilcoxon print), so nothing downstream is affected.

Cell timings, for reference: cell 29 **1603 s**, cell 36 **2990 s**, cell 80 750 s. Under
the deposited `st.pearsonr` cell 29 alone ran 4h48m without finishing.

### `Figure3_fast` — the declared performance variant

`Figure3.ipynb` as deposited is a ~15.5 h job, essentially all of it scipy 1.10's
`st.pearsonr` overhead: 1.08 ms per call at length 360, and **both** of its call sites take
`[0]`, discarding the p-value and confidence interval that dominate the cost. Cell 29 uses
`angle_units=2` (180 angles) and cell 36 `angle_units=10` (36 angles), over 134,424 and
962,931 neuron/session combinations respectively.

`Figure3_fast.ipynb` replaces both call sites with a centred dot product
(`VARIANT-F3F-00/01/02`), agreeing with `st.pearsonr(a,b)[0]` to **4.2e-17** over 200 random
pairs and matching scipy's edge cases (NaN input -> NaN; zero variance -> NaN). Measured
27.4 us vs 950.6 us, a 35x speedup. It is a **variant, not part of the audited minimal
copy**. The observed gain is smaller than 35x because only the `pearsonr` fraction speeds
up: cell 29 went from >4h48m (unfinished) to 1603 s.

### `Figure5_Figure6` — queued

Will run with `--skip 9`. Cell 9 is a standalone exporter with no consumers in its own
notebook, and the deposited `Xneuron_correlations` joblib cannot substitute for Figure3
cell 45 anyway: it carries measures `Max_bins`/`Correlations`/`angle_units` with **no
`Angles`**, and covers 3 of 25 recdays.

*Results to follow.*

### `Figure5_Regression` — running (the priority notebook)

Three edits: the path repoint plus `limited=False → True` in cells 32 and 38. As
deposited, cells 15/21/26 write the 12-lag no-suffix files while 32/38 read the 24-lag
`_beyond` files — so the plotted histogram came from a run the earlier cells never
produce. 12 lags is what the Methods specify. **Cell 21 runs unedited**, per correction (a) above — and it completed across all 25
recdays in 6309 s, which is the definitive refutation of the handoff's claim that it was
"unfixable without deviating from the deposited code".

#### Cell 15 complete (558 s) — and it validates `F2-03` directly

```
GLM_anchoring_prep_dic_regressors: 25/25 recdays written
  regressor columns (324,)  ->  25 recdays, folds [4, 5, 6, 7]
ALL 324 COLUMNS: True         (= num_nodes 9 x num_task_phases 3 x num_lags 12)
GLM_anchoring_prep_dic_Location: 25/25
GLM_anchoring_prep_dic_Neuron:   25/25

Phases_raw2_<rd>_0 distinct values: [0, 1, 2]        <- 3-bin, consumed here
Phases_raw_<rd>_0  distinct values: [0, 1, 2, 3, 4]  <- 5-bin, consumed by Figure2 c23/c25/c53
```

This is the concrete payoff of the single scientific edit. In the deposit, cell 18 assigned
the **same list object** to both `Phases_raw_dic` and `Phases_raw_dic2`, so `Phases_raw2_`
was a 5-bin array and `num_phases2 = 3` was dead code.

#### What a 5-bin `Phases_raw2_` would actually have done

Worth being precise, because the loose version of this claim ("phase bins 3 and 4 would be
structurally unreachable, producing dead regressor columns") is **wrong**, and was
corrected after being challenged.

Cell 15 allocates `module_anchor_progress = np.zeros((9, num_task_phases=3, num_lags=12))`
regardless of what the file contains. `Task_phaset = int(phases[t])` can then be 0–4. The
*input* condition `Task_phaset == Task_phase_` never fires for phases 3/4 — but
`move_phase`, and hence the bump `np.roll`, fires on **every** phase change. Cell 21 then
fits only on `phases_conc_nonan == pref_phase`, where `pref_phase` is an argmax over three
phases, so phase-3/4 bins are simply **excluded from the fit**. Nothing crashes, and a
faithful transcription of the cell-15 loop on synthetic 3-bin and 5-bin input gives:

```
3-bin phase array: 324/324 non-zero regressor columns
5-bin phase array: 324/324 non-zero regressor columns
```

**No regressor column is dead either way.** The real structural difference is in the
(anchor_phase, lag) cells reachable *within* one neuron's fit:

| loaded array | bins used in fit | reachable (ap, lag) cells | lags covered |
|---|---|---|---|
| 3-bin | 32.3 % | **12 / 36** | all 12 |
| 5-bin | 18.7 % | **7–8 / 36** | 7 of 12 |

3-bin yields a clean mod-3 stripe — for `pref_phase = 0`: `ap0` at lags {0,3,6,9}, `ap1` at
{2,5,8,11}, `ap2` at {1,4,7,10}, every lag reachable by exactly one anchor phase. 5-bin
yields a mod-**5** pattern truncated to `ap ∈ {0,1,2}`: `ap0` at {0,5,10}, `ap1` at {4,9},
`ap2` at {3,8} — leaving lags 1,2,6,7,11 unreachable by any anchor phase. Intersecting the
3-bin stripe with the 5-bin reachable set leaves only `(ap0, lag0)`.

That difference in reachable cells is what the Fig 5g raster argument keys on; see
`../../code/ELGABY_FIGURE5_RECONCILIATION.md` §0.1 for that analysis. **We did not verify
it against the raster ourselves** and do not assert its cell counts here.

#### Confirmed on the real fitted coefficients

The prediction above is borne out exactly by cell 21's actual output, which is stronger
evidence than the raster appeal because it comes from this run. Taking
`Poisson_GLM_anchoring_coeffs_all_ab03_01092023_02092023.npy` (63 neurons × 6 folds ×
324), reshaping each fit to `(9 locations, 3 anchor_phases, 12 lags)` and reading each
neuron's preferred phase from `tuning_phase_boolean_max_`:

```
non-zero (anchor_phase, lag) cells ON  the stripe ap == (pref_phase - lag) % 3 : 4536
non-zero cells OFF the stripe                                                  :    0
stripe purity                                                                  : 100.0000%
overall non-zero coefficient fraction                                          : 0.3333  (= 12/36)
```

Every non-zero coefficient lies on the mod-3 stripe, and exactly one third of the grid is
live — precisely the 3-bin prediction. Under a 5-bin array the fraction would be ≈ 0.21
with 5 of the 12 lags dead for every neuron. So the two hypotheses are empirically
distinguishable, though by the *shape and size of the reachable set* rather than by any
column being empty.

#### The argument for 3 bins that needs no external evidence

The deposited code is internally 3-bin throughout, which settles the intent on its own:

* cells 15, 21 and 26 all hardcode `num_task_phases = 3`;
* `num_lags = num_task_states × num_task_phases = 4 × 3 = 12` equals **exactly one
  complete ABCD loop** only if there are 3 phases per state. Under 5-bin, 12 lags spans
  12/5 = 2.4 states and the lag axis stops being a loop at all;
* cell 26 builds `phase_norm_mean` by tiling `np.arange(3)` across the 90-bin normalised
  curve, independently of anything read from disk;
* cell 18 defines `num_phases2 = 3` for precisely the array it then fails to use.

So the 5-bin `Phases_raw2_` is a bug in the deposit, not a design choice, and `F2-03`
supplies the array the rest of the code already assumes.

**Consequence for scoring cell 38:** his stored numbers **482 / 278 / 69 are not our
target**, because they came from a `_beyond` (24-lag) run. Expect the V5 reimplementation
band, **447–481 / 246–287 / 79–92** (`../code/ELASTICNET_V5.md`). The paper's
489 / 329 / 224 is not a pass criterion — its 90° panel is not reproduced by any definition
readable from the deposited code.

#### Cell 21 complete (6309 s) — runs unedited, 25/25 recdays

```
Poisson_GLM_anchoring_coeffs_all_: 25/25 recdays   (GLM_anchoring_coeffs_all_: 0, since
                                                    Poisson_regression=True)
total neurons across 25 recdays : 1252     <- his cell 32 reports 1252  (exact match)
regressor columns               : 324      finite 100.0%
fold counts                     : {4: 1 recday, 5: 2, 6: 20, 7: 2}
me11_05122021_06122021          : (46, 7, 324)
```

`me11_05122021_06122021` reproduces his fold structure exactly — **7 folds and 46
neurons**, matching his cell-21 stdout and identifying precisely the 46 neurons affected by
the cell-26 hardcode `num_non_repeat_ses_found = 6`, which we deliberately left in place
(see "deliberately not edited" in `EDITS.md`).

#### Cells 26, 32, 38 complete -- and the ElasticNet comparison

The full notebook: 10 cells, 0 failed, 9926 s. The `Figure5_Regression_elasticnet.ipynb`
variant (`Poisson_regression=False` -> `ElasticNet(alpha=0.01, positive=True)`) then ran in
7592 s. Both write to distinct filename prefixes, so the two result sets coexist.

| panel | Poisson | **ElasticNet** | V5 (ElasticNet a=0.01) | paper | his stored (`_beyond`) |
|---|---|---|---|---|---|
| all state-tuned | 581, r=+0.272, t=17.15 | **623, r=+0.203, t=15.56** | 447-481, t~14.5 | 489, t=9.3 | 482, t=14.47 |
| non-zero-lag 30 deg | 296, r=+0.174, t=6.76 | **307, r=+0.073, t=2.87** | 246-287, t~2.48 | 329, t=3.9 | 278, t=8.41 |
| non-zero-lag 90 deg | 79, r=+0.160, t=2.67 | **86, r=+0.119, t=2.13** | 79-92, t<=1.8 | 224, t=2.53 | 69, t=3.77 |

Both runs: 25/25 recdays, 1252 neurons, 738 state-tuned by `State_95` -- all three exact
matches to his own figures, on either model.

**The model choice accounts for most of the gap to V5.** Switching from
`PoissonRegressor(alpha=1)` to `ElasticNet(alpha=0.01)`:

* panel 1 t: 17.15 -> **15.56** against V5's ~14.5;
* panel 2 t: **6.76 -> 2.87** against V5's ~2.48, i.e. the Poisson t was inflated ~2.4x;
* panel 3 n: 79 -> **86**, now inside V5's 79-92 band.

Panel 2's mean r collapses from +0.174 to +0.073, which is the signature of the readout
mismatch: cell 26 scores predictions as `np.sum(regressors_ses * coeffs_ses_neuron, axis=1)`
-- **linear, no `exp`, no intercept** -- so a Poisson fit was being read out through a linear
predictor and the correlations were inflated. Under ElasticNet the readout is the right one
for the model, and the effect sizes land in V5's range.

What remains is a **neuron-count** difference (623 vs V5's ~465 in panel 1), i.e. which
neurons survive the finite-correlation filter, not how strong the effect is. That is the
sharp question for `../HANDOFF_V5_VS_ELGABY.md`.

**On the 90 deg panel (ED Fig 8a).** `ELASTICNET_V5.md` treats it as unreproduced (79-92
neurons at t <= 1.8). Running his own code gives a *significant* result on both models --
t = 2.67 (Poisson) and t = 2.13 (ElasticNet), against the paper's t = 2.53 -- on 79-86
neurons rather than the published 224. So the panel reproduces in direction and
significance while resting on roughly a third of the reported neurons.

Panel statistics are extracted to `../replication_run/logs/figure5_stats.{json,csv}` by
`_preflight/96_extract_figure5_stats.py`, which recomputes them from the saved per-neuron
arrays (cell 38 only prints them). It reproduces cell 38's printed values exactly.

### `Figure7` — queued

Needs one real code edit (`F7-02`): cell 76 iterates
`np.arange(num_phase_place_diffs_)`, a name undefined anywhere in the repo, at module
level above the cell's own `try:` — a fatal `NameError`. Cell 76 is the sole producer of
`cross_corr_dic`, which cells 82/87 and thence 86/89 consume. The value 5 is recovered
from **his own stored cell-76 output**, which prints `phase_place_diff` = 0,1,2,3,4.

**Its sleep chain is valid for 21 of 25 recdays.** Cells 20/24/28 index
`binned_FR_dic_<rd>_<i>` through `np.where(All_sessions == timestamp)[0][0]`, so
`len(All_session_)` must equal the `binned_FR_dic` file count or every index past a missing
session is silently shifted. Where it does not, we deliberately publish *synthetic* tokens
so the lookup misses and Figure7's own `try/except` skips the session rather than reading
the wrong file. Excluded: `ab03_29082023_30082023` (10 sheet rows vs 9 `Task_data`),
`me08_10092021_11092021` (8 vs 6), `me10_20122021_21122021` (10 vs 6; `Structure` carries
an `-ot` one-tone suffix and one row has `Ephys` `-`), and `me10_14122021_15122021` —
where the counts agree (8 == 8) but `All` = 20 against `binned_FR` = 19, i.e. behaviour
without neural data. **That last one is the dangerous case**: it looks fine and would have
shifted silently.

*Results to follow.*

### `Figure2_UMAP` — queued, partial by construction

**Cannot run end to end, and this is a property of the deposit.** Cells 18/19 need
`Embedding_example_ABCD_08042024_1321.npy`; cells 26/28/33 need `ephys_mean_z_shuff_dic` /
`ephys_mean_z_shuff_all_alliterations`, a kernel-resident shuffle bank. None of these was
deposited and none exists anywhere on disk. We run everything that does not depend on
them and report the rest as blocked-by-deposit. We did not fabricate a shuffle bank.

---

## 6. Gotchas worth carrying forward

Each of these cost real time and none is obvious from reading the code.

1. **The deposit's bare `except` blocks convert missing inputs into silent wrong answers.**
   Figure2 cell 23's `Distances_from_reward_` load sits at the *outer* per-recday `try`,
   so a `FileNotFoundError` prints "betas not calculated" 36 times, leaves
   `GLM_dic2['mean_neuron_betas']` empty, lets cell 25 spend 4 hours on fits it cannot
   use, and only surfaces 5 hours later at cell 56 as
   `TypeError: unhashable type: 'slice'`. **That load appears twice** in cell 23 — training
   *and* held-out — and both must be redirected to the in-memory dict; fixing only one
   reproduces the failure exactly. This is why the runner now asserts on live kernel state
   between cells rather than trusting exit codes.
2. **`F2-06` shifts every later cell index by +1** relative to the deposit. Anything keyed
   by cell number — execution gates, checkpoint tables — must account for it. Gates are now
   keyed by a source substring instead, and the checkpoint comparator maps deposit index →
   executed index explicitly.
3. **Cell 20 renumbers as well as overwrites**, so "it would just regenerate them
   identically" is false.
4. **`np.hstack((x))` is grouping parens, not a tuple.** Misreading this is what produced
   the incorrect "cell 21 cannot run" conclusion.
5. **Vacuous PASS.** A check that makes zero comparisons must not report success — the
   bridge script now prints `NOT RUN (0 comparisons)` and exits non-zero if it wrote
   nothing, so running it before Figure2 cannot look like a success.
6. **Not every archived output is his.** Some carry numpy ≥ 2.2 reprs, i.e. they came from
   recent local runs. The ones that matter for the Figure 5 chain do carry his
   `Python38` / `C:\Users\moham` provenance in stderr.

---

## 7. The honest bottom line

The sensible target is **internal consistency with the deposited code**, not agreement
with the printed figures. The anchoring intermediates were never deposited, so this run
regenerates them from our reading of `Figure2.ipynb`. Where they differ from his, every
downstream number differs, and the deposit contains no way to tell which was his.

Against that target the result so far is good: every deterministic checkpoint in Figure2
matches exactly, including the 2182 / 1287 / 860 / 1252 neuron counts, and the only
divergences trace to an unseeded permutation test in his own code.

## 2026-09-15 — Prospective lags through his own code (VARIANT-PRO, `Figure5_Regression_prospective.ipynb`)

Same variant as the LEC copy (cell 15 on the time-reversed session, `_prospective` suffix); job
3594658, 3.4 h, EXIT=0; sign check vs V5 `future` regressors: max|diff| = 0 on all 6 sessions of
ab03_01092023. Prep arrays deleted after cell 26. Poisson (alpha = 1), State_95 pool, his cell-26
semantics, 25 recdays / 1252 units:

| panel | retrospective | prospective |
|---|---|---|
| all state-tuned | n 581, r +0.272, t 17.1 | n 537, r +0.271, t 15.1 |
| non-zero-lag, 30° excluded | n 296, r +0.174, t 6.8 | n 253, r +0.110, t 3.4 |
| non-zero-lag, 90° excluded | n 79, r +0.160, t 2.67 | n 50, r -0.022, t -0.26 |

Retrospective > prospective at non-zero lags, as in LEC and as in V5 specs 2 vs 5.

**Figure-file collision, found and fixed 2026-09-15 10:40.** Cell 38 names its SVGs
`Output_folder + addition + 'GLM_analysis_<name>.svg'` with `addition = 'Poisson_'` only, so the
prospective run's cell 38 overwrote the three retrospective `Poisson_GLM_analysis_*.svg` (the
`.npy` results and `logs/figure5_stats.{json,csv}` were always namespaced and unaffected).
Fix: VARIANT-PRO-10 puts `addition2` into the figure name
(`_prospectivePoisson_GLM_analysis_*.svg`); the retrospective figures were regenerated from
the intact retrospective correlation files by re-running cells 32/38 of the deposited notebook
(`executed/Figure5_Regression_figs_only.ipynb`), and the prospective ones by the same cells of
the rebuilt variant (`executed/Figure5_Regression_prospective_figs_only.ipynb`). The full-run
executed notebooks are unchanged records.
