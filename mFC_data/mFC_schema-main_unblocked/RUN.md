# Running `mFC_schema-main_unblocked`

A minimally-edited copy of El-Gaby's deposited notebooks, with only the edits required to
make them execute. The deposit is untouched at `../mFC_schema-main/`, and `../data` keeps its original
permissions -- what actually protects the deposit is the mirror plus the repointed
`Input_folder`, verified by inode-independence probes in `_preflight/00_build_mirror.sh`
step [3/4]. That script can optionally `chmod -R a-w ../data` with `LOCK_DEPOSIT=1`, but
it is off by default: `../data` is group-writable in the shared `behrens` group, so
locking it blocks collaborators too.

* `REPRODUCTION.md` — the write-up: what was blocking, what we fabricated and on what
  evidence, results per notebook, and every way our run differs from the deposit.
* `EDITS.md` — the 20 edits, each with the deposited line and why it is unavoidable,
  plus a "deliberately not edited" section. Generated, so it cannot drift.
* `_preflight/` — all new code. Nothing here is an edit to a deposited notebook.

## Layout

| Path | What |
|---|---|
| `../replication_run/data/` | writable mirror of `../data` (full copy, ~52 GB) |
| `../replication_run/data/Intermediate_objects/` | `Input_folder` for every notebook |
| `../replication_run/Output_folder/{ephys,behaviour}/` | `Output_folder` |
| `../replication_run/executed/` | notebooks with our outputs |
| `../replication_run/logs/` | env, mirror, preflight, run logs; `deposit_outputs/` |

## Environment

Conda env `mfc_replication`: python 3.9, **numpy 1.22.0**, scipy 1.10.1, sklearn 1.3.2,
pandas 2.0.3, matplotlib 3.7.3, seaborn 0.13.2, statsmodels 0.14.0, pingouin 0.5.4,
umap-learn 0.5.3, numba 0.56.4. Jupyter kernel: `mfc_replication`.

Pinning numpy is **not** a convenience — it is the only correct fix. The deposit's helpers
build ragged arrays with bare `np.asarray`, which numpy >= 1.24 raises on, but
`concatenate_complex2` is *polymorphic*: the inner call needs `dtype=object` while the
outer call returns per-bin scalars and must stay numeric so `~np.isnan(phases_conc)`
works. A blanket `dtype=object` patch makes `np.isnan` raise `TypeError` and silently
changes `partition`'s dtype on equal-length slices. Verified on this machine:

```
ragged partition            -> object, VisibleDeprecationWarning   (as in his outputs)
equal-length partition      -> int64 (3,10)     <- a dtype=object patch FAILS this
double concatenate_complex2 -> inner object, outer int64, isnan OK
```

Rebuild with `bash _preflight/01_build_env.sh`.

## Run order

```bash
E=/nfs/nhome/live/aharris/.conda/envs/mfc_replication/bin/python

bash _preflight/00_build_mirror.sh          # copy + verify (no lock by default)
$E   _preflight/10_make_bookkeeping.py      # the omitted session bookkeeping
$E   _preflight/20_apply_edits.py           # rebuild the notebooks from the deposit
$E   _preflight/90_audit_diff.py            # prove only declared edits differ

$E _preflight/40_run_notebook.py "Basic_analysis.ipynb"
$E _preflight/40_run_notebook.py "Behavioural Analysis (Figure 1).ipynb"
$E _preflight/40_run_notebook.py "Figure2.ipynb"          # the long pole
$E _preflight/30_bridge_state_aliases.py                  # State_95 / State_99
# Figure3 as deposited is a ~15.5 h job, almost all of it scipy 1.10 pearsonr overhead.
# The declared variant is numerically identical to 4.2e-17 and ~35x faster; build it with
#   $E _preflight/20_apply_edits.py --variant figure3_fast
$E _preflight/40_run_notebook.py "Figure3_fast.ipynb"     # or "Figure3.ipynb" for the slow path
$E _preflight/40_run_notebook.py "Figure5_Regression.ipynb"

# LONG NOTEBOOKS GO THROUGH SLURM, NOT THE SESSION. A detached child dies with the
# interactive cgroup: Figure5_Figure6 was killed at cell 17 that way and lost the whole
# run, because its only export is cell 76 near the end. 50_sbatch_notebook.sh asks for
# 64 GB / 2 CPUs / 24 h with OMP_NUM_THREADS=1 and writes the same run_*.log.
J=$(bash _preflight/50_sbatch_notebook.sh "Figure5_Figure6.ipynb" --skip 9 | tail -1)
bash _preflight/50_sbatch_notebook.sh "Figure7.ipynb" --dependency "afterok:$J" --allow-errors
bash _preflight/50_sbatch_notebook.sh "Figure2_UMAP.ipynb" --allow-errors  # partial by design

$E _preflight/95_checkpoints.py             # score against his stored outputs
```

`40_run_notebook.py` executes cell by cell with progress, stops on the first real
exception, and runs gates in the live kernel between cells (after Figure2 cell 16:
`speed_dic` non-empty; after cell 18: the 3-bin array is genuinely 3-bin and
length-matched to the 5-bin one). `raw` cells are skipped by the kernel, which is how
Figure2 cell 20 stays out of a Run-All.

### Why `--skip 9` on Figure5_Figure6

Cell 9 is a standalone exporter with no consumers inside its own notebook, and the
deposited `Xneuron_correlations` joblib cannot substitute for Figure3 cell 45 anyway: it
carries measures `Max_bins`/`Correlations`/`angle_units` with **no `Angles`**, and covers
3 of 25 recdays. Figure7 cell 59 needs `..._Angles_...`, which only Figure3 cell 45 writes
— so **Figure3 must run, including cell 45** (which El-Gaby never ran: `execution_count`
is `null`).

## Known limits

1. **`Figure2_UMAP` cannot fully run.** Cells 18/19 need
   `Embedding_example_ABCD_08042024_1321.npy`; cells 26/28/33 need
   `ephys_mean_z_shuff_dic` / `ephys_mean_z_shuff_all_alliterations`. Neither was
   deposited nor exists on disk. Blocked by the deposit, not by us — do not fabricate a
   shuffle bank.
2. **Figure7's sleep chain is valid for 21 of 25 ABCDonly recdays.** Cells 20/24/28 index
   `binned_FR_dic_<rd>_<i>` via `np.where(All_sessions==timestamp)[0][0]`, so
   `len(All_session_)` must equal the `binned_FR_dic` file count or every index past a
   missing session is silently shifted. Where it does not, `10_make_bookkeeping.py`
   deliberately publishes *synthetic* tokens so the lookup misses and Figure7's own
   `try/except` skips the session, rather than reading the wrong file. Excluded:
   `ab03_29082023_30082023` (10 sheet rows vs 9 `Task_data`), `me08_10092021_11092021`
   (8 vs 6), `me10_20122021_21122021` (10 vs 6; `Structure` carries an `-ot` one-tone
   suffix and one row has `Ephys` `-`), and `me10_14122021_15122021` — where counts agree
   (8 == 8) but `All`=20 vs `binned_FR`=19, i.e. behaviour without neural data. That last
   is the dangerous one: it would look fine and shift silently.
3. **The `State_95`/`State_99` mapping is an inference**, though an evidenced one — see
   the header of `_preflight/30_bridge_state_aliases.py`. The structural check
   `State_99 subset of State_95` is asserted.
4. **`Location_<rd>_<s>.npy` uses `normalise`'s default mean binning.** Measured 95.1%
   exactly-integer bins / 199 unique values on `me11_05122021_06122021_0`, versus 100% /
   21 unique for `take_max=True`. Mean is preferred because `take_max` would
   systematically prefer edge IDs 10-21 over node IDs 1-9 in any bin spanning both.
   Gated at `frac_integer > 0.90`; `take_max=True` is the sensitivity variant.
5. **`Num_trials_` has a mixed basis.** The deposited `Num_trials_dic2` (joblib, nested
   `[recday][session]`) *does* hold his real per-session trial counts and covers 19 of the
   92 recdays we build, so those are used verbatim; the rest are derived as
   `len(trialtimes_<rd>_<s>.npy)`. `preflight_manifest.json` records
   `num_trials_source` per recday. His counts run 0-3 higher than complete `trialtimes`
   rows (e.g. `ah03_18082021` his `[32,52,54,37]` vs rows `[31,49,52,37]`), so he counted
   trials slightly differently — but only `> 0` is ever consumed, so the magnitude is
   inert. Worth knowing: his `Num_trials[3]=0` for `ah04_05122021_06122021` looks like a
   duplicate-task exclusion (row 3 == row 0 exactly), but it is **redundant** —
   `non_repeat_ses_maker`'s own task-repeat test already drops that row, and the final
   `non_repeat_ses` is `[0,1,2,4,5,6]` on either basis.
6. **Our session selection is a superset of his on a few recdays, harmlessly.** Cell 23
   reports 12 "Files not found for session N" where his stored output reports 5 (all of
   them `me10_14122021_15122021` session 5, printed once per fold). We select one or two
   extra sessions on recdays `Num_trials_dic2` does not cover — sessions that have
   `trialtimes` but no `Neuron_raw`. They fail to load and are skipped, so the *usable*
   fold set is unchanged and both runs report 36 recdays with zero
   "betas not calculated". There is no way to recover his `Num_trials` for those recdays
   from the deposit.
7. **The honest target is internal consistency with the deposited code**, not agreement
   with the printed figures. The anchoring intermediates were never deposited, so this run
   regenerates them from our reading of `Figure2.ipynb`; where they differ from his, every
   downstream number differs and the deposit offers no way to tell which was his.

## Figure2 result (run of 2026-09-08)

29 cells run, 1 skipped (cell 20, by design), 0 failed, 13,734 s. Scored against El-Gaby's
own stored outputs with `_preflight/95_checkpoints.py`:

| deposit cell | quantity | his | ours |
|---|---|---|---|
| 18 | recdays iterated / "not made" | 80 / 4 | **80 / 4** |
| 23 | recdays | 36 | **36** |
| 25 | recdays | 36 | **36** |
| 31 | printed counts | 502, 1214, 1214 | **502, 1214, 1214** |
| 40 / 48 | recdays | 36 | **36** |
| 53 | recdays | 84 | **84** |
| 56 | recdays | 25 | **25** |
| 61 | total neurons `n` | 2182 | **2182** |
| 61 | state-tuned, thr 95 | 1287 | **1287** |
| 61 | state-tuned, thr 99 | 860 | **860** |
| 65 | neurons | 1252 | **1252** |
| 61 | goal-progress, thr 95 | 1825 | 1656 |
| 61 | goal-progress + state, thr 95 | 1162 | 1091 |
| 61 | goal-progress, thr 99 | 1701 | 1459 |

**Every deterministic quantity reproduces exactly.** The only divergences are the
goal-progress (phase) tuning counts, and that is expected rather than a defect:

* `Figure2.ipynb` sets **no RNG seed anywhere** (verified: zero `seed(` / `default_rng` /
  `RandomState` occurrences), and cell 25 draws its 100 circular shifts with
  `shift = random.randrange(max_roll - min_roll) + min_roll`. So
  `GLM_dic2['percentile_neuron_betas']` -- and therefore
  `Tuned_dic2['Phase']`, which cell 56 forms as
  `np.logical_and(percentile > thr, phase_bool_ttest)` -- is **stochastic**. It cannot be
  reproduced exactly without seeding, which would be a code edit beyond the minimal set.
* State tuning, by contrast, is deterministic: cell 48 derives **both**
  `State_zmax_bool` (`State_zmax < 0.05`) and `State_zmax_bool_strict`
  (`State_zmax < 0.01`) from the same real-data p-value matrix, with no shuffle. Hence
  1287 and 860 match to the neuron.

Two smaller, documented divergences:

* **cell 23 skips 12 vs his 5** -- see Known limits 6. Benign; both runs report 36 recdays
  and zero "betas not calculated".
* **cell 46 iterates 36 recdays where his stored output shows 11.** The cell iterates
  `['3_task','combined_ABCDonly']` = 11 + 25 = 36 by construction, so 36 is what the
  deposited code does; his stored output covering only the 11 `3_task` names suggests that
  cell was last run partially in his kernel. Ours is the faithful execution.

Also produced here: 439 `Neuron_<rd>_<s>.npy` and 439 `Location_<rd>_<s>.npy` (`F2-06`)
across 84 recdays, shapes `(n_neurons, n_trials, 360)` and `(n_trials, 360)`; and via the
bridge, 160 `State_95`/`State_99` files over 80 recdays -- 4181 neurons, 2373 state-tuned
at p<0.05 and 1575 at p<0.01, with `State_99 subset of State_95` holding on all 80.
`Phase_<rd>.npy` exists for all 25 `combined_ABCDonly` recdays but not for the 11
`3_task` names (cell 56 only populates `Tuned_dic['Phase']` over `combined_ABCDonly`);
every consumer reads it over `combined_ABCDonly`, so those 11 are unreachable.

## Figure 5 targets

After `UNBLOCK-F5-02/03` (cells 32/38 flipped to `limited=True`, the 12-lag files that
cells 15/21/26 actually write), his stored cell-38 numbers **482 / 278 / 69 are not the
target** — those came from a 24-lag `_beyond` run. Expect the V5 reimplementation band,
**447-481 / 246-287 / 79-92** (`../code/ELASTICNET_V5.md`). The paper's 489 / 329 / 224 is
not a pass criterion; its 90 degrees panel is not reproduced by any definition readable
from the deposited code.
