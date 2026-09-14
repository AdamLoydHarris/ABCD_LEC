# `mFC_schema-main_unblocked` -- the edit ledger

Every edit applied to El-Gaby's deposited notebooks, and why each is strictly
required to make the code execute. The deposit itself is untouched and lives at
`../mFC_schema-main/`.

This file is **generated** by `_preflight/20_apply_edits.py --ledger` from the same
edit table the patcher applies, so it cannot drift out of sync with the notebooks.
`_preflight/90_audit_diff.py` asserts a bijection between the edit IDs appearing in
the diff and the IDs listed here.

## Conventions

* Every changed or added source line carries a trailing `# UNBLOCK-<ID>` token.
* Every edit is preceded by one `# UNBLOCK-<ID> (<CLASS>): ...` comment in the notebook.
* Classes: `PATH`, `NAMEDEF` (undefined name), `ORDER` (dependency/definition order),
  `COVERAGE` (cohort coverage), `SCIENCE` (changes a computed quantity), `FLAG`,
  `SKIP`, `NEWCELL`.
* `G-01` (global, all 8 notebooks): `outputs` cleared and `execution_count` set to
  `null`, so the audit only ever compares source. The deposit's stored outputs are
  archived first, to `replication_run/logs/deposit_outputs/` -- they are the numeric
  reference for verification.

## The 20 edits

| ID | Notebook | Where | Class | Why strictly necessary |
|---|---|---|---|---|
| `UNBLOCK-BA-01` | Basic_analysis.ipynb | cell 1 L0 | `PATH` | repoint at the writable mirror. Documentation-only notebook; otherwise runs as deposited (its partition() already omits the outer np.asarray) |
| `UNBLOCK-BH-01` | Behavioural Analysis (Figure 1).ipynb | cell 3 L2, cell 3 L3 | `PATH` | repoint at the writable mirror so the deposit is never written to (already runs end to end; repoint only) **/** repoint + add the TRAILING SLASH: every save is string concatenation (Output_folder+'x.svg'), so without it figures land as a sibling file |
| `UNBLOCK-F2-01` | Figure2.ipynb | cell 1 L2, cell 1 L3 | `PATH` | repoint at the writable mirror so the deposit is never written to (Figure2) **/** repoint + add the TRAILING SLASH: every save is string concatenation (Output_folder+'x.svg'), so without it figures land as a sibling file |
| `UNBLOCK-F2-02` | Figure2.ipynb | cell 16 L4 | `COVERAGE` | cell 16 builds speed_dic over combined recdays ONLY, but cell 18 iterates ['combined_ABCDonly','3_task_all']. Without '3_task_all' here, speed_dic misses all 55 single-day recdays, so cell 18's distances=speed_dic[rd][ses] autovivifies an empty defaultdict, distances[start:end] raises TypeError: unhashable type: 'slice', cell 18's except swallows it, and ALL SIX phase/state/time dicts stay silently empty for 3_task_all -- cascading into cells 23/25/31/40/46/48/50/53. His stored cell-18 output iterates 80 recdays (25 combined_ABCDonly + 55 3_task_all) with only 4 'not made', proving his speed_dic covered the single days. With this edit our run reports 'speed_dic covers 80 recdays, 412 sessions' and cell 18 populates 80 recdays, matching him |
| `UNBLOCK-F2-03` | Figure2.ipynb | cell 18 L85, cell 18 L80, cell 18 L74, cell 18 L63, cell 18 L57, cell 18 L44, cell 18 L38 | `SCIENCE` | store the genuine 3-bin array. The deposit assigned the SAME list object (phases_all) to both dicts, making Phases_raw2_ and Phases_raw_ byte-identical 5-bin arrays and leaving num_phases2=3 dead **/** accumulate the 3-bin array per trial **/** accumulate the 3-bin array per state **/** apply the SAME residual-rounding extra bin to the 3-bin array. This must sit inside the existing else and residual_cum must be decremented only once, or the two arrays drift in length and stop aligning bin-for-bin with Location_raw **/** build the 3-bin twin of the 5-bin phase array, using the num_phases2=3 the author already defined on line 5 and never referenced. Figure5_Regression needs 3 bins (num_task_phases=3, 9x3x12=324 regressors) while Figure2 own GLM (c23/c25) and phase-map c53 use the 5-bin Phases_raw_ **/** per-trial accumulator for the 3-bin array **/** per-session accumulator for the 3-bin array |
| `UNBLOCK-F2-04` | Figure2.ipynb | cell 20 | `SKIP` | make Run-All safe. Cell 20 splits combined recdays into single days and np.save()s Neuron_raw_/Location_raw_/XY_raw_/trialtimes_ under single-day names. Its outputs already exist on disk, it reads Distances_from_reward_ (written only by cell 79) so it is a no-op on a fresh folder, and it does not merely overwrite -- it RENUMBERS (deposited me08_12092021 has session indices [0,1,3]; this would write contiguous 0,1,2). Source is untouched; only cell_type changes, so the code remains readable in place |
| `UNBLOCK-F2-05` | Figure2.ipynb | cell 23 L87, cell 23 L88, cell 23 L141, cell 23 L142 | `ORDER` | break the cell-23 <-> cell-79 circularity. Distances_from_reward_<rd>_<s>.npy is written ONLY by cell 79, the last cell, and this load sits at the OUTER per-recday try so a miss aborts the whole recday with "betas not calculated". Cell 25 lines 80/133 are the author's own verbatim in-memory form of exactly this value, so the substitution is provably equivalent **/** second line of the folded np.load call removed with it **/** same substitution for the TEST session load; cell 25 line 133 is the author's own in-memory form of exactly this value **/** second line of that folded np.load call removed with it |
| `UNBLOCK-F2-06` | Figure2.ipynb | cell 31 (new cell before) | `NEWCELL` | produce the 360-bin Neuron_/Location_ arrays that six notebooks read and no deposited notebook writes |
| `UNBLOCK-F2-07` | Figure2.ipynb | cell 46 L2 | `NAMEDEF` | uncomment. Tuned_dic is used in cells 46/48/56/60/63/72/79. Cells 56, 63 and 79 have no try/except so it is a fatal NameError there; cells 46/48/60/72 swallow it and SILENTLY DISCARD ALL THEIR WORK (cell 46 prints "Not found" per recday). Without this nothing is persisted at all |
| `UNBLOCK-F2-08` | Figure2.ipynb | cell 56 L10 | `NAMEDEF` | define use_both before its first use in cell 56 |
| `UNBLOCK-UM-01` | Figure2_UMAP.ipynb | cell 1 L2, cell 1 L3 | `PATH` | repoint at the writable mirror so the deposit is never written to (Figure2_UMAP -- was a Windows path) **/** repoint + add the TRAILING SLASH: every save is string concatenation (Output_folder+'x.svg'), so without it figures land as a sibling file |
| `UNBLOCK-F3-01` | Figure3.ipynb | cell 2 L2, cell 2 L3 | `PATH` | repoint at the writable mirror so the deposit is never written to (Figure3 -- was a Windows path) **/** repoint + add the TRAILING SLASH: every save is string concatenation (Output_folder+'x.svg'), so without it figures land as a sibling file |
| `UNBLOCK-F3-03` | Figure3.ipynb | cell 36 L47 | `ORDER` | guard the Neuron_ load, which is the only unguarded one in this notebook. Cell 36 iterates every session in awake_session_behaviour_ and gates only on num_trials_day[ses_ind]==0, but a session can have trialtimes (hence num_trials>0) and no Neuron_raw -- me10_14122021_15122021 session 5 is exactly that, so no 360-bin Neuron_ array can exist for it and the load dies with FileNotFoundError, aborting the notebook. Cell 29 wraps the IDENTICAL load in try/except and continues, and cell 36 already uses a continue-on-empty idiom two lines below, so this matches the notebook's own convention. Without it Figure3 cannot reach cell 45, which is the sole producer of the Xneuron_correlations_*_Angles_* files Figure7 needs |
| `UNBLOCK-F3-02` | Figure3.ipynb | cell 54 L91 | `ORDER` | unblock the sigma_goalprogress.npy write that cell 80 depends on |
| `UNBLOCK-F6-01` | Figure5_Figure6.ipynb | cell 2 L2, cell 2 L3 | `PATH` | repoint at the writable mirror so the deposit is never written to (Figure5_Figure6 -- was a Windows path) **/** repoint + add the TRAILING SLASH: every save is string concatenation (Output_folder+'x.svg'), so without it figures land as a sibling file |
| `UNBLOCK-F5-01` | Figure5_Regression.ipynb | cell 1 L4, cell 1 L5 | `PATH` | repoint at the writable mirror so the deposit is never written to (Figure5_Regression) **/** repoint + add the TRAILING SLASH: every save is string concatenation (Output_folder+'x.svg'), so without it figures land as a sibling file |
| `UNBLOCK-F5-02` | Figure5_Regression.ipynb | cell 32 L6 | `FLAG` | cells 15/21/26 write the 12-lag no-suffix files but cells 32/38 read the 24-lag _beyond files, so as deposited the plotted histogram comes from a run the earlier cells never produce. 12 lags is what the Methods specify |
| `UNBLOCK-F5-03` | Figure5_Regression.ipynb | cell 38 L6 | `FLAG` | same as F5-02, for the plotting cell |
| `UNBLOCK-F7-01` | Figure7.ipynb | cell 1 L2, cell 1 L3 | `PATH` | repoint at the writable mirror so the deposit is never written to (Figure7 -- was a Windows path) **/** repoint + add the TRAILING SLASH: every save is string concatenation (Output_folder+'x.svg'), so without it figures land as a sibling file |
| `UNBLOCK-F7-02` | Figure7.ipynb | cell 76 L26 | `NAMEDEF` | define the undefined loop bound in cell 76 |

**Exactly one edit changes a scientific quantity: `F2-03`.**

## Deliberately NOT edited, with the consequence recorded

Distinguishing a deliberate omission from an oversight is the point of this section.

| Deposit behaviour | Why it is left alone |
|---|---|
| **Figure2 cell 61 `use_both=False`** | Leaving it is both more minimal and *better*. `use_both=True` at cell 56 then `False` at cell 61 is exactly the kernel state his execution counts (84 then 94) imply, so it preserves cell 61's stored output as a numeric checkpoint. Its only cost is that cell 63 pickles two empty-defaultdict files, `Goal_progress_strict_` and `Place_strict_`, and **nothing reads either**. The four files that matter (`Place_`, `Goal_progress_`, `State_`, `State_strict_`) are real regardless, because `State_strict_` comes from `Tuned_dic['State_zmax_bool_strict']` either way. |
| **Figure2 cell 79 line 8 `['3_task','combined']`** | Cell 18 populated `combined_ABCDonly` and `3_task_all`, so this pair differs. Rather than edit it, `10_make_bookkeeping.py` creates `combined_days.npy`, and the unedited cell runs. Any `(rd, ses)` cell 18 skipped autovivifies an empty defaultdict that `np.save` pickles under a legitimate filename -- inert, because `Num_trials_ = 0` stops `non_repeat_ses_maker` selecting those sessions, but the empties should be counted and logged. |
| **Figure2 cell 79 overwrites cell 63's `Place_`/`State_`** | With a *different* definition (permutation boolean vs t-test boolean), purely because it runs later. That is the author's ordering; record which definition survives rather than change it. |
| **Figure5_Regression cell 21** | Runs correctly as deposited -- see the note on `REPLICATION_STATUS.md` §5(a) below. Four known scientific quirks are left intact: (a) it reads `tuning_phase_boolean_max[ses_ind_actual]`, the **held-out** session, for both fit and scoring, which is leakage; (b) cell 26 hardcodes `num_non_repeat_ses_found = 6` for `me11_05122021_06122021` while cells 15/21 build 7 folds, so folds 3-5 pair regressors from sessions 3,4,5 with phases/trial-times from 4,5,7 and session 7 is never scored (46 neurons); (c) it fits `PoissonRegressor` (log link, fitted intercept) but cell 26 reads out `np.sum(regressors*coeffs)` -- linear, no `exp`, no intercept, so the non-linear-link robustness check is discarded at readout; (d) `found_ses` and `num_non_repeat_ses_found` are derived from different criteria, so folds can silently pair with the wrong session's phases. |
| **Figure5_Figure6 cell 57's `confition` typo** | A genuine typo, but in a third `elif` that is unreachable under the deposited `condition='non-zero-strict'`, which matches the second branch. Editing it would change nothing. |
| **Figure5_Figure6 cell 89's `curve_fit` / `func_decay`** | Neither is imported or defined in that notebook (they live in Figure3's helper cell) -- but both uses sit inside `if use_kernel==True:` and `use_kernel=False` is set in the same cell and never reassigned. Dead code. Same for `_previous_chocies_coefficients_ABCD.npy`. |
| **Figure5_Figure6 cell 9** | Not edited; **skip it at run time**. It is a standalone exporter with zero consumers inside its own notebook, and the deposited `Xneuron_correlations` joblib cannot substitute for Figure3 cell 45 anyway: it has measures `Max_bins`/`Correlations`/`angle_units` with **no `Angles`**, and covers 3 of 25 recdays. Its `os.mkdir` is already wrapped in `try/except FileExistsError`, so it is a no-op rather than a crash. |
| **Figure3 cell 80 line 42's `ses_ind`** | Undefined in that cell; it leaks from cell 59's loop. Not a `NameError` if cell 59 ran, but it silently loads whichever session index happened to be left bound. Flagged, not fixed -- fixing it would require deciding what he meant. |
| **Figure7 cell 28's `except` branch** | Prints `All_session_ind`, which leaks from cell 24; if no sleep session ever matched, the handler itself raises `NameError`. Reachable only on an error path. |
| **Figure2 cell 20** | Source untouched -- only `cell_type` changed (`F2-04`), so the code stays readable in place while Run-All skips it. |

## Corrections to `REPLICATION_STATUS.md` established while building this copy

1. **§5(a) is wrong.** `Figure5_Regression` cell 21 runs as deposited and needs no fix. The doc reads `np.hstack((np.vstack([...])))` as leaving one entry per (trial, state); but those are *grouping* parens, not a tuple, so `np.hstack` on the `(n_trials, 4)` object array unpacks rows into a `(4*n_trials,)` object array of per-bin 1-D arrays and the single `concatenate_complex2` then descends to per-bin scalars. Reproduced: 1773/1773 bins, correct trial-major/state-minor order, works even on numpy 2.0.2 because the final list is scalars. **Adding a second concatenation would break it.**
2. **§2 is wrong on the paths.** Each notebook has exactly one `Input_folder`/`Output_folder` pair, not two. Four carried Windows paths (`Figure2_UMAP`, `Figure3`, `Figure5_Figure6`, `Figure7`); `Figure2`, `Figure5_Regression` and `Behavioural Analysis` were already local but lacked the trailing slash on `Output_folder`.
3. **§3(i) overstates recoverability.** Recovering session bookkeeping from `session_dic`/`Variable_dic` resolves only 3 of 25 recdays (they are keyed by single-day names), and mice `ab03`/`ah07` -- 6 of the 25 recdays -- are absent from both dicts entirely. The MetaData CSVs cover all of them.
4. **`pingouin` is not imported by `Figure2`** at all, so it never blocked the notebook that does the heavy regeneration.
5. **A blocker the doc missed:** `Figure2` cell 16 builds `speed_dic` over `['combined_ABCDonly']` only, while cell 18 iterates `['combined_ABCDonly','3_task_all']` -- see `F2-02`.
6. **Three orphan families the doc listed as "written by Figure2":** `Neuron_<rd>_<s>.npy`, `Location_<rd>_<s>.npy` and `State_95<rd>.npy` are written by **no** deposited notebook. See `F2-06` and `30_bridge_state_aliases.py`.
