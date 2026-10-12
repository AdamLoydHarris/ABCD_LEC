# El-Gaby's deposited notebooks on the LEC data: what we fabricated, how the run differs, what came out

**Status: live document.** Attempt 1 (combined recdays only) completed 2026-09-10/11.
Attempt 2 adds the 50-day single-day cohort needed for Figure 3's coherence analysis;
**attempt 2b (2026-09-11/12) completed Figure2, the bridge and Figure3_fast**, and its
results are below. Started 2026-09-10. **All notebooks have since run** -- attempt 2c
(2026-09-16), `Figure5_Figure6` through the cell-17 checkpoint split (2026-09-18) and
`Figure7` (2026-09-18); nothing is outstanding. Last updated **2026-10-08** (full audit; see
the final section).

Companion files: `EDITS.md` (the 30-edit ledger, generated), `RUN.md` (how to run this
copy, with a "Resume here" block), `../mFC_schema-main_unblocked/REPRODUCTION.md` (the PFC
run this mirrors), `../../LEC_PORT_HANDOFF.md` (the brief). Every number below was measured
on this machine.

---

## 1. The question, and the one-sentence method

What would El-Gaby's own code report if it had been given our LEC recordings? Not a port,
not a reimplementation: `data_dic_lec.pkl` is exported into the flat per-session `.npy`
layout his notebooks read, into a **separate** `Input_folder`
(`../lec_replication_run/data/Intermediate_objects/`), and a copy of the audited PFC
project runs on it. Three LEC-only notebook edits were unavoidable (section 3); none
changes a computed quantity. The PFC run in `../replication_run/` is untouched, and its
audit still passes with its 22 edits.

---

## 2. What we fabricated, and on what evidence

Nothing here is a guess where the pickle offered a measurement.

### The export (`_preflight/15_export_lec_to_flat.py`)

| decision | evidence |
|---|---|
| `trialtimes_ = int32(Trial_times * 25)` | `Trial_times.max() == Neuron_raw.shape[1]` in 182/182 usable sessions, all values integer-valued, so they are 25 ms bin indices; PFC's files are ms and every consumer does `//25`. Integer dtype is load-bearing: cell 18 slices `distances[start:end]` with them and a float slice raises. |
| `Location_raw_`: `0 -> NaN`, float64 | LEC codes untracked as 0 (3.8 % of bins); PFC uses NaN. A 0 becomes index -1 after `(Location_raw-1).astype(int)` and silently addresses the last spatial module. Values 1..9 nodes / 10..21 edges; the LEC edge order (`1-2,2-3,1-4,2-5,3-6,4-5,5-6,4-7,5-8,6-9,7-8,8-9`) is identical to his hardcoded `Edge_grid`, so no remapping. |
| `XY_raw_ + 10 cm` on both axes | LEC XY is in cm (range -4.2..51.5; adjacent nodes ~18 cm apart); PFC is pixels (0..1280). Cell 16 treats `x < 0` as untracked and sets NaN: measured, that blanks 0.84 % of LEC speed bins (122 of 182 sessions, up to 10.4 % in one), and each NaN propagates through cell 18's `cumsum` to the end of its state segment. After +10 cm: 0 bins. Speed and distance are translation-invariant. User-approved. |
| `Maze1/2_measurements.txt` with `C = R = L = W = 37` | The notebooks compute `pixels_per_cm = mean(C1..R3)/L`; LEC is already in cm so the ratio must be exactly 1 (verified by parsing the files the way cell 14 does). 37 is the measured outer-node extent in cm. Cell 14 loads only mazes '1' and '2', so both files are written and the LEC mice are mapped to maze 1. |
| `Neuron_raw_` float64 | The pickle holds uint16; all 595 PFC files are float64. scipy's filters preserve input dtype, so matching PFC removes a class of silent truncation. |
| neural arrays for all 212 sessions; tracking for 188; `trialtimes_` for 182 | Attempt 1 exported only the 182 sessions with trials. Attempt 2 writes `Neuron_raw_` for every recorded session (as his deposit does) because Figure2 cells 51/54 load `Neuron_raw_<day>_0.npy` outside any `try`, and in five recdays the second day's first session is a zero-trial one. Every per-session consumer is gated by `non_repeat_ses` or loads `trialtimes_` in the same `try`, so the 30 trial-less sessions still enter no computation: `Num_trials_ = 0` is the gate. User-approved. |
| `Task_data_` one row per session index, never renumbered | Real `Task` where present (including the 9 zero-trial ABCD sessions, as PFC's files do), zeros for the 21 Object sessions. |
| **single days** (50 pseudo recdays, `3_task_all_days.npy`) | Figure2 cell 19 loads `awake_session_`/`All_session_` for both calendar halves of every combined name outside any `try` (attempt 1's first submission died there), and Figure3 cells 23/29/32/42/51-89 analyse single days as recdays in their own right. Each session is dated from the pyControl file the registry resolves for it (`ah08-2025-06-13-114642.txt`); **every LEC session falls on exactly one of its recday's two named dates** (4+4 in 19 recdays, 4+5 in one, 5+5 or 5+6 in the five 24/25 June recdays; one undatable Object session inherits its neighbour's day). Each day gets its own `Task_data_`/`Num_trials_`/`Task_num_`/session arrays and per-session files renumbered 0..n-1 within the day (copies). `3_task_days.npy` stays **empty**, so all 50 days inherit the combined day's tuning through Figure2 cells 44/61/73 -- exactly the treatment his split two-day PFC recordings got. User-approved. **Attempt 2c (2026-09-14): a single day holds its trial-bearing sessions only** (182 kept, 30 dropped across the 50 days; combined recdays keep every row). Reason and equivalence measurement in section 6b. His PFC single days did carry zero-trial rows, so this is the one place the LEC layout deliberately departs from his. User-approved. |
| session-timestamp arrays = tokens | Figure2/3/5 consume them through `len()` only (author-confirmed for PFC); not `HH-MM-SS`, so Figure7's `np.where` cannot match a wrong file. `sleep_session_` is empty. `Task_num_` = first-occurrence label of each `Task_data_` row, as `10_make_bookkeeping.py` does. |
| cohort lists | `combined_ABCDonly_days` = `combined_days` = the 25 recdays (`<U22`); `3_task_all_days` = the 50 single days; `3_task_days` and `combined_ABCDE_days` empty. |

Export report (attempt 1): 25 recdays, 212 sessions, **182 exported / 30 skipped** (21
Object, 9 zero-trial), **2851 neurons**, both registry validators clean, a sample re-read
under numpy 1.22 with the expected dtypes. Manifest:
`../lec_replication_run/logs/lec_export_manifest.json` (attempt 1's copy under
`logs/attempt1_combined_only/`).

His own selector, recomputed on the exported files, gives 6 folds on 20 recdays, 5 on 4
and 7 on 1 (`ly06_20250618_20250619`). Four sessions enter as folds with **one** completed
trial (`ah08_20250624_20250625` s8, `ly05_20250618_20250619` s6, `ly05_20250620_20250623`
s1, `ly05_20250624_20250625` s8): his gate is `num_trials > 0`, the V5 reimplementation
required two, so his fold set is a superset of V5's by exactly these four.

---

## 3. The LEC-only edits (all `COVERAGE`; none `SCIENCE`)

The handoff predicted no notebook changes. Five were needed for Figure2/3/5 (LEC-01 to 04,
07), all because the deposit hardcodes the cohort, session layout and session counts it was
written against; LEC-05/06 belong to the Figure7 port and are described in that section.

* **`UNBLOCK-LEC-01`, Figure2 cell 7.** `Mice_maze_dic` lists PFC mice only. Cell 16 does
  `Mice_maze_dic[mouse]` inside a bare `try`, so on LEC every session printed
  `speed_dic not made`, `speed_dic` stayed empty, and cell 18's `except` then left all six
  phase/state/time dicts empty -- the whole notebook would have run to completion computing
  nothing. Runner gate 1 (`speed_dic` non-empty) is what caught it. Adds the five LEC mice
  mapped to maze 1.
* **`UNBLOCK-LEC-02`, Figure2 cell 61 (deposit index).** `np.vstack` over per-recday
  booleans of the `3_task_all` cohort, outside any `try`; empty in attempt 1, so it raised
  and aborted the run. Plot-only cell; a 3-line guard placed after its `use_both=False` so
  the kernel state matches the PFC run. (Attempt 2 populates the cohort, so the guard is
  inert there; it stays because it is harmless and audited.)
* **`UNBLOCK-LEC-03`, Figure5_Figure6 cell 17.** The cell loads each session's `Neuron_`
  into an `exec`-built name `ephys_ses_<s>_`, appending missing sessions to `ses_not_found`.
  Its first consumer loop skips `ses_not_found` before the reverse `exec`; its second (line
  173) does not. A session index no earlier recday ever defined therefore raises
  `NameError` -- `ah08_20250624_20250625` has 11 sessions of which 9 and 10 are zero-trial,
  and no recday before it has a session 9, so attempt 1 died there after 21947 s. On PFC (max
  9 sessions) the name always happened to exist from an earlier recday, holding *that*
  recday's neurons; the stale value is discarded two lines later when the `Location_` load
  fails and continues, so PFC results are unaffected. The guard makes the second loop skip
  what the first skips.

* **`UNBLOCK-LEC-04`, Figure2 cell 7.** The *second* hardcoded mouse table,
  `Mice_cohort_dic`. Cell 61 does `cohort=Mice_cohort_dic[mouse]` for every `3_task_all`
  recday, outside any `try`, and never reads `cohort` again. In attempt 1 the loop body
  never ran (empty cohort), so LEC-01 covered only the maze table -- a gap in my scan. When
  attempt 2 populated the cohort, the lookup raised `KeyError: 'ah08'` and aborted Figure2
  after 4.9 h (attempt 2a). Adds the LEC mice under an inert cohort number; nothing derived
  from the dict is consumed anywhere.

* **`UNBLOCK-LEC-07`, Figure5_Figure6 cell 44.** The simple-anchor analysis accepts an
  anchor only if >= 3 session comparisons support it (`num_ses_thr = 3`,
  `apply_minimum_sesnumber = True`). `ly05_20250620_20250623` has only three sessions with
  more than two trials, so for some neurons every (phase, location) entry of `min_value` is
  NaN, `np.nanmin` returns NaN and the deposited `np.where(...)[0][0]` indexes an empty
  array. Attempt 3 died there **after cell 17 had run for 29.9 h** and before cell 76, the
  only cell that saves anything. The guard skips the neuron, leaving the NaN the cell
  pre-fills for exactly this "no supported anchor" case. PFC never reached it: every PFC
  recday has at least four sessions with more than two trials.

Runner-side, not an edit: because of that loss, `40_run_notebook.py` now checkpoints the
post-cell-17 kernel state and can restore it with `--skip 17` (see `RUN.md`), so a failure
in cells 22-104 costs about an hour rather than thirty. Attempt 4 runs as two jobs.

Run-time skips: Figure3_fast `--skip 20` (example cell hardcoding `ah04_01122021`) with
`--allow-errors` (cells 86-89 cannot produce output on any dataset, see `EDITS.md`);
Figure5_Figure6 `--skip 9` as on PFC.

Audit: `90_audit_diff.py` PASS, 23 IDs, bijection with `EDITS.md`. Smoke tests of Figure2
cells 0-22 pass gates 1 and 2 (`speed_dic covers 25 recdays, 182 sessions`; `45293 bins,
5-bin=[0..4], 3-bin=[0,1,2]` on `ah08_20250613_20250615` session 0 -- 45293 is that
session's `Neuron_raw` bin count, the trialtimes conversion verified end to end).

---

## 4. Every way the LEC run differs from the PFC run

| | PFC run | LEC run |
|---|---|---|
| `Input_folder` | mirror of the deposit (52 GB) + bookkeeping | built from `data_dic_lec.pkl` (attempt 1: 7 GB; attempt 2 with single-day copies: ~40 GB) |
| cohorts | 25 combined + 55 single-day + 11 `3_task` + 4 ABCDE | 25 combined + 50 single days (attempt 2; none in attempt 1); `3_task` and ABCDE empty |
| edits | 22 | 30 (`LEC-01` to `LEC-08`) |
| run-time skips | Figure5_Figure6 `--skip 9` | + Figure3_fast `--skip 20 --allow-errors` |
| `Neuron_raw_` dtype | float64 as deposited | float64 by cast (pickle is uint16) |
| XY units | pixels, camera-calibrated | cm, +10 cm offset, ratio-1 calibration files |
| single-day session arrays | real timestamps from MetaData CSVs | tokens; split dated from pyControl file names |
| cell 18 residual rounding | fires (ms not multiples of 25) | never fires (bin-index origin) |
| `Num_trials_` basis | his `Num_trials_dic2` (19 recdays) or `len(trialtimes_)` | the pickle's `num_trials` (== `len(Trial_times)` in 182/182) |
| one-trial folds | none noted | 4 (section 2) |
| Figure3 single-day cells (54/59/63/66/68/80/86/87) | ran (55 days) | attempt 1: 6 cells errored under `--allow-errors` because the cohort was empty (cell 68's coherence violin was drawn from an empty array); attempt 2b: all ran, coherence over 35 of 50 days, cell 23 errored on the five days whose first session had no trials (it feeds nothing, `use_peak=False`); **attempt 2c: 0 error cells, coherence over 45 of 50 days**, the 5 exclusions all named by his own "<3 sessions" rule (section 6b) |
| Figure7 | ran | no longer out of scope: a sleep-box source exists and a separate session built the inputs (`12_make_figure7_inputs.py --sleep`, edits LEC-05/06); queued `afterok` the attempt-2c checks job. See the 2026-09-15 Figure7 section. *2026-09-29: ran as 3644578, 23 cells, 0 error outputs, EXIT=0; results in that section* |
| Figure5_Figure6 | 57091 s total, cell 17 53879 s | three failed attempts: 21947 s into cell 17 on `ephys_ses_9_` (LEC-03); at exactly 86400 s by **`40_run_notebook.py`'s own `--timeout` default**, not the SLURM walltime; then **cell 17 completed in 107489 s (29.9 h)** and cell 44 raised `IndexError` on a 3-session recday (LEC-07) before anything was saved. Attempt 4 runs cells 0-17 and the rest as separate jobs around a runner-side checkpoint. Cell 17 scales with neurons x sessions^2: 2851 neurons (PFC 2182), up to 11 sessions per recday (PFC 9) |
| Figure2_UMAP, Basic/Behavioural | partial / run | out of scope (undeposited inputs; PFC raw `Data_folder`) |
| checkpoints vs his stored outputs (`95_checkpoints.py`) | the yardstick | not applicable -- no stored LEC outputs; the yardsticks are section 5 and the PFC / V5 numbers in section 6 |

A reporting caveat found in attempt 1: under `--allow-errors` the runner's summary line says
`0 failed` even when cells raised (nbclient records the exception in the notebook and
continues). Failures are therefore counted from the executed notebook's error outputs.

**Figure 5's inputs survived the Figure2 re-run.** `92_snapshot_inputs.py` md5-compared
every file Figure5_Regression reads, before and after: **2103 of 2128 must-match files
identical**, including `State_95` 25/25, `State_99` 25/25, `State_zmax_` 25/25,
`tuning_phase_boolean_max_` 25/25, all three `GLM_anchoring_prep_dic_*` families 25/25,
`Phases_raw2_` 212/212, and `trialtimes_`/`Neuron_raw_`/`speed_`/`Location_` in full. So
the section 6 results stand without re-running the 8.5 h of regressions. The 25 files that
differ are `State_zmax_strict_*`, which my snapshot pattern `^State_zmax_` over-matched:
**no notebook loads `State_zmax_strict_`** (Figure5_Regression reads `State_zmax_`,
Figure5_Figure6 reads `State_zmax_bool_`). It differs legitimately because Figure2 cell 47
line 78 builds its null with `np.random.randint(0,4,...)`, unseeded -- a **third** stochastic
quantity in Figure2, alongside cell 25's shuffle, that the PFC write-up did not flag. The
deterministic `State_zmax_` and both tuning masks are unaffected, which is why `State_95`
and `State_99` are byte-identical across runs.

---

## 5. Structural checks (`97_lec_structural_checks.py`, attempt 1, all PASS)

| check | expected | LEC |
|---|---|---|
| `GLM_anchoring_prep_dic_regressors<rd>` columns | 324 on 25/25 | **324, 25/25** |
| `Phases_raw2_` / `Phases_raw_` values, equal length | {0,1,2} / {0..4} | **182/182 sessions correct** |
| mod-3 stripe purity, Poisson | 100 %, non-zero fraction 1/3 | **100.0000 % (1,798,632 on / 0 off), fraction 0.3333** |
| mod-3 stripe purity, ElasticNet | 100 % | **100.0000 % (60,882 on / 0 off), fraction 0.0113** |
| `State_99` subset of `State_95` | 25/25 | **25/25; 931 within 1426 of 2851** *(the check now runs over 75 recdays: 2793 within 4278 of 8553)* |
| `speed_` files == exported sessions | 182 | **182** *(superseded: check 5 now tests that every Figure 5 fold session has all six per-session files; `speed_` is 370 after the single-day cohort)* |
| pipeline `Neuron_` vs pickle `Neurons_norm` (`98_compare_neurons_norm.py`) | shapes match | **182/182 sessions, 20,809 neuron-sessions, r = 1.0000 for every one, identical trial counts** -- his `raw_to_norm` and our normalisation are the same function |

The ElasticNet non-zero fraction (1.1 % of the 324 coefficients, versus exactly one third
under the L2-only Poisson fit) says `ElasticNet(alpha=0.01, positive=True)` is extremely
sparse on LEC: most neurons' fits have a handful of live coefficients. Worth bearing in
mind when reading panel 3 below. The PFC equivalent was not measured in the PFC write-up.

Figure2 cell-level counts (from the executed notebook): cell 18 skipped exactly the 30
trial-less sessions; cells 23/25 report 0 "Files not found" and 0 "betas not calculated"
(PFC: 12 and 0); the F2-06 cell skipped the same 30; cell 66: 2851 neurons, **phase-tuned
64.5 %, state-tuned 50.0 % (1426), place-tuned 86.4 %**, state given phase 54.9 %, place given
phase-and-state 96.5 % *(re-read from the attempt-2c notebook 2026-10-08; the figures here
previously read 64.4 / 86.2 / 54.8 / 96.7, which were attempt-1 or 2b values. Phase and place
tuning come from the unseeded cell-25 shuffle, so they move between runs; state tuning does
not)* (PFC cell 61: 2182 neurons, state-tuned 59 % at p<0.05, goal-progress
~76 %).

---

## 6. Results: his code on LEC (attempt 1, combined recdays; both models, 12 lags)

`96_extract_figure5_stats.py`, `use_tuned=True`, `use_strict=False` -> `State_95`. 25/25
recdays, 2851 neurons entering, 1426 state-tuned.

| panel | **LEC, his code, Poisson** | **LEC, his code, ElasticNet** | LEC V5 corrected pool | LEC V5 superseded pool | PFC, his code, Poisson | PFC, his code, ElasticNet | paper |
|---|---|---|---|---|---|---|---|
| all state-tuned | **1235, r=+0.362, t=35.4** | **1144, r=+0.329, t=33.3** | 1170, r=+0.318, t=31.9 | 887, t=35.0 | 581, t=17.2 | 623, t=15.6 | 489, t=9.3 |
| non-zero-lag 30 deg | **541, r=+0.204, t=9.9** | **402, r=+0.165, t=7.5** | 544, r=+0.105, t=5.3 | 392, t=6.0 | 296, t=6.8 | 307, t=2.9 | 329, t=3.9 |
| non-zero-lag 90 deg | **153, r=+0.092, t=2.15 (p=0.033)** | **94, r=+0.035, t=0.64 (p=0.52)** | 203, r=+0.063, t=1.7 | 129, t=2.9 | 79, t=2.7 | 86, t=2.1 | 224, t=2.5 |

The V5 columns are `code/PANEL_POOL_CORRECTION.md`'s corrected El-Gaby-semantics pool
(primary) and the superseded every-fold pool quoted in the port brief (kept for
traceability). Reading:

* **Panels 1 and 2 reproduce V5 on LEC through his code.** ElasticNet, the model V5 uses,
  gives 1144 neurons against V5's 1170 and a stronger effect (t 33.3 vs 31.9); the panel-2
  count under Poisson (541) lands on V5's 544 by coincidence of model, while the ElasticNet
  count (402) is lower and its t (7.5) higher than V5's (5.3). The neuron-count gap that
  remained unexplained on PFC (623 vs ~465) does not appear on LEC.
* **LEC is stronger than PFC on every panel and both models**, by neuron count and by t,
  which is also what V5 found (LEC t 31.9 vs PFC ~14.5 on panel 1).
* **The 90 deg panel splits by model on LEC**: significant under Poisson (t=2.15, p=0.03,
  153 neurons), not under ElasticNet (t=0.64, 94 neurons). On PFC both models were
  significant (t 2.67 / 2.13). Given the readout defect (section 7, item 8) the Poisson
  number is the less trustworthy of the two, and the ElasticNet sparsity noted in section 5
  means panel 3's 94 neurons are those with a handful of live coefficients away from the
  anchor -- fragile at this n. Consistent with V5's "unreproduced" verdict on LEC (t=1.7).

Attempt 2b confirms the table: `92_snapshot_inputs.py` verified every Figure 5 input
byte-identical across the Figure2 re-run (section 4), so these numbers are unchanged by the
single-day work.

## 6b. Figure 3: the coherence analysis the single-day cohort unlocked

With `3_task_all` populated, Figure3's coherence chain runs and
`Coherence_tuning_violin.svg` has data (31 KB, against 8 KB of empty axes in attempt 1).
**Attempt 2c (single days = trial-bearing sessions only) is the result to quote**, over
**45 of the 50 single days**; attempt 2b's 35-day column is kept to show what the layout
change did.

| quantity | **2c (45 days)** | 2b (35 days) |
|---|---|---|
| mean coherent-tuning proportion | **0.1089** | 0.1109 |
| chance (`1/4**2`) | 0.0625 | 0.0625 |
| per-condition t vs chance | **11.45, 6.09, 5.40, 5.53** | 10.36, 4.93, 4.60, 4.86 |
| p, Holm-corrected (all 4 significant) | **6.8e-14, 8.7e-7, 3.8e-6, 3.8e-6** | 1.9e-11, 6.3e-5, 6.3e-5, 6.3e-5 |

So pairs of LEC neurons hold their relative tuning across tasks well above chance, on every
task-distance condition, and every t rises with the larger day set.

**What the layout change actually did.** In 2b, 15 of the 50 days produced no coherence
entry: 5 because cell 29 could not build them (no `Neuron_<day>_0.npy`) and 10 more dropped
*silently* inside cell 59's bare `except` -- his explicit "Less than 3 sessions" rule fired
zero times, so nothing in the output said days were missing. In 2c, cell 29 loses no day
(its 'not found' count falls from 48 to 30, which is exactly the 30 trial-less sessions of
the *combined* recdays), and the only exclusions are 5 days his own rule names and prints:
`ah08_20250624`, `ly05_20250620`, `ly05_20250623`, `ly05_20250624`, `ly07_20250625`. Four of
the five days that crashed in 2b are recovered; `ly07_20250625` has 2 non-repeat sessions and
is excluded on his criterion, not ours. Trial-less rows were padding the per-session angle
matrices with NaN and starving his 3-session test.

**Which days are in the 35, and why not 50.** Ten are dropped by cell 59's own inclusion
rules (`neurons_used_thr = 10`, and its "Less than 3 sessions" test). The other **five are
lost to a fabrication artefact worth recording**: `ah08_20250625`, `ah10_20250625`,
`ly05_20250625`, `ly06_20250625`, `ly07_20250625` -- the second day of every 24/25-June
recday -- begin with a trial-less session, so `Neuron_<day>_0.npy` (a 360-bin array, which
`raw_to_norm` cannot build without trial times) does not exist, and cells 23/29/36 load
`Neuron_<rd>_0.npy` purely for `num_neurons`. Cell 29 catches it and skips the day; cell 23
does not and raises, but **cell 23 is inert**: `use_peak=False` in cells 54/59/74, so they
consume cell 29's `Xsession_correlations`, never cell 23's `Xneuron_correlations_peaks`.
Cost: 14 of the 146 sessions the 50 days would contribute.

Measured, the fix is to build single days from trial-bearing sessions only:
`non_repeat_ses` is then **identical on all 50 days**, no trial-less row shadows a later
trial-bearing one on any day, and the only change is `X_all` on those five days -- from an
empty set (because their session 0 is an Object session carrying my all-zeros `Task`
placeholder) to a one-member set. It needs no notebook edit but costs another Figure2 and
Figure3_fast run (~9 h). **User-approved 2026-09-14, run as attempt 2c and complete
2026-09-16**: the exporter's `--single-days-only` mode rebuilt the 50 days (touching no
combined-name file, so it ran while Figure5_Figure6 was executing), then Figure2 (7.4 h,
29 cells, 0 failed), the bridge, Figure3_fast (4.7 h, 27 cells, **0 error cells** -- cell 23
no longer raises) and the checks. Attempt 2b is archived under
`logs/attempt2b_singleday_allrows/` and `executed/attempt2b_singleday_allrows/`.

Checks after the attempt-2c re-run, all PASS: 324 regressor columns on 25/25; phase arrays
correct on 182/182 sessions; mod-3 stripe purity 100 % on both models (Poisson non-zero
fraction 0.3333, ElasticNet 0.0113); `State_99` within `State_95` on all **75** recdays
(25 combined + 50 single; 4278 tuned at p<0.05, 2793 at p<0.01, of 8553 neuron-entries);
`speed_` files 370 = `XY_raw_` files 370; and the pipeline's 360-bin arrays still equal ours
at r = 1.0000 on all 20,809 neuron-sessions. `92_snapshot_inputs.py` again reports every
Figure 5 input byte-identical (2151 files) apart from the 25 `State_zmax_strict_` arrays no
notebook reads, so the section 6 table is unaffected by this re-run too.

---

## 7. Design flaws in the deposit, documented and deliberately not changed

Things a reader should know about before interpreting section 6. None was altered; the
LEC-specific ones are also in `EDITS.md` under "Deliberately NOT edited".

**Interacting with the LEC data specifically**

1. **Hardcoded cohort and session layout** (`Mice_maze_dic`, `Mice_cohort_dic`, the
   `3_task*` loops, the `ah04_01122021` example, the exec-built `ephys_ses_<s>_` names, the
   unguarded single-day loads in Figure2 cell 19). The pipeline is not portable without
   edits, and the worst failure mode is silent (bare `except`), not a crash.
2. **`x < 0` as the untracked sentinel** collides with any coordinate frame whose origin sits
   inside the arena. Handled by the export offset, not by editing his code.
3. **`frame_rate = 60` applied to 25 ms bins.** Speed is 1.5x true cm/s on both datasets
   (PFC `XY_raw_` is also 25 ms binned). Inert for Figure 5 (speed only passes a length
   check there); scales the Figure 2 speed/acceleration betas without changing their t.
4. **`num_trials > 0` admits one-trial folds** (4 on LEC). With one trial the per-trial
   z-score behind state tuning is degenerate for that session and a held-out fold is scored
   on one trial's bins.
5. **Repeat detection runs over zero-trial sessions too**, so a live session can be dropped
   for repeating a task that was never actually run. Does not occur on LEC (checked), would
   on a dataset where an aborted session precedes its re-run.
6. **Session 0 is assumed to exist** (~30 `Neuron_raw_<rd>_0.npy` reads for `num_neurons`,
   two of them outside any `try`). Holds on LEC's combined days; for single days it forced
   the export of neural arrays for trial-less sessions.
7. **Figure3's goal-progress width `sigma_goalprogress.npy` is estimated from the single-day
   cohort only** (cell 54), then consumed unguarded by cell 80; cells 59/63/66/68/74/77/80/
   86/87 share the restriction. A dataset without single days gets no coherence analysis
   and no sigma. This is why attempt 2 exists.
8. **Figure5_Figure6 cell 17's second session loop lacks the `ses_not_found` guard** its
   first loop has (LEC-03). On PFC the `exec`-built name silently held a *different recday's*
   neurons for a missing session; harmless only because the next load fails and continues.
9. **Figure5_Figure6 cell 44 assumes every neuron has at least one anchor supported by three
   session comparisons** (LEC-07); a recday with three usable sessions breaks it. More
   generally the notebook holds 30 h of results in memory until its 76th cell with no
   intermediate persistence, so any late defect costs the whole run.

**Properties of the deposit, already documented for PFC and equally present here**

10. Figure5_Regression cell 21 reads the preferred phase from the **held-out** session
   (leakage); cell 26 scores a Poisson fit through `np.sum(regressors*coeffs)` -- linear, no
   `exp`, no intercept -- which inflated PFC's panel-2 t about 2.4x versus ElasticNet; hence
   both models are run. On LEC the two models agree on panels 1-2 and disagree on panel 3.
10. Cell 26 hardcodes `num_non_repeat_ses_found = 6` for one PFC recday (cannot fire on LEC).
11. Figure2 cell 25's shuffle is unseeded, so goal-progress tuning is not exactly
    reproducible; state tuning (what Figure 5 gates on) is deterministic.
12. Bare `except` blocks turn missing inputs into silently empty dicts; the runner's live
    gates exist because of this, and `--allow-errors` hides failures from its summary line.
13. Figure3 cells 86-89 cannot produce output (`ses_ind` leaks from an earlier loop).
14. `Location_<rd>_<s>.npy` uses `normalise`'s mean binning (a PFC-run judgement call,
    carried over unchanged).

---

## 8. What the handoff predicted, and what happened

| predicted | found |
|---|---|
| no notebook edits needed | four needed (`LEC-01/02/03/04`), plus one run-time skip. Two of them (LEC-02, LEC-04) sit in the same cell and were exposed one at a time: the first by the empty single-day cohort, the second by the populated one |
| "LEC has no single-day constituents", so no per-day arrays | Figure2 cell 19 loads them unguarded (first submission died in 95 s); Figure3's coherence chain needs a full single-day cohort. Both now exported with a registry-dated split |
| `Trial_times` in bins, x25 | confirmed 182/182; **and** the result must be an integer dtype |
| `Locs_raw == 0` -> NaN, 3.8 % | confirmed; `len(Locs) == len(XY)` in 182/182 |
| 24 missing-tracking sessions are a subset of the 30 zero-trial ones | confirmed (21 Object + 3 zero-trial-without-tracking + 6 zero-trial-with-tracking) |
| `len(Locs_raw) >= Trial_times.max()` | confirmed 182/182 (excess 211..43517 bins) |
| XY: nothing said | in cm, not pixels; negative values collide with his sentinel |
| `Mice_maze_dic`: nothing said | the port's actual first blocker |
| `Neurons_norm` vs pipeline: "a large discrepancy is a real finding" | identical (r = 1.0000, 20,809 neuron-sessions) |
| V5 reference 887/392/129 | superseded by the corrected pool 1170/544/203; his code gives 1144/402/94 (ElasticNet) |
| Figure5_Figure6 "should run" | needs LEC-03 and ~26 h (cell 17 scales with neurons x sessions) |

## 2026-09-15 — Prospective lags through his own code (VARIANT-PRO, `Figure5_Regression_prospective.ipynb`)

Cell 15's bump model run on the time-reversed session and un-reversed (the V5 `future`
construction); every output namespaced `_prospective`. Job 3594655, 3.9 h, EXIT=0. Sign check:
the prospective prep arrays equal `elasticnet_regression_v5.generate_regressors_raw(...,
'future')` to max|diff| = 0 on all 6 sessions of ah08_20250613 (and the retrospective ones equal
`'past'` the same way), so the two directions are exactly his model run both ways. Prep arrays
(~16 GB) deleted after cell 26; coefficients and correlations kept.

Poisson (alpha = 1), State_95 pool, his cell-26 semantics, 25 recdays / 2851 units:

| panel | retrospective (deposited direction) | prospective |
|---|---|---|
| all state-tuned | n 1235, r +0.362, t 35.4 | n 1154, r +0.434, t 38.3 |
| non-zero-lag, 30° excluded | n 541, r +0.204, t 9.9 | n 367, r +0.136, t 5.5 |
| non-zero-lag, 90° excluded | n 153, r +0.092, t 2.15 | n 98, r +0.075, t 1.36 |

Lag-0 (place-like) prediction is as good or better prospectively; the non-zero-lag panels are
weaker prospectively in both count and effect size. Same ordering as V5 spec 12 vs 13 (his
criterion + gate, `repro_poisson_v5_{past,future}`): 30° panel 558 (t 9.6) vs 359 (t 5.0),
90° 165 (t 2.5) vs 92 (t 1.4). Stats in `logs/figure5_stats.{json,csv}` (`lag_set` =
`12-lag PROSPECTIVE (VARIANT-PRO)`).

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

## 2026-09-15 — Figure7 (sleep / offline reactivation) on LEC

"Out of scope (no sleep source)" is stale: the sleep-box arrays exist
(`data/processed_data/neuron_raw_mingyutest/Neuron_raw_<recday>_sb_<j>.npy`, 289 files, 25 ms
bins, same QC-unit rows as wake, from `code/preprocessing/extract_neuron_raw.py` with byte-exact
boundaries from `recording_sessions_in_concat.csv`), and `data/metadata/MetaData-<mouse>.csv`
already lists every Sleepbox recording with its `Ephys` start time in the 18-column schema his
cell 11 expects. Each recday is one 30-40 min morning sleep, then a 20 min sleep after every task
session -- his `preX` / `post<k>` staging.

What `_preflight/12_make_figure7_inputs.py --sleep` writes into Figure7's own Input_folder
(`lec_replication_run/data/Intermediate_objects_f7/`, a symlink overlay of the mirror, so the
mirror keeps the synthetic session tokens the Figure2/3/5 runs used):

- `binned_FR_dic_<rd>_<i>` joblib shards (float64, his format) for **every** recording of the
  25 recdays, `i` = position in chronological concat order over task + sleep, each binned fresh
  from `spike_times.npy` with origin 0 over the whole recording (his awake shards start at the
  first ephys sample, not the first A_on). Sleep shards are asserted equal to the existing
  `_sb_` arrays; task shards' mean rates are reported against the trial-anchored `Neuron_raw_`.
- `All_session_<rd>` (chronological `HH-MM-SS`, 18-24 entries), `awake_session_<rd>` /
  `awake_session_behaviour_<rd>` (the task recordings, in session-index order -- asserted equal
  to `len(Task_data_<rd>)` and increasing), `sleep_session_<rd>`.
- `MetaData.xlsx - <mouse>.csv` (the lab sheet, ISO-padded dates, `Include = 1`, commas in
  Notes -> `;`) and `MetaData.xlsx - <mouse>_sleep.csv` (the Sleepbox rows). Sleep recordings
  annotated in `Notes1` (`error sess`, `chewed through cable`, `additional sb session`; 4-6 per
  mouse) are **kept**, as his pipeline has no such exclusion (user decision 2026-09-15); they are
  listed in `logs/figure7_inputs_manifest.json`.

Notebook edits (this copy): `F7-01` (Input_folder = overlay), `F7-03`, `F7-04` as in the PFC copy;
**`LEC-05`** replaces cell 11's hardcoded PFC mouse table with the five LEC mice; **`LEC-06`**
rewrites the four `Date1=`/`Date2=` lines of cells 14/20, which slice `ddmmyyyy` recday names into
ISO dates -- on `yyyymmdd` LEC names the deposited slice gives `0615-25-20`. Audit passes with 28
IDs.

Figure7 needs `Spatial_anchoring_*` (Figure5_Figure6 cell 76, job 3591114 still in cell 17) and
the `_Angles_` files (rewritten by the queued Figure3_fast), so the LEC run is chained
`afterok` the attempt-2c checks job (3591213). Export job 3633597. **Results: pending.**
*(2026-09-15 15:19: re-pointed to `afterok` the Figure5_Figure6 resume job 3633858, because
attempt 3 of Figure5_Figure6 failed in cell 44 and its `Spatial_anchoring_*` outputs do not yet
exist; see section 3, LEC-07.)*

*(2026-09-29: the chain above is superseded. Figure7 died twice more before it ran -- 3633768 at
cell 59 on the overlay snapshot, 3644540 on the empty `post6` stage, fixed by LEC-08 -- and then
completed as 3644578. The audit now lists 30 IDs across the copy, LEC-07 and LEC-08 included.
Results below.)*

#### Results, 2026-09-18 (job 3644578; 23 cells, 0 error outputs; 5 min; EXIT=0)

Written 2026-09-29 from `lec_replication_run/executed/Figure7.ipynb`. Error outputs were counted in
the notebook, not taken from the runner's summary.

Cell 89, the headline. Close = co-anchored pairs 0-180° apart in forward distance, far = 180-360°.
Cell 87 computes one Pearson r of forward angle against 250 ms sleep co-activity per recday x sleep
stage and arm, and sets an arm to NaN when it has 10 pairs or fewer; cell 89 keeps the units where
both arms are defined. PFC and paper columns are copied from PFC `REPRODUCTION.md` §Figure7.

| | LEC, 25 recdays | PFC, ours, 21 recdays | PFC, his stored notebook | paper Fig. 7e |
|---|---|---|---|---|
| recday x stage units | 100 | 37 | 36 | 59 |
| close vs far, paired t (p) | **-7.81 (6e-12)** | -2.81 (0.008) | -5.16 (1e-5) | -5.55 (7e-7) |
| close r vs 0, t (p) | -5.17 (1e-6) | -1.15 (0.26) | -3.59 (0.001) | -5.16 (1e-4) |
| far r vs 0, t (p) | +6.98 (3e-10) | +3.70 (7e-4) | +3.00 (0.005) | +4.16 (1e-4) |
| Wilcoxon p | 3.3e-10 | 0.012 | 1.5e-5 | -- |
| cell 69 betas t (circular, forward) | -1.25, +0.33 | -1.32, -1.34 | -0.85, +1.02 | -- |

The pool: cell 87 finds 144 recday x stage units with at least one co-anchored pair, over 24
recdays; 100 have both arms above 10 pairs. The 25th recday, `ly05_20250620_20250623`, has **no**
co-anchored pair in any stage (cell 87 prints "Not used" only in that case). 13 of its 91 neurons
are anchored, against a median anchored fraction of 0.58 over the 25 recdays (range 0.14-0.71).
It is also the LEC-07 recday.

Per-stage Pearson r of forward angle with sleep co-activity (cell 82; pairs pooled over recdays, so
the p values treat pairs as independent):

| stage | close arm r (p) | far arm r (p) |
|---|---|---|
| preX | -0.13 (9e-5) | +0.21 (2e-12) |
| post0 | -0.16 (1e-6) | +0.23 (2e-14) |
| post1 | -0.09 (0.015) | +0.11 (0.003) |
| post2 | -0.10 (0.004) | +0.10 (0.004) |
| preX2 | -0.12 (3e-4) | +0.08 (0.013) |
| post3 | -0.13 (2e-5) | +0.08 (0.009) |
| post4 | -0.14 (5e-5) | +0.12 (7e-5) |
| post5 | -0.09 (0.65) | +0.37 (0.11) |

Mean sleep co-activity by forward distance (cell 86 plots cell 82's `means_mean`, 45° bins). Cell 86
prints nothing; these values are read from the marker positions in
`Output_folder/ephys/Forward_distance_vs_corr_Allsleep.svg` against its y ticks:

| forward distance | 0-45 | 45-90 | 90-135 | 135-180 | 180-225 | 225-270 | 270-315 | 315-360 |
|---|---|---|---|---|---|---|---|---|
| co-activity | 0.082 | 0.032 | 0.055 | 0.048 | 0.019 | 0.039 | 0.042 | 0.079 |

**The close-vs-far effect reproduces on LEC, more strongly than on PFC or in the paper.** Both arms
carry it: close r is negative and far r positive in all 8 stages, significant in 7 (post5 is the
exception). **The curve is not a clean V.** It is high at both ends (0.082, 0.079) and lowest at
180-225° (0.019), but it dips to 0.032 at 45-90° and recovers to 0.055 at 90-135°. His stored PFC
curve falls monotonically to its floor at 90-180°.

Secondary numbers: cell 74 within- vs between-anchor circular betas, t -0.42, p 0.337 one-sided
(N 1462 / 47648); paper t -1.80, P 0.036: not reproduced, as on PFC. Cell 72 pre- vs post-task
sleep betas: circular t -0.59 (p 0.55), forward t +0.74 (p 0.46), spatial correlation t -2.05
(p 0.040).

Not established:
- **Anything per mouse.** The 100 units come from 24 recdays of 5 mice, and cell 89 treats each
  as independent. A per-mouse split needs cell 89's per-unit arrays with their recday labels, which
  the notebook does not save; that is a rerun.
- **Anything regional.** The pool is the unsplit probe mixture (ENTl, ENTm, SUB/ProS, CA1).
- **Stage post6.** LEC-08 dropped it: no co-anchored pair in any of the 25 recdays.
- **Why the curve dips at 45-90°.** Not investigated.

Provenance. The input builder was run three times. 3633597 stopped at
`GATE ah08_20250624_20250625: 10 wake recordings vs 11 Task_data rows` and 3633710 at the cell-14/20
lookup gate, both `EXIT=1` in their logs although sacct reports COMPLETED 0:0. The build that
passed every gate, 2026-09-18 20:53, left `logs/figure7_inputs_manifest.json` (25 recdays in
`gate.pass_`, 490 recordings = 211 task + 279 sleep) but no log. "Export job 3633597" above is the
first failed attempt.

Reproduce (maze_ephys, from `mFC_data/mFC_schema-main_unblocked_lec/`): see `RUN.md` §"2026-09-15,
Figure7" for the builder order, then `bash _preflight/50_sbatch_notebook.sh "Figure7.ipynb"`.
Re-run `_preflight/12_make_figure7_inputs.py` immediately before submitting; the overlay is a
snapshot of symlinks.

## 2026-09-29 — Figure5_Figure6 printed statistics (Fig. 5b / 5c / 5f, ED 7h)

Figure5_Figure6 completed on 2026-09-18 through the cell-17 checkpoint split (`RUN.md`).
n = state-tuned neurons (his `State_`), pooled over recdays; t = one-sample t against 0 with
neurons as replicates; z = his two-proportions test against 25 %. All read from the printed cell
outputs of the executed notebooks (`{lec_,}replication_run/executed/Figure5_Figure6.ipynb`) and
the deposit (`mFC_schema-main/Figure5_Figure6.ipynb`); paper values from the Fig. 5 legend and
ED 7.

| panel (cell) | paper | his stored notebook | PFC, our run | LEC, our run |
|---|---|---|---|---|
| Fig. 5b coherent proportion vs 25 %, all state-tuned (c42) | 39 %, n 738, z 5.69 | 40.4 %, n 738, z 6.31 | 35.8 %, n 738, z 4.50 | 46.1 %, n 1426, z 11.80 |
| Fig. 5b, non-zero lag (c62) | 36 %, n 545, z 3.87 | 35.9 %, n 314, z 2.97 | 36.1 %, n 288, z 2.89 | 36.4 %, n 433, z 3.64 |
| Fig. 5c single-anchor map r, all (c69) | n 737, t 9.86 | n 737, t 11.72 | n 736, t 9.08 | n 1423, t 26.30 |
| Fig. 5c, non-zero lag (c69) | n 544, t 7.55 | n 314, t 6.84 | n 288, t 5.54 | n 433, t 11.01 |
| Fig. 5f held-out lagged-map r, all (c34) | n 738, t 22.6 | n 738, t 22.64 | n 738, t 20.21 | n 1426, t 45.51 |
| Fig. 5f, non-zero lag 30° (c34) | n 285, t 7.48 | n 285, t 7.48 | n 307, t 5.72 | n 356, t 8.42 |
| ED 7h, non-zero lag 90° (c34) | N 135, t 7.07 | n 135, t 7.07 | n 164, t 3.82 | n 179, t 5.83 |

The deposit reproduces the paper's Fig. 5f and ED 7h exactly; its Fig. 5b/5c non-zero-lag pool is
314, not the legend's 545/544. The cell-31 statistic behind Fig. 5f was never saved per neuron.
`code/lagged_spatial_maps.py` rebuilds it from `Spatial_anchoring_Phase_shifted_maps_*` and
reproduces the c34 n and t above exactly in both regions (dt = 0.0 on all six); per-mouse
results, the LEC anatomy split and the place-vs-goal frame test are in `code/LAGGED_SPATIAL.md`.
Note `Spatial_anchoring_Cross_val_corr_*` is the Fig. 5c single-anchor value (cell 47), not
Fig. 5f.
