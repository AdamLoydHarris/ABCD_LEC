# El-Gaby's deposited notebooks on the LEC data: what we fabricated, how the run differs, what came out

**Status: live document.** Attempt 1 (combined recdays only) completed 2026-09-10/11.
Attempt 2 adds the 50-day single-day cohort needed for Figure 3's coherence analysis;
**attempt 2b (2026-09-11/12) completed Figure2, the bridge and Figure3_fast**, and its
results are below. `Figure5_Figure6` is the only notebook still outstanding: it was killed
twice and is re-running as of 2026-09-14 (section 4). Started 2026-09-10.

Companion files: `EDITS.md` (the 23-edit ledger, generated), `RUN.md` (how to run this
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
audit still passes with its 20 edits.

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

## 3. The four LEC-only edits (all `COVERAGE`; none `SCIENCE`)

The handoff predicted no notebook changes. Four were needed, all because the deposit
hardcodes the cohort and session layout it was written against:

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
| edits | 20 | 23 (`LEC-01/02/03`) |
| run-time skips | Figure5_Figure6 `--skip 9` | + Figure3_fast `--skip 20 --allow-errors` |
| `Neuron_raw_` dtype | float64 as deposited | float64 by cast (pickle is uint16) |
| XY units | pixels, camera-calibrated | cm, +10 cm offset, ratio-1 calibration files |
| single-day session arrays | real timestamps from MetaData CSVs | tokens; split dated from pyControl file names |
| cell 18 residual rounding | fires (ms not multiples of 25) | never fires (bin-index origin) |
| `Num_trials_` basis | his `Num_trials_dic2` (19 recdays) or `len(trialtimes_)` | the pickle's `num_trials` (== `len(Trial_times)` in 182/182) |
| one-trial folds | none noted | 4 (section 2) |
| Figure3 single-day cells (54/59/63/66/68/80/86/87) | ran (55 days) | attempt 1: 6 cells errored under `--allow-errors` because the cohort was empty (cell 68's coherence violin was drawn from an empty array); attempt 2: *pending* |
| Figure5_Figure6 cell 17 | 53879 s | killed twice before finishing: at 21947 s on `ephys_ses_9_` (fixed by LEC-03), then at exactly 86400 s by **`40_run_notebook.py`'s own `--timeout` default**, not the SLURM walltime (which was 72 h). The third submission passes `--timeout 300000` with a 96 h walltime. Cell 17 scales with neurons x sessions^2: LEC has 2851 neurons (PFC 2182) and up to 11 sessions per recday (PFC 9), so >24 h is expected |
| Figure7, Figure2_UMAP, Basic/Behavioural | run / partial / run | out of scope (no sleep source; undeposited inputs; PFC raw `Data_folder`) |
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
| `State_99` subset of `State_95` | 25/25 | **25/25; 931 within 1426 of 2851** |
| `speed_` files == exported sessions | 182 | **182** |
| pipeline `Neuron_` vs pickle `Neurons_norm` (`98_compare_neurons_norm.py`) | shapes match | **182/182 sessions, 20,809 neuron-sessions, r = 1.0000 for every one, identical trial counts** -- his `raw_to_norm` and our normalisation are the same function |

The ElasticNet non-zero fraction (1.1 % of the 324 coefficients, versus exactly one third
under the L2-only Poisson fit) says `ElasticNet(alpha=0.01, positive=True)` is extremely
sparse on LEC: most neurons' fits have a handful of live coefficients. Worth bearing in
mind when reading panel 3 below. The PFC equivalent was not measured in the PFC write-up.

Figure2 cell-level counts (from the executed notebook): cell 18 skipped exactly the 30
trial-less sessions; cells 23/25 report 0 "Files not found" and 0 "betas not calculated"
(PFC: 12 and 0); the F2-06 cell skipped the same 30; cell 66: 2851 neurons, **phase-tuned
64.4 %, state-tuned 50.0 % (1426), place-tuned 86.2 %**, state given phase 54.8 %, place given
phase-and-state 96.7 % (PFC cell 61: 2182 neurons, state-tuned 59 % at p<0.05, goal-progress
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
Cell 66, over **35 of the 50 single days**:

| quantity | LEC |
|---|---|
| mean coherent-tuning proportion | **0.1109** |
| chance (`1/4**2`) | 0.0625 |
| per-condition t vs chance | **10.36, 4.93, 4.60, 4.86** |
| p (Holm-corrected, all 4 significant) | 1.9e-11, 6.3e-5, 6.3e-5, 6.3e-5 |

So pairs of LEC neurons hold their relative tuning across tasks well above chance, on every
task-distance condition.

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
Figure3_fast run (~9 h). **User-approved 2026-09-14 and done as attempt 2c**: the exporter's
`--single-days-only` mode rebuilt the 50 days (touching no combined-name file, so it ran
while Figure5_Figure6 was executing), and Figure2 -> bridge -> Figure3_fast -> checks were
queued behind Figure5_Figure6. The 35-day numbers above are attempt 2b's, archived under
`logs/attempt2b_singleday_allrows/`; attempt 2c's replace them here when they land:
*pending*.

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

**Properties of the deposit, already documented for PFC and equally present here**

9. Figure5_Regression cell 21 reads the preferred phase from the **held-out** session
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
