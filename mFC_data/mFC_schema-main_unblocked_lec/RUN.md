# Running `mFC_schema-main_unblocked_lec` -- El-Gaby's notebooks on the LEC data

> **Resume here (2026-09-16 09:00).** Attempt 2c is **complete**: Figure2 (7.4 h), bridge,
> Figure3_fast (4.7 h, 0 error cells) and the checks all clean, coherence now over 45 of 50
> single days (`LEC_REPRODUCTION.md` 6b). Still running / queued, in dependency order
> (IDs in `logs/JOBS.txt`): **Figure5_Figure6 part1** (3633857, cells 0-17, ~30 h, writes the
> cell-17 checkpoint) -> **resume** (3633858, `--skip 9,17`, ~2 h) -> **Figure7** (3633768,
> the other session's sleep port). Attempt 3 of Figure5_Figure6 completed cell 17 (29.9 h) and
> then died in cell 44 (`UNBLOCK-LEC-07`); see "Figure5_Figure6 in two parts" below for the
> checkpoint that makes a repeat of that cheap. When the resume lands, read
> `logs/run_Figure5_Figure6_resume.log` and fill in the *pending* Figure5_Figure6 rows of
> `LEC_REPRODUCTION.md`.
>
> Complete: **attempt 1** (combined recdays; Figure2, bridge, Figure3_fast, both
> Figure5_Regression models, checks) and **attempt 2b** (the 50-day `3_task_all` cohort;
> Figure2, bridge, Figure3_fast, checks). Results in `LEC_REPRODUCTION.md` sections 5, 6
> and 6b. Attempt 1 and the two failed runs are archived under
> `../lec_replication_run/{logs,executed}/attempt1_combined_only/` and the
> `run_*_attempt*.log` files. Figure 5's inputs were verified byte-identical across the
> Figure2 re-run (`92_snapshot_inputs.py`), so the Figure 5 regressions were deliberately
> not re-run.
>
> **Attempt 2c (2026-09-14, user-approved):** single days rebuilt from trial-bearing
> sessions only (`15_export_lec_to_flat.py --single-days-only --overwrite`), recovering the
> five days whose first session had no trials. Figure2 -> bridge -> Figure3_fast -> checks are
> queued with `--dependency afterany:<Figure5_Figure6 job>` because the Figure2 re-run
> rewrites combined-name files that notebook reads. When they land, replace the 35-day
> coherence numbers in `LEC_REPRODUCTION.md` 6b and check the 92 compare
> (`after_rerun` -> `after_rerun2`) is clean apart from `Phase_`/`State_zmax_strict_`.
>
> Reading logs: each `run_*.log` ends in `EXIT=`; under `--allow-errors` count error outputs
> in the executed notebook rather than trusting the summary's `0 failed`. If Figure2 is
> re-run and fails, SLURM cancels the dependents (`DependencyNeverSatisfied`).

A copy of `../mFC_schema-main_unblocked/` (the PFC run) regenerated from the untouched
deposit `../mFC_schema-main/` with the PFC run's 22 edits plus **eight LEC-only edits**
(`UNBLOCK-LEC-01`, `UNBLOCK-LEC-02`; see `EDITS.md`). Everything this copy reads and writes
lives under `../lec_replication_run/`; the PFC run in `../replication_run/` is never touched.

* `LEC_REPRODUCTION.md` -- the write-up: what was fabricated and on what evidence, every way
  the LEC run differs from the PFC one, results, and the design flaws documented (not fixed).
* `EDITS.md` -- the 30 edits (generated; the audit asserts a bijection with the notebooks).
* `../../LEC_PORT_HANDOFF.md` -- the brief this copy implements.
* `../mFC_schema-main_unblocked/RUN.md` and `REPRODUCTION.md` -- the PFC run this mirrors.

## Layout

| Path | What |
|---|---|
| `../lec_replication_run/data/Intermediate_objects/` | `Input_folder` for every notebook -- built from `data_dic_lec.pkl` by `15_export_lec_to_flat.py`, not mirrored from any deposit |
| `../lec_replication_run/Output_folder/{ephys,behaviour}/` | `Output_folder` |
| `../lec_replication_run/executed/` | executed notebooks |
| `../lec_replication_run/logs/` | export manifest, run logs, audit, checks, Figure 5 stats |

## Environment

Two interpreters, deliberately:

* **`mfc_replication`** (python 3.9, numpy 1.22.0) runs the notebooks, the patcher, the
  audit, the bridge and the stats -- see the PFC `RUN.md` for why numpy must be pinned.
* **`maze_ephys`** (`/nfs/nhome/live/aharris/.conda/envs/maze_ephys/bin/python3`, numpy 2.x)
  runs `15_export_lec_to_flat.py` and `98_compare_neurons_norm.py`, because
  `data_dic_lec.pkl` was written by numpy 2 and does not unpickle under 1.22. The `.npy`
  files it writes are format 1.0 and the exporter re-reads a sample with the 3.9 kernel.

## Run order

```bash
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data
E=/nfs/nhome/live/aharris/.conda/envs/mfc_replication/bin/python
R=/nfs/nhome/live/aharris/.conda/envs/maze_ephys/bin/python3
P=mFC_schema-main_unblocked_lec/_preflight

$E $P/92_snapshot_inputs.py --snapshot before_rerun   # only when re-running Figure2 over existing Figure 5 results
$R $P/15_export_lec_to_flat.py            # 212 sessions (182 with trials) + 50 single days; refuses to overwrite without --overwrite
$E $P/20_apply_edits.py                   # rebuild the 8 notebooks from the deposit (22 edits) + EDITS.md
$E $P/20_apply_edits.py --variant figure3_fast
$E $P/20_apply_edits.py --variant elasticnet
$E $P/90_audit_diff.py                    # must PASS with 22 IDs

# Notebooks go through SLURM (64 GB / 2 CPU / 24 h), never as session children.
bash $P/50_sbatch_notebook.sh "Figure2.ipynb"                       # ~4 h; gates 1-5 run between cells
$E   $P/30_bridge_state_aliases.py                                  # AFTER Figure2: State_95 / State_99
J3=$(bash $P/50_sbatch_notebook.sh "Figure3_fast.ipynb" --skip 20 --allow-errors | tail -1)
J5=$(bash $P/50_sbatch_notebook.sh "Figure5_Regression.ipynb" | tail -1)
bash $P/50_sbatch_notebook.sh "Figure5_Regression_elasticnet.ipynb" --skip 15 --dependency "afterok:$J5"
bash $P/50_sbatch_notebook.sh "Figure5_Figure6.ipynb" --skip 9 --time 72:00:00   # needs Figure2 + bridge only, not Figure3; ~26 h on LEC
# Unattended variant (what was actually queued): wrap the bridge in sbatch with
# --dependency=afterok:<Figure2 job>, give every notebook --dependency afterok:<bridge>,
# the ElasticNet run afterok:<Poisson job>, and a final --wrap job running the checks with
# --dependency=afterany:<all>. See logs/JOBS.txt for the exact IDs.

$E $P/92_snapshot_inputs.py --snapshot after_rerun && $E $P/92_snapshot_inputs.py --compare before_rerun after_rerun
$E $P/97_lec_structural_checks.py         # 324 columns, 3-bin/5-bin, mod-3 stripe, State_99 in State_95
$R $P/98_compare_neurons_norm.py          # pipeline Neuron_ vs data_dic Neurons_norm
$E $P/96_extract_figure5_stats.py         # panel statistics -> logs/figure5_stats.{json,csv}
bash $P/99_status.sh                      # progress by per-recday file counts
```

### `--allow-errors` reports "0 failed" -- count error outputs instead

Under `--allow-errors`, nbclient records a cell's exception in the executed notebook and
continues, so `40_run_notebook.py`'s summary line says `0 failed` even when cells raised.
Read failures from the executed notebook (`outputs[].output_type == 'error'`), as
`LEC_REPRODUCTION.md` section 4 does for Figure3_fast (6 cells, all single-day-cohort).

### Two timeouts, not one

A long notebook needs **both** a SLURM walltime and a runner cell timeout.
`40_run_notebook.py --timeout` defaults to **86400 s (24 h) per cell**, and nbclient raises
`CellTimeoutError` when a cell exceeds it regardless of how much walltime the job has.
Figure5_Figure6 was killed that way with 72 h of walltime still available. Pass both:

```bash
bash $P/50_sbatch_notebook.sh "Figure5_Figure6.ipynb" --skip 9 --time 96:00:00 --timeout 300000
```

### Figure5_Figure6 in two parts: the cell-17 checkpoint

Cell 17 is ~30 h on LEC and nothing persists it until cell 76; attempt 3 lost it to an
`IndexError` in cell 44 (now `UNBLOCK-LEC-07`). `40_run_notebook.py` therefore carries two
Figure5_Figure6 gates (runner-side, no notebook edit, invisible to the audit): after cell 13
it records the kernel namespace and restores `logs/checkpoints/Figure5_Figure6_after_cell17.pkl`
if present; after cell 17 it pickles every picklable name cell 17 created. Cell 17's own
`skip_analysed=True` cannot be used for this because its line 4 re-initialises
`Spatial_anchoring_dic`, so the resume run must **skip cell 17**:

```bash
JA=$(bash $P/50_sbatch_notebook.sh "Figure5_Figure6.ipynb" --tag part1 --time 72:00:00 \
        --skip 9 --only 0-17 --timeout 300000 --out Figure5_Figure6_part1_cells0-17.ipynb | tail -1)
bash $P/50_sbatch_notebook.sh "Figure5_Figure6.ipynb" --tag resume --dependency afterok:$JA \
        --skip 9,17 --timeout 300000            # ~1-2 h; re-runnable after any later-cell fix
```

`--tag` suffixes the job name, sbatch script and `run_*.log` so the two parts do not overwrite
each other. If a cell after 17 fails, fix it, regenerate, and resubmit only the `resume` job.
Delete the checkpoint before a part1 run whose inputs (Figure2 outputs) have changed.

### Figure5_Figure6 on LEC: needs `UNBLOCK-LEC-03` and >24 h

The first submission died in cell 17 after 6.1 h (`NameError: ephys_ses_9_`; see
`EDITS.md` LEC-03). Cell 17 scales with neurons x sessions: 792 of 2851 neurons took
21947 s, so the full cell is ~22 h and the notebook ~26 h -- beyond the 24 h default.
Submit with `--time 48:00:00`:

```bash
bash $P/50_sbatch_notebook.sh "Figure5_Figure6.ipynb" --skip 9 --time 48:00:00
```

### Run-time skips (not edits)

| notebook | `--skip` | why |
|---|---|---|
| `Figure3_fast.ipynb` | 20, and `--allow-errors` | cell 20 is an example cell hardcoding `mouse_recday='ah04_01122021'` (a PFC recday); `ephys_0_` is redefined in every consumer cell. `--allow-errors` because cells 54/59/74/77/80/86/87 iterate the `3_task_all` cohort only: on LEC cell 54 therefore never writes `sigma_goalprogress.npy`, cell 80 loads it unguarded and fails, and 86/87/89 fail downstream (they cannot produce output even on PFC, see `EDITS.md`). The notebook's one consumed export, cell 45's `Xneuron_correlations_*`, is written before any of this. Every failing cell is listed in `run_Figure3_fast.log`; check that list against this table |
| `Figure5_Regression_elasticnet.ipynb` | 15 | prep arrays are model-independent; reuse the Poisson run's (must run after it) |
| `Figure5_Figure6.ipynb` | 9 | standalone exporter with no consumers, as in the PFC run |

## Scripts in `_preflight/` and whether they apply here

| script | LEC |
|---|---|
| `15_export_lec_to_flat.py` | **new** -- replaces `00_build_mirror.sh` + `10_make_bookkeeping.py`; also builds the 50 single-day recdays (per-day `Task_data_` etc. and per-session copies renumbered within the day) |
| `92_snapshot_inputs.py` | **new** -- md5 snapshot / compare of Figure5_Regression's inputs, the check behind "keep the Figure 5 results, do not re-run" |
| `20_apply_edits.py`, `90_audit_diff.py`, `40_run_notebook.py`, `50_sbatch_notebook.sh`, `30_bridge_state_aliases.py`, `96_extract_figure5_stats.py`, `99_status.sh` | repointed constants only |
| `97_lec_structural_checks.py`, `98_compare_neurons_norm.py` | **new** |
| `00_build_mirror.sh`, `10_make_bookkeeping.py` | **not applicable** (no deposit mirror, no MetaData CSVs); kept for parity with the PFC copy |
| `95_checkpoints.py` | **not applicable** -- it scores against El-Gaby's stored PFC outputs |
| `01_build_env.sh` | shared env; unchanged apart from the log path |

## Out of scope

`Basic_analysis` and `Behavioural Analysis (Figure 1)` read PFC raw files through
`Data_folder`; `Figure7` needs a sleep source LEC does not have; `Figure2_UMAP` needs
undeposited inputs even on PFC.

> **2026-09-15.** `Figure5_Regression_prospective.ipynb` (VARIANT-PRO, both copies) ran to
> EXIT=0 (LEC 3594655, PFC 3594658); results in `LEC_REPRODUCTION.md` / PFC `REPRODUCTION.md`
> dated 2026-09-15. `96_extract_figure5_stats.py` now iterates `_prospective`. The Figure2 ->
> bridge -> Figure3_fast -> checks chain (3591210-3591213) is still queued behind
> Figure5_Figure6 (3591114).
> Figure-name collision fixed the same morning (VARIANT-PRO-10; see `LEC_REPRODUCTION.md`).
> `20_apply_edits.py` variants now support `line_subs` for statements that continue with a backslash.

> **2026-09-15, Figure7.** Now in scope (see `LEC_REPRODUCTION.md` §Figure7). Order:
> `sbatch ... 12_make_figure7_inputs.py --sleep` (job 3633597, ~27 GB of shards into
> `Intermediate_objects_f7/`) -> re-run the builder without `--sleep` for the gates and day list ->
> `20_apply_edits.py` -> `90_audit_diff.py` -> `40_run_notebook.py Figure7.ipynb --only 0-20`
> (smoke) -> `50_sbatch_notebook.sh "Figure7.ipynb" --dependency afterok:3591213`.
> **2026-09-18.** Figure5_Figure6 completed via the checkpoint split (part1 3633857, 29.9 h for
> cell 17; resume 3633858, 2.1 h, 35 cells, EXIT=0) and cell 76 wrote the anchoring files.
> Figure7's first attempt (3633768) then died at cell 59 on
> `FileNotFoundError: Anchor_trial_Best_shift_time_<rd>.npy` -- the overlay is a SNAPSHOT of
> symlinks, and those files did not exist when it was built. **Always re-run
> `12_make_figure7_inputs.py` after any upstream notebook writes new files, immediately before
> submitting Figure7**; it is idempotent and only adds missing links (1125 added here).
> Resubmitted as 3644540.
> Resubmitted again as 3644578 after LEC-08 (3644540 died on the empty `post6` stage): 23 cells,
> 0 error outputs, EXIT=0, 5 min.
> **2026-09-29.** Figure7 is in scope and done; the "Out of scope" line above predates the sleep
> source. Results: `LEC_REPRODUCTION.md` §"Results, 2026-09-18".
