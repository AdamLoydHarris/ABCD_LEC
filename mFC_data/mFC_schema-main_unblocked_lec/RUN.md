# Running `mFC_schema-main_unblocked_lec` -- El-Gaby's notebooks on the LEC data

> **Resume here (2026-09-14).** Everything is done except `Figure5_Figure6`, which is
> running as job 3591114 (96 h walltime, `--timeout 300000`, ~30 h expected). When it lands,
> read `logs/run_Figure5_Figure6.log` and fill in the last *pending* row of
> `LEC_REPRODUCTION.md`; nothing else depends on it.
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
deposit `../mFC_schema-main/` with the PFC run's 20 edits plus **two LEC-only edits**
(`UNBLOCK-LEC-01`, `UNBLOCK-LEC-02`; see `EDITS.md`). Everything this copy reads and writes
lives under `../lec_replication_run/`; the PFC run in `../replication_run/` is never touched.

* `LEC_REPRODUCTION.md` -- the write-up: what was fabricated and on what evidence, every way
  the LEC run differs from the PFC one, results, and the design flaws documented (not fixed).
* `EDITS.md` -- the 22 edits (generated; the audit asserts a bijection with the notebooks).
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
