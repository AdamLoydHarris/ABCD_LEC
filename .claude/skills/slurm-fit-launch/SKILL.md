---
name: slurm-fit-launch
description: "Launching and babysitting cluster jobs in this repo: per-recday GLM shards + merge (sbatch_files/), the V5 anchoring queue (code/slurm_v5/), and El-Gaby's notebooks through _preflight/50_sbatch_notebook.sh. Use whenever a fit, a notebook execution or anything that loads a data_dic must run -- never run those in the interactive shell."
---

# SLURM fit launch

## Why the queue
The interactive session is itself a SLURM step capped at 64 GB, shared with the user's VS Code and
Jupyter kernels. Long fits have been OOM-killed there (cgroup `oom_kill`), and a detached child
process dies with the session. Silent `EXIT 137` with no traceback is the cgroup, not the code.

## Before submitting
- `python code/check_mirror_parity.py` -> exit 0 (the `submit_glm_*.sh` scripts run it and refuse
  on drift; do the same by hand for anything else).
- Synthetics `--quick` pass on both trees.
- Decide the output name and check it does not already exist: a section/run name shared by two
  configurations overwrites silently.
- Disk: `/ceph/behrens` runs at ~97 %. Estimate the output size (El-Gaby prep arrays are ~16 GB
  per region per lag direction) and plan the deletion of intermediates.

## The three launchers
| what | command | notes |
|---|---|---|
| GLM, one recday per job | `bash sbatch_files/submit_glm_lec.sh --section <s> ...` then `python code/run_glm_batch.py --merge --section <s> <same flags>` | shards carry a config stamp; merge refuses mismatches |
| V5 anchoring run | `sbatch --job-name=v5_<..> --mem=100G --cpus-per-task=12 --time=1-00:00:00 code/slurm_v5/submit_v5_run.sh <spec> 12` | specs enumerated in `run_queue_v5.py`; explicit-only specs need the id |
| El-Gaby notebook | `bash mFC_data/mFC_schema-main_unblocked[_lec]/_preflight/50_sbatch_notebook.sh "<nb>.ipynb" [--skip ..] [--time HH:MM:SS] [--timeout <s>]` | 64 GB / 2 CPU / 24 h default |
| one-off analysis script | copy `sbatch_files/glm_lec.sbatch` (conda hook, thread caps, `set -e` after the env check) | keep `MPLBACKEND=Agg` for anything that plots |

## Two timeouts, not one
nbclient's per-cell `--timeout` (default 86400 s) fires independently of the SLURM walltime.
Figure5_Figure6 died at cell 17 with 72 h of walltime left. Pass both `--time` and `--timeout`.

## Watching
- `squeue -u $USER`; each run log ends in `EXIT=<rc>` -- read to that line. Under
  `--allow-errors` the runner prints `0 failed` even when cells raised: count error outputs in
  the executed notebook.
- Dependencies: `--dependency afterok:<jid>`; a failed parent leaves dependents in
  `DependencyNeverSatisfied` -- cancel and resubmit them.
- Concurrency hazard: a re-run of an upstream notebook (Figure2) rewrites files a downstream one
  reads. Check what is queued before submitting a reader.
- A "running" background process at 0 % CPU and a few MB RSS is an orphaned shell wrapper, not
  the job.

## After landing
- Verify the output exists with the expected recday count (25 LEC / 25 PFC) and that
  `run_config.json` / the shard stamp carries the settings you launched.
- Delete large intermediates you will not read again; record the job ids and the outcome in the
  relevant `RUN.md` / `*.md` with the date.
