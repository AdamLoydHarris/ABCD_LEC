# Taskspace abstraction (LEC + PFC) -- working rules

Rules that are not derivable from the code. Facts about the data live in `docs/handoff/README.md`
and the per-analysis `code/*.md` write-ups; read those before touching an analysis.

## Git
- Do not add Co-Authored-By trailers or "Generated with Claude Code" footers to commit messages.
- Commit only when asked. Outputs, logs, pickles and the El-Gaby run trees are gitignored; source,
  notebooks (< 5 MB) and `.md` write-ups are versioned.

## Two trees, one code
- `code/` (LEC) and `mFC_data/code/` (PFC) are duplicated, not imported. Shared modules must stay
  byte-identical (`check_mirror_parity.py` lists them); `glm_analysis_v*.py` differ only by the PFC
  loader block. Edit the LEC copy, `cp` it over, run `python code/check_mirror_parity.py` (exit 0)
  before any cluster submission -- the submit scripts refuse on drift.
- LEC data: `data/processed_data/data_dic_lec.pkl` (3.8 GB, ~4 min to load). PFC data is rebuilt
  from the published per-session files by `build_data_dic_from_pfc`. Never renumber sessions.

## Engines are versioned, never edited in place
- A change to what a fit computes goes in a new copy (`glm_analysis_v3.py`,
  `elasticnet_regression_v5.py`) with the previous version frozen, an equivalence control proving
  `vN(legacy flags) == vN-1` to `max|diff| = 0`, a matching notebook name, and a `*_VN.md`
  deliverable that enumerates the hunks. Exception (user's rule, 2026-09-14): a new option inside
  the current version is fine when the default preserves every existing number and a synthetic
  control pins that default; a new version is for changes to the design matrix or selection rule.
- Heavy compute runs as a `run_*.py` script under sbatch and writes a pickle; notebooks load and
  plot (`RUN_MODE = 'load'`). Do not put multi-recday fits in notebook cells.
- `code/` stays flat until the analyses are finalised for submission; do not reorganise it.
- Every new analysis ships synthetic controls that enter through the same door as the data
  (`*_synthetics.py`, `--quick`). A rejection test must also be shown to accept something. This
  has caught six plausible-looking errors; do not skip it to save an hour.

## Cluster, not the interactive shell
- The interactive session is (typically) a 64 GB SLURM cgroup shared with the user's kernels. Anything that
  loads a data_dic or fits more than one recday goes through `sbatch` (`sbatch_files/*.sbatch`,
  `code/slurm_v5/submit_v5_run.sh`, the El-Gaby `_preflight/50_sbatch_notebook.sh`). A detached
  child dies with the session; EXIT 137 with no traceback is the cgroup.
- Fits are one job per recday writing shards, then an explicit `--merge`; the merge refuses
  mismatched config stamps. Notebooks run through nbclient have TWO timeouts (SLURM walltime and
  a per-cell `--timeout`); set both. Read logs to their `EXIT=` line; `0 failed` under
  `--allow-errors` is not evidence.

## Inference
- Mice are the replicates, not recdays or neurons (5 mice; most regional contrasts are 1-3 mice).
  Use `anatomy_split`; shuffle group labels within recday; show per-mouse points.

## Writing
- `.md` deliverables: what was done, what was measured (numbers with their semantics), what was
  NOT established, and the exact commands to reproduce. Dated paragraphs, not rewritten history.
- Messages to the user: lead with the outcome; short sentences; numbers in tables; no
  parentheticals. Use plain english with minimal AI-isms.
- A number always carries its semantics, n and statistic. Never a bare count.
- Code comments say why, in one or two lines, with the measurement that forced the decision.
  No paragraphs in code. Docstrings: one line of purpose plus the I/O shape.
