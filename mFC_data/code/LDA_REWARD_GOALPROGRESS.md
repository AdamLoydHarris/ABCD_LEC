# Time in session x goal progress LDA — PFC / mFC dataset

`lda_reward_goalprogress.py` · `run_lda_reward_progress.py` · `PFC_lda_analyses.ipynb` (cell 30 on)
· pickle `mFC_data/glm_outputs/PFC_lda/reward_progress.pkl`.

Mirror of the LEC write-up at [`../../code/LDA_REWARD_GOALPROGRESS.md`](../../code/LDA_REWARD_GOALPROGRESS.md):
samples, the joint trial x progress LDA, the held-out readouts and their circular-shift null, the
legacy reward x progress path, commands and results for both datasets live there.

Run: `sbatch --job-name=lda_rp_pfc --time=0-06:00 sbatch_files/lda_reward_progress.sbatch pfc`.
PFC recdays come from `combined_ABCDonly_days.npy`; `me10_20122021_21122021` is skipped (1 PC).
