#!/usr/bin/env bash
# Submit one unblocked notebook to slurm.
#
# WHY SBATCH AND NOT A BACKGROUND PROCESS: the interactive session is a cgroup, and a
# detached child dies with it -- Figure5_Figure6 was killed at cell 17 that way, losing
# the run without writing anything (its only export is cell 76, near the end). Long
# notebooks therefore go through the queue, where they survive session restarts and get
# their own memory allocation rather than sharing the interactive limit.
#
#   bash 50_sbatch_notebook.sh "Figure5_Figure6.ipynb" --skip 9
#   bash 50_sbatch_notebook.sh "Figure7.ipynb" --dependency afterok:12345
#
# Any argument after the notebook name is forwarded to 40_run_notebook.py, except
# --dependency <spec> which is consumed here and passed to sbatch.
set -euo pipefail

ROOT=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/mFC_data
UNB="$ROOT/mFC_schema-main_unblocked_lec"
LOG="$ROOT/lec_replication_run/logs"
ENVPY=/nfs/nhome/live/aharris/.conda/envs/mfc_replication/bin/python

NB="${1:?usage: 50_sbatch_notebook.sh <Notebook.ipynb> [--dependency spec] [runner args]}"
shift || true

DEP=""
WALL="24:00:00"   # --time <spec> overrides; LEC Figure5_Figure6 needs ~48:00:00 (cell 17 alone ~22 h)
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dependency) DEP="$2"; shift 2 ;;
        --time) WALL="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

BASE="$(basename "$NB" .ipynb | tr ' ' '_')"
mkdir -p "$LOG"
SCRIPT="$LOG/sbatch_${BASE}.sh"

cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=unb_${BASE}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=$WALL
#SBATCH --output=$LOG/slurm_${BASE}_%j.out
#SBATCH --error=$LOG/slurm_${BASE}_%j.out

# One BLAS thread: measured faster for these problem sizes (a PoissonRegressor fit on
# (30000, 324) takes 0.11 s single-threaded vs 0.14 s with 4), and it keeps the job
# inside its cpus-per-task allocation instead of oversubscribing the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

cd "$ROOT"
echo "host=\$(hostname) job=\$SLURM_JOB_ID start=\$(date -Is)"
"$ENVPY" "$UNB/_preflight/40_run_notebook.py" "$NB" ${ARGS[@]+"${ARGS[@]}"} \\
    > "$LOG/run_${BASE}.log" 2>&1
rc=\$?
echo "EXIT=\$rc" >> "$LOG/run_${BASE}.log"
echo "rc=\$rc end=\$(date -Is)"
exit \$rc
EOF
chmod +x "$SCRIPT"

if [ -n "$DEP" ]; then
    JID=$(sbatch --parsable --dependency="$DEP" "$SCRIPT")
    echo "submitted $NB as job $JID (dependency: $DEP)"
else
    JID=$(sbatch --parsable "$SCRIPT")
    echo "submitted $NB as job $JID"
fi
echo "$JID"
