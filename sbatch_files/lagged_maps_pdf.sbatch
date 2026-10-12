#!/bin/bash
#SBATCH --job-name=lagmaps_pdf
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/lagmaps-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/lagmaps-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-03:00
#SBATCH --cpus-per-task 2
#SBATCH --partition cpu
#SBATCH --mem=16G
#SBATCH --qos=normal

# One page of lagged spatial maps per state-tuned neuron, one PDF per region.
# Usage:  sbatch sbatch_files/lagged_maps_pdf.sbatch <lec|pfc>
# Reads lagged_spatial_maps.pkl from run_lagged_spatial_maps.py; no data_dic, no fit.
# Measured 1.3 s and 40 KB per page: LEC 1427 pages ~31 min / ~57 MB, PFC 739 ~16 min.

echo "Script started: $(date)"
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO/code"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, pandas, matplotlib" || { echo "ENV BROKEN"; exit 1; }

REGION="${1:?usage: sbatch lagged_maps_pdf.sbatch <lec|pfc>}"
set -e
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

set +e
python -u plot_lagged_maps_pdf.py --region "$REGION"
rc=$?
set -e
echo "All done! $(date) EXIT=$rc"
exit $rc
