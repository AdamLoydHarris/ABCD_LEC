#!/bin/bash
#SBATCH --job-name=merge_spikeinterface
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/merge-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/merge-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-56:00
#SBATCH --cpus-per-task 32
#SBATCH --partition gpu
#SBATCH --gres gpu:2
#SBATCH --mem=256G
#SBATCH --qos=normal

echo "Script started: $(date)"
echo "running merge, proprocess and kilosort file "

echo "Loading python"
module load miniconda

echo "Loading conda"
source activate maze_ephys
echo "Conda environment activated"
MOUSE_ID='ah10'
DATES=('2025-06-10')
KILOSORT_THS=(9,8)

echo "running"

python merge_preprocess_kilosort_spikeinterface.py "$MOUSE_ID" "${DATES[@]} ${KILOSORT_THS[@]}"

echo "All done!"

