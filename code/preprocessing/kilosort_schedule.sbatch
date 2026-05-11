#!/bin/bash
#SBATCH --job-name=KS_AH
#SBATCH --output=/ceph/behrens/adam_harris/kilosort-3/slurm_outputs/KS-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/kilosort-3/slurm_outputs/KS-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-56:00
#SBATCH --cpus-per-task 32
#SBATCH --partition gpu
#SBATCH --gres gpu:3
#SBATCH --mem=256G
#SBATCH --qos=normal

echo "Script started: $(date)"
echo "Executing the script "$(basename $0)""

echo "Loading MATLAB r2021a"
module load matlab/R2021a

echo "Loading cuda 11.2"
module load cuda/11.2
SESSION_CODE="bp04_26032024_20250219162554_ALL"

echo "Running code for session $SESSION_CODE"

matlab -nosplash -nodesktop -r "run_kilosort_ahdraft('$SESSION_CODE'); exit"

echo "All done!"