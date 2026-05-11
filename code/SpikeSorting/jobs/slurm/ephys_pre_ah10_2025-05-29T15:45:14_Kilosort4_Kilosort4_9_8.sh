#!/bin/bash
#SBATCH --job-name=ephys_preprocessing_ah10_2025-05-29T15:45:14_Kilosort4
#SBATCH --output=SpikeSorting/jobs/out/ephys_pre_ah10_2025-05-29T15:45:14_Kilosort4_Kilosort4_9_8_IBL_True.out
#SBATCH --error=SpikeSorting/jobs/err/ephys_pre_ah10_2025-05-29T15:45:14_Kilosort4_Kilosort4_9_8_IBL_True.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00

source /etc/profile.d/modules.sh
module load miniconda
module load cuda/11.8
conda deactivate
conda deactivate
conda deactivate
conda activate maze_ephys

PYTHONPATH='.' python -c "import SpikeSorting.spikesort_session as sps; sps.preprocess_ephys_session('ah10', 'config_3', '2025-05-29T15:45:14', '../data/raw_data/ephys/ah10/config_3/2025-05-29_15-45-14', True, [9, 8], spikesort_path='../data/preprocessed_data/spikesorting')"
