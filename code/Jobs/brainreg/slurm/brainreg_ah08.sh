#!/bin/bash
#SBATCH --job-name=brainreg_ah08
#SBATCH --output='Jobs/brainreg/out/brainreg_ah08.out'
#SBATCH --error='Jobs/brainreg/err/brainreg_ah08.err'
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64GB
#SBATCH --time=23:59:00

echo $SLURMD_NODENAME
source /etc/profile.d/modules.sh

echo "Loading Brainglobe module"
module load brainglobe/2024-03-01
nvidia-smi

echo "Running brainreg"

brainreg ../data/raw_data/histology/ah08/stitchedImages_100/3 ../data/preprocessed_data/brainreg/ah08/allen_mouse_10um --additional ../data/raw_data/histology/ah08/stitchedImages_100/2 -v 20.0 4.033 4.033 --orientation psl --atlas allen_mouse_10um --debug