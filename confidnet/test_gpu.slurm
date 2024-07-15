#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o test_gpu_%j.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch-cf2

python3 test_gpu.py &> test.out

