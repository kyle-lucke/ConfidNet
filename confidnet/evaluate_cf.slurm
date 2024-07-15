#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o eval_cf_%j.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch-cf2

# MNIST
python3 test.py -c results/logs/mnist/mnist_selfconfid/config_1.yaml -e 390 > mnist_cf_test_cf.out

# SVHN
python3 test.py -c results/logs/svhn/svhn_selfconfid/config_1.yaml -e 414 > svhn_cf_test_cf.out

# CIFAR10
# python3 test.py -c  -e 448

# CIFAR100
python3 test.py -c results/logs/cifar100/cifar100_selfconfid/config_1.yaml -e 311 > cifar100_cf_test_cf.out
