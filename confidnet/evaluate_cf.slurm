#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o evaluate_cf.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch-cn

# # MNIST
# python3 test.py -c results/logs/mnist/mnist_selfconfid/config_1.yaml -e 390 > mnist_cf_test_cf.out

# SVHN
python3 test.py -c results/logs/svhn/svhn_selfconfid/config_0.yaml > svhn_cf_test.out

# CIFAR10
python3 test.py -c results/logs/cifar10/cifar10_resnet32_selfconfid/config_0.yaml > resnet32_cf_test.out

# CIFAR10
# python3 test.py -c  -e 448

# # CIFAR100
# python3 test.py -c results/logs/cifar100/cifar100_selfconfid/config_1.yaml -e 311 > cifar100_cf_test_cf.out
