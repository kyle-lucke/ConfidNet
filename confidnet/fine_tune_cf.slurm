#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH --nodelist=node01 
#SBATCH -o train_cf_models.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch-cn

echo "Training SVHN CNN"
python3 train.py -c confs/selfconfid-svhn.yaml > train_svhn_cf.out
echo

echo "Training CIFAR10 Resnet32"
python3 train.py -c confs/selfconfid-cifar10-resnet32.yaml > train_cifar10_renset32_cf.out
echo

