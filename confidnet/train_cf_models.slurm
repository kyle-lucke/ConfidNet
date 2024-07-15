#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o train_cf_models_%j.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch-cf2

# echo "Training MNIST CNN"
# python3 train.py -c confs/selfconfid-mnist.yaml -f &> train_mnist_cf.out
# echo

# echo "Training SVHN CNN"
# python3 train.py -c confs/selfconfid-svhn.yaml -f &> train_svhn_cf.out
# echo

# echo "Training CIFAR10 VGG16"
# python3 train.py -c confs/selfconfid-cifar10.yaml -f &> train_cifar10_cf.out
# echo

# echo "Training CIFAR100 VGG16"
# python3 train.py -c confs/selfconfid-cifar100.yaml -f &> train_cifar100_cf.out
# echo

echo "Training CIFAR10 Resnet32"
python3 train.py -c confs/selfconfid-cifar10-resnet32.yaml -f &> train_cifar10_renset32_cf.out
echo

echo "Training CIFAR100 Resnet32"
python3 train.py -c confs/selfconfid-cifar100-resnet32.yaml -f &> train_cifar100_renset32_cf.out
echo

