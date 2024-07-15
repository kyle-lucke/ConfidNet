#!/bin/bash -l

# # Resource request
#SBATCH -p gpu-mxian
#SBATCH --gpus 1
#SBATCH -o train_base_models.out

# change directory to directory we submit job from:
cd $SLURM_SUBMIT_DIR

# Setup conda:
source /lfs/klucke.ui/miniconda3/etc/profile.d/conda.sh
conda activate torch-cf2

# echo "Training MNIST CNN"
# python3 train.py -c confs/exp_mnist.yaml -f &> train_mnist_base.out
# echo

# echo "Training SVHN CNN"
# python3 train.py -c confs/exp_svhn.yaml -f &> train_svhn_base.out
# echo

# echo "Training CIFAR10 VGG16"
# python3 train.py -c confs/exp_cifar10.yaml -f &> train_cifar10_base.out
# echo

# echo "Training CIFAR100 VGG16"
# python3 train.py -c confs/exp_cifar100.yaml -f &> train_cifar100_base.out
# echo

echo "Training CIFAR10 ResNet32"
python3 train.py -c confs/exp_cifar10_resnet32.yaml -f &> train_cifar10_resnet32_base.out
echo

echo "Training CIFAR100 ResNet32"
python3 train.py -c confs/exp_cifar100_resnet32.yaml -f &> train_cifar100_resnet32_base.out
echo


