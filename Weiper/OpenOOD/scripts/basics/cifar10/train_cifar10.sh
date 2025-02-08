#!/bin/bash
# sh scripts/basics/cifar10/train_cifar10.sh

GPU=1
CPU=1
node=73
jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
python /home/omar/weiper/Weiper/OpenOOD/main.py \
    --config /home/omar/weiper/Weiper/OpenOOD/configs/datasets/cifar10/cifar10.yml \
    /home/omar/weiper/Weiper/OpenOOD/configs/preprocessors/base_preprocessor.yml \
    /home/omar/weiper/Weiper/OpenOOD/configs/networks/resnet18_32x32.yml \
    /home/omar/weiper/Weiper/OpenOOD/configs/pipelines/train/baseline.yml \
    --seed 0
