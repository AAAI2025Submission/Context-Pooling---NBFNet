#!/bin/bash
#PBS-N UM2KM
#PBS -l select=8:ncpus=128:mem=440G:mpiprocs=128:ompthreads=1
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -o out-run.txt
#PBS -P 12003906
#PBS -q normal
#PBS -l gpu_mem=32gb

export CUDA_LAUNCH_BLOCKING=1

export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONUNBUFFERED=1
python script/run.py -c config/inductive/wn18rr.yaml --gpus [0] --version v1 \
--accuracy_threshold 0.37 \
--recall_threshold 0.23 \
--accuracy_graph False \
--recall_graph False \
--accuracy_graph_complement False \
--recall_graph_complement False


