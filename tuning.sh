#!/bin/bash
#PBS -N UM2KM
#PBS -l select=8:ncpus=40:mem=32G:mpiprocs=128:ompthreads=1
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -o out-run.txt
#PBS -P 12003906
#PBS -q normal
#PBS -l gpu_mem=32gb

export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONUNBUFFERED=1
export RAY_BACKEND_LOG_LEVEL=warning
python script/tuning.py -c config/transductive/wn18rr.yaml --gpus [0] --version v3
