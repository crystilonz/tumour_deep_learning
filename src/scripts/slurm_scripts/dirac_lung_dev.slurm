#!/bin/bash

# Slurm job options
#SBATCH --job-name=LSTM_train
#SBATCH --time=4:0:0
#SBATCH --partition=gpu-a100-40
#SBATCH --qos=dev
# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=dp289 

# Request right number of full nodes (32 cores by node fits any GPU compute nodes))
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

# Load the correct modules
module load gcc/9.3.0
module load cuda/12.3
module load openmpi/4.1.5-cuda12.3 

export OMP_NUM_THREADS=8
export OMP_PLACES=cores

# pytorch
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# These will need to be changed to match the actual application you are running
application="python"
options="/home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/pytorch_run_lstm.py \
--pkl /home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/datasets/lung_text/z_vec.pkl \
--vocab_dir /home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/datasets/lung_text/vocab.json \
--num_workers 8 \
--save_model \
--epochs 30
--save_dir /home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src/dist_models/LSTM_dev"

# We have reserved the full nodes, now distribute the processes as
# required: 4 MPI processes per node, stride of 8 cores between 
# MPI processes
# 
# Note use of gpu_launch.sh wrapper script for GPU and NIC pinning 
source /home/dp289/dp289/dc-siha1/project/tumour_deep_learning/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/dp289/dp289/dc-siha1/project/tumour_deep_learning/src"
srun --nodes=2 --ntasks-per-node=4 --cpus-per-task=8 \
     --hint=nomultithread --distribution=block:block \
     gpu_launch.sh \
     ${application} ${options}
# srun gpu_launch.sh ${application} ${options}