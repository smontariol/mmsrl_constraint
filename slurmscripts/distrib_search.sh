#!/bin/bash
#SBATCH -A lco@gpu
#SBATCH --job-name=mmsrl             # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=5:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=new_old_freq_%j_%x_%A_%a.out  # output file name
#SBATCH --error=new_old_freq_%j_%x_%A_%a.err   # error file name
#SBATCH --array=0-19


cd $WORK/mmsrl

module purge
source ~/.bashrc
conda activate pytorch
export DATA_PATH=$WORK/data

FACTOR=('0.1' '0.3' '0.5' '0.7' '0.9')
BIAS=('0.02' '0.05' '0.1' '0.2')
ifactor=$(($SLURM_ARRAY_TASK_ID%5))
ibias=$(($SLURM_ARRAY_TASK_ID/5))
python -m mmsrl.train configs/ofa_new_old_best.py --frequency_factor=${FACTOR[$ifactor]} --frequency_bias=${BIAS[$ibias]} --output_val=\"val_${FACTOR[$ifactor]}_${BIAS[$ibias]}.pkl\" --output_test=\"test_${FACTOR[$ifactor]}_${BIAS[$ibias]}.pkl\"
