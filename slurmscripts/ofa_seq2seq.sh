#!/bin/bash
#SBATCH -A lco@gpu
#SBATCH -C v100-32g
#SBATCH --job-name=mmsrl             # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=10:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/lco/url46ht/outputs/ofa_seq2seq_%j_%x_%A_%a.out  # output file name
#SBATCH --error=/gpfswork/rech/lco/url46ht/outputs/ofa_seq2seq_%j_%x_%A_%a.err   # error file name
#SBATCH --array=0-11

set -x

cd $WORK/mmsrl

module purge
source ~/.bashrc
conda activate pytorch
export DATA_PATH=$WORK/data

base_config=(""
             "--ofa_path=OFA/checkpoints/ofa_base.pt"
             "--weight_decay=0.01"
             "--ofa_path=OFA/checkpoints/ofa_base.pt --weight_decay=0.01")
base_name=("s2s_vqa_large"
           "s2s_base"
           "s2s_vqa_large_ddecay"
           "s2s_base_ddecay")
iconfig=$(($SLURM_ARRAY_TASK_ID%4))
config=${base_config[$iconfig]}
name=${base_name[$iconfig]}
python -m mmsrl.train configs/ofa_ours.py configs/ofa_vqa.py configs/ofa_seq2seq.py --patience=26 $config --output_val=\"$WORK/outputs/val_${name}_${SLURM_ARRAY_TASK_ID}.pkl\" --output_test=\"$WORK/outputs/test_${name}_${SLURM_ARRAY_TASK_ID}.pkl\"
