#!/bin/bash
#SBATCH -A lco@gpu
#SBATCH --job-name=mmsrl             # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=10:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/lco/url46ht/outputs/generic_features_visualbert/generic_features_visualbert_%j_%x_%A_%a.out  # output file name
#SBATCH --error=/gpfswork/rech/lco/url46ht/outputs/generic_features_visualbert/generic_features_visualbert_%j_%x_%A_%a.err   # error file name
#SBATCH --array=0-95


cd $WORK/mmsrl

module purge
source ~/.bashrc
conda activate pytorch
export DATA_PATH=$WORK/data
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface/transformers
export TRANSFORMERS_OFFLINE=1

base_config=("--features_path='all' --use_sbert=True --use_caption=True --use_entities=True"
             "--features_path=features_frcnn"
             "--features_path=features_b7"
             "--features_path=features_vgg"
             "--features_path=features_b7 --use_sbert=True  --use_entities=True"
             "--features_path=features_b7 --use_caption=True")

base_name=("all"
           "frcnn"
           "b7"
           "vgg"
           "entities"
           "caption")
base_pooling=(attention mlp)
base_lr=(5e-3 1e-4 5e-4 1e-5)

iconfig=$(($SLURM_ARRAY_TASK_ID%6))
ipooling=$(($SLURM_ARRAY_TASK_ID/6%2))
ilr=$(($SLURM_ARRAY_TASK_ID/12%4))

config=${base_config[$iconfig]}
pooling=${base_pooling[$ipooling]}
lr=${base_lr[$ilr]}
name=${base_name[$iconfig]}_${pooling}_lr${lr}

python -m mmsrl.train configs/generic.py --use_clip=False --pooling=$pooling $config --patience=26 \
        --batch_per_epoch=1000 --use_visualbert=True\
        --learning_rate=$lr \
        --cyclic_subsample=None --subsample_labels='interpolate_micro_to_macro' \
        --output_val=\"$WORK/outputs/generic_features_visualbert/val_${name}_${SLURM_ARRAY_TASK_ID}.pkl\" \
        --output_test=\"$WORK/outputs/generic_features_visualbert/test_${name}_${SLURM_ARRAY_TASK_ID}.pkl\"
