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
#SBATCH --output=/gpfswork/rech/lco/url46ht/outputs/generic_features/generic_feature_%j_%x_%A_%a.out  # output file name
#SBATCH --error=/gpfswork/rech/lco/url46ht/outputs/generic_features/generic_feature_%j_%x_%A_%a.err   # error file name
#SBATCH --array=0-17


cd $WORK/mmsrl

module purge
source ~/.bashrc
conda activate pytorch
export DATA_PATH=$WORK/data
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1


base_config=("--features_path='all' --use_sbert=True --use_caption=True --use_entities=True --use_visualbert=True --use_clip=True"
             "--features_path=features_frcnn --use_sbert=True --use_caption=True --use_entities=True --use_visualbert=True --use_clip=True"
             "--features_path=features_b7 --use_sbert=True --use_caption=True --use_entities=True --use_visualbert=True --use_clip=True"
             "--features_path=features_vgg --use_sbert=True --use_caption=True --use_entities=True --use_visualbert=True --use_clip=True"
             "--features_path=features_frcnn --use_sbert=True  --use_entities=True --use_visualbert=True --use_clip=True"
             "--features_path=features_frcnn --use_sbert=True --use_caption=True --use_visualbert=True --use_clip=True"
             "--features_path=features_frcnn --use_sbert=True --use_caption=True --use_entities=True --use_clip=True"
             "--features_path=features_frcnn --use_sbert=True --use_caption=True --use_entities=True --use_visualbert=True"
             "--features_path=features_frcnn --use_caption=True --use_entities=True --use_visualbert=True --use_clip=True")

base_name=("all"
           "frcnn"
           "b7"
           "vgg"
           "nocaption"
           "noentities"
           "novisualbert"
           "noclip"
           "nosbert")
base_pooling=(attention mlp)

iconfig=$(($SLURM_ARRAY_TASK_ID%9))
ipooling=$(($SLURM_ARRAY_TASK_ID/9%2))
config=${base_config[$iconfig]}
pooling=${base_pooling[$ipooling]}
name=${base_name[$iconfig]}_${pooling}
python -m mmsrl.train configs/generic.py --pooling=$pooling $config --patience=26 --output_val=\"$WORK/outputs/generic_features/val_${name}_${SLURM_ARRAY_TASK_ID}.pkl\" --output_test=\"$WORK/outputs/generic_features/test_${name}_${SLURM_ARRAY_TASK_ID}.pkl\"
