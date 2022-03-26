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
#SBATCH --output=/gpfswork/rech/lco/url46ht/outputs/generic_subsample/generic_subsample_%j_%x_%A_%a.out  # output file name
#SBATCH --error=/gpfswork/rech/lco/url46ht/outputs/generic_subsample/generic_subsample_%j_%x_%A_%a.err   # error file name
#SBATCH --array=0-104


cd $WORK/mmsrl

module purge
source ~/.bashrc
conda activate pytorch
export DATA_PATH=$WORK/data
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface/transformers
export TRANSFORMERS_OFFLINE=1

base_config=("--cyclic_subsample=None --subsample_labels=micro"
             "--cyclic_subsample=None --subsample_labels=macro"
             "--cyclic_subsample=None --subsample_labels=target --frequency_target=[2,1.5,1.75,1]"
             "--cyclic_subsample=None --subsample_labels='interpolate_micro_to_macro'"
             "--cyclic_subsample=None --subsample_labels=affine --frequency_factor=0.762 --frequency_bias=0.06"
             "configs/cyclic.py"
             "configs/cyclic.py --cyclic_subsample=cyclic_subsample_shorter")

base_name=("micro"
           "macro"
           "supramacro"
           "interpolate"
           "affine"
           "long_cycle"
           "short_cycle")
base_bpe=(500 1000 2000)

mkdir $WORK/outputs/generic_subsample


iconfig=$(($SLURM_ARRAY_TASK_ID%7))
ibpe=$(($SLURM_ARRAY_TASK_ID/7%3))
config=${base_config[$iconfig]}
bpe=${base_bpe[$ibpe]}
name=${base_name[$iconfig]}_bpe${bpe}
python -m mmsrl.train configs/generic.py --pooling=mlp $config --patience=26 --batch_per_epoch=$bpe --output_val=\"$WORK/outputs/generic_subsample/val_${name}_${SLURM_ARRAY_TASK_ID}.pkl\" --output_test=\"$WORK/outputs/generic_subsample/test_${name}_${SLURM_ARRAY_TASK_ID}.pkl\"
