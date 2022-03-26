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
#SBATCH --output=/gpfswork/rech/lco/url46ht/outputs/ofa_subsample_bpe_%j_%x_%A_%a.out  # output file name
#SBATCH --error=/gpfswork/rech/lco/url46ht/outputs/ofa_subsample_bpe_%j_%x_%A_%a.err   # error file name
#SBATCH --array=0-53


cd $WORK/mmsrl

module purge
source ~/.bashrc
conda activate pytorch
export DATA_PATH=$WORK/data

base_config=("--cyclic_subsample=None --subsample_labels=micro"
             "--cyclic_subsample=None --subsample_labels=macro"
             "--cyclic_subsample=None --subsample_labels=target --frequency_target=[2,1.5,1.75,1]"
             "--cyclic_subsample=None --subsample_labels='interpolate_micro_to_macro'"
             "configs/alternate_macro_micro.py"
             "configs/alternate_macro_micro.py --cyclic_subsample=cyclic_subsample_shorter")
base_name=("micro"
           "macro"
           "supramacro"
           "interpolate"
           "long_cycle"
           "short_cycle")
base_bpe=(500 1000 2000)

iconfig=$(($SLURM_ARRAY_TASK_ID%6))
ibpe=$(($SLURM_ARRAY_TASK_ID/6%3))
config=${base_config[$iconfig]}
bpe=${base_bpe[$ibpe]}
name=${base_name[$iconfig]}_bpe${bpe}
python -m mmsrl.train configs/ofa_ours.py configs/ofa_vqa.py --ofa_classification_head --patience=26 --batch_per_epoch=$bpe $config --output_val=\"$WORK/outputs/val_${name}_${SLURM_ARRAY_TASK_ID}.pkl\" --output_test=\"$WORK/outputs/test_${name}_${SLURM_ARRAY_TASK_ID}.pkl\"
