#!/bin/bash
#
#
# Comments starting with #OAR are used by the resource manager if using "oarsub -S"
#
# Note : quoting style of parameters matters, follow the example
#
# The job is submitted to the default queue
#OAR -q default
# Path to the binary to run

module purge
module load conda
source activate pytorch
module load cuda/10.2
module load gcc/7.3.0
module load cudnn/7.6-cuda-10.2

model=$1
unfreeze=$2
learningrate=$3
subsamble_other=$4

echo python -m mmsrl.train \
         --ofa_filename=$model \
                configs/ofa_vqa.py --unfreeze=$unfreeze --learning_rate=$learningrate --subsamble_other=$subsamble_other\
                                        --output_val=\"outputs/ofa_vqa_${model}-${unfreeze}-${learningrate}-${subsamble_other}_val.pkl\"\
                                        --output_test=\"outputs/ofa_vqa_${model}-${unfreeze}-${learningrate}-${subsamble_other}_test.pkl\"\
                                        --save_model=\"models/ofa_vqa_${model}-${unfreeze}-${learningrate}-${subsamble_other}.bin\"

python -m mmsrl.train --ofa_filename=$model configs/ofa_vqa.py --ofa_path=ofa_dir+ofa_filename --unfreeze=$unfreeze --learning_rate=$learningrate --subsamble_other=$subsamble_other\
                                        --output_val=\"outputs/ofa_vqa_${model}-${unfreeze}-${learningrate}-${subsamble_other}_val.pkl\"\
                                        --output_test=\"outputs/ofa_vqa_${model}-${unfreeze}-${learningrate}-${subsamble_other}_test.pkl\"\
                                        --save_model=\"models/ofa_vqa_${model}-${unfreeze}-${learningrate}-${subsamble_other}.bin\"

# oarsub -p "gpu='YES'" -l /gpunum=1,walltime=20:00:00 -t besteffort ./oarscripts/train_vqa.sh
# oarsub -p "gpu='YES' and host='nefgpu39.inria.fr'" -l /gpunum=1,walltime=20:00:00 -t besteffort ./oarscripts/train_vqa.sh
# oarsub -p "gpu='YES' and host='nefgpu50.inria.fr'" -l /nodes=1/gpunum=1,walltime=20:00:00 --array-param-file oarscripts/ofa_params.txt -t besteffort ./oarscripts/train_vqa.sh


#OAR_JOB_ID=12236582
# "--ofa_input='ocr question'" \