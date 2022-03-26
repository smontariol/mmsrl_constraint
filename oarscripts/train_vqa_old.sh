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

echo python -m mmsrl.train configs/ofa_old_best.py --output_val=outputs/ofa_vqa_old_max35_lr5_val.pkl \
                                        --output_test=outputs/ofa_vqa_old_max35_lr5_test.pkl \
                                        --save_model=models/ofa_vqa_old.bin

python -m mmsrl.train configs/ofa_old_best.py --output_val=outputs/ofa_vqa_old_max35_lr5_val.pkl \
                                        --output_test=outputs/ofa_vqa_old_max35_lr5_test.pkl \
                                        --save_model=models/ofa_vqa_old.bin

# oarsub -p "gpu='YES'" -l /gpunum=1,walltime=20:00:00 -t besteffort ./oarscripts/train_vqa_old.sh
# oarsub -p "gpu='YES' and mem>80000 and host='nefgpu39.inria.fr'" -l /gpunum=1,walltime=20:00:00 -t besteffort ./oarscripts/train_vqa_old.sh
#OAR_JOB_ID=12236582
# --unfreeze=\"classifier\" \

#ipython --pdb -c "%run -m mmsrl.train configs/ofa_vqa.py \"--ofa_path=ofa_path.replace('vqa_large_best','ofa_large')\" --ofa_unfreeze="all" --learning_rate=1e-4 --output_val=outputs/ofa_vqa_large_val_lr4.pkl --output_test=outputs/ofa_vqa_large_test_lr4.pkl --save_model=models/ofa_vqa_large_lr4.bin"