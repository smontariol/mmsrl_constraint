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


echo python -m mmsrl.train configs/ofa_vqa.py --ofa_path=OFA/checkpoints/snli_ve_large_best.pt \
                                        --output_val=outputs/ofa_snli_val.pkl \
                                        --output_test=outputs/ofa_snli_test.pkl \
                                        --save_model=models/ofa_snli.bin

python -m mmsrl.train configs/ofa_vqa.py --ofa_path=OFA/checkpoints/snli_ve_large_best.pt \
                                        --output_val=outputs/ofa_snli_val.pkl \
                                        --output_test=outputs/ofa_snli_test.pkl \
                                        --save_model=models/ofa_snli.bin

# oarsub -p "gpu='YES'" -l /gpunum=1,walltime=20:00:00 -t besteffort ./oarscripts/train_vqa.sh
# oarsub -p "gpu='YES' and host='nefgpu47.inria.fr'" -l /gpunum=1,walltime=20:00:00 -t besteffort ./oarscripts/train_snli.sh

#OAR_JOB_ID=12239573
# "--ofa_input='ocr question'" \