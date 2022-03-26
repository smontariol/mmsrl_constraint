#!/bin/bash
#
#
# Comments starting with #OAR are used by the resource manager if using "oarsub -S"
#
# The job is submitted to the default queue
#OAR -q default
#OAR -p gpu='YES'
#OAR -l /gpunum=1,walltime=20:00:00 
#OAR -t besteffort


module purge
module load conda
source activate pytorch
module load cuda/10.2
module load gcc/7.3.0
module load cudnn/7.6-cuda-10.2

#python -m mmsrl.generate_captions
python mmsrl/extract_features_vgg.py

# PYTHONPATH=/data/almanach/user/smontari/scratch/constraint_challenge/mmf: python3 -u \
#     /data/almanach/user/smontari/scratch/constraint_challenge/mmf/tools/scripts/features/frcnn/extract_features_frcnn.py \
#   --model_file /data/almanach/user/smontari/scratch/constraint_challenge/Multimodal-semantic-role-labeling/mmfsrl/fcrnn/pytorch_model.bin \
#   --config_file /data/almanach/user/smontari/scratch/constraint_challenge/Multimodal-semantic-role-labeling/mmfsrl/fcrnn/config.yaml \
#   --output_folder "/data/almanach/user/smontari/scratch/constraint_challenge/mmsrl/all/features_frcnn/" \
#     --image_dir "/data/almanach/user/smontari/scratch/constraint_challenge/mmsrl/all/images/" 

# oarsub -p "gpu='YES' and host='nefgpu39.inria.fr'" -l /gpunum=1,walltime=20:00:00 -t besteffort ./oarscripts/extract_image_features.sh
# oarsub -p "gpu='YES' and host='nefgpu47.inria.fr'" -l /gpunum=1,walltime=20:00:00 -t besteffort -S "./oarscripts/train_clip.sh mlp"
# oarsub -S "./oarscripts/train_clip.sh attention"
