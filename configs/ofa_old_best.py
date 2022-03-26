dataname = "all"
model = "ofa"

multilabel = False
from configs.label_distribution import label_frequencies
subsample_labels = "macro"
subsample_other = 0.5

# OFA parameters
ofa_task = "vqa_gen"
ofa_dir = "OFA/checkpoints/"
ofa_path = f"{ofa_dir}/ofa_base.pt"
vqa_input = "question ocr"
decoder_input = ""
unfreeze = "all"
ofa_classification_head = True

keep_all_batches = True
group_image_entities = True

# Optimizer parameters
learning_rate = 1e-5
weight_decay = 0.0001
max_grad_norm = 1

# Trainer parameters
batch_size = 1
accumulation = 16
max_epoch = 35
patience = 25

max_text_len = 128
max_nb_ent = 10
image_max_edge = 384

amp = True
workers = 3
