dataname = "all"
model = "ofa"

from configs.label_distribution import label_frequencies
from configs.micro_to_macro import multilabel, batch_per_epoch, cyclic_subsample, subsample_labels

# OFA parameters
unfreeze = "all"
ofa_classification_head = True

# Optimizer parameters
learning_rate = 2e-5
weight_decay = 0.0001
max_grad_norm = 1

# Trainer parameters
batch_size = 8
accumulation = 16
max_epoch = 25
patience = 3

max_text_len = 128
max_nb_ent = 10
image_max_edge = 384

amp = True
workers = 3
