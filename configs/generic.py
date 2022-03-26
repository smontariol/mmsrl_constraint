dataname = "all"
model = "generic"

use_clip = True
modelname = 'ViT-L/14' #"ViT-B/16"
workers = 3
amp = False
unfreeze="classifier"
pooling = "mlp"
use_caption = False
use_entities = False
features_path = False #"features_b7"
use_sbert = False
sbert_model = '/gpfswork/rech/lco/url46ht/.cache/torch/sentence_transformers/sentence-transformers_paraphrase-distilroberta-base-v1/'

use_visualbert = False
visualbert_name = 'uclanlp/visualbert-vcr-coco-pre'#'uclanlp/visualbert-nlvr2-coco-pre'
visualbert_text_model = "bert-base-uncased" #limjiayi/bert-hateful-memes-expanded facebook/bart-large

from configs.label_distribution import label_frequencies
from configs.micro_to_macro import multilabel, batch_per_epoch, cyclic_subsample, subsample_labels

# Model parameters
hidden_size = 768
dropout = 0

# Optimizer parameters
learning_rate = 1e-4
weight_decay = 0.001
max_grad_norm = 1

# Trainer parameters
batch_size = 8
accumulation = 16
max_epoch = 25
patience = 3

max_text_len = 128
max_nb_ent = 10
max_ent_len = 15
image_max_edge = 384
