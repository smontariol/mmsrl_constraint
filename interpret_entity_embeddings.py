import numpy as np
import os
import json
import collections
import ipdb
import clip
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

labels_list = ["hero", "villain", "victim", "other"]
label_frequencies = [0.027121160214685396, 0.13857485440219253, 0.05195843325339728, 0.7823455521297248]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
min_occurences=30 #min nb of occurrences of entities

entities_dict = collections.defaultdict(list)
annot_file = os.path.join('../mmsrl', 'all', 'annotations', 'train' + '.jsonl')
with open(annot_file, 'r') as json_file:
    dataset = list(map(json.loads, json_file))
for sample in dataset:
    for i, label in enumerate(labels_list):
        for entity in sample[label]:
            entities_dict[entity].append(label)

def normalise_label_freq(counter):
    counts = np.array([counter.get(label, 0) for label in labels_list])
    counts = np.divide(counts, label_frequencies)
    counts = counts / sum(counts)
    return counts


entities = []
main_label = []
label_distrib = []
frequency = []
for entity, labels in entities_dict.items():
    entities.append(entity)
    counter = collections.Counter(labels)
    distrib = normalise_label_freq(counter)
    main_label.append(distrib.argmax())
    label_distrib.append(distrib)
    frequency.append(len(labels))

print(f"Extracted {len(entities_dict)} entities.")
entities_dict_reduc = {k:v for k,v in entities_dict.items() if len(v)>min_occurences}
print(f"keeping {len(entities_dict_reduc)} entities appearing more than {min_occurences} times in the dataset.")
labels_reduc = [l for l, e in zip(main_label, entities) if e in entities_dict_reduc]
distrib_reduc = [l for l, e in zip(label_distrib, entities) if e in entities_dict_reduc]

print(collections.Counter(labels_reduc))
# Get list of entities and encode them
entities_list_final = entities_dict_reduc.keys()
entities_input_ids = [clip.tokenize(entity, truncate=True).squeeze(0).to(device) for entity in entities_list_final]
embeddings = []

# load model and get entities embeddings

model_path = '/data/almanach/user/smontari/scratch/constraint_challenge/Multimodal-semantic-role-labeling/models/model_clip_0.bin'
model = torch.load(model_path, map_location=torch.device(device))

print("loading model...")
clip_model, _ = clip.load('ViT-L/14', device=device)
for entity_encoding in tqdm.tqdm(entities_input_ids):
    entities_features = clip_model.encode_text(entity_encoding.unsqueeze(0)).to(dtype=model['entities_linear.weight'].dtype)
    entities_embedding = torch.nn.functional.linear(entities_features, weight = model['entities_linear.weight'], bias = model['entities_linear.bias'])
    embeddings.append(np.array(entities_embedding.squeeze(0).cpu().detach(), dtype='float32'))


# tsne = TSNE(n_components=2, init='random').fit_transform(embeddings)


pca = PCA(n_components=2)
pca.fit(embeddings)
print(pca.explained_variance_ratio_) #array([0.3405144 , 0.18068498])

pca_features = pca.transform(embeddings)

df = pd.DataFrame()
df["comp-1"] = pca_features[:,0]
df["comp-2"] = pca_features[:,1]

final_dict = {}
for i, entity in enumerate(entities_list_final):
    final_dict[entity] = {}
    counter = collections.Counter(entities_dict[entity])
    final_dict[entity]['distrib'] = np.array([counter.get(label, 0) for label in labels_list])
    final_dict[entity]['feature'] = pca_features[i]

pickle.dump(final_dict, open('pca_coordinates.pkl', 'wb'))

plt.figure(figsize=(8,5))

sns.scatterplot(x="comp-1", y="comp-2", hue=labels_reduc,
                palette=sns.color_palette("hls", 4),
                data=df, legend="full").set(title="T-SNE projection") 
plt.savefig('embeddings_pca_correct.png')
