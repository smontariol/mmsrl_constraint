import numpy as np
import pandas as pd
import os
import json
from scipy.stats import entropy
import collections
from transformers import AutoTokenizer

import mmsrl.utils


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

dataset = {}

#datasetnames = ['uspolitics','covid19']
datasetnames = ['all']
for name in datasetnames:
    dataset[name] = {}
    for split in ['train', 'val', 'test']:
        annot_file = os.path.join('../mmsrl', name, 'annotations', split + '.jsonl')
        with open(annot_file, 'r') as json_file:
            dataset[name][split] = list(map(json.loads, json_file))

labels_list = mmsrl.utils.LABELS
stats_table = pd.DataFrame()
metrics_names = ['size', 'avg_nb_tokens', 'max_nb_tokens', 'nb_token_quantile95','class_distrib', 'avg_nb_ents', 'max_nb_ents', 'avg_nb_token_ents', 'max_nb_token_ents', 'total_nb_ents', 'percent_ents_in_text', 'total_nb_unique_ents', 'nb_duplicated_ent', 'samples_with_duplicate', 'entropy']
stats_table['metric_name'] = metrics_names

def fill_dict(sample, ent_label_dict):
    for i, label in enumerate(labels_list):
        for ent in sample[label]:
            if ent not in ent_label_dict:
                ent_label_dict[ent] = [0]*len(labels_list)
            ent_label_dict[ent][i] +=1

for name in datasetnames:
    for split in ['train', 'val', 'test']:
        colname = name + '_' + split
        metrics = []
        data = dataset[name][split]
        metrics.append(len(data))
        text = [data[i]['OCR'] for i in range(len(data))]
        tokenized = [tokenizer.tokenize(t) for t in text]
        nbtok = [len(t) for t in tokenized]
        metrics.append(np.mean(nbtok))
        metrics.append(np.max(nbtok))
        metrics.append(round(np.percentile(nbtok, 95), 2))
        #print(pd.Series(nbtok).describe())
        class_distrib = [0]*len(labels_list)
        nbents = []
        entities = []
        duplicated_ents = 0
        sample_with_duplicates = 0
        ent_label_dict = {}
        nb_ent_in_text = []
        for sample in data:
            sample_ents = []
            # add entities and labels to the global dictionary
            fill_dict(sample, ent_label_dict)
            for i, label in enumerate(labels_list):
                # compute class distribution
                class_distrib[i] += len(sample[label])
                sample_ents.extend(sample[label])
            # check if some entities are in several classes
            if len(sample_ents) != len(set(sample_ents)):
                #print("Duplicated entity!")
                #print(sample)
                duplicated_ents += (len(sample_ents)- len(set(sample_ents)))
                sample_with_duplicates +=1
            entities.extend(sample_ents)
            nbents.append(len(sample_ents))
            # check if the entities are in the text
            ents_in_text = [(ent in sample['OCR']) for ent in sample_ents]
            nb_ent_in_text.append(np.sum(ents_in_text))
        
        class_distrib /= np.sum(class_distrib)
        class_distrib = [round(num, 2) for num in class_distrib]
        metrics.append(class_distrib)
        metrics.append(np.mean(nbents))
        metrics.append(np.max(nbents))

        tokenized = [tokenizer.tokenize(t) for t in entities]
        nbtokent = [len(t) for t in tokenized]
        metrics.append(np.mean(nbtokent))
        metrics.append(np.max(nbtokent))

        metrics.append(len(entities))
        
        percent_ents_in_text = np.sum(nb_ent_in_text) / len(entities) * 100
        metrics.append(round(percent_ents_in_text, 2))
        metrics.append(len(set(entities)))

        metrics.append(duplicated_ents)
        metrics.append(sample_with_duplicates)        

        # entropy: how are the classes distributed fow each entity

        dataset_avg_var = 0
        dataset_avg_entropy = 0
        for ent, distrib  in ent_label_dict.items():
            mean_abs_dev_ent_ditrib = np.mean(distrib - np.mean(distrib)**2)
            dataset_avg_var += mean_abs_dev_ent_ditrib
            dataset_avg_entropy += entropy(distrib)
            #mean_abs_dev_ent_ditrib = np.mean(np.absolute(distrib - np.mean(distrib)))
        dataset_avg_var /= len(ent_label_dict)
        dataset_avg_entropy /= len(ent_label_dict)

        metrics.append((round(dataset_avg_var,2), round(dataset_avg_entropy, 2)))

        stats_table[colname] = metrics
    

    stats_table[colname] = metrics
    

print(stats_table)

stats_table.to_csv('stats_table.csv', sep='\t')
