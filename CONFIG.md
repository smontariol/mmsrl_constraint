# Configuration

## How-to Use

Train the model described in the `configs/baseline.py` file:
```
python -m mmsrl.train configs/baseline.py
```

Same but changing the value of the `batch_size` parameter:
```
python -m mmsrl.train configs/baseline.py --batch_size=42
```

When a command line argument starts with `--`, the right hand side of the `=` is `eval()`uated, to set a string parameter, the quotes should be escaped:
```
python -m mmsrl.train configs/baseline.py --dataname=\"covid19\"
```

Indeed, the right hand side is first evaluated in the context of the config dictionary, and can be resolved if it refers to a variable, in the following example, the value of modelname is set to the value of the config variable dataname (which is set to `all` in `configs/baseline.py`).
```
python -m mmsrl.train configs/baseline.py --modelname=dataname
```

This allows for some on-the-fly manipulation of config variables:
```
python -m mmsrl.train configs/ofa_vqa.py '--ofa_path=ofa_path.replace("vqa_large_best", "ofa_base")'
```

However, if the right hand side of `=` can't be `eval()`uated, it is cast to a string. In the following example `modelname` is set to `"datanamex"`. (This feature was asked for by Syrielle, blame her for the confusion.)
```
python -m mmsrl.train configs/baseline.py --modelname=datanamex
```


## Available Config Parameters

- `dataname`	Choose dataset, either 'uspolitics' or 'covid19', or 'all' to use both.
- `model`	Which model to use: ofa or perceiver or baseline.
- `multilabel`	Multi-label classification or not.
- `output_val`	Path to the validation output file.
- `output_train`	Path to the train output file.
- `output_test`	Path to the test output file.
- `save_model`	Save best model there.

### Multimodel
- `unfreeze`	Part of the model to train (supported: "classifier", "none", "all")

### OFA parameters
- `ofa_task`	Task for ofa: vqa_gen or snli_ve.
- `ofa_path`	Path to OFA model checkpoint.
- `ofa_classification_head`	Use a classification on the model's output instead of recasting the MMSRL problem as VQA or SNLI.
- `vqa_input`	Input to the OFA encoder for the VQA task (either "question ocr", "ocr question", or "question").
- `only_score_answer`	The prediction is only made by looking at the likelihood of answer tokens, not tokens part of the question ocr, etc.

#### OFA SNLI parameters
The output of the SNLI model for an entity looks like the following row-normalized table:

|           |   no  | maybe |  yes  |
| :---      | :---: | :---: | :---: |
| hero      |   A   |   B   |   C   |
| victim    |   A   |   B   |   C   |
| villain   |   A   |   B   |   C   |
| *(other)* |  *a*  |  *b*  |  *c*  |

The output must be of size 4: one score for each of hero, victim, villain and other.
The conversion of the above table into this prediction on size 4 is parametrized by the following options:

- `maybe_is`	Whether to group `maybe` with `yes`, `no` (or ignore it).
- `neither_for_other`	Whether a prompt is generated for *other*, if this is false the "other" class is the sum of the `no` column(s), otherwise a prompt "entity is neither a hero, villain nor victim" is used.

When `neither_for_other` is `True`, the prediction is simply the column C (or B+C if `maybe_is="yes"`).
Otherwise, when the last row does not exist, the predictions for hero, victim and villain remain the same, but the prediction for other is set to the sum of unsused cells.
    
### Baseline model parameters
- `modelname`	Which transformers model to use.
- `hidden_size`	Size of the hidden layer of the MLP for the baseline model.
- `dropout`	Dropout of the transformers' output, before the MLP.

### Optimizer parameters
- `learning_rate`	Learning rate.
- `weight_decay`	Weight decay.
- `max_grad_norm`	Max gradient normalization.
- `subsample_other`	Subsampling of the others class to counter the large class imbalance.
- `subsample_labels`	Same but per-label.
- `class_weights`	Class weights for loss computation. Doesn't work because the class imbalance is too big. Because when we have a class with low frequency, the loss gets huge and the gradient too.

### Trainer parameters
- `batch_size`	Model batch size.
- `accumulation`	Nb of accumulation steps.
- `batch_per_epoch`	Batch per epoch.
- `no_initial_valid`	Do not validate before training.
- `max_epoch`	Max epoch.
- `patience`	If the dev score worsen 3 epoch in a row, stop the training.
- `amp`	Automatic Mixed Precision training.
- `workers`	Number of workers used to generate data.

### Dataset parameters (Numbers chosen from exploration of the datasets)
- `max_text_len`	Max text length (nb of tokens). (observed: 321)
- `max_nb_ent`	Max nb of entities per image, for padding. (observed: 29)
- `max_ent_len`	Max entity name length (nb of tokens).
- `image_max_edge`	Resize images such that the longest edge has this length.
