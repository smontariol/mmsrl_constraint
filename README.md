# Multimodal-semantic-role-labeling

https://codalab.lisn.upsaclay.fr/competitions/906

## How-to

Export the `DATA_PATH` variable to the location where you want the dataset stored:
```
$ export DATA_PATH="$PWD/../data"
```

Run the dataset preparation script (the data will be downloaded if not already present):
```
$ ./prepare_data.sh
```

Run the baseline and create a `predictions.pkl` files with the probabilities on the validation set:
```
$ python -m mmsrl.train configs/baseline.py --output_val=predictions.pkl
```

Show the content `predictions.pkl`:
```
$ python -m mmsrl.show_predictions predictions.pkl
```

Create a new ensemble prediction in the current directory `.` from directories containing several prediction files (they should contain `val` and `test` in their name):
```
$ python -m mmsrl.ensembling . path/to/directory1 path/to/directory2 …
```

Create a `.zip` containing a `.jsonl` for submission on Codalab val dataset:
```
$ python -m mmsrl.submission submission.zip predictions.pkl
```

To run using ipython:
```
ipython --pdb -c "%run -m mmsrl.train -- configs/ofa_vqa.py --learning_rate=1e-5"
```

See `CONFIG.md` for more details on the handling of hyperparameters.

## Preparing environment
```
pip install -r requirements
```

Install `CLIP`:
```
pip install git+https://github.com/openai/CLIP.git
```

Install `fairseq`:
```
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install --use-feature=in-tree-build .
```

Use `OFA` (from the repository root) and get checkpoints:
```
git submodule update --init --recursive
mkdir OFA/checkpoints
cd OFA/checkpoints
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_base.pt
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vqa_large_best.pt
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/snli_ve_large_best.pt
```


## Extracting features

### Image features:
python mmsrl/generate_image_features.py --output_folder /data/mmsrl/all/features --image_dir /data/mmsrl/all/images/ --modelname vgg
python mmsrl/generate_image_features.py --output_folder/data/mmsrl/all/features --image_dir /data/mmsrl/all/images/ --modelname b7

### Generating captions:
python -m mmsrl.generate_captions.py