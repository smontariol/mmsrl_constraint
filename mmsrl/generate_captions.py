import numpy as np
import os, sys
import torch
from fairseq import utils, tasks
from fairseq import checkpoint_utils
import PIL
import torchvision
import json
import pathlib
import pandas as pd
import tqdm

import mmsrl
sys.path.append(mmsrl.__path__[0] + "/../OFA")

from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel


class ImageCaption():
    def __init__(self, images_path):
        # Register caption task
        tasks.register_task('caption',CaptionTask)
        self.images_path = images_path

        # turn on cuda if GPU is available
        self.use_cuda = torch.cuda.is_available()
        # use fp16 only when GPU is available
        self.use_fp16 = False

        # Load pretrained ckpt & config
        overrides={"bpe_dir":"OFA/utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
        self.models, self.cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
                utils.split_paths('OFA/checkpoints/caption_large_best_clean.pt'),
                arg_overrides=overrides
            )

        # Move models to GPU
        for model in self.models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        # Text preprocess
        self.bos_item = torch.LongTensor([self.task.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.task.src_dict.eos()])
        self.pad_idx = self.task.src_dict.pad()

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = torchvision.transforms.Compose([
            lambda image: image.convert("RGB"),
            torchvision.transforms.Resize((self.cfg.task.patch_image_size, self.cfg.task.patch_image_size), interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
            ])

    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    def construct_sample(self, data, length=None, ocr=False):
        image = PIL.Image.open(self.images_path / data['image'])
        patch_image = self.patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = self.encode_text(" what does the image describe?", append_bos=True, append_eos=True)
        if ocr:
            ocr_text = data['OCR'].replace('\n', ' ')
            ocr_encoded = self.encode_text(ocr_text, length=length, append_bos=True, append_eos=True)
            src_text = torch.cat((ocr_encoded, src_text))
        
        src_length = torch.LongTensor([src_text.ne(self.pad_idx).long().sum()])
        sample = {
            "id":np.array(['42']),
            "net_input": {
                "src_tokens": src_text.unsqueeze(0),
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask
            }
        }
        return sample

    def generate(self, data, length=None, ocr = False):

        # Construct input sample & preprocess for GPU if cuda available
        sample = self.construct_sample(data, length, ocr)
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if self.use_fp16 else sample

        with torch.no_grad():
            result, scores = eval_step(self.task, self.generator, self.models, sample)

        return result[0]['caption']


if __name__=='__main__':

    data_path = pathlib.Path(os.environ['DATA_PATH'])
    images_path = data_path / "mmsrl" / "all" / 'images'
    caption_generator = ImageCaption(images_path)
    captions_path = data_path / "mmsrl" / "all" / "captions.csv"
    with open(captions_path, 'w', buffering=1) as f:
        for split in ['val', 'unseen_test', 'train']:
            annot_file = data_path / "mmsrl" / "all" / 'annotations' / f"{split}.jsonl"
            with annot_file.open('r') as json_file:
                dataset = list(map(json.loads, json_file))
            #print("split: ", len(dataset), "samples")
            for data in tqdm.tqdm(dataset, desc=f"{split}", leave=False):
                caption = caption_generator.generate(data)
                f.write(data['image'] + '\t' + caption + '\n')