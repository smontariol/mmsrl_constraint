# Copyright (c) Facebook, Inc. and its affiliates.

"""
  Run with for example:
  python3 mmf/tools/scripts/features/frcnn/extract_features_vgg.py \
 --image_dir ./example_images --output_folder ./output_features
"""

import argparse
import copy
import logging
import os
import torchvision
import numpy as np
import torch
import glob
import math
import cv2
import os


def get_image_files(
    image_dir,
    exclude_list=None,
    partition=None,
    max_partition=None,
    start_index=0,
    end_index=None,
    output_folder=None,
):
    files = glob.glob(os.path.join(image_dir, "*.png"))
    files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
    files.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))

    files = set(files)
    exclude = set()

    if os.path.exists(exclude_list):
        with open(exclude_list) as f:
            lines = f.readlines()
            for line in lines:
                exclude.add(line.strip("\n").split(os.path.sep)[-1].split(".")[0])
    output_ignore = set()
    if output_folder is not None:
        output_files = glob.glob(os.path.join(output_folder, "*.npy"))
        for f in output_files:
            file_name = f.split(os.path.sep)[-1].split(".")[0]
            output_ignore.add(file_name)

    for f in list(files):
        file_name = f.split(os.path.sep)[-1].split(".")[0]
        if file_name in exclude or file_name in output_ignore:
            files.remove(f)

    files = list(files)
    files = sorted(files)

    if partition is not None and max_partition is not None:
        interval = math.floor(len(files) / max_partition)
        if partition == max_partition:
            files = files[partition * interval :]
        else:
            files = files[partition * interval : (partition + 1) * interval]

    if end_index is None:
        end_index = len(files)

    files = files[start_index:end_index]

    return files


def chunks(array, chunk_size):
    for i in range(0, len(array), chunk_size):
        yield array[i : i + chunk_size], i


class ModelFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ModelFeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = torch.nn.Sequential(*self.features)
        
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = torch.nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        #self.fc = model.classifier[0]
  
    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        feature = self.features(x)
        pooled = self.pooling(feature)
        pooled = self.flatten(pooled)
        #out = self.fc(out) 
        return feature, pooled 


class FeatureExtractor:
    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.args.output_folder = self.args.output_folder + '_' + self.args.modelname + '/'
        if not os.path.exists(self.args.output_folder):
            os.mkdir(self.args.output_folder)
        if self.args.modelname == 'vgg':
            model_vgg_pretrained = torchvision.models.vgg16(pretrained=True)
            self.model = ModelFeatureExtractor(model_vgg_pretrained)
        elif self.args.modelname == 'b7':
            model_b7_pretrained = torchvision.models.efficientnet_b7(pretrained=True)
            self.model = ModelFeatureExtractor(model_b7_pretrained)
        else:
            print("Model unknown")

        # Change the device to GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Transform the image, so it becomes readable with the model
        self.transform_center = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.Resize(448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="/data/almanach/user/smontari/scratch/constraint_challenge/mmsrl/all/features", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file", default='/data/almanach/user/smontari/scratch/constraint_challenge/mmsrl/all/images/')
        parser.add_argument(
            "--modelname",
            type=str,
            help="The name of the feature to extract",
            default="vgg",
        )
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
            + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        # TODO finish background flag
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        parser.add_argument(
            "--padding",
            type=str,
            default=None,
            help="You can set your padding, i.e. 'max_detections'",
        )
        parser.add_argument(
            "--visualize",
            type=bool,
            default=False,
            help="Add this flag to save the extra file used for visualization",
        )
        parser.add_argument(
            "--partition",
            type=int,
            default=None,
            help="Add this flag to save the extra file used for visualization",
        )
        parser.add_argument(
            "--max_partition",
            type=int,
            default=None,
            help="Add this flag to save the extra file used for visualization",
        )
        return parser

    def _save_feature(self, file_name, feature, pooled):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        feat_list_base_name = file_base_name + ".npy"
        np.save(
            os.path.join(self.args.output_folder, file_base_name + ".npy"),
            feature.cpu().numpy(),
        )
        np.save(
            os.path.join(self.args.output_folder, file_base_name + "_pooled.npy"),
            pooled.cpu().numpy(),
        )

    def extract_features(self):
        image_dir = self.args.image_dir
        if os.path.isfile(image_dir):
            feature, pooled = self.get_image_center([image_dir])
            self._save_feature(image_dir, feature, pooled)
        else:
            files = get_image_files(
                self.args.image_dir,
                exclude_list=self.args.exclude_list,
                partition=self.args.partition,
                max_partition=self.args.max_partition,
                start_index=self.args.start_index,
                end_index=self.args.end_index,
            )

            finished = 0
            total = len(files)
            failed = 0
            failedNames = []

            file_names = copy.deepcopy(files)

            for chunk, begin_idx in chunks(files, self.args.batch_size):
                try:
                    feature, pooled = self.get_image_center(chunk)
                    for idx, file_name in enumerate(chunk):
                        self._save_feature(
                            file_names[begin_idx + idx],
                            feature, pooled,
                        )
                    finished += len(chunk)

                    if finished % 200 == 0:
                        print(f"Processed {finished}/{total}")
                except Exception:
                    failed += len(chunk)
                    for idx, file_name in enumerate(chunk):
                        failedNames.append(file_names[begin_idx + idx])
                    logging.exception("message")
            if self.args.partition is not None:
                print("Partition " + str(self.args.partition) + " done.")
            print("Failed: " + str(failed))
            print("Failed Names: " + str(failedNames))

    def get_image_center(self, chunk):
        # Set the image path
        imgs: torch.Tensor = torch.empty((len(chunk), 3, 448, 448), dtype=torch.float32)
        for i, path in enumerate(chunk):
            img = cv2.imread(path)
            img = self.transform_center(img)
            # Reshape the image. PyTorch model reads 4-dimensional tensor
            # [batch_size, channels, width, height]
            img = img.reshape(3, 448, 448)
            imgs[i] = img

        imgs = imgs.to(self.device)
        # We only extract features, so we don't need gradient
        with torch.no_grad():
            # Extract the feature from the image
            #feature = model_vgg(imgs).squeeze()
            feature, pooled  = self.model(imgs)
        
        return feature, pooled



if __name__ == "__main__":

    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
