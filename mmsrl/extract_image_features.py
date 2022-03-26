import numpy as np
import os
import torch

class ImageFeatureExtractor():
    def __init__(self, config):
        self.config = config

    def process_image(self, images):
        """
        return pooled or full vecteur, for mlp or attention pooling respectively.
        If all features are used, return pooled features concatenated (only for mlp model)
        """
        
        if self.config.features_path == "all":
            if self.config.pooling == 'mlp':
                b7_pooled = np.load(str(images).replace('.png', '_pooled.npy').replace('images', "features_b7")).squeeze(0)
                vgg_pooled = np.load(str(images).replace('.png', '_pooled.npy').replace('images', "features_vgg")).squeeze(0)
                frcnn = np.load(str(images).replace('.png', '.npy').replace('images', "features_frcnn")).mean(0)
                all_features = np.concatenate((vgg_pooled, b7_pooled, frcnn), axis=0)[None,:]
            else:
                b7 = np.load(str(images).replace('.png', '.npy').replace('images', "features_b7")).squeeze(0).reshape((2560,196)).transpose()
                vgg = np.load(str(images).replace('.png', '.npy').replace('images', "features_vgg")).squeeze(0).reshape((512,196)).transpose()
                frcnn = np.load(str(images).replace('.png', '.npy').replace('images', "features_frcnn")) # 2048, 36
                all_features = [vgg, b7, frcnn]
                raise NotImplementedError()
            return all_features
        elif self.config.features_path in "features_frcnn":
            feature = np.load(str(images).replace('.png', '.npy').replace('images', self.config.features_path))
            return feature.mean(0)[None,:] if self.config.pooling == 'mlp' else feature
        elif self.config.features_path in ["features_b7", "features_vgg"]:
            extension = '_pooled.npy' if self.config.pooling == 'mlp' else '.npy'
            vec = np.load(str(images).replace('.png', extension).replace('images', self.config.features_path))
            return vec if self.config.pooling == 'mlp' else vec.reshape((vec.shape[1],196)).transpose()
        else:
            raise RuntimeError(f"Unknown config value for feature path: {self.config.features_path}")
