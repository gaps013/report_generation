import torch
import os
from torch import nn

class FeatureExtraction(nn.Module):
    def __init__(self, model):
        super(FeatureExtraction, self).__init__()

        modules = list(model.children())[:-3]
        # print(modules[-1][14])
        # self.visual_feature_size = modules[-1].size(0)
        self.model = nn.Sequential(*modules)
    def forward(self, image):
        features = self.model(image)
        features = features.reshape((features.shape[0], features.shape[1]))
        return features
    # @property
    # def visual_feature_size(self) -> int:
    #     r"""
    #     Size of the channel dimension of output from forward pass. This
    #     property is used to create layers (heads) on top of this backbone.
    #     """
    #     return self._visual_feature_size