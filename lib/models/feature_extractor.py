import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F


class JAADFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(JAADFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size), 
                                        nn.ReLU()) 
    def forward(self, inputs):
        box_input = inputs
        embedded_box_input= self.box_embed(box_input)

        return embedded_box_input

class ETHUCYFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(ETHUCYFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.embed = nn.Sequential(nn.Linear(6, self.embbed_size), 
                                        nn.ReLU()) 


    def forward(self, inputs):
        box_input = inputs

        embedded_box_input= self.embed(box_input)

        return embedded_box_input

class PIEFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(PIEFeatureExtractor, self).__init__()

        self.embbed_size = args.hidden_size
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size), 
                                        nn.ReLU()) 
    def forward(self, inputs):
        box_input = inputs
        embedded_box_input= self.box_embed(box_input)
        return embedded_box_input

_FEATURE_EXTRACTORS = {
    'PIE': PIEFeatureExtractor,
    'JAAD': JAADFeatureExtractor,
    'ETH': ETHUCYFeatureExtractor,
    'HOTEL': ETHUCYFeatureExtractor,
    'UNIV': ETHUCYFeatureExtractor,
    'ZARA1': ETHUCYFeatureExtractor,
    'ZARA2': ETHUCYFeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)
