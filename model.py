#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:42:14 2020

@author: ajay
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy
import math
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch import optim
from loss import TripletLoss
import pretrainedmodels as ptm
import pretrainedmodels.utils as utils

class ResNet50(nn.Module):
    
    """
    Class for pretrained ResNet50 to create the embeddings. Can access last few layers through representation 
    """
    def __init__(self, embed_dim=128, loss = 'triplet', not_pretrained = False,list_style=False, no_norm=False):
        super(ResNet50, self).__init__()

        self.embed_dim = embed_dim
        self.loss_type = loss
        self.not_pretrained = not_pretrained

        if not self.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, embed_dim)
        
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        mod_x = self.model.last_linear(x)
        #No Normalization is used if N-Pair Loss is the target criterion.
        return mod_x if self.loss_type=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)
    
# Separate model    
#model = ResNet50()
#model.train()    
    
    
    
    
    
#def resnet_train_model(num_classes = 12):
##    num_classes = 16
#    """
#    Builds pretrained Inception v3 model, removes the last output layer, and freezes training of parameters
#    """
#    
#    resnet_model = torchvision.models.resnet18(pretrained = True, progress = True)
#    feat = resnet_model.fc.in_features
##    inception_model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
##    resnet_model.classifier=resnet_model.classifier[:-1]
#    resnet_model.fc = nn.Linear(feat, num_classes)
#    input_size = 224
#    for param in resnet_model.parameters():
#        param.requires_grad = False
#    return resnet_model 
#def model():
#    """ 
#    Builds the last layer as a combination two layers and an ELU activation. T 
#    Trains batch-wise based on L2 norm
#    """
##     model = pretrain_inception()
#    model = resnet_train_model()
#    
#    num_classes = 12
#    num_embedding = 128
#    feat = model.fc.in_features
#    model.classifier[47] = nn.Sequential(
#            nn.Linear(feat, 512),
#            nn.ELU(),
#            nn.Linear(512,num_embedding))
##    for batch in filter(
##        lambda m: type(m) == torch.nn.BatchNorm2d, model.modules()
##    ):
##        batch.eval()
##        batch.train = lambda _: None
#    return model
#def parameters(model):
#    """
#    Finds number of trainable parameters and prints
#    """
##    model = model()
#    total_params = sum(p.numel() for p in model.parameters())
#    print(f'{total_params:,} total parameters.')
#    total_trainable_params = sum(
#    p.numel() for p in model.parameters() if p.requires_grad)
#    print(f'{total_trainable_params:,} training parameters.')
#    return total_params, total_trainable_params
#    
#def toGPU(model, single = True):
#    """
#    Move to one single GPU or across two GPU's
#    """
#    if (single):
#        model = model.to('cuda')
#    else:
#        model = nn.DataParallel(model)
#
#def loss(model, loss = 'Triplet', lr = 0.0001):
#    """
#    Return loss function and criterion based on different loss functions: Offline Triplet, online triplet, Proxy NCA, Margin loss
#    """
#    if (loss == 'Triplet'):
#        loss = TripletLoss(model)
#         
#        optimizer = optim.Adam(model.parameters(), lr = lr)
#        
#    
#    return loss, optimizer

    
    
    
    
    
    

    
        
    
#inception_model = torchvision.models.inception_v3(pretrained = True, progress = True)
##print (inception_model)
#print (inception_model.fc.in_features)
