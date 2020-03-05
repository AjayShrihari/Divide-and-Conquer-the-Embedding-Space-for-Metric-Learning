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
from torch.nn import Linear, Dropout, AvgPool2d, MaxPool2d
from torch.nn.init import xavier_normal
from torch import optim
from loss import TripletLoss

def resnet_model(num_classes = 12):
#    num_classes = 16
    """
    Builds pretrained Inception v3 model, removes the last output layer, and freezes training of parameters
    """
    
    resnet_model = torchvision.models.resnet18(pretrained = True, progress = True)
    feat = resnet_model.fc.in_features
#    inception_model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
#    resnet_model.classifier=resnet_model.classifier[:-1]
    resnet_model.fc = nn.Linear(feat, num_classes)
    input_size = 224
     for param in model.parameters():
        param.requires_grad = False
    return inception_model 
def model():
    """ 
    Builds the last layer as a combination two layers and an ELU activation. T 
    Trains batch-wise based on L2 norm
    """
#     model = pretrain_inception()
    model = resnet_model()
    
    num_classes = 12
    num_embedding = 128
    feat = resnet_model.fc.in_features
    model.classifier[47] = nn.Sequential(
            nn.Linear(feat, 512),
            nn.ELU(),
            nn.Linear(512,num_embedding))
    for batch in filter(
        lambda m: type(m) == torch.nn.BatchNorm2d, model.modules()
    ):
        batch.eval()
        batch.train = lambda _: None
    return model
def parameters(model):
    """
    Finds number of trainable parameters and prints
    """
#    model = model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params
    
def toGPU(model, single = True):
    """
    Move to one single GPU or across two GPU's
    """
    if (single):
        model = model.to('cuda')
    else:
        model = nn.DataParallel(model)

def loss(model, loss = 'Triplet', lr = 0.0001):
    """
    Return loss function and criterion based on different loss functions: Offline Triplet, online triplet, Proxy NCA, Margin loss
    """
    if (loss == 'Triplet')
        loss = TripletLoss(model)
         
        optimizer = optim.Adam(model.parameters(), lr = lr)
        
    
    return loss, optimizer

    
    
    
    
    
    

    
        
    
#inception_model = torchvision.models.inception_v3(pretrained = True, progress = True)
##print (inception_model)
#print (inception_model.fc.in_features)
