#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:42:14 2020
Code for model used for metric learning
"""
import torch
import torch.nn as nn
import pretrainedmodels as ptm
# import pretrainedmodels.utils as utils

class ResNet50(nn.Module):
    """
    Class for pretrained Resnet 50, and dividing the embedding space and assigning the Resnet 50 learners to parts of the embedding space.
    """
    def __init__(self, args):
        super(ResNet50, self).__init__()

        self.embed_dim = args.embed_dim
        self.loss_type = args.loss_type
        self.not_pretrained = not args.pretrained
        self.nb_classes = args.num_learners

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

        # self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, embed_dim)
        
        self.k_emb = nn.ModuleList([nn.Linear(self.model.last_linear.in_features,self.embed_dim//self.nb_classes) for _ in range(self.nb_classes)])
        
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, embed_num = -1):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        # mod_x = self.model.last_linear(x)
        
        # d = self.embed_dim//self.nb_classes
        
        if (embed_num == -1):
            mod_x = []    
            for i,linear in zip(range(self.nb_classes),self.k_emb):
                mod_x.append(linear(x))
            
            mod_x = torch.cat(mod_x,dim = 1)
        else:
            mod_x = self.k_emb[embed_num](x)
        
        return mod_x if self.loss_type=='npair' else torch.nn.functional.normalize(mod_x, dim=-1)
    
# model = ResNet50()
# model.train()
# for name, child in model.named_children():
#     for name2, params in child.named_parameters():
#         print(name, name2)  
