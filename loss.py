#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:18:25 2020

@author: ajay
"""
import numpy as np
import torch, itertools as it,random
import torch.nn as nn
import torch.nn.functional as F

def randomsampling(batch, labels):
    """
    Find triplets within a sampling of the batch
    """
    if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
    unique_classes = np.unique(labels)
    indices        = np.arange(len(batch))
    class_dict     = {i:indices[labels==i] for i in unique_classes}

    sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
    sampled_triplets = np.array(sampled_triplets).reshape(-1,3)

    sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
    sampled_triplets = [x for y in sampled_triplets for x in y]

    #NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
    sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
    return sampled_triplets

def pdist(A, eps = 1e-4):
    """
    Find the distance matrix for a given matrix
    """
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min = 0)
    return res.clamp(min = eps).sqrt()
    
def semihardsampling(batch, labels):
    """
    Find triplets within a sampling using semi hard negative mining 
    
    """
    if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
    bs = batch.size(0)
    #Return distance matrix for all elements in batch (BSxBS)
    distances = pdist(batch.detach()).detach().cpu().numpy()
    
    positives, negatives = [], []
    anchors = []
    
    for i in range(bs):
        
        l, d = labels[i], distances[i]
        # print(d,l)
        anchors.append(i)
        #1 for batchelements with label l
        neg = labels!=l; pos = labels==l
        #0 for current anchor
        pos[i] = False
        
        #Find negatives that violate triplet constraint semi-negatives
        neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
        #Find positives that violate triplet constraint semi-hardly
        pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())
        
        if pos_mask.sum()>0:
            positives.append(np.random.choice(np.where(pos_mask)[0]))
        else:
            positives.append(np.random.choice(np.where(pos)[0]))
        
        if neg_mask.sum()>0:
            negatives.append(np.random.choice(np.where(neg_mask)[0]))
        else:
            negatives.append(np.random.choice(np.where(neg)[0]))
    
    sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
    return sampled_triplets
          
class TripletLoss(nn.Module):
    """
    Function to calculate triplet loss. Finds embeddings using a set of positive, negative and anchor points 
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def distance(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum()  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum()  # .pow(.5)
        return F.relu(distance_positive - distance_negative + self.margin)
    
    def forward(self,batch,labels,sampling_type = 1):
        if(sampling_type==0): triplets = randomsampling(batch, labels)
        else: triplets = semihardsampling(batch,labels)
        loss =  torch.stack([self.distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in triplets])
        
        return torch.mean(loss)
