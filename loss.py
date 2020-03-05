#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:18:25 2020

@author: ajay
"""
import numpy as np
import torch, itertools as it,random
import torch.nn as nn

def randomsampling(self, batch, labels):
    """
    This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
    selects <len(batch)> triplets.
    Args:
        batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
        labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
    Returns:
        list of sampled data tuples containing reference indices to the position IN THE BATCH.
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
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = nn.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()