# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 06:19:36 2020

@author: aniket
"""

import numpy as np
import torch, itertools as it,random
import torch.nn as nn
import torch.nn.functional as F

def randomsampling(batch, labels):
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

def pdist(A, eps = 1e-4):
    """
    Efficient function to compute the distance matrix for a matrix A.
    Args:
        A:   Matrix/Tensor for which the distance matrix is to be computed.
        eps: float, minimal distance/clampling value to ensure no zero values.
    Returns:
        distance_matrix, clamped to ensure no zero values are passed.
    """
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min = 0)
    return res.clamp(min = eps).sqrt()
    
def semihardsampling(batch, labels,embed_num=-1):
    """
    This methods finds all available triplets in a batch given by the classes provided in labels, and select
    triplets based on semihard sampling introduced in 'Deep Metric Learning via Lifted Structured Feature Embedding'.
    Args:
    batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
    labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
    Returns:
    list of sampled data tuples containing reference indices to the position IN THE BATCH.
    """
    if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
    if(len(np.unique(labels)) < 2): 
        return False
    bs = batch.size(0)
    #Return distance matrix for all elements in batch (BSxBS)
    distances = pdist(batch.detach()).detach().cpu().numpy()
    
    positives, negatives = [], []
    anchors = []
    
    for i in range(bs):
        
        l, d = labels[i], distances[i]
        # print(d,l)
        neg = labels!=l; pos = labels==l
        pos[i] = False
        if(embed_num != -1 and np.sum(pos) < 2):
            continue
        
        anchors.append(i)
        
        #Find negatives that violate triplet constraint semi-negatives
        try:
            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
        except:
            neg_mask = np.array([0])
        #Find positives that violate triplet constraint semi-hardly
        
        try:
            pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())
        except:
            pos_mask = np.array([0])
        
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
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, args):
        super(TripletLoss, self).__init__()
        self.margin = args.triplet_margin
        self.sampling_type = args.sampling_type

    def distance(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum()  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum()  # .pow(.5)
        return F.relu(distance_positive - distance_negative + self.margin)
    
    def forward(self,batch,labels,embed_num=-1):
        if(self.sampling_type=='semihard'): triplets = semihardsampling(batch, labels, embed_num=embed_num)
        else: triplets = randomsampling(batch,labels)
        if(triplets):
            loss =  torch.stack([self.distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in triplets])
        else:
            loss = batch*0
        return torch.mean(loss)
