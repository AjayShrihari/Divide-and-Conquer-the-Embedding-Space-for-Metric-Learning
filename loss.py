# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 06:19:36 2020
Code for sampling: random, semihard negative mining sampling.
Code for Triplet Loss, Margin Loss and Proxy NCA loss
"""

import numpy as np
import torch, itertools as it,random
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def randomsampling(batch, labels):
    """
    Method for randomly sampling a set of triplets in a given batch, given the class labels.
    Returns:
    A list of triplets 
    """
    if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
    unique_classes = np.unique(labels)
    indices        = np.arange(len(batch))
    class_dict     = {i:indices[labels==i] for i in unique_classes}

    sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
    sampled_triplets = np.array(sampled_triplets).reshape(-1,3)
    sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
    sampled_triplets = [x for y in sampled_triplets for x in y]
    sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
    return sampled_triplets

def pdist(A, eps = 1e-4):
    """
    Method for finding a distance matrix, given a matrix A
    """
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min = 0)
    return res.clamp(min = eps).sqrt()
    
def semihardsampling(batch, labels,embed_num=-1):
    """
    Method to find a set of triplets in a given batch using semi hard negative mining
    Returns:
    List of triplets
    """
    if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
    if(len(np.unique(labels)) < 2): 
        return False
    bs = batch.size(0)
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
        
        #Find negatives points that violate constraint
        try:
            neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
        except:
            neg_mask = np.array([0])
        #Find positives that violate constraint
        
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
    Takes embeddings of an anchor sample, a positive sample and a negative sample.
    """

    def __init__(self, args):
        super(TripletLoss, self).__init__()
        self.margin = args.triplet_margin
        self.sampling_type = args.sampling_type

    def distance(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum()  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum()  # .pow(.5)
        return F.relu(distance_positive - distance_negative + self.margin)
    
    def forward(self, args, batch,labels,embed_num=-1):
        if(self.sampling_type=='semihard'): triplets = semihardsampling(batch, labels, embed_num=embed_num)
        else: triplets = randomsampling(batch,labels)
        if(triplets):
            loss =  torch.stack([self.distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in triplets])
        else:
            loss = batch*0
        return torch.mean(loss)

class MarginLoss(torch.nn.Module):
    def __init__(self, args, beta_constant=False):
        """
        Class for margin loss.
        beta - boundary between positive and negative
        alpha - denotes margin of separation
        """
        super(MarginLoss, self).__init__()
        self.margin             = args.triplet_margin
        self.n_classes          = args.num_classes
        self.beta_constant     = beta_constant

        self.beta_val = args.margin_beta
        self.beta     = self.beta_val if beta_constant else torch.nn.Parameter(torch.ones(self.n_classes)*self.beta_val)

        self.sampling_type            = args.sampling_type


    def forward(self, args, batch, labels,embed_num=-1):
        """
        Do a forward pass on the triplets using random sampling and semihard sampling
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        if(self.sampling_type=='semihard'): triplets = semihardsampling(batch, labels, embed_num=embed_num)
        else: triplets = randomsampling(batch,labels)

        #Compute distances between anchor-positive and anchor-negative points in te triplet.
        if(triplets):
            d_ap, d_an = [],[]
            for triplet in triplets:
                train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}
    
                pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
                neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)
    
                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)
    
            
            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in triplets]).type(torch.cuda.FloatTensor)
    
            
            pos_loss = F.relu(d_ap-beta+self.margin)
            neg_loss = F.relu(beta-d_an+self.margin)
    
            
            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)
    
            #Compute Margin Loss using the formula
            loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count
        else:
            loss = torch.mean(batch*0)
        

        return loss

class ProxyNCALoss(torch.nn.Module):
    def __init__(self, args):
        """
        Class for 
        """
        super(ProxyNCALoss, self).__init__()
        self.num_proxies   = args.num_classes
        # self.embedding_dim = args.embed_dim
        # self.PROXIES = torch.nn.Parameter(torch.randn(self.num_proxies, self.embedding_dim) / 8)
        self.all_classes = torch.arange(self.num_proxies)


    def forward(self, args, batch, labels,embed_num = -1):
        """
        Do a forward 
        """
        self.embedding_dim = args.embed_dim if embed_num==-1 else args.embed_dim//args.num_learners
        self.PROXIES = torch.nn.Parameter(torch.randn(self.num_proxies, self.embedding_dim) / 8)
        batch       = 3*torch.nn.functional.normalize(batch, dim=1)
        PROXIES     = 3*torch.nn.functional.normalize(self.PROXIES, dim=1)
        #Set proxies for positive and negative
        pos_proxies = torch.stack([PROXIES[pos_label:pos_label+1,:] for pos_label in labels])
        neg_proxies = torch.stack([torch.cat([self.all_classes[:class_label],self.all_classes[class_label+1:]]) for class_label in labels])
        neg_proxies = torch.stack([PROXIES[neg_labels,:] for neg_labels in neg_proxies])
        
        dist_to_neg_proxies = torch.sum((batch[:,None,:]-neg_proxies.to(device)).pow(2),dim=-1)
        dist_to_pos_proxies = torch.sum((batch[:,None,:]-pos_proxies.to(device)).pow(2),dim=-1)
        negative_log_proxy_nca_loss = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
        return negative_log_proxy_nca_loss
