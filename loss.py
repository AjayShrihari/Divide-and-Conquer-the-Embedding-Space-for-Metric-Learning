# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 06:19:36 2020

@author: aniket
"""

import numpy as np
import torch, itertools as it,random
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        Basic Margin Loss as proposed in 'Sampling Matters in Deep Embedding Learning'.
        Args:
            margin:          float, fixed triplet margin (see also TripletLoss).
            nu:              float, regularisation weight for beta. Zero by default (in literature as well).
            beta:            float, initial value for trainable class margins. Set to default literature value.
            n_classes:       int, number of target class. Required because it dictates the number of trainable class margins.
            beta_constant:   bool, set to True if betas should not be trained.
            sampling_method: str, sampling method to use to generate training triplets.
        Returns:
            Nothing!
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
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            margin loss (torch.Tensor(), batch-averaged)
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        if(self.sampling_type=='semihard'): triplets = semihardsampling(batch, labels, embed_num=embed_num)
        else: triplets = randomsampling(batch,labels)

        #Compute distances between anchor-positive and anchor-negative.
        if(triplets):
            d_ap, d_an = [],[]
            for triplet in triplets:
                train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}
    
                pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
                neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)
    
                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)
    
            #Group betas together by anchor class in sampled triplets (as each beta belongs to one class).
            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in triplets]).type(torch.cuda.FloatTensor)
    
            #Compute actual margin postive and margin negative loss
            pos_loss = F.relu(d_ap-beta+self.margin)
            neg_loss = F.relu(beta-d_an+self.margin)
    
            #Compute normalization constant
            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)
    
            #Actual Margin Loss
            loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count
        else:
            loss = torch.mean(batch*0)
        #(Optional) Add regularization penalty on betas.
        # if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss

class ProxyNCALoss(torch.nn.Module):
    def __init__(self, args):
        """
        Basic ProxyNCA Loss as proposed in 'No Fuss Distance Metric Learning using Proxies'.
        Args:
            num_proxies:     int, number of proxies to use to estimate data groups. Usually set to number of classes.
            embedding_dim:   int, Required to generate initial proxies which are the same size as the actual data embeddings.
        Returns:
            Nothing!
        """
        super(ProxyNCALoss, self).__init__()
        self.num_proxies   = args.num_classes
        # self.embedding_dim = args.embed_dim
        # self.PROXIES = torch.nn.Parameter(torch.randn(self.num_proxies, self.embedding_dim) / 8)
        self.all_classes = torch.arange(self.num_proxies)


    def forward(self, args, batch, labels,embed_num = -1):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            proxynca loss (torch.Tensor(), batch-averaged)
        """
        self.embedding_dim = args.embed_dim if embed_num==-1 else args.embed_dim//args.num_learners
        self.PROXIES = torch.nn.Parameter(torch.randn(self.num_proxies, self.embedding_dim) / 8)
        #Normalize batch in case it is not normalized (which should never be the case for ProxyNCA, but still).
        #Same for the PROXIES. Note that the multiplication by 3 seems arbitrary, but helps the actual training.
        batch       = 3*torch.nn.functional.normalize(batch, dim=1)
        PROXIES     = 3*torch.nn.functional.normalize(self.PROXIES, dim=1)
        #Group required proxies
        pos_proxies = torch.stack([PROXIES[pos_label:pos_label+1,:] for pos_label in labels])
        neg_proxies = torch.stack([torch.cat([self.all_classes[:class_label],self.all_classes[class_label+1:]]) for class_label in labels])
        neg_proxies = torch.stack([PROXIES[neg_labels,:] for neg_labels in neg_proxies])
        #Compute Proxy-distances
        dist_to_neg_proxies = torch.sum((batch[:,None,:]-neg_proxies.to(device)).pow(2),dim=-1)
        dist_to_pos_proxies = torch.sum((batch[:,None,:]-pos_proxies.to(device)).pow(2),dim=-1)
        #Compute final proxy-based NCA loss
        negative_log_proxy_nca_loss = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
        return negative_log_proxy_nca_loss
