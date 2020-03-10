#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:39:02 2020

@author: ajay
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, sys, pandas as pd, random

import torch, torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist


sys.path.insert(1, 'dataloaders/')

import SOP_Loader as loader
import loss as tripletloss
import model as net
import faiss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.00001
decay = 0.0004
tau = [30,35]
gamma = 0.3

k_vals = [1,10,100,1000]
num_epochs = 20
margin = 0.2

#source_path = '~/Academics/ECE3rdyear/3-2/CV/Project/Stanford_Online_Products'
source_path = '../Stanford_Online_Products'
dataloaders = loader.give_dataloaders(source_path)

model = net.ResNet50().to(device)

to_optim   = [{'params':model.parameters(),'lr':lr,'weight_decay':decay}]
optimizer    = torch.optim.Adam(to_optim)
scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=tau, gamma=gamma)

criterion = tripletloss.TripletLoss(margin)

def train_one_epoch(train_dataloader,model,optimizer,criterion,epoch):
    '''Training for one epoch '''
    
    losses = []
    iterator = iter(train_dataloader)
    
    for i,(class_labels,image) in enumerate(iterator):
        
        features = model(image.to(device))
        
        loss = criterion(features,class_labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        losses.append(loss.item())
        print ("Loss for epoch:",loss.item())
        if i==len(train_dataloader)-1:
#        if i==301-1: 
            print('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(losses)))
            break

def eval_one_epoch(test_dataloader,model,k_vals,epoch):
    '''Evaluating for one epoch'''
    
    torch.cuda.empty_cache()
    test_dataloader = dataloaders['testing']
    n_classes = len(test_dataloader.dataset.avail_classes)
#    n_classes= 100
    with torch.no_grad():
        target_labels, feature_coll = [],[]
        test_iter = iter(test_dataloader)
        image_paths= [x[0] for x in test_dataloader.dataset.image_list]
        for idx,inp in enumerate(test_iter):
            input_img,target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            feature_coll.extend(out.cpu().detach().numpy().tolist())
#            print (idx)
#            if(idx==150): break

        target_labels = np.hstack(target_labels).reshape(-1,1)
        feature_coll  = np.vstack(feature_coll).astype('float32')

        torch.cuda.empty_cache()
        
#        kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(feature_coll)
#        model_generated_cluster_labels = kmeans.labels_
#        computed_centroids = kmeans.cluster_centers_
#
#        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))
#        
#        k_closest_points  = squareform(pdist(feature_coll)).argsort(1)[:, :int(np.max(k_vals)+1)]
#        k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]
        cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
        kmeans            = faiss.Clustering(feature_coll.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        kmeans.train(feature_coll, cpu_cluster_index)
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, feature_coll.shape[-1])

        ### Assign feature points to clusters
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        faiss_search_index.add(computed_centroids)
        _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)

        ### Compute NMI
        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))


        ### Recover max(k_vals) nearest neighbours to use for recall computation
        faiss_search_index  = faiss.IndexFlatL2(feature_coll.shape[-1])
        faiss_search_index.add(feature_coll)
        _, k_closest_points = faiss_search_index.search(feature_coll, int(np.max(k_vals)+1))
        k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
            recall_all_k.append(recall_at_k)
        
        result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(k_vals, recall_all_k))
        print('Epoch (Test) {0}: NMI [{1:.4f}] | Recall [{2}]'.format(epoch, NMI, result_str))
    
    return NMI,recall_all_k,feature_coll

for epoch in range(num_epochs):
    '''Run the training and per epoch testing for a certain number of epochs''' 
    # print ("Epoch:",epoch)
    model.train()
    train_one_epoch(dataloaders['training'],model,optimizer,criterion,epoch)
    
    model.eval()
    NMI,_,_ = eval_one_epoch(dataloaders['testing'],model,k_vals,epoch)
    
    scheduler.step()
#save model
torch.save(model.state_dict(), './model_dict_shnm1.pth')
