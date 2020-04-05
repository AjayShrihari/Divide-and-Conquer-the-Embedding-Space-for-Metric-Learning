# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:31:16 2020

@author: aniket
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
# import os, sys, pandas as pd, random
# from multiprocessing import freeze_support

import torch
import torch.nn as nn
# from torch.utils.data import Dataset,DataLoader
# from torchvision import transforms

# from PIL import Image
# import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist

try:
    import faiss
    using_faiss = True
    print('using faiss for clustering')
except:
    using_faiss = False
    print('using sklearn.cluster for clustering')

import loss as loss
import utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(args,train_dataloader,model,optimizer,criterion,epoch,embed_num = -1):
    
    losses = []
    iterator = iter(train_dataloader)
    for i,(class_labels,image) in enumerate(iterator):
        
        features = model(image.to(device),embed_num = embed_num)
        loss = criterion(features,class_labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        losses.append(loss.item())
        if(not args.debug):
            if i==len(train_dataloader)-1:
                print('Epoch (Train) {0} Learner {1}: Mean Loss [{2:.4f}]'.format(epoch, embed_num, np.mean(losses)))
        else:
            if i==49:
                # print(loss.item())
                print('Epoch (Train) {0} Learner {1}: Mean Loss [{2:.4f}]'.format(epoch, embed_num, np.mean(losses)))
                break

def eval_one_epoch(args,test_dataloader,model,k_vals,epoch,embed_num = -1):
    
    torch.cuda.empty_cache()
    if(not args.debug):
        n_classes = len(test_dataloader.dataset.avail_classes)
    else:
        n_classes = 10
    with torch.no_grad():
        target_labels, feature_coll = [],[]
        test_iter = iter(test_dataloader)
        # image_paths= [x[0] for x in test_dataloader.dataset.image_list]
        for idx,inp in enumerate(test_iter):
            input_img,target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device),embed_num = embed_num)
            feature_coll.extend(out.cpu().detach().numpy().tolist())
            if(args.debug):
                # print(idx)
                if(idx==49): break

        target_labels = np.hstack(target_labels).reshape(-1,1)
        feature_coll  = np.vstack(feature_coll).astype('float32')

        torch.cuda.empty_cache()
        
        if(using_faiss):
            d = feature_coll.shape[-1]
            if(args.faiss_type == 'gpu'):
                cluster_index = utils.create_GpuIndex(d = d)
            else:
                cluster_index = faiss.IndexFlatL2(d)
                
            kmeans            = faiss.Clustering(d, n_classes)
            kmeans.niter = 20
            kmeans.min_points_per_centroid = 1
            kmeans.max_points_per_centroid = 1000000000
    
            ### Train Kmeans
            kmeans.train(feature_coll,cluster_index)
            computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes,d)
    
            ### Assign feature points to clusters
            if(args.faiss_type == 'gpu'):
                faiss_search_index = utils.create_GpuIndex(d = d)
            else:
                faiss_search_index = faiss.IndexFlatL2(d)
            
            faiss_search_index.add(computed_centroids)
            _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)
    
            ### Compute NMI
            NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))
    
            ### Recover max(k_vals) nearest neighbours to use for recall computation
            if(args.faiss_type == 'gpu'):
                faiss_search_index = utils.create_GpuIndex(d = d)
            else:
                faiss_search_index  = faiss.IndexFlatL2(d)
            
            faiss_search_index.add(feature_coll)
            _, k_closest_points = faiss_search_index.search(feature_coll, int(np.max(k_vals)+1))
            k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]
            
        else:
            kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(feature_coll)
            model_generated_cluster_labels = kmeans.labels_
            # computed_centroids = kmeans.cluster_centers_
    
            NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))
            
            k_closest_points  = squareform(pdist(feature_coll)).argsort(1)[:, :int(np.max(k_vals)+1)]
            k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]

        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
            recall_all_k.append(recall_at_k)
        
        result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(k_vals, recall_all_k))
        print('Epoch (Test) {0}: NMI [{1:.4f}] | Recall [{2}]'.format(epoch, NMI, result_str))

    return NMI,recall_all_k,feature_coll


def train_embed_full(args,model,dataloaders,k_vals,optimizer,scheduler,criterion,num_epochs):
    
    for epoch in range(num_epochs):
        
        _ = model.train()
        train_one_epoch(args,dataloaders['training'],model,optimizer,criterion,epoch)
    
    
        _ = model.eval()
        NMI,_,_ = eval_one_epoch(args,dataloaders['testing'],model,k_vals,epoch)
            
        scheduler.step()


def train_embed_k(args,model,train_loader,test_loader,k_vals,optimizer,scheduler,criterion,num_epochs,embed_num):
    
    for epoch in range(num_epochs):
        
        _ = model.train()
        train_one_epoch(args,train_loader,model,optimizer,criterion,epoch,embed_num = embed_num)
    
    
        _ = model.eval()
        NMI,_,_ = eval_one_epoch(args,test_loader,model,k_vals,epoch,embed_num = -1)
            
        scheduler.step()
        

def train(args,model,dataloaders,k_vals):
    
    to_optim   = [{'params':model.parameters(),'lr':args.lr,'weight_decay':args.decay}]
    optimizer    = torch.optim.Adam(to_optim)
    tau = list(map(int,args.tau.split(',')))
    scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=tau, gamma=args.gamma)
    
    if(args.loss_type == 'tripletloss'):
        criterion = loss.TripletLoss(args)
    
    train_embed_full(args,model, dataloaders, k_vals, optimizer, scheduler, criterion, num_epochs = args.initial_epochs)
    print('Epoch {0} : Clustering data',args.initial_epochs - 1)
    train_clusters = utils.load_clusters(args,dataloaders['training'],model)
    print('Epoch {0} : Done clustering',args.initial_epochs - 1)
    if(args.debug):
        return
    
    for epoch in range(args.initial_epochs,args.num_epochs):
        
        if(epoch % args.num_T == 0):
            print('Epoch {0} : Clustering data',epoch)
            train_clusters = utils.load_clusters(args,dataloaders['training'],model)
            print('Epoch {0} : Done Clustering', epoch)
        
        for j in range(args.num_learner):
            
            _ = model.train()
            train_one_epoch(train_clusters[j],model,optimizer,criterion,epoch,embed_num = j)
        
            _ = model.eval()
            NMI,_,_ = eval_one_epoch(dataloaders['testing'],model,k_vals,epoch,embed_num = -1)
                
            scheduler.step()
    
    if(args.save_model):
        torch.save(model.state_dict(), args.model_dict_path)




        
        
        
        
        
        
        
        
        
        
        
