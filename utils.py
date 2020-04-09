#-*- coding: utf-8 -*-
"""
Created on Sat Apr  4 01:17:30 2020

@author: aniket
"""
import os
import shutil
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.cluster import KMeans

try:
    import faiss
    using_faiss = True
except:
    using_faiss = False
    
    
def create_GpuIndex(d,ngpu = torch.cuda.device_count()):
    
    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpu)]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)
    
    return index

class cluster_data(Dataset):
    
    def __init__(self,labels,images):
        self.labels = labels
        self.images = images
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        return self.labels[idx],self.images[idx]
    
class cluster_saved_data(Dataset):
    
    def __init__(self,clus_id,labels):
        self.clus_id = clus_id
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        im = np.load('./clusters/cluster_' + str(self.clus_id) + '/im_' + str(idx+1) + '.npy')
        #trans = transforms.Compose([transforms.ToTensor()])
        im = torch.from_numpy(im).float()
        label = self.labels[idx]
        return label,im

def load_clusters(args,dloader,model):
    
    num_clusters = args.num_learners
    batch_sz = args.cluster_batch_sz
    num_workers = args.nb_workers
    save = args.cluster_save
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cluster_labels = [[] for _ in range(num_clusters)]
    cluster_im = [[] for _ in range(num_clusters)]
    cluster = [[] for _ in range(num_clusters)]
    num_elements_in_cluster = np.zeros(num_clusters).astype('uint')

    dloader.shuffle = False
    
    with torch.no_grad():
        
        target_labels, feature_coll = [],[]
        iterator = iter(dloader)
        
        for idx,inp in enumerate(iterator):
            input_img,target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            feature_coll.extend(out.cpu().detach().numpy().tolist())
            if(args.debug):
                if(idx==99): break
            
        target_labels = np.hstack(target_labels).reshape(-1,1)
        feature_coll  = np.vstack(feature_coll).astype('float32')
        
        if(using_faiss):
            d = feature_coll.shape[-1]
            if(args.faiss_type == 'gpu'):
                cluster_index = create_GpuIndex(d = d)
            else:
                cluster_index = faiss.IndexFlatL2(d)
                
            kmeans            = faiss.Clustering(d, num_clusters)
            kmeans.niter = 20
            kmeans.min_points_per_centroid = 1
            kmeans.max_points_per_centroid = 1000000000
    
            ### Train Kmeans
            kmeans.train(feature_coll,cluster_index)
            computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(num_clusters,d)
    
            ### Assign feature points to clusters
            if(args.faiss_type == 'gpu'):
                faiss_search_index = create_GpuIndex(d = d)
            else:
                faiss_search_index = faiss.IndexFlatL2(d)
            
            faiss_search_index.add(computed_centroids)
            _, pred_labels = faiss_search_index.search(feature_coll, 1) 
        else:
            kmeans = KMeans(n_clusters = num_clusters, random_state=0).fit(feature_coll)
            pred_labels = kmeans.labels_
        
        if(save):
            if(os.path.isdir('./clusters')):
                shutil.rmtree('./clusters')
            os.mkdir('./clusters')
            for i in range(num_clusters):
                os.mkdir('./clusters/cluster_' + str(i))
        iterator1=iter(dloader)
        for idx,inp in enumerate(iterator1):
            sz = inp[0].numpy().shape[0]
            for i in range(sz):
                clus_id = pred_labels[i + idx*sz][0]
         
                if(save):
                    num_elements_in_cluster[clus_id] += 1
                    np.save('./clusters/cluster_'+ str(clus_id) + '/im_' + str(num_elements_in_cluster[clus_id]) + '.npy', inp[1][i].numpy().astype("float16"))
                else:
                    cluster_im[clus_id].append((inp[1][i]))
                
                cluster_labels[clus_id].append((inp[0][i]))
            if(args.debug):
                if(idx==99): break
        print("num_elemnets_in_cluster:",num_elements_in_cluster)
        for i in range(num_clusters):
            if(save):
                data = cluster_saved_data(clus_id = i, labels = cluster_labels[i])
            else:
                data = cluster_data(cluster_labels[i],cluster_im[i])
            cluster[i] = DataLoader(data,batch_size = batch_sz,shuffle = True,num_workers = num_workers)
    
    dloader.shuffle = True
        
    return cluster
