# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:57:12 2020

@author: aniket
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, sys, pandas as pd, csv, copy
import torch, torch.nn as nn, matplotlib.pyplot as plt, random

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# enable for Windows
# from multiprocessing import freeze_support

def give_dataloaders(source_path, arch = 'resnet50', bs = 80, nb_kernels = 0,samples_per_class = 4):
    """
    Create a pytorch library comaptible dataloader for our code
    """
   
    datasets = give_OnlineProducts_datasets(source_path,arch,samples_per_class)

    #Move datasets to dataloaders.
    dataloaders = {}
    for key,dataset in datasets.items():
        is_val = dataset.is_validation
        dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=nb_kernels, shuffle=not is_val, pin_memory=True, drop_last=not is_val)

    return dataloaders


def give_OnlineProducts_datasets(source_path,arch = 'resnet50',samples_per_class = 4):
    """
    Generates training, testing, validation data for the given Stanford Online products dataset in a dictionary
    """
    image_sourcepath  = source_path
    
    training_files = pd.read_table(source_path+'/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(source_path+'/Ebay_test.txt', header=0, delimiter=' ')

    
    conversion = {}
    for class_id, path in zip(training_files['class_id'],training_files['path']):
        conversion[class_id] = path.split('/')[0]
    for class_id, path in zip(test_files['class_id'],test_files['path']):
        conversion[class_id] = path.split('/')[0]

    
    train_image_dict, val_image_dict  = {},{}
    for key, img_path in zip(training_files['class_id'],training_files['path']):
        key = key-1
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(test_files['class_id'],test_files['path']):
        key = key-1
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(image_sourcepath+'/'+img_path)

    
    super_train_image_dict,super_conversion = {},{}
    for super_class_id, path in zip(training_files['super_class_id'],training_files['path']):
        conversion[super_class_id] = path.split('/')[0]
    for key, img_path in zip(training_files['super_class_id'],training_files['path']):
        key = key-1
        if not key in super_train_image_dict.keys():
            super_train_image_dict[key] = []
        super_train_image_dict[key].append(image_sourcepath+'/'+img_path)
    super_train_dataset = BaseTripletDataset(super_train_image_dict, arch, is_validation=True)
    super_train_dataset.conversion = super_conversion

    
    train_dataset       = BaseTripletDataset(train_image_dict, arch, samples_per_class=samples_per_class)
    val_dataset         = BaseTripletDataset(val_image_dict,   arch, is_validation=True)
    eval_dataset        = BaseTripletDataset(train_image_dict, arch, is_validation=True)

    train_dataset.conversion       = conversion
    val_dataset.conversion         = conversion
    eval_dataset.conversion        = conversion

    # return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}
    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset, 'super_evaluation':super_train_dataset}


class BaseTripletDataset(Dataset):
    """
    Data augment class for mormalizing and resizing images to 224 for ResNet50 and imagenet standards
    """
    def __init__(self, image_dict, arch = 'resnet50', samples_per_class=8, is_validation=False):
       
        
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])

        self.is_validation = is_validation

        self.image_dict  = image_dict

        self.avail_classes    = sorted(list(self.image_dict.keys()))

        
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))

        
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0

        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        if not self.is_validation:
            transf_list.extend([transforms.RandomResizedCrop(size=224) if arch=='resnet50' else transforms.RandomResizedCrop(size=224),
                                transforms.RandomHorizontalFlip(0.5)])
        else:
            transf_list.extend([transforms.Resize(256),
                                transforms.CenterCrop(224) if arch=='resnet50' else transforms.CenterCrop(224)])

        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]


        self.is_init = True


    def ensure_3dim(self, img):
        """
        Ensure that the image is a 3D matrix
        """
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):
        
        if self.is_init:
            self.current_class = self.avail_classes[idx%len(self.avail_classes)]
            self.is_init = False

        if not self.is_validation:
            if self.samples_per_class==1:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

            if self.n_samples_drawn==self.samples_per_class:
               
                counter = copy.deepcopy(self.avail_classes)
                for prev_class in self.classes_visited:
                    if prev_class in counter: counter.remove(prev_class)

                self.current_class   = counter[idx%len(counter)]
                self.classes_visited = self.classes_visited[1:]+[self.current_class]
                self.n_samples_drawn = 0

            class_sample_idx = idx%len(self.image_dict[self.current_class])
            self.n_samples_drawn += 1

            out_img = self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
            return self.current_class,out_img
        else:
            return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return self.n_files





#source_path = '../../Stanford_Online_Products'
#
#dataloaders = give_dataloaders(source_path)

#trainloader = iter(dataloaders['training'])
#
#def main():
#     for i,(class_labels,image) in enumerate(trainloader,0):
#         print('i:',i,'labels:',class_labels)
#
#if __name__ == '__main__':
#     main()

# if __name__ == '__main__':
#     freeze_support() # enable for multiprocessing in Windows
#     source_path = 'Stanford_Online_Products'
#     dataloaders = give_dataloaders(source_path)
#     trainloader = iter(dataloaders['training'])


#     for i,(class_labels,_) in enumerate(trainloader,0):
#         print('i:',i,'labels:',class_labels)

