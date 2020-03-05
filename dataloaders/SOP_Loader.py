#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
import PIL.Image as Image
import pandas as pd


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im


class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


def trans(sz_resize = 256, sz_crop = (229,299), mean = [104, 117, 128],
        std = [1, 1, 1], rgb_to_bgr = True, is_train = True,
        intensity_scale = None):
    return transforms.Compose([
        RGBToBGR() if rgb_to_bgr else Identity(),
        transforms.RandomResizedCrop(sz_crop) if is_train else Identity(),
        transforms.Resize(sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(sz_crop) if not is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.ToTensor(),
        ScaleIntensities(
            *intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])



class SOPdata(torch.utils.data.Dataset):
    
    def __init__(self,txt_file,root_dir,transform = trans):
        self.root_dir = root_dir
        file = os.path.join(root_dir,txt_file)
        self.super_class_id = pd.read_csv(file,sep=" ",usecols=range(2,3))
        self.image_path = pd.read_csv(file,sep=" ",usecols=range(3,4))
        self.transform = transform
    
    def __len__(self):
        return len(self.super_class_id)
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir,self.image_path.iloc[idx,0])
        image = Image.open(img_name)
        image = self.transform(image)
        labels = self.super_class_id.iloc[idx,0]
        sample = {'image':image,'labels':labels}
        
        return sample


# trainset = SOPdata(txt_file = 'Ebay_train.txt',root_dir = 'Stanford_Online_Products')

# trainloader = torch.utils.data.DataLoader(trainset,batch_size = 10000,shuffle=True,num_workers=2)

# for i_batch,sample_batched in enumerate(trainloader,0):
#     print(i_batch,sample_batched['image'],sample_batched['labels'])

