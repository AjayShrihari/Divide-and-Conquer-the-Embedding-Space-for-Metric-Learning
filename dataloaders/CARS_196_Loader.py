# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:23:11 2020

@author: aniket
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, copy
from scipy.io import loadmat
import torch

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def give_CARS_dataloaders(args):
    """
    Args:
        dataset:     string, name of dataset for which the dataloaders should be returned.
        source_path: path leading to dataset folder
        arch:        network architecture used
        bs:          batch size
        nb_kernels:  number of workers for generating DataLoader
    Returns:
        dataloaders: dict of dataloaders for training, testing and evaluation on training.
    """
   
    datasets = give_CARS_datasets(args,args.dataset_path,args.arch,args.samples_per_class)

    #Move datasets to dataloaders.
    dataloaders = {}
    for key,dataset in datasets.items():
        is_val = dataset.is_validation
        dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=args.batch_sz, num_workers=args.nb_workers, shuffle=not is_val, pin_memory=True, drop_last=not is_val)

    return dataloaders

def readAnno(path):
    ann = loadmat(path)['annotations'][0]
    ret = [[] for _ in range(len(ann))]
    for idx in range(len(ann)):
        _,_,_,_,target,imgfn = ann[idx]
        r = {
        'class_id': target.item() - 1,
        'path': imgfn.item()
        }   

        ret[idx] = r
    return pd.DataFrame(ret)
    
def give_CARS_datasets(args,source_path,arch = 'resnet50',samples_per_class = 4):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the Online-Products dataset.
    For Metric Learning, training and test sets are provided by given text-files, Ebay_train.txt & Ebay_test.txt.
    So no random shuffling of classes.
    Args:
        arch: network architecture used
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    image_sourcepath  = source_path
    #Load text-files containing classes and imagepaths.
    training_files = readAnno(source_path + '/cars_train_annos.mat')
    test_files     = readAnno(source_path + '/cars_test_annos_withlabels.mat')

    #Generate Conversion dict.
    conversion = {}
    for class_id, path in zip(training_files['class_id'],training_files['path']):
        conversion[class_id] = path.split('/')[0]
    for class_id, path in zip(test_files['class_id'],test_files['path']):
        conversion[class_id] = path.split('/')[0]

    #Generate image_dicts of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict, val_image_dict  = {},{}
    for key, img_path in zip(training_files['class_id'],training_files['path']):
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath+'/cars_train/'+img_path)

    for key, img_path in zip(test_files['class_id'],test_files['path']):
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(image_sourcepath+'/cars_test/'+img_path)

    
    train_dataset       = BaseTripletDataset(args,train_image_dict, arch, samples_per_class=samples_per_class)
    val_dataset         = BaseTripletDataset(args,val_image_dict,   arch, is_validation=True)
    eval_dataset        = BaseTripletDataset(args,train_image_dict, arch, is_validation=True)

    train_dataset.conversion       = conversion
    val_dataset.conversion         = conversion
    eval_dataset.conversion        = conversion

    # return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}
    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}


class BaseTripletDataset(Dataset):
    """
    Dataset class to provide (augmented) correctly prepared training samples corresponding to standard DML literature.
    This includes normalizing to ImageNet-standards, and Random & Resized cropping of shapes 224 for ResNet50 and 227 for
    GoogLeNet during Training. During validation, only resizing to 256 or center cropping to 224/227 is performed.
    """
    def __init__(self, args, image_dict, arch = 'resnet50', samples_per_class=8, is_validation=False):
        """
        Dataset Init-Function.
        Args:
            image_dict:         dict, Dictionary of shape {class_idx:[list of paths to images belong to this class] ...} providing all the training paths and classes.
            arch:               network architecture used
            samples_per_class:  Number of samples to draw from one class before moving to the next when filling the batch.
            is_validation:      If is true, dataset properties for validation/testing are used instead of ones for training.
        Returns:
            Nothing!
        """
        #Define length of dataset
        self.args = args
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])

        self.is_validation = is_validation

        self.image_dict  = image_dict

        self.avail_classes    = sorted(list(self.image_dict.keys()))

        #Convert image dictionary from classname:content to class_idx:content, because the initial indices are not necessarily from 0 - <n_classes>.
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))

        #Init. properties that are used when filling up batches.
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            #Select current class to sample images from up to <samples_per_class>
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0

        #Data augmentation/processing methods.
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

        #Convert Image-Dict to list of (image_path, image_class). Allows for easier direct sampling.
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        #Flag that denotes if dataset is called for the first time.
        self.is_init = True


    def ensure_3dim(self, img):
        """
        Function that ensures that the input img is three-dimensional.
        Args:
            img: PIL.Image, image which is to be checked for three-dimensionality (i.e. if some images are black-and-white in an otherwise coloured dataset).
        Returns:
            Checked PIL.Image img.
        """
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):
        """
        Args:
            idx: Sample idx for training sample
        Returns:
            tuple of form (sample_class, torch.Tensor() of input image)
        """
        if self.is_init:
            self.current_class = self.avail_classes[idx%len(self.avail_classes)]
            self.is_init = False

        if not self.is_validation:
            if self.samples_per_class==1:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

            if self.n_samples_drawn==self.samples_per_class:
                #Once enough samples per class have been drawn, we choose another class to draw samples from.
                #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                #previously or one before that.
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
            if(not self.args.query):
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))
            else:
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0]))),self.image_list[idx][0]
    
    def __len__(self):
        return self.n_files

