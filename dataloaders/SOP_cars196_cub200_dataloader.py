import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, copy
import torch

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def give_dataloaders(args, set = 'cars196'):
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
   
    # datasets = give_OnlineProducts_datasets(args.dataset_path,args.arch,args.samples_per_class)
    if set == 'cars196':
        datasets = give_CARS196_datasets(args.dataset_path,args.arch,args.samples_per_class)
    if set == 'cub200':
        datasets = give_CUB200_datasets(args.dataset_path,args.arch,args.samples_per_class)


    #Move datasets to dataloaders.
    dataloaders = {}
    for key,dataset in datasets.items():
        is_val = dataset.is_validation
        dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=args.batch_sz, num_workers=args.nb_workers, shuffle=not is_val, pin_memory=True, drop_last=not is_val)

    return dataloaders

def give_CARS196_datasets(source_path, arch = 'resnet50', samples_per_class = 4):
    """
    Dataloader for Cars 196 dataset
    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    image_sourcepath  = source_path
    #Find available data classes.
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    #Make a index-to-labelname conversion dict.
    conversion    = {i:x for i,x in enumerate(image_classes)}
    #Generate a list of tuples (class_label, image_path)
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    #Image-dict of shape {class_idx:[list of paths to images belong to this class] ...}
    image_dict    = {}
    for key, img_path in image_list:
        key = key
        # key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))

    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]
    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}

    train_dataset       = BaseTripletDataset(train_image_dict, arch, samples_per_class=samples_per_class)
    val_dataset         = BaseTripletDataset(val_image_dict,   arch, is_validation=True)
    eval_dataset        = BaseTripletDataset(train_image_dict, arch, is_validation=True)

    train_dataset.conversion = conversion
    val_dataset.conversion   = conversion
    eval_dataset.conversion  = conversion

    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}
    # return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset, 'super_evaluation':super_train_dataset}

def give_CUB200_datasets(source_path, arch = 'resnet50', samples_per_class = 4):
    """
    Function to load Cub200 dataset
    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    image_sourcepath  = source_path
    #Find available data classes.
    image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    #Make a index-to-labelname conversion dict.
    conversion    = {int(x.split('.')[0]):x.split('.')[-1] for x in image_classes}
    #Generate a list of tuples (class_label, image_path)
    image_list    = {int(key.split('.')[0]):sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    #Image-dict of shape {class_idx:[list of paths to images belong to this class] ...}
    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))

    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]
    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}


    train_dataset       = BaseTripletDataset(train_image_dict, arch, samples_per_class=samples_per_class)
    val_dataset         = BaseTripletDataset(val_image_dict,   arch, is_validation=True)
    eval_dataset        = BaseTripletDataset(train_image_dict, arch, is_validation=True)

    train_dataset.conversion = conversion
    val_dataset.conversion   = conversion
    eval_dataset.conversion  = conversion

    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}


class BaseTripletDataset(Dataset):
    """
    Dataset class to provide (augmented) correctly prepared training samples corresponding to standard DML literature.
    This includes normalizing to ImageNet-standards, and Random & Resized cropping of shapes 224 for ResNet50 and 227 for
    GoogLeNet during Training. During validation, only resizing to 256 or center cropping to 224/227 is performed.
    """
    def __init__(self, image_dict, arch = 'resnet50', samples_per_class=8, is_validation=False):
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
            return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return self.n_files