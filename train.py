#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:39:02 2020

@author: ajay
"""
from model import *
from SOP_Loader import *
source_path = '../../Stanford_Online_Products'

dataloaders = give_dataloaders(source_path)


def train():
    global source_path
    global dataloaders
    trainloader = iter(dataloaders['training'])
    for i,(class_labels,image) in enumerate(trainloader,0):
        
    
def test():
    