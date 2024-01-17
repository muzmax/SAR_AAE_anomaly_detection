import torch.utils.data as data
import torch
import numpy as np
import glob
import random
import os


class test_data(data.Dataset):
    """ Store the eval images (1xHxWxC) and get normalized version"""
    def __init__(self, dataset, process_func=None,label=False,nb_ch=4):
        
        self.ch = nb_ch
        self.Transform = process_func
        self.is_label = label
        if label :
            self.dataset_ = dataset[0]
            self.label_ = dataset[1]
        else :
            self.dataset_ = dataset

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, item):
        path = self.dataset_[item]
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
            
        im = np.abs(np.load(path))
        if len(im.shape) == 2:
            im = im[:,:,np.newaxis]
        else:
            shape = im.shape
            if self.ch == 2:
                im_ = np.zeros((shape[0],shape[1],2))
                im_[:,:,0] = im[:,:,0]
                im_[:,:,1] = im[:,:,3]
                im = im_
            elif self.ch == 3:
                im_ = np.zeros((shape[0],shape[1],3))
                im_[:,:,0] = im[:,:,0]
                im_[:,:,1] = 0.5*(im[:,:,1]+im[:,:,2])
                im_[:,:,2] = im[:,:,3]
                im = im_

        assert (len(im.shape) == 3)
        x = self.Transform(im)

        if self.is_label:
            label = np.load(self.label_[item])
            return x,label,name
            
        return x,torch.empty(0),name
    
class train(data.Dataset):
    """ Get a patch """
    def __init__(self, dataset, process_func=None,process_ref=None):
        self.dataset_ = dataset
        self.Transform = process_func

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, item):
        x = self.Transform(self.dataset_[item])
        return (x)

class train_RX(data.Dataset):
    """ Get a patch """
    def __init__(self, dataset, RX_map, process_func=None, process_RX = None):
        self.dataset_ = dataset
        self.score_ = RX_map
        self.Transform = process_func
        self.Transform_RX = process_RX

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, item):
        
        x = self.Transform(self.dataset_[item])
        ano = self.Transform_RX(self.score_[item])
        return (x,ano)



    
    
    
    
    
    
    
    
    