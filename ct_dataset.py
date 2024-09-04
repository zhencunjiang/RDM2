from __future__ import print_function
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms,models
import matplotlib.pyplot as plt

import torch
from torch import nn
import time
import pydicom
import h5py

def ImageRescale(im, I_range):
    im_range = im.max() - im.min()
    target_range = I_range[1] - I_range[0]

    if im_range == 0:
        target = np.zeros(im.shape, dtype=np.float32)
    else:
        target = I_range[0] + target_range / im_range * (im - im.min())
    return np.float32(target)
def read_dcm_max(file):
    data = pydicom.read_file(file)
    data = np.asarray(data.pixel_array)
    data_max=data.max()
    data_min=data.min()
    # print(data_max)
    # print(data_min)
    data= ImageRescale(data,[0,1])
    # data=data/255
    data=np.float32(data)
    data=np.expand_dims(data,axis=0)
    # print(data)
    return data,data_max


def read_dcm(file):
    data = pydicom.read_file(file)
    data = np.asarray(data.pixel_array)
    data= ImageRescale(data,[0,1])
    # data=data/255
    data=np.float32(data)
    data=np.expand_dims(data,axis=0)
    # print(data)
    return data

class ctDataset(Dataset):
    def __init__(self, csv,transform=None):
        self.data=pd.read_csv(csv)

        self.transform=transform

    def __getitem__(self, index):
        noiseimgae=self.data["noiseimage"][index]
        nimg=read_dcm(noiseimgae)
        nimg=torch.tensor(nimg).type(torch.FloatTensor)
        cleanimage=self.data["cleanimage"][index]
        cimg=read_dcm(cleanimage)
        cimg=torch.tensor(cimg).type(torch.FloatTensor)
        if self.transform is not None:
            nimg = self.transform(nimg)
            cimg=self.transform(cimg)
        return nimg,cimg

    def __len__(self):
        return (self.data.shape[0])


def read_h5(file):
    f = h5py.File(file, 'r')
    x = f['data']
    y = f['label']
    x=np.asarray(x)
    y=np.asarray(y)
    x= ImageRescale(x, [0, 1])
    y = ImageRescale(y, [0, 1])
    x = np.expand_dims(x, axis=0)
    y =np.expand_dims(y,axis=0)
    return x,y

class stage2Dataset(Dataset):
    def __init__(self, dir=''):
        self.data = sorted([os.path.join(dir, name) for name in
                                  os.listdir(os.path.join(dir))])

    def __getitem__(self, index):
        nimg,cimg=read_h5(self.data[index])
        nimg = torch.from_numpy(nimg).float()
        cimg = torch.from_numpy(cimg).float()
        return nimg,cimg

    def __len__(self):
        return (len(self.data))







