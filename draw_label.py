#-*- coding: utf-8 -*-
import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.quantization import QuantStub
from tqdm import tqdm



def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)



class SatelliteDataset(Dataset):
    def __init__(self, path_now, transform=None, infer=False):
        self.data = pd.read_csv(path_now+'/open/train.csv')
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(path_now+"/open"+img_path[1:])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask



path_now=os.getcwd()
if not os.path.exists(str(path_now)+'/open/train_label'):
    os.makedirs(str(path_now)+'/open/train_label')

if not os.path.exists(str(path_now)+'/mmsegmentation/dataset'):
    os.makedirs(str(path_now)+'/mmsegmentation/dataset')

dataset = SatelliteDataset(path_now=path_now, transform=None)
for i in range(7140):
    image,label=dataset[i]
    label_array=np.array(label)
    index=str(i)
    cv2.imwrite(path_now+"/open/train_label/LABEL_"+index.zfill(4)+".png",label)
