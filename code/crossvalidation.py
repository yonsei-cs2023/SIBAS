#-*- coding: utf-8 -*-
import os
# import shutil, random
from random import shuffle
import torch
from PIL import Image
import numpy as np
import mmcv
from mmengine.registry import init_default_scope
from PIL import Image
from mmseg.datasets.transforms import *  # noqa
from mmseg.datasets.transforms import RandomCrop
from mmseg.registry import TRANSFORMS
from tqdm import tqdm
import time
# Library
import torch, torchvision
import mmseg
import mmcv
import mmengine
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import os
import shutil
import sys

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine import Config
from mmengine.runner import Runner #mmengine 봐보기!!!

init_default_scope('mmseg')

torch.manual_seed(0)

#1. Make Crop_Train/Val dataset, crop_train/val{i}.txt - just once(DONE)

train_img_dir = "rename_open2/train_img"
train_label_dir = 'rename_open2/train_label'
split_dir = 'output/splits' #"../input/rename_open2/splits" #.txt files
crop_train_img_dir ='output/crop_train_image'#"../input/rename_open2/train_img"
crop_train_label_dir = 'output/crop_train_label'#"../input/rename_open2/train_img"
os.mkdir(crop_train_img_dir)
os.mkdir(crop_train_label_dir)
os.mkdir(split_dir)

def pipeline_randomcrop(img_num, n_crops): #img, seg...? path -> return results,,,? 아니면 저장
  
  img_path = os.path.join(train_img_dir, img_num)
  seg_path = os.path.join(train_label_dir, img_num)
  img = np.array(Image.open(img_path))
  seg = np.array(Image.open(seg_path))
  results = dict()
  results['img'] = img
  results['gt_seg_map'] = seg
  results['seg_fields'] = ['gt_seg_map']
  results['img_shape'] = img.shape
  results['ori_shape'] = img.shape

  pipeline = RandomCrop(crop_size=(224,224))
  crop_results = pipeline(results)
  crop_img = Image.fromarray(crop_results['img']) # NumPy array to PIL image
  crop_seg = Image.fromarray(crop_results['gt_seg_map'])
  num = img_num[:-4]
  filename = str(num)+'_' + str(n_crops) + '.png'
  crop_img.save(os.path.join(crop_train_img_dir,filename))
  crop_seg.save(os.path.join(crop_train_label_dir, filename))

# filenamelist = os.listdir(train_img_dir)

# print(len(filenamelist))
# filenamelist.sort()

# sub_filenamelist = filenamelist[3016:]
# print("start cropping")
# for file in tqdm(sub_filenamelist, desc='file', position=0):
#   time.sleep(0.01)
#   for i in tqdm(range(25), desc='num', position=1, leave=False):
#     time.sleep(0.01)
#     pipeline_randomcrop(file, i)
  
# print("end cropping")
# outputlist = os.listdir(crop_train_img_dir)
# print(len(outputlist)) #178501

# outputlist2 = os.listdir(crop_train_label_dir)
# print(len(outputlist2)) #178501


# #2. Split K-fold (train, val(need to be cropped)) - just once
K = 4 #7140*25 = 178500 -> 44625 * 4 // 133875 (train), 44625(val)

train_splits= [] # = ['train1.txt', 'train2.txt', 'train3.txt', 'train4.txt']
val_splits = [] #['val1.txt', 'val2.txt', 'val3.txt', 'val4.txt']
for i in range(K):
    train_filename = 'train' + str(i+1) + '.txt'
    val_filename = 'val' + str(i+1) + '.txt'
    train_splits.append(train_filename)
    val_splits.append(val_filename)
# print(train_splits)
# print(val_splits)

# #split K(=4) folds randomly
# filenamelist = os.listdir(crop_train_img_dir)
# print(len(filenamelist)) #178500
# # print(filenamelist[0])
# shuffle(filenamelist)

# # print(filenamelist[0])
# for i in range(K):
#     #validation.txt
#     with open(os.path.join(split_dir, val_splits[i]), 'w') as f:
#         #select 1/K
#         kfold_length = int(len(filenamelist)*1/K) #44625
#         print("kfold_len", kfold_length)
#         f.writelines(line + '\n' for line in filenamelist[i*kfold_length:i*kfold_length+kfold_length])
#     #train.txt
#     with open(os.path.join(split_dir, train_splits[i]), 'w') as f:
#         trainset = filenamelist[0:i*kfold_length]
#         trainset.extend(filenamelist[i*kfold_length+kfold_length:])
#         print("train_len", len(trainset)) #133875
#         f.writelines(line + '\n' for line in trainset)

# #3. Train K models - save train_loss, val_mIoU,accuracy using Runner.Train

#DATASET path - after unzip
split_dir = 'crop_data/splits' #"../input/rename_open2/splits" #.txt files
crop_train_img_dir ='crop_data/crop_train_image'#"../input/rename_open2/train_img"
crop_train_label_dir = 'crop_data/crop_train_label'#"../input/rename_open2/train_img"

data_path = '../input/crop_data/' #../input/satellite-image-segmentation/' #'../input/satellite-image-segmentation-sample/'  # Replace with the path to the directory containing the images

# img_src = os.path.join(data_path, 'train_img/')
# lab_src = os.path.join(data_path, 'train_label/')

# define dataset root and directory for images and annotations
data_root = data_path
img_dir = 'crop_train_image' #whole train dataset
ann_dir = 'crop_train_label'

# define class and palette for better visualization
classes = ('background', 'building')
palette = [[128, 0, 128], [255,255,0]]

#class_weight = [0.1, 0.9]

@DATASETS.register_module()
class SatelliteImageDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)


cfg = Config.fromfile('configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py')

print(f'Config:\n{cfg.pretty_text}')

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='SyncBN', requires_grad=True)
cfg.crop_size = (224,224) #(256, 256) #(156,156) or (112,112)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 2 #8

# Modify dataset type and path
cfg.dataset_type = 'SatelliteImageDataset' #'StanfordBackgroundDataset'
cfg.data_root = data_root
# cfg.model.class_weight = [0.1, 0.9] #Bg, Building

class_weight = [0.5, 1.0] 
loss_list = []
loss_list.append(dict(type='CrossEntropyLoss',  use_sigmoid=False, loss_weight = 1.0, class_weight=class_weight))
# loss_list.append(dict(type='DiceLoss', loss_weight = 1.0, class_weight=class_weight))           
loss_list.append(dict(type='FocalLoss', loss_weight = 1.0))           
#loss_list.append(dict(type='LovaszLoss', loss_weight = 1.0, reduction='none', class_weight=class_weight))
cfg.model.decode_head.loss_decode = loss_list

cfg.train_dataloader.batch_size = 16 #8
cfg.train_dataloader.num_workers = 8

cfg.img_norm_cfg = dict(
    mean=[0.32519105, 0.35761357, 0.34220385], std=[0.16558432, 0.17289196, 0.19330389], to_rgb=True
    #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='RandomResize', scale=(224, 224), ratio_range=(0.5, 2.0), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=cfg.crop_size,cat_max_ratio=0.75), #cat_max_ratio=0.75
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    #dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='Resize', scale=(224, 224), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    #dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = '../input/crop_data/splits/train4.txt' #../splits/train.txt' #'../input/satellite-image-segmentation-sample/splits/train.txt' #'../code/splits/train.txt' #'/content/drive/MyDrive/데이터셋/splits/train.txt' #'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = '../input/crop_data/splits/val4.txt' #'../splits/val.txt' #'../input/satellite-image-segmentation-sample/splits/val.txt'#'../code/splits/val.txt' #'/content/drive/MyDrive/데이터셋/splits/val.txt' # 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader

# Load the pretrained weights
#/content/mmsegmentation/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth
cfg.load_from = 'segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
#'./checkpoints/segformer_mit-b5_8xb1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth
#cfg.load_from = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '../output/'

cfg.train_cfg.max_iters = 140000 #160000 #20000 // 총 iterations 140K
cfg.train_cfg.val_interval = 1000 #200. // validation 하는 간격
cfg.default_hooks.logger.interval = 1000 #10 # 로그 찍는 간격
cfg.default_hooks.checkpoint.interval = 1000 # 200 #checkpoint 모델 파일 .pth 저장 간격

# cfg.default_hooks.visualization.draw = True
# cfg.default_hooks.visualization.interval = 1

# cfg.log_config.interval = 1 #10 AttributeError: 'ConfigDict' object has no attribute 'log_config'
# Set seed to facilitate reproducing the result
cfg['randomness'] = dict(seed=0)

# We can also use tensorboard to log the training process
cfg.vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
cfg.visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=cfg.vis_backends,
    name='visualizer')

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

runner = Runner.from_cfg(cfg)

print("runner.train")
# start training
runner.train()

print("training end")

sys.exit(0)

# # #4. Load scalar values from csv file(<- Tensorboard)


# # #5. Calculate the final average Performance score


# # #6. Make eventfile or graph showing the average trend



