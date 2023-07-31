#-*- coding: utf-8 -*-

# 주피터 명령어
'''!pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# nvcc -V && gcc —version &&
!pip install -U openmim
!mim install mmengine
!pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

!rm -rf mmsegmentation
!git clone -b main https://github.com/open-mmlab/mmsegmentation.git

%cd mmsegmentation/
!pip install -e .
!pip install "mmsegmentation==1.0.0"

# Download config and checkpoint files
!mim download mmsegmentation --config segformer_mit-b0_8xb1-160k_cityscapes-1024x1024 --dest .

# Check nvcc version
!nvcc -V
# Check GCC version
!gcc --version

!pip install future tensorboard

'''
# 터미널 명령어

'''
nvcc -V && gcc --version && pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 && pip install -U openmim && mim install mmengine && pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html && cd ../mmsegmentation/ && pip install -e . && pip install "mmsegmentation==1.0.0" && pip install future tensorboard && mim download mmsegmentation --config segformer_mit-b5_8xb1-160k_cityscapes-1024x1024 --dest . && python ../code/yj_main.py

'''
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

print(os.getcwd()) #'root/code'

data_path = '../input/rename_open2/' #../input/satellite-image-segmentation/' #'../input/satellite-image-segmentation-sample/'  # Replace with the path to the directory containing the images

# img_src = os.path.join(data_path, 'train_img/')
# lab_src = os.path.join(data_path, 'train_label/')

# define dataset root and directory for images and annotations
data_root = data_path
img_dir = 'train_img' #whole train dataset
ann_dir = 'train_label'

img_dir2 = 'crop_image' #validation dataset cropped
ann_dir2 = 'crop_label'
# define class and palette for better visualization
classes = ('background', 'building')
palette = [[128, 0, 128], [255,255,0]]

#class_weight = [0.1, 0.9]



@DATASETS.register_module()
class SatelliteImageDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)


# cfg = Config.fromfile('configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py')
cfg = Config.fromfile('../code/71_ce_focal_RC_RF_PH_config.py')

# segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py
# configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py
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
    dict(type='RandomCrop', crop_size=cfg.crop_size,cat_max_ratio=0.75), #cat_max_ratio=0.75
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
cfg.train_dataloader.dataset.ann_file = '../input/rename_open2/splits/train.txt' #../splits/train.txt' #'../input/satellite-image-segmentation-sample/splits/train.txt' #'../code/splits/train.txt' #'/content/drive/MyDrive/데이터셋/splits/train.txt' #'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir2, seg_map_path=ann_dir2)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = '../input/rename_open2/splits/crop_val.txt' #'../splits/val.txt' #'../input/satellite-image-segmentation-sample/splits/val.txt'#'../code/splits/val.txt' #'/content/drive/MyDrive/데이터셋/splits/val.txt' # 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader

# Load the pretrained weights
#/content/mmsegmentation/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth
# from mmseg.apis import init_model, inference_model
checkpoint_path = '../code/71_ce_focal_RC_RF_PH_iter_40000.pth'

# model = init_model(cfg, checkpoint_path, 'cuda:0')
cfg.load_from = checkpoint_path
#cfg.load_from = 'segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
#'./checkpoints/segformer_mit-b5_8xb1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth
#cfg.load_from = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '../output/'

cfg.train_cfg.max_iters = 30000 #160000 #20000 // 총 iterations
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

