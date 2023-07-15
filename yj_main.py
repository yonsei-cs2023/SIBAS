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

data_path = '../input/satellite-image-segmentation/' #'../input/satellite-image-segmentation-sample/'  # Replace with the path to the directory containing the images

img_src = os.path.join(data_path, 'train_img/')
lab_src = os.path.join(data_path, 'train_label/')

# 처음에만!!! (vessel은 계속)
# def renameFiles(data_directory, target_label):
#     # Iterate over all files in the source directory
#     for filename in os.listdir(data_directory):
#         if filename.startswith(target_label) and os.path.isfile(os.path.join(data_directory, filename)):
#             # Remove the 'TRAIN_' prefix from the filename
#             new_filename = filename.replace(target_label, '')
#             # Create the destination path with the updated filename
#             source_path = os.path.join(data_directory, filename)
#             destination_path = os.path.join(data_directory, new_filename)

#             # Rename the file
#             os.rename(source_path, destination_path)

# renameFiles(img_src, 'TRAIN_') #TRAIN_0000.png -> 0000.png
# renameFiles(lab_src, 'LABEL_')

# define dataset root and directory for images and annotations
data_root = data_path
img_dir = 'train_img'
ann_dir = 'train_label'
# define class and palette for better visualization
classes = ('background', 'building')
palette = [[128, 0, 128], [255,255,0]]

#class_weight = [0.1, 0.9]

# # 전체 데이터셋으로 할 때는 주석처리하기
# print("split train/val txt")
# # split train/val set randomly
# split_dir = 'splits'
# mmengine.mkdir_or_exist(osp.join(data_root, split_dir)) #'../input/satellite-image-segmentation-sample/splits'
# filename_list = [osp.splitext(filename)[0] for filename in mmengine.scandir(
#     osp.join(data_root, ann_dir), suffix='.png')]
# with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
#   # select first 4/5 as train set
#   train_length = int(len(filename_list)*4/5)
#   f.writelines(line + '\n' for line in filename_list[:train_length])
# with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
#   # select last 1/5 as train set
#   f.writelines(line + '\n' for line in filename_list[train_length:])



@DATASETS.register_module()
class SatelliteImageDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)


cfg = Config.fromfile('configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py')
# segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py
# configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py
print(f'Config:\n{cfg.pretty_text}')

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='SyncBN', requires_grad=True)
cfg.crop_size = (224,224) #(256, 256)
cfg.model.data_preprocessor.size = cfg.crop_size
#cfg.model.backbone.norm_cfg = cfg.norm_cfg #segmentor 할 때만 주석처리
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
#cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg  #segmentor 할 때만 주석처리
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 2 #8
#cfg.model.auxiliary_head.num_classes = 2 #8

# Modify dataset type and path
cfg.dataset_type = 'SatelliteImageDataset' #'StanfordBackgroundDataset'
cfg.data_root = data_root
# cfg.model.class_weight = [0.1, 0.9] #Bg, Building

cfg.train_dataloader.batch_size = 16 #8

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='RandomResize', scale=(224, 224), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = '../splits/train.txt' #'../input/satellite-image-segmentation-sample/splits/train.txt' #'../code/splits/train.txt' #'/content/drive/MyDrive/데이터셋/splits/train.txt' #'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = '../splits/val.txt' #'../input/satellite-image-segmentation-sample/splits/val.txt'#'../code/splits/val.txt' #'/content/drive/MyDrive/데이터셋/splits/val.txt' # 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader

# Load the pretrained weights
#/content/mmsegmentation/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth
cfg.load_from = 'segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
#'./checkpoints/segformer_mit-b5_8xb1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth
#cfg.load_from = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '../output/'

cfg.train_cfg.max_iters = 37500 #160000 #20000 // 총 iterations
cfg.train_cfg.val_interval = 200 #200. // validation 하는 간격
cfg.default_hooks.logger.interval = 1000 #10 # 로그 찍는 간격
cfg.default_hooks.checkpoint.interval = 2000 # 200 #checkpoint 모델 파일 .pth 저장 간격

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


# cfg.log_config.hooks = [
#     dict(type='TextLoggerHook'),
#     dict(type='TensorboardLoggerHook')]

# vis_backends = [dict(type='LocalVisBackend'),
#                 dict(type='TensorboardVisBackend')]
# visualizer = dict(
#     type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


runner = Runner.from_cfg(cfg)

print("runner.train")
# start training
runner.train()