import torch, torchvision
import mmseg
import mmcv
import mmengine
import os
import sys

# Replace with the path to the directory containing the images
data_path = '../input/'  

# define dataset root and directory for images and annotations
data_root = data_path
img_dir = 'train_img'
ann_dir = 'train_label'

img_dir2 = 'val_img'
ann_dir2 = 'val_label'

# define class and palette for better visualization
classes = ('background', 'building')
palette = [[128, 0, 128], [255,255,0]]

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class SatelliteImageDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

"""### Create a config file
In the next step, we need to modify the config for the training. To accelerate the process, we finetune the model from trained weights.
"""

from mmengine import Config
cfg = Config.fromfile('configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py')
print(f'Config:\n{cfg.pretty_text}')

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.data_preprocessor.size = cfg.crop_size
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg

b = int(os.getenv('BATCH'))
resize = str(os.getenv('RESIZE'))
loss = str(os.getenv('LOSS'))

crop = str(os.getenv('CROP'))

total_iter = int(os.getenv('ITER'))
log_intv = int(os.getenv('LOG_INTERVAL'))
val_intv = int(os.getenv('VAL_INTERVAL'))
chk_intv = int(os.getenv('CHK_INTERVAL'))

w1 = float(os.getenv('WEIGHT1'))
w2 = float(os.getenv('WEIGHT2'))
class_weight = [w1, w2]

loss_list = []

if ('C' in loss):
    loss_list.append(dict(type='CrossEntropyLoss',  use_sigmoid=False, loss_weight = 1.0, class_weight=class_weight))

if ('D' in loss):
    loss_list.append(dict(type='DiceLoss', loss_weight = 1.0, class_weight=class_weight))           

if ('F' in loss):
    loss_list.append(dict(type='FocalLoss', loss_weight = 1.0, class_weight=class_weight))           

if ('L' in loss):
    loss_list.append(dict(type='LovaszLoss', loss_weight = 1.0, reduction='none', class_weight=class_weight))
    
cfg.model.decode_head.loss_decode = loss_list
# modify num classes of the model in decode/auxiliary head

cfg.model.decode_head.num_classes = 2

# Modify dataset type and path
cfg.dataset_type = 'SatelliteImageDataset'
cfg.data_root = data_root

cfg.train_dataloader.batch_size = b
cfg.train_dataloader.num_workers = 8

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

if crop == 'T':
    cropsize = int(os.getenv('CROP_SIZE'))
    cfg.crop_size = (cropsize, cropsize)
    cfg.train_pipeline.insert(2, dict(type='RandomCrop', crop_size=cfg.crop_size))

if resize == 'T':
    cfg.train_pipeline.insert(2, dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 1.5), keep_ratio=True))


cfg.val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir2, seg_map_path=ann_dir2)
cfg.val_dataloader.dataset.pipeline = cfg.val_pipeline
cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader
cfg.vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]

cfg.visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=cfg.vis_backends,
    name='visualizer') 

# Load the pretrained weights
cfg.load_from = 'segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '../output/'

cfg.train_cfg.max_iters = total_iter
cfg.train_cfg.val_interval = val_intv
cfg.default_hooks.logger.interval = log_intv
cfg.default_hooks.checkpoint.interval = chk_intv

# Set seed to facilitate reproducing the result
cfg['randomness'] = dict(seed=0)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

"""### Train and Evaluation"""

from mmengine.runner import Runner

runner = Runner.from_cfg(cfg)

# start training
runner.train()