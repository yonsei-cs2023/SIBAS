import torch, torchvision
import mmseg
import mmcv
import mmengine
import matplotlib.pyplot as plt
import os.path as osp
import os
import numpy as np
from PIL import Image
import cv2
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model, show_result_pyplot


data_root = 'dataset'
img_dir = 'images'
ann_dir = 'labels'
# define class and palette for better visualization
classes = ('bg','building')
palette = [[255,0,0],[0, 0, 255]]


@DATASETS.register_module()
class StanfordBackgroundDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette,)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)


from mmengine import Config
cfg = Config.fromfile('configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py')
#print(f'Config:\n{cfg.pretty_text}')


cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.crop_size = (224, 224)
cfg.model.data_preprocessor.size = cfg.crop_size
#cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
#cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 2
#cfg.model.auxiliary_head.num_classes = 2

# Modify dataset type and path
cfg.dataset_type = 'StanfordBackgroundDataset'
cfg.data_root = data_root

cfg.train_dataloader.batch_size = 16

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 1.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=cfg.crop_size),
    dict(type='RandomFlip', prob=0.5),
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
cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader


# Load the pretrained weights
cfg.load_from = './checkpoints/segformer_mit-b5_8xb1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

#'/home/fai/workspace/YCS/mmsegmentation/work_dirs/tutorial/iter_200.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/segformer_loss4'

cfg.train_cfg.max_iters = 44625
cfg.train_cfg.val_interval = 3570
cfg.default_hooks.logger.interval = 10
cfg.default_hooks.checkpoint.interval = 1785

# Set seed to facilitate reproducing the result
cfg['randomness'] = dict(seed=0)

# Let's have a look at the final config used for training
#print(f'Config:\n{cfg.pretty_text}')


runner = Runner.from_cfg(cfg)

runner.train()
#runner.val()

# Init the model from the config and the checkpoint

checkpoint_path = '/home/fai/workspace/YCS/mmsegmentation/work_dirs/segformer_loss4/iter_44625.pth'
model = init_model(cfg, checkpoint_path, 'cuda:0')

'''
test_path="/home/fai/workspace/YCS/open/test_img"
for image_num in os.listdir(test_path):
    img = mmcv.imread(test_path+"/"+image_num)
    result = inference_model(model, img)

    label_tensor=result.pred_sem_seg.data
    label_tensor=label_tensor.cpu()

    np_arr = np.array(label_tensor, dtype=np.uint8)
    np_arr2 = np_arr[0,...]
    img2 = Image.fromarray(np_arr2)
    print(image_num)
    img2.save('/home/fai/workspace/YCS/open/test_label/'+image_num)

'''

#plt.figure(figsize=(8, 6))
#vis_result = show_result_pyplot(model, img, result)
#plt.imshow(mmcv.bgr2rgb(vis_result))



