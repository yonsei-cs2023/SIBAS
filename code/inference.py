#-*- coding: utf-8 -*-
import os
import mmcv
import numpy as np
from PIL import Image
import zipfile
import csv
import pandas as pd
import cv2
from mmengine import Config
# import mmseg

#71_ce_focal_RC_RF_PH
cfg = Config.fromfile('../code/71_ce_focal_RC_RF_PH_config.py')

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

from mmseg.apis import init_model, inference_model
checkpoint_path = '../code/71_ce_focal_RC_RF_PH_iter_40000.pth'

model = init_model(cfg, checkpoint_path, 'cuda:0')

def rle_encode(mask):
    non_zero_pixels=np.count_nonzero(mask)
    if(non_zero_pixels==0):
        return -1
    else:
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

output_path = '../input/test/sample_submission.csv'
# data = [['img_id', 'mask_rle']] 

data = []
data_path ="../input/test/test_img"

print("start inference")
for image_num in os.listdir(data_path):
    img_path = os.path.join(data_path, image_num)
    img = mmcv.imread(img_path)
    result = inference_model(model, img)
    label_tensor=result.pred_sem_seg.data.cpu()

    np_arr = np.array(label_tensor, dtype=np.uint8)
    img_rle = rle_encode(np_arr)
    data.append([image_num, str(img_rle)]) 
   
#data sort
data.sort(key = lambda x:x[0])
rle = []
for i in range(len(data)):
    rle.append(data[i][1])


print("making csv file")

submit = pd.read_csv('../input/test/sample_submission.csv')
submit['mask_rle'] = rle

submit.to_csv('../output/submit.csv', index=False)