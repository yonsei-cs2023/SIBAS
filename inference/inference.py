import os
import mmcv
import numpy as np
from PIL import Image
import zipfile
import csv
import pandas as pd
import cv2
from mmengine import Config

config_file = os.getenv('CONFIG_FILENAME')
chkpt_file = os.getenv('CHECKPOINT_FILENAME')

config_path = os.path.join('../config', config_file)
cfg = Config.fromfile(config_path)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

from mmseg.apis import init_model, inference_model
checkpoint_path = os.path.join('../config', chkpt_file)

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

output_path = '../output/submission.csv'
data = [['img_id', 'mask_rle']] 

data_path ="../input/"

for image_num in os.listdir(data_path):
    img_path = os.path.join(data_path, image_num)
    img = mmcv.imread(img_path)
    result = inference_model(model, img)
    label_tensor=result.pred_sem_seg.data.cpu()

    np_arr = np.array(label_tensor, dtype=np.uint8)
    img_rle = rle_encode(np_arr)
    data.append([image_num, str(img_rle)]) 
   
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)