#-*- coding: utf-8 -*-

# import torch, torchvision
# import mmseg
# import mmcv
# import mmengine
# import matplotlib.pyplot as plt
import os.path as osp
# import numpy as np
import os
import shutil
import sys

#dacon에서 다운로드 후 data폴더 저장 경로
data_path = 'open/' ##
#TRAIN_0000.png -> 0000.png와 같이 rename한 data폴더 저장할 경로
output_path = "mmsegmentation/dataset" ##

img_src = os.path.join(data_path, 'train_img/')
lab_src = os.path.join(data_path, 'train_label/')

img_output = os.path.join(output_path, 'train_img/')
lab_output = os.path.join(output_path, 'train_label/')

# 처음에만!!! 
def renameFiles(data_directory,output_directory, target_label):
    # Iterate over all files in the source directory
    for filename in os.listdir(data_directory):
        if filename.startswith(target_label) and os.path.isfile(os.path.join(data_directory, filename)):
            # Remove the 'TRAIN_' prefix from the filename
            new_filename = filename.replace(target_label, '')
            # Create the destination path with the updated filename
            source_path = os.path.join(data_directory, filename)
            destination_path = os.path.join(output_directory, new_filename)

            # Rename the file
            os.rename(source_path, destination_path)


path_now=os.getcwd()
if not os.path.exists(str(path_now)+'/mmsegmentation/dataset/train_img'):
    os.makedirs(str(path_now)+'/mmsegmentation/dataset/train_img')

if not os.path.exists(str(path_now)+'/mmsegmentation/dataset/train_label'):
    os.makedirs(str(path_now)+'/mmsegmentation/dataset/train_label')

renameFiles(img_src,img_output, 'TRAIN_') #TRAIN_0000.png -> 0000.png
renameFiles(lab_src,lab_output, 'LABEL_')
