#-*- coding: utf-8 -*-
import numpy as np
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
import torch
import os
import mmcv
# from torchvision import datasets
# from tqdm.notebook import tqdm


# Train image만 따로 폴더에 넣어 만들기!!! 
# 만약, Normalizing After Splitting, BUT Before Cross Validation


def image_stats(data_path, annotationfile): #data_path
    # total_images = len(data)
    total_pixels = 0
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)

    filename_list = []
    with open(annotationfile, mode = 'r') as f:
        for line in f:
            filename_list.append(line.strip()) #\n 없애주기

    print("filename_list length is ",len(filename_list))

    # for i in range(total_images):
    for filename in filename_list: #os.listdir(data_path):
        imgfilename = str(filename) + '.png'
        img_path = os.path.join(data_path, imgfilename)
        img = mmcv.imread(img_path)
        # 이미지 데이터를 numpy 객체로 변환하고 정규화(0~1)
        # img, _ = data[i]
        img_np = np.asarray(img) / 255.0
        total_pixels += img_np.size // img_np.shape[-1]

        # 픽셀값의 합과 제곱의 합을 계산
        pixel_sum += np.sum(img_np, axis=(0, 1))
        pixel_squared_sum += np.sum(np.square(img_np), axis=(0, 1))

    # 평균과 표준편차 계산 및 반환
    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_squared_sum / total_pixels - np.square(mean))
    return mean, std

# train_mean, train_std를 validation, test에 그대로 쓴다고 한다!
train_mean, train_std = image_stats('../input/rename_open2/train_img', '../input/rename_open2/splits/train.txt')
print("Train Mean:", train_mean)
print("Train Std:", train_std)

# test_mean, test_std = image_stats('../input/rename_open/test_img')
# print("Train Mean:", test_mean)
# print("Train Std:", test_std)
