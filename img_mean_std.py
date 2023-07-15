import numpy as np
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
import torch
import os
import mmcv
# from torchvision import datasets
# from tqdm.notebook import tqdm

def image_stats(data_path):
    # total_images = len(data)
    total_pixels = 0
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)

    # for i in range(total_images):
    for filename in os.listdir(data_path):
        img_path = os.path.join(data_path, filename)
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
    
    
#train_data = datasets.CocoDetection('train데이터셋의 위치', 'instances_train.json파일의 위치')
#test_data = datasets.CocoDetection('test데이터셋의 위치', 'instances_test.json파일의 위치')

mean, std = image_stats('/content/drive/MyDrive/데이터셋/train_img')
print("Mean:", mean)
print("Std:", std)

#pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 && pip install -U openmim && mim install mmengine && pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html && cd ../mmsegmentation/ && pip install -e . && pip install "mmsegmentation==1.0.0" && pip install future tensorboard && mim download mmsegmentation --config segformer_mit-b5_8xb1-160k_cityscapes-1024x1024 --dest . && python ../code/yj_main.py