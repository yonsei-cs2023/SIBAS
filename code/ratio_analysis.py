import cv2
import numpy as np
import os 
import csv
import pandas as pd

path="crop_data/crop_train_label"#"/home/fai/workspace/YCS/open/test_label"
split_path = "crop_data/splits/train3.txt"
img_list = []
with open(split_path, 'r')as f:
    lines = f.readlines()
    for line in lines:
        img_list.append(line.strip())

print(len(img_list))
# print(img_list[0]) 

f=open("crop_data/ratio_train3.csv",'w',newline="")

data=[]
data.append([])
data[0].append('img_id')
data[0].append('ratio')
#readline으로 train.txt읽어오기
i=0
for num in img_list: #os.listdir(path):
    data.append([])
    i+=1
    index=str(i-1)
    image=cv2.imread(os.path.join(path, num),cv2.IMREAD_GRAYSCALE)
    #data[i].append("Image_"+index.zfill(5))
    data[i].append(str(num))
    non_zero_pixels = np.count_nonzero(image)
    print(num)
    total_pixels = image.shape[0] * image.shape[1]
    non_zero_ratio = non_zero_pixels / total_pixels
    #print(non_zero_ratio)
    data[i].append(str(non_zero_ratio))
    # print(i)

writer=csv.writer(f)
writer.writerows(data)
f.close

# print(len(os.listdir(path)))