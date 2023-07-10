import csv
import numpy as np
import os 
import pandas as pd
import cv2
import matplotlib.pyplot as plt

'''

def rle_encode(mask):
    mask_gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    non_zero_pixels=np.count_nonzero(mask_gray)
    if(non_zero_pixels==0):
        return -1
    else:
        pixels = mask_gray.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)


path="/home/fai/workspace/YCS/open/test_label"
image_files=sorted(os.listdir(path))

f=open('YCS.csv','w',newline='')

data=[]
data.append([])
data[0].append('img_id')
data[0].append('mask_rle')

j=0
for image_file in image_files:
    j+=1
    data.append([])
    index=str(j-1)
    data[j].append("TEST_"+index.zfill(5))
    image_path = os.path.join(path, image_file)
    mask=cv2.imread(image_path)
    name=rle_encode(mask)
    data[j].append(str(name))
    print(j)
    
   
writer=csv.writer(f)
writer.writerows(data)
f.close

'''
path="/home/fai/workspace/YCS/open/test_label"
for image_file in os.listdir(path):
    mask=cv2.imread(path+"/"+image_file)
    res,thr=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    cv2.imwrite("/home/fai/workspace/YCS/open/test_see/"+image_file,thr)

