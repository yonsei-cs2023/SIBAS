#-*- coding: utf-8 -*-
import os
from PIL import Image

def crop_and_save_image_label(image_path, label_path, output_dir, num, x, y, size):
    image = Image.open(image_path)
    label = Image.open(label_path)
    num, tp = num.split(".")

    for i in range(5):
        for j in range(5):
            left = x * i
            upper = y * j
            right = left + size
            lower = upper + size

            cropped_image = image.crop((left, upper, right, lower))
            cropped_label = label.crop((left, upper, right, lower))

            cropped_image.save(f"{output_dir}/crop_image/{num}_{i * 5 + j + 1}.png")
            cropped_label.save(f"{output_dir}/crop_label/{num}_{i * 5 + j + 1}.png")

def main(path_now):
    
    input_image_dir = path_now+"/mmsegmentation/dataset/train_img"
    input_label_dir = path_now+"/mmsegmentation/dataset/train_label"
    output_dir = path_now+"/mmsegmentation/dataset"

    nu = 0
    for image_num in os.listdir(input_image_dir):
        image_path = os.path.join(input_image_dir, image_num)
        label_path = os.path.join(input_label_dir, image_num)

        crop_and_save_image_label(image_path, label_path, output_dir, image_num, 200, 200, 224)

        nu += 1
        print(nu)

if __name__ == "__main__":
    path2=os.getcwd()
    if not os.path.exists(str(path2)+'/mmsegmentation/dataset/crop_image'):
      os.makedirs(str(path2)+'/mmsegmentation/dataset/crop_image')
    if not os.path.exists(str(path2)+'/mmsegmentation/dataset/crop_label'):
      os.makedirs(str(path2)+'/mmsegmentation/dataset/crop_label')  
    main(path2)
