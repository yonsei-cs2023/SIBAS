#!/bin/bash


python3 ./draw_label.py
python3 ./rename.py
python3 ./data_crop.py

#./deidentification.sh value1 value2
#$1:value1(video_name)
#$2:value2(type)
#type 1:blur_only
#type 2:blur_edge
#type 3:blur_only + blur_edge