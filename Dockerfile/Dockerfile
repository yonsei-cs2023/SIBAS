# Use an appropriate base image with CUDA support
FROM nvidia/cuda:11.3.1-base-ubuntu20.04
FROM python:3.8
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    sudo \ 
    python3-pip \ 
    unzip \
    git \ 
    python3-opencv

# Alias python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python    

# Install PyTorch and torchvision
#RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install opencv-python pandas
RUN pip install future tensorboard

# Install OpenMIM and dependencies
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install 'mmcv >= 2.0.0rc1'

WORKDIR /

# Clone the mmsegmentation repository
RUN git clone https://github.com/open-mmlab/mmsegmentation

# Change to the mmsegmentation directory
WORKDIR /mmsegmentation

# Install mmsegmentation
RUN pip install -e .

RUN pip install "mmsegmentation==1.0.0"
