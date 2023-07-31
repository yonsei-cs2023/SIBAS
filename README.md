# Model Training and Inference with MMsegmentation

This repository contains code for training and inference of semantic segmentation models using [MMsegmentation](https://github.com/open-mmlab/mmsegmentation), a powerful and efficient toolbox developed by the OpenMMLab team. 

## Library Requirements

To ensure smooth setup and execution, make sure you have Docker installed and use the provided Dockerfile. The Dockerfile includes all the necessary library requirements for running the code:

- opencv-python
- pandas
- future
- tensorboard
- openmim
- mmengine
- mmcv (version >= 2.0.0rc1)
- torch (version == 2.0.0+cu118)
- torchvision (version == 0.15.1+cu118)
- torchaudio (version == 2.0.1)
- mmsegmentation (version == 1.0.0)

## Setup Instructions

1. Clone this repository to your local machine.
2. Make sure you have Docker installed and run the Dockerfile provided to install all the required libraries.
3. Download the configuration files for the models you want to use from MMsegmentation:

   For Segformer model:

```mim download mmsegmentation --config segformer_mit-b5_8xb1-160k_cityscapes-1024x1024 --dest .```

   For UNet model:

```mim download mmsegmentation --config unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024 --dest .```


4. Configure the data paths for your dataset in the model config files:

- `data_path`: The main directory where your dataset is located.
- `img_dir`: The directory containing the training images.
- `ann_dir`: The directory containing the training labels.
- `img_dir2`: The directory containing the validation images.
- `ann_dir2`: The directory containing the validation labels.

5. Make sure you set the necessary environment variables for your training settings. The model config files are designed to read parameters from the OS environment variables, but you can modify them as needed.

6. Execute the model python file inside model folder to start training 


## Inference

For inference, you need to prepare the model config file and the checkpoint file. Set the test data path as the data path for inference:

1. Prepare the model config file with the required settings and the checkpoint file for your trained model.

2. Run the inference.py script
