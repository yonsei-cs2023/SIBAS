# SIBAS
Satellite Image Building Area Segmentation

# 0. Library Requirements

- opencv-python (Version == 4.8.0)
- pandas (Version == 1.3.5)
- future  (Version == 0.18.3)
- tensorboard (Version == 2.11.2)
- openmim (Version == 0.3.9)
- mmengine (Version == 0.8.2 )
- mmcv (version >= 2.0.0rc1)
- torch (version == 2.0.0+cu118)
- torchvision (version == 0.15.1+cu118)
- torchaudio (version == 2.0.1)
- mmsegmentation (version == 1.0.0)

## 0. 개발 환경

```markdown

1.
개발 환경: 
GPU(NVIDIA-GeForce-GTX-1080Ti) 1개 사용
conda)
Python 3.7.4
CUDA 12.0(nvidia-smi 기준)
PyTorch 2.0.1+cu118
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 10.1, V10.1.243

+ gcc --version
gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

2.
개발 환경:
GPU(NVIDIA-GeForce-RTX-3060) 1개 사용
conda)
Python 3.7.16
CUDA 12.2(nvidia-smi 기준)
PyTorch 1.12.0
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
	
+ gcc --version
gcc (Ubuntu 11.3.0-1ubuntu1~22.04.1) 11.3.0

3.
개발 환경:
GPU GPU(NVIDIA-GeForce-RTX-3090) 1개 사용
Python 3.8.10
CUDA 11.8.0
PyTorch 2.0.0+cu118
nvcc: cuda compilation tools, release 11.8, v11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
	
+ gcc --version
gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
```

## 0.  사용한 pretrained 모델 출처

```markdown
Method: Segformer
Backbone: MIT-B5
traindata: Cityscapes

#pretrained model 출처: https://github.com/open-mmlab/mmsegmentation
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'

#segformer 논문 출처
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```

## 1. 데이터셋과 필요 코드, 모델 weight파일 준비

데이콘에서 다운로드 후 압축 푼 open 폴더 구조는 다음과 같다. 

```markdown
	open
  	├── train_img
	│   ├── TRAIN_0000.png
	│   ├── ...
	│   └── TRAIN_7139.png
	├── test_img
	│   ├── TEST_00000.png
	│   ├── ...
	│   └── TEST_60639.png
	├── train.csv
	├── test.csv
  	└── sample_submission.csv
```

제출한 zipfile 압축을 풀면 구조는 다음과 같다.

```markdown
	YCS
  	├── data_preprocessing
	│   ├── splits                        # txt files with train/val filenames
	│   ├── data_crop.py
	│   ├── draw_label.py
	│   ├── mv_splits.py
	│   ├── rename.py
	│   └── preprocessing.sh
	├── train
	│   ├── checkpoints                   # model weights
	│   ├── config_ABC                    # model configuration python files
	│   ├── inference.py
	│   ├── inference_after_train.py
	│   ├── trainA.py
	│   ├── trainB.py
	│   └── trainC.py
	├── readme.md
  	└── requirements.txt
```

## 2. 라이브러리 등 필요 환경 구축  (requirements.txt)

1. **필요한 라이브러리를 requirements.txt를 이용해 다운 받는다.**
    

```python
conda create -n "가상환경 명" python=3.7 -y
conda activate "가상환경 명"
pip install -r requirements.txt
```

명령어 실행 후 실행된 경로에 src 폴더가 생기고 그 안에 open-mmlab의 mmsegementation이 git clone된다.

1-2. **만약 mmcv 모듈 관련 오류가 생긴다면 아래의 명령어를 터미널에 입력한다.**

```python
pip install mmcv==2.0.0
pip install tensorboard
```

2. **만약 requirements.txt를 install 했을 때 버전 충돌이 있다면 아래의 명령어를 터미널에 순서대로 입력한다.**

#1) CUDA 11.7 version
```python
conda create -n mmseg python=3.7 -y
conda activate mmseg
pip3 install torch torchvision torchaudio
pip install -U openmim
mim install mmengine
mim install 'mmcv >= 2.0.0rc1'
rm -rf mmsegmentation
git clone -b main https://github.com/open-mmlab/mmsegmentation.git 
cd mmsegmentation
pip install -e .
pip install tensorboard
```

#2) CUDA 11.8.0 버전
```python
conda create -n "가상환경 명" python=3.7 -y
conda activate "가상환경 명"
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine 
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html 
git clone -b main https://github.com/open-mmlab/mmsegmentation.git 
cd mmsegmentation
pip install -e . 
pip install "mmsegmentation==1.0.0" 
pip install future tensorboard 
```

# #3-4: private score 복원 가능한 코드 파일로 학습 후 inference하는 법

private score 복원 가능한 코드파일: data_preprocessing, train 폴더 안의 코드들

## 3. Preprocess 코드 실행

**1) data_preprocessing 폴더 안에 있는 preprocessing.sh bash 파일에서 데이터셋 폴더와 preprocessed된 데이터셋폴더 위치를 지정한다.**

```python
#Path to original dacon data
export DATAPATH="../open"

#Path to save preprocessed data
export PREPROCESSED="../preprocessed"
```

**2) data_preprocessing 폴더 안에 있는  preprocessing.sh bash파일을 실행한다.**

```python
./preprocessing.sh
```

**3) 만약 실행했을 때  /bin/bash: bad interpreter: Permission denied error가 난다면 preprocessing.sh의 파일 권한을 다음과 같이 수정해주면 된다.**

```python
chmod 755 preprocessing.sh
```

참고: https://pinggoopark.tistory.com/301

이때 preprocessing.sh를 실행하면 draw_label.py → rename.py → data_crop.py → mv_splits.py가 순서대로 실행되고

- **draw_label.py**: rle_decode함수를 사용해 label image를 생성한다.
- **rename.py**: TRAIN_0000.png → 0000.png, LABEL_0000.png→0000.png와 같이 train image와 생성된 label image들의 파일 이름을 바꿔준 후 PREPROCESSED 위치에  train_img/, train_label/ 폴더를 저장한다.
- **data_crop.py**:  train image와 label을 200단위로 한 이미지당 총 25개의 244x244 이미지로 crop한다. 이를 PREPROCESSED 위치에 crop_image, crop_label 폴더에 저장한다. 즉 preprocessed dataset을 만든다.
- **mv_splits.py**: splits 폴더를 PREPROCESSED 위치로 이동시킨다.

## 4. Train 코드 실행 방법 (학습 후 inference하는 방법)

**1) train/ 폴더 안에 있는 trainA.py 파일을 실행한다.**

```python
python trainA.py --mmseg [git clone한 mmsegmentation 경로] --data [preprocessed된 dataset 폴더 경로]
```

- `[preprocessed된 dataset 폴더 경로]` 는 3-1의 PREPROCESSED 경로와 일치해야 한다.

**2) trainA.py 실행이 끝난 후 trainB.py 파일을 실행한다.**

```python
python trainB.py --mmseg [git clone한 mmsegmentation 경로] --data [preprocessed된 dataset 폴더 경로로]
```

- `[preprocessed된 dataset 폴더 경로]` 는 3-1의 PREPROCESSED 경로와 일치해야 한다.

**3)  trainB.py 실행이 끝난 후 trainC.py 파일을 실행한다.**

```python
python trainC.py --mmseg [git clone한 mmsegmentation 경로] --data [preprocessed된 dataset 폴더 경로]
```

- `[preprocessed된 dataset 폴더 경로]` 는 3-1의 PREPROCESSED 경로와 일치해야 한다.

**4) trainC.py 실행이 끝난 후 학습이 완료되었음으로 아래 명령어를 통해 inference_after_train.py를 실행한다.**

```python
python3 inference_after_train.py --data [test_img 폴더가 위치한 경로] --output_dir [csv 파일 저장되는 경로]
```

- `[test_img 폴더가 위치한 경로]` : Dacon 원래 데이터셋인 open 폴더 경로와 같다.



즉, trainA.py →trainB.py→ trainC.py→ inference_after_train.py가 순서대로 실행되고 각 파일이 실행 및 의미하는 바는 다음과 같다.

- **trainA.py:**
    
    mmseg의 pretrained model weights를 불러오고 crop된 train image를 이용해 53550 iterations만큼 학습
    
    생성된 segformer_modelA폴더에 model weight 파일(.pth)이 저장되고 vis_data 폴더 안에 config.py 파일과 tensorboard log가 찍힌다.
    
- **trainB.py:**
    
    modelA의 학습된 weights(segformer_modelA 폴더에 저장된 .pth 파일)를 불러오고 non crop된 train image를 이용해 35700 iterations만큼 학습
    
    생성된 segformer_modelB 폴더에 model weight 파일(.pth)이 저장되고 vis_data 폴더 안에 config.py 파일과 tensorboard log가 찍힌다.
    
- **trainC.py:**
    
    modelB의 학습된 weights(segformer_modelB 폴더에 저장된 .pth 파일)를 불러오고 non crop된 train image를 이용해 120000 iterations만큼 학습
    
    생성된 segformer_modelC 폴더에 model weight 파일(.pth)이 저장되고 vis_data 폴더 안에 config.py 파일과 tensorboard log가 찍힌다.
    
- **inference_after_train.py:**
    
    최종 학습 완료된 modelC의 weights를 불러와 test image를 inference하고 output_dir 경로에 **inference 결과인 YCS1.csv 파일이 생성된다.**
    

# #5: private score 복원 가능한 모델 weight파일(.pth)로 inference하는 법

private score 복원 가능한 모델 weight 파일: ‘train/checkpoints’ 폴더 안의 iter_110670.pth 파일

## 5. Inference 코드 실행 방법 (weight 불러와서 inference 하는 방법)

**1) train 폴더 안으로 이동한 후 아래 명령어를 통해 inference.py  파일 실행한다.**

```python
python inference.py --data [test_img 폴더가 위치한 경로] --output_dir [csv 파일 저장되는 경로]
```

- `[test_img 폴더가 위치한 경로]` : Dacon 원래 데이터셋인 open 폴더 경로와 같다.
- inference.py:
    
    제출한 복원 가능한 weight 파일을 불러와 inference 후 output_dir 경로에 **inference 결과인 YCS2.csv 파일이 생성된다.**
