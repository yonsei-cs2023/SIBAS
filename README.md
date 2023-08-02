# SIBAS
Satellite Image Building Area Segmentation

# readme

## 0. Library Requirements (버전 기입하기)

- opencv-python
- pandas
- future tensorboard
- openmim
- mmengine (Version == 0.8.2 )
- mmcv (version >= 2.0.0rc1)
- torch (version == 2.0.0+cu118)
- torchvision (version == 0.15.1+cu118)
- torchaudio (version == 2.0.1)
- mmsegmentation (version == 1.0.0)

## 0. 개발 환경 (버전 기입하기)

```python

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
```

## 0.  사용한 pretrained 모델 출처

```python
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

데이콘에서 다운로드 후 압축 푼 Open 폴더 구조는 다음과 같다.

- train_img/
- test_img/
- train.csv
- test.csv
- sample_submission.csv

제출한 zipfile 압축을 풀면 구조는 다음과 같다.

- data_preprocessing/
- train/

**YCS 폴더를 하나 만들고 그 안에 open folder와 data_preprocessing, train 폴더를 이동시킨다.**

![스크린샷 2023-08-02 오후 11.49.29.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b2810b6c-ac15-4036-8352-c3fe8f42482a/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2023-08-02_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.49.29.png)

## 2. 도커를 사용해 라이브러리 등 필요 환경 구축 (도커 컨테이너 생성 및 실행 방법)

**0) 도커를 다운로드 받는다.**

**1) 제출한 도커파일을 이용해 직접 도커이미지를 빌드하거나 이미 생성된 도커이미지 다운을 받는다.**

1-1) 도커파일을 이용해 직접 도커이미지를 빌드하는 경우 (dockerfile이 있는 위치에서 아래 cmd 명령어를 입력한다.)

```python
docker build -t ycs-image [도커파일 directory 경로]
```

1-2) 이미 생성된 도커이미지 다운받는 경우

```python
docker pull docker.io/cygbbhx/ycs-image:latest
```

**2) 데이터셋 및 코드 마운트하고 도커 이미지를 이용해 도커 컨테이너 생성하기**

cmd에서 Open folder와, 제출 zipfile 압축 푼 후 생긴 data_preprocessing, train, inference 폴더를 포함한 폴더(YCS) 위치로 이동한 후 아래 명령어를 입력한다.

```python
docker run -ti -d --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p 6006:6006/tcp --name=ycs --ipc=host -v ${PWD}:/workspace cygbbhx/ycs-image:latest
```

**3) 생성된 도커 컨테이너를 실행한다**

```python
docker exec -ti ycs /bin/bash
```

```python
#만약 ycs 도커 컨테이너가 running 상태가 아니면 docker start후 docker exec 명령어 실행
docker start ycs 
```

**4) 처음 실행 후 도커 컨테이너에서의 초기 위치는 /mmsegmentation이므로  /workspace/data_preprocessing 폴더안으로 이동**

```python
cd /workspace/data_preprocessing

```

# #3-4: private score 복원 가능한 코드 파일로 학습 후 inference하는 법

# #5: private score 복원 가능한 모델 weight파일(.pth)로 inference하는 법

## 3. Preprocess 코드 실행

**1) preprocessing.sh bash파일을 실행한다.**

```python
./preprocessing.sh
```

만약 실행했을 때 

/bin/bash: bad interpreter: Permission denied error가 난다면 preprocessing.sh의 파일 권한을 다음과 같이 수정해주면 된다.

https://pinggoopark.tistory.com/301

이때 preprocessing.sh를 실행하면 draw_label.py → rename.py→ data_crop.py가 순서대로 실행되고

- draw_label.py: rle_decode함수를 사용해 label image를 생성한다.
- rename.py: TRAIN_0000.png → 0000.png, LABEL_0000.png→0000.png와 같이 train image와 생성된 label image들의 파일 이름을 바꿔준 후 /workspace/dataset/ 폴더 아래로  train_img/, train_label/ 폴더를 저장한다.
- data_crop.py:  train image와 label을 200단위로 한 이미지당 총 25개의 244x244 이미지로 crop한다. 이를 /workspace/dataset/ 폴더아래로 crop_image, crop_label 폴더에 저장한다.

따라서 위 명령어를 실행한 후 폴더 구조는 다음과 같다.

- workspace/
    - dataset/
        - train_img/
        - train_label/
        - crop_image/
        - crop_label/
        - splits/
    - open/
    - data_preprocessing/
    - train/

## 4. Train 코드 실행 방법 (학습 후 inference하는 방법)

**1) /workspace/train/ 폴더 안으로 이동한 후 trainA.py 파일 실행한다.**

아래 두 명령어를 순서대로 cmd에 입력한다.

```python
cd /workspace/train/

python3 trainA.py
```

**2) trainA.py 실행이 끝난 후 trainB.py 파일 실행한다.**

아래 두 명령어를 순서대로 cmd에 입력한다.

```python
cd /workspace/train/

python3 trainB.py
```

**3)  trainB.py 실행이 끝난 후 trainC.py 파일 실행한다.**

아래 두 명령어를 순서대로 cmd에 입력한다.

```python
cd /workspace/train/

python3 trainB.py
```

**4) trainC.py 실행이 끝난 후 학습이 완료되었음으로 inference_after_train.py를 실행한다.**

아래 두 명령어를 순서대로 cmd에 입력한다.

```python
cd /workspace/train/

python3 inference_after_train.py
```

즉, trainA.py →trainB.py→ trainC.py→ inference_after_train.py가 순서대로 실행되고 각 파일이 실행 및 의미하는 바는 다음과 같다.

- trainA.py:
    
    mmseg의 pretrained model weights를 불러오고 crop된 train image를 이용해 53550 iterations만큼 학습
    
    segformer_modelA에 model weight 파일(.pth)이 저장되고 vis_data 폴더 안에 config.py 파일과 tensorboard log가 찍힌다.
    
- trainB.py:
    
    modelA의 학습된 weights(segformer_modelA 폴더에 저장된 .pth 파일)를 불러오고 non crop된 train image를 이용해 35700 iterations만큼 학습
    
    segformer_modelB에 model weight 파일(.pth)이 저장되고 vis_data 폴더 안에 config.py 파일과 tensorboard log가 찍힌다.
    
- trainC.py:
    
    modelB의 학습된 weights(segformer_modelB 폴더에 저장된 .pth 파일)를 불러오고 non crop된 train image를 이용해 120000 iterations만큼 학습
    
    segformer_modelC에 model weight 파일(.pth)이 저장되고 vis_data 폴더 안에 config.py 파일과 tensorboard log가 찍힌다.
    
- inference_after_train.py:
    
    최종 학습 완료된 modelC의 weights를 불러와 /workspace/test_image/의 test image를 inference하고 /workspace/ 위치에 **inference 결과인 YCS1.csv 파일이 생성된다.**
    

따라서 위 명령어를 실행한 후 폴더 구조는 다음과 같다.

- workspace/
    - YCS1.csv (생성됨)
    - dataset/
        - train_img/
        - train_label/
        - crop_image/
        - crop_label/
        - splits/
    - open/
    - data_preprocessing/
    - train/
        - segformer_modelA/ (생성됨)
        - segformer_modelB/ (생성됨)
        - segformer_modelC/ (생성됨)
        - checkpoints/
        - config_ABC/
        - inference_after_train.py
        - inference.py
        - train+inference.sh
        - trainA.py
        - trainB.py
        - trainC.py

## 5. Inference 코드 실행 방법 (weight 불러와서 inference 하는 방법)

**1) /workspace/train/ 폴더 안으로 이동한 후 inference.py  파일 실행한다.**

아래 두 명령어를 순서대로 cmd에 입력한다.

```python
cd /workspace/train/

python inference.py
```

- inference.py:
    
    제출한 복원 가능한 weight 파일을 불러와 inference 후 /workspace/ 위치에 **inference 결과인 YCS2.csv 파일이 생성된다.**
    

따라서 위 명령어를 실행한 후 폴더 구조는 다음과 같다.

- workspace/
    - YCS1.csv
    - YCS2.csv (생성됨)
    - dataset/
        - train_img/
        - train_label/
        - crop_image/
        - crop_label/
        - splits/
    - open/
    - data_preprocessing/
    - train/
        - segformer_modelA/
        - segformer_modelB/
        - segformer_modelC/
