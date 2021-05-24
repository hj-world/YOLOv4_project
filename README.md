# YOLOv4_project 

### 🚢 자율운행 선박을 위한 카메라 기반 장애물 인식 및 경고 시스템 🚢  
![@@@ yolov4 튜닝(2)](https://user-images.githubusercontent.com/60416651/118238305-501b5600-b4d3-11eb-907c-b235ae0210ac.jpg)

객체 탐지 모델 중 YOLO(You Only Look Once)를 이용하여 실시간으로 해양 환경의 물체를 인식하고 주행에 방해가 되는 장애물에 대해 경고 알림을 울리는 시스템을 구현한다. 빠르고 정확한 장애물 탐지를 위해 기존 YOLOv4를 수정, 보완하여 학습 모델을 만든다.  

## 01. 개발 환경
Hardware  
+ CPU: Intel(R) Xeon(R) CPU E5-2620 v3 2.4GHz * 2  
+ GPU: GeForce GTX 1080 Ti * 12  
+ RAM: 16GB  

Software  
+ OS: Ubuntu 18.04.5 LTS  
+ CUDA: 11.2  
+ cuDNN: 8.0.5  
+ Framework: Darknet  


## 02. 실행 순서

### (1) github 저장소 복제
git을 clone하거나 현재 이 페이지에서 zip파일을 다운로드 한다.

```
git clone http://github.com/hjp0503/yolov4_project.git 
```

YOLO를 실행시키키 위한 신경망 프레임워크인 Darknet( https://github.com/AlexeyAB/darknet.git )은 이미 저장소 안에 설치된 상태이다.

### (2) darknet make
자신의 컴퓨터 환경에 맞게 makefile 파일을 수정한다.
```
GPU=1              # GPU 사용하여 가속하기 위한 CUDA를 포함  
CUDNN=1            # GPU 사용하여 벼림가속을 위한 CUDNN을 포함  
CUDNN_HALF=1       # 텐서코어에 대해 검출 3x, 벼림 2x 가속됨  
OPENCV=1           # 이미지/동영상/카메라로부터 객체 검출시 opencv 사용  
AVX=0                
OPENMP=0           # 다중-코어 CPU를 사용하여 가속하기 위해 openmp 사용  
LIBSO=1            # 추후 darknet.so 파일 생성  
...
```

make 명령어를 통해 darknet을 컴퓨터에 make 한다.  
```
make
```

### (3) 데이터셋 추가
싱가포르 해양 데이터셋(Sigapore Maritime Dataset, SMD)이라는 오픈 데이터셋을 이용하여 학습에 사용할 데이터셋을 생성한다.
https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset 에서 내려받은 후 `data_pre-processing.ipynb` 코드를 실행하여 데이터 전처리 작업을 진행한다.
전처리가 끝나면 img 폴더에 이미지와 레이블 파일이 생성된다.  
(웹 크롤링한 이미지를 추가적으로 사용했지만, 저작권 상 올리지 않음)

### (4) 사전학습된(pre-trained) 가중치 사용
yolov2의 사전학습된 가중치: darknet19_448.conv.23 사용 (https://pjreddie.com/darknet/yolov2/)  
yolov4의 사전학습된 가중치: yolov4.conv.137 사용 (https://github.com/AlexeyAB/darknet/releases)

### (5) 모델 훈련
train 명령어를 이용하여 훈련을 시작한다.  

명령어 형식  
`*./darknet detector train [폴더/data파일] [폴더/cfg파일] [미리 학습된 weights파일] [옵션]`  
[pre-trained 가중치 파일] [옵션] 은 선택사항이다.

YOLOv2 훈련 명령어
```
./darknet detector train yolov2/smd.data yolov2/yolov4.cfg darknet19_448.conv.23 -map
```
YOLOv4 훈련 명령어
```
./darknet detector train yolov4/smd.data yolov4/yolov2.cfg yolov4.conv.137 -map
```

### (6) 모델 성능 확인
map 명령어를 이용하여 모델 성능을 확인한다.  

명령어 형식  
`*./darknet detector map [폴더/data파일] [폴더/cfg파일] [폴더/weights파일]`    

YOLOv2 성능 확인 명령어
```
/darknet detector map yolov2/smd.data yolov2/yolov2.cfg backup_yolov2/yolov2_best.weights
```
YOLOv4 성능 확인 명령어
```
/darknet detector map yolov4/smd.data yolov4/yolov4.cfg backup_yolov4/yolov4_best.weights
```

### (7) 모델 테스트
test 명령어를 이용하여 모델의 객체 탐지 결과를 확인한다.    

이미지 테스트 명령어 형식  
`*./darknet detector test [폴더/data파일] [폴더/cfg파일] [폴더/weights파일] [폴더/테스트.jpg] -i 0 -thresh [임계값]`  
비디오 테스트 명령어 형식  
`*./darknet detector demo [폴더/data파일] [폴더/cfg파일] [폴더/weights파일] [폴더/테스트.avi] -i 0 -thresh [임계값]`  
[폴더/테스트파일]은 이미지(.jpg) 또는 비디오(.avi) 파일을 입력할 수 있다.  
[임계값]은 0~1까지의 값으로 정한 값 이상으로 검출된 개체만 표시한다.  


YOLOv2 테스트 명령어
```
# 이미지  
./darknet detector test yolov2/smd.data yolov2/yolov2-test.cfg backup_yolov2/yolov2_best.weights data/ship.jpg -i 0 -thresh 0.25  
# 비디오  
./darknet detector demo yolov2/smd.data yolov2/yolov2-test.cfg backup_yolov2/yolov2_best.weights data/ship.avi -i 0 -thresh 0.25  
```
YOLOv4 테스트 명령어
```
# 이미지  
./darknet detector test yolov4/smd.data yolov4/yolov4-test.cfg backup_yolov4/yolov4_best.weights data/ship.jpg -i 0 -thresh 0.25  
# 비디오
./darknet detector demo yolov4/smd.data yolov4/yolov4-test.cfg backup_yolov4/yolov4_best.weights data/ship.avi -i 0 -thresh 0.25  
```

### (8) 경고알림 기능 실행
비디오 테스트에 -ext_output 옵션을 붙이면, 가까운 객체에 대해 경고알림이 울린다.  
```
./darknet detector demo yolov4/smd.data yolov4/yolov4-test.cfg backup_yolov4/yolov4_best.weights data/ship.avi -i 0 -thresh 0.25 -ext_output
```

