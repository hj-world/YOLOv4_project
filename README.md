# YOLOv4_project 

### 🚢 자율운행 선박을 위한 카메라 기반 장애물 인식 시스템 🚢  
![image](https://user-images.githubusercontent.com/60416651/113977243-43289880-987d-11eb-893f-fa6cb7ce69a8.png)

## 1. 개요 및 필요성
  차량이 인간의 도움 없이 스스로 주행하기 위해서는 주변 장애물에 대한 빠른 탐지와 신속한 대응이 중요하다. 그렇기 때문에 비디오 카메라, 레이더, LIDAR, GPS, SONAR 등의 기술들을 동시에 활용하여 자율주행 능력을 높인다. 그 중에서도 특히 비디오 카메라는 물체를 구별하고 장애물을 피하며 경로를 탐색하는 ‘시각적 인식’을 하는 데에 더욱 핵심적인 역할을 한다.  
  해양 환경에서 선박은 AIS(Automatic Identification System)를 사용하여 주변 배를 인식한다. 하지만 부표, 소형보트, 카약과 같이 규모가 작고 AIS 송신기가 없는 물체는 인식이 불가능하기 때문에 해양 사고가 발생할 수 있다. 그러므로 비디오 카메라를 이용하여 자율주행 선박에서 해양 환경의 장애물을 인식하고 회피할 수 있는 시스템을 제안한다. 


## 2. 개발 내용
   취지에 적합한 기술을 알아본 결과, 딥러닝의 Object Detection 기법 중 **YOLOv4**가 과제에 적합하다고 판단하였다.  
   YOLOv4는 1개의 GPU를 사용하는 일반적인 학습 환경에서도 실시간 수준의 빠른 속도와 기존 YOLO 모델보다 개선된 정확도를 제공하고 있다. 그러므로 YOLOv4를 해양 환경에 맞도록 커스터마이징하여 구현하는 것이 이번 과제의 목표이다.  
   학습에 사용하는 커스텀 데이터셋은 싱가포르 해양 데이터셋(SMD, singapore maritime dataset)로 알려진 공개 데이터셋을 가공하여 만들었다. 컴퓨터에 실행 환경을 구축하고 YOLO를 실행시키기 위한 신경망 프레임워크인 Darknet을 설치하였다. 실행 환경과 커스텀 데이터셋에 맞게 YOLOv4의 네트워크 구조와 학습 하이퍼파라미터를 수정한 후 본격적으로 학습을 시작하였다.  


## 3. 개발 환경
Hardware  
+ CPU: Intel(R) Xeon(R) CPU E5-2620 v3 2.4GHz * 2  
+ GPU: GeForce GTX 1080 Ti * 12  
+ RAM: 16GB  

Software  
+ OS: Ubuntu 18.04.5 LTS  
+ CUDA: 11.2  
+ cuDNN: 8.0.5  
+ Framework: Darknet  

## 4. 파일 설명

## 5. 실행 순서

### (1) github 저장소 복제
git을 clone하거나 현재 이 페이지에서 zip파일을 다운로드 한다.

```
git clone http://github.com/hjp0503/yolov4_project.git 
```
YOLO를 실행시키키 위한 신경망 프레임워크인 Darknet(https://github.com/AlexeyAB/darknet.git)은 이미 저장소 안에 설치된 상태이다.


### (2) darknet make
자신의 컴퓨터 환경에 맞게 makefile 파일을 수정한다.
```
GPU=1              # GPU 사용하여 가속하기 위한 CUDA를 포함
CUDNN=1            # GPU 사용하여 벼림가속을 위한 CUDNN을 포함
CUDNN_HALF=1       # 텐서코어에 대해 검출 3x, 벼림 2x 가속됨
OPENCV=1           # 이미지/동영상/카메라로부터 객체 검출시 opencv 사용
AVX=0              # 
OPENMP=0           # 다중-코어 CPU를 사용하여 가속하기 위해 openmp 사용
LIBSO=1            # 추후 darknet.so 파일 생성
...
```

make 명령어를 통해 darknet을 컴퓨터에 make 한다.  
```
make
```

### (3) 데이터셋 추가
img 폴더에 가공된 데이터셋을 추가한다.  
여기서 가공된 데이터셋은 이미지 + 텍스트파일(ground truth label 정보 포함)을 의미한다.  
SMD(Singapore Maritime Dataset)을 가공한 데이터셋은 img/img.zip에, 웹크롤링하여 가공한 데이터셋은 img/ 에 있다.  
모델 훈련시 압축을 풀어서 사용 해야한다.

### (4) 모델 훈련
train 명령어를 이용하여 훈련을 시작한다.  
명령어 형식 =>  **./darknet detector train [폴더/data파일] [폴더/cfg파일] [미리 학습된 weights파일] [옵션]**  
[pre-trained 가중치 파일] [옵션] 은 선택사항이다.

YOLOv2 훈련 명령어
```
./darknet detector train yolov2/smd.data yolov2/yolov4.cfg yolov4.conv.137 -map
```
YOLOv4 훈련 명령어
```
./darknet detector train yolov4/smd.data yolov4/yolov2.cfg darknet19_448.conv.23 -map
```

### (5) 모델 성능 확인
map 명령어를 이용하여 모델 성능을 확인한다.  
명령어 형식 =>  **./darknet detector map [폴더/data파일] [폴더/cfg파일] [폴더/weights파일]**  

YOLOv2 성능 확인 명령어
```
/darknet detector map yolov2/smd.data yolov2/yolov2.cfg backup_yolov2/yolov2_best.weights
```
YOLOv4 성능 확인 명령어
```
/darknet detector map yolov4/smd.data yolov4/yolov2.cfg backup_yolov2/yolov4_best.weights
```

### (6) 모델 테스트
test 명령어를 이용하여 이미지를 입력하고 모델의 객체 탐지 결과를 확인한다.  
명령어 형식 =>  **./darknet detector test [폴더/data파일] [폴더/cfg파일] [폴더/weights파일] [폴더/.jpg파일] -i 0 -thresh [임계값]**  
[임계값]은 0~1까지의 값으로 정한 값 이상으로 검출된 개체만 표시한다.  


YOLOv2 성능 확인 명령어
```
./darknet detector test yolov2/smd.data yolov2/yolov2-test.cfg backup_yolov2/yolov2_best.weights data/ship.jpg -i 0 -thresh 0.25
```
YOLOv4 성능 확인 명령어
```
./darknet detector test yolov4/smd.data yolov4/yolov4-test.cfg backup_yolov4/yolov4_best.weights data/ship.jpg -i 0 -thresh 0.25
```

## 6. 현재까지의 수행 결과
  학습 모델이 결과로 도출되었고, 성능을 확인해보았다. 대표적인 성능 평가 지표인 mAP는 78.17%, precision, recall, average iou는 각각 90%, 93%, 76.10%로 나왔다.  
  샘플 이미지 몇 장으로 모델을 테스트해보았을 때, 10개의 클래스에 대하여 대부분 정확하게 분류하는 것을 확인하였다. 하지만 크기가 작거나 거리가 멀어서 작아 보이는 소형 object에 대한 탐지성능, 큰 object에 완전히 겹쳐진 object에 대한 탐지성능, 전체가 아닌 부분으로 보여지는 object에 대한 탐지성능이 다소 부족하여 이에 대한 개선이 필요하다고 판단하였다.
  
  
## 7. 기대효과 및 개선방향
  사람의 눈으로도 확인이 어려운 부분까지 잘 탐지하는 것으로 보아 모델을 조금 더 개선시키면, 현장에서의 실질적인 사용도 가능하다고 생각한다. 탐지성능이 떨어지는 문제는 관련 논문과 책에서 비슷한 사례를 찾아보고, 하나씩 고쳐나갈 예정이다.

