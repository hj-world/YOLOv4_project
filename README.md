# YOLOv4_project 

### ๐ข ์์จ์ดํ ์ ๋ฐ์ ์ํ ์นด๋ฉ๋ผ ๊ธฐ๋ฐ ์ฅ์ ๋ฌผ ์ธ์ ๋ฐ ๊ฒฝ๊ณ  ์์คํ ๐ข  
![@@@ yolov4 ํ๋(2)](https://user-images.githubusercontent.com/60416651/118238305-501b5600-b4d3-11eb-907c-b235ae0210ac.jpg)

๊ฐ์ฒด ํ์ง ๋ชจ๋ธ ์ค YOLO(You Only Look Once)๋ฅผ ์ด์ฉํ์ฌ ์ค์๊ฐ์ผ๋ก ํด์ ํ๊ฒฝ์ ๋ฌผ์ฒด๋ฅผ ์ธ์ํ๊ณ  ์ฃผํ์ ๋ฐฉํด๊ฐ ๋๋ ์ฅ์ ๋ฌผ์ ๋ํด ๊ฒฝ๊ณ  ์๋ฆผ์ ์ธ๋ฆฌ๋ ์์คํ์ ๊ตฌํํ๋ค. ๋น ๋ฅด๊ณ  ์ ํํ ์ฅ์ ๋ฌผ ํ์ง๋ฅผ ์ํด ๊ธฐ์กด YOLOv4๋ฅผ ์์ , ๋ณด์ํ์ฌ ํ์ต ๋ชจ๋ธ์ ๋ง๋ ๋ค.  

## 01. ๊ฐ๋ฐ ํ๊ฒฝ
Hardware  
+ CPU: Intel(R) Xeon(R) CPU E5-2620 v3 2.4GHz * 2  
+ GPU: GeForce GTX 1080 Ti * 12  
+ RAM: 16GB  

Software  
+ OS: Ubuntu 18.04.5 LTS  
+ CUDA: 11.2  
+ cuDNN: 8.0.5  
+ Framework: Darknet  


## 02. ์คํ ์์

### (1) github ์ ์ฅ์ ๋ณต์ 
git์ cloneํ๊ฑฐ๋ ํ์ฌ ์ด ํ์ด์ง์์ zipํ์ผ์ ๋ค์ด๋ก๋ ํ๋ค.

```
git clone http://github.com/hjp0503/yolov4_project.git 
```

YOLO๋ฅผ ์คํ์ํคํค ์ํ ์ ๊ฒฝ๋ง ํ๋ ์์ํฌ์ธ Darknet( https://github.com/AlexeyAB/darknet.git )์ ์ด๋ฏธ ์ ์ฅ์ ์์ ์ค์น๋ ์ํ์ด๋ค.

### (2) darknet make
์์ ์ ์ปดํจํฐ ํ๊ฒฝ์ ๋ง๊ฒ makefile ํ์ผ์ ์์ ํ๋ค.
```
GPU=1              # GPU ์ฌ์ฉํ์ฌ ๊ฐ์ํ๊ธฐ ์ํ CUDA๋ฅผ ํฌํจ  
CUDNN=1            # GPU ์ฌ์ฉํ์ฌ ๋ฒผ๋ฆผ๊ฐ์์ ์ํ CUDNN์ ํฌํจ  
CUDNN_HALF=1       # ํ์์ฝ์ด์ ๋ํด ๊ฒ์ถ 3x, ๋ฒผ๋ฆผ 2x ๊ฐ์๋จ  
OPENCV=1           # ์ด๋ฏธ์ง/๋์์/์นด๋ฉ๋ผ๋ก๋ถํฐ ๊ฐ์ฒด ๊ฒ์ถ์ opencv ์ฌ์ฉ  
AVX=0                
OPENMP=0           # ๋ค์ค-์ฝ์ด CPU๋ฅผ ์ฌ์ฉํ์ฌ ๊ฐ์ํ๊ธฐ ์ํด openmp ์ฌ์ฉ  
LIBSO=1            # ์ถํ darknet.so ํ์ผ ์์ฑ  
...
```

make ๋ช๋ น์ด๋ฅผ ํตํด darknet์ ์ปดํจํฐ์ make ํ๋ค.  
```
make
```

### (3) ๋ฐ์ดํฐ์ ์ถ๊ฐ
์ฑ๊ฐํฌ๋ฅด ํด์ ๋ฐ์ดํฐ์(Sigapore Maritime Dataset, SMD)์ด๋ผ๋ ์คํ ๋ฐ์ดํฐ์์ ์ด์ฉํ์ฌ ํ์ต์ ์ฌ์ฉํ  ๋ฐ์ดํฐ์์ ์์ฑํ๋ค.
https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset ์์ ๋ด๋ ค๋ฐ์ ํ `data_pre-processing.ipynb` ์ฝ๋๋ฅผ ์คํํ์ฌ ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ ์์์ ์งํํ๋ค.
์ ์ฒ๋ฆฌ๊ฐ ๋๋๋ฉด img ํด๋์ ์ด๋ฏธ์ง์ ๋ ์ด๋ธ ํ์ผ์ด ์์ฑ๋๋ค.  
(์น ํฌ๋กค๋งํ ์ด๋ฏธ์ง๋ฅผ ์ถ๊ฐ์ ์ผ๋ก ์ฌ์ฉํ์ง๋ง, ์ ์๊ถ ์ ์ฌ๋ฆฌ์ง ์์)

### (4) ์ฌ์ ํ์ต๋(pre-trained) ๊ฐ์ค์น ์ฌ์ฉ
yolov2์ ์ฌ์ ํ์ต๋ ๊ฐ์ค์น: darknet19_448.conv.23 ์ฌ์ฉ (https://pjreddie.com/darknet/yolov2/)  
yolov4์ ์ฌ์ ํ์ต๋ ๊ฐ์ค์น: yolov4.conv.137 ์ฌ์ฉ (https://github.com/AlexeyAB/darknet/releases)

### (5) ๋ชจ๋ธ ํ๋ จ
train ๋ช๋ น์ด๋ฅผ ์ด์ฉํ์ฌ ํ๋ จ์ ์์ํ๋ค.  

๋ช๋ น์ด ํ์  
`*./darknet detector train [ํด๋/dataํ์ผ] [ํด๋/cfgํ์ผ] [๋ฏธ๋ฆฌ ํ์ต๋ weightsํ์ผ] [์ต์]`  
[pre-trained ๊ฐ์ค์น ํ์ผ] [์ต์] ์ ์ ํ์ฌํญ์ด๋ค.

YOLOv2 ํ๋ จ ๋ช๋ น์ด
```
./darknet detector train yolov2/smd.data yolov2/yolov4.cfg darknet19_448.conv.23 -map
```
YOLOv4 ํ๋ จ ๋ช๋ น์ด
```
./darknet detector train yolov4/smd.data yolov4/yolov2.cfg yolov4.conv.137 -map
```

### (6) ๋ชจ๋ธ ์ฑ๋ฅ ํ์ธ
map ๋ช๋ น์ด๋ฅผ ์ด์ฉํ์ฌ ๋ชจ๋ธ ์ฑ๋ฅ์ ํ์ธํ๋ค.  

๋ช๋ น์ด ํ์  
`*./darknet detector map [ํด๋/dataํ์ผ] [ํด๋/cfgํ์ผ] [ํด๋/weightsํ์ผ]`    

YOLOv2 ์ฑ๋ฅ ํ์ธ ๋ช๋ น์ด
```
/darknet detector map yolov2/smd.data yolov2/yolov2.cfg backup_yolov2/yolov2_best.weights
```
YOLOv4 ์ฑ๋ฅ ํ์ธ ๋ช๋ น์ด
```
/darknet detector map yolov4/smd.data yolov4/yolov4.cfg backup_yolov4/yolov4_best.weights
```

### (7) ๋ชจ๋ธ ํ์คํธ
test ๋ช๋ น์ด๋ฅผ ์ด์ฉํ์ฌ ๋ชจ๋ธ์ ๊ฐ์ฒด ํ์ง ๊ฒฐ๊ณผ๋ฅผ ํ์ธํ๋ค.    

์ด๋ฏธ์ง ํ์คํธ ๋ช๋ น์ด ํ์  
`*./darknet detector test [ํด๋/dataํ์ผ] [ํด๋/cfgํ์ผ] [ํด๋/weightsํ์ผ] [ํด๋/ํ์คํธ.jpg] -i 0 -thresh [์๊ณ๊ฐ]`  
๋น๋์ค ํ์คํธ ๋ช๋ น์ด ํ์  
`*./darknet detector demo [ํด๋/dataํ์ผ] [ํด๋/cfgํ์ผ] [ํด๋/weightsํ์ผ] [ํด๋/ํ์คํธ.avi] -i 0 -thresh [์๊ณ๊ฐ]`  
[ํด๋/ํ์คํธํ์ผ]์ ์ด๋ฏธ์ง(.jpg) ๋๋ ๋น๋์ค(.avi) ํ์ผ์ ์๋ ฅํ  ์ ์๋ค.  
[์๊ณ๊ฐ]์ 0~1๊น์ง์ ๊ฐ์ผ๋ก ์ ํ ๊ฐ ์ด์์ผ๋ก ๊ฒ์ถ๋ ๊ฐ์ฒด๋ง ํ์ํ๋ค.  


YOLOv2 ํ์คํธ ๋ช๋ น์ด
```
# ์ด๋ฏธ์ง  
./darknet detector test yolov2/smd.data yolov2/yolov2-test.cfg backup_yolov2/yolov2_best.weights data/ship.jpg -i 0 -thresh 0.25  
# ๋น๋์ค  
./darknet detector demo yolov2/smd.data yolov2/yolov2-test.cfg backup_yolov2/yolov2_best.weights data/ship.avi -i 0 -thresh 0.25  
```
YOLOv4 ํ์คํธ ๋ช๋ น์ด
```
# ์ด๋ฏธ์ง  
./darknet detector test yolov4/smd.data yolov4/yolov4-test.cfg backup_yolov4/yolov4_best.weights data/ship.jpg -i 0 -thresh 0.25  
# ๋น๋์ค
./darknet detector demo yolov4/smd.data yolov4/yolov4-test.cfg backup_yolov4/yolov4_best.weights data/ship.avi -i 0 -thresh 0.25  
```

### (8) ๊ฒฝ๊ณ ์๋ฆผ ๊ธฐ๋ฅ ์คํ
๋น๋์ค ํ์คํธ์ -ext_output ์ต์์ ๋ถ์ด๋ฉด, ๊ฐ๊น์ด ๊ฐ์ฒด์ ๋ํด ๊ฒฝ๊ณ ์๋ฆผ์ด ์ธ๋ฆฐ๋ค.  
```
./darknet detector demo yolov4/smd.data yolov4/yolov4-test.cfg backup_yolov4/yolov4_best.weights data/ship.avi -i 0 -thresh 0.25 -ext_output
```

