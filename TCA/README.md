# TCA                                                                                                               

## Introduction
TCA is used for traffic congestion analysis.

## Installation
oneflow==0.3.4<br>
python-opencv==4.5.1<br>
argparse==1.1<br>
numpy==1.19.2<br>


## Get started

### Download the dataset and model
- Please download our dataset of car and jam [BaiduNetdis](https://pan.baidu.com/s/1jURriB8vg7daobBeeWu16w) code:5b0r.
- Please download our model of car detector [BaiduNetdis](https://pan.baidu.com/s/1RAABK6MubYRfEIz5xpxhRg) code:rz6v,decompress and put it in the folder of saved. 

### Run the car detector

- try detect the car for image  
```
 cd detector
 python demo.py
```   
the result is below:  
![1](https://user-images.githubusercontent.com/26379410/119995968-bee2ce00-c000-11eb-9a92-e8b8edf70530.png)


- try detect the car for video   
```
 cd detector
 python detect_video.py -path=your path/of_dataset/jamDataset
```

### Run and evaluate the jam detector
- Run the jam detector with the detect result of detect_video.py
```
 cd jamDetector
 python main.py -jam_dataset_path=your path/of_dataset/jamDataset
```
- Run the jam detector with the car detector
```
 cd jamDetector
 python main.py -jam_dataset_path=your path/of_dataset/jamDataset -with_detector=True
```
- Run the jam detector with visualization
```
 cd jamDetector
 python main.py -jam_dataset_path=your path/of_dataset/jamDataset -is_visualization=True
```
the result of jam is below:  
![2](https://user-images.githubusercontent.com/26379410/119996100-e8035e80-c000-11eb-82fd-32324d8ba0d4.png)  
the result of no jam is below:  
![3](https://user-images.githubusercontent.com/26379410/119996131-f3ef2080-c000-11eb-8866-09206b3c20e8.png)


### How to train car detector
```
 python generate_txt.py -dataset_path=your path/of_dataset/dataset
 cd jamDetector
 python train.py 
```
## The evaluate result of jam detection

| method | tiny-yolo-v3|    F1   | Precision | Recall | Switch Rate | Hit Rate |
| :----: | :---------: | :-----: |:--------: | :----: | :---------: | :------: |
|  ours  |   Oneflow   |  0.937  |   0.936   |  0.937 |     2.5     |   1.00   |
