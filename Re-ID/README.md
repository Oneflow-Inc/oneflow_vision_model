# Re-ID

## Introduction
This repo provides two methods for Person Re-identification task.
## Get started


### Prerequisites

```
pip install scipy
pip install opencv-python
pip install oneflow
```
**Note:** Install the latest oneflow, which can be download [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/oneflow_cu102-0.2b2-cp37-cp37m-manylinux2014_x86_64.whl).


### Data preparation

We use [Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) to train and evaluate our Re-ID models.
Please download and unzip it in `./dataset` first.



### Train a model

To train a model, run ```python train.py```


Our model is trained using [ResNet50](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/reid_resnet50.zip) as backbones, download and unzip it in  `./pretrained`.



### Evaluate a model
To evaluate a model, run ```python eval.py```


If you only want to evaluate the model, we provide pretrained [Re-ID](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/reid_model.zip) and [HS_ReID](https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/HS_reid_model.zip) model for evaluation.  
The performance on Market1501 is shown below.

|          | Re-ID  |HS_Re-ID|
|  :----:  | :----: | :----: |
|  mAP     | 85.7%  |  80.2% |
|  rank-1  | 94.1%  |  93.2% |
|  rank-5  | 98.0%  |  97.1% |
|  rank-10 | 98.7%  |  98.0% |
