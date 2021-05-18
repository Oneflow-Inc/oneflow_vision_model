# Posenet                                                                                                                       
 
## Introduction
Posenet is a backbone network, we use it to classify face pose.
 
## Installation
oneflow==0.3.2

## Get started
 
### Prepare data
The face pose dataset can be downloaded from [BaiduNetdis](https://pan.baidu.com/s/1KbrMUrUIS_cCzpDgdgjMRQ) code: o9tg .

### Train a model
```
python3 of_cnn_train_val.py
	--train_data_dir=./dataset/train_set
	--train_data_part_num=1
	--val_data_dir=./dataset/val_set
	--val_data_part_num=1
	--num_nodes=1
	--gpu_num_per_node=1
	--optimizer="sgd"
	--learning_rate=0.0001
	--loss_print_every_n_iter=100
	--batch_size_per_device=32
	--val_batch_size_per_device=32
	--num_epoch=150
	--model="poseNet"
	--num_examples=7459
	--num_val_examples=1990
```
### Download pretrained model

The pre-trained model can be downloaded from [BaiduNetdis](https://pan.baidu.com/s/1vzasLy5MKVe6ILOhmb86TA) code: lln8 .
  
### Validation result

Validation Top1: 92.64%
  
### Inference
```
python3 of_cnn_inference.py
	--model="poseNet"
	--image_path="test/xx.jpg"
	--model_load_dir="snapshot_epoch_xx"
  
```
### Test result and results demonstration 
We test the model on the test_set, which contains 1000 face images, and the accuracy rate achieves 94.2%.

Face pose categories:{ 0: 'frontal', 1: 'profile45', 2: 'profile75', 3: 'upward', 4: 'downward'} 

Some test results are as follows:<br>
![0-f](https://user-images.githubusercontent.com/51501381/118591284-852aef80-b7d6-11eb-9e99-23b06d554675.png)<br>
![1-profile45](https://user-images.githubusercontent.com/51501381/118591356-9ffd6400-b7d6-11eb-9e94-33934de4bda3.png)<br>
![2-profile75](https://user-images.githubusercontent.com/51501381/118591366-a25fbe00-b7d6-11eb-815d-22ad187b0e49.png)<br>
![3-upward](https://user-images.githubusercontent.com/51501381/118591372-a390eb00-b7d6-11eb-90f8-fe0d245e5398.png)<br>
![4-downward](https://user-images.githubusercontent.com/51501381/118591377-a4c21800-b7d6-11eb-8236-91f5b4021671.png)<br>
