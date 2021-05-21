# Recurrent Feature Reasoning for Image Inpainting


## Reproducibility

We've checked the reproducibilities of the results in the paper. 
| |Reproducible|
|:----:|:----:|
|Paris StreetView|True|
|CelebA|True|


## Running the program

To perform training or testing, use 
```
python run1.py
```
There are several arguments that can be used, which are
```
--data_root +str #where to get the images for training/testing
--mask_root +str #where to get the masks for training/testing
--model_save_path +str #where to save the model during training
--result_save_path +str #where to save the inpainting results during testing
--model_path +str #the pretrained generator to use during training/testing
--target_size +int #the size of images and masks
--mask_mode +int #which kind of mask to be used, 0 for external masks with random order, 1 for randomly generated masks, 2 for external masks with fixed order
--batch_size +int #the size of mini-batch for training
--n_threads +int
--iter +int #the number of iter is the same as model_path
--gpu_id +int #which gpu to use
--finetune #to finetune the model during training
--test #test the model
--vgg_path #the path of the vgg model
--epoch #the number of the epoch
```
For example, to train the network using gpu 3, with pretrained models
```
python run1.py --data_root data --mask_root mask --model_path checkpoint/epoch_0_g_10000 --batch_size 6 --gpu 3 --vgg_path Path
```
to test the network
```
python run1.py --data_root data/images --mask_root data/masks --model_path checkpoint/epoch_0_g_10000 --test --mask_mode 2 --iter 10000
```
The RFR-Net for filling smaller holes is added. The only difference is the smaller number of pixels fixed in each iteration.  If you are fixing small holes, you can use that version of code, to gain some speep-up.
## Training procedure
To fully exploit the performance of the network, we suggest to use the following training procedure, in specific

1. Train the network, i.e. use the command
```
python run1.py --data_root data/images --mask_root data/masks --batch_size 6 --gpu 3 --vgg_path Path
```



2. Test the model
```
python run1.py --test --model_path path --iter iternum
```


## Dataset
CelebA：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Paris StreetView： https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
