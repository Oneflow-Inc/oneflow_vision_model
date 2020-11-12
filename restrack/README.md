# ResTrack

天枢项目，restrack 多目标跟踪

基于yolo3的deepsort跟踪框架，使用resid作为reid模型进行目标的外观特征提取。

install & run:

docker run -p 端口:22 -it --gpus all harbor.dubhe.ai/oneflow/oneflow:cudnn7-py36-of

apt-get install ssh

apt-get install vim

pip install scipy opencv-python numpy

cd yolov3
 
pip install oneflow_yolov3-0.0.0-py3-none-any.whl

python res_track.py


