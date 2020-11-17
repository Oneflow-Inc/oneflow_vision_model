"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import argparse
import numpy as np
import cv2
import json
import shutil
import pdb
import time
import os
from yolo_net import YoloPredictNet
#import oneflow_yolov3
#from oneflow_yolov3.ops.yolo_decode import yolo_predict_decoder

parser = argparse.ArgumentParser(description="flags for predict")
parser.add_argument("-g", "--gpu_num_per_node", type=int, default=1, required=False)
parser.add_argument("-load", "--model_load_dir", type=str, default="yolov3/of_model/yolov3_model_python/", required=False)
parser.add_argument("-image_height", "--image_height", type=int, default=608, required=False)
parser.add_argument("-image_width", "--image_width", type=int, default=608, required=False)
parser.add_argument("-img_list", "--image_list_path", type=str, default="", required=False)
parser.add_argument("-image_path_list", "--image_path_list", type=str, nargs='+', required=False)
parser.add_argument("-label_to_name_file", "--label_to_name_file", default="yolov3/data/coco.names", type=str, required=False)
parser.add_argument("-batch_size", "--batch_size", type=int, default=1, required=False)
parser.add_argument("-total_batch_num", "--total_batch_num", type=int, default=1, required=False)
parser.add_argument("-loss_print_steps", "--loss_print_steps", type=int, default=1, required=False)
parser.add_argument("-use_tensorrt", "--use_tensorrt", type=int, default=0, required=False)
args = parser.parse_args()

#flow.config.load_library(oneflow_yolov3.lib_path())
func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float) 
if args.use_tensorrt != 0:
    func_config.use_tensorrt(True)

# func_config = flow.FunctionConfig()
# func_config.default_logical_view(flow.scope.consistent_view())
# func_config.default_data_type(flow.float)
# if args.use_tensorrt != 0:
#     func_config.use_tensorrt(True)


label_2_name=[]

with open(args.label_to_name_file,'r') as f:
  label_2_name=f.readlines()

nms=True
#print("nms:", nms)
input_blob_def_dict = {
    "images" : flow.FixedTensorDef((args.batch_size, 3, args.image_height, args.image_width), dtype=flow.float),
    "origin_image_info" : flow.FixedTensorDef((args.batch_size, 2), dtype=flow.int32),
}
    
def print_detect_box(positions, probs):
    if nms==True:
        batch_size = positions.shape[0]
        for k in range(batch_size):
            for i in range(1, 81):
                for j in range(positions.shape[2]):
                    if positions[k][i][j][1]!=0 and positions[k][i][j][2]!=0 and probs[k][i][j]!=0:
                        print(label_2_name[i-1], " ", probs[k][i][j]*100,"%", "  ", positions[k][i][j][0], " ", positions[k][i][j][1], " ", positions[k][i][j][2], " ", positions[k][i][j][3])
    else:
        for j in range(positions.shape[1]):
            for i in range(1, 81):
                if positions[0][j][1]!=0 and positions[0][j][2]!=0 and probs[0][j][i]!=0:
                    print(label_2_name[i-1], " ", probs[0][j][i]*100,"%", "  ",positions[0][j][0], " ", positions[0][j][1], " ", positions[0][j][2], " ", positions[0][j][3])

def xywh_2_x1y1x2y2(x, y, w, h, origin_image):
    x1  = (x - w / 2.) * origin_image[1]
    x2  = (x + w / 2.) * origin_image[1]
    y1   = (y - h / 2.) * origin_image[0]
    y2   = (y + h / 2.) * origin_image[0]
    return x1, y1, x2, y2


def batch_boxes(positions, probs, origin_image_info):
    batch_size = positions.shape[0]
    batch_list=[]
    if nms==True:
        for k in range(batch_size):
            box_list = []
            for i in range(1, 81):
                for j in range(positions.shape[2]):
                    if positions[k][i][j][2]!=0 and positions[k][i][j][3]!=0 and probs[k][i][j]!=0:
                        x1, y1, x2, y2 = xywh_2_x1y1x2y2(positions[k][i][j][0], positions[k][i][j][1], positions[k][i][j][2], positions[k][i][j][3], origin_image_info[k])
                        bbox = [i-1, x1, y1, x2, y2, probs[k][i][j]]
                        box_list.append(bbox)
            batch_list.append(np.asarray(box_list))
    else:
        for k in range(batch_size):
            box_list = []
            for j in range(positions.shape[1]):
                for i in range(1, 81):
                    if positions[k][j][2]!=0 and positions[k][j][3]!=0 and probs[k][j][i]!=0:
                        x1, y1, x2, y2 = xywh_2_x1y1x2y2(positions[k][j][0], positions[k][j][1], positions[k][j][2], positions[k][j][3], origin_image_info[k])
                        bbox = [i-1, x1, y1, x2, y2, probs[k][j][i]]
                        box_list.append(bbox)
            batch_list.append(np.asarray(box_list))
    return batch_list

def striplist(l):
    return([x.strip() for x in l])

@flow.global_function(func_config)
def yolo_user_op_eval_job(images=input_blob_def_dict["images"],origin_image_info=input_blob_def_dict["origin_image_info"]):
    yolo_pos_result, yolo_prob_result = YoloPredictNet(images, origin_image_info, trainable=False)
    yolo_pos_result = flow.identity(yolo_pos_result, name="yolo_pos_result_end")
    yolo_prob_result = flow.identity(yolo_prob_result, name="yolo_prob_result_end")
    return yolo_pos_result, yolo_prob_result, origin_image_info

def yolo_show(image_path_list, batch_list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img_path, batch in zip(image_path_list, batch_list):
        result_list = batch.tolist()
        img = cv2.imread(img_path)        
        for result in result_list:
            cls = int(result[0])
            bbox = result[1:-1]
            score = result[-1]
            print('img_file:', img_path)
            print('cls:', cls)
            print('bbox:', bbox)
            c = ((int(bbox[0]) + int(bbox[2]))/2, (int(bbox[1] + int(bbox[3]))/2))
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 1)
            cv2.putText(img, str(cls), (int(c[0]), int(c[1])), font, 1, (0, 0, 255), 1)
        result_name = img_path.split('/')[-1]
        cv2.imwrite("data/results/"+result_name, img)

def resize_image(img, origin_h, origin_w, image_height, image_width):
    w = image_width
    h = image_height
    resized=np.zeros((3, image_height, image_width), dtype=np.float32)
    part=np.zeros((3, origin_h, image_width), dtype = np.float32)
    w_scale = (float)(origin_w - 1) / (w - 1)
    h_scale = (float)(origin_h - 1) / (h - 1)

    for c in range(w):
        if c == w-1 or origin_w == 1:
            val = img[:, :, origin_w-1]
        else:
            sx = c * w_scale
            ix = int(sx)
            dx = sx - ix 
            val = (1 - dx) * img[:, :, ix] + dx * img[:, :, ix+1]
        part[:, :, c] = val
    for r in range(h):
        sy = r * h_scale
        iy = int(sy)
        dy = sy - iy
        val = (1-dy)*part[:, iy, :]
        resized[:, r, :] = val 
        if r==h-1 or origin_h==1:
            continue
        resized[:, r, :] = resized[:, r, :] + dy * part[:, iy+1, :]
    return resized

def image_preprocess_v2(img_path, image_height, image_width):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    img = img.transpose(2, 0, 1).astype(np.float32) # hwc->chw
    img = img / 255  # /255
    img[[0,1,2], :, :] = img[[2,1,0], :, :] #bgr2rgb

    w = image_width 
    h = image_height
    origin_h = img.shape[1]
    origin_w = img.shape[2]
    new_w = origin_w
    new_h = origin_h
    if w/origin_w < h/origin_h:
        new_w = w 
        new_h = origin_h * w // origin_w
    else:
        new_h = h 
        new_w = origin_w * h // origin_h    
    resize_img = resize_image(img, origin_h, origin_w, new_h, new_w)

    dw = (w-new_w)//2
    dh = (h-new_h)//2

    padh_before = int(dh)
    padh_after = int(h - new_h - padh_before)
    padw_before = int(dw)
    padw_after = int(w - new_w - padw_before)
    result = np.pad(resize_img, pad_width = ((0,0),(padh_before, padh_after),(padw_before, padw_after)), mode='constant', constant_values=0.5)
    result = np.expand_dims(result, axis=0)
    origin_image_info = np.zeros((1,2), dtype=np.int32)
    origin_image_info[0,0] = origin_h
    origin_image_info[0,1] = origin_w
    return result.astype(np.float32), origin_image_info

def one_frame_preprocess_v2(img_ori, image_height, image_width):
    img = img_ori.transpose(2, 0, 1).astype(np.float32) # hwc->chw
    img = img / 255  # /255
    img[[0,1,2], :, :] = img[[2,1,0], :, :] #bgr2rgb

    w = image_width 
    h = image_height
    origin_h = img.shape[1]
    origin_w = img.shape[2]
    new_w = origin_w
    new_h = origin_h
    if w/origin_w < h/origin_h:
        new_w = w 
        new_h = origin_h * w // origin_w
    else:
        new_h = h 
        new_w = origin_w * h // origin_h    
    resize_img = resize_image(img, origin_h, origin_w, new_h, new_w)

    dw = (w-new_w)//2
    dh = (h-new_h)//2

    padh_before = int(dh)
    padh_after = int(h - new_h - padh_before)
    padw_before = int(dw)
    padw_after = int(w - new_w - padw_before)
    result = np.pad(resize_img, pad_width = ((0,0),(padh_before, padh_after),(padw_before, padw_after)), mode='constant', constant_values=0.5)
    result = np.expand_dims(result, axis=0)
    origin_image_info = np.zeros((1,2), dtype=np.int32)
    origin_image_info[0,0] = origin_h
    origin_image_info[0,1] = origin_w
    return result.astype(np.float32), origin_image_info

def batch_image_preprocess_v2(img_path_list, image_height, image_width):
    result_list = []
    origin_info_list = []
    for img_path in img_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = img.transpose(2, 0, 1).astype(np.float32) # hwc->chw
        img = img / 255  # /255
        img[[0,1,2], :, :] = img[[2,1,0], :, :] #bgr2rgb

        w = image_width 
        h = image_height
        origin_h = img.shape[1]
        origin_w = img.shape[2]
        new_w = origin_w
        new_h = origin_h
        if w/origin_w < h/origin_h:
            new_w = w 
            new_h = origin_h * w // origin_w
        else:
            new_h = h 
            new_w = origin_w * h // origin_h    
        resize_img = resize_image(img, origin_h, origin_w, new_h, new_w)

        dw = (w-new_w)//2
        dh = (h-new_h)//2

        padh_before = int(dh)
        padh_after = int(h - new_h - padh_before)
        padw_before = int(dw)
        padw_after = int(w - new_w - padw_before)
        result = np.pad(resize_img, pad_width = ((0,0),(padh_before, padh_after),(padw_before, padw_after)), mode='constant', constant_values=0.5)
        origin_image_info = [origin_h, origin_w]
        result_list.append(result)
        origin_info_list.append(origin_image_info)
    results = np.asarray(result_list).astype(np.float32)
    origin_image_infos = np.asarray(origin_info_list).astype(np.int32)
    return results, origin_image_infos

def coco_format(type_, id_list, file_list, result_list, label_list): 
    annotations = []    
    for i, result in enumerate(result_list):
        temp = {}
        id_name = id_list[i]
        file_path = file_list[i]
        temp['id'] = id_name
        temp['annotation'] = []      
        im = cv2.imread(file_path)
        height, width, _ = im.shape
        if result.shape[0] == 0:
            temp['annotation'] = json.dumps(temp['annotation'])
            annotations.append(temp)
            continue
        else:
            for j in range(result.shape[0]):
                cls_id = int(result[j][0])+1
                x1 = result[j][1]
                x2 = result[j][3]
                y1 = result[j][2]
                y2 = result[j][4]
                score = result[j][5]
                width = max(0, x2-x1)
                height = max(0, y2-y1)   
                if cls_id in label_list:
                    temp['annotation'].append({
                        'area': width * height,
                        'bbox': [x1, y1, width, height],
                        'category_id': cls_id,
#                            'id': id_index,
#                            'image_id': id_name,
                        'iscrowd': 0,
                        # mask, 矩形是从左上角点按顺时针的四个顶点
                        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
                        'score': score
                        })
        if type_ == 0 and len(temp['annotation']) > 0:
#            pdb.set_trace()
            temp['annotation'] = [temp['annotation'][0]]
#            temp['annotation'][0]['area'] = ''
#            temp['annotation'][0]['bbox'] = ''
#            temp['annotation'][0]['segmentation'] = ''
            temp['annotation'][0].pop('area')
            temp['annotation'][0].pop('bbox')
            temp['annotation'][0].pop('iscrowd')
            temp['annotation'][0].pop('segmentation')
        temp['annotation'] = json.dumps(temp['annotation'])
        annotations.append(temp)
    return annotations

class YoloInference(object):
    def __init__(self, ):
        flow.config.gpu_device_num(args.gpu_num_per_node)
        flow.env.ctrl_port(9789)
        
        check_point = flow.train.CheckPoint()
        if not args.model_load_dir:
            check_point.init()
        else:
            check_point.load(args.model_load_dir)        

    def yolo_inference(self, type_, id_list, image_path_list, label_list): 
        if len(image_path_list) == 1:
            t0 = time.time() 
            images, origin_image_info = batch_image_preprocess_v2(image_path_list, args.image_height, args.image_width)
            yolo_pos, yolo_prob, origin_image_info = yolo_user_op_eval_job(images, origin_image_info).get()  
            batch_list = batch_boxes(yolo_pos, yolo_prob, origin_image_info)   
            print('result:', batch_list)
            annotations = coco_format(type_, id_list, image_path_list, batch_list, label_list)
            t1 = time.time()
            print('t1-t0:', t1-t0)
        return annotations

    def yolo_inference_one_frame(self, image_ori): 
        images, origin_image_info = one_frame_preprocess_v2(image_ori, args.image_height, args.image_width)
        yolo_pos, yolo_prob, origin_image_info = yolo_user_op_eval_job(images, origin_image_info).get()  
        batch_list = batch_boxes(yolo_pos, yolo_prob, origin_image_info)
        #print('batch_list:', batch_list)
        return batch_list

if __name__ == "__main__":
    
    yolo_obj = YoloInference()
    img = cv2.imread("./yolov3/demo.jpg", cv2.IMREAD_COLOR) 
    batch_list = yolo_obj.yolo_inference_one_frame(img)
    
    if batch_list is not None:
        for i, results in enumerate(batch_list):
            for i, result in enumerate(results):
                print('result:', result)
                cls_id = int(result[0])+1
                x1 = result[1]
                x2 = result[3]
                y1 = result[2]
                y2 = result[4]
                score = result[5]
                print("cls_id: {}, x1: {}, y1 : {},  x2: {}, y2 : {}, score:{}".format(cls_id, x1, x2, y1, y2, score))
        


