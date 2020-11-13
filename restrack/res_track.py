#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
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
 * Copyright (C) 2020 by ZheJiang Lab. All rights reserved.
 * restrack demo
 * created: zhangwenguang   (Email:zhangwg@zhejianglab.com) ,fujiaqing
 * date:07/01/2020
 * version:0.0.1
'''
import os
import sys
sys.path.append('./yolov3')
import cv2
import numpy as np
from datetime import datetime
import argparse
import json

from mot_track_kc import KCTracker
from util import COLORS_10, draw_bboxes, draw_bboxes_conf_cls
import time

import yolov3
from yolov3.predict_with_print_box_in import *

def bbox_to_xywh_cls_conf(bbox_xyxyc, conf_thresh=0.5):
    if any(bbox_xyxyc[:, 4] >= conf_thresh):
        bbox = bbox_xyxyc[bbox_xyxyc[:, 4] >= conf_thresh, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  #
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  #
        return bbox
    else:
        return []

class Detector(object):
    def __init__(self, vid_path, min_confidence=0.4, max_cosine_distance=0.2, max_iou_distance=0.7, max_age=30,
                 out_dir='res/'):
        self.vdo = cv2.VideoCapture()
        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.yolo_detect = YoloInference()
        self.class_ = 80
        self.kc_tracker = []
        if self.class_ > 0:
            for cls_id in range(self.class_):
                kc_tracker = KCTracker(confidence_l=0.01, confidence_h=0.02,use_filter=True, max_cosine_distance=max_cosine_distance,
                                    max_iou_distance=max_iou_distance, max_age=max_age, cls_=cls_id)
                self.kc_tracker.append(kc_tracker)
        else:
            print("class_ is error!")
            return None

        _, filename = os.path.split(vid_path)
        self.mot_txt = os.path.join(self.out_dir, filename[:-4] + '.txt')
        self.mot_txt_filter = os.path.join(self.out_dir, filename[:-4] + '_filter.txt')
        self.mot_txt_bk = os.path.join(self.out_dir, filename[:-4] + '_bk.txt')
        self.det_txt = os.path.join(self.out_dir, filename[:-4] + '_det.txt')
        self.video_name = os.path.join(self.out_dir, filename[:-4] + '_res.avi')
        self.features_npy = os.path.join(self.out_dir, filename[:-4] + '_det.npy')
        self.save_feature = False
        self.all_features = []
        self.write_det_txt = False
        self.write_video = False
        self.use_tracker = True
        self.person_id = 1
        self.write_img = True
        self.write_json = True
        self.read_json = True
        self.write_bk = False
        self.temp_dir = filename[:-4]
        if self.write_img:
            self.img_dir = os.path.join(self.out_dir + '/' + self.temp_dir, 'imgs')
            os.makedirs(self.img_dir, exist_ok=True)
        if self.write_json or self.read_json:
            self.json_dir = os.path.join(self.out_dir + '/' + self.temp_dir, 'json')
            if self.write_json:
                os.makedirs(self.json_dir, exist_ok=True)
        print("Track Detector init sucessed!\n")

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        print("video_path %s \n" %video_path)
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.vdo.get(cv2.CAP_PROP_FPS)

        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output = cv2.VideoWriter(self.video_name, fourcc, self.fps, (self.im_width, self.im_height))
        if self.im_width > 0 and self.im_height > 0:
            print("open video sucessed!\n")
    def save_file(self, path, item):
        item = json.dumps((item))
        try:
            if not os.path.exists(path):
                with open(path, "w", encoding='utf-8') as f:
                    f.write(item + "\n")
            else:
                with open(path, "w", encoding='utf-8') as f:
                    f.write(item + "\n")
        except Exception as e:
            print("write error==>", e)

    def de_coco_format(self, ann_json):
        json_data = []
        if ann_json:
            for i, annotation in enumerate(ann_json):
                conf_ = float(annotation['score']) / 100
                cls_ = int(annotation['category_id'] - 1)
                x1 = float(annotation['bbox'][0])
                x2 = float(annotation['bbox'][2] + x1)
                y1 = float(annotation['bbox'][1])
                y2 = float(annotation['bbox'][3] + y1)
                object_ = [x1, y1,x2,y2,conf_,cls_]
                json_data.append(np.array(object_))
        return np.array(json_data)

    def coco_format(self, type_, id_name, im, result): 
        temp = []
        height, width, _ = im.shape
        if result.shape[0] == 0:
            return temp
        else:
            for j in range(result.shape[0]):
                cls_id = int(result[j][6])+1
                x1 = int(result[j][0])
                x2 = int(result[j][2])
                y1 = int(result[j][1])
                y2 = int(result[j][3])
                track_id = int(result[j][5])
                score = float(result[j][4])
                width = max(0, x2-x1)
                height = max(0, y2-y1)   
                temp.append({
                    'category_id': cls_id,
                    'area': width * height,
                    'iscrowd': 0,
                    'bbox': [x1, y1, width, height],
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
                    'score': score,
                    'track_id':track_id
                    })
        return temp

    def get_result(self, result_list): 
        results = []    
        for i, result in enumerate(result_list):
            if result.shape[0] == 0:
                continue
            else:
                for j in range(result.shape[0]):
                    cls_id = int(result[j][0])+1
                    x1 = result[j][1]
                    x2 = result[j][3]
                    y1 = result[j][2]
                    y2 = result[j][4]
                    score = result[j][5]
                    object_ = [x1, y1,x2,y2,score,cls_id]
                    results.append(np.array(object_))
        return results


    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        #for path, img, ori_im, vid_cap in self.dataset:
        while 1:
            ret, ori_im = self.vdo.read()
            if ori_im is None:
                break
            frame_no += 1
            start = time.time()
            im = ori_im[ymin:ymax, xmin:xmax]
            t1 = time.time()
            results = []
            batch_list = self.yolo_detect.yolo_inference_one_frame(ori_im)
            results = self.get_result(batch_list)

            if self.write_json:
                jsonname= self.json_dir + '/' + str(frame_no) + '.json'
                if len(results) > 0:
                    ann = self.coco_format(1, 1, ori_im, results)
                    self.save_file(jsonname, ann)
            if self.read_json:
                jsonname= self.json_dir + '/' + str(frame_no) + '.json'
                with open(jsonname,'r',encoding='utf8')as fp:
                    ann_json = json.load(fp)
                    results = self.de_coco_format(ann_json)
            t2 = time.time()
            bbox_xywhcs = []
            if len(results) > 0:
                outputs = []
                for cls_id in range(self.class_):
                    results_cls = np.array(results)
                    results_cls = results_cls[results_cls[:, 5] == cls_id]
                    results_cls_ = results_cls[:, [0,1,2,3,4]]
                    bbox_xywhcs = bbox_to_xywh_cls_conf(results_cls_, conf_thresh=0.05)
                    if len(bbox_xywhcs) > 0:
                        output = []
                        features = []
                        feature_type = 0
                        if cls_id == 0:
                            feature_type = 0
                        if self.use_tracker:

                            output, features_ = self.kc_tracker[cls_id].update(frame_no, bbox_xywhcs, ori_im, type = feature_type)
                            features.append(features_)
                            if self.save_feature:
                                if self.all_features is None:
                                    self.all_features = features
                                else:
                                    self.all_features = np.vstack((self.all_features, features))
                        if len(output) > 0:
                            bbox_xyxy = output[:, 0:4]
                            identities = output[:, 5]
                            confs = output[:, 4]
                            ori_im = draw_bboxes_conf_cls(ori_im, bbox_xyxy, confs, identities, offset=(0, 0), cls_id_ = cls_id)
                            output = np.insert(output, 6, values=cls_id, axis=1)
                            if len(outputs) == 0:
                                outputs = output
                            else:
                                outputs = np.vstack((outputs, output))
                if self.write_json:
                    jsonname= str(label_name)
                    if len(outputs) > 0:
                        ann = self.coco_format(1, 1, ori_im, outputs)
                        self.save_file(jsonname, ann)
            else:
                for cls_id in range(self.class_):
                    self.kc_tracker[cls_id].update(frame_no, bbox_xywhcs, im)
            end = time.time()
            fps = 1 / (end - start)
            avg_fps += fps
            if frame_no % 100 == 0:
                print("detect cost time: {}s, fps: {}, frame_no : {} track cost:{}".format(end - start, fps, frame_no,
                                                                                           end - t2))

            if self.write_video:
                self.output.write(ori_im)
            if self.write_img:
                cv2.imwrite(os.path.join(self.img_dir, '{:06d}.jpg'.format(frame_no)), ori_im)

        self.vdo.release()
        if self.save_feature:
            self.saveFeature(self.features_npy, self.all_features)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    start_time = datetime.now()
    print('start time:', start_time)
    vid_path = './data/video/demo.avi'
    filename = 'demo.avi'
    det = Detector(filename, min_confidence=0.35, max_cosine_distance=0.2,
                    max_iou_distance=0.7, max_age=30, out_dir='results/res_20200326')
    det.save_feature = False
    det.write_det_txt = False
    det.use_tracker = True
    det.write_video = True
    det.write_bk = False
    det.write_json = False
    det.read_json = False
    det.open(vid_path)
    det.detect()
    end_time = datetime.now()
    print(' cost hour:', (end_time - start_time) / 3600)
