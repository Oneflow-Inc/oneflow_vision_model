#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import web
import os
import pdb
urls = ('/track_label','Upload')
import os.path as osp
import string
import _thread
import logging
import urllib
import base64
from queue import Queue
import time
import random
import json

from hog_track import *
import argparse

#import sys
#import codecs
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

taskQueue = Queue()
taskInImages = {}
base_path = "/nfs/"

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--path", type=str, default="/home/")
parser.add_argument("--callback_urls", type=str, default="10.5.24.119:8100")
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--log_name", type=str, default="10.5.24.119:8100")
args = parser.parse_args()
base_path = args.path
callback_urls = args.callback_urls

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='tracklog_' + args.log_name + '.txt',
                    filemode='a',
                    format=
                    '%(asctime)s - %(message)s'
                    )
logging.info('------------------------------------track server start-----------------------------')
logging.info(args.path)
logging.info(args.callback_urls)
logging.info(args.port)
#logging.info(sys.stdout.encoding)

def get_code():
    return ''.join(random.sample(string.ascii_letters + string.digits, 8))
def get_32code():
    return ''.join(random.sample(string.ascii_letters + string.digits, 32))


class Upload:
    def GET(self):
        x = web.input()
        print(x)
        web.header("Access-Control-Allow-Origin", "*")
        web.header("Access-Control-Allow-Credentials", "true")
        web.header('Access-Control-Allow-Headers',  'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')
        web.header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')
        return """<html><head></head><body>please send data in post
</body></html>"""

    def POST(self):
        try:
            web.header("Access-Control-Allow-Origin", "*")
            web.header("Access-Control-Allow-Credentials", "true")
            web.header('Access-Control-Allow-Headers',  'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')
            web.header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')            
            x = web.data()
            x = json.loads(x.decode())
            taskId = get_code() 
            taskInImages = {}
            taskInImages[taskId] = {'data':x}
            print("track Random_code:", taskId)
            logging.info(taskId)
            video_id = x['id']
            images_data = x['images']
            image_num = len(images_data)
            logging.info(video_id)
            logging.info(image_num)
            web.t_queue.put(taskInImages)
            return {"code":200, "msg":"", "data":taskId}         
        except Exception as e:
                logging.error("Error post")
                logging.error(e)            
                print(e)
                print("Error Post")
                return 'post error'
def trackProcess():
    global taskQueue
    global callback_urls
    global callback_urls_addr
    while True:
        try: 
            task_dict =  taskQueue.get()  
            for taskId in task_dict:
                task_data = task_dict[taskId]['data']
                video_id = task_data['id']
                image_list = []
                label_list = []
                images_data = task_data['images']

                for file in images_data:
                    filePath = base_path + file['filePath']
                    annotationPath = base_path + file['annotationPath']
                    image_list.append(filePath)
                    label_list.append(annotationPath)
                image_num = len(label_list)
                logging.info(image_num)

                if len(image_list) != len(label_list):
                    logging.error("Error image_list len != label_list len")
                    print("Error image_list len != label_list len")
                track_det = Detector('xxx.avi', min_confidence=0.35, max_cosine_distance=0.2, max_iou_distance=0.7, max_age=30, out_dir='results/')
                track_det.write_img = False
                RET = track_det.run_track(image_list, label_list)
                logging.info(RET)
                if RET == 'OK':
                    result = {"code": 200,	"msg": 'success',	"data": 'null',	"traceId": 'null'}
                else:
                    result = {"code": 199, "msg": RET, "data": 'null', "traceId": 'null'}
                send_data = json.dumps(result).encode()
                callback_urls_addr = 'http://'+callback_urls+'/api/data/datasets/files/annotations/auto/track/'+str(video_id)
                logging.info(callback_urls_addr)
                headers = {'Content-Type':'application/json'}   
                req = urllib.request.Request(callback_urls_addr, headers=headers)
                response = urllib.request.urlopen(req, data=send_data, timeout=5)              
                logging.info(response.read())
                logging.info("End track")
                print("End track")
        except Exception as e:
            logging.error("Error trackProcess")
            logging.error(e)
            print("Error trackProcess")
            print(e)
        time.sleep(0.01)

def track_thread(no, interval):
    print('track_thread on')
    trackProcess()

class MyApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

if __name__ == "__main__":
    _thread.start_new_thread(track_thread, (5,5))
    app = MyApplication(urls, globals())

    web.t_queue = taskQueue
    web.taskInImages = taskInImages
    app.run(port=args.port)
