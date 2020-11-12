#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import web
#from web.wsgiserver import CherryPyWSGIServer
import os
import pdb
urls = ('/auto_annotate','Upload')
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
import predict_with_print_box as yolo_demo
import argparse
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

parser = argparse.ArgumentParser(description="config for label server")
parser.add_argument("-p", "--port", type=int, default=8080, required=False)
parser.add_argument("-m", "--mode", type=str, default="test", required=False)
args = parser.parse_args()
url_json = './config/url.json'
with open(url_json) as f:
    url_dict = json.loads(f.read())
url = url_dict[args.mode]
port = args.port
taskQueue = Queue()
taskInImages = {}
base_path = "/nfs/"

des_folder = os.path.join('./log', args.mode)
if not os.path.exists(des_folder):
    os.makedirs(des_folder)
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename=des_folder+'/labellog.txt',
                    filemode='w',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    #'%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    '%(asctime)s - %(message)s'
                    #日志格式
                    )
def get_code():
    return ''.join(random.sample(string.ascii_letters + string.digits, 8))
def get_32code():
    return ''.join(random.sample(string.ascii_letters + string.digits, 32))


class Upload:
    def GET(self):
#        x = web.input()
#        print(x)
        #web.header('content-type', 'text/json')
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
            type_ = x['annotateType']
            task_id = get_code() 
            task_images = {}
            task_images[task_id] = {"input":{'type':type_, 'data':x},"output":{"annotations":[]}}
            print("Random_code:", task_id)
            logging.info(task_id)
            web.t_queue.put(task_images)
            return {"code":200, "msg":"", "data":task_id}         
        except Exception as e:
                print(e)
                print("Error Post")
                logging.error("Error post")
                logging.error(e)            
                return 'post error'
def bgProcess():
    global taskQueue
    global url
    print('------------------------------------label server start-----------------------------')
    logging.info('------------------------------------label server start-----------------------------')
    print(url)
    logging.info(url)
    while True:
        try: 
            task_dict =  taskQueue.get()  
            for task_id in task_dict:
                id_list = []
                image_path_list = []
                type_ = task_dict[task_id]["input"]['type']
                for file in task_dict[task_id]["input"]['data']["files"]:
                    id_list.append(file["id"])
                    image_path_list.append(base_path+file["url"])
                label_list = task_dict[task_id]["input"]['data']["labels"]
                image_num = len(image_path_list)
                if image_num < 16:
                    for i in range(16-image_num):
                        image_path_list.append(image_path_list[0])
                        id_list.append(id_list[0])
                #pdb.set_trace()
                print("image_num", image_num)
                print("image_path_list", image_path_list) 
                logging.info(image_num)
                logging.info(image_path_list)               
                annotations = yolo_obj.yolo_inference(type_, id_list, image_path_list, label_list)
                annotations = annotations[0:image_num]
                result = {"annotations":annotations}
                print("result", result)
                logging.info(result)                
                send_data = json.dumps(result).encode()               
                task_url = url + task_id
                headers = {'Content-Type':'application/json'}   
                req = urllib.request.Request(task_url, headers=headers)
                response = urllib.request.urlopen(req, data=send_data, timeout=5)    
                print("task_url:", task_url)
                print("response.read():", response.read())
                print("End mayechi")
                logging.info(task_url)
                logging.info(response.read())
                logging.info("End mayechi")

        except Exception as e:
            print(url)
            print("Error bgProcess")
            print(e)            
            logging.error("Error bgProcess")
            logging.error(e)
            logging.info(url)
#        pass
#        if len(taskInImages) > 500:
#            print('------------clear taskInImages----------------------')
#            taskInImages.clear()
        time.sleep(0.01)
            
 
def bg_thread(no, interval):
    print('bg_thread on')
    bgProcess()

class MyApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

if __name__ == "__main__":  
    yolo_obj = yolo_demo.YoloInference()
    _thread.start_new_thread(bg_thread, (5,5))
#    app = web.application(urls, globals())
    app = MyApplication(urls, globals())
    
    web.t_queue = taskQueue
    web.taskInImages = taskInImages
    app.run(port=port)
