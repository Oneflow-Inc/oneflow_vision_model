import time
import oneflow as flow
import oneflow.typing as tp
import numpy as np
from yolov3_tiny import Yolov3_tiny
from config import cfg
import utils as utils
import cv2
import os
import argparse

test_images = tp.Numpy.Placeholder((1, 3, cfg.TEST.INPUT_SIZE, cfg.TEST.INPUT_SIZE))
anchors_s = tp.Numpy.Placeholder((cfg.YOLO.ANCHOR_PER_SCALE, 2))
anchors_l = tp.Numpy.Placeholder((cfg.YOLO.ANCHOR_PER_SCALE, 2))

func_config = flow.FunctionConfig()
model = Yolov3_tiny(cfg, trainable=False)

@flow.global_function(type="predict", function_config=func_config)
def test_job(images: test_images, anchors_s: anchors_s, anchors_l: anchors_l) \
        -> tp.Numpy:
    pred = model.predict(images, anchors_s, anchors_l)

    return pred

anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
anchors[0,...]/=cfg.YOLO.STRIDES[0]
anchors[1,...]/=cfg.YOLO.STRIDES[1]

flow.load_variables(flow.checkpoint.get(cfg.TEST.WEIGHT_FILE))

def predict(original_image):
    '''

    :param original_image: [H, W, 3]
    :return: (xmin, ymin, xmax, ymax, score, class)
    '''

    image = utils.image_preporcess(np.copy(original_image), [cfg.TEST.INPUT_SIZE, cfg.TEST.INPUT_SIZE])
    image = image[np.newaxis, ...]
    image_ = np.transpose(image, [0,3,1,2])
    image_ = np.copy(image_, order='C')
    pred = test_job(image_, anchors[0], anchors[1])[0,...]
    original_image_size = original_image.shape[0:2]
    bboxes = utils.postprocess_boxes(pred, original_image_size, cfg.TEST.INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
    bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
    return bboxes

def double_rect(rect):
    left, top, right, bottom = rect
    x = (left+right)/2
    y = (top+bottom)/2
    w = (right-left)
    h = (bottom - top)
    rect = [int(x-w),int(y-h),int(x+w),int(y+h)]
    return rect

def detect_video(video_filename, show = False, interest_rect = None):
    cap = cv2.VideoCapture(video_filename)
    assert cap.isOpened()
    bboxs=[]
    while True:
        res, img = cap.read()
        if not res:
            break
        start_time = time.time()
        detect_img = img
        if interest_rect is not None:
            detect_img = img.copy()
            image_h, image_w, _ = detect_img.shape
            l,t,r,b = double_rect(interest_rect)
            l = max(0,l)
            t = max(0,t)
            r = min(r,image_w)
            b = min(b, image_h)
            detect_img = detect_img[t:b,l:r,...]
            bbox = predict(detect_img)
            for box in bbox:
                box[0] = box[0] + l
                box[2] = box[2] + l
                box[1] = box[1] + t
                box[3] = box[3] + t
        else:
            bbox = predict(detect_img)

        bboxs.append(bbox)
        if show:
            img = utils.draw_bbox(img, bbox, ['car', ])
            consume = (time.time() - start_time)*1000
            consume = max(0,consume)
            cv2.imshow('video', img)
            cv2.waitKey(max(1, 45 - int(consume)))
    return bboxs

def write_to_file(bboxes, path):
    for i, bbox in enumerate(bboxes):
        filename = os.path.join(path, str(i+1)+'.txt')
        with open(filename, 'w') as file:
            for b in bbox:
                line = str(b[0])+' '+str(b[1])+' '+str(b[2])+' '+str(b[3])
                file.write(line + '\n')

def main():
    parser = argparse.ArgumentParser(description='flags for detect video')
    parser.add_argument(
        '-path',
        '--path',
        type=str,
        required=True
    )
    args = parser.parse_args()

    video_path = os.path.join(args.path, 'video')
    calib_path = os.path.join(args.path, 'calib')
    bboxes_output_path = os.path.join(args.path, 'detect_res')
    for filename in os.listdir(video_path):
        if not filename.endswith('.mp4'):
            continue
        print('process start '+filename)
        calib_txt = os.path.join(calib_path,filename[:-4]+'.txt')
        interest_rect = None
        if os.path.exists(calib_txt):
            with open(calib_txt) as file:
                line = file.readline()
                left, top, right, bottom = line.split()
                interest_rect = [int(left), int(top), int(right), int(bottom)]
        video_filename = os.path.join(video_path, filename)
        bboxes = detect_video(video_filename, False, interest_rect)
        output_path = os.path.join(bboxes_output_path, filename[:-4])
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        write_to_file(bboxes, output_path)
        print('process end '+filename)


if __name__ == '__main__':
    main()
