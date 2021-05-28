import oneflow as flow
import oneflow.typing as tp
import numpy as np
from detector.yolov3_tiny import Yolov3_tiny
from detector.config import cfg
import detector.utils as utils
import cv2
from PIL import Image

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

def main(filename):
    flow.load_variables(flow.checkpoint.get(cfg.TEST.WEIGHT_FILE))

    image = cv2.imread(filename)
    bboxes = predict(image)
    image = utils.draw_bbox(image, bboxes, ['car',])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.show()

if __name__ == '__main__':
    filename = r'../data/test_img/171.jpg'
    main(filename)
    pass