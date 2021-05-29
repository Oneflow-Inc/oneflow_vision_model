from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES = "../data/classes/car.names"
__C.YOLO.CLASS_NUM = 1
__C.YOLO.ANCHORS = "../data/anchors/anchors.txt"
__C.YOLO.STRIDES = [16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "../data/dataset/train_of.txt"
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.INPUT_SIZE = [416,]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.MAX_BBOX_PER_SCALE = 150
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.WARMUP_EPOCHS = 4
__C.TRAIN.EPOCHS = 500
__C.TRAIN.BATCH_NUM_PER_EPOCH = None
__C.TRAIN.SAVE_MODEL_PATH = '../checkpoint/'
__C.TRAIN.INITIAL_WEIGHT = None
__C.TRAIN.FOCUS_LOSS_ALPHA = 0.25
__C.TRAIN.FOCUS_LOSS_GAMMA = 2
__C.TRAIN.LOSS_GIOU_ALPHA = 1
__C.TRAIN.LOSS_CONF_ALPHA = 1
__C.TRAIN.LOSS_PRED_ALPHA = 1

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "../data/dataset/val_of.txt"
__C.TEST.BATCH_SIZE = 1
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = "../saved/yolov3_snapshot"
__C.TEST.SHOW_LABEL = True
# __C.TEST.SCORE_THRESHOLD = 0.3
# __C.TEST.IOU_THRESHOLD = 0.45
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45
