import oneflow as flow
import oneflow.typing as tp
from typing import Tuple
import numpy as np
import time
import os
from detector.yolov3_tiny import Yolov3_tiny
from detector.dataset import Dataset
from detector.config import cfg

np.set_printoptions(threshold=np.inf)

train_label_sbbox_input_size = int(cfg.TRAIN.INPUT_SIZE[0]/cfg.YOLO.STRIDES[0])
train_label_lbbox_input_size = int(cfg.TRAIN.INPUT_SIZE[0]/cfg.YOLO.STRIDES[1])
train_output_channel = (cfg.YOLO.CLASS_NUM+5)*cfg.YOLO.ANCHOR_PER_SCALE

dataset = Dataset('train')
cfg.TRAIN.BATCH_NUM_PER_EPOCH = dataset.num_batchs

train_images = tp.Numpy.Placeholder((cfg.TRAIN.BATCH_SIZE, 3, cfg.TRAIN.INPUT_SIZE[0], cfg.TRAIN.INPUT_SIZE[0]))
train_label_sbbox = tp.Numpy.Placeholder((cfg.TRAIN.BATCH_SIZE, train_label_sbbox_input_size,
                                          train_label_sbbox_input_size, cfg.YOLO.ANCHOR_PER_SCALE, cfg.YOLO.CLASS_NUM+5))
train_label_lbbox = tp.Numpy.Placeholder((cfg.TRAIN.BATCH_SIZE, train_label_lbbox_input_size,
                                          train_label_lbbox_input_size, cfg.YOLO.ANCHOR_PER_SCALE, cfg.YOLO.CLASS_NUM+5))
train_true_sbbox = tp.Numpy.Placeholder((cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.MAX_BBOX_PER_SCALE, 4))
train_true_lbbox = tp.Numpy.Placeholder((cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.MAX_BBOX_PER_SCALE, 4))
anchors_s = tp.Numpy.Placeholder((cfg.YOLO.ANCHOR_PER_SCALE, 2))
anchors_l = tp.Numpy.Placeholder((cfg.YOLO.ANCHOR_PER_SCALE, 2))

func_config = flow.FunctionConfig()
model = Yolov3_tiny(cfg, trainable=True)

@flow.global_function(type="train", function_config=func_config)
def train_job(images: train_images, label_sbbox: train_label_sbbox, label_lbbox: train_label_lbbox,
              true_sbbox: train_true_sbbox, true_lbbox: train_true_lbbox,
              anchors_s: anchors_s, anchors_l: anchors_l
              ) -> Tuple[tp.Numpy, tp.Numpy, tp.Numpy, tp.Numpy]:
    total_loss, giou_loss, conf_loss, prob_loss = model.train(images, label_sbbox, label_lbbox, true_sbbox, true_lbbox, anchors_s, anchors_l)
    wramup_steps = cfg.TRAIN.WARMUP_EPOCHS * cfg.TRAIN.BATCH_NUM_PER_EPOCH
    warmup_scheduler = flow.optimizer.warmup.linear(wramup_steps, cfg.TRAIN.LEARN_RATE_INIT)
    end_steps = (cfg.TRAIN.EPOCHS-cfg.TRAIN.WARMUP_EPOCHS) * cfg.TRAIN.BATCH_NUM_PER_EPOCH
    lr_scheduler = flow.optimizer.CosineScheduler(base_lr=cfg.TRAIN.LEARN_RATE_INIT, steps=end_steps, alpha=0, warmup=warmup_scheduler)
    flow.optimizer.Adam(lr_scheduler).minimize(total_loss)
    # flow.optimizer.SGD(lr_scheduler).minimize([giou_loss, conf_loss, prob_loss])
    return total_loss, giou_loss, conf_loss, prob_loss

if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    if not cfg.TRAIN.INITIAL_WEIGHT:
        check_point.init()
    else:
        check_point.load(cfg.TRAIN.INITIAL_WEIGHT)
    fmt_str = "{:>12}  {:>12}  {:>12.3f}  {:>12.4f}  {:>12.4f}  {:>12.4f}  {:>12.4f}"
    print("{:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}".format('epoch','iter', 'time', 'giou_loss',
                                                                                      'conf_loss', 'prob_loss',
                                                                                      'total_loss'))

    global cur_time
    cur_time = time.time()
    for epoch in range(cfg.TRAIN.EPOCHS):
        for iter_, train_data in enumerate(dataset):
            total_loss, giou_loss, conf_loss, prob_loss = train_job(*train_data)
            if iter_%10==0:
                print(fmt_str.format(epoch, iter_, time.time() - cur_time,
                                     np.abs(giou_loss).mean(), np.abs(conf_loss).mean(),
                                     np.abs(prob_loss).mean(), np.abs(total_loss).mean()))
                cur_time = time.time()
        # check_point.save(os.path.join(cfg.TRAIN.SAVE_MODEL_PATH, "yolov3_snapshot_") + str(epoch + 1))
        if (epoch+1)%50==0:
            check_point.save(os.path.join(cfg.TRAIN.SAVE_MODEL_PATH, "yolov3_snapshot_") + str(epoch + 1))