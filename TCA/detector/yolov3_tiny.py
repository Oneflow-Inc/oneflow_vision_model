import numpy as np
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from flow_utils import *
from resnet import resnet50

class Yolov3_tiny:
    def __init__(self, cfg, trainable, data_format='NCHW'):
        self.class_num = cfg.YOLO.CLASS_NUM
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.strides = cfg.YOLO.STRIDES
        self.focus_loss_alpha = cfg.TRAIN.FOCUS_LOSS_ALPHA
        self.focus_loss_gamma = cfg.TRAIN.FOCUS_LOSS_GAMMA
        self.loss_giou_alpha = cfg.TRAIN.LOSS_GIOU_ALPHA
        self.loss_conf_alpha = cfg.TRAIN.LOSS_CONF_ALPHA
        self.loss_preb_alpha = cfg.TRAIN.LOSS_PRED_ALPHA
        self.trainable = trainable
        self.data_format = data_format

    def backbone(self, in_blob):
        '''
           backbone
        :param in_blob:  [N, 3, 416, 416]
        :return:  [[N, 256, 26, 26],[N, 1024, 13, 13]]
        '''
        backbone_descripts = [
            {'op': 'conv', 'kernal_size': 3, 'stride': 1, 'output_channel': 16},  # 416*416*16
            {'op': 'max_pool', 'kernal_size': 2, 'stride': 2},  # 208*208*16
            {'op': 'conv', 'kernal_size': 3, 'stride': 1, 'output_channel': 32},  # 208*208*32
            {'op': 'max_pool', 'kernal_size': 2, 'stride': 2},  # 104*104*16
            {'op': 'conv', 'kernal_size': 3, 'stride': 1, 'output_channel': 64},  # 104*104*64
            {'op': 'max_pool', 'kernal_size': 2, 'stride': 2},  # 52*52*64
            {'op': 'conv', 'kernal_size': 3, 'stride': 1, 'output_channel': 128},  # 52*52*128
            {'op': 'max_pool', 'kernal_size': 2, 'stride': 2},  # 26*26*128
            {'op': 'conv', 'kernal_size': 3, 'stride': 1, 'output_channel': 256, 'route': True},  # 26*26*256
            {'op': 'max_pool', 'kernal_size': 2, 'stride': 2},  # 13*13*256
            {'op': 'conv', 'kernal_size': 3, 'stride': 1, 'output_channel': 512},  # 13*13*512
            {'op': 'max_pool', 'kernal_size': 2, 'stride': 1},  # 13*13*512
            {'op': 'conv', 'kernal_size': 3, 'stride': 1, 'output_channel': 1024, 'route': True},  # 13*13*1024
        ]
        blob = in_blob
        routes = []
        # backbone
        for i, desc in enumerate(backbone_descripts):
            if desc['op'] == 'conv':
                blob = conv_unit(blob, num_filter=desc['output_channel'],
                                 kernel=[desc['kernal_size'], desc['kernal_size']],
                                 stride=[desc['stride'], desc['stride']],
                                 data_format=self.data_format, use_bias=False,
                                 trainable=self.trainable, prefix='yolo-backbone' + str(i))
            elif desc['op'] == 'max_pool':
                blob = max_pooling(blob, kernel=desc['kernal_size'], stride=desc['stride'],
                                   data_format=self.data_format,
                                   name='yolo-backbone' + str(i) + 'max_pool')

            if 'route' in desc and desc['route'] is not None and desc['route']:
                routes.append(blob)

        return routes

    # def backbone_resnet(self, in_blob):
    #     routes = resnet50(in_blob, self.trainable, self.trainable)
    #     return routes

    def network(self, in_blob):
        '''
            :param in_blob:  [N, 3, 416, 416]
            :return:   list[[N, 3 * (5 + num_class), 13, 13], [N, 3 * (5 + num_class), 26, 26]
        '''

        blobs = self.backbone(in_blob)

        # yolo_blob1
        blob = conv_unit(blobs[-1], num_filter=256,
                         kernel=[1, 1], stride=[1, 1],
                         data_format=self.data_format, use_bias=False,
                         trainable=self.trainable, prefix='yolo-detect1-layer1')
        blob1 = conv_unit(blob, num_filter=512,
                          kernel=[3, 3], stride=[1, 1],
                          data_format=self.data_format, use_bias=False,
                          trainable=self.trainable, prefix='yolo-detect1-layer2')
        output_channel = (self.class_num + 5) * self.anchor_per_scale
        # blob1 = conv2d_layer(name='yolo-detect1-pred', input=blob1, filters=output_channel, kernel_size=(1,1), strides=1,
        #                      padding='same', data_format=self.data_format, dilation_rate=1, activation=None,
        #                      use_bias=False, trainable=self.trainable)
        blob1 = conv2d_layer(name='yolo-detect1-pred', input=blob1, filters=output_channel, kernel_size=(1,1), strides=1,
                             padding='same', data_format=self.data_format, dilation_rate=1, activation=None,
                             use_bias=True, trainable=self.trainable)
        # yolo_blob2
        blob = conv_unit(blob, num_filter=128,
                         kernel=[1, 1], stride=[1, 1],
                         data_format=self.data_format, use_bias=False,
                         trainable=self.trainable, prefix='yolo-detect2-layer1')
        blob = upsample(blob, name='yolo-detect2-upsample')  # 26*26*128
        blob = flow.concat([blob, blobs[0]], name='yolo-detect2-concat', axis=1)  # 26*26*384
        blob = conv_unit(blob, num_filter=256,
                         kernel=[3, 3], stride=[1, 1],
                         data_format=self.data_format, use_bias=False,
                         trainable=self.trainable, prefix='yolo-detect2-layer2')
        blob = conv2d_layer(name='yolo-detect2-pred', input=blob, filters=output_channel, kernel_size=(1,1), strides=1,
                            padding='same', data_format=self.data_format, dilation_rate=1, activation=None,
                            use_bias=True, trainable=self.trainable)
        conv_sbbox = blob
        conv_lbbox = blob1

        return [conv_lbbox, conv_sbbox]

    def decode(self, feature_map, anchors, stride, prefix='yolo'):
        '''
            return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        :param feature_map: [N, H, W, 3 * (5 + num_class)]
        :param anchors: [3, 2]
        :param stride:
        :return: (x, y, w, h, score, probability)
           [pred_xywh, pred_conf, pred_prob]:  [N, H, W, 3, 4+1+class_num]
        '''
        # [N, H, W, 3, 5 + num_class]
        feature_map = flow.reshape(feature_map, shape=(feature_map.shape[0], feature_map.shape[1], feature_map.shape[2],
                                                       self.anchor_per_scale, -1))

        # shape: [N, H, W, 3, 2]
        box_centers = flow.slice(feature_map, begin=[None, None, None, None, 0], size=[None, None, None, None, 2])
        # shape: [N, H, W, 3, 2]
        box_sizes = flow.slice(feature_map, begin=[None, None, None, None, 2], size=[None, None, None, None, 2])
        # shape: [N, H, W, 3, 1]
        conf_logits = flow.slice(feature_map, begin=[None, None, None, None, 4], size=[None, None, None, None, 1])
        # shape: [N, H, W, 3, class_num]
        prob_logits = flow.slice(feature_map, begin=[None, None, None, None, 5],
                                 size=[None, None, None, None, feature_map.shape[-1] - 5])

        # obtain the x_y_offset
        grid_size = feature_map.shape[1:3]
        grid_x = flow.range(grid_size[1], dtype=flow.float32, name=prefix+'_decode_range1')
        grid_x = flow.expand_dims(grid_x, axis=0)
        like_tensor = flow.constant(value=1.0, dtype=flow.float32, shape=(grid_size[0], grid_size[1]))
        grid_x = flow.broadcast_like(grid_x, like_tensor,broadcast_axes=(0, ), name = prefix+'yolo_grid_x')
        grid_y = flow.range(grid_size[0], dtype=flow.float32, name=prefix+'_yolo_decode_range2')
        grid_y = flow.expand_dims(grid_y, axis=1)
        grid_y = flow.broadcast_like(grid_y, like_tensor,broadcast_axes=(1, ), name = prefix+'yolo_grid_y')
        x_offset = flow.expand_dims(grid_x, axis=-1)
        y_offset = flow.expand_dims(grid_y, axis=-1)
        #shape: [1, H, W, 1 ,2]
        x_y_offset = flow.concat([x_offset, y_offset], axis=-1)
        x_y_offset = flow.expand_dims(x_y_offset, axis=0)
        x_y_offset = flow.expand_dims(x_y_offset, axis=-2)

        pred_xy = (flow.math.sigmoid(box_centers) + x_y_offset) * stride
        pred_wh = (flow.math.exp(box_sizes) * anchors) * stride  # anchor relative to the feature map
        # shape: [N, H, W, 3, 4]
        pred_xywh = flow.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = flow.math.sigmoid(conf_logits)
        pred_prob = flow.math.sigmoid(prob_logits)

        pred = flow.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
        # shape:
        #  pred: [N, H, W, 3, 4+1+class_num]
        #  x_y_offset: [1, H, W, 1, 2]
        return pred, x_y_offset

    def bbox_giou(self, boxes1, boxes2):
        ''' (x, y, w, h)
        :param boxes1: [N, H, W, 3, 4]  (x, y, w, h)
        :param boxes2:  [N, H, W, 3, 4]  (x, y, w, h)
        :return: [N, H, W, 3, 1]
        '''

        def convert(box_xywh):
            box_xy = flow.slice(box_xywh, begin=[None, None, None, None, 0], size=[None, None, None, None, 2])
            box_wh = flow.slice(box_xywh, begin=[None, None, None, None, 2], size=[None, None, None, None, 2])
            box_lt = box_xy - box_wh * 0.5
            box_rb = box_xy + box_wh * 0.5
            box_lt = flow.math.minimum(box_lt, box_rb)
            box_rb = flow.math.maximum(box_lt, box_rb)
            return box_lt, box_rb

        boxes1_lt, boxes1_rb = convert(boxes1)
        boxes1_wh = boxes1_rb - boxes1_lt
        # boxes1_wh = flow.math.clip_by_value(boxes1_rb - boxes1_lt, min_value=0)
        boxes1_area = flow.slice(boxes1_wh, begin=[None, None, None, None, 0], size=[None, None, None, None, 1]) * \
                      flow.slice(boxes1_wh, begin=[None, None, None, None, 1], size=[None, None, None, None, 1])

        boxes2_lt, boxes2_rb = convert(boxes2)
        boxes2_wh = boxes2_rb - boxes2_lt
        # boxes2_wh = flow.math.clip_by_value(boxes2_rb - boxes2_lt, min_value=0)
        boxes2_area = flow.slice(boxes2_wh, begin=[None, None, None, None, 0], size=[None, None, None, None, 1]) * \
                      flow.slice(boxes2_wh, begin=[None, None, None, None, 1], size=[None, None, None, None, 1])

        left_up = flow.math.maximum(boxes1_lt, boxes2_lt)
        right_down = flow.math.minimum(boxes1_rb, boxes2_rb)

        inter_section_wh = flow.math.clip_by_value(right_down - left_up, min_value = 0.0)
        inter_area = flow.slice(inter_section_wh, begin=[None, None, None, None, 0], size=[None, None, None, None, 1]) * \
                     flow.slice(inter_section_wh, begin=[None, None, None, None, 1], size=[None, None, None, None, 1])
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

        enclose_left_up = flow.math.minimum(boxes1_lt, boxes2_lt)
        enclose_right_down = flow.math.maximum(boxes1_rb, boxes2_rb)
        enclose_wh = flow.math.clip_by_value(enclose_right_down - enclose_left_up, min_value = 0.0)
        enclose_area = flow.slice(enclose_wh, begin=[None, None, None, None, 0], size=[None, None, None, None, 1]) * \
                       flow.slice(enclose_wh, begin=[None, None, None, None, 1], size=[None, None, None, None, 1])
        giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-6)
        # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

        return giou

    def bbox_iou(self, boxes1, boxes2):
        '''
        :param boxes1: [N, H, W, 3, 1, 4]  (x, y, w, h)
        :param boxes2:  [N, 1, 1, 1, V 4]  (x, y, w, h)
        :return: [N, H, W, 3, V, 1]
        '''

        def convert(box_xywh):
            box_xy = flow.slice(box_xywh, begin=[None, None, None, None, None, 0],
                                size=[None, None, None, None, None, 2])
            box_wh = flow.slice(box_xywh, begin=[None, None, None, None, None, 2],
                                size=[None, None, None, None, None, 2])
            box_lt = box_xy - box_wh * 0.5
            box_rb = box_xy + box_wh * 0.5
            box_lt = flow.math.minimum(box_lt, box_rb)
            box_rb = flow.math.maximum(box_lt, box_rb)
            return box_lt, box_rb

        boxes1_lt, boxes1_rb = convert(boxes1)
        boxes1_wh = boxes1_rb - boxes1_lt
        boxes1_area = flow.slice(boxes1_wh, begin=[None, None, None, None, None, 0],
                                 size=[None, None, None, None, None, 1]) * \
                      flow.slice(boxes1_wh, begin=[None, None, None, None, None, 1],
                                 size=[None, None, None, None, None, 1])

        boxes2_lt, boxes2_rb = convert(boxes2)
        boxes2_wh = boxes2_rb - boxes2_lt
        boxes2_area = flow.slice(boxes2_wh, begin=[None, None, None, None, None, 0],
                                 size=[None, None, None, None, None, 1]) * \
                      flow.slice(boxes2_wh, begin=[None, None, None, None, None, 1],
                                 size=[None, None, None, None, None, 1])

        left_up = flow.math.maximum(boxes1_lt, boxes2_lt)
        right_down = flow.math.minimum(boxes1_rb, boxes2_rb)

        inter_section_wh = flow.math.clip_by_value(right_down - left_up, min_value = 0.0)
        inter_area = flow.slice(inter_section_wh, begin=[None, None, None, None, None, 0],
                                size=[None, None, None, None, None, 1]) * \
                     flow.slice(inter_section_wh, begin=[None, None, None, None, None, 1],
                                size=[None, None, None, None, None, 1])
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / (union_area + 1e-6)

        return iou

    def focal(self, target, actual):
        focal_loss = flow.math.abs(self.focus_loss_alpha + target - 1) * flow.math.pow(flow.math.abs(target - actual), self.focus_loss_gamma)
        # focal_loss = self.focus_loss_alpha * flow.math.pow(flow.math.abs(target - actual), self.focus_loss_gamma)
        return focal_loss

    def loss_layer(self, feature_map, pred, label, bboxes, stride, prefix = 'loss_layer'):
        '''

        :param feature_map: [N, H, W, 3*(5+class_num)]
        :param pred: [N, H, W, 3, 4+1+class_num]
        :param label:  [N, H, W, 3, 4+1+class_num]
        :param bboxes:  [N, V, 4]
        :param stride:
        :param anchor_per_scale:
        :return:
            giou_loss:
            conf_loss:
            prob_loss:
        '''
        feature_map = flow.reshape(feature_map, shape=(
            feature_map.shape[0], feature_map.shape[1], feature_map.shape[2], self.anchor_per_scale, -1))
        # shape: [N, H, W, 3, 1]
        raw_conf = flow.slice(feature_map, begin=[None, None, None, None, 4], size=[None, None, None, None, 1])
        # shape: [N, H, W, 3, class_num]
        raw_prob = flow.slice(feature_map, begin=[None, None, None, None, 5],
                              size=[None, None, None, None, feature_map.shape[-1] - 5])

        #  [N, H, W, 3, 4]
        pred_xywh = flow.slice(pred, begin=[None, None, None, None, 0], size=[None, None, None, None, 4])
        pred_conf = flow.slice(pred, begin=[None, None, None, None, 4], size=[None, None, None, None, 1])

        #flow.slice(label, begin=[None, None, None, None, 0], size=[None, None, None, None, 4])
        label_xywh = flow.slice(label, begin=[None, None, None, None, 0], size=[None, None, None, None, 4])
        respond_bbox = flow.slice(label, begin=[None, None, None, None, 4], size=[None, None, None, None, 1])
        label_prob = flow.slice(label, begin=[None, None, None, None, 5],
                                size=[None, None, None, None, label.shape[-1] - 5])
        # [N, H, W, 3, 1]
        giou = self.bbox_giou(pred_xywh, label_xywh)
        # label_w = flow.slice(label, begin=[None, None, None, None, 2], size=[None, None, None, None, 1])
        # label_h = flow.slice(label, begin=[None, None, None, None, 3], size=[None, None, None, None, 1])
        # bbox_loss_scale = 2.0 - 1.0 * label_w * label_h / ((stride * feature_map.shape[1]) ** 2)  #???
        # [N, H, W, 3, 1]
        # giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
        giou_loss = respond_bbox * (1 - giou)

        # [N, 1, 1, 1, V, 4]
        bboxes_ = flow.expand_dims(bboxes, axis = 1)
        bboxes_ = flow.expand_dims(bboxes_, axis = 1)
        bboxes_ = flow.expand_dims(bboxes_, axis = 1)
        # [N, H, W, 3, V]
        iou = self.bbox_iou(flow.expand_dims(pred_xywh, axis=-2),bboxes_)
        iou = flow.squeeze(iou, axis=[-1,])
        # [N, H, W, 3, 1]
        max_iou = flow.math.reduce_max(iou, axis=-1, keepdims=True)
        # respond_bgd = (1.0 - respond_bbox) * (max_iou < self.iou_loss_thresh)
        tmp = flow.math.less(max_iou, flow.constant_like(like=max_iou, value=self.iou_loss_thresh, dtype=flow.float32))
        # respond_bgd = (1.0 - respond_bbox) * tmp
        respond_bgd = flow.where(tmp, 1.0 - respond_bbox,
                                 flow.zeros_like(respond_bbox, dtype=flow.float32))
        # [N, H, W, 3, 1]
        # ce = flow.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=raw_conf)
        # alpha_t = respond_bbox*self.focus_loss_alpha+(1.0-respond_bbox)*(1.0-self.focus_loss_alpha)
        # conf_loss = alpha_t*flow.math.pow(1.0-flow.math.exp(flow.math.negative(ce)), self.focus_loss_gamma)*ce
        # conf_loss = (respond_bbox+respond_bgd)*conf_loss
        conf_focal = self.focal(respond_bbox, pred_conf)
        conf_loss = conf_focal * (
                respond_bbox * flow.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=raw_conf)
                +
                respond_bgd * flow.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=raw_conf)
        )
        # [N, H, W, 3, 1]
        prob_loss = respond_bbox * flow.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=raw_prob)

        #??
        # label_w = flow.slice(label, begin=[None, None, None, None, 2], size=[None, None, None, None, 1])
        # label_h = flow.slice(label, begin=[None, None, None, None, 3], size=[None, None, None, None, 1])
        # bbox_loss_scale = 2.0 - 1.0 * label_w * label_h / ((stride * feature_map.shape[1]) * (stride * feature_map.shape[2]))  #???
        # # [N, H, W, 3, 1]
        # giou_loss = respond_bbox * bbox_loss_scale * flow.smooth_l1_loss(prediction=pred_xywh, label=label_xywh)

        giou_loss = flow.math.reduce_mean(flow.math.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = flow.math.reduce_mean(flow.math.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = flow.math.reduce_mean(flow.math.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, feature_map_s, feature_map_l, label_sbbox, label_lbbox, true_sbbox, true_lbbox, anchors_s,
                     anchors_l):
        '''

        :param feature_map_s: [N, 3 * (5 + num_class), 13, 13]
        :param feature_map_l: [N, 3 * (5 + num_class), 26, 26]
        :param label_sbbox:  [N, 13, 13, 3, 4+1+class_num]
        :param label_lbbox:  [N, 26, 26, 3, 4+1+class_num]
        :param true_sbbox:  [N, V, 4]
        :param true_lbbox:  [N, V, 4]
        :param anchors_s: [3,2]
        :param anchors_l: [3,2]
        :return:
        '''
        # [N, H, W, 3 * (5 + num_class)]
        feature_map_s = flow.transpose(feature_map_s, perm=[0, 2, 3, 1])
        # [N, H, W, 3, 4+1+class_num]
        pred_s, _ = self.decode(feature_map_s, anchors_s, self.strides[0], prefix= 'decode_s')
        loss_sbbox = self.loss_layer(feature_map_s, pred_s, label_sbbox, true_sbbox, self.strides[0], prefix= 'loss_later_s')
        # [N, H, W, 3 * (5 + num_class)]
        feature_map_l = flow.transpose(feature_map_l, perm=[0, 2, 3, 1])
        # [N, H, W, 3, 4+1+class_num]
        pred_l, _ = self.decode(feature_map_l, anchors_l, self.strides[1], prefix= 'decode_l')
        loss_lbbox = self.loss_layer(feature_map_l, pred_l, label_lbbox, true_lbbox, self.strides[1], prefix= 'loss_later_l')

        giou_loss = loss_sbbox[0] + loss_lbbox[0]
        conf_loss = loss_sbbox[1] + loss_lbbox[1]
        prob_loss = loss_sbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss

    def train(self, images, label_sbbox, label_lbbox, true_sbbox, true_lbbox, anchors_s, anchors_l):
        '''

        :param images: [N, 3, H, W]
        :param label_sbbox: [N, 13, 13, 3, 4+1+class_num]
        :param label_lbbox: [N, 26, 26, 3, 4+1+class_num]
        :param true_sbbox: [N, V, 4]
        :param true_lbbox: [N, V, 4]
        :param anchors_s: [anchor_per_scale, 2]
        :param anchors_l: [anchor_per_scale, 2]
        :return:
        '''
        conv_lbbox, conv_sbbox = self.network(images)

        giou_loss, conf_loss, prob_loss = self.compute_loss(conv_sbbox, conv_lbbox, label_sbbox, label_lbbox,
                                                            true_sbbox, true_lbbox, anchors_s, anchors_l)
        total_loss = self.loss_giou_alpha * giou_loss + self.loss_conf_alpha * conf_loss + self.loss_preb_alpha * prob_loss
        return total_loss, giou_loss, conf_loss, prob_loss
        # return {
        #     'total_loss': total_loss,
        #     'giou_loss': giou_loss,
        #     'conf_loss': conf_loss,
        #     'prob_loss': prob_loss
        # }

    def predict(self, images, anchors_s, anchors_l):
        '''
        :param images: [N, 3, 416, 416]
        :param anchors_s: [anchor_per_scale, 2]
        :param anchors_l: [anchor_per_scale, 2]
        :return: [N, -1, 4+1+class_num]
            pred_bbox: [N, -1, 4]
            pred_conf: [N, -1, 1]
            pred_pred: [N, -1, class_num]
        '''
        conv_lbbox, conv_sbbox = self.network(images)
        conv_sbbox = flow.transpose(conv_sbbox, perm=[0, 2, 3, 1])
        conv_lbbox = flow.transpose(conv_lbbox, perm=[0, 2, 3, 1])
        pred_s,_ = self.decode(conv_sbbox, anchors_s, self.strides[0], prefix= 'decode_s')
        pred_l,_ = self.decode(conv_lbbox, anchors_l, self.strides[1], prefix= 'decode_l')
        pred_s = flow.reshape(pred_s, [pred_s.shape[0], -1, pred_s.shape[-1]])
        pred_l = flow.reshape(pred_l, [pred_l.shape[0], -1, pred_l.shape[-1]])
        pred = flow.concat([pred_s, pred_l], axis=-2)
        # pred_bbox = flow.slice(pred, begin=[None, None, 0], size=[None, None, 4])
        # pred_conf = flow.slice(pred, begin=[None, None, 4], size=[None, None, 1])
        # pred_pred = flow.slice(pred, begin=[None, None, 5], size=[None, None, pred.shape[-1]-5])
        # return pred_bbox, pred_conf, pred_pred
        return pred
