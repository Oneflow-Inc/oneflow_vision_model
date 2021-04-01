# import tensorflow as tf
import oneflow as flow
import numpy as np

class JointsMSELoss(object):
    def __init__(self):
        super(JointsMSELoss, self).__init__()

    def call(self, y_pred, target, target_weight):
        batch_size = y_pred.shape[0]
        num_of_joints = y_pred.shape[-1]

        pred = flow.reshape(x=y_pred, shape=(batch_size, -1, num_of_joints))

        heatmap_pred_list = []
        for i in range(num_of_joints):
          tensor = flow.slice(pred, begin=[None, None, i*1], size=[None, None, 1])
          heatmap_pred_list.append(tensor)
        

        gt = flow.reshape(x=target, shape=(batch_size, -1, num_of_joints))

        heatmap_gt_list = []
        for i in range(num_of_joints):
          tensor = flow.slice(gt, begin=[None, None, i*1], size=[None, None, 1])
          heatmap_gt_list.append(tensor)

        loss = 0.0
        for i in range(num_of_joints):
            heatmap_pred = flow.squeeze(heatmap_pred_list[i])
            heatmap_gt = flow.squeeze(heatmap_gt_list[i])

            y_true = heatmap_pred * flow.reshape(flow.slice(target_weight, begin=[None,i*1, None], size=[None,1,None]),[batch_size,1])

            y_pred = heatmap_gt * flow.reshape(flow.slice(target_weight, begin=[None,i*1, None], size=[None,1,None]),[batch_size,1])

            loss += 0.5 * flow.nn.MSELoss(y_true, y_pred, reduction="mean")
            
        return loss / num_of_joints


