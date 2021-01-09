from __future__ import absolute_import
import oneflow as flow
import oneflow.nn as nn
import numpy as np
import datetime
import oneflow.math as math
import oneflow.typing as tp
from typing import Tuple

__all__ = ['TripletLoss']

def addmm(mat,mat1,mat2,beta=1,alpha=1):    
    temp=flow.matmul(mat,mat2)
    out=(beta*mat+alpha*temp)
    return out
class _CrossEntropyLoss():
    """Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \epsilon) \times y + \frac{\epsilon}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\epsilon` is a weight. When
    :math:`\epsilon = 0`, the loss function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        epsilon (float, optional): weight. Default is 0.1.
    """

    def __init__(self, num_classes, epsilon=0.1):
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs, label):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        one_hot_label = flow.one_hot(indices=flow.cast(label, flow.int32), depth=self.num_classes, axis=-1,
                                     dtype=flow.float32)
        log_probs = math.log(flow.nn.softmax(inputs, axis=1))
        targets = math.add(math.multiply((1 - self.epsilon), one_hot_label), (self.epsilon / self.num_classes))
        temp = math.multiply(log_probs, math.multiply(-1, targets))
        temp2 = math.reduce_mean(temp, axis=0)
        loss = math.reduce_sum(temp2)
        return loss

# class _TripletLoss(object):
#     """Triplet loss with hard positive/negative mining.
#     Reference:
#         Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
#     Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
#     Args:
#         margin (float, optional): margin for triplet. Default is 0.3.
#     """
#     def __init__(self, margin=0.3):
#         self.margin = margin​
#     def _MarginRankingLoss(self, input1, input2, target, reduction='mean'):
#         if reduction == 'none':
#             ret = flow.clip(
#                 math.add(self.margin, math.multiply(target, math.multiply(-1, math.subtract(input1, input2)))),
#                 min_value=0)
#         else:
#             ret = math.reduce_mean(flow.clip(
#                 math.add(self.margin, math.multiply(target, math.multiply(-1, math.subtract(input1, input2)))),
#                 min_value=0))
#         return ret
# ​
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
#             targets (torch.LongTensor): ground truth labels with shape (num_classes).
#         """
#         n = inputs.shape[0]
#         dist = math.reduce_sum(math.pow(inputs, flow.constant_like(inputs, 2, dtype=flow.float32)), axis=1)
# ​
#         shape_tensor = flow.constant(value=0.0, dtype=flow.float32, shape=(n, n))
#         dist = flow.broadcast_like(dist, like=shape_tensor, broadcast_axes=[1])
# ​
#         dist = math.add(dist, flow.transpose(dist, perm=(1, 0), batch_axis_non_change=True))
# ​
#         temp1 = math.multiply(-2, flow.matmul(inputs, flow.transpose(inputs, perm=(1, 0), batch_axis_non_change=True)))
#         dist = math.add(dist, temp1)
#         dist = math.sqrt(flow.clamp(dist, min_value=1e-12))
# ​
#         mask = math.equal(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
#                           flow.transpose(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
#                                          perm=(1, 0), batch_axis_non_change=True))
#         mask_rev = math.not_equal(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
#                                   flow.transpose(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
#                                                  perm=(1, 0), batch_axis_non_change=True))
# ​
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             temp_dist = flow.slice_v2(dist, [(i, i + 1, 1)])
#             temp_mask = flow.slice_v2(mask, [(i, i + 1, 1)])
#             temp_mask_rev = flow.slice_v2(mask_rev, [(i, i + 1, 1)])
#             dist_ap.append(math.reduce_max(flow.gather_nd(temp_dist, flow.where(temp_mask))))
#             dist_an.append(math.reduce_min(flow.gather_nd(temp_dist, flow.where(temp_mask_rev))))
#         dist_ap = flow.concat(dist_ap)
# ​
#         dist_an = flow.concat(dist_an)
# ​
#         y = flow.ones_like(dist_an)
# ​
#         return self._MarginRankingLoss(dist_an, dist_ap, y)

class _TripletLoss():
    """Triplet loss with hard positive/negative mining.
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    Args:
       margin (float, optional): margin for triplet. Default is 0.3."""
    def __init__(self, margin=0.3):
        self.margin = margin
    def _MarginRankingLoss(self, input1, input2, target, reduction='mean'):
        if reduction == 'none':
            ret = flow.clip(
                math.add(self.margin, math.multiply(target, math.multiply(-1, math.subtract(input1, input2)))),
                min_value=0)
        else:
            ret = math.reduce_mean(flow.clip(
                math.add(self.margin, math.multiply(target, math.multiply(-1, math.subtract(input1, input2)))),
                min_value=0))
        return ret

    def MarginRankingLoss(self,input1, input2, target, reduction='mean'):
        low_bound = flow.constant_like(target, 0, dtype= flow.float32)

        if reduction == 'none':
            ret = math.maximum(low_bound,
                        math.add(self.margin,math.multiply(target,math.multiply(-1,math.subtract(input1,input2)))))
        else:
            ret = math.reduce_mean(math.maximum(low_bound,
                        math.add(self.margin,math.multiply(target,math.multiply(-1,math.subtract(input1,input2))))))
        return ret
        
    def build(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.shape[0]
        dist = math.reduce_sum(math.pow(inputs, flow.constant_like(inputs, 2, dtype=flow.float32)), axis=1)
        shape_tensor = flow.constant(value=0.0, dtype=flow.float32, shape=(n, n))
        dist = flow.broadcast_like(dist, like=shape_tensor, broadcast_axes=[1])
        dist = math.add(dist, flow.transpose(dist, perm=(1, 0), batch_axis_non_change=True))
        temp1 = math.multiply(-2, flow.matmul(inputs, flow.transpose(inputs, perm=(1, 0), batch_axis_non_change=True)))
        dist = math.add(dist, temp1)
        dist = math.sqrt(flow.clamp(dist, min_value=1e-12))
        mask = math.equal(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
                          flow.transpose(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
                                         perm=(1, 0), batch_axis_non_change=True))
        mask_rev = math.not_equal(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
                                  flow.transpose(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
                                                 perm=(1, 0), batch_axis_non_change=True))
        dist_ap, dist_an = [], []
        for i in range(n):
            temp_dist = flow.slice_v2(dist, [(i, i + 1, 1)])
            temp_mask = flow.slice_v2(mask, [(i, i + 1, 1)])
            temp_mask_rev = flow.slice_v2(mask_rev, [(i, i + 1, 1)])
            dist_ap.append(math.reduce_max(flow.gather_nd(temp_dist, flow.where(temp_mask))))
            dist_an.append(math.reduce_min(flow.gather_nd(temp_dist, flow.where(temp_mask_rev))))
        dist_ap = flow.concat(dist_ap,0)
        dist_an = flow.concat(dist_an,0)
        y = flow.ones_like(dist_an)
       # return dist_an, dist_ap, y
        
        return self._MarginRankingLoss(dist_an, dist_ap, y)


class TestTripletLoss():
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='euclidean'):
        super(TestTripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
    def _MarginRankingLoss(self, input1, input2, target, reduction='mean'):
        if reduction == 'none':
            ret = flow.clip(
                math.add(self.margin, math.multiply(target, math.multiply(-1, math.subtract(input1, input2)))),
                min_value=0)
        else:
            ret = math.reduce_mean(flow.clip(
                math.add(self.margin, math.multiply(target, math.multiply(-1, math.subtract(input1, input2)))),
                min_value=0))
        return ret
    def forward(self, inputs, targets):
        n=inputs.shape[0]
        # Compute pairwise distance, replace by the official when merged
        tempname=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')    
        shape_tensor = flow.constant(value=0.0, dtype=flow.float32, shape=(n, n))
        if self.distance == 'euclidean':
            blob_2=flow.get_variable(
                "blob_2_"+tempname,
                shape=inputs.shape,
                initializer=flow.constant_initializer(2),
                dtype=inputs.dtype
            )
            dist=flow.math.pow(inputs,blob_2)
           
            dist=flow.math.reduce_sum(dist, axis=1, keepdims=True)
            dist=flow.broadcast_like(dist,shape_tensor)
            tempdist=flow.transpose(dist)
            dist=dist+tempdist
            inputs_t=flow.transpose(inputs)
            dist=addmm(dist,inputs,inputs_t,beta=1,alpha=-2)
            dist=flow.clamp(dist,min_value=1e-12)
            dist=flow.math.sqrt(dist)
        elif self.distance=='cosine':
            #fnorm=flow.math.l2_normalize(inputs, axis=1)
            fnorm=flow.math.reduce_mean(flow.math.divide(inputs,flow.math.l2_normalize(inputs, axis=1)),axis=1,keepdims=True)
        
            expand_fnorm=flow.broadcast_like(fnorm,like=inputs,broadcast_axes=[1])
            l2norm=flow.math.divide(inputs,expand_fnorm)
            l2norm_t=flow.transpose(l2norm,perm=(1, 0))
            dist=flow.math.negative(flow.matmul(l2norm,l2norm_t))
        # For each anchor, find the hardest positive and negative
        mask = math.equal(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
                          flow.transpose(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
                                         perm=(1, 0), batch_axis_non_change=True))
        mask_rev = math.not_equal(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
                                  flow.transpose(flow.broadcast_like(targets, like=shape_tensor, broadcast_axes=[1]),
                                                 perm=(1, 0), batch_axis_non_change=True))
        dist_ap, dist_an = [], []
        for i in range(n):
            temp_dist = flow.slice_v2(dist, [(i, i + 1, 1)])
            temp_mask = flow.slice_v2(mask, [(i, i + 1, 1)])
            temp_mask_rev = flow.slice_v2(mask_rev, [(i, i + 1, 1)])
            temp_dist_ap=flow.expand_dims(math.reduce_max(flow.gather_nd(temp_dist, flow.where(temp_mask))),0)
            temp_dist_an=flow.expand_dims(math.reduce_min(flow.gather_nd(temp_dist, flow.where(temp_mask_rev))),0)
            dist_ap.append(temp_dist_ap)
            dist_an.append(temp_dist_an)
        dist_ap = flow.concat(dist_ap,0)
        dist_an = flow.concat(dist_an,0)
        y = flow.ones_like(dist_an)
        return self._MarginRankingLoss(dist_an, dist_ap, y)
