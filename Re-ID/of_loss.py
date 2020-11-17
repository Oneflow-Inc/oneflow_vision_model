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
# Version: 0.0.1
# Author: puchazhong(zhonghw@zhejianglab.com)
# Data: 11/03/2020
import oneflow as flow
import numpy as np
import oneflow.typing as tp
import oneflow.math as math


class _TripletLoss():
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

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

    def forward(self, inputs, targets):
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
        dist_ap = flow.concat(dist_ap)

        dist_an = flow.concat(dist_an)

        y = flow.ones_like(dist_an)

        return self._MarginRankingLoss(dist_an, dist_ap, y)


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
