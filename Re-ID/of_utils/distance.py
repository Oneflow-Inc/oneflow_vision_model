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

# -*- coding:utf-8 -*-

import oneflow as flow
import oneflow.math as math


def _compute_distance_matrix(input1, input2, metric='euclidean'):
    if metric == 'euclidean':
        distmat = _euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = _cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def _euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 : 2-D feature matrix.
        input2 : 2-D feature matrix.

    Returns:
        distance matrix.
    """
    m, n = input1.shape[0], input2.shape[0]
    temp1 = math.reduce_sum(math.pow(input1, flow.constant_like(input1, 2, dtype=flow.float32)), axis=1)
    temp2 = math.reduce_sum(math.pow(input2, flow.constant_like(input2, 2, dtype=flow.float32)), axis=1)
    shape_tensor1 = flow.constant(value=0.0, dtype=flow.float32, shape=(m, n))
    shape_tensor2 = flow.constant(value=0.0, dtype=flow.float32, shape=(n, m))
    temp1 = flow.broadcast_like(temp1, like=shape_tensor1, broadcast_axes=[1])
    temp2 = flow.transpose(flow.broadcast_like(temp2, like=shape_tensor2, broadcast_axes=[1]), perm=(1, 0))

    dismat = math.add(temp1, temp2)

    return math.add(dismat, math.multiply(-2, flow.matmul(input1, flow.transpose(input2, perm=(1, 0)))))


def _cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 : 2-D feature matrix.
        input2 : 2-D feature matrix.

    Returns:
        distance matrix.
    """
    input1_normed = math.l2_normalize(input1, axis=1)
    input2_normed = math.l2_normalize(input2, axis=1)
    distmat = math.subtract(1, flow.matmul(input1_normed, flow.transpose(input2_normed, perm=(1, 0))))
    return distmat
