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

import os
import numpy as np
from PIL import Image

from Scnet.utils import config as configs
from model import resnet50

parser = configs.get_parser()
args = parser.parse_args()
configs.print_args(args)

import oneflow as flow
import oneflow.typing as tp
from Scnet.utils.clsidx_to_labels import clsidx_2_labels

def load_image(image_path='data/img_red.png'):
    print(image_path)
    im = Image.open(image_path)
    im = im.resize((224, 224))
    im = im.convert('RGB')  # 有的图像是单通道的，不加转换会报错
    im = np.array(im).astype('float32')
    im = (im - args.rgb_mean) / args.rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, 'float32')


@flow.global_function("predict", flow.function_config())
def InferenceNet(images: tp.Numpy.Placeholder((1, 3, 224, 224), dtype=flow.float)) -> tp.Numpy:
    body,logits = resnet50(images, args, training=False)
    predictions = flow.nn.softmax(logits)
    return predictions


def main():

    flow.env.log_dir(args.log_dir)
    assert os.path.isdir(args.model_load_dir)
    check_point = flow.train.CheckPoint()
    check_point.load(args.model_load_dir)

    image = load_image(args.image_path)
    predictions = InferenceNet(image)
    clsidx = predictions.argmax()
    print(predictions.max(), clsidx_2_labels[clsidx])


if __name__ == "__main__":
    main()
