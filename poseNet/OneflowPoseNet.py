import oneflow as flow
import oneflow.typing as tp


def _get_kernel_initializer():
    return flow.variance_scaling_initializer(distribution="random_normal", data_format="NCHW")

def _get_regularizer():
    return flow.regularizers.l2(0.00005)

def _get_bias_initializer():
    return flow.zeros_initializer()

def conv2d_layer(
    input, filters, kernel_size, strides=1, padding="VALID",
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    name=None,
):
    output = flow.layers.conv2d(
        input, filters, kernel_size, strides, padding, dilation_rate=1,
        activation=flow.nn.relu,
        kernel_initializer=weight_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=weight_regularizer,
        bias_regularizer=bias_regularizer,
        name=name,
    )
    return output

def block35(in_blob):
    """Builds the 35x35 resnet block."""
    with flow.scope.namespace("Block35"):
        with flow.scope.namespace("Branch_0"):
            tower_conv = conv2d_layer(in_blob, 32, 1, padding="SAME", name="Conv2d_1x1", )
        with flow.scope.namespace("Branch_1"):
            tower_conv1_0 = conv2d_layer(in_blob, 32, 1, padding="SAME", name="Conv2d_0a_1x1", )
            tower_conv1_1 = conv2d_layer(tower_conv1_0, 32, 3, padding="SAME", name='Conv2d_0b_3x3', )
        with flow.scope.namespace("Branch_2"):
            tower_conv2_0 = conv2d_layer(in_blob, 32, 1, padding="SAME", name="Conv2d_0a_1x1", )
            tower_conv2_1 = conv2d_layer(tower_conv2_0, 48, 3, padding="SAME", name="Conv2d_0b_3x3", )
            tower_conv2_2 = conv2d_layer(tower_conv2_1, 64, 3, padding="SAME", name="Conv2d_0c_3x3", )

        mixed_B35 = []
        mixed_B35.append(tower_conv)
        mixed_B35.append(tower_conv1_1)
        mixed_B35.append(tower_conv2_2)
        mixed = flow.concat(values=mixed_B35, axis=1, name="concat")

        up = conv2d_layer(mixed, in_blob.shape[1], 1, padding="SAME", name="Conv2d_1x1", )
        scaled_up = up * 1.0

        in_blob += scaled_up
        in_blob = flow.nn.relu(in_blob)
    return in_blob

def block17(in_blob):
    """Builds the 17x17 resnet block."""
    with flow.scope.namespace("Block17"):
        with flow.scope.namespace("Branch_0"):
            tower_conv = conv2d_layer(in_blob, 192, 1, padding="SAME", name="Conv2d_1x1")
        with flow.scope.namespace("Branch_1"):
            tower_conv1_0 = conv2d_layer(in_blob, 128, 1, padding="SAME", name="Conv2d_0a_1x1")
            tower_conv1_1 = conv2d_layer(tower_conv1_0, 160, [1, 7], padding="SAME", name="Conv2d_0b_1x7")
            tower_conv1_2 = conv2d_layer(tower_conv1_1, 192, [7, 1], padding="SAME", name="Conv2d_0c_7x1")

        mixed_B17 = []
        mixed_B17.append(tower_conv)
        mixed_B17.append(tower_conv1_2)
        mixed = flow.concat(values=mixed_B17, axis=1, name="concat")

        up = conv2d_layer(mixed, in_blob.shape[1], 1, padding="SAME", name="Conv2d_1x1")

        scaled_up = up * 1.0
        in_blob += scaled_up
        in_blob = flow.nn.relu(in_blob)
    return in_blob

def block8(in_blob):
    """Builds the 8x8 resnet block."""
    with flow.scope.namespace("Block8"):
        with flow.scope.namespace("Branch_0"):
            tower_conv = conv2d_layer(in_blob, 192, 1, padding="SAME", name="Conv2d_1x1")
        with flow.scope.namespace("Branch_1"):
            tower_conv1_0 = conv2d_layer(in_blob, 192, 1, padding="SAME", name="Conv2d_0a_1x1")
            tower_conv1_1 = conv2d_layer(tower_conv1_0, 224, [1, 3], padding="SAME", name="Conv2d_0b_1x3")
            tower_conv1_2 = conv2d_layer(tower_conv1_1, 256, [3, 1], padding="SAME", name="Conv2d_0c_3x1")

        mixed_B8 = []
        mixed_B8.append(tower_conv)
        mixed_B8.append(tower_conv1_2)
        mixed = flow.concat(values=mixed_B8, axis=1, name="concat")

        up = conv2d_layer(mixed, in_blob.shape[1], 1, padding="SAME", name="Conv2d_1x1")

        scaled_up = up * 1.0
        in_blob += scaled_up
        in_blob = flow.nn.relu(in_blob)
    return in_blob


def poseNet(inputs, train=False):
    # 149 x 149 x 32
    net = conv2d_layer(inputs, 32, 3, 2, padding="VALID", name="Conv2d_1a_3x3",)
    # 147 x 147 x 32
    net = conv2d_layer(net, 32, 3, padding="VALID", name="Conv2d_2a_3x3",)
    # 147 x 147 x 64
    net = conv2d_layer(net, 64, 3, padding="SAME", name="Conv2d_2b_3x3", )
    # 73 x 73 x 64
    net = flow.nn.max_pool2d(net, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="MaxPool_3a_3x3",)
    # 73 x 73 x 80
    net = conv2d_layer(net, 80, 1, padding="VALID", name="Conv2d_3b_1x1",)
    # 71 x 71 x 192
    net = conv2d_layer(net, 192, 3, padding="VALID", name="Conv2d_4a_3x3",)
    # 35 x 35 x 192
    net = flow.nn.max_pool2d(net, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="MaxPool_5a_3x3")

    # 35 x 35 x 320
    with flow.scope.namespace("Mixed_5b"):
        with flow.scope.namespace("Branch_0"):
            tower_conv = conv2d_layer(net, 96, 1, padding="SAME", name="Conv2d_1x1",)
        with flow.scope.namespace("Branch_1"):
            tower_conv1_0 = conv2d_layer(net, 48, 1, padding="SAME", name="Conv2d_0a_1x1",)
            tower_conv1_1 = conv2d_layer(tower_conv1_0, 64, 5, padding="SAME", name="Conv2d_0b_5x5",)
        with flow.scope.namespace("Branch_2"):
            tower_conv2_0 = conv2d_layer(net, 64, 1, padding="SAME", name="Conv2d_0a_1x1",)
            tower_conv2_1 = conv2d_layer(tower_conv2_0, 96, 3, padding="SAME", name="Conv2d_0b_3x3",)
            tower_conv2_2 = conv2d_layer(tower_conv2_1, 96, 3, padding="SAME", name="Conv2d_0c_3x3",)
        with flow.scope.namespace('Branch_3'):
            tower_pool = flow.nn.avg_pool2d(net, ksize=3, strides=1, padding='SAME', data_format="NCHW", name="AvgPool_0a_3x3")
            tower_pool_1 = conv2d_layer(tower_pool, 64, 1, padding="SAME", name="Conv2d_0b_1x1",)

        Mixed_5b = []
        Mixed_5b.append(tower_conv)
        Mixed_5b.append(tower_conv1_1)
        Mixed_5b.append(tower_conv2_2)
        Mixed_5b.append(tower_pool_1)
        net = flow.concat(values=Mixed_5b, axis=1, name="concat")

    #net = flow.repeat(block35(net), 10, name="repeat")
    net = block35(net)

    #Bran1 8x8x320
    netB1 = conv2d_layer(net, 320, 3, strides=2, padding="SAME", name="conv_ls1")
    netB1 = flow.nn.max_pool2d(netB1, ksize=3, strides=2, padding="VALID", name='MaxPool_3x3_ls1')

    #17 x 17 x 1088
    with flow.scope.namespace("Mixed_6a"):
        with flow.scope.namespace("Branch_0"):
            tower_conv = conv2d_layer(net, 384, 3,  strides=2, padding="VALID", name="Conv2d_1a_3x3")
        with flow.scope.namespace("Branch_1"):
            tower_conv1_0 = conv2d_layer(net, 256, 1, padding="SAME", name="Conv2d_0a_1x1")
            tower_conv1_1 = conv2d_layer(tower_conv1_0, 256, 3, padding="SAME", name='Conv2d_0b_3x3')
            tower_conv1_2 = conv2d_layer(tower_conv1_1, 384, 3, strides=2, padding="VALID", name="Conv2d_1a_3x3")
        with flow.scope.namespace("Branch_2"):
            tower_pool = flow.nn.max_pool2d(net, ksize=3, strides=2, padding="VALID", name="MaxPool_1a_3x3")

        Mixed_6a = []
        Mixed_6a.append(tower_conv)
        Mixed_6a.append(tower_conv1_2)
        Mixed_6a.append(tower_pool)
        net = flow.concat(values=Mixed_6a, axis=1, name="concat")
    #net = flow.repeat(block17(net), 10, name="repeat2")
    net = block17(net)

    #Bran2 8x8x1088
    netB2 = conv2d_layer(net, 1088, 3, strides=2, padding="VALID", name="conv_ls2")

    # 8 x 8 x 2080
    with flow.scope.namespace("Mixed_7a"):
        with flow.scope.namespace("Branch_0"):
            tower_conv = conv2d_layer(net, 256, 1, padding="SAME", name="Conv2d_0a_1x1")
            tower_conv_1 = conv2d_layer(tower_conv, 384, 3, strides=2, padding="VALID", name="Conv2d_1a_3x3")
        with flow.scope.namespace("Branch_1"):
            tower_conv1 = conv2d_layer(net, 256, 1, padding="SAME", name="Conv2d_0a_1x1")
            tower_conv1_1 = conv2d_layer(tower_conv1, 288, 3, strides=2, padding="VALID", name="Conv2d_1a_3x3")
        with flow.scope.namespace("Branch_2"):
            tower_conv2 = conv2d_layer(net, 256, 1, padding="SAME", name="Conv2d_0a_1x1")
            tower_conv2_1 = conv2d_layer(tower_conv2, 288, 3, padding="SAME", name="Conv2d_0b_3x3")
            tower_conv2_2 = conv2d_layer(tower_conv2_1, 320, 3, strides=2, padding="VALID", name="Conv2d_1a_3x3")
        with flow.scope.namespace("Branch_3"):
            tower_pool = flow.nn.max_pool2d(net, ksize=3, strides=2, padding="VALID", name="MaxPool_1a_3x3")

        Mixed_7a = []
        Mixed_7a.append(tower_conv_1)
        Mixed_7a.append(tower_conv1_1)
        Mixed_7a.append(tower_conv2_2)
        Mixed_7a.append(tower_pool)
        net = flow.concat(values=Mixed_7a, axis=1, name="concat")
    #net = flow.repeat(block8(net), 10, name="repeat3")
    net = block8(net)

    #BranAll  8x8x3488
    Mixed_netB3 = []
    Mixed_netB3.append(netB1)
    Mixed_netB3.append(netB2)
    Mixed_netB3.append(net)
    netB3 = flow.concat(values=Mixed_netB3, axis=1, name="concatAll")

    #8x8x2080
    netB3 = conv2d_layer(netB3, 2080, 1, padding="VALID", name="conv_ls3")

    # 8 x 8 x 1536
    net = conv2d_layer(netB3, 1536, 1, padding="SAME", name="Conv2d_7b_1x1")

    kernel_size = net.shape[2:]

    net = flow.nn.avg_pool2d(net, kernel_size[0], kernel_size[1], padding='VALID', name="AvgPool_1a_8x8")
    reshape = flow.reshape(net, [net.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        5,
        activation=flow.nn.relu,
        kernel_initializer=_get_kernel_initializer(),
        name="dense",
    )

    if train:
        hidden = flow.nn.dropout(hidden, 0.8, name="dropout")
    return hidden

def testNet(inputs, train=False):
    # 149 x 149 x 32
    net = conv2d_layer(inputs, 32, 3, 2, padding="VALID", name="Conv2d_1a_3x3", )
    # 147 x 147 x 32
    net = conv2d_layer(net, 32, 3, padding="VALID", name="Conv2d_2a_3x3", )
    # 147 x 147 x 64
    net = conv2d_layer(net, 64, 3, padding="SAME", name="Conv2d_2b_3x3", )
    # 73 x 73 x 64
    net = flow.nn.max_pool2d(net, ksize=3, strides=2, padding="VALID", data_format="NCHW", name="MaxPool_3a_3x3", )
    # 1 x 1 x 1536
    kernel_size = net.shape[2:]
    net = flow.nn.avg_pool2d(net, kernel_size[0], kernel_size[1], padding='VALID', name="AvgPool_1a_8x8")
    reshape = flow.reshape(net, [net.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        #activation=flow.nn.relu,
        kernel_initializer=_get_kernel_initializer(),
        name="dense",
    )
    if train:
        hidden = flow.nn.dropout(hidden, 0.5, name="dropout")
    #return hidden
    return flow.layers.dense(hidden, 8, kernel_initializer=_get_kernel_initializer(), name="dense2")

def ofrecord_decode():
    # batch_size = 25
    batch_size = 32
    color_space = "RGB"
    ofrecord = flow.data.ofrecord_reader(
        "./poseData",
        batch_size=batch_size,
        # You should set the corresponding data_part_num
        data_part_num=5,
        # part_name_suffix_length=-1,
        random_shuffle=True,
        shuffle_after_epoch=True,
    )
    image = flow.data.OFRecordImageDecoderRandomCrop(
        ofrecord, "images", color_space=color_space, random_area=(0.95, 1.0), random_aspect_ratio=(0.99, 1.0)
    )
    labels = flow.data.OFRecordRawDecoder(
        ofrecord, "labels", shape=(1,), dtype=flow.int32
    )
    rsz, scale, new_size = flow.image.Resize(
        image, target_size=(299, 299), channels=3
    )
    normal = flow.image.CropMirrorNormalize(
        rsz,
        color_space=color_space,
        #mean=[0.0],
        #std=[255.0],
        output_dtype=flow.float,
    )
    return normal, labels

image = tp.Numpy.Placeholder((4,3,299,299), dtype=flow.float32)
labels = tp.Numpy.Placeholder((4,1), dtype=flow.float32)
@flow.global_function(type="train")
def train_job1(images:image, labels:labels) -> tp.Numpy:
    # images, labels = ofrecord_decode()
    # with flow.scope.placement("cpu", "0:0"):
    #     logits = poseNet(images, train=True)
    #     loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
    #         labels, logits, name="softmax_loss"
    #     )
    #     loss = flow.math.reduce_mean(loss)
    logits = poseNet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )
    loss = flow.math.reduce_mean(loss)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return loss

@flow.global_function(type="train")
def train_job() -> tp.Numpy:
    images, labels = ofrecord_decode()
    # with flow.scope.placement("cpu", "0:0"):
    #     logits = poseNet(images, train=True)
    #     loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
    #         labels, logits, name="softmax_loss"
    #     )
    #     loss = flow.math.reduce_mean(loss)
    logits = poseNet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )
    loss = flow.math.reduce_mean(loss)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return loss

if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()
    for epoch in range(20000):
        image, labels = read()
        loss = train_job()
        if epoch % 10 == 0: print(str(epoch)+":  "+str(loss))
    check_point.save('./model/models_PoseNet')