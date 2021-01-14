import oneflow as flow
import oneflow.nn as nn
import oneflow.distribute as distribute_util


def meanshift(x, rgb_range, rgb_mean, rgb_std, sign=-1, name="Meanshift"):
    # Concat the rgb_std
    _new_constant_std_0 = flow.constant_scalar(rgb_std[0], dtype=flow.float32, name=name + "_std_0")
    _new_constant_std_1 = flow.constant_scalar(rgb_std[1], dtype=flow.float32, name=name + "_std_1")
    _new_constant_std_2 = flow.constant_scalar(rgb_std[2], dtype=flow.float32, name=name + "_std_2")
    _std = flow.concat(
        inputs=[_new_constant_std_0, _new_constant_std_1, _new_constant_std_2],
        axis=0,
    )

    _reshaped_std = flow.reshape(_std, (3, 1, 1, 1), name=name + "reshape_std")

    # Concat the rgb_mean
    _new_constant_mean_0 = flow.constant_scalar(rgb_mean[0], dtype=flow.float32, name=name + "_mean_0")
    _new_constant_mean_1 = flow.constant_scalar(rgb_mean[1], dtype=flow.float32, name=name + "_mean_1")
    _new_constant_mean_2 = flow.constant_scalar(rgb_mean[2], dtype=flow.float32, name=name + "_mean_2")

    _mean = flow.concat(
        inputs=[_new_constant_mean_0, _new_constant_mean_1, _new_constant_mean_2],
        axis=0,
    )

    _weight_ones = flow.constant(1.0, dtype=flow.float32, shape=(3, 3), name=name + "_ones")

    # Generate eye matrix

    # [[1, 0, 0],    [[0, 0, 0],
    #  [1, 1, 0], -   [1, 0, 0],
    #  [1, 1, 1]]     [1, 1, 0]]

    weight = flow.math.tril(_weight_ones, 0) - flow.math.tril(_weight_ones, -1)
    weight = flow.reshape(weight, shape=(3, 3, 1, 1), name=name + "_reshaped_weight")
    weight = flow.math.divide(weight, _reshaped_std)

    bias = sign * rgb_range * _mean
    bias = flow.math.divide(bias, _std)

    _conv = flow.nn.conv2d(x, filters=weight, strides=1, padding="SAME", name=name + "_mean_shift_conv")
    output = flow.nn.bias_add(_conv, bias, data_format="NCHW", name=name + "_addbias")
    return output


def _batch_norm(inputs, name, trainable=True, training=True):
    params_shape = [inputs.shape[1]]
    # Float32 required to avoid precision-loss when using fp16 input/output
    params_dtype = flow.float32 if inputs.dtype == flow.float16 else inputs.dtype
    if not flow.current_global_function_desc().IsTrainable() or not trainable:
        training = False
    with flow.scope.namespace(name):
        beta = flow.get_variable(
            name="beta",
            shape=params_shape,
            dtype=params_dtype,
            initializer=flow.zeros_initializer(),
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )
        gamma = flow.get_variable(
            name="gamma",
            shape=params_shape,
            dtype=params_dtype,
            initializer=flow.ones_initializer(),
            trainable=trainable,
            distribute=distribute_util.broadcast(),
        )
        moving_mean = flow.get_variable(
            name="moving_mean",
            shape=params_shape,
            dtype=params_dtype,
            initializer=flow.zeros_initializer(),
            trainable=False,
            distribute=distribute_util.broadcast(),
        )
        moving_variance = flow.get_variable(
            name="moving_variance",
            shape=params_shape,
            dtype=params_dtype,
            initializer=flow.ones_initializer(),
            trainable=False,
            distribute=distribute_util.broadcast(),
        )
    builder = (
        flow.user_op_builder(name)
        .Op("normalization")
        .Input("x", [inputs])
        .Input("moving_mean", [moving_mean])
        .Input("moving_variance", [moving_variance])
        .Input("gamma", [gamma])
        .Input("beta", [beta])
        .Output("y")
        .Attr("axis", 1)
        .Attr("epsilon", 1.001e-5)
        .Attr("training", training)
        .Attr("momentum", 0.997)
    )
    if trainable and training:
        builder = builder.Output("mean").Output("inv_variance")
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


def batch_norm(input, name, axis=1, reuse=False, trainable=True):
    # use separated BN from real and fake batch
    name = name+'_reuse' if reuse else name
    return _batch_norm(input, name, trainable=trainable)


def max_pool2d(input, size, strides, name, padding="VALID", data_format="NCHW", reuse=False):
    name = name+'_reuse' if reuse else name
    return flow.nn.max_pool2d(input, ksize=size, strides=strides, padding=padding, data_format=data_format, name=name)


init = flow.random_normal_initializer(stddev=0.02)


def conv1x1(
    input,
    filters=64,
    name="conv1x1",
    strides=1,
    trainable=True,
    reuse=False,
    use_bias=True,
):
    name_ = name if reuse == False else name + "_reuse"

    # (output_dim, k_h, k_w, input.shape[3]) if NHWC
    weight_shape = (filters, input.shape[1], 1, 1)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=init,
        trainable=trainable,
        reuse=reuse,
    )

    output = flow.nn.conv2d(
        input,
        weight,
        strides=strides,
        padding="valid",
        data_format="NCHW",
        name=name_,
    )

    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
            reuse=reuse,
        )

        output = flow.nn.bias_add(output, bias, "NCHW")
    return output


def conv3x3(
    input,
    filters=64,
    name="conv3x3",
    strides=1,
    trainable=True,
    reuse=False,
    use_bias=True,
):
    name_ = name if reuse == False else name + "_reuse"

    # (output_dim, k_h, k_w, input.shape[3]) if NHWC
    weight_shape = (filters, input.shape[1], 3, 3)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=init,
        trainable=trainable,
        reuse=reuse,
    )

    output = flow.nn.conv2d(
        input,
        weight,
        strides=strides,
        padding="same",
        data_format="NCHW",
        name=name_,
    )

    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=flow.constant_initializer(0.0),
            trainable=trainable,
            reuse=reuse,
        )

        output = flow.nn.bias_add(output, bias, "NCHW")
    return output


def residual_block(inputs, name, trainable=True):
    with flow.scope.namespace(name):
        conv1 = conv3x3(inputs, name="conv1", trainable=trainable)
        # prelu1 = flow.layers.prelu(bn1, name='prelu', trainable=trainable)
        relu1 = flow.math.relu(conv1)
        conv2 = conv3x3(relu1, name="conv2", trainable=trainable)

    return inputs + conv2


def residual_blocks(inputs, block_num, name, trainable=True):
    output = inputs
    # outputs = []
    for i in range(block_num):
        block_name = name + "block%d" % (i)
        output = residual_block(output, block_name, trainable=trainable)
        # outputs.append(output)
    return output


def SFE(x, trainable=True):
    x = conv3x3(x, 64, "SFE_0", trainable=trainable)
    x = flow.math.relu(x)
    x1 = x
    x = residual_blocks(x, 16, "rb_", trainable=trainable)
    x = conv3x3(x, 64, "SFE_1", trainable=trainable)
    x = x + x1
    return x


def CSFI2(x1, x2, trainable=True):
    x12 = flow.layers.upsample_2d(x1, (2, 2), interpolation='bilinear', name='upsanple2')
    x12 = conv1x1(x12, 64, "CSFI2_0", trainable=trainable)
    x12 = flow.math.relu(x12)
    x21 = conv3x3(x2, 64, "CSFI2_1", 2, trainable)
    x21 = flow.math.relu(x21)

    x1 = conv3x3(flow.concat((x1, x21), axis=1), 64, "CSFI2_2", trainable=trainable)
    x1 = flow.math.relu(x1)
    x2 = conv3x3(flow.concat((x2, x12), axis=1), 64, "CSFI2_3", trainable=trainable)
    x2 = flow.math.relu(x2)

    return x1, x2


def CSFI3(x1, x2, x3, trainable=True):
    x12 = flow.layers.upsample_2d(x1, (2, 2), interpolation='bilinear', name='upsanple4')
    x12 = conv1x1(x12, 64, "CSFI3_0", trainable=trainable)
    x12 = flow.math.relu(x12)
    x13 = flow.layers.upsample_2d(x1, (4, 4), interpolation='bilinear', name='upsanple5')
    x13 = conv1x1(x13, 64, "CSFI3_1", trainable=trainable)
    x13 = flow.math.relu(x13)

    x21 = conv3x3(x2, 64, "CSFI3_2", 2, trainable)
    x13 = flow.math.relu(x13)
    x23 = flow.layers.upsample_2d(x2, (2, 2), interpolation='bilinear', name='upsanple6')
    x23 = conv1x1(x23, 64, "CSFI3_3", trainable=trainable)
    x23 = flow.math.relu(x23)

    x31 = conv3x3(x3, 64, "CSFI3_4", 2, trainable)
    x31 = flow.math.relu(x31)
    x31 = conv3x3(x31, 64, "CSFI3_5", 2, trainable)
    x31 = flow.math.relu(x31)
    x32 = conv3x3(x3, 64, "CSFI3_6", 2, trainable)
    x32 = flow.math.relu(x32)

    x1 = conv3x3(flow.concat((x1, x21, x31), axis=1), 64, "CSFI3_7", trainable=trainable)
    x1 = flow.math.relu(x1)
    x2 = conv3x3(flow.concat((x2, x12, x32), axis=1), 64, "CSFI3_8", trainable=trainable)
    x2 = flow.math.relu(x2)
    x3 = conv3x3(flow.concat((x3, x13, x23), axis=1), 64, "CSFI3_9", trainable=trainable)
    x3 = flow.math.relu(x3)

    return x1, x2, x3


def MergeTail(x1, x2, x3, trainable=True):
    x13 = flow.layers.upsample_2d(x1, (4, 4), interpolation='bilinear', name='upsanple7')
    x13 = conv1x1(x13, 64, "MergeTail_0", trainable=trainable)
    x13 = flow.math.relu(x13)
    x23 = flow.layers.upsample_2d(x2, (2, 2), interpolation='bilinear', name='upsanple8')
    x23 = conv1x1(x23, 64, "MergeTail_1", trainable=trainable)
    x23 = flow.math.relu(x23)

    x = conv3x3(flow.concat((x3, x13, x23), axis=1), 64, "MergeTail_2", trainable=trainable)
    x = flow.math.relu(x)
    x = conv3x3(x, 32, "MergeTail_3", trainable=trainable)
    x = conv1x1(x, 3, "MergeTail_4", trainable=trainable)
    x = flow.clamp(x, -1, 1)

    return x


def PixelShuffle(input, r):
    b, c, h, w = input.shape
    assert c % (r * r) == 0
    new_c = int(c / (r * r))
    out = flow.reshape(input, [b, new_c, r * r, h, w])
    out = flow.reshape(out, [b * new_c, r, r, h, w])
    out = flow.transpose(out, [0, 3, 1, 4, 2])
    out = flow.reshape(out, [b, new_c, h * r, w * r])
    return out