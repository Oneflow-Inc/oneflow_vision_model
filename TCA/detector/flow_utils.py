import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util

# set up initialize i
# def _get_kernel_initializer():
#     return flow.variance_scaling_initializer(distribution="random_normal", data_format="NCHW")
#
# def _get_regularizer():
#     return flow.regularizers.l2(0.0005)
#
# def _get_bias_initializer():
#     return flow.zeros_initializer()

def _get_regularizer():
    #all decay
    return flow.regularizers.l2(0.00004)


def _get_initializer(model_name):
    if model_name == "weight":
        return flow.random_normal_initializer(stddev=0.01)
        # return flow.variance_scaling_initializer(2.0, mode="fan_out", distribution="random_normal", data_format="NCHW")
    elif model_name == "bias":
        return flow.zeros_initializer()
    elif model_name == "gamma":
        return flow.ones_initializer()
    elif model_name == "beta":
        return flow.zeros_initializer()
    elif model_name == "dense_weight":
        return flow.random_normal_initializer(0, 0.01)


def conv2d_layer(name,
                  input,
                  filters,
                  kernel_size=[3,3],
                  strides=1,
                  padding="SAME",
                  group_num=1,
                  data_format="NCHW",
                  dilation_rate=1,
                  activation=op_conf_util.kRelu,
                  use_bias=False,
                  weight_initializer=_get_initializer("weight"),
                  bias_initializer=_get_initializer("bias"),
                  # weight_regularizer=_get_regularizer(),
                  # bias_regularizer=_get_regularizer(),
                  trainable=True,
                  ):
    if data_format == "NCHW":
        weight_shape = (int(filters), int(input.shape[1]), int(kernel_size[0]), int(kernel_size[1]))
    elif data_format == "NHWC":
        weight_shape = (int(filters), int(kernel_size[0]), int(kernel_size[1]), int(input.shape[3]))
    else:
        raise ValueError('data_format must be "NCHW" or "NHWC".')
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
        # regularizer=weight_regularizer,
        trainable=trainable,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
            # regularizer=bias_regularizer,
            model_name="bias",
            trainable=trainable,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError
    return output


def _batch_norm(inputs, axis, momentum, epsilon, center=True, scale=True, trainable=True, name=None):
    if trainable:
        training = True
    else:
        training = False
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=flow.zeros_initializer(),
        gamma_initializer=flow.ones_initializer(),
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=trainable,
        training=training,
        name=name
    )


def _leaky_relu(input, alpha=None, name=None):
    return flow.nn.leaky_relu(input, alpha=alpha, name=name)


def conv_unit(data, num_filter=1, kernel=(1, 1), stride=(1, 1), padding="same", data_format="NCHW", use_bias=False,
              trainable=True, prefix=''):
    conv = conv2d_layer(name=prefix + '-conv', input=data, filters=num_filter, kernel_size=kernel, strides=stride,
                         padding=padding, data_format=data_format, dilation_rate=1, activation=None, use_bias=use_bias,
                         trainable=trainable)
    bn = _batch_norm(conv, axis=1, momentum=0.99, epsilon=1.0001e-5, trainable=trainable, name=prefix + '-bn')
    leaky_relu = _leaky_relu(bn, alpha=0.1, name=prefix + '-leakyRelu')
    return leaky_relu


def upsample(input, name=None, data_format="NCHW"):
    return flow.layers.upsample_2d(input, size=2, data_format=data_format, interpolation="nearest", name=name)


def max_pooling(data, kernel=2, stride=2, padding='same', data_format="NCHW", name=None):
    out = flow.nn.max_pool2d(
        input=data,
        ksize=kernel,
        strides=stride,
        padding=padding,
        data_format=data_format,
        name=name
    )
    return out
