import oneflow as flow
import oneflow.distribute as distribute_util

init = flow.random_normal_initializer(stddev=0.02)

def conv2d(
    input,
    filters,
    size,
    name,
    strides=1,
    padding="same",
    trainable=True,
    reuse=False,
    const_init=False,
    use_bias=True,
):
    name_ = name if reuse == False else name + "_reuse"

    # (output_dim, k_h, k_w, input.shape[3]) if NHWC
    weight_shape = (filters, input.shape[1], size, size)
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
        padding=padding,
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

def deconv2d(
    input,
    filters,
    size,
    name,
    strides=2,
    trainable=True,
    reuse=False,
    const_init=False,
    use_bias=False,
):
    name_ = name if reuse == False else name + "_reuse"
    # weight : [in_channels, out_channels, height, width]
    weight_shape = (input.shape[1], filters, size, size)
    output_shape = (
        input.shape[0],
        input.shape[1],
        input.shape[2] * strides,
        input.shape[3] * strides,
    )

    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=init
        if not const_init
        else get_const_initializer(),
        trainable=trainable,
    )

    output = flow.nn.conv2d_transpose(
        input,
        weight,
        strides=[strides, strides],
        output_shape=output_shape,
        padding="SAME",
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
        )

        output = flow.nn.bias_add(output, bias, "NCHW")
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
    name =  name+'_reuse' if reuse else name
    return _batch_norm(input, name, trainable=trainable)


def avg_pool2d(input, name, size, strides, padding, reuse=False):
    name = name+'_reuse' if reuse else name
    return flow.nn.avg_pool2d(input, ksize=size, strides=strides, padding=padding, name=name)

def max_pool2d(input, size, strides, name, padding="VALID", data_format="NCHW", reuse=False):
    name = name+'_reuse' if reuse else name
    return flow.nn.max_pool2d(input, ksize=size, strides=strides, padding=padding, data_format=data_format, name=name)


def residual_block(inputs, name, filters=64, size=3, trainable=True, reuse=False):
    with flow.scope.namespace(name):
        conv1=conv2d(inputs, filters=filters, size=size, name="conv1", strides=1, trainable=trainable, reuse=reuse)
        bn1 = batch_norm(conv1, "bn1", trainable=trainable, reuse=reuse)
        # prelu1 = flow.layers.prelu(bn1, name='prelu', trainable=trainable)
        relu1 = flow.math.relu(bn1)
        conv2 = conv2d(relu1, filters, size=size, name="conv2", strides=1, trainable=trainable, reuse=reuse)
        bn2 = batch_norm(conv2, "bn2", trainable=trainable, reuse=reuse)

    return inputs + bn2

def residual_blocks(inputs, filters, block_num, trainable=True):
    output = inputs
    # outputs = []
    for i in range(block_num):
        block_name = "block%d" % (i)
        output = residual_block(output, block_name, filters=filters, trainable=trainable)
        # outputs.append(output)
    return output

def upsample_blocks(inputs, filters, block_num, trainable=True):
    output = inputs
    # outputs = []
    for i in range(block_num):
        block_name = "block%d" % (i)
        output = upsample_block(output, block_name, filters=filters, trainable=trainable)
        # outputs.append(output)
    return output

def upsample_block(inputs, name, filters, size=3, trainable=True, reuse=False):
    output = inputs
    with flow.scope.namespace(name):
        deconv = deconv2d(output, name="deconv", filters=filters, size=size, trainable=trainable, reuse=reuse)
        bn = batch_norm(deconv, name="bn", trainable=trainable)   
        # output = flow.layers.prelu(bn, name='prelu', trainable=trainable)
        output = flow.math.relu(bn)
    return output