
import oneflow as flow
import oneflow.nn as nn
import oneflow.distribute as distribute_util

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