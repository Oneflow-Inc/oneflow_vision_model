import oneflow as flow
import oneflow.typing as tp
import numpy as np

def _get_regularizer(model_name):
    #all decay
    return flow.regularizers.l2(0.00004)

def _get_initializer(model_name):
    if model_name == "weight":
        return flow.variance_scaling_initializer(2.0, mode="fan_in", distribution="random_normal", data_format="NCHW")
    elif model_name == "bias":
        return flow.zeros_initializer()
    elif model_name == "gamma":
        return flow.ones_initializer()
    elif model_name == "beta":
        return flow.zeros_initializer()
    elif model_name == "dense_weight":
        return flow.random_normal_initializer(0, 0.01)

def _conv2d(name, x, filters, kernel_size, strides, num_group, padding="SAME", data_format="NCHW"):  # tested
    assert data_format=="NCHW", "Mobilenet does not support channel_last mode."
    weight_initializer = _get_initializer("weight")
    weight_regularizer=_get_regularizer("beta")  # review it

    shape = (filters, x.shape[1] // num_group, kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "-weight",
        shape=shape,
        dtype=x.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
        model_name="--weight",
        trainable=True,  # review it
    )
    return flow.nn.conv2d(x, weight, strides, padding, data_format, name=name, groups=num_group)

def _batch_norms(name, x, axis, momentum, epsilon, center=True, scale=True, trainable=True):  # tested
    return flow.layers.batch_normalization(
        name=name,
        inputs=x,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer = _get_initializer("beta"),
        gamma_initializer = _get_initializer("gamma"),
        beta_regularizer = _get_regularizer("beta"),
        gamma_regularizer = _get_regularizer("gamma"),
        trainable=trainable
    )

def hswish(x):
    out =  x * flow.nn.relu6(x + 3) / 6
    return out

def hsigmoid(x):
    out = flow.nn.relu6(x + 3) / 6
    return out

# valid numbers outputted
def SeModule(name, x, channel, reduction=4):
    N, C, H, W = x.shape

    y = flow.nn.avg_pool2d(x, ksize=[H, W], strides=None, padding="SAME")  # check here
    y = flow.flatten(y, start_dim=1, end_dim=-1)
    y = flow.layers.dense(
        y, 
        units=channel // reduction, 
        use_bias=False,
        kernel_initializer=_get_initializer("dense_weight"),
        bias_initializer=_get_initializer("bias"),
        kernel_regularizer=_get_regularizer("dense_weight"),
        bias_regularizer=_get_regularizer("bias"),
        name=name+"dense1a",
        )
    y = flow.math.relu(y)
    y = flow.layers.dense(
        y, 
        units=channel, 
        use_bias=False,
        kernel_initializer=_get_initializer("dense_weight"),
        bias_initializer=_get_initializer("bias"),
        kernel_regularizer=_get_regularizer("dense_weight"),
        bias_regularizer=_get_regularizer("bias"),
        name=name+"dense2",
        )
    y = hsigmoid(y)
    y = flow.expand_dims(input=y, axis=2)
    y = flow.expand_dims(input=y, axis=3)
    y_expand = flow.broadcast_like(y, x, broadcast_axes=(2, 3))
    out = x * y_expand
    return out

# valid number outputted
def small_unit(name, x, kernel_size=1, num_filter=1, strides=1, padding="SAME", num_group=1, data_format="NCHW", act=None):
    conv = _conv2d(name=name+"-small_unit", x=x, filters=num_filter, kernel_size=kernel_size, strides=strides,
            num_group=num_group, padding=padding, data_format=data_format)
    bn = _batch_norms(name+"-small_unit_bn", conv, axis=1, momentum=0.9, epsilon=1e-5)
    if act == "_relu":
        return flow.math.relu(bn)
    elif act == "_hswish":
        return hswish(bn)
    else:
        return bn

# valid numbers outputted. But may be strange.
def mnv3_unit(name, x, kernel_size, expansion, num_filter, shortcut, strides, act, sechannel, data_format="NCHW"):
    # num_exp_filter = int(round(num_in_filter * expansion_factor))
    y = small_unit(name+"-mnv3_unit1", x, kernel_size=1, num_filter=expansion, strides=1, padding="VALID", num_group=1, act=act)
    y = small_unit(name+"-mnv3_unit2", y, kernel_size=kernel_size, num_filter=expansion, strides=strides, 
        padding=([0, 0, kernel_size//2, kernel_size//2]), num_group=expansion, act=act)  # check the padding
    out = small_unit(name+"-mnv3_unit3", y, kernel_size=1, num_filter=num_filter, strides=1, padding="VALID", num_group=1, act=None)
    if sechannel != None:
        out = SeModule(name+"-semodule", out, sechannel)
    if shortcut:
        _x = small_unit(name+"-mnv3_unit_shortcut", x, kernel_size=1, num_filter=num_filter, strides=1, padding="VALID", num_group=1, act=None)
    return out

def MobileNetV3_Large(x, data_format="NCHW", num_class=1000):
    layer1 = small_unit("large-layer1", x, num_filter=16, kernel_size=3, strides=2, padding="SAME", num_group=1, data_format=data_format, act="_hswish")
    
    layerneck = mnv3_unit("large-neck1", layer1, 3, 16, 16, False, 1, "_relu", None)
    layerneck = mnv3_unit("large-neck2", layerneck, 3, 64, 24, False, 2, "_relu", None)
    layerneck = mnv3_unit("large-neck3", layerneck, 3, 72, 24, False, 1, "_relu", None)
    layerneck = mnv3_unit("large-neck4", layerneck, 5, 72, 40, False, 2, "_relu", 40) 
    layerneck = mnv3_unit("large-neck5", layerneck, 5, 120, 40, False, 1, "_relu", 40)
    layerneck = mnv3_unit("large-neck6", layerneck, 5, 120, 40, False, 1, "_relu", 40)
    layerneck = mnv3_unit("large-neck7", layerneck, 3, 240, 80, False, 2, "_hswish", None)
    layerneck = mnv3_unit("large-neck8", layerneck, 3, 200, 80, False, 1, "_hswish", None)
    layerneck = mnv3_unit("large-neck9", layerneck, 3, 184, 80, False, 1, "_hswish", None)
    layerneck = mnv3_unit("large-neck10", layerneck, 3, 184, 80, False, 1, "_hswish", None)
    layerneck = mnv3_unit("large-neck11", layerneck, 3, 480, 112, True, 1, "_hswish", 112)
    layerneck = mnv3_unit("large-neck12", layerneck, 3, 672, 112, False, 1, "_hswish", 112)
    layerneck = mnv3_unit("large-neck13", layerneck, 5, 672, 160, True, 1, "_hswish", 160)
    layerneck = mnv3_unit("large-neck14", layerneck, 5, 672, 160, False, 2, "_hswish", 160)
    layerneck = mnv3_unit("large-neck15", layerneck, 3, 960, 160, False, 1, "_hswish", 160)
    layer2 = small_unit("large-layer2", layerneck, num_filter=960, act="_hswish", padding="VALID")  # number > 1 exists

    layer_avg = flow.nn.avg_pool2d(layer2, ksize=[layer2.shape[2], layer2.shape[3]], strides=None, padding="VALID")  # review it 
    layer_view = flow.reshape(layer_avg, (layer_avg.shape[0], -1))  # review it

    dense3 = flow.layers.dense(
        layer_view, 
        units=1280, 
        use_bias=False,
        kernel_initializer=_get_initializer("dense_weight"),
        bias_initializer=_get_initializer("bias"),
        kernel_regularizer=_get_regularizer("dense_weight"),
        bias_regularizer=_get_regularizer("bias"),
        name="dense3-large",
        )
    bn3 = _batch_norms("bn3-large", dense3, axis=1, momentum=0.9, epsilon=1e-5)
    hs3 = hswish(bn3)
    dense4 = flow.layers.dense(
        hs3, 
        units=num_class, 
        use_bias=False,
        kernel_initializer=_get_initializer("dense_weight"),
        bias_initializer=_get_initializer("bias"),
        kernel_regularizer=_get_regularizer("dense_weight"),
        bias_regularizer=_get_regularizer("bias"),
        name="dense4-large",
    )

    return dense4

def MobileNetV3_Small(x, data_format="NCHW", num_class=1000):
    layer1 = small_unit("small-layer1", x, num_filter=16, kernel_size=3, strides=2, padding="SAME", num_group=1, data_format=data_format, act="_hswish")

    layerneck = mnv3_unit("small-neck1", layer1, 3, 16, 16, True, 1, "_relu", 16)
    layerneck = mnv3_unit("small-neck2", layerneck, 3, 72, 24, False, 2, "_relu", None)
    layerneck = mnv3_unit("small-neck3", layerneck, 3, 88, 24, False, 1, "_relu", None)
    layerneck = mnv3_unit("small-neck4", layerneck, 5, 96, 40, False, 2, "_hswish", 40)
    layerneck = mnv3_unit("small-neck5", layerneck, 5, 240, 40, True, 1, "_hswish", 40)
    layerneck = mnv3_unit("small-neck6", layerneck, 5, 240, 40, True, 1, "_hswish", 40)
    layerneck = mnv3_unit("small-neck7", layerneck, 5, 120, 48, False, 1, "_hswish", 48)
    layerneck = mnv3_unit("small-neck8", layerneck, 5, 144, 48, True, 1, "_hswish", 48)
    layerneck = mnv3_unit("small-neck9", layerneck, 5, 288, 96, False, 2, "_hswish", 96)
    layerneck = mnv3_unit("small-neck10", layerneck, 5, 576, 96, True, 1, "_hswish", 96)
    layerneck = mnv3_unit("small-neck11", layerneck, 5, 576, 96, True, 1, "_hswish", 96)

    layer2 = small_unit("small-layer2", layerneck, num_filter=576, act="_hswish")
    layer_avg = flow.nn.avg_pool2d(layer2, ksize=[layer2.shape[2], layer2.shape[3]], strides=None, padding="VALID")  # review it
    layer_view = flow.reshape(layer_avg, (layer_avg.shape[0], -1))  # review it
    dense3 = flow.layers.dense(layer_view, 1280)
    bn3 = _batch_norms("bn3-large", dense3, axis=1, momentum=0.9, epsilon=1e-5)
    hs3 = hswish(bn3)
    dense4 = flow.layers.dense(
        hs3, 
        units=num_class, 
        use_bias=False,
        kernel_initializer=_get_initializer("dense_weight"),
        bias_initializer=_get_initializer("bias"),
        kernel_regularizer=_get_regularizer("dense_weight"),
        bias_regularizer=_get_regularizer("bias"),
        name="dense4-large",
        )
    print(dense4.shape)

    return dense4

def Mobilenet_Large(input_data, args, trainable=True, training=True, num_classes=1000, prefix = ""):
    assert   args.channel_last==False, "Mobilenet does not support channel_last mode, set channel_last=False will be right!"
    data_format="NHWC" if args.channel_last else "NCHW"
    out = MobileNetV3_Large(input_data, data_format=data_format, num_class=num_classes)
    return out

def Mobilenet_Small(input_data, args, trainable=True, training=True, num_classes=1000, prefix = ""):
    assert   args.channel_last==False, "Mobilenet does not support channel_last mode, set channel_last=False will be right!"
    data_format="NHWC" if args.channel_last else "NCHW"
    out = MobileNetV3_Small(input_data, data_format=data_format, num_class=num_classes)
    return out
