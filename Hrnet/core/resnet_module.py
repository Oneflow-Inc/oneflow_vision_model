# import tensorflow as tf
import oneflow as flow
import datetime
# global time=0

def _conv2d_layer(name,
        input,
        filters,
        kernel_size,
        strides=1,
        padding="VALID",
        groups_num=1,
        data_format="NHWC",
        # dilation_rate=1,
        # activation='Relu',
        use_bias=True,
        # use_bn=True,
        weight_initializer=flow.glorot_uniform_initializer(),
        bias_initializer=flow.zeros_initializer(),
        trainable=True,
        groups=1,
        ):
    # if isinstance(kernel_size, int):
    #     kernel_size_1 = kernel_size
    #     kernel_size_2 = kernel_size
    # if isinstance(kernel_size, tuple):
    #     kernel_size_1 = kernel_size[0]
    #     kernel_size_2 = kernel_size[1]
    # if data_format == "NHWC":
    #     weight_shape = (filters, kernel_size_1, kernel_size_2, input.shape[3])
    #     # weight_shape = (int(filters), int(input.shape[1]), int(kernel_size_1), int(kernel_size_2))
    #     # weight_shape = (filters, input.shape[1], kernel_size[0], kernel_size[0])
    # elif data_format == "NCHW":
    #     weight_shape = (filters, input.shape[1], kernel_size_1, kernel_size_2)
    #     # weight_shape = (filters, int(input.shape[1]), int(kernel_size[0]), int(kernel_size[0]))
    #     # weight_shape = (int(filters), int(kernel_size_2), int(kernel_size_1), int(input.shape[3]))
    # else:
    #     raise ValueError('data_format must be "NCHW" or "NHWC".')
    # weight = flow.get_variable(
    #     name + "-weight",
    #     shape=weight_shape,
    #     dtype=input.dtype,
    #     initializer=weight_initializer,
    #     trainable=trainable,
    # )
    # time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # output = flow.layers.conv2d(
    #     input=input, filters=weight, strides=strides, padding=padding, dilations=1,data_format=data_format, name=name, groups=groups
    # )
    
    # if use_bias:
    #     bias = flow.get_variable(
    #         name + "-bias",
    #         shape=(filters,),
    #         dtype=input.dtype,
    #         initializer=bias_initializer,
    #         model_name="bias",
    #         trainable=trainable,
    #     )
    #     output = flow.nn.bias_add(output, bias, data_format)

    # if activation is not None:
    #     if activation == 'Relu':
    #         output = flow.nn.relu(output)
    #     else:
    #         raise NotImplementedError

    return flow.layers.conv2d(
                input, filters, kernel_size, strides, padding,
                data_format=data_format, dilation_rate=1, groups=groups,
                activation=None, use_bias=use_bias,
                kernel_initializer=flow.xavier_normal_initializer(),
                bias_initializer=flow.zeros_initializer(),
                # kernel_regularizer=flow.variance_scaling_initializer(2.0, mode="fan_out", distribution="random_normal", data_format="NHWC"),
                # bias_regularizer=flow.zeros_initializer(),
                trainable=trainable, name=name)


def _batch_norm(inputs, momentum, epsilon, name, training=True):
    
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        # beta_initializer=flow.zeros_initializer(),
        # gamma_initializer=flow.ones_initializer(),
        # beta_regularizer=flow.zeros_initializer(),
        # gamma_regularizer=flow.ones_initializer(),
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=True,
        training=training,
        name=name
    )

def bottleneck_block(inputs, filters_num, strides=1, downsample=False,
                     name='bottleneck'):
    expansion = 4

    residual = inputs

    x = _conv2d_layer(f'{name}_conv1'+ str(time), inputs,  filters_num//expansion, 1, padding="SAME", use_bias=False)
    x = _batch_norm(x, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1'+ str(time), training=training)
    x = flow.nn.relu(x, name=f'{name}_relu1')
    x = _conv2d_layer(f'{name}_conv2'+ str(time), x, filters_num//expansion, 3, strides, padding="SAME", use_bias=False)
    x = _batch_norm(x, momentum=0.1, epsilon=1e-5, name=f'{name}_bn2'+ str(time), training=training)
    x = flow.nn.relu(x, name=f'{name}_relu2')

    x = _conv2d_layer(f'{name}_conv3'+ str(time), x, filters, 1, 1, use_bias=False)
    x = _batch_norm(x, momentum=0.1, epsilon=1e-5, name=f'{name}_bn3'+ str(time), training=training)


    if downsample:
        residual = _conv2d_layer(f'{name}_down_conv'+ str(time), inputs, filters, 1, strides, use_bias=False)
        residual = _batch_norm(residual, momentum=0.1, epsilon=1e-5, name=f'{name}_down_bn'+ str(time), training=training)

    output = flow.nn.relu(flow.math.add_n([x, residual],name=f'{name}_res'), name=f'{name}_out')

    return output

def transion_layer1(inputs, filters=[32, 64], name='stage1_transition'):

    x1 = _conv2d_layer(f'{name}_conv1' + str(time), inputs, filters[0], 3, 1, padding="SAME")
    x1 = _batch_norm(x1, momentum=0.1, epsilon=1e-5, training=training, name=f'{name}_bn1' + str(time))
    x1 = flow.nn.relu(x1, name=f'{name}_relu1')

    x2 = _conv2d_layer(f'{name}_conv2' + str(time), inputs, filters[1], 3, 2, padding="SAME", use_bias=False)
    x2 = _batch_norm(x2, momentum=0.1, epsilon=1e-5, name=f'{name}_bn2' + str(time), training=training)
    x2 = flow.nn.relu(x2, name=f'{name}_relu2')
    return [x1, x2]

def make_branch(inputs, filters, name='branch'):
    x = basic_block(inputs, filters, downsample=False, name=f'{name}_basic1')
    x = basic_block(x, filters, downsample=False, name=f'{name}_basic2')
    x = basic_block(x, filters, downsample=False, name=f'{name}_basic3')
    x = basic_block(x, filters, downsample=False, name=f'{name}_basic4')
    return x

def basic_block(inputs, filters_num, strides=1,training=True, downsample=False, name='basic'):
    expansion = 1
    residual = inputs

    x = _conv2d_layer(f'{name}_conv1'+ str(time), inputs, filters//expansion, 3, strides, padding="SAME")
    x = _batch_norm(x, momentum=0.1, epsilon=1e-5, training=training, name=f'{name}_bn1'+ str(time))
    x = flow.nn.relu(x)

    x = _conv2d_layer(f'{name}_conv2'+ str(time), x, filters//expansion, 3, 1, padding="SAME")
    x = _batch_norm(x, momentum=0.1, epsilon=1e-5, training=training, name=f'{name}_bn2'+ str(time))
    if downsample:
        residual = _conv2d_layer(f'{name}_down_conv'+ str(time), inputs, filters, 1, strides)
        residual = _batch_norm(residual, momentum=0.1, epsilon=1e-5, name=f'{name}_down_bn'+ str(time))

    output = flow.nn.relu(flow.math.add_n([x, residual],name=f'{name}_res'), name=f'{name}_out')

    return output

# def BasicBlock(name, inputs, filter_num, stride=1, training=True, **kwargs):
#     # residual = inputs
#     # global time
#     time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
#     conv1 = _conv2d_layer(name+"_1_"+str(time), inputs, filter_num, (3, 3), stride, "SAME")
#     bn1 = _batch_norm(conv1, momentum=0.1, epsilon=1e-5, training=training, name=name+"_2_"+str(time))
#     relu = flow.nn.relu(bn1)
#     conv2 = _conv2d_layer(name+"_3_"+str(time), relu, filter_num, (3, 3), 1, "SAME")
#     bn2 = _batch_norm(conv2, momentum=0.1, epsilon=1e-5, training=training, name=name+"_4_"+str(time))
#
#     if stride != 1:
#         residual = _conv2d_layer(name+"_5_"+str(time), inputs, filter_num, (1, 1), stride, "SAME")
#         residual = _batch_norm(residual, momentum=0.1, epsilon=1e-5, name=name+"_6_"+str(time))
#     else:
#         residual = lambda inputs: inputs
#
#     print
#     output = flow.nn.relu(flow.math.add(residual, bn2))
#     return output




# def BottleNeck(name, inputs, filter_num, stride=1, training=True, **kwargs):
#     time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S_%f')
#     residual = _conv2d_layer('0'+name+str(time), inputs, filter_num * 4, (1, 1), stride, "SAME",use_bias=False)
#     residual = _batch_norm(residual, momentum=0.1, epsilon=1e-5, name=name+'1'+str(time), training=training)
#
#     conv1 = _conv2d_layer('2'+name+str(time), inputs, filter_num, (1, 1), 1, "SAME", use_bias=False)
#     bn1 = _batch_norm(conv1, momentum=0.1, epsilon=1e-5, name='3'+name+ str(time), training=training)
#     relu1 = flow.nn.relu(bn1)
#     conv2 = _conv2d_layer('4'+name+str(time), relu1, filter_num, (3, 3), stride, "SAME", use_bias=False)
#     bn2 = _batch_norm(conv2, momentum=0.1, epsilon=1e-5, name='5'+name+ str(time), training=training)
#     relu2 = flow.nn.relu(bn2)
#     conv3 = _conv2d_layer('6'+name+str(time), relu2, filter_num * 4, (1, 1), 1, "SAME", use_bias=False)
#     bn3 = _batch_norm(conv3, momentum=0.1, epsilon=1e-5, name='7'+name+str(time), training=training)
#
#     output = flow.nn.relu(flow.math.add(residual, bn3))
#     return output

def fuse_layer1(inputs, filters=[32, 64], name='stage2_fuse'):
    x1, x2 = inputs

    x11 = x1

    x21 = _conv2d_layer(f'{name}_conv_2_1' + str(time), x2, filters[0], 1, 1,
                          use_bias=False)
    x21 = _batch_norm(x21, momentum=0.1, epsilon=1e-5, name=f'{name}_bn_2_1' + str(time), training=training)
    x21 = flow.layers.upsample_2d(x=x21,size=(2,2),data_format="NHMC", name=f'{name}_up_2_1')
    x1 = flow.nn.relu(flow.math.add_n([x11, x21],name=f'{name}_add1'), name=f'{name}_branch1_out')


    x22 = x2
    x12 = _conv2d_layer(f'{name}_conv1_2'+ str(time), x1, filters[1], 3, 2, padding="SAME",
                          use_bias=False)
    x12 = _batch_norm(x12, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_2' + str(time), training=training)
    x2 = flow.nn.relu(flow.math.add_n([x12, x22],name=f'{name}_add2'), name=f'{name}_branch2_out')

    return [x1, x2]

def transition_layer2(inputs, filters, name='stage2_transition'):
    x1, x2 = inputs

    x1 = _conv2d_layer(f'{name}_conv1'+ str(time), x1, filters[0], 3, 1, padding="SAME",
                          use_bias=False)
    x1 = _batch_norm(x1, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1' + str(time), training=training)
    x1 = flow.nn.relu(x1, name=f'{name}_relu1')

    x21 = _conv2d_layer(f'{name}_conv2'+ str(time), x2, filters[1], 3, 1, padding="SAME",
                          use_bias=False)
    x21 = _batch_norm(x21, momentum=0.1, epsilon=1e-5, name=f'{name}_bn2' + str(time), training=training)
    x21 = flow.nn.relu(x21, name=f'{name}_relu2')

    x22 = _conv2d_layer(f'{name}_conv3'+ str(time), x2, filters[2], 3, 2, padding="SAME",
                          use_bias=False)
    x22 = _batch_norm(x22, momentum=0.1, epsilon=1e-5, name=f'{name}_bn3' + str(time), training=training)
    x22 = flow.nn.relu(x22, name=f'{name}_relu3')
    return [x1, x21, x22]

def fuse_layer2(inputs, filters=[32, 64, 128], name='stage3_fuse'):
    x1, x2, x3 = inputs

    # branch 1
    x11 = x1

    x21 = _conv2d_layer(f'{name}_conv2_1'+ str(time), x2, filters[0], 1, 1,
                          use_bias=False)
    x21 = _batch_norm(x21, momentum=0.1, epsilon=1e-5, name=f'{name}_bn2_1' + str(time), training=training)
    x21 = flow.layers.upsample_2d(x=x21, size=(2, 2), data_format="NHMC")

    x31 = _conv2d_layer(f'{name}_conv3_1'+ str(time),x3, filters[0], 1, 1,
                          use_bias=False)
    x31 = _batch_norm(x31, momentum=0.1, epsilon=1e-5, name=f'{name}_bn3_1' + str(time), training=training)
    x31 = flow.layers.upsample_2d(x=x31, size=(4,4), data_format="NHMC", name=f'{name}_up3_1')

    x1 = flow.nn.relu(flow.math.add([x11, x21, x31],name=f'{name}_add1'), name=f'{name}_branch1_out')

    # branch 2
    x22 = x2

    x12 = _conv2d_layer(f'{name}_conv1_2'+ str(time),x1, filters[1], 3, 2, padding="SAME",
                          use_bias=False)
    x12 = _batch_norm(x12, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_2' + str(time), training=training)

    x32 = _conv2d_layer(f'{name}_conv3_2'+ str(time), x3, filters[1], 1, 1,
                           use_bias=False)
    x32 = _batch_norm(x32, momentum=0.1, epsilon=1e-5, name=f'{name}_bn3_2' + str(time), training=training)
    x32 = flow.layers.upsample_2d(x=x32, size=(2, 2), data_format="NHMC", name=f'{name}_up3_2')

    x2 = flow.nn.relu(flow.math.add([x12, x22, x32], name=f'{name}_add2'), name=f'{name}_branch2_out')

    # branch 3
    x33 = x3

    x13 = _conv2d_layer(f'{name}_conv1_3_1'+ str(time), x1, filters[0], 3, 2, padding="SAME",
                           use_bias=False)
    x13 = _batch_norm(x13, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_3_1' + str(time), training=training)
    x13 = flow.nn.relu(x13,name=f'{name}_relu1_3_1')
    x13 = _conv2d_layer(f'{name}_conv1_3_2'+ str(time), x13, filters[2], 3, 2, padding="SAME",
                           use_bias=False)
    x13 = _batch_norm(x13, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_3_2' + str(time), training=training)

    x23 = _conv2d_layer(f'{name}_conv2_3'+ str(time), x2, filters[2], 3, 2, padding="SAME",
                           use_bias=False)
    x23 = _batch_norm(x23, momentum=0.1, epsilon=1e-5, name=f'{name}_bn2_3' + str(time), training=training)

    x3 = flow.nn.relu(flow.math.add_n([x13, x23, x33],name=f'{name}_add3'), name=f'{name}_branch3_out')

    return [x1, x2, x3]

def transition_layer3(inputs, filters, name='stage3_transition'):
    x1, x2, x3 = inputs

    x1 = conv2d_bn(x1, filters[0], 3, 1, padding='same', activation='relu',
                   name=f'{name}_conv1')
    x2 = conv2d_bn(x2, filters[1], 3, 1, padding='same', activation='relu',
                   name=f'{name}_conv2')
    x31 = conv2d_bn(x3, filters[2], 3, 1, padding='same', activation='relu',
                    name=f'{name}_conv3')
    x32 = conv2d_bn(x3, filters[3], 3, 2, padding='same', activation='relu',
                    name=f'{name}_conv4')

    return [x1, x2, x31, x32]

def fuse_layer3(inputs, filters=[32, 64, 128, 256], name='stage4_fuse'):
    x1, x2, x3, x4 = inputs

    # branch 1
    x11 = x1

    x21 = _conv2d_layer(f'{name}_conv2_1'+ str(time), x2, filters[0], 1, 1,
                           use_bias=False)
    x21 = _batch_norm(x21, momentum=0.1, epsilon=1e-5, name=f'{name}_bn21' + str(time), training=training)
    x21 = flow.layers.upsample_2d(x=x21, size=(2, 2), data_format="NHMC", name=f'{name}_up2_1')

    x31 =  _conv2d_layer(f'{name}_conv3_1'+ str(time), x3, filters[0], 1, 1,
                           use_bias=False)
    x31 = _batch_norm(x31, momentum=0.1, epsilon=1e-5, name=f'{name}_bn31' + str(time), training=training)
    x31 = flow.layers.upsample_2d(x=x31, size=(4,4), data_format="NHMC", name=f'{name}_up3_1')

    x41 = _conv2d_layer(f'{name}_conv4_1'+ str(time), x4, filters[0], 1, 1,
                           use_bias=False)
    x41 = _batch_norm(x41, momentum=0.1, epsilon=1e-5, name=f'{name}_bn21' + str(time), training=training)
    x41 = flow.layers.upsample_2d(x=x41, size=(8, 8), data_format="NHMC", name=f'{name}_up4_1')

    x1 = flow.nn.relu(flow.math.add_n([x11, x21, x31, x41],name=f'{name}_add1'), name=f'{name}_branch1_out')

    # branch 2
    x22 = x2

    x12 = _conv2d_layer(f'{name}_conv1_2'+ str(time), x1, filters[1], 3, 2, padding="SAME",
                           use_bias=False)
    x12 = _batch_norm(x12, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_2' + str(time), training=training)

    x32 = _conv2d_layer(f'{name}_conv3_2'+ str(time), x3, filters[1], 1, 1,
                           use_bias=False)
    x32 = _batch_norm(x32, momentum=0.1, epsilon=1e-5, name=f'{name}_bn3_2' + str(time), training=training)
    x32 = flow.layers.upsample_2d(x=x32, size=(2,2), data_format="NHMC", name=f'{name}_up3_2')


    x42 = _conv2d_layer(f'{name}_conv4_2'+ str(time), x4, filters[1], 1, 1,
                           use_bias=False)
    x42 =  _batch_norm(x42, momentum=0.1, epsilon=1e-5, name=f'{name}_bn4_2' + str(time), training=training)
    x42 =  flow.layers.upsample_2d(x=x42, size=(4,4), data_format="NHMC", name=f'{name}_up4_2')

    x2 = flow.nn.relu(flow.math.add_n([x12, x22, x32, x42],name=f'{name}_add2'), name=f'{name}_branch2_out')

    # branch 3
    x33 = x3

    x13 = _conv2d_layer(f'{name}_conv1_3_1'+ str(time), x1, filters[0], 3, 2, padding="SAME",
                           use_bias=False)
    x13 = _batch_norm(x13, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_3_1' + str(time), training=training)
    x13 = flow.nn.relu(x13,name=f'{name}_relu1_3_1')

    x13 = _conv2d_layer(f'{name}_conv1_3_2'+ str(time), x13, filters[2], 3, 2, padding="SAME",
                           use_bias=False)
    x13 = _batch_norm(x13, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_3_2' + str(time), training=training)


    x23 = _conv2d_layer(f'{name}_conv2_3'+ str(time), x2, filters[2], 3, 2, padding="SAME",
                           use_bias=False)
    x23 = _batch_norm(x23, momentum=0.1, epsilon=1e-5, name=f'{name}_bn23' + str(time), training=training)

    x43 = _conv2d_layer(f'{name}_conv4_3'+ str(time), x4, filters[2], 1, 1,
                           use_bias=False)
    x43 = _batch_norm(x43, momentum=0.1, epsilon=1e-5, name=f'{name}_bn423' + str(time), training=training)
    x43 = flow.layers.upsample_2d(x=x43, size=(2,2), data_format="NHMC", name=f'{name}_up4_3')

    x3 = flow.nn.relu(flow.math.add_n([x13, x23, x33, x43],name=f'{name}_add3'), name=f'{name}_branch3_out')

    # branch 4
    x44 = x4

    x14 = _conv2d_layer(f'{name}_conv1_4_1'+ str(time), x1, filters[0], 3, 2, padding="SAME",
                           use_bias=False)
    x14 = _batch_norm(x14, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_4_1' + str(time), training=training)
    x14 = flow.nn.relu(x14,name=f'{name}_relu1_4_1')
    x14 = _conv2d_layer(f'{name}_conv1_4_2' + str(time), x14, filters[0], 3, 2, padding="SAME",
                        use_bias=False)
    x14 = _batch_norm(x14, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_4_2' + str(time), training=training)
    x14 = flow.nn.relu(x14, name=f'{name}_relu1_4_2')
    x14 = _conv2d_layer(f'{name}_conv1_4_3' + str(time), x14, filters[3], 3, 2, padding="SAME",
                        use_bias=False)
    x14 = _batch_norm(x14, momentum=0.1, epsilon=1e-5, name=f'{name}_bn1_4_3' + str(time), training=training)


    x24 = _conv2d_layer(f'{name}_conv2_4_1'+ str(time),x2, filters[1], 3, 2, padding="SAME",
                           use_bias=False)
    x24 = _batch_norm(x24, momentum=0.1, epsilon=1e-5, name=f'{name}_bn2_4_1' + str(time), training=training)
    x24 = flow.nn.relu(x24, name=f'{name}_relu2_4_1')
    x24 = _conv2d_layer(f'{name}_conv2_4_2' + str(time),x24, filters[3], 3, 2, padding="SAME",
                        use_bias=False)
    x24 = _batch_norm(x24, momentum=0.1, epsilon=1e-5, name=f'{name}_bn2_4_2' + str(time), training=training)


    x34 = _conv2d_layer(f'{name}_conv3_4'+ str(time),x3, filters[3], 3, 2, padding="SAME",
                           use_bias=False)
    x34 = _batch_norm(x34, momentum=0.1, epsilon=1e-5, name=f'{name}_bn3_4' + str(time), training=training)

    x4 = flow.nn.relu(flow.math.add_n([x14, x24, x34, x44],name=f'{name}_add4'), name=f'{name}_branch4_out')

    return [x1, x2, x3, x4]

def fuse_layer4(inputs, filters=32, name='final_fuse'):
    x1, x2, x3, x4 = inputs

    x11 = x1

    x21 = _conv2d_layer(f'{name}_conv2_1'+ str(time),x2, filters, 1, 1,
                           use_bias=False)
    x21 = _batch_norm(x21, momentum=0.1, epsilon=1e-5, name=f'{name}_bn2_1' + str(time), training=training)
    x21 = flow.layers.upsample_2d(x=x21, size=(2,2), data_format="NHMC", name=f'{name}_up2_1')

    x31 = _conv2d_layer(f'{name}_conv3_1'+ str(time),x3, filters, 1, 1,
                           use_bias=False)
    x31 = _batch_norm(x31, momentum=0.1, epsilon=1e-5, name=f'{name}_bn3_1' + str(time), training=training)
    x31 = flow.layers.upsample_2d(x=x31, size=(4,4), data_format="NHMC", name=f'{name}_up3_1')

    x41 = _conv2d_layer(f'{name}_conv4_1'+ str(time),x4, filters, 1, 1,
                           use_bias=False)
    x41 = _batch_norm(x41, momentum=0.1, epsilon=1e-5, name=f'{name}_bn4_1' + str(time), training=training)
    x41 = flow.layers.upsample_2d(x=x41, size=(8,8), data_format="NHMC", name=f'{name}_up4_1')

    x = flow.concat(inputs=[x11, x21, x31, x41],axis=-1,name=f'{name}_out')
    return x



def make_basic_layer(inputs, filter_num, blocks, stride=1, training=None):
    # global time
    # res_block = []
    time = datetime.datetime.now().strftime('_%f')
    # res_block = flow.math.add(BasicBlock('make_basic0', inputs, filter_num, stride=stride), res_block)
    # time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    x = BasicBlock('make_basic0'+str(time), inputs, filter_num, stride=stride, training=training)
    # time+=1
    # res_block = flow.math.add(res_block,x)
    for _ in range(1, blocks):
        # res_block = flow.math.add(BasicBlock('make_basic', res_block, filter_num, stride=1), res_block)
        x = BasicBlock('make_basic'+str(time)+str(_), x, filter_num, stride=1, training=training)
        # time+=time
        # res_block = flow.math.add(res_block,BasicBlock('make_basic', inputs, filter_num, stride=1))
    return x


def make_bottleneck_layer(inputs, filter_num, blocks, stride=1, training=None):
    # global time
    # res_block = tf.keras.Sequential()
    # res_block.add(BottleNeck(filter_num, stride=stride))
    time = datetime.datetime.now().strftime('_%f')
    # res_block = []
    # res_block = flow.math.add(BottleNeck('make_bottle0', inputs, filter_num, stride=stride), res_block)
    # time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    x = BottleNeck('make_bottle0'+str(time), inputs, filter_num, stride=stride, training=training)
    # time+=1
    # res_block = flow.math.add(res_block,x)
    for _ in range(1, blocks):
        # res_block = flow.math.add(BottleNeck('make_bottle0', res_block, filter_num, stride=1), res_block)
        x = BottleNeck('make_bottle'+str(time)+str(_), x, filter_num, stride=1, training=training)
        # time+=1
        # res_block = flow.math.add(res_block,BottleNeck('make_bottle0', inputs, filter_num, stride=1))
    return x
