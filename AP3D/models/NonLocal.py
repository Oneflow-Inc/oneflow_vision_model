import oneflow as flow
import oneflow.nn as nn
import datetime
def _get_kernel_initializer():
    return flow.variance_scaling_initializer(distribution="random_normal", data_format="NCDHW")

def _get_regularizer():
    return flow.regularizers.l2(0.00005)

def _get_bias_initializer():
    return flow.zeros_initializer()
def conv3d_layer(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCDHW",
    dilation_rate=1,
    activation=None,
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    trainable=True
):
    if isinstance(kernel_size,int):
        kernel_size_1=kernel_size
        kernel_size_2 = kernel_size
        kernel_size_3 = kernel_size
    if isinstance(kernel_size,list):
        kernel_size_1=kernel_size[0]
        kernel_size_2=kernel_size[1]
        kernel_size_3=kernel_size[2]
    weight_shape=(filters,inputs.shape[1]//groups,kernel_size_1,kernel_size_2,kernel_size_3)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
        trainable=trainable
    )
 
    output=flow.nn.conv3d(
         inputs, weight, strides, padding, data_format, dilation_rate, groups, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=inputs.dtype,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output
def conv2d_layer(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    trainable=True
):
    if isinstance(kernel_size, int):
        kernel_size_1 = kernel_size
        kernel_size_2 = kernel_size
    if isinstance(kernel_size, list):
        kernel_size_1 = kernel_size[0]
        kernel_size_2 = kernel_size[1]

    weight_shape = (filters, inputs.shape[1]//groups, kernel_size_1, kernel_size_2)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
        trainable=trainable
    )
    output = flow.nn.conv2d(
        inputs, weight, strides, padding, data_format, dilation_rate, groups, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=inputs.dtype,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output

def conv1d_layer(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    trainable=True
):
    if isinstance(kernel_size, int):
        kernel_size_1 = kernel_size

    if isinstance(kernel_size, list):
        kernel_size_1 = kernel_size[0]


    weight_shape = (filters, inputs.shape[1]//groups,kernel_size_1)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
        trainable=trainable
    )
    output = flow.nn.conv1d(
        inputs, weight, strides, padding, data_format, dilation_rate, groups, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=inputs.dtype,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output


def conv3d_layer_with_bn(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCDHW",
    dilation_rate=1,
    activation=None,
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    use_bn=True,
    trainable=True
):
    output = conv3d_layer(name=name,
                          inputs=inputs,
                          filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          dilation_rate=dilation_rate,
                          activation=activation,
                          use_bias=use_bias,
                          groups=groups,
                          weight_initializer=weight_initializer,
                          bias_initializer=bias_initializer,
                          weight_regularizer=weight_regularizer,
                          bias_regularizer=bias_regularizer)
    
    if use_bn:
        tempname=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')    

        output = flow.layers.batch_normalization(inputs=output,
                                                 axis=1,
                                                 momentum=0.997,
                                                 epsilon=1.001e-5,
                                                 center=True,
                                                 scale=True,
                                                 trainable=trainable,
                                                 name=name +"_"+tempname +"_bn")
    return output


def conv2d_layer_with_bn(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCHW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    use_bn=True,
    trainable=True
):
    output = conv2d_layer(name=name,
                          inputs=inputs,
                          filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          dilation_rate=dilation_rate,
                          activation=activation,
                          use_bias=use_bias,
                          groups=groups,
                          weight_initializer=weight_initializer,
                          bias_initializer=bias_initializer,
                          weight_regularizer=weight_regularizer,
                          bias_regularizer=bias_regularizer)

    if use_bn:
        output = flow.layers.batch_normalization(inputs=output,
                                                 axis=1,
                                                 momentum=0.997,
                                                 epsilon=1.001e-5,
                                                 center=True,
                                                 scale=True,
                                                 trainable=trainable,
                                                 name=name + "_bn")
    return output

def conv1d_layer_with_bn(
    name,
    inputs,
    filters,
    kernel_size=3,
    strides=1,
    padding="VALID",
    data_format="NCW",
    dilation_rate=1,
    activation="Relu",
    use_bias=False,
    groups=1,
    weight_initializer=_get_kernel_initializer(),
    bias_initializer=_get_bias_initializer(),
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    use_bn=True,
    trainable=True
):
    output = conv1d_layer(name=name,
                          inputs=inputs,
                          filters=filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          dilation_rate=dilation_rate,
                          activation=activation,
                          use_bias=use_bias,
                          groups=groups,
                          weight_initializer=weight_initializer,
                          bias_initializer=bias_initializer,
                          weight_regularizer=weight_regularizer,
                          bias_regularizer=bias_regularizer)

    if use_bn:
        output = flow.layers.batch_normalization(inputs=output,
                                                 axis=1,
                                                 momentum=0.997,
                                                 epsilon=1.001e-5,
                                                 center=True,
                                                 scale=True,
                                                 trainable=trainable,
                                                 name=name + "_bn")
    return output


class NonLocalBlockND(object):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True,trainable=True,time=0):
        super(NonLocalBlockND,self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.bn_layer=bn_layer
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.trainable=trainable
        self.time=str(time)
    def build_maxpool(self,dimension,inputs):
        if dimension==3:
            return nn.max_pool3d(
                input=inputs,
                ksize=[1,2,2],
                data_format='NCDHW',
                strides=None,
                padding="VALID"
            )
        elif dimension==2:
            return nn.max_pool2d(
                input=inputs,
                kszie=2,
                data_format="NCHW",
                strides=None,
                padding="VALID"
            )
        else:
            return nn.max_pool1d(
                input=inputs,
                kszie=2,
                data_format="NCW",
                strides=None,
                padding="VALID"
            )
    def build_conv(self,dimension,name,inputs,bn=False,trainable=True):
        init=flow.random_normal_initializer(mean=0, stddev=1)
        #init=flow.kaiming_initializer(shape=inputs.shape,mode="fan_out",nonlinearity="relu")
        if dimension==3:
            if bn==False:
                return conv3d_layer(
                    "conv3d_layer_"+name+"_"+self.time,inputs,self.inter_channels,
                    kernel_size=1, strides=1, use_bias=True,weight_initializer=init,
                    trainable=trainable
                )
            else:
                return conv3d_layer_with_bn(
                    "conv3d_layer_"+name+"_"+self.time,inputs,self.in_channels,
                    kernel_size=1,strides=1,use_bias=True,weight_initializer=init,
                    trainable=trainable
                )

        elif dimension==2:
            if bn==False:
                return conv2d_layer(
                    "conv2d_layer_"+name+"_"+self.time,inputs,self.inter_channels,
                    kernel_size=1, strides=1,  use_bias=True,weight_initializer=init,
                    trainable=trainable
                )
            else:
                 return conv2d_layer_with_bn(
                    "conv2d_layer_"+name+"_"+self.time,inputs,self.in_channels,
                    kernel_size=1,strides=1,use_bias=True,weight_initializer=init,
                    trainable=trainable
                )
        else:
            if bn==False:
                return conv1d_layer(
                    "conv1d_layer_"+name+"_"+self.time,inputs,self.inter_channels,
                    kernel_size=1, strides=1, use_bias=True,weight_initializer=init,
                    trainable=trainable
                )
            else:
                return conv1d_layer_with_bn(
                    "conv1d_layer_"+name+"_"+self.time,inputs,self.in_channels,
                    kernel_size=1,strides=1,use_bias=True,weight_initializer=init,
                    trainable=trainable
                )

        

    def build_network(self,inputs):
        batch_size=inputs.shape[0]
        g=self.build_conv(self.dimension,"g",inputs,bn=False,trainable=self.trainable)
        g=self.build_maxpool(self.dimension,g)
        g_inputs=flow.reshape(g,shape=[batch_size, self.inter_channels, -1])
        g_inputs=flow.transpose(g_inputs,perm=[0,2,1])

        theta=self.build_conv(self.dimension,"theta",inputs,bn=False,trainable=self.trainable)
        theta_inputs=flow.reshape(theta,shape=[batch_size, self.inter_channels, -1])
        theta_inputs=flow.transpose(theta_inputs,perm=[0,2,1])

        phi=self.build_conv(self.dimension,"phi",inputs,bn=False,trainable=self.trainable)
        phi=self.build_maxpool(self.dimension,phi)
        phi_inputs=flow.reshape(phi,shape=[batch_size, self.inter_channels, -1])

        f=flow.linalg.matmul(theta_inputs,phi_inputs)
        f=nn.softmax(f,axis=-1)

        y=flow.linalg.matmul(f,g_inputs)
        y=flow.transpose(y,perm=[0,2,1])
        y=flow.reshape(y,shape=[batch_size, self.inter_channels, *inputs.shape[2:]])

        W=self.build_conv(self.dimension,"W",y,bn=self.bn_layer,trainable=self.trainable)
        output=W+inputs
        return output

class NonLocalBlock1D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True,trainable=True):
        super(NonLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer,trainable=trainable)


class NonLocalBlock2D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True,trainable=True):
        super(NonLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer,trainable=trainable)


class NonLocalBlock3D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True,trainable=True,time=0):
        super(NonLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer,trainable=trainable,time=time)
