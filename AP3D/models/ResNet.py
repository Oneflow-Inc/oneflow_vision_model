import math
import copy
import oneflow as flow
import oneflow.nn as nn
import datetime
import numpy as np
import models.NonLocal as NonLocal
__all__ = ['AP3DResNet50', 'AP3DNLResNet50']
global time
time=0

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
    weight_regularizer=_get_regularizer(),
    bias_regularizer=_get_regularizer(),
    bias_initializer=_get_bias_initializer(),
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
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == "Relu":
            output = flow.nn.relu(output)
        else:
            raise NotImplementedError

    return output

class APM(object):
    def __init__(self, out_channels, time_dim=3, temperature=4, contrastive_att=True,trainable=True,time=0):
        super(APM,self).__init__()
        self.time_dim=time_dim
        self.temperature=temperature
        self.contrastive_att=contrastive_att
        self.out_channels=out_channels
        self.trainable=trainable
        self.time=time
    def build_network(self,inputs):
        global time
        b,c,t,h,w=inputs.shape
        N=self.time_dim
        
        templist=[np.arange(0,t)+i for i in range(N) if i!=N//2]
        templist=np.expand_dims(templist,axis=0)
        neighbor_time_index=np.concatenate(
            templist,axis=0
        )
        neighbor_time_index=np.transpose(neighbor_time_index)
        neighbor_time_index=np.ndarray.flatten(neighbor_time_index)
   
        neighbor_time_index=np.int64(neighbor_time_index)
        
        # feature map registration

        init=flow.kaiming_initializer(shape=inputs.shape,mode="fan_out",nonlinearity="relu")
        semantic=conv3d_layer("conv_semantic_"+str(time),inputs,self.out_channels,
            kernel_size=1,use_bias=False,padding="VALID",trainable=self.trainable,
            weight_initializer=init
        )
        inputs_norm=flow.math.l2_normalize(
            semantic,axis=1
        )


        inputs_norm_padding=flow.pad(inputs_norm,paddings=[
            (0,0),(0,0),((self.time_dim-1)//2,(self.time_dim-1)//2), (0,0),(0,0)]
        )
        inputs_norm_expand=flow.expand_dims(inputs_norm,axis=3)
        temp_inputs_norm_expand=inputs_norm_expand
        for i in range(N-2):
            inputs_norm_expand=flow.concat(
               inputs=[ inputs_norm_expand,temp_inputs_norm_expand],
                axis=3
            )
       
        inputs_norm_expand=flow.transpose(inputs_norm_expand,perm=[0, 2, 3, 4, 5, 1])
        inputs_norm_expand=flow.reshape(inputs_norm_expand,shape=[-1, h*w, c//16])

        slice_list=[]
        for index in  neighbor_time_index:
            temp=flow.slice(
                inputs_norm_padding,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )      
            slice_list.append(temp)
        neighbor_norm=flow.concat(
            slice_list,axis=2
        )
        neighbor_norm=flow.transpose(neighbor_norm,perm=[0, 2, 1, 3, 4])
        neighbor_norm=flow.reshape(neighbor_norm,shape=[-1, c//16, h*w])

        similarity=flow.matmul(inputs_norm_expand,neighbor_norm)*self.temperature
        similarity=nn.softmax(similarity,axis=-1)

        inputs_padding=flow.pad(inputs,
        paddings=[
            (0,0),(0,0),((self.time_dim-1)//2,(self.time_dim-1)//2), (0,0),(0,0)]
        ) 
        slice_list=[]
        for index in  neighbor_time_index:
            temp=flow.slice(
                inputs_padding,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )      
            slice_list.append(temp)
        neighbor=flow.concat(
            slice_list,axis=2
        )
        neighbor=flow.transpose(neighbor,perm=[0,2,3,4,1])
        neighbor=flow.reshape(neighbor,shape=[-1, h*w, c]) 

        neighbor_new=flow.matmul(similarity,neighbor)
        neighbor_new=flow.reshape(neighbor_new,shape=[b, t*(N-1), h, w, c])
        neighbor_new=flow.transpose(neighbor_new,perm=[0, 4, 1, 2, 3])

         # contrastive attention
        if self.contrastive_att:        
            temp_input=flow.expand_dims(inputs,axis=3)
            temp_temp_input=temp_input
            for i in range(N-2):
                temp_input=flow.concat(
                inputs=[ temp_input,temp_temp_input],
                axis=3
            )
            temp_input=flow.reshape(temp_input,shape=[b, c, (N-1)*t, h, w])
            input_att=conv3d_layer(
                "conv3d_inputmapping_"+str(time),temp_input,self.out_channels,
                kernel_size=1, use_bias=False,trainable=False,weight_initializer=flow.kaiming_initializer(shape=temp_input.shape,mode="fan_out",nonlinearity="relu")
            )

            n_att=conv3d_layer(
                "conv3d_nmapping_"+str(time),neighbor_new,self.out_channels,
                kernel_size=1, use_bias=False,trainable=False,weight_initializer=flow.kaiming_initializer(shape=neighbor_new.shape,mode="fan_out",nonlinearity="relu")
            )
            temp_input=input_att*n_att
            contrastive_att_net=conv3d_layer(
                "conv3d_att_net_"+str(time),temp_input,1,
                kernel_size=1, use_bias=False,trainable=self.trainable,weight_initializer=flow.kaiming_initializer(shape=temp_input.shape,mode="fan_out",nonlinearity="relu")
            )
            contrastive_att_net=flow.math.sigmoid(contrastive_att_net)
            neighbor_new=flow.math.multiply(
                neighbor_new,contrastive_att_net
            )
        # integrating feature maps        
        init = flow.zeros_initializer()
        input_offset = flow.get_variable(
            "input_offset_"+str(time),
            shape=(b, c, N*t, h, w),
            initializer=init,
            dtype=inputs.dtype,
            trainable=self.trainable)
        input_index=np.array(
            [i for i in range(t*N) if i%N==N//2]
        )
        neighbor_index=np.array(
            [i for i in range(t*N) if i%N!=N//2])
        input_offset_list=[]
        inputs_list=[]
        neighbor_new_list=[]
        for index in  range(input_offset.shape[2]):
            temp=flow.slice(
                input_offset,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )  
            input_offset_list.append(temp)
        for index in range(inputs.shape[2]):
            temp=flow.slice(
                inputs,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )
            inputs_list.append(temp)
        for index in range(neighbor_new.shape[2]):
            temp=flow.slice(
                neighbor_new,
                begin=[None,None,int(index),None,None],
                size=[None,None,1,None,None]
            )
            neighbor_new_list.append(temp)
        temp_index=0
        for index in input_index:
            input_offset_list[index]+=inputs_list[temp_index]
            temp_index+=1

        temp_index=0
        for index in neighbor_index:
            input_offset_list[index]+=neighbor_new_list[temp_index]
            temp_index+=1
        input_offset=flow.concat(
            input_offset_list,axis=2
        )
        return input_offset


class C2D(object):
    def __init__(self,name,conv2d,trainable=True,**kwargs):
        super(C2D, self).__init__()
        self.conv2d=conv2d
        self.kernel_dim = [1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride = [1, conv2d.stride[0], conv2d.stride[0]]
        self.padding = [0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.trainable=trainable
        self.name=name
    def build_network(self,inputs):
        output=conv3d_layer(self.name,inputs=inputs,filters=self.conv2d.out_channels,
                kernel_size=self.kernel_dim,strides=self.stride, padding=self.padding,
                use_bias=False,weight_initializer=flow.xavier_normal_initializer(data_format='NCDHW'),trainable=self.trainable)
        return output

class APP3DC(object):
    def __init__(self, name,conv2d, time_dim=3,  temperature=4, contrastive_att=True,time_stride=1,trainable=True):
        super(APP3DC, self).__init__() 
        global time
        self.APM=APM(conv2d.out_channels//16,time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att,time=time)
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.time_dim=time_dim
        self.conv2d=conv2d
        self.time_stride=time_stride
        self.trainable=trainable
        self.name=name
        self.time=time
    def build_network(self,inputs):
       
        init=flow.kaiming_initializer(shape=inputs.shape,mode="fan_out")
        out=conv3d_layer(
            self.name,inputs,self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, strides=self.stride,
            padding="SAME", use_bias=True,weight_initializer=init,trainable=self.trainable)
        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride*self.time_dim,1,1]
        residual=self.APM.build_network(out)
        init=flow.kaiming_initializer(shape=residual.shape,mode="fan_out")
        residual=conv3d_layer(
            "APP3DC_temporal_"+str(self.time),residual,self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, 
            strides=self.stride, padding="VALID",use_bias=False,weight_initializer=init,
            trainable=self.trainable)
        global time
        time+=1
        out=out+residual
        return out





def inflate_conv(inputs,conv2d,time_dim=1,time_padding=0,time_stride=1,time_dilation=1,trainable=True,name=None):
    kernel_dim=[time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]]
    padding = [0,0,time_padding, conv2d.padding[0],conv2d.padding[1]]
    stride = [time_stride, conv2d.stride[0], conv2d.stride[0]]
    dilation = [time_dilation, conv2d.dilation[0], conv2d.dilation[1]]
    output=conv3d_layer(
        name,inputs,conv2d.out_channels,kernel_size=kernel_dim,
        dilation_rate=dilation,strides=stride,
        padding=padding,
        weight_initializer=flow.xavier_normal_initializer(data_format='NCDHW'),
        use_bias=False,trainable=trainable
    )
    return output

def inflate_batch_norm(inputs,num_features,trainable=True,name=None):
    output = flow.layers.batch_normalization(inputs=inputs,
                                            axis=1,
                                            momentum=0.997,
                                            epsilon=1.001e-5,
                                            center=True,
                                            scale=True,
                                            trainable=trainable,
                                            name=name+"_bn")

    return output
            


def inflate_pool(inputs,kernel_size,padding,stride,dilation,time_dim=1,time_padding=0,time_stride=None,time_dilation=1):
    kernel_dim = [time_dim, kernel_size, kernel_size]
    padding = [0,0,time_padding, padding, padding]
    if time_stride is None:
        time_stride=time_dim
    stride = [time_stride, stride, stride]
    dilation = [time_dilation, dilation,dilation]
    pool3d=nn.max_pool3d(
            inputs,
            ksize=kernel_dim,
            strides=stride,
            padding=padding
        )
    return pool3d


class Conv2d(object):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1,padding=1,bias=False,dilation=1):
        super(Conv2d, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.bias=bias
        self.dilation=dilation

        
class Bottleneck3D(object):
    def __init__(self, name,bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True,trainable=True):
        super(Bottleneck3D, self).__init__()
        self.inflate_time=inflate_time
        self.bottleneck2d=bottleneck2d
        self.block=block
        self.temperature=temperature
        self.contrastive_att=contrastive_att
        self.trainable=trainable
        self.name=name
    def _inflate_downsample(self,name,inputs ,downsample2d, time_stride=1):
        out=inflate_conv(inputs,downsample2d.conv2d,time_dim=1,time_stride=time_stride,
                trainable=self.trainable,name=name)
        out=inflate_batch_norm(out,downsample2d.bn2d.num_features,trainable=self.trainable,name=name)
        return out
    def build_network(self,inputs):
        global time
        residual = inputs
        #conv1
        out=inflate_conv(inputs,self.bottleneck2d.conv1,time_dim=1,trainable=self.trainable,
            name=self.name+"branch2a")
        #bn1
        out=inflate_batch_norm(out,self.bottleneck2d.bn1.num_features,trainable=self.trainable,name=self.name+"branch2a")
        #relu
        out=nn.relu(out)
        #conv2
        if self.inflate_time == True:
            out=self.block(self.name+"branch2b",self.bottleneck2d.conv2, temperature=self.temperature,
                    contrastive_att=self.contrastive_att,trainable=self.trainable).build_network(out)
        else:
            out=inflate_conv(out,self.bottleneck2d.conv2, time_dim=1,trainable=self.trainable,name=self.name+"branch2b")
        #bn2
        out=inflate_batch_norm(out,self.bottleneck2d.bn2.num_features,trainable=self.trainable,name=self.name+"branch2b")
        #relu
        out=nn.relu(out)
        #conv3
        out=inflate_conv(out,self.bottleneck2d.conv3, time_dim=1,trainable=self.trainable,name=self.name+"branch2c")
        #bn3
        out=inflate_batch_norm(out,self.bottleneck2d.bn3.num_features,trainable=self.trainable,name=self.name+"branch2c")
        if self.bottleneck2d.downsample is not None:
            residual=self._inflate_downsample(self.name+"branch1",inputs,self.bottleneck2d.downsample)
        out+=residual
        out=nn.relu(out)
        return out


class ResNet503D(object):
    def __init__(self, num_classes, block, c3d_idx, nl_idx, temperature=4, contrastive_att=True,resnetblock=None, training=True):
        super(ResNet503D, self).__init__()
        self.block = block
        self.temperature = temperature
        self.contrastive_att = contrastive_att
        self.num_classes=num_classes
        self.c3d_idx=c3d_idx
        self.nl_idx=nl_idx
        self.training=training
        self.resnet2d=resnetblock
    
    def _inflate_reslayer(self,name,inputs,reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        global time
        layer3d=inputs
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(name+str(i)+"_",layer2d, C2D, inflate_time=False,trainable=self.training).build_network(layer3d)
            else:
                layer3d = Bottleneck3D(name+str(i)+"_",layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att,trainable=self.training).build_network(layer3d)

            if i in nonlocal_idx:
                layer3d = NonLocal.NonLocalBlock3D(nonlocal_channels, sub_sample=True,trainable=self.training,time=time).build_network(layer3d) 
                time+=1
        return layer3d

    def build_network(self,inputs):
        conv2d=Conv2d(3,64,[7,7],[2,2],[3,3],dilation=[1, 1])
        #conv1
        out= inflate_conv(inputs,conv2d, time_dim=1,trainable=self.training,name="Resnet-conv1") 
        #bn1
        out=inflate_batch_norm(out,64,trainable=self.training,name="Resnet-conv1")
        resnet2d=self.resnet2d.getResnet()     
        #relu
        out=nn.relu(out)
        #maxpool
        out=inflate_pool(out,kernel_size=3,padding=1,stride=2,dilation=1, time_dim=1)
        out=self._inflate_reslayer("Resnet-res2_",out,resnet2d[0], c3d_idx=self.c3d_idx[0], 
                                             nonlocal_idx=self.nl_idx[0], nonlocal_channels=256)
        out=self._inflate_reslayer("Resnet-res3_",out,resnet2d[1], c3d_idx=self.c3d_idx[1], \
                                             nonlocal_idx=self.nl_idx[1], nonlocal_channels=512)

        out=self._inflate_reslayer("Resnet-res4_",out,resnet2d[2], c3d_idx=self.c3d_idx[2], \
                                             nonlocal_idx=self.nl_idx[2], nonlocal_channels=1024)
        out= self._inflate_reslayer("Resnet-res5_",out,resnet2d[3], c3d_idx=self.c3d_idx[3], \
                                             nonlocal_idx=self.nl_idx[3], nonlocal_channels=2048)
        b,c,t,h,w=out.shape
        out =flow.transpose(out,perm=[0,2,1,3,4])
        out=flow.reshape(out,shape=[b*t, c, h, w])
        out=nn.max_pool2d(
            input=out,
            ksize=out.shape[2:],
            strides=None,
            padding="VALID"
        )
        out=flow.reshape(out,shape=[b,t,-1])
        if not self.training:
            return out
        out=flow.math.reduce_mean(out,axis=1)
        f = flow.layers.batch_normalization(inputs=out,

                                            center=False,
                                            trainable=self.training,
                                            axis=1,
                                            beta_initializer=flow.constant_initializer(0),
                                            gamma_initializer=flow.random_normal_initializer(mean=1, stddev=0.02),
                                            name= "Resnet503D_linear_bn")
        
        y=flow.layers.dense(f, 
                            self.num_classes, 
                            use_bias=False,
                            activation=None,
                            kernel_initializer=flow.random_normal_initializer(mean=0, stddev=0.001),
                            bias_initializer=flow.zeros_initializer(),
                            kernel_regularizer=_get_regularizer(),  # weght_decay
                            bias_regularizer=_get_regularizer(),
                            name="fcRes")
        return y,f
        
def AP3DResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]
    return ResNet503D(num_classes, C2D, c3d_idx, nl_idx, **kwargs)

def AP3DNLResNet50(num_classes, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]
    return ResNet503D(num_classes, C2D, c3d_idx, nl_idx, **kwargs)
