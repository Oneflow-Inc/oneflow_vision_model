import oneflow as flow
import oneflow.nn as nn
import numpy as np
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
        trainable=trainable,
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



class APM(object):
    def __init__(self, out_channels, time_dim=3, temperature=4, contrastive_att=True,trainable=True):
        super(APM,self).__init__()
        self.time_dim=time_dim
        self.temperature=temperature
        self.contrastive_att=contrastive_att
        self.out_channels=out_channels
        self.trainable=trainable
    def build_network(self,inputs):
        b,c,t,h,w=inputs.shape
        N=self.time_dim
        templist=[]
        for i in range(N):
            tempname=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')    
            if i!=N//2:
                out = flow.range(t, dtype=flow.int64)
                one = flow.constant_like(out, i, dtype= flow.int64)
                out=flow.math.add(out, one)
                out=flow.expand_dims(out,axis=0)
                templist.append(out)
        neighbor_time_index=flow.concat(templist,axis=0)
        neighbor_time_index=flow.transpose(neighbor_time_index,[1,0])
        neighbor_time_index=flow.flatten(neighbor_time_index, start_dim=0, end_dim=-1)


    
        # feature map registration
        tempname=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')    

        init=flow.kaiming_initializer(shape=inputs.shape,mode="fan_out",nonlinearity="relu")
        semantic=conv3d_layer("conv_semantic_"+tempname,inputs,self.out_channels,
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
                "conv3d_inputmapping_"+tempname,temp_input,self.out_channels,
                kernel_size=1, use_bias=False,trainable=False,weight_initializer=flow.kaiming_initializer(shape=temp_input.shape,mode="fan_out",nonlinearity="relu")
            )

            n_att=conv3d_layer(
                "conv3d_nmapping_"+tempname,neighbor_new,self.out_channels,
                kernel_size=1, use_bias=False,trainable=False,weight_initializer=flow.kaiming_initializer(shape=neighbor_new.shape,mode="fan_out",nonlinearity="relu")
            )
            temp_input=input_att*n_att
            contrastive_att_net=conv3d_layer(
                "conv3d_att_net_"+tempname,temp_input,1,
                kernel_size=1, use_bias=False,trainable=self.trainable,weight_initializer=flow.kaiming_initializer(shape=temp_input.shape,mode="fan_out",nonlinearity="relu")
            )
            contrastive_att_net=flow.math.sigmoid(contrastive_att_net)
            neighbor_new=flow.math.multiply(
                neighbor_new,contrastive_att_net
            )
        # integrating feature maps

        
        init = flow.zeros_initializer()
        tempname=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')    

        input_offset = flow.get_variable(
            "input_offset_"+tempname,
            shape=(b, c, N*t, h, w),
            initializer=init,
            dtype=inputs.dtype,
            trainable=self.trainable)
        with flow.scope.placement("cpu", "0:0"):

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
    def __init__(self,name,conv2d,trainable=True ,**kwargs):
        super(C2D, self).__init__()
        self.conv2d=conv2d
        self.kernel_dim = [1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride = [1, conv2d.stride[0], conv2d.stride[0]]
        self.padding = [0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.trainable=trainable
        self.name=name
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weigt_3d=np.zeros(weight_2d.shape)   
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        #init=flow.constant_initializer(weight_3d)
      #  init=flow.kaiming_initializer(shape=inputs.shape,mode="fan_out",nonlinearity="relu")
        init=flow.random_normal_initializer(mean=0, stddev=1)
        output=conv3d_layer(self.name,inputs=inputs,filters=self.conv2d.out_channels,
                kernel_size=self.kernel_dim,strides=self.stride, padding=self.padding,
                use_bias=True,weight_initializer=init,trainable=self.trainable)
        return output
class I3D(object):
    def __init__(self,conv2d,time_dim=3,time_stride=1,trainable=True,**kwargs):
        super(I3D,self).__init__()
        self.kernel_dim=[time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[time_stride, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,time_dim//2, conv2d.padding[0], conv2d.padding[1]]
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.trainable=trainable
    def build_network(self,inputs):
        #pytorch中的repeat ==>numpy tile
        #由于上面使用了numpy的zeros函数导致weight3d 变成了np类型的对象，无法使用
        #flow相关的函数，因此这里的后续补充需要从zero开始。
        # oneflow.repeat(input: oneflow.python.framework.remote_blob.BlobDef, repeat_num: int, 
        # name: Optional[str] = None) → oneflow.python.framework.remote_blob.BlobDef
        #weight_3d=flow.repeat(weight_3d,)
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d=np.tile(weight_3d,(1,1,self.time_dim,1,1))
        # middle_dix=self.time_dim//2
        # weight_3d[:, :, middle_idx, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        #init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        init=flow.kaiming_initializer(shape=inputs.shape,mode="fan_out",nonlinearity="relu")
        output=conv3d_layer("conv_I3D_",inputs,self.conv2d.out_channels,
                kernel_size=self.kernel_dim,strides=self.stride, padding=self.padding,
                use_bias=True, weight_initializer=init,trainable=self.trainable
        )
        return output
class API3D(object):
    def __init__(self, conv2d, time_dim=3, time_stride=1, temperature=4, contrastive_att=True,trainable=True):
        super(API3D, self).__init__()
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.APM=APM(conv2d.in_channels//16, 
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        self.kernel_dim=[time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[time_stride*time_dim, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.trainable=trainable
    def build_network(self,inputs):
        # weight_2d = self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d=np.tile(
        #     weight_3d,(1,1,self.time_dim,1,1)
        # )
        # middle_idx = self.time_dim // 2
        # weight_3d[:, :, middle_idx, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)

        init=flow.random_uniform_initializer(minval=0, maxval=0.5)

        out=conv3d_layer(
            "conv3d_API3D_",self.APM.build_network(inputs),self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            weight_initializer=init,use_bias=True,trainable=self.trainable
        )
        return out
        





class P3DA(object):
    def __init__(self, conv2d,time_stride=1,time_dim=3,trainable=True, **kwargs):
        super(P3DA, self).__init__()
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.time_stride=time_stride
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.trainable=trainable
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)        
        spatial_conv3d=conv3d_layer(
            "P3DA_spatial_",inputs,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            use_bias=True,weight_initializer=init,trainable=self.trainable
        )

        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride,1,1]
        self.padding=[self.time_dim//2,0,0]

        # weight_2d=np.eye(self.conv2d.out_channels)
        # weight_2d=flow.expand_dims(weight_2d,axis=2)
        # weight_2d=flow.expand_dims(weight_2d,axis=2)
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d=np.tile(weight_3d,(1,1,self.time_dim,1,1))
        # middle_dix=self.time_dim//2
        # weight_3d[:, :, middle_idx, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        out=conv3d_layer(
            "P3DA_temporal_",spatial_conv3d,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            weight_initializer=init,use_bias=False,trainable=self.trainable
        )
        return out


class P3DB(object):
    def __init__(self, conv2d,time_dim=3, time_stride=1, trainable=True,**kwargs):
        super(P3DB, self).__init__()
        self.conv2d=conv2d
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.time_dim=time_dim
        self.time_stride=time_stride
        self.trainable=trainable
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
    
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)        
        out1=conv3d_layer(
            "P3DB_spatial_",inputs,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            weight_initializer=init,use_bias=True,trainable=self.trainable
        )

        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride, self.conv2d.stride[0], self.conv2d.stride[0]]
        self.padding=[self.time_dim//2,0,0]
        init=flow.constant_initializer(0)
        out2=conv3d_layer(
            "P3DB_temporal_",inputs, self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            use_bias=False,weight_initializer=init,trainable=self.trainable
        )
        out1= out1+out2
        return out1

class P3DC(object):
    def __init__(self, conv2d,  time_dim=3,time_stride=1, trainable=True, **kwargs):
        super(P3DC, self).__init__()

        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.time_stride=time_stride
        self.trainable=trainable
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)        
        out=conv3d_layer(
            "P3DC_spatial_",inputs,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            use_bias=True,weight_initializer=init,trainable=self.trainable
        )

        self.kernel_dim=[self.time_dim, 1, 1]
        self.stride=[self.time_stride, 1, 1]
        self.padding=[self.time_dim//2, 0, 0]
        init=flow.constant_initializer(0)

        residual=conv3d_layer(
            "P3DC_temporal_",out,self.conv2d.out_channels,
            kernel_size=self.kernel_dim,strides=self.stride,padding=self.padding,
            use_bias=False,weight_initializer=init,trainable=self.trainable
        )
        out= out+residual
        return out


class APP3DA(object):
    def __init__(self, conv2d, time_dim=3, temperature=4, contrastive_att=True,time_stride=1,trainable=True):
        super(APP3DA, self).__init__()
        self.APM = APM(conv2d.out_channels//16, 
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.time_stride=time_stride
        self.trainable=trainable
    def build_network(self,inputs):
        # weight_2d = self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        out=conv3d_layer(
            "APP3DA_spatial_",inputs, self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, 
            strides=self.stride, padding=self.padding,use_bias=True,weight_initializer=init,
            trainable=self.trainable
        
        )
        self.kernel_dim=[self.time_dim, 1, 1]
        self.stride= [self.time_stride*self.time_dim, 1, 1]
        # weight_2d=np.eye(self.conv2d.out_channels)
        # weight_2d=flow.expand_dims(weight_2d,axis=2)
        # weight_2d=flow.expand_dims(weight_2d,axis=2)
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d=np.tile(weight_3d,(1,1,self.time_dim,1,1))
        # middle_idx = self.time_dim // 2
        # weight_3d[:, :, middle_idx, :, :] = weight_2d
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        #init=flow.constant_initializer(weight_3d)
        out=conv3d_layer(
    
            "APP3DA_temporal_",self.APM.build_network(inputs),self.conv2d.out_channels, 
            kernel_size=self.kernel_dim,
            strides=self.stride, padding="SAME",use_bias=False,weight_initializer=init,
             trainable=self.trainable
        )
        return out
class APP3DB(object):
    def __init__(self, conv2d, time_dim=3, temperature=4, contrastive_att=True,time_stride=1,trainable=True):
        super(APP3DB, self).__init__()
        self.APM = APM( conv2d.in_channels//16, 
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.conv2d=conv2d
        self.time_dim=time_dim
        self.time_stride=time_stride
        self.trainable=trainable
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        init=flow.random_uniform_initializer(minval=0, maxval=0.5)

        out=conv3d_layer(
            "APP3DB_spatial_",inputs,self.conv2d.out_channels, 
            kernel_size=self.kernel_dim,
            strides=self.stride, padding=self.padding,use_bias=True,weight_initializer=init,
            trainable=self.trainable
        )
        
        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride*self.time_dim,self.conv2d.stride[0],self.conv2d.stride[0]]
        init=flow.constant_initializer(0)
        out2=conv3d_layer(
            "APP3DB_temporal_",self.APM.build_network(inputs), self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, strides=self.stride, 
            padding="SAME",use_bias=False,weight_initializer=init,
            trainable=self.trainable
        )
        out=out+out2
        return out

class APP3DC(object):
    def __init__(self, name,conv2d, time_dim=3,  temperature=4, contrastive_att=True,time_stride=1,trainable=True):
        super(APP3DC, self).__init__() 
        self.APM=APM(conv2d.out_channels//16, 
                       time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        self.kernel_dim=[1, conv2d.kernel_size[0], conv2d.kernel_size[1]]
        self.stride=[1, conv2d.stride[0], conv2d.stride[0]]
        self.padding=[0,0,0, conv2d.padding[0], conv2d.padding[1]]
        self.time_dim=time_dim
        self.conv2d=conv2d
        self.time_stride=time_stride
        self.trainable=trainable
        self.name=name
    def build_network(self,inputs):
        # weight_2d=self.conv2d.weight.data
        # weight_3d=np.zeros(weight_2d.shape)
        # weight_3d=flow.expand_dims(weight_3d,axis=2)
        # weight_3d[:, :, 0, :, :] = weight_2d
        # init=flow.constant_initializer(weight_3d)
        #init=flow.random_uniform_initializer(minval=0, maxval=0.5)
        init=flow.kaiming_initializer(shape=inputs.shape,mode="fan_out",nonlinearity="relu")
        out=conv3d_layer(
            self.name,inputs,self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, strides=self.stride,
            padding="SAME", use_bias=True,weight_initializer=init,trainable=self.trainable
        )

        self.kernel_dim=[self.time_dim,1,1]
        self.stride=[self.time_stride*self.time_dim,1,1]
        #init=flow.constant_initializer(0)
        residual=self.APM.build_network(out)
        #init=flow.random_normal_initializer(mean=0, stddev=1)
        init=flow.kaiming_initializer(shape=residual.shape,mode="fan_out",nonlinearity="relu")
        #self.padding = "SAME" if self.stride > 1 or self.kernel_dim > 1 else "VALID"
        tempname=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')    

        residual=conv3d_layer(
            "APP3DC_temporal_"+tempname,residual,self.conv2d.out_channels, 
            kernel_size=self.kernel_dim, 
            strides=self.stride, padding="VALID",use_bias=False,weight_initializer=init,
            trainable=self.trainable
        )

        out=out+residual
        return out