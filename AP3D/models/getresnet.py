
class conv2d(object):
    def __init__(self, in_channels, out_channels, kernel_size,stride=[1,1],padding=[0,0],bias=False,dilation=[1,1]):
        super(conv2d, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.bias=bias
class bn2d(object):
    def __init__(self, num_features,  eps, momentum,affine,track_running_stats):
        super(bn2d, self).__init__()
        self.num_features=num_features
        self.eps=eps
        self.momentum=momentum
        self.affine=affine
        self.track_running_stats=track_running_stats
class downsample(object):
    def __init__(self, conv2d,bn2d):
        super(downsample, self).__init__()
        self.conv2d=conv2d
        self.bn2d=bn2d
class Bottleneck(object):
    def __init__(self, conv1, bn1,conv2,bn2,conv3,bn3,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1=conv1
        self.bn1=bn1
        self.conv2=conv2
        self.bn2=bn2
        self.conv3=conv3
        self.bn3=bn3
        self.downsample=downsample

def getResnet():
    layer_all=[]
    layer=[]
    layer.append(
            Bottleneck(
                conv2d(64, 64, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(64, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                downsample(
                    conv2d(64, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                    bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        ))
    layer.append(
            Bottleneck(
                conv2d(256, 64, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(64, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ))
    layer.append(
            Bottleneck(
                conv2d(256, 64, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(64, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        ))
    layer_all.append(layer)
    layer=[]
    layer.append(
            Bottleneck(
                conv2d(256, 128, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(128, 128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False),
                bn2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(128, 512, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                downsample(
                    conv2d(256, 512, kernel_size=[1, 1], stride=[2, 2], bias=False),
                    bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        ))
    layer.append(
            Bottleneck(
                conv2d(512, 128, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(128, 512, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        ))
    layer.append(
            Bottleneck(
                conv2d(512, 128, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(128, 512, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        ))
    layer.append(
            Bottleneck(
                conv2d(512, 128, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(128, 512, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        ))
    layer_all.append(layer)
    layer=[]
    layer.append(
            Bottleneck(
                conv2d(512, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 1024, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                downsample(
                    conv2d(512, 1024, kernel_size=[1, 1], stride=[2, 2], bias=False),
                    bn2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        ))
    layer.append(
            Bottleneck(
                conv2d(1024, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 1024, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
        )) 
    layer.append(
            Bottleneck(
                conv2d(1024, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 1024, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
        ))  
    layer.append(
            Bottleneck(
                conv2d(1024, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 1024, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
        ))  
    layer.append(
            Bottleneck(
                conv2d(1024, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 1024, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
        ))  
    layer.append(
            Bottleneck(
                conv2d(1024, 256, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(256, 1024, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
        )) 
    layer_all.append(layer)
    layer=[]
    layer.append(
            Bottleneck(
                conv2d(1024, 512, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(512, 512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(512, 2048, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                downsample(
                    conv2d(1024, 2048, kernel_size=[1, 1], stride=[1, 1], bias=False),
                    bn2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        ))
    layer.append(
            Bottleneck(
                conv2d(2048, 512, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(512, 512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(512, 2048, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
        ))
    layer.append(
            Bottleneck(
                conv2d(2048, 512, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(512, 512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False),
                bn2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                conv2d(512, 2048, kernel_size=[1, 1], stride=[1, 1], bias=False),
                bn2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                
        ))
    layer_all.append(layer)
        
    return layer_all




        
        