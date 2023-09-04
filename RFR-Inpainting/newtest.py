import  oneflow as flow
import numpy as np
import oneflow.typing as tp

@flow.global_function(type="train")
def train_job(
            maskedimg: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
            masks: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
            images: tp.Numpy.Placeholder((6, 3, 256, 256), dtype=flow.float),
) -> tp.Numpy:
    init = RFRNetModel()
    ret, mmask, fake, comp = init.buildnet(maskedimg, masks, images)
    loss = init.get_g_loss(ret, mmask, fake, comp)
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [2e-4])
    # Set Adam optimizer
    flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss)
    return loss


x = np.random.randn(6, 3, 256, 256).astype(np.float32)
y = np.random.randn(6, 3, 256, 256).astype(np.float32)
z = np.random.randn(6, 3, 256, 256).astype(np.float32)
kk = train_job(x,y,z)
print(kk.shape)
print(kk)