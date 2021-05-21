import oneflow as flow


# @flow.global_function()
# def conv2d_Job(
#         x: tp.Numpy.Placeholder((3, 1, 3, 2))
#
# ) -> Tuple[tp.Numpy, tp.Numpy]:
#     A_feat = flow.slice(x=x, begin=[0, None, None, None], size=[1, None, None, None])
#     B_feat = flow.squeeze(input=A_feat, axis=0)
#     return A_feat,B_feat
#
#
# x = np.random.randn(3, 1, 3, 2).astype(np.float32)
# print(x)
#
# out1,out2 = conv2d_Job(x)
#
# print(out1.shape)
# print(out1)
# print(out2.shape)
# print(out2)

@flow.global_function()
def conv2d_Job(
        x: tp.Numpy.Placeholder((6,3,256,256)),
        y: tp.Numpy.Placeholder((6,3,256,256))



) ->tp.Numpy:

    conv1=flow.math.multiply(x,y)
    return conv1


images = np.ones((6,3,256,256)).astype(np.float32)
masks = np.ones((6,3,256,256)).astype(np.float32)

out1= conv2d_Job(images,masks)
print("image :",out1.shape)
print(out1)