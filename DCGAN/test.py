import oneflow as flow
import numpy as np
import oneflow.typing as tp


def conv2d(input, filters, kernel_size, strides, padding, name):
    input_shape = input.shape
    weight_initializer = flow.truncated_normal(0.1)
    weight_regularizer = flow.regularizers.l2(0.0005)
    weight_shape = (filters,
                    input_shape[1],
                    kernel_size[0],
                    kernel_size[1])

    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        initializer=weight_initializer,
        regularizer=weight_regularizer,
    )
    return flow.nn.compat_conv2d(input, weight, strides, padding, name=name)


@flow.global_function()
def conv2d_Job(x: tp.Numpy.Placeholder((1, 64, 32, 32))
) -> tp.Numpy:
    conv = conv2d(x,
                filters=128,
                kernel_size=[3, 3],
                strides=2,
                padding='SAME',
                name="Convlayer")
    return conv


x = np.random.randn(1, 64, 32, 32).astype(np.float32)
out = conv2d_Job(x)
print(out.shape)

# out.shape (1, 128, 16, 16)