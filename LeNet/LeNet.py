import oneflow as flow
import numpy as np
import oneflow.typing as tp
from typing import Tuple
import matplotlib.pyplot as plt

batch_size = 100
flow.config.enable_legacy_model_io(False)

def LeNet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    
    conv_1 = flow.layers.conv2d(data, 6, 5, padding="SAME", activation=flow.nn.relu, name="conv_1",
        kernel_initializer=initializer)
    
    pool_1 = flow.nn.max_pool2d(conv_1, ksize=2, strides=2, padding="SAME", name="pool_1", data_format="NCHW")
    
    conv_2 = flow.layers.conv2d(pool_1, 16, 5,padding="SAME", activation=flow.nn.relu, name="conv_2",
        kernel_initializer=initializer,)
    
    pool_2 = flow.nn.max_pool2d(conv_2, ksize=2, strides=2, padding="SAME", name="pool_2", data_format="NCHW")
    
    reshape = flow.reshape(pool_2, [pool_2.shape[0], -1])
    
    dense_1 = flow.layers.dense(reshape, 120, activation=flow.nn.relu, name="dense_1",
        kernel_initializer=initializer,)
    
    dense_2 = flow.layers.dense(dense_1, 84, activation=flow.nn.relu, kernel_initializer=initializer, name="dense_2")
    
    if train:
        dense_2 = flow.nn.dropout(dense_2, rate=0.5, name="dropout")

    dense_3 = flow.layers.dense(dense_2, 10, kernel_initializer=initializer, name="dense_3")
        
    return dense_3

@flow.global_function(type='train')
def train(
    images: tp.Numpy.Placeholder((batch_size, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((batch_size,), dtype=flow.int32),) -> tp.Numpy:
    
    with flow.scope.placement("gpu", "0:0"):
        logits = LeNet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
        
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([],[0.001])
    flow.optimizer.Adam(lr_scheduler,do_bias_correction=False).minimize(loss)
    return loss

@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((batch_size,1,28,28), dtype = flow.float),
    labels: tp.Numpy.Placeholder((batch_size,), dtype = flow.int32),
)-> Tuple[tp.Numpy, tp.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = LeNet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name = 'softmax_loss')
    return (labels, logits)

def acc(labels, logits, total, corr):
    pred = np.argmax(logits,1)
    true_pred = np.sum(pred == labels)
    total += labels.shape[0]
    corr += true_pred
    return total, corr

def main():
    (X_train, y_train), (X_test, y_test) = flow.data.load_mnist(batch_size ,batch_size)
    train_loss, test_acc = [],[]
    for epoch in range(10):
        total = 0
        corr = 0
        for i, (images, labels) in enumerate(zip(X_train, y_train)):
            loss = train(images, labels)
        train_loss.append(loss.mean())
        for i, (images, labels) in enumerate(zip(X_test, y_test)):
            labels, logits = eval_job(images, labels)
            total, corr = acc(labels, logits, total, corr)
        test_acc.append(corr/total)
        print("Epoch: {} Train Loss: {} Test Accuracy: {}"
              .format(epoch+1,loss.mean(), corr/total))
    return train_loss, test_acc   
       
def resplot(x, color, label):
    plt.figure()
    plt.plot(x,color)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.xlim(0,len(x)-1)
    plt.ylim(0,1.1)
    plt.title(label)
    plt.show()
      
if __name__ == "__main__":
    train_loss, test_acc = main()
    flow.checkpoint.save("./lenet_models_1")
    print("model saved")

    resplot(train_loss, color='blue', label='train_loss')
    
    resplot(test_acc, color='orange', label='test_acc')
