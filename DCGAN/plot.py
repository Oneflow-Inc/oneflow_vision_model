import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob

def is_outlier(points, thresh=3):
    return points >= thresh

def plot(root_dir, epoch):
    g_loss = np.load(os.path.join(root_dir, 'g_loss_{}.npy'.format(epoch)))
    d_loss = np.load(os.path.join(root_dir, 'd_loss_{}.npy'.format(epoch)))
    print("last g_loss: {}.".format(g_loss[-10:-1]))
    print("last d_loss: {}.".format(d_loss[-10:-1]))
    filtered_g_loss = g_loss[~is_outlier(g_loss)]
    filtered_d_loss = d_loss[~is_outlier(d_loss)]
    plt.figure(figsize=(15,5))
    plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(g_loss,label="G")
    # plt.plot(d_loss,label="D")
    plt.plot(filtered_g_loss,label="G")
    plt.plot(filtered_d_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(root_dir, 'loss_{}.png'.format(epoch)))

if __name__ == "__main__":
    root_dir = "/home/zjhuangzhenhua/zjcdy/DCGAN/of_model"
    epoch = 100
    plot(root_dir, epoch)

