import oneflow as flow
class scloss(object):
    def __init__(self,cnum = 3):
        self.cnum = cnum
    def ccmp(self,input, kernel_size, stride):
        input = flow.transpose(input, perm=[0, 3, 2, 1])
        input = flow.nn.max_pool2d(input, kernel_size, stride, padding="VALID")
        input = flow.transpose(input, perm=[0, 3, 2, 1])
        return input

    def loss_div(self,feature):
        branch = feature
        branch = flow.reshape(branch, (branch.shape[0], branch.shape[1], branch.shape[2] * branch.shape[3]))
        branch = flow.nn.softmax(branch, 2)
        branch = flow.reshape(branch, (branch.shape[0], branch.shape[1], feature.shape[2], feature.shape[2]))
        branch = self.ccmp(branch, kernel_size=(1, self.cnum), stride=(1, self.cnum))
        branch = flow.reshape(branch, (branch.shape[0], branch.shape[1], branch.shape[2] * branch.shape[3]))
        loss_dis = 1.0 - 1.0 * flow.math.reduce_mean(flow.math.reduce_sum(branch, 2)) / self.cnum  # set margin = 3.0
        return loss_dis

    def loss_con(self,one_hot_labels, feature):
        branch = feature
        fc_part1 = flow.layers.dense(
            flow.reshape(branch, (branch.shape[0], -1)),
            units=8,  # 车辆颜色类别8
            use_bias=True,
            kernel_initializer=flow.variance_scaling_initializer(2, 'fan_in', 'random_normal'),
            bias_initializer=flow.zeros_initializer(),
            name="fc1",
        )
        loss_con = flow.nn.softmax_cross_entropy_with_logits(one_hot_labels, fc_part1, name="softmax_loss1")
        return loss_con

    def loss_pre(self,one_hot_labels, fc8):
        return flow.nn.softmax_cross_entropy_with_logits(one_hot_labels, fc8)


def sc_loss(one_hot_labels, logits, fc8):
    loss = scloss(3)
    return loss.loss_con(one_hot_labels, logits) + loss.loss_div(logits) + loss.loss_pre(one_hot_labels, fc8)