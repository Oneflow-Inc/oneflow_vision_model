import os
import copy


class Utility(object):

    @staticmethod
    def get_intersection(rect1, rect2):
        '''
        计算两个矩形的交集
        :param rect1: [left, top, right, bottom]
        :param rect2: [left, top, right, bottom]
        :return:  [left, top, right, bottom]
        '''
        left = max(rect1[0], rect2[0])
        top = max(rect1[1], rect2[1])
        right = min(rect1[2], rect2[2])
        bottom = min(rect1[3], rect2[3])
        left = min(left, right)
        top = min(top, bottom)
        return [left, top, right, bottom]

    @staticmethod
    def get_x(point1, point2, y):
        '''

        :param point1: [x,y]
        :param point2: [x,y]
        :param y: float
        :return: float
        '''
        return (point1[0] - point2[0]) * (y - point2[1]) / (point1[1] - point2[1]) + point2[0]

    @staticmethod
    def get_y(point1, point2, x):
        '''

        :param point1: [x,y]
        :param point2: [x,y]
        :param x: float
        :return: float
        '''
        return (point1[1] - point2[1]) * (x - point2[0]) / (point1[0] - point2[0]) + point2[1]

    @staticmethod
    def is_in_rect(point, rect):
        '''
            点point是否在矩形rect内
        :param point: [x, y]
        :param rect:   [left, top, right, bottom]
        :return: bool
        '''
        return point[0] >= rect[0] and point[0] <= rect[2] \
               and point[1] >= rect[1] and point[1] <= rect[3]

    @staticmethod
    def get_rect_area(rect):
        '''
            获取矩形面积
        :param rect: [left, top, right, bottom]
        :return: float
        '''
        width = max(rect[2] - rect[0], 0)
        height = max(rect[3] - rect[1], 0)
        return width * height

    @staticmethod
    def get_center(rect):
        '''
            获取矩形中点
        :param rect: [left, top, right, bottom]
        :return: [x, y]
        '''
        center_x = (rect[0] + rect[2]) / 2
        center_y = (rect[1] + rect[3]) / 2
        return [center_x, center_y]

    @staticmethod
    def get_dist(point1, point2):
        '''
            计算两点距离
        :param point1: [x, y]
        :param point2: [x, y]
        :return:
        '''
        return pow(pow(point1[1] - point2[1], 2) + pow(point1[0] - point2[0], 2), 0.5)

    @staticmethod
    def get_slope(point1, point2):
        '''
            获取两点斜率
        :param point1: [x, y]
        :param point2: [x, y]
        :return:
        '''
        return (point1[1] - point2[1]) * 1.0 / (point1[0] - point2[0])

    @staticmethod
    def get_iou(rect1, rect2):
        '''
            计算IoU
        :param rect1:  [left, top, right, bottom]
        :param rect2:  [left, top, right, bottom]
        :return: float
        '''
        inter_area = Utility.get_rect_area(Utility.get_intersection(rect1, rect2))
        union_area = Utility.get_rect_area(rect1) + Utility.get_rect_area(rect2) - inter_area
        res = 0
        if union_area > 0:
            res = inter_area * 1.0 / union_area
        return res

    @staticmethod
    def get_obj_from_file(filename, has_id=False):
        '''
            从文件中读取检测结果或者是跟踪结果
        :param filename:
        :param has_id:
        :return: if has_id==False
                        [[left, top, right, bottom], ...]]
                 else
                        [[id, left, top, right, bottom], ...]]

        '''
        res = []
        with open(filename) as file:
            for line in file.readlines():
                tmp = line.split()
                tmp = list([float(i) for i in tmp])
                if has_id:
                    tmp = [int(tmp[-1]), ] + tmp[:-1]
                res.append(tmp)
        return res

    @staticmethod
    def overlapping_length(left, right):
        start_time = max(left[0], right[0])
        end_time = min(left[1], right[1])
        return max(0, end_time - start_time + 1)

    @staticmethod
    def getPRF1(preds_, gts):
        '''
            计算指标
        :param preds:  [[[start_time, end_time], ...],...]
        :param gts:   [[[start_time, end_time], ...],...]
        :return: precise, recall, f1, switch_rate, hit_rate
        '''
        preds = copy.deepcopy(preds_)
        total_gt_len = 0
        total_pred_len = 0
        for gt in gts:
            for gt_ in gt:
                total_gt_len += gt_[1] - gt_[0] + 1
        for pred in preds:
            pred.sort(key=lambda x: x[0])
            i = 0
            while i < len(pred):
                first = pred[i]
                j = i + 1
                while j < len(pred):
                    second = pred[j]
                    if first[1] + 1 >= second[0]:
                        first[1] = max(first[1], second[1])
                        pred.pop(j)
                    else:
                        break
                i += 1
        for pred in preds:
            for p in pred:
                total_pred_len += p[1] - p[0] + 1
        ok_jam = 0  # 成功匹配的时间
        switch_num = 0  # 切换数量
        total_jam_during_num = 0  # 总的拥堵时间段
        pred_jam_during_num = 0  # 预测到的时间段
        for k in range(len(gts)):
            pred = preds[k]
            switch_num += len(pred)
            gt = gts[k]
            total_jam_during_num += len(gt)

            ok_jam_ = 0
            total_gt_ = 0
            total_pred_ = 0
            for gt_ in gt:
                total_gt_ += gt_[1] - gt_[0] + 1
            for pred_ in pred:
                total_pred_ += pred_[1] - pred_[0] + 1
            for i in range(len(gt)):
                flag = True
                for j in range(len(pred)):
                    len_ = Utility.overlapping_length(pred[j], gt[i])
                    if len_ > 0 and flag:
                        pred_jam_during_num += 1
                        flag = False
                    ok_jam_ += len_
                    ok_jam += len_
            precise_ = 1 if total_pred_ == 0 else (ok_jam_ / total_pred_)
            recall_ = 1 if total_gt_ == 0 else (ok_jam_ / total_gt_)
            # print(precise_)
            # print(recall_)
        precise = ok_jam / total_pred_len
        recall = ok_jam / total_gt_len
        f1 = 2 * precise * recall / (precise + recall)
        switch_rate = switch_num / total_jam_during_num
        hit_rate = pred_jam_during_num / total_jam_during_num
        return precise, recall, f1, switch_rate, hit_rate
        pass

    @staticmethod
    def get_gt(path, filenames, suffix='.txt'):
        '''
            获取拥堵GT
        :param path:
        :param filenames:
        :param suffix:
        :return:  [[[start_time, end_time],...],...]
        '''
        gts = []
        for filename in filenames:
            filename = os.path.join(path, str(filename) + suffix)
            gt = []
            with open(filename) as file:
                for line in file.readlines():
                    line = line.split()
                    gt.append([float(i) for i in line])
            gts.append(gt)
        return gts
        pass

    @staticmethod
    def TCI_2(pa, pq, v, region, config):
        '''

        :param pa:
        :param pq:
        :param v:
        :param region:
        :param config:
        :return:
        '''
        avg_v = region.avg_v
        if v == config.max_speend:
            v = avg_v
        res = (pq + pa + 1 - v / avg_v) / 3
        return res
