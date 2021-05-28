import os
from jamDetector.Utility import Utility
from jamDetector.Config import Config
import cv2


class Region:

    def __init__(self, filename, config):
        self.lane_lines = []

        self.c_a = []
        self.c_q = []
        self.c_q_num = []

        self.total_k = 0
        self.total_k_num = 0

        self.avg_v = 1
        self.avg_v_num = 0

        self.config = config

        with open(filename) as fin:
            line = fin.readline()
            left, top, right, bottom = line.split()
            self.jam_detect_region = [int(left), int(top), int(right), int(bottom)]
            lines = fin.readlines()
            for line in lines:
                left, top, right, bottom = line.split()
                self.lane_lines.append([int(left), int(top), int(right), int(bottom)])
            self.lane_size = len(self.lane_lines) - 1
            self.lane_line_size = len(self.lane_lines)
            total_c_q = 0
            for i in range(self.lane_size):
                self.c_a.append(self.get_area(self.jam_detect_region, i))
                self.c_q.append(self.config.C_q_init)
                self.c_q_num.append(1)
                total_c_q += self.c_q[-1]
            self.c_a.append(self.get_area(self.jam_detect_region, -1))
            self.c_q.append(total_c_q)
            self.c_q_num.append(1)

    def draw(self, img):
        for lane_line in self.lane_lines:
            cv2.line(img, tuple(lane_line[:2]), tuple(lane_line[2:]), color=(0, 255, 0))
        cv2.rectangle(img, tuple(self.jam_detect_region[:2]), tuple(self.jam_detect_region[2:]), color = (0, 255, 0))

        pass

    def in_which_region(self, point):
        '''
        点在哪一个车道内
        :param point: [x,y]
        :return:  int, 表示车道index（-1表示不在区域内）
        '''
        if Utility.is_in_rect(point, self.jam_detect_region):
            for i in range(self.lane_size):
                lane = self.lane_lines[i]
                left = Utility.get_x(lane[0:2], lane[2:], point[1]) - point[0]
                lane = self.lane_lines[i + 1]
                right = Utility.get_x(lane[0:2], lane[2:], point[1]) - point[0]
                if left * right <= 0:
                    return i
        return -1

    def get_area(self, rect, lane_index):
        '''
            获取rect和某一个车道的交集面积
            lane_index=-1表示和最外围的车道的交集面积
        :param rect:
        :param lane_index:
        :return:
        '''
        assert lane_index >= -1 and lane_index < self.lane_size
        which_line1 = 0 if lane_index < 0 else lane_index
        which_line2 = self.lane_size if lane_index < 0 else lane_index + 1
        area = 0
        left, top, right, bottom = Utility.get_intersection(self.jam_detect_region, rect)
        for y in range(top, bottom):
            for x in range(left, right):
                if self.is_in_region([x, y], which_line1, which_line2):
                    area += 1
        return area

    def is_in_region(self, point, which_line1, which_line2):
        '''
            判断点是否在车道内
        :param point: [x,y]
        :param which_line1: int
        :param which_line2: int
        :return: bool
        '''
        assert which_line1 >= 0 and which_line1 < self.lane_line_size
        assert which_line2 >= 0 and which_line2 < self.lane_line_size
        left = Utility.get_x(self.lane_lines[which_line1][0:2],
                             self.lane_lines[which_line1][2:], point[1]) - point[0]
        right = Utility.get_x(self.lane_lines[which_line2][0:2],
                              self.lane_lines[which_line2][2:], point[1]) - point[0]
        return left * right <= 0

    def get_slope(self, lane_id):
        '''
            获取车道线斜率
        :param lane_id:
        :return:
        '''
        assert lane_id >= 0 and lane_id < self.lane_size
        return Utility.get_slope(self.lane_lines[lane_id][:2], self.lane_lines[lane_id][2:])

    def get_CA(self, lane_id):
        if lane_id == -1:
            lane_id = self.lane_size
        return self.c_a[lane_id]

    def get_CQ(self, lane_id):
        if lane_id == -1:
            lane_id = self.lane_size
        return self.c_q[lane_id]

    def update_CQ_lane(self, pq, lane_id):
        '''
            更新CQ
        :param pq:
        :param lane_id:
        :return:
        '''
        assert lane_id >= 0 and lane_id <= self.lane_size
        if lane_id == self.lane_size:
            return False
        tmp = pq * self.c_q[lane_id]
        flag = False
        if tmp >= self.config.C_q_init:
            self.c_q[lane_id] = (self.c_q[lane_id] * self.c_q_num[lane_id] + tmp) / (self.c_q_num[lane_id] + 1)
            self.c_q_num[lane_id] += 1
            flag = True
        return flag

    def update_avg_v(self, v):
        '''
            更新平均速度
        :param v:
        :return:
        '''
        self.avg = (self.avg_v * self.avg_v_num + v) / (self.avg_v_num + 1)
        self.avg_v_num += 1

    def update_total_K(self, tracking_objs):
        '''
            更新total_k
        :param tracking_objs:  [TrackingObj, TrackingObj,...]
        :return:
        '''
        for obj in tracking_objs:
            bboxs = obj.bboxs
            if len(bboxs) < 2:
                continue
            front = bboxs[0]
            back = bboxs[-1]

            def fn(rect1, rect2, jam_detection_region):
                '''

                :param rect1: [left, top, right, bottom]
                :param rect2: [left, top, right, bottom]
                :param jam_detection_region: [left, top, right, bottom]
                :return:
                '''
                tmp1 = rect1[1] - jam_detection_region[1]
                tmp2 = rect2[3] - jam_detection_region[3]
                tmp3 = rect1[0] - jam_detection_region[0]
                tmp4 = rect2[2] - jam_detection_region[2]
                return (tmp1 <= 0 and tmp2 >= 0) or (tmp3 <= 0 and tmp4 >= 0)

            if fn(front, back, self.jam_detect_region) or fn(back, front, self.jam_detect_region):
                flag = True
                lane_id = obj.lane_id
                for i in range(1, len(bboxs)):
                    if self.in_which_region(Utility.get_center(bboxs[i])) != lane_id:
                        flag = False
                        break
                if flag:
                    self.total_k = (self.total_k * self.total_k_num + obj.existing_frame_number()) / (
                            self.total_k_num + 1)
                    self.total_k_num += 1

    def update_CQ_all(self):
        self.c_q[self.lane_size] = 0
        for i in range(self.lane_size):
            self.c_q[self.lane_size] += self.c_q[i]
