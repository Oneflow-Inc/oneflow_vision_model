from Utility import Utility
from Region import Region
import random
import cv2


class TrackingObj:

    def __init__(self, id, lane_id, start_time):
        self.id = id
        self.lane_id = lane_id
        self.start_time = start_time
        self.end_time = start_time
        self.is_update = False
        self.miss_frame = 0
        self.track_frame = 0
        self.bboxs = []
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        self.color = (b, g, r)
        pass

    def get_bbox(self):
        '''

        :return:
        '''
        assert len(self.bboxs) > 0
        return self.bboxs[-1]

    def get_moving_dist(self):
        '''
            移动距离
        :return:
        '''
        if len(self.bboxs) < 2:
            return 0
        center1 = Utility.get_center(self.bboxs[0])
        center2 = Utility.get_center(self.bboxs[-1])
        return Utility.get_dist(center1, center2)

    def get_avg_length(self, region_info: Region):
        '''
            获取平均车速长度
        :param region_info:
        :return:
        '''
        slope = region_info.get_slope(self.lane_id)
        res = 0
        if abs(slope) >= 1:
            for box in self.bboxs:
                res += box[3] - box[1]
        else:
            for box in self.bboxs:
                res += box[2] - box[0]
        res /= len(self.bboxs)
        return res

    def existing_frame_number(self):
        return len(self.bboxs)

    def un_update(self):
        self.is_update = False

    def update(self, rect, lane_id, cur_time):
        self.bboxs.append(rect)
        self.lane_id = lane_id
        self.end_time = cur_time
        self.is_update = True
        if self.miss_frame == 0:
            self.track_frame += 1
        else:
            self.track_frame = 1
        self.miss_frame = 0

    def missing(self):
        self.miss_frame += 1

    def draw(self, img):
        box = self.bboxs[-1]
        box = list([int(i) for i in box])
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=self.color)
        for box in self.bboxs:
            center = Utility.get_center(box)
            cv2.circle(img, (int(center[0]), int(center[1])), 1, color=self.color)
        pass
