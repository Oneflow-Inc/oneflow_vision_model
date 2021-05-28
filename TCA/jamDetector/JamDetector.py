import time
from jamDetector.Region import Region
from jamDetector.TrackingObj import TrackingObj
from jamDetector.Utility import Utility
from jamDetector.JamState import JamState
import os
import cv2


class JamDetector:

    def __init__(self, config):
        self.id = 0
        self.config = config

        self.is_jam = False  # 当前是否拥堵

        self.trackingObjs = []  # 记录跟踪的目标
        self.cur_timestamp = 0  # 当前时间

        self.is_skipping = False
        self.skip_timestamp = 0

        self.INVALID = -1000
        self.last_index = self.INVALID

        pass

    def set_index_function(self, index_function):
        self.index_function = index_function  # 计算指标的回调

    def reset(self):
        self.id = 0
        self.is_jam = False
        self.trackingObjs.clear()
        self.cur_timestamp = 0
        self.is_skipping = False
        self.skip_timestamp = 0
        self.last_index = self.INVALID
        pass

    def tracking(self, bboxs):
        '''
            跟踪
        :param bboxs: [[left, top, right, bottom],[left, top, right, bottom],...]
        :return:
        '''
        old_len = len(self.trackingObjs)
        for obj in self.trackingObjs:
            obj.un_update()
        for bbox in bboxs:
            flag = False
            lane_id = self.region_info.in_which_region(Utility.get_center(bbox))
            if lane_id < 0:
                continue
            for i in range(old_len):
                if not self.trackingObjs[i].is_update and Utility.get_iou(bbox, self.trackingObjs[
                    i].get_bbox()) >= self.config.sigma_iou:
                    self.trackingObjs[i].update(bbox, lane_id, self.cur_timestamp)
                    flag = True
                    break
            if not flag:
                self.id += 1
                self.trackingObjs.append(TrackingObj(self.id, lane_id, self.cur_timestamp))
                self.trackingObjs[-1].update(bbox, lane_id, self.cur_timestamp)
        i = 0
        while i < len(self.trackingObjs):
            if not self.trackingObjs[i].is_update:
                self.trackingObjs[i].missing()
                if self.trackingObjs[i].track_frame <= self.region_info.total_k \
                        or self.trackingObjs[i].miss_frame >= self.config.miss_track_max_number:
                    self.trackingObjs.pop(i)
                    continue
            i += 1

    def tracking_with_id(self, bboxs):
        '''
            跟踪还原
        :param bboxs: [[id, left, top, right, bottom],[id, left, top, right, bottom],...]
        :return:
        '''
        old_len = len(self.trackingObjs)
        for obj in self.trackingObjs:
            obj.un_update()
        for bbox in bboxs:
            flag = False
            lane_id = self.region_info.in_which_region(Utility.get_center(bbox[1:]))
            if lane_id < 0:
                continue
            for i in range(old_len):
                if self.trackingObjs[i].id == bbox[0]:
                    self.trackingObjs[i].update(bbox[1:], lane_id, self.cur_timestamp)
                    flag = True
                    break
            if not flag:
                self.trackingObjs.append(TrackingObj(bbox[0], lane_id, self.cur_timestamp))
                self.trackingObjs[-1].update(bbox[1:], lane_id, self.cur_timestamp)
        i = 0
        while i < len(self.trackingObjs):
            if not self.trackingObjs[i].is_update:
                self.trackingObjs[i].missing()
                if self.trackingObjs[i].track_frame <= self.region_info.total_k \
                        or self.trackingObjs[i].miss_frame >= self.config.miss_track_max_number:
                    self.trackingObjs.pop(i)
                    continue
            i += 1
        pass

    def is_need_skip(self):
        '''
            当前是否需要跳帧
        :return:
        '''
        # if not self.use_skip:
        #     return False
        if self.is_jam:
            pass
        elif self.is_skipping:
            if self.cur_timestamp - self.skip_timestamp < self.config.T_s:
                return True
            self.is_skipping = False
        return False

    # def detect(self, timestamp, img, detector, tracker):
    #     '''
    #     :param timestamp:
    #     :param img:
    #     :param detector:
    #     :param tracker:
    #     :return:
    #     '''
    #     begin_time = time.time()
    #     self.frame_number += 1
    #     self.cur_timestamp = timestamp
    #     if self.is_need_skip():
    #         self.last_index = self.INVALID
    #         self.skip_frame_number += 1
    #         end_time = time.time()
    #         self.total_time += end_time - begin_time
    #         return False
    #     detect_bbox = detector(img)
    #     track_bbox = tracker(timestamp, img, detect_bbox)
    #
    #     self.detectJam()
    #
    #     flag, jam_info = self.jam_state.get_jam_info(self.config.T_dic)
    #     if flag:
    #         if len(self.jam_detection_res) > 0:
    #             self.jam_detection_res.append(jam_info)
    #         else:
    #             if self.jam_detection_res[-1][1] <= jam_info[0] \
    #                     or self.jam_detection_res[-1][0] >= jam_info[1]:
    #                 self.jam_detection_res.append(jam_info)
    #             else:
    #                 self.jam_detection_res[-1][0] = min(self.jam_detection_res[-1][0], jam_info[0])
    #                 self.jam_detection_res[-1][1] = max(self.jam_detection_res[-1][1], jam_info[1])
    #     end_time = time.time()
    #     self.total_time += end_time - begin_time

    def detect_jam(self):
        '''
            进行拥堵检测
        :return:
        '''
        lane_line_size = self.region_info.lane_line_size
        lane_size = self.region_info.lane_size

        pa = [0, ] * lane_line_size
        pq = [0, ] * lane_line_size

        v = [0, ] * lane_line_size
        numv = [0, ] * lane_line_size

        tmpv = [0, ] * lane_line_size
        tmpnumv = [0, ] * lane_line_size

        for obj in self.trackingObjs:
            if obj.miss_frame > 0:
                continue
            lane_id = obj.lane_id
            assert lane_id >= 0 and lane_id < lane_line_size
            pa[lane_id] += Utility.get_rect_area(obj.get_bbox())
            pq[lane_id] += 1

            pa[lane_size] += Utility.get_rect_area(obj.get_bbox())
            pq[lane_size] += 1
            if self.cur_timestamp == obj.start_time:
                continue
            t = (self.cur_timestamp - obj.start_time) / self.config.FPS  # fixed
            tmp = obj.get_moving_dist() / (obj.get_avg_length(self.region_info) * t)
            tmpv[lane_id] += tmp
            tmpnumv[lane_id] += 1
            if obj.existing_frame_number() >= max(self.config.min_k, self.config.aphpa_k * self.region_info.total_k):
                v[lane_id] += tmp
                numv[lane_id] += 1
        for i in range(lane_size):
            tmpv[lane_size] += tmpv[i]
            tmpnumv[lane_size] += tmpnumv[i]
            if numv[i] > 0:
                v[lane_size] += v[i]
                numv[lane_size] += numv[i]
        flag = False
        for lane_id in range(lane_line_size):
            if not self.config.use_lane_jam and lane_id != lane_size:
                continue
            tmp_pa = pa[lane_id] / self.region_info.get_CA(lane_id)
            tmp_pq = pq[lane_id] / self.region_info.get_CQ(lane_id)
            tmp_v = self.config.max_speend
            if numv[lane_id] != 0:
                tmp_v = v[lane_id] / numv[lane_id]
            index = self.index_function(tmp_pa, tmp_pq, tmp_v, self.region_info, self.config)
            if self.last_index[lane_id] != self.INVALID:
                index = self.config.beta * index + (1 - self.config.beta) * self.last_index[lane_id]
            self.last_index[lane_id] = index
            # if self.last_index[lane_id] != self.INVALID:
            #     index = self.config.beta * index + (1 - self.config.beta) * self.last_index[lane_id]
            # self.last_index[lane_id] = index

            if lane_id == lane_size:
                self.display1 = 'Traffic area density: ' + str(round(tmp_pa, 2))
                self.display2 = 'Traffic quantity density: ' + str(round(tmp_pq, 2))
                self.display3 = 'Traffic quantity velocity: ' + str(round(tmp_v, 2))
                self.display4 = 'Traffic congestion index: ' + str(round(index, 2))

            is_jam = index >= self.config.sigma_h
            if is_jam and self.config.use_update_dynamic:
                flag = self.region_info.update_CQ_lane(tmp_pq, lane_id) | flag
            if not is_jam and lane_id == self.region_info.lane_size:
                if index <= self.config.sigma_l:
                    if self.config.use_update_dynamic:
                        self.region_info.update_total_K(self.trackingObjs)
                        if tmp_v != self.config.max_speend and tmp_v != 0:
                            self.region_info.update_avg_v(tmp_v)
                    if self.config.use_skip:
                        self.is_skipping = True
                        self.skip_timestamp = self.cur_timestamp
            self.jam_state.update_jam(lane_id, self.cur_timestamp, index, is_jam)

        if flag:
            self.region_info.update_CQ_all()

    def detect_with_filename(self, video_filename, region_filename, bbox_path, is_visualization=True):
        '''
            进行拥堵检测
            检测器结果从文件中读取（检测器结果被预先存在文件中国）
        :param video_filename:
        :param region_filename:
        :param bbox_path:
        :param is_visualization:
        :return: [consumeTime, skipFrame, totalFrame, pred=[[start_time, end_time], ... ]]
        '''
        self.reset()
        self.region_info = Region(region_filename, self.config)  # 道路信息
        self.jam_state = JamState(self.region_info.lane_size)  # 拥堵状态

        self.last_index = [self.INVALID, ] * self.region_info.lane_line_size

        filenames = os.listdir(bbox_path)
        key_fn = lambda key: int(key.split('.')[0])
        filenames.sort(key=key_fn)
        total_time = 0
        # [[start_time,end_time,index], ]
        jam_detection_res = []
        cur_frame = -1
        skip_frame = 0
        if is_visualization:
            cap = cv2.VideoCapture(video_filename)
        for filename in filenames:
            bboxs = Utility.get_obj_from_file(os.path.join(bbox_path, filename), self.config.is_load_track)
            ret, img = False, None
            if is_visualization:
                if cap.isOpened():
                    ret, img = cap.read()
                if ret:
                    self.region_info.draw(img)
            begin_time = time.time()
            cur_frame += 1
            self.cur_timestamp = cur_frame
            if self.is_need_skip():
                self.last_index = self.INVALID
                self.skip_frame_number += 1
                # end_time = time.time()
                # cur_consume_time = end_time - begin_time
                # total_time += cur_consume_time
                if ret:
                    cv2.imshow(video_filename, img)
                    cv2.waitKey(int(1000 / self.config.FPS))
                continue
            if self.config.is_load_track:
                self.tracking_with_id(bboxs)
            else:
                self.tracking(bboxs)
            self.detect_jam()

            flag, jam_info = self.jam_state.get_jam_info(self.config.T_dic)
            if flag:
                if len(jam_detection_res) == 0:
                    jam_detection_res.append(jam_info)
                else:
                    if jam_detection_res[-1][1] < (jam_info[0] - 1) \
                            or (jam_detection_res[-1][0] - 1) > jam_info[1]:
                        jam_detection_res.append(jam_info)
                    else:
                        jam_detection_res[-1][0] = min(jam_detection_res[-1][0], jam_info[0])
                        jam_detection_res[-1][1] = max(jam_detection_res[-1][1], jam_info[1])
            end_time = time.time()
            cur_consume_time = end_time - begin_time
            cur_consume_time = max(0, cur_consume_time)
            total_time += cur_consume_time

            if ret and is_visualization:
                for track_obj in self.trackingObjs:
                    track_obj.draw(img)

                font = cv2.FORMATTER_FMT_DEFAULT
                color = (255, 0, 0) if not flag else (0, 0, 255)
                cv2.putText(img, self.display1, (10, 20), font, 0.5, color)
                cv2.putText(img, self.display2, (10, 40), font, 0.5, color)
                cv2.putText(img, self.display3, (10, 60), font, 0.5, color)
                cv2.putText(img, self.display4, (10, 80), font, 0.5, color)

                cv2.imshow(video_filename, img)
                cv2.imwrite("res//"+str(cur_frame)+".jpg", img)
                cv2.waitKey(max(1,int(1000 / self.config.FPS - cur_consume_time)))
        if is_visualization:
            cv2.destroyAllWindows()
        return total_time, skip_frame, cur_frame, jam_detection_res

    def detect(self, video_filename, region_filename, detector, is_visualization=False):
        '''
            进行拥堵检测
        :param video_filename:
        :param region_filename:
        :param detector:
        :param is_visualization:
        :return: [consumeTime, skipFrame, totalFrame, pred=[[start_time, end_time], ... ]]
        '''
        self.reset()
        self.region_info = Region(region_filename, self.config)  # 道路信息
        self.jam_state = JamState(self.region_info.lane_size)  # 拥堵状态

        self.last_index = [self.INVALID, ] * self.region_info.lane_line_size

        total_time = 0
        # [[start_time,end_time,index], ]
        jam_detection_res = []
        cur_frame = -1
        skip_frame = 0
        cap = cv2.VideoCapture(video_filename)
        while cap.isOpened():
            ret, img = cap.read()
            detect_img = img.copy()
            if not ret:
                break
            if is_visualization:
                self.region_info.draw(img)
            begin_time = time.time()
            cur_frame += 1
            self.cur_timestamp = cur_frame
            if self.is_need_skip():
                self.last_index = self.INVALID
                self.skip_frame_number += 1
                # end_time = time.time()
                # total_time += end_time - begin_time
                if ret:
                    cv2.imshow(video_filename, img)
                    cv2.waitKey(int(1000 / self.config.FPS))
                continue
            bboxs = detector.detect(detect_img, self.region_info.jam_detect_region)
            self.tracking(bboxs)
            self.detect_jam()

            flag, jam_info = self.jam_state.get_jam_info(self.config.T_dic)
            if flag:
                if len(jam_detection_res) == 0:
                    jam_detection_res.append(jam_info)
                else:
                    if jam_detection_res[-1][1] < (jam_info[0] - 1) \
                            or (jam_detection_res[-1][0] - 1) > jam_info[1]:
                        jam_detection_res.append(jam_info)
                    else:
                        jam_detection_res[-1][0] = min(jam_detection_res[-1][0], jam_info[0])
                        jam_detection_res[-1][1] = max(jam_detection_res[-1][1], jam_info[1])
            end_time = time.time()
            cur_consume_time = end_time - begin_time
            cur_consume_time = max(0, cur_consume_time)
            total_time += cur_consume_time
            if is_visualization:
                for track_obj in self.trackingObjs:
                    track_obj.draw(img)

                font = cv2.FORMATTER_FMT_DEFAULT
                color = (255, 0, 0) if not flag else (0, 0, 255)
                cv2.putText(img, self.display1, (10, 20), font, 0.5, color)
                cv2.putText(img, self.display2, (10, 40), font, 0.5, color)
                cv2.putText(img, self.display3, (10, 60), font, 0.5, color)
                cv2.putText(img, self.display4, (10, 80), font, 0.5, color)

                cv2.imshow(video_filename, img)
                cv2.waitKey(max(1,int(1000 / self.config.FPS - cur_consume_time)))
        if is_visualization:
            cv2.destroyAllWindows()
        return total_time, skip_frame, cur_frame, jam_detection_res
