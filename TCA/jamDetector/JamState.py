class JamState:

    def __init__(self, lane_size):
        self.jam_lane_id = -1
        self.lane_size = lane_size
        self.jam_flags = [False, ] * (lane_size + 1)
        self.jam_start_times = [0, ] * (lane_size + 1)
        self.jam_end_times = [0, ] * (lane_size + 1)
        self.jam_indexs = [[], ] * (lane_size + 1)

    def reset(self):
        '''
            重置
        :return:
        '''
        for i in range(len(self.jam_flags)):
            self.jam_flags[i] = False
        for i in range(len(self.jam_start_times)):
            self.jam_start_times[i] = 0
        for i in range(len(self.jam_end_times)):
            self.jam_end_times[i] = 0
        for i in range(len(self.jam_indexs)):
            self.jam_indexs[i].clear()
        self.jam_lane_id = -1

    def update_jam(self, lane_id, cur_time, index=0, is_jam=True):
        '''
            更新
        :param lane_id:
        :param cur_time:
        :param index:
        :param is_jam:
        :return:
        '''
        assert lane_id >= -1 and lane_id <= self.lane_size
        if lane_id < 0:
            lane_id = self.lane_size
        if not is_jam:
            self.jam_indexs[lane_id].clear()
            self.jam_flags[lane_id] = False
        else:
            if self.jam_flags[lane_id]:
                self.jam_end_times[lane_id] = cur_time
            else:
                self.jam_flags[lane_id] = True
                self.jam_start_times[lane_id] = self.jam_end_times[lane_id] = cur_time
            self.jam_indexs[lane_id].append(index)

    def get_jam_start_time(self):
        assert self.jam_lane_id >= 0 and self.jam_lane_id < len(self.jamEndTimeVec_)
        return self.jam_end_times[self.jam_lane_id]

    def get_jam_info(self, thr):
        '''
            获取拥堵信息
        :param thr:
        :return: (flag, [start_time, end_time])
        '''
        jam_start_time = 0
        jam_end_time = 0
        flag = False
        for i in range(len(self.jam_end_times)):
            if self.jam_flags[i] and (self.jam_end_times[i] - self.jam_start_times[i]) >= thr:
                if not flag:
                    jam_start_time = self.jam_start_times[i]
                    jam_end_time = self.jam_end_times[i]
                    flag = True
                else:
                    jam_start_time = min(jam_start_time, self.jam_start_times[i])
                    jam_end_time = max(jam_end_time, self.jam_end_times[i])
        return flag, [jam_start_time, jam_end_time]
