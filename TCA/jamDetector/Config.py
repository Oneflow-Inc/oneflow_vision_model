class Config:
    def __init__(self):
        self.max_speend = 10000
        # IoU Tracking
        self.sigma_iou = 0.5
        self.miss_track_max_number = 0

        # Jam
        self.use_skip = False
        self.aphpa = 1.25
        self.K = 10
        self.min_k = 2
        self.aphpa_k = 0.4

        self.sigma_h = 0.47
        self.sigma_l = 0.25

        self.FPS = 25

        self.T_dic = 5 * self.FPS  # 拥堵持续时间
        self.T_r = 30 * self.FPS  # 拥堵时沉睡时间
        self.T_s = 3 * self.FPS  # 跳帧时间

        # 是否开启车道拥堵检测
        self.use_lane_jam = True

        self.beta = 0.7
        self.C_q_init = 3
        self.use_update_dynamic = True  # 是否启动动态更新
        self.is_load_track = False  # 是否从文件中加载sort跟踪结果

