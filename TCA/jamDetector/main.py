from jamDetector.Utility import Utility
from jamDetector.JamDetector import JamDetector
from jamDetector.Detector import Detector
from jamDetector.Config import Config
import os
import argparse

parser = argparse.ArgumentParser(description='flags for detect video')
parser.add_argument(
    '-jam_dataset_path',
    '--jam_dataset_path',
    type = str,
    required = True
)
parser.add_argument(
    '-with_detector',
    '--with_detector',
    type = bool,
    default=False,
    required = False
)
parser.add_argument(
    '-is_visualization',
    '--is_visualization',
    type = bool,
    default = False,
    required = False
)
args = parser.parse_args()
PATH = args.jam_dataset_path
BBOX = os.path.join(PATH, r'detect_res')
REGION = os.path.join(PATH, 'calib')
GT = os.path.join(PATH, 'gt')
VIDEO = os.path.join(PATH, 'video')


def Task(jam_detector, indexs, detector=None, is_visualization=False):
    '''

    :param jam_detector:
    :param indexs:
    :param detector:
    :param is_visualization:
    :return:
    '''
    preds = []
    gts = Utility.get_gt(GT, indexs)

    total_time = 0
    total_frame = 0
    total_skip_frame = 0
    for i in range(len(indexs)):
        index = str(indexs[i])
        print(" index: " + index)
        video_filename = os.path.join(VIDEO, index + ".mp4")
        region_filename = os.path.join(REGION, index + ".txt")
        if detector is None:
            bbox_path = os.path.join(BBOX, index)
            total_time_, skip_frame, cur_frame, jam_detection_res = \
                jam_detector.detect_with_filename(video_filename, region_filename, bbox_path, is_visualization)
        else:
            total_time_, skip_frame, cur_frame, jam_detection_res = \
                jam_detector.detect(video_filename, region_filename, detector, is_visualization)
        preds.append(jam_detection_res)
        total_time += total_time_
        total_frame += cur_frame
        total_skip_frame += skip_frame
    precise, recall, f1, switch_rate, hit_rate = Utility.getPRF1(preds, gts)
    print("precise: " + str(precise) + " recall: " + str(recall) + " F1: "
          + str(f1) + " switch rate: " + str(switch_rate)
          + " hit rate: " + str(hit_rate))
    print("total_time: " + str(total_time))
    print("total_frame: " + str(total_frame))
    print("total_skip_frame: " + str(total_skip_frame))
    pass


def main():
    indexs = []
    for i in range(27):
        indexs.append(i + 1)
    config = Config()
    jam_detector = JamDetector(config)
    jam_detector.set_index_function(Utility.TCI_2)
    detector = None
    if args.with_detector:
        detector = Detector()
    Task(jam_detector, indexs, detector, args.is_visualization)


if __name__ == '__main__':
    main()
    pass
