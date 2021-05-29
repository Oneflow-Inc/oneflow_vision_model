import sys

sys.path.append('../detector')
import detect_video as detect_video


class Detector:
    def __init__(self):
        pass

    def detect(self, img, roi=None):
        '''

        :param img:
        :param roi: [left, top, right, bottom]
        :return:  [[left, top, right, bottom], ...]
        '''
        detect_img = img
        if roi is not None:
            image_h, image_w, _ = detect_img.shape
            l, t, r, b = roi
            l = max(0, l)
            t = max(0, t)
            r = min(r, image_w)
            b = min(b, image_h)
            detect_img = detect_img[t:b, l:r, ...]
            bboxs = detect_video.predict(detect_img)
            for box in bboxs:
                box[0] = box[0] + l
                box[2] = box[2] + l
                box[1] = box[1] + t
                box[3] = box[3] + t
        else:
            bboxs = detect_video.predict(detect_img)
        if len(bboxs) > 0:
            bboxs = [box[:4] for box in bboxs]
        return bboxs
