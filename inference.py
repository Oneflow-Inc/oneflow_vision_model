from face_seg_linknet34.models.LinkNet34 import LinkNet34
from face_seg_linknet34.utils.numpy_data_utils import load_image,image_transform
import oneflow.experimental as flow
import cv2
import time
import numpy as np
import sys
import imageio
from PIL import Image
from face_seg_linknet34.utils.load_model_params import load_model_params


class CaptureFrames():

    def __init__(self, model, source, show_mask=False):
        self.model = model.to('cuda')
        self.source = source
        self.show_mask = show_mask

    def __call__(self, source):
        self.capture_frames(source)

    def capture_frames(self, source):
        camera = cv2.VideoCapture(source)
        time.sleep(2)
        self.model.eval()
        (grabbed, orig) = camera.read()
        camera.set(cv2.CAP_PROP_FPS, 25.0)

        if (camera.isOpened() == False):
            print("Unable to read video")

        time.sleep(2)

        self.model.eval()
        time_1 = time.time()

        fps = 27
        size = (640,480)
        fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
        print('111111')
        result_video = cv2.VideoWriter("result_video2.mp4", fourcc, fps, size)
        self.frames_count = 0
        save_num = 100
        while grabbed:
            (grabbed, orig) = camera.read()
            if not grabbed:
                continue

            shape = orig.shape[0:2]
            imgs = image_transform(Image.fromarray(orig))
            imgs = flow.Tensor(imgs).to('cuda')
            pred = self.model(imgs)

            mask = pred.data.numpy()
            mask = mask.squeeze()
            mask = cv2.resize(mask, (shape[1], shape[0]))
            mask = mask > 0.8

            rgba = cv2.cvtColor(orig, cv2.COLOR_BGR2BGRA)
            ind = np.where(mask == 0)
            rgba[ind] = rgba[ind] - [0, 0, 0, 180]

            canvas = Image.new('RGBA', (rgba.shape[1], rgba.shape[0]), (255, 255, 255, 255))
            canvas.paste(Image.fromarray(rgba), mask=Image.fromarray(rgba))
            rgba = np.array(canvas)
            rgb = cv2.cvtColor(rgba, cv2.COLOR_BGRA2BGR)
            k = cv2.waitKey(1)

            if self.show_mask:
                cv2.imshow('mask', rgb)

            if save_num > 0:
                result_video.write(rgb)
                save_num = save_num - 1

            if self.frames_count % 30 == 29:
                time_2 = time.time()
                sys.stdout.write(f'\rFPS: {30 / (time_2 - time_1)}')
                sys.stdout.flush()
                time_1 = time.time()

            if k != -1:
                self.terminate(camera)
                break
            self.frames_count += 1
        self.terminate(camera)

    def terminate(self, camera):
        cv2.destroyAllWindows()
        camera.release()


# #set path=0 for webcam or set to a video file

flow.env.init()
flow.enable_eager_execution()

#path = "output_liyue.avi"
path = 0
model_path = "linknet.pth"
pretrained_model_path = "resnet34-333f7ec4.pth"

model = LinkNet34(pretrained=False, pretrained_model_path = pretrained_model_path)
linknet34 = load_model_params(model,model_path)
print('Load successfully')
c = CaptureFrames(model, 0, True)
c(path)