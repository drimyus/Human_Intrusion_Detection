from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os
# import pygame
import sys
from script.person_det.detect import PersonDetect
from script.helmet_det.imgnet_utils import ImgNetUtils
from script.climb_det.detect import ClimbDet


COLOR_SAFE = (255, 255, 0)
COLOR_ALARM = (0, 0, 255)
SAFE_MSG = "Motion Detected"
ALARM_MSG = "Possible Intruders...Activating Alarm"

SOUND_ALARM = os.path.dirname(os.path.realpath(__file__)) + "/toolur_EsXo2k.mp3"
# pygame.init()
# pygame.mixer.init()
# pygame.mixer.music.load(SOUND_ALARM)
#
pd = PersonDetect()
cd = ClimbDet()
inu = ImgNetUtils()


class HIDM:
    def __init__(self):
        self.g_alarm_counter = 0
        self.g_motion_counter = 0
        self.DURATION = 20

        self.dst_width, self.dst_height = None, None
        self.skip = None

    def init_HIDM(self, frame_size, zoom_ratio=0.5, skip=20):
        self.dst_width = int(frame_size[0] * zoom_ratio)
        self.dst_height = int(frame_size[1] * zoom_ratio)
        self.skip = skip

    def show_detects(self, show_img, img, persons):
        h_r = float(show_img.shape[0]) / img.shape[0]
        w_r = float(show_img.shape[1]) / img.shape[1]

        # draw the prediction on the frame
        for p in persons:
            (x, y, w, h) = (p['box'] * np.array([w_r, h_r, w_r, h_r])).astype(np.int)
            cv2.rectangle(show_img, (x, y), (x + w, y + h), COLOR_SAFE, 2)
            t_y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(show_img, "person ", (x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SAFE, 2)

        return show_img

    def predict_labels(self, img, mode, persons):
        if len(persons) > 0:
            sys.stdout.write("\r{}".format(SAFE_MSG))
        flag_det = False
        for p in persons:
            (x, y, w, h) = p['box']
            crop = img[int(y):int(y + h), int(x):int(x + w)]

            feature = inu.get_feature(img=crop)
            if mode == "helmet":
                flag_det = inu.helmet_detect(predictions=feature)
            elif mode == "climb":
                flag_det, _, _ = cd.climb_detect(feature=feature)
            if flag_det:
                break
        if flag_det:
            self.func_sound_play()
            sys.stdout.write("\r{}".format(ALARM_MSG))

    #
    def release(self):
        cv2.destroyAllWindows()

    #
    def run_pi_cam(self, feed, mode, show=False):
        cap = cv2.VideoCapture(feed)
        if cap is None:
            sys.stdout.write("can not load video capture.\n")
            sys.exit(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.init_HIDM(frame_size=(width, height))

        cnt = -1
        while True:
            cnt += 1
            suc, frame = cap.read()
            if not suc:
                break

            show_img = frame.copy()
            resize = cv2.resize(frame, (self.dst_width, self.dst_height))
            if cnt % self.skip == 0:
                persons = pd.detect_persons(img=resize)
                self.predict_labels(img=resize, mode=mode, persons=persons)
                if show:
                    result = self.show_detects(show_img=show_img, img=resize, persons=persons)
                    cv2.imshow("result", result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        cap.release()

    #
    def run(self, feed, mode, show=False):
        cap = cv2.VideoCapture(feed)
        if cap is None:
            sys.stdout.write("can not load video capture.\n")
            sys.exit(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.init_HIDM(frame_size=(width, height))

        cnt = -1
        while True:
            cnt += 1
            suc, frame = cap.read()
            if not suc:
                sys.stderr.write("can not read frame.\n")
                break

            show_img = frame.copy()
            resize = cv2.resize(frame, (self.dst_width, self.dst_height))
            if cnt % self.skip == 0:
                persons = pd.detect_persons(img=resize)
                self.predict_labels(img=resize, mode=mode, persons=persons)
                if show:
                    result = self.show_detects(show_img=show_img, img=resize, persons=persons)
                    cv2.imshow("result", result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        cap.release()

    def func_sound_play(self):
        # if not pygame.mixer.music.get_busy() and self.g_alarm_counter == self.DURATION:
        #     print("alarm beef sound")
        #     pygame.mixer.music.play()
        print("alarm beef sound")


if __name__ == '__main__':

    folder = "../data/video/helmet/"
    fns = ["Armed Robbery at I Care Pharmacy May 23 2011.mp4",
           "Bloody robbery caught by a cctv.mp4",
           "cctv_robbery.mp4",
           "Helmet thief stealing in Semenyih pharmacy recorded by GASS HD CCTV.mp4"]
    video = folder + fns[2]
    print(video)

    import json
    with open("../input.json", "r") as fp:
        settings = json.load(fp)

    mode = settings['mode']
    feed = settings['input']
    show = settings['show']
    print("mode: {} \ninput: {}\nshow: {}\n".format(mode, feed, show))

    if mode not in ["climb", "helmet"]:
        sys.stdout.write("not defined mode among ['climb', 'helmet'].\n")
        sys.exit(0)
    elif feed == "PI-CAM":
        HIDM().run_pi_cam(feed=0, mode=mode, show=(show == "True"))
    elif feed == "USB-CAM":
        HIDM().run(feed=0, mode=mode, show=(show == "True"))
    elif os.path.exists(feed):
        HIDM().run(feed=feed, mode=mode, show=(show == "True"))
    else:
        sys.stdout.write("error\n")
        sys.exit(0)
