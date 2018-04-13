from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os
import pygame
from script.person_det.detect import PersonDetect
from script.helmet_det.imgnet_utils import ImgNetUtils
from script.climb_det.detect import ClimbDet


COLOR_SAFE = (255, 255, 0)
COLOR_ALARM = (0, 0, 255)
SAFE_MSG = "Motion Detected"
ALARM_MSG = "Possible Intruders...Activating Alarm"

SOUND_ALARM = os.path.dirname(os.path.realpath(__file__)) + "/toolur_EsXo2k.mp3"
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(SOUND_ALARM)

pd = PersonDetect()
inu = ImgNetUtils()
cd = ClimbDet()


class HIDM:
    def __init__(self):
        self.g_alarm_counter = 0
        self.g_motion_counter = 0
        self.DURATION = 20

        self.dst_width, self.dst_height = None, None
        self.skip = None

    def init_HIDM(self, frame_size, zoom_ratio = 0.5, skip=5):
        self.dst_width = int(frame_size[0] * zoom_ratio)
        self.dst_height = int(frame_size[1] * zoom_ratio)
        self.skip = skip

    def show_detects(self, show_img, img, trackers):
        h_r = float(show_img.shape[0]) / img.shape[0]
        w_r = float(show_img.shape[1]) / img.shape[1]

        # draw the prediction on the frame
        for p in trackers:
            self.g_motion_counter = self.DURATION
            label = p['label']
            (x, y, w, h) = (p['box'] * np.array([w_r, h_r, w_r, h_r])).astype(np.int)
            if label == '':
                color = COLOR_SAFE
            else:
                color = COLOR_ALARM
                self.g_alarm_counter = self.DURATION

            cv2.rectangle(show_img, (x, y), (x + w, y + h), color, 2)
            t_y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(show_img, "person " + label, (x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.g_motion_counter > 0:
            cv2.putText(show_img, SAFE_MSG, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SAFE, 3)
        if self.g_alarm_counter > 0:
            cv2.putText(show_img, ALARM_MSG, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ALARM, 3)

        self.func_sound_play()
        self.g_alarm_counter -= 1
        self.g_motion_counter -= 1

        return show_img

    def predict_labels(self, img, persons):
        for p in persons:
            (x, y, w, h) = p['box']
            crop = img[int(y):int(y + h), int(x):int(x + w)]

            feature = inu.get_feature(img=crop)
            is_helmet = inu.helmet_detect(predictions=feature)
            is_climb, _, _ = cd.climb_detect(feature=feature)

            p['label'] = is_climb * "Climb" + is_helmet * "Helmet"

    #
    def proc(self, frame, pos):
        show_img = frame.copy()
        resize = cv2.resize(frame, (self.dst_width, self.dst_height))
        if pos % self.skip == 0:
            pd.update_trackers(img=resize)
            self.predict_labels(img=resize, persons=pd.get_trackers())
        else:
            pd.upgrade_trackers(img=resize, chk_num_thresh=3)

        result = self.show_detects(show_img=show_img, img=resize, trackers=pd.get_trackers())
        return result

    #
    def release(self):
        cv2.destroyAllWindows()

    #
    def run(self, cap):
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.init_HIDM(frame_size=(width, height))

        cnt = -1
        while True:
            cnt += 1
            suc, frame = cap.read()
            if not suc:
                break

            result = self.proc(frame=frame, pos=cnt)

            cv2.imshow("result", result)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        cap.release()

    def func_sound_play(self):
        if not pygame.mixer.music.get_busy() and self.g_alarm_counter == self.DURATION:
            print "alarm beef sound"
            pygame.mixer.music.play()


if __name__ == '__main__':

    folder = "../data/video/helmet/"
    fns = ["Armed Robbery at I Care Pharmacy May 23 2011.mp4",
           "Bloody robbery caught by a cctv.mp4",
           "cctv_robbery.mp4",
           "Helmet thief stealing in Semenyih pharmacy recorded by GASS HD CCTV.mp4"]
    video = folder + fns[2]
    print(video)

    if os.path.exists(video):
        HIDM().run(cap=cv2.VideoCapture(video))
    else:
        print "no exist file"
