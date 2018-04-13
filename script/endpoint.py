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


import argparse
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path of the input video file(string) or usb camera id(int)")
a = parser.parse_args()


pd = PersonDetect()
inu = ImgNetUtils()
cd = ClimbDet()

COLOR_SAFE = (255, 255, 0)
COLOR_ALARM = (0, 0, 255)

SAFE_MSG = "Motion Detected"
ALARM_MSG = "Possible Intruders...Activating Alarm"

DURATION = 20

g_alarm_counter = 0
g_motion_counter = 0

SOUND_ALARM = os.path.dirname(os.path.realpath(__file__)) + "/toolur_EsXo2k.mp3"
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(SOUND_ALARM)

# stuff for playing sounds


def show_rects(show_img, img, trackers):
    global g_alarm_counter, g_motion_counter

    h_r = float(show_img.shape[0]) / img.shape[0]
    w_r = float(show_img.shape[1]) / img.shape[1]

    # draw the prediction on the frame
    for p in trackers:
        g_motion_counter = DURATION
        label = p['label']
        (x, y, w, h) = (p['box'] * np.array([w_r, h_r, w_r, h_r])).astype(np.int)
        if label == '':
            color = COLOR_SAFE
        else:
            color = COLOR_ALARM
            g_alarm_counter = DURATION

        cv2.rectangle(show_img, (x, y), (x + w, y + h), color, 2)
        t_y = y - 15 if y - 15 > 15 else y + 15
        cv2.putText(show_img, "person " + label, (x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if g_motion_counter > 0:
        cv2.putText(show_img, SAFE_MSG, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SAFE, 3)
    if g_alarm_counter > 0:
        cv2.putText(show_img, ALARM_MSG, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ALARM, 3)

    func_sound_play()
    g_alarm_counter -= 1
    g_motion_counter -= 1

    return show_img


def alarm_msg(show_img):
    cv2.putText(show_img, "Alarm Activating..", (20, show_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                COLOR_ALARM, 3)


def predict_labels(img, persons):
    for p in persons:
        (x, y, w, h) = p['box']
        crop = img[int(y):int(y + h), int(x):int(x + w)]

        feature = inu.get_feature(img=crop)
        is_helmet = inu.helmet_detect(predictions=feature)
        is_climb, _, _ = cd.climb_detect(feature=feature)

        p['label'] = is_climb * "Climb" + is_helmet * "Helmet"


def func_run(video, zoom_ratio=0.5, skip=5):

    print("starting video stream...")

    cap = cv2.VideoCapture(video)
    time.sleep(2.0)
    fps = FPS().start()
    dst_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * zoom_ratio)
    dst_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * zoom_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    saver = cv2.VideoWriter("result.avi", fourcc, 30.0, (dst_width, dst_height))

    cnt = -1
    # loop over the frames from the video stream
    while True:
        cnt += 1
        suc, frame = cap.read()
        if not suc:
            break
        show_img = frame.copy()
        resize = imutils.resize(frame, width=dst_width)

        if cnt % skip == 0:
            pd.update_trackers(img=resize)
            predict_labels(img=resize, persons=pd.get_trackers())
        else:
            pd.upgrade_trackers(img=resize, chk_num_thresh=3)

        result = show_rects(show_img=show_img, img=resize, trackers=pd.get_trackers())

        # show the output frame
        cv2.imshow("result", result)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()
        saver.write(result)

    # stop the timer and display FPS information
    fps.stop()
    print("elapsed time: {:.2f}".format(fps.elapsed()))
    print("approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    saver.release()
    cap.release()


def func_sound_play():
    global g_alarm_counter
    if not pygame.mixer.music.get_busy() and g_alarm_counter == DURATION:
        print "alarm beef sound"
        pygame.mixer.music.play()


if __name__ == '__main__':
    if a.input is None:
        folder = "../data/video/helmet/"
        fns = ["Armed Robbery at I Care Pharmacy May 23 2011.mp4",
               "Bloody robbery caught by a cctv.mp4",
               "cctv_robbery.mp4",
               "Helmet thief stealing in Semenyih pharmacy recorded by GASS HD CCTV.mp4"]
        video = folder + fns[2]
    else:
        video = a.input

        # "python script/endpoint.py --video data/video/helmet/cctv_robbery.mp4"
    print(video)
    if (len(video) == 1 and int(video) == 0) or os.path.exists(video):
        if len(video) == 1:
            video = int(video)
        func_run(video=video)
    else:
        print "no exist file"
