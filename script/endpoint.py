from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import sys
import os

from person_det.detect import PersonDetect
pd = PersonDetect()


COLOR_TRACK = (255, 255, 0)
COLOR_TARGET = (0, 0, 255)


def show_rects(show_img, img, trackers):
    h_r = float(show_img.shape[0]) / img.shape[0]
    w_r = float(show_img.shape[1]) / img.shape[1]

    # draw the prediction on the frame
    for p in trackers:
        label = "{}: {:.2f}%".format(CLASSES[p['label']], p['score'])
        (x, y, w, h) = (p['box'] * np.array([w_r, h_r, w_r, h_r])).astype(np.int)
        if not self.det_flag:
            color = COLOR_TRACK
        else:
            color = COLOR_TARGET
        cv2.rectangle(show_img, (x, y), (x + w, y + h), color, 2)
        t_y = y - 15 if y - 15 > 15 else y + 15
        cv2.putText(show_img, label, (x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return show_img


def run(self, video, zoom_ratio=0.5, skip=5):
    if self.bSave:
        zoom_ratio = 1.0
        skip = 5

    print("starting video stream...")
    cap = cv2.VideoCapture(video)
    time.sleep(2.0)
    fps = FPS().start()
    dst_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * zoom_ratio)
    dst_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * zoom_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    saver = cv2.VideoWriter("result.avi", fourcc, 30.0, (dst_width, dst_height))

    cnt = -1
    self.det_flag = False
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
        else:
            pd.upgrade_trackers(img=resize, chk_num_thresh=3)

        result = show_rects(show_img=show_img, img=resize, trackers=pd.get_trackers())

        # show the output frame
        cv2.imshow("result", result)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        elif key == ord("s"):
            self.det_flag = True
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


if __name__ == '__main__':
    dir = "../data/video/helmet/"
    fns = ["Armed Robbery at I Care Pharmacy May 23 2011.mp4",
           "Bloody robbery caught by a cctv.mp4",
           "cctv robbery.mp4",
           "Helmet thief stealing in Semenyih pharmacy recorded by GASS HD CCTV.mp4"]

    det.run(video=dir+fns[3])