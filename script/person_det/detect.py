from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import sys
import os
import math
import dlib


# initialize the list of class labels MobileNet SSD was trained to
# person_det, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
TARGET_OBJs = [15]  # person


class PersonDetect:
    def __init__(self, bSave=False, start_id=0):
        root = os.path.dirname(__file__)
        prototxt = os.path.join(root, "model/MobileNetSSD_deploy.prototxt.txt")
        model = os.path.join(root, "model/MobileNetSSD_deploy.caffemodel")

        if not os.path.exists(prototxt) or not os.path.exists(model):
            sys.stderr.write("can not load pre traind models")
            return

        # load our serialized model from disk
        print("loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        self.confidence = 0.5
        self.thresh_same_rect = 0.5
        self.margin = 10

        self.person_trackers = []

        self.bSave = bSave
        self.save_dir = "../../data/train_data"
        self.uid = start_id

    def __dist_pt2pt(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def __detect_persons(self, img):
        # grab the frame dimensions and convert it to a blob
        (h, w) = img.shape[:2]
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        # (note: normalization is done via the authors of the MobileNet SSD
        # implementation)

        # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        blob = cv2.dnn.blobFromImage(img, 0.007843, (w, h), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        persons = []
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence and idx in TARGET_OBJs:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                persons.append({'label': '',  # idx,
                                'score': confidence * 100,
                                'box': (x1, y1, x2 - x1, y2 - y1)})
        return persons

    def get_trackers(self):
        return self.person_trackers

    def __same_rect(self, before, after):
        (x1, y1, w1, h1) = before
        (x2, y2, w2, h2) = after

        # overlay_rect
        _x = max(x1, x2)
        _y = max(y1, y2)
        _w = min(x1 + w1, x2 + w2) - _x
        _h = min(y1 + h1, y2 + h2) - _y

        min_sz = min(w1 * h1, w2 * h2)
        overlay_sz = _w * _h

        if _w <= 0 or _h <= 0 or overlay_sz < min_sz * self.thresh_same_rect:
            return False
        else:
            return True

    def __update(self, img, rects):
        for p in self.person_trackers:
            p['chk_num'] -= 1

        # upgrade trackers
        j = 0
        while j < len(rects):
            matched = None
            min_dis = None

            r = rects[j]

            i = 0
            while i < len(self.person_trackers):
                p = self.person_trackers[i]
                (x1, y1, w1, h1) = p['box']
                (x2, y2, w2, h2) = r['box']
                cur_dis = self.__dist_pt2pt(pt1=(x1 + w1 // 2, y1 + h1 // 2), pt2=(x2 + w2 // 2, y2 + h2 // 2))

                if min_dis is None or min_dis > cur_dis:
                    min_dis = cur_dis
                    matched = p

                i += 1

            b_same = False
            if matched is not None:
                b_same = self.__same_rect(before=matched['box'], after=r['box'])

            if b_same:
                avg_box = tuple((np.array(matched['box']) * 0.5 + np.array(r['box']) * 0.5).astype(np.int))

                matched['box'] = avg_box
                (x, y, w, h) = avg_box
                matched['tracker'].start_track(img, dlib.rectangle(x, y, x + w, y + h))
                matched['label'] = r['label']
                matched['score'] = r['score']
                matched['chk_num'] = 0

            else:
                # create new tracker
                self.uid += 1
                (x, y, w, h) = r['box']
                tracker = dlib.correlation_tracker()
                tracker.start_track(img, dlib.rectangle(x, y, x + w, y + h))
                a = {
                    'uid': self.uid,
                    'tracker': tracker,
                    'box': r['box'],
                    'label': r['label'],
                    'score': r['score'],
                    'chk_num': 0
                }
                self.person_trackers.append(a)
                if self.bSave:
                    self.__save_rect(img=img, rect=a['box'])

            j += 1

    def update_trackers(self, img):
        persons = self.__detect_persons(img=img)
        self.__update(img=img, rects=persons)

    def upgrade_trackers(self, img, chk_num_thresh):
        height, width = img.shape[:2]

        to_dels = []
        for p in self.person_trackers:
            _ = p['tracker'].update(img)
            if p['chk_num'] < -chk_num_thresh:
                to_dels.append(p)

            loc = p['tracker'].get_position()
            [x, y, w, h] = [int(loc.left()), int(loc.top()), int(loc.width()), int(loc.height())]
            p['box'] = [x, y, w, h]

            if not (self.margin < x < width - w - self.margin) or not (self.margin < y < height - h - self.margin):
                to_dels.append(p)

        for d in to_dels:
            self.person_trackers.remove(d)

    def train_data(self, video, zoom_ratio=1.0, skip=5):
        if self.bSave:
            zoom_ratio = 1.0
            skip = 5

        print("starting video stream...")
        cap = cv2.VideoCapture(video)
        time.sleep(2.0)
        fps = FPS().start()
        dst_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * zoom_ratio)
        dst_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * zoom_ratio)

        cnt = -1
        self.det_flag = False
        # loop over the frames from the video stream
        while True:
            cnt += 1
            suc, frame = cap.read()
            frame = cv2.resize(frame, (int(dst_width*2), int(dst_height*2)))
            if not suc:
                break
            show_img = frame.copy()
            resize = imutils.resize(frame, width=dst_width)

            if cnt % skip == 0:
                self.update_trackers(img=resize)
            else:
                self.upgrade_trackers(img=resize, chk_num_thresh=3)

            result = self.__show_rects(show_img=show_img, img=resize)

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

        # stop the timer and display FPS information
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.elapsed()))
        print("approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        cap.release()

    def __save_rect(self, img, rect):
        (x, y, w, h) = rect
        crop = img[y - self.margin:y + h + self.margin, x - self.margin:x + w + self.margin]
        path = os.path.join(self.save_dir, str(self.uid) + ".jpg")
        cv2.imwrite(path, crop)
        print(self.uid)

    def __show_rects(self, show_img, img):
        h_r = float(show_img.shape[0]) / img.shape[0]
        w_r = float(show_img.shape[1]) / img.shape[1]

        # draw the prediction on the frame
        for p in self.person_trackers:
            label = "person: {:.2f}%".format(p['score'])
            (x, y, w, h) = (p['box'] * np.array([w_r, h_r, w_r, h_r])).astype(np.int)
            if not self.det_flag:
                color = (255, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(show_img, (x, y), (x + w, y + h), color, 2)
            t_y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(show_img, label, (x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return show_img


if __name__ == '__main__':
    pd = PersonDetect(bSave=True, start_id=42)
    fn = [
        "/home/be/Documents/BE/HIDM/data/video/climbing/crop1/crop.mp4",
        "/home/be/Documents/BE/HIDM/data/video/climbing/crop1/crop2.mp4",
        "/home/be/Documents/BE/HIDM/data/video/climbing/Home Robbers Climb Over Spiked Fence Caught On CCTV.mp4"
        ]
    pd.train_data(video=fn[1])
