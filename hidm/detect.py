from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import sys
import os
import math


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
TARGET_OBJs = [15]  # person   14, 7, 6
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
TRACK_TYPES = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']


class Det:
    def __init__(self, prototxt, model, confidence=0.5):
        """
        :param prototxt: path to Caffe 'deploy' prototxt file
        :param model: path to Caffe pre-trained model
        :param confidence: minimum probability to filter weak detections  default = 0.2
        """
        if not os.path.exists(prototxt) or not os.path.exists(model):
            sys.stderr.write("can not load pre traind models")
            return

        prototxt = prototxt  # help="path to Caffe 'deploy' prototxt file"
        model = model
        self.tracker_type = TRACK_TYPES[1]
        self.confidence = confidence
        self.thresh = 0.3
        self.margin = 10

        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        self.trackers = []

    def __detect_rects(self, frame):
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        # (note: normalization is done via the authors of the MobileNet SSD
        # implementation)

        # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        rects = []
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
                rects.append({'label': idx,
                              'score': confidence * 100,
                              'box': (x1, y1, x2 - x1, y2 - y1)})
        return rects

    def __show_rects(self, show_img, frame):

        h_r = float(show_img.shape[0]) / frame.shape[0]
        w_r = float(show_img.shape[1]) / frame.shape[1]

        # draw the prediction on the frame
        for t in self.trackers:
            label = "{}: {:.2f}%".format(CLASSES[t['label']], t['score'])
            (x, y, w, h) = (t['box'] * np.array([w_r, h_r, w_r, h_r])).astype(np.int)
            if not self.det_flag:
                color = COLORS[t['label']]
            else:
                color = (0, 0, 255)
            cv2.rectangle(show_img, (x, y), (x+w, y+h), color, 2)
            t_y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(show_img, label, (x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return show_img

    def __same_rect(self, rect1, rect2):
        (x1, y1, w1, h1) = rect1
        (x2, y2, w2, h2) = rect2

        # overlay_rect
        _x = max(x1, x2)
        _y = max(y1, y2)
        _w = min(x1 + w1, x2 + w2) - _x
        _h = min(y1 + h1, y2 + h2) - _y

        min_sz = min(w1 * h1, w2 * h2)
        overlay_sz = _w * _h

        if _w <= 0 or _h <= 0 or overlay_sz < min_sz * self.thresh:
            return False
        else:
            return True

    def __update_trackers(self, frame, rects):
        to_adds = []
        for t in self.trackers:
            t['updated'] -= 1

        for rect in rects:
            _flag = False
            for t in self.trackers:
                if self.__same_rect(t['box'], rect['box']):
                    avg_box = rect['box']
                    # avg_box = tuple((np.array(rect['box']) / 2.0 + np.array(t['box']) / 2.0).astype(np.int))
                    t['tracker'].init(frame, avg_box)
                    t['label'] = rect['label']
                    t['score'] = rect['score']
                    t['box'] = avg_box
                    t['updated'] = 0

                    _flag = True
                    break
            if not _flag:
                if self.tracker_type == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                elif self.tracker_type == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                elif self.tracker_type == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                elif self.tracker_type == 'TLD':
                    tracker = cv2.TrackerTLD_create()
                elif self.tracker_type == 'MEDIANFLOW':
                    tracker = cv2.TrackerMedianFlow_create()
                elif self.tracker_type == 'GOTURN':
                    tracker = cv2.TrackerGOTURN_create()

                try:
                    tracker.init(frame, rect['box'])
                    to_adds.append({
                        'tracker': tracker,
                        'box': rect['box'],
                        'label': rect['label'],
                        'score': rect['score'],
                        'updated': 0,
                    })
                except Exception:
                    pass

        for add in to_adds:
            self.trackers.append(add)
        i = 0
        while i < len(self.trackers):
            t = self.trackers[i]
            if t['updated'] < - 10:
                self.trackers.remove(t)
                continue
            else:
                i += 1

    def __track_rects(self, frame):
        height, width = frame.shape[:2]
        to_dels = []
        for t in self.trackers:
            _, rect = t['tracker'].update(frame)
            t['box'] = rect
            (x, y, w, h) = rect
            if not (self.margin < x < width - w - self.margin) or not (self.margin < y < height - h - self.margin):
                to_dels.append(t)

        for d in to_dels:
            self.trackers.remove(d)

    def run(self, video, width=None, skip=5):
        # initialize the video stream, allow the cammera sensor to warmup,
        # and initialize the FPS counter
        print("[INFO] starting video stream...")

        cap = cv2.VideoCapture(video)
        time.sleep(2.0)
        fps = FPS().start()
        if width is None:
            dst_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            dst_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            dst_width = width
            dst_height = int(cap.shape[0] * width / cap.shape[1])

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

            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            if cnt % skip == 0:
                rects = self.__detect_rects(frame=resize)
                self.__update_trackers(frame=resize, rects=rects)
            else:
                self.__track_rects(frame=resize)

            result = self.__show_rects(show_img=show_img, frame=resize)

            # show the output frame
            cv2.imshow("result", result)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            elif key == ord("s"):
                self.det_flag = not self.det_flag
            # update the FPS counter
            fps.update()
            saver.write(result)

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        saver.release()
        cap.release()
