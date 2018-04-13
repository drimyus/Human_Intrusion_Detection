import os
import sys
import cv2
import tkinter
from kivy.app import App
from tkinter import filedialog
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen

from kivy.lang import Builder
from script.hidm import HIDM


Builder.load_file(os.path.join(os.path.dirname(__file__), 'kv', 'Start.kv'))


class StartScreen(Screen):
    app = None

    cap = None
    fps = 0
    frame_pos = 0
    event_take_video = None

    hidm = HIDM()

    def build(self, **kwargs):
        super(StartScreen, self).__init__(**kwargs)
        self.app = App.get_running_app()

    def on_btn_usb(self, *args):
        self.ids.txt_path.text = "usb camera" + str(0)
        self.cap = cv2.VideoCapture(0)

    def on_btn_video(self, *args):
        tk = tkinter.Tk()
        tk.withdraw()
        select_file = (tkinter.filedialog.askopenfile(initialdir='.', title='select a video file'))
        if select_file is None:
            return
        str_path = select_file.name
        if os.path.splitext(str_path)[1] not in [".mp4", ".avi"]:
            return
        tk.update()
        tk.destroy()

        self.ids.txt_path.text = str_path
        self.cap = cv2.VideoCapture(str_path)

    def on_btn_start(self, *args):
        if self.cap is not None:
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.hidm.init_HIDM(frame_size=(width, height), zoom_ratio=0.5, skip=5)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps == 0:
                self.fps = 30.0

            if self.event_take_video is None:
                self.event_take_video = Clock.schedule_interval(self.live_video, 1.0 / self.fps)
            elif not self.event_take_video.is_triggered:
                self.event_take_video()

    def on_btn_stop(self):
        if self.event_take_video is not None and self.event_take_video.is_triggered:
            self.event_take_video.cancel()

    def on_btn_exit(self):
        self.app.stop()

    def __frame2buf(self, frame, size):
        frame = cv2.resize(frame, (int(size[0]), int(size[1])))
        frame = cv2.flip(frame, 0)
        buf = frame.tostring()
        texture = Texture.create(size=size)
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def __show_frame(self, frame):
        if frame is not None:
            texture = self.__frame2buf(frame=frame, size=self.ids.shower.size)
            self.ids.shower.texture = texture

    def __release(self):
        self.hidm.release()
        self.cap.release()

    @mainthread
    def live_video(self, *args):
        try:
            ret, frame = self.cap.read()
            if not ret:
                if self.event_take_video is not None and self.event_take_video.is_triggered:
                    self.event_take_video.cancel()
                self.__release()

            result_frame = self.hidm.proc(frame=frame, pos=self.frame_pos)
            self.__show_frame(frame=result_frame)
            self.frame_pos += 1
        except Exception as e:
            self.__release()
            print(e)
