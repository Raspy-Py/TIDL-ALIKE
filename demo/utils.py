import os
import cv2
import json
import stat
import numpy as np
from copy import copy

#========================================================================================
#
# Helper classes for reading video from an arbitrary source and streaming it
#
#========================================================================================

class Streamer(object):
    def __init__(self, host='10.42.0.1', w=256, h=256, fps=30, bitrate=4000, port=5000):
        gst_out = (
            f'appsrc ! videoconvert ! videoscale ! x264enc tune=zerolatency bitrate={bitrate} speed-preset=superfast ! '
            f'rtph264pay config-interval=1 pt=96 ! udpsink host={host} port={port} auto-multicast=true'
        )
        self.w, self.h = w, h
        self.fps = fps
        self.out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (w, h), True)

    def __del__(self):
        self.out.release()

    def write(self, frame):
        if frame is not None:
            self.out.write(frame)
        else:
            print("Received a None frame.")

class ImageLoader(object):
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Can't open video file: {self.path}")

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        file_stat = os.stat(self.path)
        if stat.S_ISREG(file_stat.st_mode):
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_frames = 9999999

    def __del__(self):
        if self.cap:
            self.cap.release()

    def __len__(self):
        return self.total_frames
    
    def get_fps(self):
        return self.fps
    
    def get_res(self):
        return self.w, self.h

    def read(self):
        ret, frame = self.cap.read()
        if ret < 0:
            print("Error: Could not read frame.")
        return frame

