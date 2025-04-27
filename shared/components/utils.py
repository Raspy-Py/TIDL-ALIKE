import numpy as np
import stat
import cv2
import os

def save_tensor_to_file(tensor, filename):
    with open(filename, 'wb') as f:
        # Save shape information
        shape = np.array(tensor.shape, dtype=np.int64)
        np.array([len(shape)], dtype=np.int64).tofile(f)
        shape.tofile(f)
        print(f"saving data to {filename}")
        # Save data 
        tensor.astype(np.float32).tofile(f)

def load_tensor_from_file(filename):
    with open(filename, 'rb') as f:
        # Read shape information
        shape_len = np.fromfile(f, dtype=np.int64, count=1)[0]
        shape = np.fromfile(f, dtype=np.int64, count=shape_len)

        # Read data
        tensor = np.fromfile(f, dtype=np.float32).reshape(shape)

    return tensor


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
        return self.cap.read()
