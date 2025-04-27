import os
import cv2

class CameraReader(object):
    def __init__(self, device="video2", w=256, h=256, fps=30):
        cam_h, cam_w = 480, 640
        self.device_path = os.path.join("/dev", device)
        self.cap = cv2.VideoCapture(self.device_path)

        self.h, self.w = h, w
        self.offset_h = (cam_h - h) // 2
        self.offset_w = (cam_w - w) // 2
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise IOError(f"Can't open: {self.device_path}")
        
    def get_frame(self):
        ret, img = self.cap.read()
        return img[
            self.offset_h:self.offset_h + self.h, 
            self.offset_w:self.offset_w + self.w, 
            :]


if __name__ == "__main__":
    camera_reader = CameraReader()
    img = camera_reader.get_frame()
    print(img.shape)
