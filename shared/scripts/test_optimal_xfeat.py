import sys
sys.path.append('/home/workdir/components')
sys.path.append('/home/workdir/XFEAT')

import os
import cv2
import onnx 
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from utils import ImageLoader
from modules.xfeat import XFeat


DEFAULT_IMAGE_PATH = "/home/workdir/assets/images/input_image.png"
REPORTS_FOLDER = "/home/workdir/assets/reports"
VIDEO_PATH = "/home/workdir/assets/videos/field_short.mp4"

H, W = 288, 512

# Get input tensor from OpenCV image
def preprocess_image(frame, w, h):
    image = frame.copy()
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image.astype(np.float32)) / 255


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU
    xfeat = XFeat()
    image_loader = ImageLoader(VIDEO_PATH)
    W_input, H_input = image_loader.get_res()   
    inv_scale = W_input / W 
    prev_frame = None

    output_file = os.path.join(REPORTS_FOLDER, 'field_short_start.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, image_loader.get_fps(), (W_input, H_input))

    for _ in tqdm(range(len(image_loader))):
        ret, input = image_loader.read()
        if not ret:
            break

        frame = preprocess_image(input, 512, 288)
        if prev_frame is None:
            prev_frame = frame
            continue
        
        #mkpts_0, mkpts_1 = xfeat.match_xfeat(prev_frame, frame)
        mkpts_0, mkpts_1 = xfeat.match_xfeat_star(prev_frame, frame)
        prev_frame = frame

        output = input
        for pt1, pt2 in zip(mkpts_0, mkpts_1):
            pt1 = (int(pt1[0] * inv_scale), int(pt1[1] * inv_scale))
            pt2 = (int(pt2[0] * inv_scale), int(pt2[1] * inv_scale))
            cv2.circle(output, pt2, 5, (0, 0, 255), -1)
            cv2.line(output, pt1, pt2, (255, 0, 0), 2)

        out.write(output)
        



