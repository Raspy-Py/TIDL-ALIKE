import os
import numpy as np
from fire import Fire
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import tqdm

# PC only
import sys
sys.path.append('/home/workdir/ALIKE')
sys.path.append('/home/workdir/components')

from dkd import DKD
from feature_extractor import FeatureExtractor, FeatureExtractorORT
from matcher import SimpleTracker, NotSoSimpleTracker
from utils import ImageLoader


VIDEO_PATH = "/home/workdir/assets/data/video/original.mp4"
WRITE_PATH = "/home/workdir/assets/data/video/processed.mp4"
MODELS_PATH = "/home/workdir/ALIKE/models"
ORT_MODELS_PATH = "/home/workdir/assets/models"

def preprocess_image_ort(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)

def resize_image(image, model_w, model_h):
    H_, W_ = image.shape[:2]
    if H_ > W_:
        new_h = model_h
        new_w = int(W_ * new_h / H_)
    else:
        new_w = model_w
        new_h = int(H_ * new_w / W_)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def undistort_image_fisheye(image, K, D, P):
    height, width = image.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, 
        D, 
        np.eye(3), 
        P, 
        (width, height), 
        cv2.CV_32FC1
    )
    
    undistorted_img = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    
    return undistorted_img


def get_camera_intrinsics(P_scale=0.45):
    D = np.array([
        -0.037558942767288876,
        -0.003653583628271442,
        -0.0007515566565807187,
        0.0004978366553628334
    ])

    K = np.array([
        [596.6634447325672, 0, 931.8577858158648],
        [0, 597.2993744845463, 638.494948230778],
        [0, 0, 1]
    ])

    P = K.copy()
    P[0, 0] *= P_scale
    P[1, 1] *= P_scale

    return K, D, P


if __name__ == "__main__":
    # check if cuda is available
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")


    K, D, P = get_camera_intrinsics(0.45)

    model_path = os.path.join(MODELS_PATH, "small_192_256.onnx")
    feature_extractor = FeatureExtractorORT(model_path, use_cuda=use_cuda)
    inputs = feature_extractor.get_input_dict()

    print(inputs)

