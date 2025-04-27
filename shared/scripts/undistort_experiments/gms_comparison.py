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
from feature_extractor import FeatureExtractor
from matcher import SimpleTracker, NotSoSimpleTracker
from utils import ImageLoader
from gms_matcher import *

VIDEO_PATH = "/home/workdir/assets/data/video/original.mp4"
WRITE_PATH = "/home/workdir/assets/data/video/gms_processed.mp4"
MODELS_PATH = "/home/workdir/ALIKE/models"
CONFIGS = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 
                'model_path': os.path.join(MODELS_PATH, 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True,
                'model_path': os.path.join(MODELS_PATH, 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True,
                'model_path': os.path.join(MODELS_PATH, 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False,
                'model_path': os.path.join(MODELS_PATH, 'alike-l.pth')},
}

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    #image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    image = image.astype(np.float32)
    return torch.from_numpy(image)

def preprocess_mask(image):
    image = image.copy() / 255.0
    #image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return torch.from_numpy(image)


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

def pad_tensor(image, model_w, model_h):
    b, c, h, w = image.shape
    h_, w_ = model_h, model_w

    if h_ != h:
        h_padding = torch.zeros(b, c, h_ - h, w)
        image = torch.cat([image, h_padding], dim=2)
    if w_ != w:
        w_padding = torch.zeros(b, c, h_, w_ - w)
        image = torch.cat([image, w_padding], dim=3)

    return image

def unpad_output(descriptor_map, scores_map, input_w, input_h):
    b, c, h_, w_ = scores_map.shape
    h, w = input_h, input_w

    if h_ != h or w_ != w:
        descriptor_map = descriptor_map[:, :, :h, :w]
        scores_map = scores_map[:, :, :h, :w]

    return descriptor_map, scores_map


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

if __name__ == "__main__":

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

    scale = 0.45
    P = K.copy()
    P[0, 0] *= scale
    P[1, 1] *= scale


    H, W = 160, 256
    mask = cv2.imread("/home/workdir/scripts/undistort_experiments/mask.png", cv2.IMREAD_GRAYSCALE)
    mask = undistort_image_fisheye(mask, K, D, P)
    mask = resize_image(mask, W, H)
    mask_tensor = preprocess_mask(mask)

    feature_extractor = FeatureExtractor(**CONFIGS["alike-s"], device="cpu", manual_norm=True)
    dkd = DKD(radius=2, top_k=400, detector="maxpool")
    # tracker = SimpleTracker()
    #tracker = NotSoSimpleTracker(top_k=400)

    # cap = cv2.VideoCapture(VIDEO_PATH)
    image_loader = ImageLoader(VIDEO_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for mp4
    fps = image_loader.get_fps()
    input_w, input_h = image_loader.get_res()
    video_writer = cv2.VideoWriter(WRITE_PATH, fourcc, fps, (256 * 2, 144))

    # if not cap.isOpened():
    #     print(f"Failed to open video file: {VIDEO_PATH}")
    #     exit(-1)
    
    feature_extractor.eval()
    counter = 0
    prev_frame = None
    prev_kpts = None
    prev_desc = None

    orb = cv2.ORB_create(400)
    orb.setFastThreshold(0)
    if cv2.__version__.startswith('3'):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    gms = GmsMatcher(orb, matcher)

    for i in tqdm(range(len(image_loader))):
        ret, frame = image_loader.read()

        if not ret:
            break

        if i < 500:
            continue
        
        # Resize and pad image the way it's done in OpenVINS
        frame = undistort_image_fisheye(frame, K, D, P)
        frame = resize_image(frame, W, H)
        frame_h, frame_w = frame.shape[:2]

        input_tensor = preprocess_image(frame)
        input_tensor = pad_tensor(input_tensor, W, H)

        # Run ALike 
        descriptor_map, scores_map = feature_extractor(input_tensor)
        descriptor_map, scores_map = unpad_output(descriptor_map, scores_map, frame_w, frame_h)
        scores_map = scores_map * mask_tensor

        xy_keypoints, descriptors, kptscores = dkd(scores_map, descriptor_map)
        descriptors = descriptors.detach().numpy()

        # convert keypoints tensor to list of KeyPoint objects
        # keypoints = [cv2.KeyPoint(x=k[0], y=k[1], _size=1) for k in keypoints]
        
        keypoints = []
        for k in xy_keypoints:
            kpt = cv2.KeyPoint()
            kpt.pt = (k[0], k[1])
            kpt.size = 1
            keypoints.append(kpt)

        
        if prev_frame is None:
            prev_frame = frame
            prev_kpts = keypoints
            prev_desc = descriptors
            continue


        orb = cv2.ORB_create(400)
        orb.setFastThreshold(0)
        if cv2.__version__.startswith('3'):
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        gms = GmsMatcher(orb, matcher)

        matches = gms.match_gms(prev_kpts, prev_desc, descriptors, keypoints, frame.shape)
        print(f"Number of matches: {len(matches)}")
        # gms.draw_matches(img1, img2, DrawingType.ONLY_LINES)
        #out = gms.draw_matches(prev_frame, frame, DrawingType.COLOR_CODED_POINTS_XpY)

        #video_writer.write(out)

        prev_frame = frame
        prev_kpts = keypoints
        prev_desc = descriptors
    
    video_writer.release()
    #tracker.print_stats()

