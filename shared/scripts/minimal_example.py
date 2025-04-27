"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""
import sys
sys.path.append('/home/workdir/components')
sys.path.append('/home/workdir/XFEAT')

import cv2
import numpy as np
import os
import torch
import tqdm

from modules.xfeat import XFeat


DEVICE = "cpu"
WEIGHTS_PATH = "/home/workdir/XFEAT/weights/xfeat.pt"
DEFAULT_IMAGE_PATH = "/home/workdir/assets/images/input_image.png"
REPORTS_FOLDER = "/home/workdir/assets/reports"
TUM_FOLDER ="/home/workdir/assets/data/tum"

def preprocess_image(frame):
    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image.astype(np.float32)) / 255

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat = XFeat(WEIGHTS_PATH)

# XFEAT* matching
image1 = cv2.imread(os.path.join(TUM_FOLDER, "1.png"))
image2 = cv2.imread(os.path.join(TUM_FOLDER, "10.png"))

H, W, _ = image1.shape

input_tensor1 = preprocess_image(image1)
input_tensor2 = preprocess_image(image2)

pts1, pts2 = xfeat.match_xfeat_star(input_tensor1, input_tensor2)

result_image = np.concatenate((image1, image2), axis=1)
for _pt1, _pt2 in zip(pts1, pts2):
	#cv2.line(result_image, (int(_pt1[0]), int(_pt1[1])), (int(_pt2[0]) + W, int(_pt2[1])), (0,255,0), 1)
	cv2.circle(result_image, (int(_pt1[0]), int(_pt1[1])), 2, (0, 0, 255), -1)
	cv2.circle(result_image, (int(_pt2[0]) + W, int(_pt2[1])), 2, (0, 0, 255), -1)

cv2.imwrite(os.path.join(REPORTS_FOLDER, "xfeat_result.png"), result_image)
