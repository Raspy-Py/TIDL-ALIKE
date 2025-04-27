import sys
sys.path.append('/home/workdir/components')

import os
import cv2
import torch
import argparse
import numpy as np
import onnx 
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F

from xfeat.extractor import XFeatModel


DEVICE = "cpu"
WEIGHTS_PATH = "/home/workdir/XFEAT/weights/xfeat.pt"
DEFAULT_IMAGE_PATH = "/home/workdir/assets/images/input_image.png"
REPORTS_FOLDER = "/home/workdir/assets/reports"
TUM_FOLDER ="/home/workdir/assets/data/tum"

# Create points indices grid
def create_xy(h, w, dev):
    y, x = torch.meshgrid(torch.arange(h, device = dev), 
                            torch.arange(w, device = dev))
    xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
    return xy

# Get input tensor from OpenCV image
def preprocess_image(frame):
    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image.astype(np.float32)) / 255

# Extract keypoints from maps 
def extract_dense(features, heatmap, top_k = 4096):     
    B, C, _H1, _W1 = features.shape
        
    xy1 = (create_xy(_H1, _W1, DEVICE) * 8).expand(B,-1,-1)

    features = features.permute(0,2,3,1).reshape(B, -1, C)
    heatmap = heatmap.permute(0,2,3,1).reshape(B, -1)

    _, top_k = torch.topk(heatmap, k = min(len(heatmap[0]), top_k), dim=-1)

    feats = torch.gather(features, 1, top_k[...,None].expand(-1, -1, 64))
    mkpts = torch.gather(xy1, 1, top_k[...,None].expand(-1, -1, 2)).to(torch.float32)
    sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

    return {'keypoints': mkpts,
				'descriptors': feats,
				'scales': sc }

def perform_matching(feats1, feats2, min_cossim = 0.82):
    cossim = feats1 @ feats2.t()
    cossim_t = feats2 @ feats1.t()
    
    _, match12 = cossim.max(dim=1)
    _, match21 = cossim_t.max(dim=1)

    idx0 = torch.arange(len(match12), device=match12.device)
    mutual = match21[match12] == idx0

    if min_cossim > 0:
        cossim, _ = cossim.max(dim=1)
        good = cossim > min_cossim
        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]
    else:
        idx0 = idx0[mutual]
        idx1 = match12[mutual]

    return idx0, idx1

def subpix_softmax2d(heatmaps, temp = 3):
    N, H, W = heatmaps.shape
    heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
    x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ))
    x = x - (W//2)
    y = y - (H//2)

    coords_x = (x[None, ...] * heatmaps)
    coords_y = (y[None, ...] * heatmaps)
    coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
    coords = coords.sum(1)

    return coords

def refine_matches(d0, d1, idx0, idx1, fine_matcher, fine_conf = 0.25):
    # FIXME: remove batch dimension
    feats1 = d0['descriptors'][0][idx0]
    feats2 = d1['descriptors'][0][idx1]
    mkpts_0 = d0['keypoints'][0][idx0]
    mkpts_1 = d1['keypoints'][0][idx1]
    sc0 = d0['scales'][0][idx0]

    #Compute fine offsets
    offsets = fine_matcher(torch.cat([feats1, feats2],dim=-1))
    conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
    offsets = subpix_softmax2d(offsets.view(-1,8,8))

    mkpts_0 += offsets * (sc0[:,None]) #*0.9 #* (sc0[:,None])

    mask_good = conf > fine_conf
    mkpts_0 = mkpts_0[mask_good]
    mkpts_1 = mkpts_1[mask_good]

    return torch.cat([mkpts_0, mkpts_1], dim=-1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_path = DEFAULT_IMAGE_PATH
    else:
        image_path = sys.argv[1]

    print(f"Loading model from: {WEIGHTS_PATH}")
    model = XFeatModel()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))

    # print(f"Loading image from: {image_path}")
    # image = cv2.imread(image_path)
    # input_tensor = preprocess_image(image)

    # print("Running XFEAT...")
    # features, keypoints, heatmap = model(input_tensor)

    # print(f"features: {type(features)} {features.shape}")
    # print(f"keypoints: {type(keypoints)} {keypoints.shape}")
    # print(f"heatmap: {type(heatmap)} {heatmap.shape}")

    # Save heatmap
    # heatmap = heatmap.squeeze(0).squeeze(0)
    # heatmap_numpy = heatmap.detach().numpy() * 255.0
    # heatmap_numpy = heatmap_numpy.astype(np.uint8)
    # heatmap_path = os.path.join(REPORTS_FOLDER, "xfeat_heatmap.png")
    # print(f"Saving heatmap to {heatmap_path}")
    # cv2.imwrite(heatmap_path, heatmap_numpy)

    # Extract dense postprocessing
    # feats, mkpts = extract_dense(features, heatmap, top_k=1024)

    # print(f"Postprocessed features: {type(feats)} {feats.shape}")
    # print(f"Postprocessed mkpts: {type(mkpts)} {mkpts.shape}")

    # keypoints = mkpts.squeeze(0).detach().numpy()

    # original_image = cv2.imread(image_path)
    # for point in keypoints:
    #     cv2.circle(original_image, (point[0], point[1]), 1, (0, 0, 255), -1)
    # cv2.imwrite(os.path.join(REPORTS_FOLDER, "xfeat_result.png"), original_image)


    # XFEAT* matching
    image1 = cv2.imread(os.path.join(TUM_FOLDER, "1.png"))
    image2 = cv2.imread(os.path.join(TUM_FOLDER, "10.png"))

    H, W, _ = image1.shape

    input_tensor1 = preprocess_image(image1)
    input_tensor2 = preprocess_image(image2)

    features1, _, heatmap1 = model(input_tensor1)
    features2, _, heatmap2 = model(input_tensor2)

    heatmap = heatmap1.squeeze(0).squeeze(0)
    heatmap_numpy = heatmap.detach().numpy() * 255.0
    heatmap_numpy = heatmap_numpy.astype(np.uint8)
    heatmap_path = os.path.join(REPORTS_FOLDER, "xfeat_heatmap.png")
    print(f"Saving heatmap to {heatmap_path}")
    cv2.imwrite(heatmap_path, heatmap_numpy)


    out1 = extract_dense(features1, heatmap1, top_k = 1000)
    out2 = extract_dense(features2, heatmap2, top_k = 1000)

    desc_shape1 = out1["descriptors"].shape
    idx1, idx2 = perform_matching(out1["descriptors"].squeeze(0), out2["descriptors"].squeeze(0))

    matches = refine_matches(out1, out2, idx1, idx2, model.fine_matcher)
    pts1, pts2 = matches[:, :2].detach().numpy(), matches[:, 2:].detach().numpy()

    result_image = np.concatenate((image1, image2), axis=1)
    for _pt1, _pt2 in zip(pts1, pts2):
        cv2.line(result_image, (int(_pt1[0]), int(_pt1[1])), (int(_pt2[0]) + W, int(_pt2[1])), (0,255,0), 1)
        cv2.circle(result_image, (int(_pt1[0]), int(_pt1[1])), 2, (0, 0, 255), -1)
        cv2.circle(result_image, (int(_pt2[0]) + W, int(_pt2[1])), 2, (0, 0, 255), -1)

    print(f"Writing result into: {os.path.join(REPORTS_FOLDER, 'xfeat_result.png')}")
    cv2.imwrite(os.path.join(REPORTS_FOLDER, "xfeat_result.png"), result_image)


