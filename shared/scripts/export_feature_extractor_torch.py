
import sys
sys.path.append('/home/workdir/ALIKE')
sys.path.append('/home/workdir/components')

import os
import cv2
import torch
import argparse
import numpy as np
import onnx 

from feature_extractor import FeatureExtractor, FeatureExtractorSingleOut
from onnx.utils import extract_model


RANDOM_IMAGE_SHAPE = (3, 480, 640)
CALIBRATION_IMAGE_PATH = "/home/workdir/ALIKE/assets/tum/1311868169.163498.png"
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
EXPORT_FOLDER = "/home/workdir/assets/models"
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
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="ALIKE configuration preset.")
    parser.add_argument("--input", type=str, default="image", help="Input node name.")
    parser.add_argument("--file", type=str, default="model", help="Output file name.")
    parser.add_argument("--cut", type=str, default="", help="Optional cutting point.")
    parser.add_argument("--shape", type=str, default="640x480", help="Shape of the calibration image.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX Operation set version.")
    parser.add_argument("--export-folder", type=str, default=EXPORT_FOLDER, help="Directory to save the the exported model.")

    args = parser.parse_args()

    print(f"Loading calibration image.")
    image_shape = [int(dim) for dim in args.shape.split('x')]
    assert len(image_shape) == 2, f"Error: shape dimension should be 2, but got {len(image_shape)}"
    W, H = image_shape
    image = torch.randn(1, 3, H, W).to(DEVICE)
    feature_extractor = FeatureExtractorSingleOut(**CONFIGS[args.config], device=DEVICE, manual_norm=True)

    print(f"Exporting model to ONNX.")
    torch_model_file = args.file + ".pt"

    torch_output_path = os.path.join(EXPORT_FOLDER, torch_model_file)

    # Also save the torch model
    feature_extractor.eval()
    trace_model = torch.jit.trace(feature_extractor, torch.Tensor(1,3,H,W))
    trace_model.save(torch_output_path)
