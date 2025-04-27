import sys
sys.path.append('/home/workdir/XFEAT')
sys.path.append('/home/workdir/components')

import os
import cv2
import torch
import argparse
import numpy as np
import onnx 

from onnx.utils import extract_model
# from modules.model import XFeatModel
from xfeat.extractor import XFeatModel



RANDOM_IMAGE_SHAPE = (3, 480, 640)
CALIBRATION_IMAGE_PATH = "/home/workdir/ALIKE/assets/tum/1311868169.163498.png"
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
EXPORT_FOLDER = "/home/workdir/assets/models"
WEIGHTS_FOLDER = "/home/workdir/XFEAT/weights"


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="image", help="Input node name.")
    parser.add_argument("--file", type=str, default="model", help="Output file name.")
    parser.add_argument("--cut", type=str, default="", help="Optional cutting point.")
    parser.add_argument("--shape", type=str, default="640x480", help="Shape of the calibration image.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX Operation set version.")
    parser.add_argument("--export-folder", type=str, default=EXPORT_FOLDER, help="Directory to save the the exported model.")

    args = parser.parse_args()

    print(f"Generating calibration image.")
    image_shape = [int(dim) for dim in args.shape.split('x')]
    assert len(image_shape) == 2, f"Error: shape dimension should be 2, but got {len(image_shape)}"
    W, H = image_shape
    image = torch.randn(1, 1, H, W)


    weigts_path = os.path.join(WEIGHTS_FOLDER, "xfeat.pt")
    print(f"Loading model {weigts_path}...")
    xfeat = XFeatModel()
    xfeat.load_state_dict(torch.load(weigts_path, map_location=DEVICE))

    print(f"Exporting model to ONNX.")
    model_file = args.file + ".onnx"
    temp_name = os.path.join(EXPORT_FOLDER, model_file if args.cut == "" else "temp_" + model_file)
    torch.onnx.export(
        xfeat, image, temp_name,
        input_names=[args.input], 
        output_names=['feats', 'keypoints', 'heatmap'],
        opset_version=args.opset,

        # ...
        export_params=True,
        do_constant_folding=False,
    )
    output_path = os.path.join(EXPORT_FOLDER, model_file)
    
    if args.cut != "":
        print(f"Extracting model subgraph: [{args.input}] -> [{args.cut}].")
        extract_model(
            input_path=temp_name, 
            output_path=output_path, 
            input_names=args.input.split(','), 
            output_names=args.cut.split(',')
        )

        os.remove(temp_name)
    print(f"Peforming shape inference on: [{output_path}].")
    onnx.shape_inference.infer_shapes_path(output_path, output_path)

    print(f"Successfully exported to: [{output_path}]")
