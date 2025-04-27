import os
import onnxruntime
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
from matcher import SimpleTracker
from visualizers import plot_detected_points, create_gif
from utils import ImageLoader

def preprocess_image(image, widht, height):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (widht, height), interpolation=cv2.INTER_LINEAR)
    image = image.copy() / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)

def raw_image(image, widht, height):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (widht, height), interpolation=cv2.INTER_LINEAR)
    image = image.copy()
    return np.array(image, dtype=np.uint8)

class ALIKEPipeline:
    def __init__(self, model, shape="480x640", n_desc=400, detector="maxpool"):
        # Feature Extractor
        self.model_name = model
        self.onnx_path = os.path.join("/home/workdir/assets/models", self.model_name + ".onnx")
        self.reports_folder = os.path.join("/home/workdir/assets/reports", self.model_name)
        self.artifacts_folder = os.path.join("/home/workdir/assets/artifacts", self.model_name)

        dim_list = [int(dim) for dim in shape.split('x')]
        assert len(dim_list) == 2, f"Error: shape dimension should be 2, but got {len(dim_list)}"
        self.w, self.h = dim_list
        self.input_shape = (1, 3, self.w, self.h)

        # DKD
        self.sort = True
        self.dkd = DKD(radius=2, top_k=n_desc, scores_th=0.5, n_limit=5000, detector=detector)

        # Tracker
        self.tracker = SimpleTracker()

        # Data
        self.input_node = "image"

    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 0,
        }

    def run_tidl(self, input_video_path):
        inference_tidl_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLExecutionProvider"],
            provider_options=[self.get_inference_options()],
            sess_options=onnxruntime.SessionOptions(),
        )
        # Input reader
        image_loader = ImageLoader(input_video_path)
        print(f"Total frames: {len(image_loader)}")

        width, height = image_loader.get_res()
        fps = image_loader.get_fps()

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 files
        video_writer = cv2.VideoWriter(os.path.join(self.reports_folder, "matched_sequence.mp4"), fourcc, fps, (self.w, self.h))

        for _ in tqdm(range(len(image_loader))):
            image = image_loader.read()
            inputs = preprocess_image(image, self.w, self.h)
            raw = raw_image(image, self.w, self.h)
            features = inference_tidl_session.run(None, {self.input_node: inputs})
            descriptors, scores_map = features

            keypoints, descriptors, scores = self.dkd(scores_map, descriptors)

            if self.sort:
                indices = np.argsort(scores)[::-1]  # Get indices for sorting in descending order
                keypoints = keypoints[indices]
                descriptors = descriptors[indices]
                scores = scores[indices]
            out, N_matches = self.tracker.update(raw, keypoints, descriptors)
            video_writer.write(out)

    def run_cpu(self, input_video_path):
        inference_tidl_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=onnxruntime.SessionOptions(),
        )
        # Input reader
        image_loader = ImageLoader(input_video_path)
        print(f"Total frames: {len(image_loader)}")

        width, height = image_loader.get_res()
        fps = image_loader.get_fps()
        print(f"FPS: {fps}")

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 files
        video_writer = cv2.VideoWriter(os.path.join(self.reports_folder, "matched_sequence.mp4"), fourcc, fps, (self.w, self.h))

        for idx in tqdm(range(5000)):
            image = image_loader.read()
            if idx < 3000:
                continue
            inputs = preprocess_image(image, self.w, self.h)
            raw = raw_image(image, self.w, self.h)
            features = inference_tidl_session.run(None, {self.input_node: inputs})
            descriptors, scores_map = features

            keypoints, descriptors, scores = self.dkd(scores_map, descriptors)

            if self.sort:
                indices = np.argsort(scores)[::-1]  # Get indices for sorting in descending order
                keypoints = keypoints[indices]
                descriptors = descriptors[indices]
                scores = scores[indices]
            out, N_matches = self.tracker.update(raw, keypoints, descriptors)
            video_writer.write(out)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="small_192_256", help="Model name")
    parser.add_argument("--shape", type=str, default="256x192", help="Input shape")
    #input video
    parser.add_argument("--input", type=str, default="/home/workdir/assets/videos/sequence.mp4", help="Input video path")
    args = parser.parse_args()

    alike = ALIKEPipeline(args.model, shape=args.shape, n_desc=200, detector="maxpool")

    # Run CPU
    alike.run_cpu(args.input)

    print("Done")
