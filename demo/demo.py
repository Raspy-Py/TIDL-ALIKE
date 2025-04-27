import cv2
import sys
import numpy as np

from utils import ImageLoader, Streamer
from alike import ALikeConfig, ALikePipeline


def preprocess(image, w, h):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    image = image.copy() / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_points_file> <output_image_path>")
        sys.exit(1)
    
    video_src = sys.argv[1]
    host_ip = sys.argv[2]

    image_loader = ImageLoader(video_src)
    W, H = image_loader.get_res()
    FPS = image_loader.get_fps()
    streamer = Streamer(host_ip, W, H, FPS)

    alike_config = ALikeConfig("./config.yaml")
    alike_pipeline = ALikePipeline(alike_config, preprocess=lambda image: preprocess(image, alike_config.w, alike_config.h))

    h_inv_scale = H / alike_config.h
    w_inv_scale = W / alike_config.w

    for _ in range(len(image_loader)):
        image = image_loader.read()
        keypoints, descriptors, scores = alike_pipeline.run(image.copy())
        keypoints = keypoints.astype(np.float64)
        keypoints[:, 0] *= w_inv_scale
        keypoints[:, 1] *= h_inv_scale
        keypoints = keypoints.astype(np.int64)
        
        for point in keypoints:
            cv2.circle(image, tuple(point), 3, (0, 0, 255), -1)

        streamer.write(image)

