import os
import cv2
import yaml
import onnxruntime
import numpy as np
    
# Inference code
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    image = image.copy() / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


class ALikeConfig:
    def __init__(self, config_path: str):
        self.h = 640
        self.w = 640
        self.kernel = 4 
        self.padding = 2
        self.n_descriptors = 500
        self.model_path = "./model.onnx"
        self.artifacts_path = "./artifacts"

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.__dict__.update(config)

    def __str__(self):
        return str(self.__dict__)

class DKD(object):
    def __init__(self, top_k=500, kernel=4, padding=2):
        """
        Args:
            radius: size of the inner zero padding (used to eliminate false keypoints)
            top_k: return top k keypoints
        """
        super().__init__()
        self.top_k = top_k
        self.kernel = kernel
        self.padding = padding

    def tiled_detect_keypoints(self, scores_map):
        _, _, h, w = scores_map.shape
        scores = scores_map.squeeze()
        
        p = self.padding + 1
        scores[:p, :] = scores[-p:, :] = scores[:, :p] = scores[:, -p:] = 0

        num_tiles_h, num_tiles_w = h // self.kernel, w // self.kernel
        
        reshaped = scores.reshape(num_tiles_h, self.kernel, num_tiles_w, self.kernel).swapaxes(1, 2)
        reshaped = reshaped.reshape(num_tiles_h, num_tiles_w, -1)
        argmax_indices = reshaped.argmax(axis=2)
        values = reshaped[np.arange(num_tiles_h)[:, None], np.arange(num_tiles_w), argmax_indices]

        row_indices = (np.arange(num_tiles_h) * self.kernel)[:, None] + argmax_indices // self.kernel
        col_indices = (np.arange(num_tiles_w) * self.kernel) + argmax_indices % self.kernel

        flat_indices = np.argpartition(values.ravel(), -self.top_k)[-self.top_k:]
        top_values = values.ravel()[flat_indices]
        top_row_indices = row_indices.ravel()[flat_indices]
        top_col_indices = col_indices.ravel()[flat_indices]

        keypoints_xy = np.column_stack((top_col_indices, top_row_indices))

        return keypoints_xy, top_values
    
    def sample_descriptor(self, descriptor_map, kpts):
        descriptors = descriptor_map[:, :, kpts[:, 1], kpts[:, 0]]  # 1xDxHxW
        descriptors = descriptors.squeeze(0) # DxHxW
        descriptors = descriptors / np.linalg.norm(descriptors, axis=0)
        return descriptors.T

    def run(self, scores_map, descriptor_map):
        """
        :param scores_map:  1x1xHxW
        :param descriptor_map: 1xCxHxW
        :return: keypoints: np.ndarray; descriptors: np.ndarray; scores: np.ndarray
        """

        keypoints, scores = self.tiled_detect_keypoints(scores_map)
        descriptors = self.sample_descriptor(descriptor_map, keypoints)
       
        return keypoints, descriptors, scores

class FeatureExtractor(object):
    def __init__(self, model_path: str, artifacts_path: str):

        self.input_node = "image"
        self.artifacts_folder = artifacts_path
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["TIDLExecutionProvider"],
            provider_options=[self.get_inference_options()],
            sess_options=onnxruntime.SessionOptions(),
        )

    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 0,
        }
    
    def run(self, image):
        descriptors, scores_map = self.session.run(None, {self.input_node: image})
        return descriptors, scores_map

class ALikePipeline:
    def __init__(self, config: ALikeConfig, preprocess=None):

        self.preprocess = preprocess
        self.onnx_path = config.model_path
        self.artifacts_folder = config.artifacts_path

        self.feature_extractor = FeatureExtractor(config.model_path, config.artifacts_path)
        self.dkd = DKD(config.n_descriptors, config.kernel, config.padding)

    def run(self, image):
        if self.preprocess:
            image = self.preprocess(image)

        descriptors, scores_map = self.feature_extractor.run(image)
        keypoints, descriptors, scores = self.dkd.run(scores_map, descriptors)

        # keypoints: (N, 2)
        # descriptors: (N, D) D=128 for ALike-n
        # scores: (N, )
        return keypoints, descriptors, scores


import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = ALikeConfig("./config.yaml")
    pipeline = ALikePipeline(config, preprocess_image)
    
    image = cv2.imread("image.jpg")

    keypoints, descriptors, scores = pipeline.run(image)


    image = cv2.resize(image, (640, 640), interolation=cv2.INTER_LINEAR)
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='r', s=1)
    plt.axis("off")
    plt.savefig("detection.png")
    plt.close()

