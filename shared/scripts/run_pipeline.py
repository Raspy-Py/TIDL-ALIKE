import os
import shutil
import onnxruntime
import numpy as np
from fire import Fire
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# PC only
import sys
sys.path.append('/home/workdir/ALIKE')
sys.path.append('/home/workdir/components')

from dkd import DKD

def plot_image_columns(rgb_images, scoremaps, keypoints, output_file='image_columns.png'):
    N = len(rgb_images)
    
    fig, axes = plt.subplots(N, 3, figsize=(10, N * 3))

    # Handle the case where N == 1
    if N == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ['Source image', 'Scores Map', 'Detected Keypoints']
    for ax, col in zip(axes[0], column_titles):
        ax.set_title(col)

    for i in range(N):
        image = np.transpose(rgb_images[i], (0, 2, 3, 1))
        axes[i, 0].imshow(image.squeeze(0))
        axes[i, 0].axis('off')

        scoremaps_img = scoremaps[i][1].squeeze(0).squeeze(0)
        axes[i, 1].imshow(scoremaps_img, cmap='viridis')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(scoremaps_img, cmap='viridis')
        axes[i, 2].scatter(keypoints[i][:, 0], keypoints[i][:, 1], c='r', s=1)
        axes[i, 2].axis("off")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)

class ALIKEPipeline:
    def __init__(self, model, shape="480x640", frames=1, n_desc=500):
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
        self.dkd = DKD(radius=2, top_k=n_desc, scores_th=0.5, n_limit=5000)

        # Data
        self.input_node = "image"
        self.test_frames = int(frames)
        self.data = self.get_calibration_tensors()


    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 0,
        }

    def get_calibration_tensors(self):
        calibration_dataset_path = "/home/workdir/assets/data"
        data = pd.read_csv(os.path.join(calibration_dataset_path, "data.csv"))
        data["image_path"] = (
            f"{calibration_dataset_path}/"
            + data["image_folder_path"]
            + "/"
            + data["filename"]
        )

        preprocessed_images = []
        for row_index, row in data.iterrows():
            if len(preprocessed_images) == self.test_frames:
                break
            image = cv2.imread(row["image_path"])
            if image is None:
                print(f"Image not found: {row['image_path']}")
                continue
            preprocessed_images.append(preprocess_image(image)[:,:,:self.w,:self.h])
        
        print("Preprocessed images: ", len(preprocessed_images))
        return preprocessed_images


    def run_tidl(self):
        inference_tidl_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLExecutionProvider"],
            provider_options=[self.get_inference_options()],
            sess_options=onnxruntime.SessionOptions(),
        )

        outputs_tidl = []
        for inputs in tqdm(self.data[:self.test_frames]):
            outputs_tidl.append(inference_tidl_session.run(None, {self.input_node: inputs}))


        # ==================== extract keypoints
        #start = time.time()
        keypoints_list = []
        descriptors_list = []
        scores_list = []
        with torch.no_grad():
            for descriptors, scores_map in outputs_tidl:
                descriptors = torch.tensor(descriptors).squeeze(0)
                scores_map = torch.tensor(scores_map).squeeze(0)
                plt.imshow(scores_map.squeeze(0))
                plt.savefig("scores_map.png")
                keypoints, descriptors, scores = self.dkd(scores_map, descriptors)
                keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[self.w - 1, self.h - 1]])


                if self.sort:
                    indices = torch.argsort(scores, descending=True)
                    keypoints = keypoints[indices]
                    descriptors = descriptors[indices]
                    scores = scores[indices]
            
                keypoints_list.append(keypoints)
                descriptors_list.append(descriptors)
                scores_list.append(scores)

        print(f"PLOTTING RESULTS.")
        plot_image_columns(self.data, outputs_tidl, keypoints_list,
                           output_file=os.path.join(self.reports_folder, "pipeline.png"))

if __name__ == "__main__":
    Fire(ALIKEPipeline)
    print("Done")
