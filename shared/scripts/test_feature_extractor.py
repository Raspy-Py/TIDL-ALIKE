import os
import shutil
import onnxruntime
import numpy as np
from fire import Fire
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_image_columns(rgb_images, grayscale_images1, grayscale_images2, output_file='image_columns.png'):
    N = len(rgb_images)
    
    fig, axes = plt.subplots(N, 4, figsize=(12, N * 3))

    # Handle the case where N == 1
    if N == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ['Source image', 'Original ONNX', 'TIDL Compiled']
    for ax, col in zip(axes[0], column_titles):
        ax.set_title(col)

    for i in range(N):
        image = np.transpose(rgb_images[i], (0, 2, 3, 1))
        axes[i, 0].imshow(image.squeeze(0))
        axes[i, 0].axis('off')

        axes[i, 1].imshow(grayscale_images1[i][1].squeeze(0).squeeze(0), cmap='viridis')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(grayscale_images2[i][1].squeeze(0).squeeze(0), cmap='viridis')
        axes[i, 2].axis('off')

        error_info = []
        for error_margin in [0.1, 0.01, 0.001, 0.0001]:
            error_value = compare_float_3d_arrays(
                arr1=grayscale_images1[i][0],
                arr2=grayscale_images2[i][0],
                error_margin=error_margin,
            )
            error_info.append(f"Error margin: {error_margin}, Value: {error_value:.2f}")

        # Add text to the figure
        # Add text to the fourth column
        error_text = "\n".join(error_info)
        axes[i, 3].text(0.5, 0.5, error_text, ha='center', va='center', fontsize=8, wrap=True)
        axes[i, 3].axis('off')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


def compare_float_3d_arrays(arr1, arr2, error_margin=0.01):
    total_elements = arr1.size
    matching_elements = np.sum(np.abs(arr1 - arr2) <= error_margin)

    similarity = (matching_elements / total_elements) * 100
    return similarity
    #print(f"The arrays are {similarity:.2f}% similar.")


def relative_difference_for_arrays(arr1, arr2):
    rel_diff = np.mean(np.abs(arr1 - arr2) / np.abs(arr1)) * 100
    print(f"Relative difference: {rel_diff:.2f}%")


class Tester:
    def __init__(self, model, shape="480x640", frames=1):
        self.model_name = model
        self.onnx_path = os.path.join("/home/workdir/assets/models", self.model_name + ".onnx")
        self.reports_folder = os.path.join("/home/workdir/assets/reports", self.model_name)
        self.artifacts_folder = os.path.join("/home/workdir/assets/artifacts", self.model_name)

        dim_list = [int(dim) for dim in shape.split('x')]
        assert len(dim_list) == 2, f"Error: shape dimension should be 2, but got {len(dim_list)}"
        self.w, self.h = dim_list
        self.input_shape = (1, 3, self.h, self.w)

        self.input_node = "image"
        self.test_frames = int(frames)
        self.data = self.get_calibration_tensors()


    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 0,
        }

    def generate_fixed_seed_arrays(self, seed=47):
        np.random.seed(seed)
        return [
            np.random.rand(*self.input_shape).astype(np.float32)
            for _ in range(self.test_frames)
        ]

    def get_calibration_tensors(self):
        calibration_dataset_path = "/home/workdir/assets/data"
        data = pd.read_csv(os.path.join(calibration_dataset_path, "hpatch.csv"))
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
            image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            preprocessed_images.append(preprocess_image(image))
        
        print("Preprocessed images: ", len(preprocessed_images))
        return preprocessed_images


    def compare(self):
        inference_tidl_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLExecutionProvider"],
            provider_options=[self.get_inference_options()],
            sess_options=onnxruntime.SessionOptions(),
        )
        inference_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["CPUExecutionProvider"],
            provider_options=[{}],
            sess_options=onnxruntime.SessionOptions(),
        )

        outputs = []
        outputs_tidl = []
        for inputs in tqdm(self.data[:self.test_frames]):
            outputs.append(inference_session.run(None, {self.input_node: inputs}))
            outputs_tidl.append(inference_tidl_session.run(None, {self.input_node: inputs}))
  
        plot_image_columns(self.data, outputs, outputs_tidl, 
                           output_file=os.path.join(self.reports_folder, "visuals.png"))



if __name__ == "__main__":
    Fire(Tester)
    print("Done")
