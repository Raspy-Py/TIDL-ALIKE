import os
import shutil
import onnxruntime
import numpy as np
from fire import Fire
import cv2
import pandas as pd


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    #image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image.astype(np.float32)


def compare_float_3d_arrays(arr1, arr2, error_margin=0.01):
    total_elements = arr1.size
    matching_elements = np.sum(np.abs(arr1 - arr2) <= error_margin)

    similarity = (matching_elements / total_elements) * 100
    print(f"The arrays are {similarity:.2f}% similar.")


def relative_difference_for_arrays(arr1, arr2):
    rel_diff = np.mean(np.abs(arr1 - arr2) / np.abs(arr1)) * 100
    print(f"Relative difference: {rel_diff:.2f}%")


class SubGraphCompiler:
    def __init__(self, model, shape="640x480", frames=1, iters=1, bits=16, data="data"):
        self.model_name = model
        self.onnx_path = os.path.join("/home/workdir/assets/models", self.model_name + ".onnx")
        self.artifacts_folder = os.path.join("/home/workdir/assets/artifacts", self.model_name)

        dim_list = [int(dim) for dim in shape.split('x')]
        assert len(dim_list) == 2, f"Error: shape dimension should be 2, but got {len(dim_list)}"
        self.w, self.h = dim_list
        self.input_shape = (1, 3, self.h, self.w)

        self.bits = int(bits)
        self.input_node = "image"
        self.calibration_frames = int(frames)
        self.calibration_iterations = int(iters)
        self.data_path = data
        self.data = self.get_calibration_tensors()

    def get_compilation_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "accuracy_level": 1,
            "debug_level": 0,
            "tensor_bits": self.bits,
            "advanced_options:calibration_frames": self.calibration_frames,
            "advanced_options:calibration_iterations": self.calibration_iterations,
            "advanced_options:add_data_convert_ops": 1,
            #"advanced_options:inference_mode":1,                # does not change anything, because device has anly 1 DSP core
            #"advanced_options:high_resolution_optimization": 1, # breaks the network
            "debugTraceLevel": 2,
            "deny_list:layer_type": "Slice,Split,Reshape,Squeeze",
        }

    def get_inference_options(self):
        return {
            "tidl_tools_path": os.environ.get("TIDL_TOOLS_PATH", "/home/workdir/tidl_tools"),
            "artifacts_folder": self.artifacts_folder,
            "debug_level": 7,
        }

    def generate_fixed_seed_arrays(self, seed=47):
        np.random.seed(seed)
        return [
            np.random.rand(*self.input_shape).astype(np.float32)
            for _ in range(self.calibration_frames)
        ]

    def get_calibration_tensors(self):
        calibration_dataset_path = "/home/workdir/assets/data"
        data = pd.read_csv(os.path.join(calibration_dataset_path, f"{self.data_path}.csv"))
        data["image_path"] = (
            f"{calibration_dataset_path}/"
            + data["image_folder_path"]
            + "/"
            + data["filename"]
        )

        preprocessed_images = []
        for row_index, row in data.iterrows():
            if len(preprocessed_images) == self.calibration_frames:
                break
            image = cv2.imread(row["image_path"])
            if image is None:
                print(f"Image not found: {row['image_path']}")
                continue
            
            _, _, three = image.shape
            if three != 3:
                print(f"Image {row['image_path']} is not RGB, skipping")
                continue

            image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            preprocessed_images.append(preprocess_image(image))
        
        print("Preprocessed images: ", len(preprocessed_images))
        return preprocessed_images

    def compile(self):
        if os.path.exists(self.artifacts_folder):
            shutil.rmtree(self.artifacts_folder)
        os.makedirs(self.artifacts_folder)
        compilation_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=["TIDLCompilationProvider"],
            provider_options=[self.get_compilation_options()],
            sess_options=onnxruntime.SessionOptions(),
        )
        print("STARTING COMPILATION")
        for _ in range(self.calibration_iterations):
            for inputs in self.data:
                ort_inputs = {self.input_node: inputs}
                _ = compilation_session.run(None, ort_inputs)

    def inference(self):
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
        all_outputs = []
        for inputs in self.data[:5]:
            outputs = inference_session.run(None, {self.input_node: inputs})[0]
            outputs_tidl = inference_tidl_session.run(None, {self.input_node: inputs})[0]
            all_outputs.append((outputs_tidl, outputs))
        for outputs_tidl, outputs in all_outputs:
            for error_margin in [0.1, 0.01, 0.001, 0.0001]:
                print(f"Error margin: {error_margin}")
                compare_float_3d_arrays(
                    arr1=outputs_tidl[0],
                    arr2=outputs[0],
                    error_margin=error_margin,
                )


if __name__ == "__main__":
    Fire(SubGraphCompiler)
    print("Done")
