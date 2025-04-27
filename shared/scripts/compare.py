import numpy as np
import sys

sys.path.append('/home/workdir/')
from components.utils import load_tensor_from_file

src_dir = "assets/reports/small_3_sq/"

if __name__ == "__main__":
    python_scores_map = load_tensor_from_file(src_dir + "python_scores_map.tensor")
    cpp_scores_map = load_tensor_from_file(src_dir + "cpp_scores_map.tensor")

    python_descriptor_map = load_tensor_from_file(src_dir + "python_descriptor_map.tensor")
    cpp_descriptor_map = load_tensor_from_file(src_dir + "cpp_descriptor_map.tensor")

    np.testing.assert_allclose(cpp_scores_map, python_scores_map, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(cpp_descriptor_map, python_descriptor_map, rtol=1e-6, atol=1e-5)

    mse_scores_map = np.mean((python_scores_map - cpp_scores_map) ** 2)
    mse_descriptor_map = np.mean((python_descriptor_map - cpp_descriptor_map) ** 2)

    print(f"mse_scores_map : {mse_scores_map}")
    print(f"mse_descriptor_map : {mse_descriptor_map}")