# Development enviromnet for conveting, testing, benchmarking and deploying ALIKE to BeagleBone AI-64

# ALIKE Project Documentation

## Table of Contents
Model Manipulation:
- [Installation](#installation)
- [ALIKE Model Export Script](#alike-model-export-script)
- [SubGraphCompiler Script](#subgraphcompiler-script)
- [Tester Script](#tester-script)
- [ALIKE HPatches Benchmark Script](#alike-hpatches-benchmark-script)

Deployment:
- [Model Packaging & Deployment Script](#model-packaging--deployment-script)


Misc
- [Credits](#credits)


## Installation

`All model manipulations, including testing and benchmarking, are performed inside the container!`

To build and run it:
```bash
cd shared
make build && make run && make exec
```

Then, inside the container:

## ALIKE Model Export Script

### Overview

This script exports ALIKE (Accurate LIKElihood keypoint detector) models to ONNX format. ALIKE is a deep learning-based keypoint detector that can be used for feature matching, image alignment, and other computer vision tasks.

### Available Model Configurations

The script supports four pre-configured ALIKE models with different computational requirements and capabilities:

- `alike-t`: Tiny model (smallest, fastest, least accurate)
- `alike-s`: Small model
- `alike-n`: Normal model (balanced)
- `alike-l`: Large model (largest, slowest, most accurate)

### Usage

```bash
python scripts/export_feature_extractor.py --config MODEL_CONFIG [options]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | *required* | ALIKE configuration preset (`alike-t`, `alike-s`, `alike-n`, or `alike-l`) |
| `--input` | str | "image" | Input node name in the ONNX model |
| `--file` | str | "model" | Output file name for the exported model |
| `--cut` | str | "" | Optional cutting point in the model (for partial exports) |
| `--shape` | str | "640x480" | Shape of the calibration image (WIDTHxHEIGHT) |
| `--opset` | int | 11 | ONNX operation set version |
| `--export-folder` | str | DEFAULT_EXPORT_FOLDER | Directory to save the exported model |

### Examples

Export the tiny ALIKE model:
```bash
python scripts/export_feature_extractor.py --config alike-t
```

Export the large ALIKE model with custom dimensions:
```bash
python scripts/export_feature_extractor.py --config alike-l --shape 640x480
```

Export with custom input node name and output file:
```bash
python scripts/export_feature_extractor.py --config alike-n --file alike_normal_model
```

## SubGraphCompiler Script

### Overview

This script compiles and tests ONNX models for deployment on TI devices using the TIDL (Texas Instruments Deep Learning) framework. It provides functionality for model compilation, quantization, and comparative inference between CPU and TIDL execution providers.

### Usage

```bash
python scripts/compile_feature_extractor.py --model MODEL_NAME [options]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | *required* | Name of the ONNX model to compile |
| `--shape` | str | "640x480" | Input image dimensions (WIDTHxHEIGHT) |
| `--frames` | int | 1 | Number of calibration frames to use |
| `--iters` | int | 1 | Number of calibration iterations |
| `--bits` | int | 16 | Quantization bit-depth (typically 8 or 16) |
| `--data` | str | "data" | Base name of calibration dataset CSV file |

### Examples

Compile a model with default settings:
```bash
python scripts/compile_feature_extractor.py --model model_name
```

Compile a model with custom quantization and calibration:
```bash
python scripts/compile_feature_extractor.py --model model_name --shape 1280x720 --frames 10 --iters 2 --bits 8
```

### Notes

- The artifacts folder is created at `/home/workdir/assets/artifacts/MODEL_NAME/`
- Compilation options include:
  - Accuracy level: 1
  - Tensor bits: 16 (default)
  - Denied layer types: Slice, Split, Reshape, Squeeze
- The script compares outputs between CPU and TIDL executions using multiple error margins (0.1, 0.01, 0.001, 0.0001)

## Tester Script

### Overview

This script tests and compares the performance of ONNX models between CPU and TIDL (Texas Instruments Deep Learning) execution providers. It generates visual reports to evaluate the accuracy of TIDL-compiled models against the original ONNX implementation.

### Usage

```bash
python scripts/test_feature_extractor.py --model MODEL_NAME [options]
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | *required* | Name of the ONNX model to test |
| `--shape` | str | "480x640" | Input image dimensions (WIDTHxHEIGHT) |
| `--frames` | int | 1 | Number of test frames to process |

### Examples

Test a model with default settings:
```bash
python scripts/test_feature_extractor.py --model model_name
```

Test a model with custom dimensions and more test frames:
```bash
python scripts/test_feature_extractor.py --model model_name --shape 1280x720 --frames 10
```

### Notes
- Visual reports are generated in `/home/workdir/assets/reports/MODEL_NAME/`
- The script uses the artifacts folder at `/home/workdir/assets/artifacts/MODEL_NAME/`

## ALIKE HPatches Benchmark Script

### Overview

This scriptextracts keypoints and descriptors from images in the dataset and saves them for later evaluation of matching performance.

### Usage

The script is designed to be run directly without command-line arguments:

```bash
python scripts/run_extraction.py
```

### Configuration Parameters

Configure the following variables at the top of the script:

| Parameter | Description |
|-----------|-------------|
| `dataset_root` | Path to the HPatches dataset |
| `methods` | List of ALIKE models to benchmark (e.g., 'alike-n', 'alike-l', 'alike-n-ms', 'alike-l-ms') |
| `use_cuda` | Automatically set based on GPU availability |

### Dataset
Before running, donwload HPatches dataset and place it inside
`shared/ALIKE/hseq/hpatches-sequences-release`

### Output

The script saves extracted features for each image as NPZ files in the HPatches dataset directory:
- Format: `[sequence_name]/[image_number].ppm.[method_name]`
- Each NPZ file contains:
  - `keypoints`: Nx2 array of (x,y) coordinates
  - `descriptors`: NxD array of feature descriptors 
  - `scores`: N array of keypoint confidence scores


## Model Packaging & Deployment Script

### Overview
This script packages an ONNX model with its TIDL artifacts and deployment scripts, then transfers the package to a target device.

### Usage
```bash
./deploy/deploy_model.sh MODEL_NAME
```

### Arguments
- `MODEL_NAME`: Name of the model to package and deploy (without file extension)

### Workflow
1. Creates a package folder structure under `packages/MODEL_NAME/`
2. Copies TIDL artifacts from `../shared/assets/artifacts/MODEL_NAME/`
3. Renames the artifacts directory to match expected structure
4. Removes any temporary directories
5. Copies the ONNX model and renames it to `model.onnx`
6. Copies Python scripts from the `scripts/` directory
7. Deploys the package to a device named 'beagle' at `/opt/model_zoo/` via SSH


### Example
```bash
./deploy/deploy_model.sh alike-n
```

## Credits

### ALIKE

This project uses the ALIKE (Accurate LIKElihood keypoint detector) model developed by:

- Jiaming Zhang, Xiaolong Jiang, Jian Wang, Pengfei Li, Kuo-Chin Lien, Yingtian Liu, Zhe Liu, and Yun Fu
- Northeastern University and Lumen Technologies
- Paper: [ALIKE: Accurate and Lightweight Keypoint Detection and Descriptor Extraction](https://arxiv.org/abs/2112.02906)
- GitHub: [https://github.com/Shiaoming/ALIKE](https://github.com/Shiaoming/ALIKE)

### Environment

The deployment environment is based on the TIDL Toy Docker project:
- GitHub: [https://github.com/FoxFourAI/tidl_toy_docker](https://github.com/FoxFourAI/tidl_toy_docker)
- This provides a containerized environment for working with the Texas Instruments Deep Learning (TIDL) framework

### Acknowledgments

- This project leverages the Texas Instruments Deep Learning (TIDL) framework for deploying deep learning models on TI processors
- The HPatches dataset is used for benchmarking feature matching performance