import os
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import onnxruntime
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from copy import deepcopy
from torchvision.transforms import ToTensor

# PC only
import sys

dataset_root = '/home/workdir/ALIKE/hseq/hpatches-sequences-release'
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
methods = ['alike-n']#, 'alike-l', 'alike-n-ms', 'alike-l-ms']

# =============
# DIFF UTILS
# =============

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    #image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    #image = np.expand_dims(image, axis=0)
    #image = np.transpose(image, (0, 3, 1, 2))
    #return image.astype(np.float32)
    return image

def resize_image(image, model_w, model_h):
    H_, W_, three = image.shape
    if H_ > W_:
        new_h = model_h
        new_w = int(W_ * new_h / H_)
    else:
        new_w = model_w
        new_h = int(H_ * new_w / W_)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def pad_tensor(image, size):
    b, c, h, w = image.shape
    h_, w_ = size

    if h_ != h:
        h_padding = torch.zeros(b, c, h_ - h, w)
        image = torch.cat([image, h_padding], dim=2)
    if w_ != w:
        w_padding = torch.zeros(b, c, h_, w_ - w)
        image = torch.cat([image, w_padding], dim=3)

    return image

def unpad_output(descriptor_map, scores_map, size):
    b, c, h_, w_ = scores_map.shape
    h, w = size

    if h_ != h or w_ != w:
        descriptor_map = descriptor_map[:, :, :h, :w]
        scores_map = scores_map[:, :, :h, :w] 

    return descriptor_map, scores_map

# =============
# MAIN ENTITIES
# =============

class HPatchesDataset(data.Dataset):
    def __init__(self, root: str = dataset_root, alteration: str = 'all', transform = None):
        """
        Args:
            root: dataset root path
            alteration: # 'all', 'i' for illumination or 'v' for viewpoint
        """
        assert (Path(root).exists()), f"Dataset root path {root} dose not exist!"
        self.root = root
        self.transform = transform

        # get all image file name
        self.image0_list = []
        self.image1_list = []
        self.homographies = []
        folders = [x for x in Path(self.root).iterdir() if x.is_dir()]
        self.seqs = []
        for folder in folders:
            if alteration == 'i' and folder.stem[0] != 'i':
                continue
            if alteration == 'v' and folder.stem[0] != 'v':
                continue

            self.seqs.append(folder)

        self.len = len(self.seqs)
        assert (self.len > 0), f'Can not find PatchDataset in path {self.root}'

    def __getitem__(self, item):
        folder = self.seqs[item]

        imgs = []
        homos = []
        for i in range(1, 7):
            img = cv2.imread(str(folder / f'{i}.ppm'), cv2.IMREAD_COLOR)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HxWxC
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

            if i != 1:
                homo = np.loadtxt(str(folder / f'H_1_{i}')).astype('float32')
                homos.append(homo)

        return imgs, homos, folder.stem

    def __len__(self):
        return self.len

    def name(self):
        return self.__class__

class DKD(nn.Module):
    def __init__(self, radius=2, top_k=500, scores_th=0.5, n_limit=5000, detector="default"):
        """
        Args:
            radius: soft detection radius, kernel size is (2 * radius + 1)
            top_k: top_k > 0: return top k keypoints
            scores_th: top_k <= 0 threshold mode:  scores_th > 0: return keypoints with scores>scores_th
                                                   else: return keypoints with scores > scores.mean()
            n_limit: max number of keypoint in threshold mode
        """
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.detector_type = detector
        self.keypoint_detector = self.tiled_detect_keypoints

    def tiled_detect_keypoints(self, scores_map):
        def reshape_split(image, kernel):
            img_h, img_w = image.shape
            image = image.reshape(img_h // kernel, kernel, img_w // kernel, kernel).swapaxes(1, 2)
            flattened_tiles = image.reshape(img_h // kernel, img_w // kernel, kernel * kernel)
            return flattened_tiles
        
        _, _, h, w = scores_map.shape
        scores_map[:, :, :self.radius + 1, :] = 0
        scores_map[:, :, :, :self.radius + 1] = 0
        scores_map[:, :, h - self.radius:, :] = 0
        scores_map[:, :, :, w - self.radius:] = 0

        kernel = 4

        reshaped_image = reshape_split(scores_map.squeeze(0).squeeze(0), kernel)
        num_tiles_h, num_tiles_w, _ = reshaped_image.shape
    
        argmax_indices = np.argmax(reshaped_image, axis=2)
        values = np.take_along_axis(reshaped_image, argmax_indices[:, :, np.newaxis], axis=2).squeeze()

        # global indices
        tile_row_starts = np.arange(num_tiles_h) * kernel
        tile_col_starts = np.arange(num_tiles_w) * kernel

        global_row_indices = tile_row_starts[:, np.newaxis] + argmax_indices // kernel
        global_col_indices = tile_col_starts + argmax_indices % kernel
        flat_indices = np.argsort(values.ravel())[-self.top_k:]
        top_values = values.ravel()[flat_indices]
        top_row_indices = global_row_indices.ravel()[flat_indices]
        top_col_indices = global_col_indices.ravel()[flat_indices]

        keypoints_xy = np.vstack((top_col_indices, top_row_indices)).T
        #keypoints_xy = keypoints_xy / keypoints_xy.new_tensor([w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)

        return keypoints_xy, top_values
    
    def sample_descriptor(self, descriptor_map, kpts, bilinear_interp=False):
        """
        :param descriptor_map: CxHxW
        :param kpts: list, len=B, each is Nx2 (keypoints) [h,w]
        :param bilinear_interp: bool, whether to use bilinear interpolation
        :return: descriptors: list, len=B, each is NxD
        """
        _, _, height, width = descriptor_map.shape

        kptsi = kpts # Nx2,(x,y)

        if bilinear_interp:
            descriptors_ = torch.nn.functional.grid_sample(descriptor_map.unsqueeze(0), kptsi.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, :, 0, :]  # CxN
        else:
            #kptsi = (kptsi + 1) / 2 * kptsi.new_tensor([[width - 1, height - 1]])
            #kptsi = kptsi.long()
            descriptors_ = descriptor_map[:, :, kptsi[:, 1], kptsi[:, 0]]  # CxN
        descriptors_ = descriptors_ / np.linalg.norm(descriptors_, axis=1)

        return descriptors_.T

    def forward(self, scores_map, descriptor_map):
        """
        :param scores_map:  1xHxW
        :param descriptor_map: CxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """

        keypoints, kptscores = self.tiled_detect_keypoints(scores_map)
        descriptors = self.sample_descriptor(descriptor_map, keypoints)
        return keypoints, descriptors, kptscores

class ALIKEPipeline:
    def __init__(self, shape="256x256", n_desc=500):
        # Feature Extractor
        self.onnx_path = "./model.onnx"

        dim_list = [int(dim) for dim in shape.split('x')]
        assert len(dim_list) == 2, f"Error: shape dimension should be 2, but got {len(dim_list)}"
        self.w, self.h = dim_list   
        self.input_shape = (1, 3, self.w, self.h)
        self.input_node = "image"

        # DKD
        self.sort = False
        self.dkd = DKD(radius=2, top_k=n_desc)

        # TIDL runtime
        self.inference_session = onnxruntime.InferenceSession(
            self.onnx_path,
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

    def extract_dense_maps(self, image):
        descriptors_map, scores_map = self.inference_session.run(None, {self.input_node: image})
        return descriptors_map, scores_map # NumPy arrays
    
    def detect_keypoints(self, scores_map, descriptor_map):
        with torch.no_grad():
            keypoints, descriptors, scores = self.dkd(scores_map, descriptor_map)
        return keypoints, descriptors, scores # NumPy arrays

def extract_multiscale(model, img, model_h, model_w, n_k=0, sort=False):
    H_, W_, three = img.shape
    assert three == 3, "input image shape should be [HxWx3]"

    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # ==================== image size constraint
    img = resize_image(img, model_h=model_h, model_w=model_w)
    # transform into a model consumable form
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img_tensor = pad_tensor(torch.tensor(img))

    # extract features
    descriptors_map, scores_map = model.extract_dense_map(img_tensor)
    # remove padding to simpify the job for keypoint detector
    descriptors_map, scores_map = unpad_output(descriptors_map, scores_map, (H_, W_))
    keypoints, descriptors, scores = model.detect_keypoints(descriptors_map, scores_map)
    
    # restore value
    torch.backends.cudnn.benchmark = old_bm

    return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}

def extract_method(m):
    hpatches = HPatchesDataset(root=dataset_root, alteration='all')
    model = m[:7]
    min_scale = 0.3 if m[8:] == 'ms' else 1.0

    model = ALIKEPipeline(n_desc=500)

    progbar = tqdm(hpatches, desc='Extracting for {}'.format(m))
    for imgs, homos, seq_name in progbar:
        for i in range(1, 7):
            img = imgs[i - 1]
            pred = extract_multiscale(model, img, min_scale=min_scale, max_scale=1, sort=False, n_k=5000)
            kpts, descs, scores = pred['keypoints'], pred['descriptors'], pred['scores']

            with open(os.path.join(dataset_root, seq_name, f'{i}.ppm.{m}'), 'wb') as f:
                np.savez(f, keypoints=kpts.cpu().numpy(),
                         scores=scores.cpu().numpy(),
                         descriptors=descs.cpu().numpy())

if __name__ == '__main__':
    for method in methods:
        extract_method(method)
