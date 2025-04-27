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

dataset_root = '/opt/model_zoo/hseq/hpatches-sequences-release'
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
methods = ['alike-o']#, 'alike-l', 'alike-n-ms', 'alike-l-ms']

model_shape = (256, 256) # (h, w)

# =============
# DIFF UTILS
# =============

def transform(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.copy() / 255.0
    #image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    #image = np.expand_dims(image, axis=0)
    #image = np.transpose(image, (0, 3, 1, 2))
    #return image.astype(np.float32)
    return image.astype(np.float32)

def resize_and_pad(image, model_w, model_h):
    H_, W_, three = image.shape
    if H_ > W_:
        new_h = model_h
        new_w = int(W_ * new_h / H_)
    else:
        new_w = model_w
        new_h = int(H_ * new_w / W_)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    if model_h != new_h:
        h_padding = np.zeros((1, 3, model_h - new_h, new_w), dtype=np.float32)
        image = np.concatenate([image, h_padding], axis=2)
    if model_w != new_w:
        w_padding = np.zeros((1, 3, model_h, model_w - new_w), dtype=np.float32)
        image = np.concatenate([image, w_padding], axis=3)

    return image

def unpad_output(descriptors_map, scores_map, original_size, align_kernel=4):
    o_h, o_w = original_size

    if o_h > o_w:
        new_h = model_shape[0]
        new_w = int(model_shape[1] * o_w / o_h)
    else:
        new_w = model_shape[1]
        new_h = int(model_shape[0] * o_h / o_w)
    
    new_h = ((new_h + align_kernel - 1) // align_kernel) * align_kernel
    new_w = ((new_w + align_kernel - 1) // align_kernel) * align_kernel

    descriptors_map = descriptors_map[:, :, :new_h, :new_w]
    scores_map = scores_map[:, :, :new_h, :new_w] 

    return descriptors_map, scores_map

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

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
    def __init__(self, radius=2, top_k=500, scores_th=0.5, n_limit=5000, detector="tiled"):
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
        if detector == "maxpool":
            self.keypoint_detector = self.detect_keypoints
        else:
            self.keypoint_detector = self.tiled_detect_keypoints

    def tiled_detect_keypoints(self, scores_map):
        _, _, h, w = scores_map.shape
        scores = scores_map.squeeze()
        
        # Zero out borders
        r = self.radius + 1
        scores[:r, :] = scores[-r:, :] = scores[:, :r] = scores[:, -r:] = 0

        kernel = 4
        num_tiles_h, num_tiles_w = h // kernel, w // kernel
        
        # Reshape and find max in each tile
        reshaped = scores.reshape(num_tiles_h, kernel, num_tiles_w, kernel).swapaxes(1, 2)
        reshaped = reshaped.reshape(num_tiles_h, num_tiles_w, -1)
        argmax_indices = reshaped.argmax(axis=2)
        values = reshaped[np.arange(num_tiles_h)[:, None], np.arange(num_tiles_w), argmax_indices]

        # Calculate global indices
        row_indices = (np.arange(num_tiles_h) * kernel)[:, None] + argmax_indices // kernel
        col_indices = (np.arange(num_tiles_w) * kernel) + argmax_indices % kernel

        # Find top k points
        flat_indices = np.argpartition(values.ravel(), -self.top_k)[-self.top_k:]
        top_values = values.ravel()[flat_indices]
        top_row_indices = row_indices.ravel()[flat_indices]
        top_col_indices = col_indices.ravel()[flat_indices]

        keypoints_xy = np.column_stack((top_col_indices, top_row_indices))

        return keypoints_xy, top_values

    def detect_keypoints(self, scores_map):
        global simple_nms_time
        _, _, h, w = scores_map.shape
        scores_map = torch.tensor(scores_map, requires_grad=False).squeeze(0)
        scores_nograd = scores_map
        nms_scores = simple_nms(scores_nograd, self.radius)

        # remove border
        nms_scores[:, :self.radius*2, :] = 0
        nms_scores[:, :, :self.radius*2] = 0
        nms_scores[:, h - self.radius*2:, :] = 0
        nms_scores[:, :, w - self.radius*2:] = 0

        # detect keypoints without grad
        if self.top_k > 0:
            topk = torch.topk(nms_scores.view(-1), self.top_k)
            indices_kpt = topk.indices  # B x top_k
        else:
            if self.scores_th > 0:
                mask = nms_scores > self.scores_th
                if mask.sum() == 0:
                    th = scores_nograd.reshape(-1).mean(dim=1)  # th = self.scores_th
                    mask = nms_scores > th.reshape(1, 1, 1)
            else:
                th = scores_nograd.reshape(-1).mean(dim=1)  # th = self.scores_th
                mask = nms_scores > th.reshape(1, 1, 1)
            mask = mask.reshape(-1)

            scores = scores_nograd.reshape(-1)

            indices = mask.nonzero(as_tuple=False)[:, 0]
            if len(indices) > self.n_limit:
                kpts_sc = scores[indices]
                sort_idx = kpts_sc.sort(descending=True)[1]
                sel_idx = sort_idx[:self.n_limit]
                indices = indices[sel_idx]
            indices_kpt = indices


        keypoints_xy_nms = torch.stack([indices_kpt % w, indices_kpt // w], dim=1)  # Mx2
        keypoints_xy = keypoints_xy_nms / keypoints_xy_nms.new_tensor([w - 1, h - 1]) * 2 - 1  # (w,h) -> (-1~1,-1~1)
        kptscore = torch.nn.functional.grid_sample(scores_map.unsqueeze(0),
                                                    keypoints_xy.view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True)[0, 0, 0, :]  # CxN
        keypoints_xy = (keypoints_xy + 1) / 2 * keypoints_xy.new_tensor([[w - 1, h - 1]])

        return keypoints_xy.cpu().numpy().astype(np.int64), kptscore.squeeze(0).cpu().numpy()
    
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
        descriptors_ = descriptors_.squeeze(0)
        descriptors_ = descriptors_ / np.linalg.norm(descriptors_, axis=0)
        return descriptors_.T

    def forward(self, scores_map, descriptor_map):
        """
        :param scores_map:  1xHxW
        :param descriptor_map: CxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """
        h, w = descriptor_map.shape[2:]
        keypoints, kptscores = self.keypoint_detector(scores_map)
        #print(f"TYPES (keypoints, kptscores):  {type(keypoints)}, {type(kptscores)}")
        #print(f"DTYPES (keypoints, kptscores):  {keypoints.dtype}, {kptscores.dtype}")
        #print(f"SHAPES (keypoints, kptscores): {keypoints.shape}, {kptscores.shape}")
        #exit(0)
        descriptors = self.sample_descriptor(descriptor_map, keypoints)
        keypoints = keypoints / np.array([w - 1, h - 1]) * 2 - 1
        return keypoints, descriptors, kptscores

class ALIKEPipeline:
    def __init__(self, n_desc=200):
        # Feature Extractor
        self.model = "small_3_sq"
        #self.onnx_path = f"/home/workdir/assets/models/{self.model}.onnx"
        self.onnx_path = f"/opt/model_zoo/{self.model}/model.onnx"
        #self.artifacts_folder = f"/home/workdir/assets/artifacts/{self.model}/"
        self.artifacts_folder = f"/opt/model_zoo/{self.model}/artifacts"

        # DKD
        self.sort = False
        self.dkd = DKD(radius=2, top_k=n_desc, detector="tiled")

        # TIDL runtime
        self.input_node = "image"
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

    def extract_dense_map(self, image):
        descriptors_map, scores_map = self.inference_session.run(None, {self.input_node: image})
        return descriptors_map, scores_map # NumPy arrays
    
    def detect_keypoints(self, scores_map, descriptor_map):
        with torch.no_grad():
            keypoints, descriptors, scores = self.dkd(scores_map, descriptor_map)
        return keypoints, descriptors, scores # NumPy arrays

def extract_multiscale(model, img, model_h, model_w):
    H_, W_, three = img.shape
    assert three == 3, "input image shape should be [HxWx3]"

    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    img = resize_and_pad(img, model_h=model_h, model_w=model_w)

    # ========= extract features
    descriptors_map, scores_map = model.extract_dense_map(img)
    # ========= remove padding
    descriptors_map, scores_map = unpad_output(descriptors_map, scores_map, (H_, W_), align_kernel=4)
    keypoints, descriptors, scores = model.detect_keypoints(scores_map, descriptors_map)


    # restore value
    torch.backends.cudnn.benchmark = old_bm

    # ========= sort descending
    indices = np.argsort(scores)[::-1]  # Sort in descending order
    keypoints = keypoints[indices]
    descriptors = descriptors[indices]
    scores = scores[indices]

    keypoints = (keypoints + 1) / 2 * np.array([W_ - 1, H_ - 1])

    return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_concatinated(name, images, keypoints):
    for i in range(len(images)):
        o_h, o_w, _ = images[i].shape
        #print(f"orginal shape: {images[i].shape}")
        image = resize_and_pad(images[i], model_shape[1], model_shape[0])
        #image = image[:, :, :o_h, :o_w]
        #print(f"unpadded shape: {image.shape}")
        images[i] = np.transpose(image.squeeze(0), (1, 2, 0))
        

    fig, ax = plt.subplots(1, len(images), figsize=(len(images) * 5, 5))

    if len(images) == 1:
        ax = [ax]

    for i in range(len(images)):
        ax[i].imshow(images[i])
        ax[i].scatter(keypoints[i][:, 0], keypoints[i][:, 1], c='r', s=1)
        ax[i].axis('off')

    # Save the result
    output_file = f'./images/{name}.png'
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()

def my_extract_method(m):
    hpatches = HPatchesDataset(root=dataset_root, transform=transform, alteration='all')
    model = m[:7]
    points = 400

    if m == "alike-f": # original feature extractor
        pass
    elif m == "alike-k": # original DKD
        pass

    model = ALIKEPipeline(n_desc=points)

    progbar = tqdm(hpatches, desc='Extracting for {}'.format(m))
    for imgs, homos, seq_name in progbar:
        #logs = (seq_name, [], []) # example | images | keypoints
        for i in range(1, 7):
            img = imgs[i - 1]
            pred = extract_multiscale(model, img, model_shape[0], model_shape[1])
            kpts, descs, scores = pred['keypoints'], pred['descriptors'], pred['scores']
            #logs[1].append(img)
            #logs[2].append(kpts)

            if kpts.shape[0] != points or descs.shape[0] != points or scores.shape[0] != points:
                print(f"Inhomogeneous output detected from: {seq_name}_{i}")
                print(f" - kpts.shape: {kpts.shape}")
                print(f" - descs.shape: {descs.shape}")
                print(f" - scores.shape: {scores.shape}")
                exit(0)

            with open(os.path.join(dataset_root, seq_name, f'{i}.ppm.{m}'), 'wb') as f:
                np.savez(f, keypoints=kpts,
                         scores=scores,
                         descriptors=descs)
        #name, images, keypoints = logs
        #save_concatinated(name, images, keypoints)

if __name__ == '__main__': 
    for method in methods:
        my_extract_method(method)
