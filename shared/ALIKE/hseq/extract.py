import os
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from copy import deepcopy
from torchvision.transforms import ToTensor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from alike import ALike, configs

dataset_root = './hpatches-sequences-release'
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
methods = ['alike-n', 'alike-l', 'alike-n-ms', 'alike-l-ms']


model_shape = (256, 256) # (h, w)

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

def unpad_output(descriptors_map, scores_map, original_size):
    o_h, o_w = original_size

    if o_h > o_w:
        new_h = model_shape[0]
        new_w = int(model_shape[1] * o_w / o_h)
    else:
        new_w = model_shape[1]
        new_h = int(model_shape[0] * o_h / o_w)

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


def extract_multiscale(model, img, scale_f=2 ** 0.5,
                       min_scale=1., max_scale=1.,
                       min_size=0., max_size=256.,
                       image_size_max=256,
                       n_k=0, sort=True):
    H_, W_, three = img.shape
    assert three == 3, "input image shape should be [HxWx3]"

    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # ==================== image preprocess
    image = deepcopy(img)
    keypoints, descriptors, scores = [], [], []
    model_h, model_w = model_shape
    image = resize_and_pad(image, model_h=model_h, model_w=model_w)
    image = torch.from_numpy(image).to(device)

    #print(f"image shape|min|max: {image.shape}|{image.min()}|{image.max()}")

    # ==================== forward pass
    with torch.no_grad():
        descriptors_map, scores_map = model.extract_dense_map(image)
        descriptors_map, scores_map = unpad_output(descriptors_map, scores_map, (H_, W_))

        keypoints_, descriptors_, scores_, _ = model.dkd(scores_map, descriptors_map)
        #print(f"shapes: {keypoints_[0].shape}, {descriptors_[0].shape}, {scores_[0].shape}")

    keypoints.append(keypoints_[0])
    descriptors.append(descriptors_[0])
    scores.append(scores_[0])


    # restore value
    torch.backends.cudnn.benchmark = old_bm

    keypoints = torch.cat(keypoints)
    descriptors = torch.cat(descriptors)
    scores = torch.cat(scores)
    keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W_ - 1, H_ - 1]])

    if sort or 0 < n_k < len(keypoints):
        indices = torch.argsort(scores, descending=True)
        keypoints = keypoints[indices]
        descriptors = descriptors[indices]
        scores = scores[indices]

    if 0 < n_k < len(keypoints):
        keypoints = keypoints[0:n_k]
        descriptors = descriptors[0:n_k]
        scores = scores[0:n_k]

    return {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}


def extract_method(m):
    hpatches = HPatchesDataset(root=dataset_root, alteration='all', transform=transform)
    model = m[:7]
    min_scale = 0.3 if m[8:] == 'ms' else 1.0

    model = ALike(**configs[model], device=device, top_k=400, scores_th=0.2, n_limit=5000)

    progbar = tqdm(hpatches, desc='Extracting for {}'.format(m))
    for imgs, homos, seq_name in progbar:
        for i in range(1, 7):
            img = imgs[i - 1]
            pred = extract_multiscale(model, img, min_scale=min_scale, max_scale=1, sort=False, n_k=400, image_size_max=256)
            kpts, descs, scores = pred['keypoints'], pred['descriptors'], pred['scores']

            with open(os.path.join(dataset_root, seq_name, f'{i}.ppm.{m}'), 'wb') as f:
                np.savez(f, keypoints=kpts.cpu().numpy(),
                         scores=scores.cpu().numpy(),
                         descriptors=descs.cpu().numpy())


if __name__ == '__main__':
    for method in methods:
        extract_method(method)
