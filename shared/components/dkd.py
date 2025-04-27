import torch
import numpy as np
from torch import nn


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

def simpler_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    supp_mask = max_pool(max_mask.float()) > 0
    supp_scores = torch.where(supp_mask, zeros, scores)
    new_max_mask = supp_scores == max_pool(supp_scores)
    max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)
    
def simplest_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    return torch.where(max_mask, scores, zeros)

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

    def detect_keypoints(self, scores_map):
        global simple_nms_time
        _, _, h, w = scores_map.shape
        scores_map = torch.tensor(scores_map, requires_grad=False).squeeze(0)
        scores_nograd = scores_map
        nms_scores = simplest_nms(scores_nograd, self.radius)

        # remove border
        nms_scores[:, :self.radius + 1, :] = 0
        nms_scores[:, :, :self.radius + 1] = 0
        nms_scores[:, h - self.radius:, :] = 0
        nms_scores[:, :, w - self.radius:] = 0

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
        descriptors_ = descriptors_ / np.linalg.norm(descriptors_, axis=1)
        descriptors_ = descriptors_.T
        return descriptors_.squeeze(2)

    def forward(self, scores_map, descriptor_map):
        """
        :param scores_map:  1xHxW
        :param descriptor_map: CxHxW
        :param sub_pixel: whether to use sub-pixel keypoint detection
        :return: kpts: list[Nx2,...]; kptscores: list[N,....] normalised position: -1.0 ~ 1.0
        """

        keypoints, kptscores = self.keypoint_detector(scores_map)
        #print(f"TYPES (keypoints, kptscores):  {type(keypoints)}, {type(kptscores)}")
        #print(f"DTYPES (keypoints, kptscores):  {keypoints.dtype}, {kptscores.dtype}")
        #print(f"SHAPES (keypoints, kptscores): {keypoints.shape}, {kptscores.shape}")
        #exit(0)
        descriptors = self.sample_descriptor(descriptor_map, keypoints)
        return keypoints, descriptors, kptscores
    



def tiled_detect_keypoints(self, scores_map):
    def reshape_split(image, kernel):
        img_h, img_w = image.shape
        image = image.reshape(img_h // kernel, kernel, img_w // kernel, kernel).swapaxes(1, 2)
        flattened_tiles = image.reshape(img_h // kernel, img_w // kernel, kernel * kernel)
        return flattened_tiles
    _, _, h, w = scores_map.shape
    kernel = 4
    reshaped_image = reshape_split(scores_map.squeeze(0).squeeze(0), kernel)
    num_tiles_h, num_tiles_w, _ = reshaped_image.shape
    argmax_indices = np.argmax(reshaped_image, axis=2)
    values = np.take_along_axis(reshaped_image, argmax_indices[:, :, np.newaxis], axis=2).squeeze()
    tile_row_starts = np.arange(num_tiles_h) * kernel
    tile_col_starts = np.arange(num_tiles_w) * kernel
    global_row_indices = tile_row_starts[:, np.newaxis] + argmax_indices // kernel
    global_col_indices = tile_col_starts + argmax_indices % kernel
    flat_indices = np.argsort(values.ravel())[-self.top_k:]
    top_values = values.ravel()[flat_indices]
    top_row_indices = global_row_indices.ravel()[flat_indices]
    top_col_indices = global_col_indices.ravel()[flat_indices]
    keypoints_xy = np.vstack((top_col_indices, top_row_indices)).T
    return keypoints_xy, top_values