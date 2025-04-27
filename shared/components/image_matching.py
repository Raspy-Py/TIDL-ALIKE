import cv2
import copy
import torch
import numpy as np
import kornia.feature as KF
from kornia.testing import is_mps_tensor_safe


class Matcher(object):
    def __init__(self):
        pass

    def _cdist(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        r"""Manual `torch.cdist` for M1."""
        if (not is_mps_tensor_safe(d1)) and (not is_mps_tensor_safe(d2)):
            return torch.cdist(d1, d2)
        d1_sq = (d1**2).sum(dim=1, keepdim=True)
        d2_sq = (d2**2).sum(dim=1, keepdim=True)
        dm = d1_sq.repeat(1, d2.size(0)) + d2_sq.repeat(1, d1.size(0)).t() - 2.0 * d1 @ d2.t()
        dm = dm.clamp(min=0.0).sqrt()
        return dm
    
    def _compute_lafs(self, keypoints, image_size = (256, 144), default_scale=0.02):
        """
        Compute Local Affine Frames (LAFs) using Kornia's laf_from_center_scale_ori function.
        
        Parameters:
        - keypoints (numpy array): Array of keypoints in UV format (shape: [num_keypoints, 2]).
        - image_size (tuple): Tuple containing the (height, width) of the image.
        - default_scale (float): Scale factor for the LAFs, relative to the image size.
        
        Returns:
        - lafs (torch.Tensor): Tensor of LAFs, each represented by a 2x3 matrix (shape: [num_keypoints, 2, 3]).
        """
        # Convert keypoints to a tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)  # shape: [1, num_keypoints, 2]
        
        # Calculate a scale in pixels based on the image size
        scale = default_scale * np.sqrt(image_size[0] * image_size[1])
        
        # Initialize orientations to zero (or any other constant if needed)
        orientations = torch.zeros((1, keypoints.shape[1]), dtype=torch.float32)  # shape: [1, num_keypoints]
        
        # Generate LAFs
        lafs = KF.laf_from_center_scale_ori(keypoints, torch.tensor([scale]), orientations)
        
        return lafs  # shape: [1, num_keypoints, 2, 3]

class MNNMatcher(Matcher):
    def __init__(self, thr=0.95):
        super().__init__()
        self.thr = thr

    def __call__(self, desc1, desc2, kpts1, kpts2):
        sim = desc1 @ desc2.transpose()
        sim[sim < self.thr] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()
    
class FGINMatcher(Matcher):
    def __init__(self, num_kpts = 400, th=0.8, spatial_th=10.0, mutual=True):
        super().__init__()
        self.th = th
        self.spatial_th = spatial_th
        self.mutual = mutual
        self.num_kpts = num_kpts

    
    def __call__(self, desc1, desc2, kpts1, kpts2):
        lafs1 = self._compute_lafs(kpts1)  # (1, N, 2, 3)
        lafs2 = self._compute_lafs(kpts2)  # (1, N, 2, 3)

        distances, indices = KF.match_fginn(desc1, desc2, lafs1, lafs2, th=self.th, spatial_th=self.spatial_th, mutual=self.mutual)

        indices = indices.detach().numpy()
        #distances = distances.squeeze(1).detach().numpy()
        #indices = indices[distances < 0.1]
        return indices


class AdalamMatcher(Matcher):
    def __init__(self, num_kpts = 400, th=0.8, spatial_th=10.0, mutual=True):
        super().__init__()
        self.th = th
        self.spatial_th = spatial_th
        self.mutual = mutual
        self.num_kpts = num_kpts

    def __call__(self, desc1, desc2, kpts1, kpts2):
        lafs1 = self._compute_lafs(kpts1)  # (1, N, 2, 3)
        lafs2 = self._compute_lafs(kpts2)  # (1, N, 2, 3)

        distances, indices = KF.match_adalam(desc1, desc2, lafs1, lafs2)

        indices = indices.detach().numpy()
        #distances = distances.squeeze(1).detach().numpy()
        #indices = indices[distances < 0.1]
        return indices

class SimpleTracker(object):
    def __init__(self, matcher):
        self.pts_prev = None
        self.desc_prev = None
        self.max_matches = 1
        self.matches_history = []
        if matcher is None:
            self.matcher = MNNMatcher()
        else:
            self.matcher = matcher

    def update(self, img, pts, desc):
        N_matches = 0
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc

            out = copy.deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 2, (255, 0, 0), -1, lineType=16)
        else:
            matches = self.matcher(self.desc_prev, desc, self.pts_prev, pts)    
            N_matches = len(matches)

            self.max_matches = max(self.max_matches, N_matches)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]

            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 255), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return out, N_matches


class NotSoSimpleTracker(object):
    def __init__(self, matcher, history_size=50, top_k=400):
        self.pts_prev = None
        self.desc_prev = None
        self.top_k = top_k
        self.max_matches = 1
        self.matches_history = []
        self.history_size = history_size
        self.total_matches = 0
        self.total_frames = 0

        if matcher is None:
            self.matcher = MNNMatcher()
        else:
            self.matcher = matcher

    def update(self, img, pts, desc):
        self.total_frames += 1
        # Compute matches and update history
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc
            out = copy.deepcopy(img)
            pts_np = pts.detach().numpy()
            for pt1 in pts_np:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 2, (255, 0, 0), -1, lineType=16)
            N_matches = 0
            self.matches_history.append(N_matches)
        else:
            matches = self.matcher(self.desc_prev, desc, self.pts_prev, pts)
            N_matches = len(matches)
            self.total_matches += N_matches
            #self.max_matches = max(self.max_matches, N_matches)
            self.matches_history.append(N_matches)

            # Limit the matches history to the defined history size
            if len(self.matches_history) > self.history_size:
                self.matches_history.pop(0)

            # Draw matches on the image
            pts_np = pts.detach().numpy()
            prev_pts_np = self.pts_prev.detach().numpy()
            mpts1, mpts2 = prev_pts_np[matches[:, 0]], pts_np[matches[:, 1]]
            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 255), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        # Draw the number of matches on the image
        out = self.draw_number_of_matches(out, N_matches)

        # Draw the dynamic graph at the bottom
        out = self.draw_history_graph(out)
        return out, N_matches

    def draw_number_of_matches(self, img, n_matches):
        # Position for the text
        text_position = (10, img.shape[0] - 30)  # Adjust as needed
        cv2.putText(img, f'Matches: {n_matches}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def draw_history_graph(self, img):
        # Parameters for the graph
        graph_height = 50
        graph_width = img.shape[1]
        graph_bottom = img.shape[0] - graph_height
        #max_matches = max(1, max(self.matches_history))  # Avoid division by zero

        # Prepare a blank graph area
        #img[graph_bottom:, :] = 0

        # Draw the graph line
        for i in range(1, len(self.matches_history)):
            x1 = int((i - 1) * graph_width / self.history_size)
            x2 = int(i * graph_width / self.history_size)
            y1 = graph_bottom + graph_height - int(self.matches_history[i - 1] * graph_height / self.top_k)
            y2 = graph_bottom + graph_height - int(self.matches_history[i] * graph_height / self.top_k)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return img
    
    def print_stats(self):
        print(f"Total frames: {self.total_frames}")
        print(f"Total matches: {self.total_matches}")
        print(f"Average matches per frame: {self.total_matches / self.total_frames}")
        print(f"Max matches in a frame: {self.max_matches}")
