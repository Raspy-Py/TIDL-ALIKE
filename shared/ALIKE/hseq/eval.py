import cv2
import os
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
#from extract import extract_method
import json
from extract_optimized import my_extract_method

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

'''
methods = ['d2', 'lfnet', 'superpoint', 'r2d2', 'aslfeat', 'disk',
           'alike-n', 'alike-l', 'alike-n-ms', 'alike-l-ms', 'alike-o']
names = ['D2-Net(MS)', 'LF-Net(MS)', 'SuperPoint', 'R2D2(MS)', 'ASLFeat(MS)', 'DISK',
         'ALike-N', 'ALike-L', 'ALike-N(MS)', 'ALike-L(MS)', "ALike-O"]
'''
methods = ['alike-o']   
names = ['ALike-o']

# ALike-O for TIDL optimized ALike

top_k = 400
n_i = 57
n_v = 49
cache_dir = './cache'
dataset_path = './hpatches-sequences-release'


def generate_read_function(method, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(dataset_path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'], aux['descriptors']
        else:
            assert ('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k:]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]

    return read_function


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    #sim[sim < 0.9] = 0
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()

def homo_trans(coord, H):
    kpt_num = coord.shape[0]
    homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
    proj_coord = np.matmul(H, homo_coord.T).T
    proj_coord = proj_coord / proj_coord[:, 2][..., None]
    proj_coord = proj_coord[:, 0:2]
    return proj_coord


err_log = {}
def print_nicely_formatted_array(arr):
    formatted_array = np.array([[f"{val:6.5f}" for val in row] for row in arr])
    for row in formatted_array:
        print(" ".join(row))

#import matplotlib.pyplot as plt
def plot_topk_keypoints(read_functions: list, model_names: list, top_k = 200):
    seq_names = sorted(os.listdir(dataset_path))

    N = 6 # 6 images in each sequence
    M = len(read_functions)
    figsize = (10, 10)

    for seq_name in tqdm(seq_names):
        fig, axes = plt.subplots(M, N, figsize=figsize)
        for m, model_name in enumerate(model_names):
            reader = read_function[m]
            for n in range(1, 7):
                keypoints, descriptors = reader(seq_name, n)

                ref_img = cv2.imread(os.path.join(dataset_path, seq_name, '1.ppm'))
                img = deepcopy(ref_img)
                H_, W_, three = ref_img.shape
                model_size = 256
                max_hw = max(H_, W_)
                ratio = float(model_size / max_hw)
                img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)


def benchmark_features(read_feats):
    lim = [1, 5]
    rng = np.arange(lim[0], lim[1] + 1)

    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    i_err_homo = {thr: 0 for thr in rng}
    v_err_homo = {thr: 0 for thr in rng}

    fail_homo = 0
    nice_homo = 0

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        err_log[seq_name] = {"err":[], "err_homo":[], "matches":[]}

        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        ref_img = cv2.imread(os.path.join(dataset_path, seq_name, '1.ppm'))
        img = deepcopy(ref_img)
        H_, W_, three = ref_img.shape
        model_size = 256
        max_hw = max(H_, W_)
        ratio = float(model_size / max_hw)
        img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)

        ref_img_shape = img.shape
        trans_scale_factor = ref_img.shape[0] / img.shape[0]

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )
            err_log[seq_name]["matches"].append(matches.shape[0])


            homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))

            pos_a = keypoints_a[matches[:, 0], : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

            err_log[seq_name]["err"].append(np.mean(dist <= 3)) # MMA@3matches
            

            # =========== compute homography
            gt_homo = homography
            pred_homo, _ = cv2.findHomography(keypoints_a[matches[:, 0], : 2], keypoints_b[matches[:, 1], : 2], cv2.RANSAC)            

            if pred_homo is None:
                fail_homo += 1
                homo_dist = np.array([float("inf")])
                print(f'failed homography for: {seq_name}[{im_idx}]')
            else:
                nice_homo += 1
                #pred_homo[:-1,-1] = homography[:-1,-1]
                corners = np.array([[0, 0],
                                    [ref_img_shape[1] - 1, 0],
                                    [0, ref_img_shape[0] - 1],
                                    [ref_img_shape[1] - 1, ref_img_shape[0] - 1]])
                real_warped_corners = homo_trans(corners, gt_homo)
                warped_corners = homo_trans(corners, pred_homo)
                homo_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err_homo[thr] += np.mean(homo_dist <= thr)
                else:
                    v_err_homo[thr] += np.mean(homo_dist <= thr)
            err_log[seq_name]["err_homo"].append(np.mean(homo_dist <= 3)) # MHA@3
            

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    print(f"successful homographies: {nice_homo / (nice_homo + fail_homo) * 100:.2f}%")

    return i_err, v_err, i_err_homo, v_err_homo, [seq_type, n_feats, n_matches]


if __name__ == '__main__':
    errors = {}
    for method in methods:
        output_file = os.path.join(cache_dir, method + '.npy')
        read_function = generate_read_function(method)
        if method == "alike-o": # ALike optimized for TIDL:
            my_extract_method(method)
            errors[method] = benchmark_features(read_function)
            with open("results.json", "w") as file:
                json.dump(err_log, file)
            #np.save(output_file, errors[method])
        elif os.path.exists(output_file) and False:
            errors[method] = np.load(output_file, allow_pickle=True)
        else:
            #extract_method(method)
            errors[method] = benchmark_features(read_function)
            with open("results.json", "w") as file:
                json.dump(err_log, file)
            #np.save(output_file, errors[method])

    for name, method in zip(names, methods):
        i_err, v_err, i_err_hom, v_err_hom, _ = errors[method]

        print(f"====={name}=====")
        print(f"MMA@1 | MMA@2 | MMA@3 | MHA@1 | MHA@2 | MHA@3 | MHA@4 | MHA@5 ")
        print(  "====================[ILLUMINATION]============================+")
        for thr in range(1, 4):
            err = i_err[thr] / (n_i * 5)
            print(f"{err * 100:>5.2f}%", end='| ')
        for thr in range(1, 6):
            err_hom = i_err_hom[thr] / (n_i * 5)
            print(f"{err_hom * 100:>5.2f}%", end='| ')

        print("\n======================[VIEWPORT]==============================+")
        for thr in range(1, 4):
            err = v_err[thr] / (n_v * 5)
            print(f"{err * 100:>5.2f}%", end='| ')
        for thr in range(1, 6):
            err_hom = v_err_hom[thr] / (n_v * 5)
            print(f"{err_hom * 100:>5.2f}%", end='| ')
        print("\n========================[TOTAL]===============================+")
        for thr in range(1, 4):
            err = (v_err[thr] + i_err[thr]) / ((n_v + n_i) * 5)
            print(f"{err * 100:>5.2f}%", end='| ')
        for thr in range(1, 6):
            err_hom = (v_err_hom[thr] + i_err_hom[thr]) / ((n_v + n_i) * 5)
            print(f"{err_hom * 100:>5.2f}%", end='| ')
        print('')
