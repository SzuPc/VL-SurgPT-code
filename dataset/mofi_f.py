import os
import copy
import io
import torch
import pickle
import numpy as np
import random

from glob import glob

import pickle
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as T


# These classes are adapted from https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/kubric_movif_dataset.py

class Movi_F_Base(torch.utils.data.Dataset):
    def __init__(self, args):
        super(Movi_F_Base, self).__init__()

        self.data_root = args.movi_f_root
        self.seq_len = args.T
        self.traj_per_sample = args.N
        self.sample_vis_1st_frame = True
        self.use_augs = True
        self.crop_size = args.input_size # (384, 512)
        self.augmentation = args.augmentation

        # photometric augmentation
        self.photo_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = T.GaussianBlur(11, sigma=(0.1, 2.0))

        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15

        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

    def getitem_helper(self, index):
        return NotImplementedError

    def __getitem__(self, index):

        rgbs, trajs, visibles, got_k_points = self.getitem_helper(index)
        # rgbs: (T, 3, H, W)
        # trajs: (T, got_k_points, 2)
        # visibles: (T, got_k_points)
        # got_k_points: int

        # <Trajectories>
        trajs_zero = torch.zeros(self.seq_len, self.traj_per_sample, 2)
        trajs_zero[:, :got_k_points] = trajs
        trajs = trajs_zero
        # </Trajectories>

        # <Visibility>
        visibles_zero = torch.zeros(self.seq_len, self.traj_per_sample, dtype=torch.bool)
        visibles_zero[:, :got_k_points] = visibles
        visibles = visibles_zero
        # </Visibility>
        
        return rgbs, trajs, visibles, got_k_points

    def add_photometric_augs(self, rgbs, trajs, visibles, eraser=True, replace=True):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.eraser_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:
            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs
            ]
            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt
            ]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                            np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles

    def add_spatial_augs(self, rgbs, trajs, visibles, target_crop_size):
        T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]

        ############ spatial transform ############

        # padding
        pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
        pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

        rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
        trajs[:, :, 0] += pad_x0
        trajs[:, :, 1] += pad_y0
        H, W = rgbs[0].shape[:2]

        # scaling + stretching
        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        scale_x = scale
        scale_y = scale
        H_new = H
        W_new = W

        scale_delta_x = 0.0
        scale_delta_y = 0.0

        rgbs_scaled = []
        for s in range(S):
            if s == 1:
                scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                scale_delta_x = (
                    scale_delta_x * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
                scale_delta_y = (
                    scale_delta_y * 0.8
                    + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                )
            scale_x = scale_x + scale_delta_x
            scale_y = scale_y + scale_delta_y

            # bring h/w closer
            scale_xy = (scale_x + scale_y) * 0.5
            scale_x = scale_x * 0.5 + scale_xy * 0.5
            scale_y = scale_y * 0.5 + scale_xy * 0.5

            # don't get too crazy
            scale_x = np.clip(scale_x, 0.2, 2.0)
            scale_y = np.clip(scale_y, 0.2, 2.0)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # make it at least slightly bigger than the crop area,
            # so that the random cropping can add diversity
            H_new = np.clip(H_new, target_crop_size[0] + 10, None)
            W_new = np.clip(W_new, target_crop_size[1] + 10, None)
            # recompute scale in case we clipped
            scale_x = (W_new - 1) / float(W - 1)
            scale_y = (H_new - 1) / float(H - 1)
            rgbs_scaled.append(cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR))
            trajs[s, :, 0] *= scale_x
            trajs[s, :, 1] *= scale_y
        rgbs = rgbs_scaled

        ok_inds = visibles[0, :] > 0
        vis_trajs = trajs[:, ok_inds]  # S,?,2

        if vis_trajs.shape[1] > 0:
            mid_x = np.mean(vis_trajs[0, :, 0])
            mid_y = np.mean(vis_trajs[0, :, 1])
        else:
            mid_y = target_crop_size[0]
            mid_x = target_crop_size[1]

        x0 = int(mid_x - target_crop_size[1] // 2)
        y0 = int(mid_y - target_crop_size[0] // 2)

        offset_x = 0
        offset_y = 0

        for s in range(S):
            # on each frame, shift a bit more
            if s == 1:
                offset_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                offset_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
            elif s > 1:
                offset_x = int(
                    offset_x * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                )
                offset_y = int(
                    offset_y * 0.8
                    + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                )
            x0 = x0 + offset_x
            y0 = y0 + offset_y

            H_new, W_new = rgbs[s].shape[:2]
            if H_new == target_crop_size[0]:
                y0 = 0
            else:
                y0 = min(max(0, y0), H_new - target_crop_size[0] - 1)

            if W_new == target_crop_size[1]:
                x0 = 0
            else:
                x0 = min(max(0, x0), W_new - target_crop_size[1] - 1)

            rgbs[s] = rgbs[s][y0 : y0 + target_crop_size[0], x0 : x0 + target_crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0

        H_new = target_crop_size[0]
        W_new = target_crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]

        return rgbs, trajs

    def crop(self, rgbs, trajs, sampled_crop_size):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W
        
        if self.augmentation: # simple random crop
            y0 = 0 if sampled_crop_size[0] >= H_new else np.random.randint(0, H_new - sampled_crop_size[0])
            x0 = 0 if sampled_crop_size[1] >= W_new else np.random.randint(0, W_new - sampled_crop_size[1])

        else:                 # center crop
            y0 = (H_new - sampled_crop_size[0]) // 2
            x0 = (W_new - sampled_crop_size[1]) // 2

        rgbs = [rgb[y0 : y0 + sampled_crop_size[0], x0 : x0 + sampled_crop_size[1]] for rgb in rgbs]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return rgbs, trajs


class Movi_F(Movi_F_Base):
    def __init__(self, args):
        super(Movi_F, self).__init__(args)

        self.root = args.movi_f_root
        self.crop_size = args.input_size    # (448, 448)
            
        self.original_size = (512, 512)
        self.augmentation = args.augmentation

        self.T = args.T
        self.N = args.N
    
        self.seq_names = [fname for fname in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, fname))]
        self.seq_names = sorted(self.seq_names)

    def __len__(self):
        return len(self.seq_names)
    
    def getitem_helper(self, idx):
        # :return rgbs: (T, 3, 448, 448)
        # :return trajs: (T, N, 2)
        # :return visibles: (T, N)

        seq_name = self.seq_names[idx]

        npy_path = os.path.join(self.root, seq_name, seq_name + ".npy")
        rgb_path = os.path.join(self.root, seq_name, "frames")

        img_paths = sorted(os.listdir(rgb_path))
        rgbs_list = []
        for i, img_path in enumerate(img_paths):
            if img_path[-3:] == "png":
                rgbs_list.append(Image.open(os.path.join(rgb_path, img_path)).convert("RGB"))

        assert len(rgbs_list) == 24, f"len(rgbs): {len(rgbs_list)}"


        rgbs = np.stack(rgbs_list)       # (24, 512, 512, 3)
        annot_dict = np.load(npy_path, allow_pickle=True).item()
        traj_2d = annot_dict["coords"]
        visibility = annot_dict["visibility"]

        assert self.T <= len(rgbs)

        if self.T < len(rgbs):
            if self.augmentation:
                start_ind = np.random.choice(len(rgbs) - self.T, 1)[0]
            else:
                start_ind = 0

            rgbs = rgbs[start_ind : start_ind + self.T]
            traj_2d = traj_2d[:, start_ind : start_ind + self.T]
            visibility = visibility[:, start_ind : start_ind + self.T]

        traj_2d = np.transpose(traj_2d, (1, 0, 2))
        visibility = np.transpose(np.logical_not(visibility), (1, 0))

        if self.augmentation:
            sampled_crop_size = self.crop_size
            rgbs, traj_2d, visibility = self.add_photometric_augs(rgbs, traj_2d, visibility)
            rgbs, traj_2d = self.add_spatial_augs(rgbs, traj_2d, visibility, sampled_crop_size)
        else:
            sampled_crop_size = self.crop_size
            rgbs, traj_2d = self.crop(rgbs, traj_2d, sampled_crop_size)

        visibility[traj_2d[:, :, 0] > sampled_crop_size[1]] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > sampled_crop_size[0]] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        visibility = torch.from_numpy(visibility)       # (T, 2048)
        traj_2d = torch.from_numpy(traj_2d)             # (T, 2048, 2), in [0, 448] range

        visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        visibile_pts_inds = visibile_pts_first_frame_inds
        
        if self.augmentation:
            point_inds = torch.randperm(len(visibile_pts_inds))[: self.N]
        else:
            point_inds = visibile_pts_inds[: self.N]
            

        got_k_points = len(point_inds)

        visible_inds_sampled = visibile_pts_inds[point_inds]
        trajs = traj_2d[:, visible_inds_sampled].float()                          # (T, N, 2)
        visibles = visibility[:, visible_inds_sampled]                            # (T, N)
        rgbs = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float()       # (T, 3, sampled_crop_size[0], sampled_crop_size[1])

        return rgbs, trajs, visibles, got_k_points
