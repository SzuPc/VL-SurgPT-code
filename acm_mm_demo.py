import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as Tr
import torchvision.transforms.functional as TF
from model.track_on_ff import TrackOnFF    # Frame Inputs


from glob import glob
from PIL import Image
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio.v2 as imageio



# === Helper functions ===
def read_video(video_path):
    reader = imageio.get_reader(video_path)
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    video = np.stack(frames)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # (T, 3, 720, 1920)
    
    print(f"{video.shape[0]} frames in video")
    
    plt.imshow(video[0].permute(1, 2, 0).long())
    
    return video
    
def write_gif(png_dir, out_dir):
    images = []
    
    sorted_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    
    for z, file_name in enumerate(sorted_files):    
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

    imageio.mimsave(out_dir, images, fps=30)



# === Set Model Arguments ===
from utils.train_utils import restart_from_checkpoint_not_dist

class Args:
    def __init__(self):
        self.input_size = [384, 512]

        self.N = 384
        self.T = 18
        self.stride = 4
        self.transformer_embedding_dim = 256
        self.cnn_corr = False
        self.linear_visibility = False
        
        self.num_layers = 3
        self.num_layers_offset_head = 3
        
        self.num_layers_rerank = 3
        self.num_layers_rerank_fusion = 1
        self.top_k_regions = 16

        self.num_layers_spatial_writer = 3
        self.num_layers_spatial_self = 1
        self.num_layers_spatial_cross = 1
        
        self.memory_size = 12
        self.val_memory_size = 96
        self.val_vis_delta = 0.9
        self.random_memory_mask_drop = 0

        self.lambda_point = 5.0
        self.lambda_vis = 1.0
        self.lambda_offset = 1.0
        self.lambda_uncertainty = 1.0
        self.lambda_top_k = 1.0
        
        self.epoch_num = 4
        self.lr = 1e-3
        self.wd = 1e-4
        self.bs = 1
        self.gradient_acc_steps = 1

        self.validation = False
        self.checkpoint_path = "/home/lenovo/acm_mm_dataset/track_on/checkpoints/track_on_checkpoint.pt"
        self.seed = 1234
        self.loss_after_query = True

        self.gpus = torch.cuda.device_count()

args = Args()

# === 加载模型 ===
model = TrackOnFF(args)
restart_from_checkpoint_not_dist(args, run_variables={}, model=model)
model.cuda().eval()
model.set_memory_size(args.val_memory_size, args.val_memory_size)
model.visibility_treshold = args.val_vis_delta

# === 推理函数 ===
def infer(video_path, queries):
    """对输入视频和查询点进行推理"""
    video = read_video(video_path)  # (T, 3, H, W)
    queries = torch.tensor(queries).cuda()  # 查询点 (N, 2)
    T, N = video.shape[0], queries.shape[0]

    results = []  # 存储每帧的点结果
    with torch.no_grad():
        for t in range(T):
            if t == 0:
                model.init_queries_and_memory(queries, video[t].unsqueeze(0).cuda())
            point, vis = model.ff_forward(video[t].unsqueeze(0).cuda())
            results.append((point.cpu().numpy(), vis.cpu().numpy()))
    return results

# === 示例使用 ===
if __name__ == "__main__":
    video_path = "/home/lenovo/acm_mm_dataset/tissue_dataset/0/left/seq000/frames/00000000ms-00000266ms-visible.mp4"  # 输入视频路径
    queries = [[1140, 300], [1160, 350], [1180, 400]]  # 查询点 (x, y)

    results = infer(video_path, queries)
    for t, (points, visibility) in enumerate(results):
        print(f"Frame {t}:")
        for i, (point, vis) in enumerate(zip(points, visibility)):
            print(f"  Query {i}: Point={point}, Visible={bool(vis)}")