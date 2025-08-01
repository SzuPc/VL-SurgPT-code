import os
import sys
import numpy as np
from tqdm import tqdm
import time
import datetime
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import wandb
os.environ["WANDB_MODE"] = "offline"

from read_args import get_args, print_args
from utils.train_utils import init_distributed_mode, fix_random_seeds
from utils.train_utils import get_scheduler
from utils.train_utils import restart_from_checkpoint, save_on_master

from utils.log_utils import init_wandb, log_eval_metrics, log_batch_loss, log_epoch_loss

from utils.eval_utils import Evaluator, compute_tapvid_metrics

from utils.coord_utils import get_queries

from model.track_on import TrackOn

from load_dataset import get_dataloader  # 替换数据加载方式


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
        checkpoint_path = "/mnt/data1_hdd/wa/bleeding_zrl/track_on/checkpoints/track_on_checkpoint.pt"

        self.checkpoint_path = checkpoint_path
        self.seed = 1234
        self.loss_after_query = True

        self.gpus = torch.cuda.device_count()



def train(args, train_dataloader, model, optimizer, lr_scheduler, scaler, epoch):
    model.train()
    total_loss = 0



    model.module.extend_queries = False
    model.module.set_memory_mask_ratio(args.random_memory_mask_drop)

    train_dataloader = tqdm(train_dataloader, disable=args.rank != 0, file=sys.stdout)
    printed_memory = False
    update_num = 0
    device = torch.device(f"cuda:{args.gpu}")

    torch.cuda.reset_peak_memory_stats(device=args.gpu)
    for i, batch in enumerate(train_dataloader):

        seq_info = batch['seq_info']
        scale = batch['scale']  # 获取缩放比例 (scale_w, scale_h)
        video = batch['video'].to(device)                  # [1, T, C, H, W]
        gt_points = batch['points'].to(device)             # [1, N, T, 2]
        gt_visibility = batch['visibility'].to(device)     # [1, N, T]

        # 构造 queries: [B, N, 3] = (t, y, x)
        queries_xy = gt_points[:, :, 0, :]  # 初始帧的点
        t_index = torch.zeros((queries_xy.shape[0], queries_xy.shape[1], 1), device=device)
        queries = torch.cat([t_index, queries_xy], dim=-1)  # [1, N, 3]

        # 调整 tracks 和 visibility 的维度
        gt_tracks = gt_points.permute(0, 2, 1, 3)        # [1, T, N, 2]
        gt_visibility = gt_visibility.permute(0, 2, 1)   # [1, T, N]

        # 模型推理
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            out = model(video, queries, gt_tracks, gt_visibility)


        if args.rank == 0 and epoch == 0 and not printed_memory:
            print(f"\nMemory Usage after forward: {torch.cuda.max_memory_allocated(device=args.gpu) / 1024 ** 3:.1f} GB")
            torch.cuda.reset_peak_memory_stats(device=args.gpu)

        # Compute loss
        loss = torch.zeros(1).cuda()
        for key, value in out.items():
            if "loss" in key:
                loss += value

        total_loss += loss.item()

        # Backward pass
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if args.rank == 0 and epoch == 0 and not printed_memory:
            print(f"Memory Usage after backward: {torch.cuda.max_memory_allocated(device=args.gpu) / 1024 ** 3:.1f} GB")
            printed_memory = True

        if args.amp:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update_num += 1

        total_update_num = epoch * len(train_dataloader) + update_num
        log_batch_loss(args, optimizer, train_dataloader, total_update_num, i, out)

    log_epoch_loss(args, total_loss, epoch, train_dataloader)


@torch.no_grad()
def evaluate(args, val_dataloader, model, epoch, verbose=False):
    model.eval()
    model.module.extend_queries = True
    model.module.set_memory_mask_ratio(0)

    evaluator = Evaluator()
    total_frames = 0
    total_time = 0

    for j, (video, trajectory, visibility, query_points_i) in enumerate(tqdm(val_dataloader, disable=verbose, file=sys.stdout)):
        # Timer start
        start_time = time.time()
        total_frames += video.shape[1]

        query_points_i = query_points_i.cuda(non_blocking=True)      # (1, N, 3)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)              # (1, T, N)
        video = video.cuda(non_blocking=True)                    # (1, T, 3, H, W)
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device

        # Change (t, y, x) to (t, x, y)
        queries = query_points_i.clone().float()
        queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)


        if args.online_validation:
            out = model.module.inference(video, queries)
        else:
            out = model.module(video, queries, trajectory, visibility)
        pred_trajectory = out["points"]                # (1, T, N, 2)
        pred_visibility = out["visibility"]            # (1, T, N)

        # Timer end
        total_time += time.time() - start_time
 
        # === === ===
        # From CoTracker
        traj = trajectory.clone()
        query_points = query_points_i.clone().cpu().numpy()
        gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
        # === === ===


        out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first")
        if verbose:
            print(f"Video {j}/{len(val_dataloader)}: AJ: {out_metrics['average_jaccard'][0] * 100:.2f}, delta_avg: {out_metrics['average_pts_within_thresh'][0] * 100:.2f}, OA: {out_metrics['occlusion_accuracy'][0] * 100:.2f}", flush=True)
        evaluator.update(out_metrics)
        
    fps = total_frames / total_time
    print(f"Evaluation FPS: {fps:.2f}", flush=True)

    results = evaluator.get_results()
    smaller_delta_avg = results["delta_avg"]
    aj = results["aj"]
    oa = results["oa"]

    
    log_eval_metrics(args, results, epoch)

    return smaller_delta_avg, aj, oa


def main_worker(args):
    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    print_args(args)
    start_time = time.time()

    # ##### Data #####
    dataset_root = Path('/home/lenovo/acm_mm_dataset/tissue_dataset')  # 数据集路径
    train_dataloader = get_dataloader(dataset_root, batch_size=args.bs)  # 加载训练数据
    # val_dataloader = get_dataloader(dataset_root / 'val', batch_size=1)  # 加载验证数据
    if train_dataloader is not None:
        print(f"Total number of iterations: {len(train_dataloader) * args.epoch_num / 1000:.1f}K")
    # ##### ##### #####
    
    # ##### Model & Training #####
    model = TrackOn(args).to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻 offset_head 的参数
    for param in model.module.offset_head.parameters():
        param.requires_grad = True

    # 检查解冻的参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Training parameter: {name}")

    if not args.validation:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = get_scheduler(args, optimizer, train_dataloader)
        scaler = torch.GradScaler() if args.amp else None
        init_wandb(args)

        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 10**6:.2f}M")
        print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f}M")
        print()
    # === === ===

    # === Load from checkpoint ===
    to_restore = {"epoch": 0}
    if args.checkpoint_path is not None:
        if args.validation:
            restart_from_checkpoint(args, 
                                run_variables=to_restore, 
                                model=model)
        else:
            restart_from_checkpoint(args, 
                                    run_variables=to_restore, 
                                    model=model,
                                    scaler=scaler,
                                    optimizer=optimizer, 
                                    scheduler=lr_scheduler)
    
    
    start_epoch = to_restore["epoch"]

    # if args.validation and args.rank == 0:
    #     model.module.visibility_treshold = args.val_vis_delta
    #     model.module.set_memory_size(args.val_memory_size, args.val_memory_size)
    #     evaluate(args, val_dataloader, model, -1, verbose=True)
    #     total_time = time.time() - start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     print()
    #     print('Validation time {}'.format(total_time_str))

    dist.barrier()
    if args.validation:
        dist.destroy_process_group()
        return
    
    print("Training starts")


    best_models = {"aj": [-1, -1], "oa": [-1, -1], "delta_avg": [-1, -1]}       # [epoch, value]
    for epoch in range(start_epoch, args.epoch_num):
        # train_dataloader.sampler.set_epoch(epoch)

        print(f"=== === Epoch {epoch} === ===")

        # === === Training === ===

        train(args, train_dataloader, model, optimizer, lr_scheduler, scaler, epoch)
        print()
        # === === ===
        


        dist.barrier()

        print(f"=== === === === === ===")
        print()

    # print best results
    if args.rank == 0:
        print("Best Results")
        print(f"Best AJ: {best_models['aj'][1]:.3f} at epoch {best_models['aj'][0]}")
        print(f"Best OA: {best_models['oa'][1]:.3f} at epoch {best_models['oa'][0]}")
        print(f"Best Smaller Delta Avg: {best_models['delta_avg'][1]:.3f} at epoch {best_models['delta_avg'][0]}")
    
    wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args()
    # args = Args()  # 使用 TrackOn 的参数
    main_worker(args)