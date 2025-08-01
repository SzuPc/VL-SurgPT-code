import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from load_dataset import get_dataloader
import utils.vis_utils as vu
from computer_metrics import TrackingEvaluator, save_sequence_results
import argparse
from model.track_on_ff import TrackOnFF
from utils.train_utils import restart_from_checkpoint_not_dist

logger = logging.getLogger(__name__)

def setup_logging():
    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=format)

@torch.no_grad()
def evaluate_sequence(model, batch):
    """对单个序列进行推理"""
    video = batch['video'].squeeze(0)  # [T, C, H, W]
    gt_points = batch['points'].squeeze(0)  # [N, T, 2]
    gt_visibility = batch['visibility'].squeeze(0)  # [N, T]
    seq_info = batch['seq_info']
    scale = batch['scale']  # 获取缩放比例 (scale_w, scale_h)

    
    T = video.shape[0]
    results = []
    
    # 获取第一帧的点作为查询点
    queries = gt_points[:, 0, :].cuda()  # [N, 2]
    results.append((queries.cpu().numpy(), np.zeros(len(queries), dtype=bool)))

    # 初始化模型
    model.init_queries_and_memory(queries, video[0].unsqueeze(0).cuda())
    
    # 逐帧处理
    for t in tqdm(range(T), desc=f"Processing frames", leave=False):
        frame = video[t].unsqueeze(0).cuda()  # [1, C, H, W]
        point, vis = model.ff_forward(frame)
        results.append((point.cpu().numpy(), vis.cpu().numpy()))
        
    return results, seq_info, scale

def visualize_results(video_frames, tracked_points, gt_points, visibility, out_path):
    """可视化追踪结果"""
    video_writer = vu.VideoWriter(out_path, fps=15)
    
    for t in range(len(video_frames)):
        frame = video_frames[t].cpu().numpy().transpose(1, 2, 0)
        frame = (frame * 255).astype(np.uint8)
        canvas = frame.copy()
        
        # 绘制跟踪点
        pred_pts = tracked_points[t][0].cpu().numpy()
        occluded = tracked_points[t][1].cpu().numpy()
        
        for i in range(len(pred_pts)):
            # 绘制预测点
            color = vu.RED if not occluded[i] else vu.GRAY
            vu.circle(canvas, pred_pts[i], radius=3, color=color, thickness=-1)
            
            # 绘制真值点
            if visibility[i, t]:
                vu.circle(canvas, gt_points[i, t], radius=3, color=vu.GREEN, thickness=1)
        
        video_writer.write(canvas)
    
    video_writer.close()

def main():
    args = parse_args()
    torch.cuda.set_device(args.device)

    setup_logging()

    # 配置
    dataset_root = Path('/home/lenovo/acm_mm_dataset/tissue_dataset')
    baseline_dir = Path('/home/lenovo/acm_mm_dataset/baseline_result')
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)

    # 加载模型
    logger.info("Loading TrackOnFF model")
    args_model = Args()  # 使用 TrackOnFF 的参数
    model = TrackOnFF(args_model)
    restart_from_checkpoint_not_dist(args_model, run_variables={}, model=model)
    model.cuda().eval()
    model.set_memory_size(args_model.val_memory_size, args_model.val_memory_size)
    model.visibility_treshold = args_model.val_vis_delta

    # 如果指定 index，仅使用对应子目录
    if args.index != -1:
        subfolders = sorted([f for f in os.listdir(dataset_root) if os.path.isdir(dataset_root / f)])
        if args.index >= len(subfolders):
            raise ValueError(f"Index {args.index} out of range. Found {len(subfolders)} folders.")
        dataset_path = dataset_root / subfolders[args.index]
        logger.info(f"Only processing: {dataset_path}")
    else:
        dataset_path = dataset_root
        logger.info(f"Processing ALL folders under: {dataset_root}")

    # 加载数据
    loader = get_dataloader(dataset_path, batch_size=1)
    logger.info(f"Found {len(loader)} sequences")
    
    evaluator = TrackingEvaluator()

    # 对每个序列进行评估
    for batch in tqdm(loader, desc="Processing sequences"):
        # 运行追踪
        seq_info = batch['seq_info']
        case = str(seq_info['case'][0] if isinstance(seq_info['case'], list) else seq_info['case'])
        side = str(seq_info['side'][0] if isinstance(seq_info['side'], list) else seq_info['side'])
        seq = str(seq_info['seq'][0] if isinstance(seq_info['seq'], list) else seq_info['seq'])
        
        # 检查是否已经存在结果文件
        save_path = baseline_dir / case / side / seq
        result_file = save_path / f"{args.model_name}_result.json"
        # if result_file.exists():
        #     logger.info(f"Skipping {case}/{side}/{seq} as results already exist.")
        #     continue
        results, seq_info, scale = evaluate_sequence(model, batch)  
        scale_w, scale_h = scale  # 获取缩放比例

        # 计算误差
        video = batch['video'].squeeze(0)
        gt_points = batch['points'].squeeze(0)
        visibility = batch['visibility'].squeeze(0)
        
        original_results = []
        for coords, occlusions in results:
            coords[:, 0] /= scale_w  # x 坐标还原
            coords[:, 1] /= scale_h  # y 坐标还原
            original_results.append((coords, occlusions))
        
        original_gt_points = gt_points.clone()
        original_gt_points[..., 0] /= scale_w  # x 坐标还原
        original_gt_points[..., 1] /= scale_h  # y 坐标还原
        
        # 使用还原后的结果计算误差
        sequence_result = evaluator.compute_sequence_error(
            original_results, 
            original_gt_points, 
            visibility, 
            seq_info
        )
        
        save_sequence_results(results, sequence_result, save_path, args.model_name)
        
        logger.info(f"Sequence {case}/{side}/{seq}")
        
        del batch, results, sequence_result, video, gt_points, visibility
        torch.cuda.empty_cache()

        # logger.info(f"Mean tracking error: {sequence_result['sequence_metrics']['mean_error']:.2f} pixels")
        
        # 可视化（如果需要）
        # out_path = output_dir / f"{seq_info['case']}_{seq_info['side']}_{seq_info['seq']}_tracked.mp4"
        # visualize_results(video, results, gt_points, visibility, out_path)
    
    # 计算并保存总体结果

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=4, help='Folder index to process, -1 means all')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--model_name', default="trackon_offline")
    return parser.parse_args()

if __name__ == '__main__':
    main()