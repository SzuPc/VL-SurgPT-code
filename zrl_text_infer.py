import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging
import numpy as np
from model.track_on_ff import TrackOnFF
from model.track_on_text import Text_guidance_track
from load_dataset import get_dataloader
from utils.train_utils import restart_from_checkpoint_not_dist

from transformers import CLIPTokenizer, CLIPTextModel

from model.loss_tg import point_loss, status_cls_loss, kalman_trajectory_supervision, smoothness_self_supervision
STATUS_CLASS_NAMES = [
    "Pulled", "Clear view", "Reflection", "Obscured by tissue",
    "Obscured by smoke", "Obscured by instrument", "Off camera"
]
status_name_to_idx = {name: i for i, name in enumerate(STATUS_CLASS_NAMES)}

def encode_status_classes(clip_tokenizer, clip_model, device):
    tokens = clip_tokenizer(STATUS_CLASS_NAMES, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model(**tokens).last_hidden_state[:, 0]  # (7, D)
    return embeddings  # Tensor[7, D]

def map_status_to_embeddings(filled_status_texts, status_class_embeddings, name_to_idx, device):
    """
    输入:
        filled_status_texts: List[List[List[str]]] → shape (N, T, 1)
        status_class_embeddings: Tensor (7, D)
    返回:
        Tensor: (N, T, D) 对应每个点、每帧的状态嵌入向量
    """
    N = len(filled_status_texts)
    T = len(filled_status_texts[0])
    D = status_class_embeddings.shape[1]

    embeddings = torch.zeros(N, T, D, device=device)

    for i in range(N):
        for t in range(T):
            raw_status = filled_status_texts[i][t]
            if isinstance(raw_status, list):
                status_str = raw_status[0] if len(raw_status) > 0 else ''
            else:
                status_str = raw_status

            idx = name_to_idx.get(status_str, -1)
            if idx >= 0:
                embeddings[i, t] = status_class_embeddings[idx]
            else:
                embeddings[i, t] = torch.zeros(D, device=device)

    return embeddings

def encode_status_classes(clip_tokenizer, clip_model, device):
    tokens = clip_tokenizer(STATUS_CLASS_NAMES, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = clip_model(**tokens).last_hidden_state[:, 0]  # (7, D)
    return embeddings 


def map_status_to_embeddings(filled_status_texts, status_class_embeddings, name_to_idx, device):
    """
    输入:
        filled_status_texts: List[List[str or list]] → shape (N, T)
        status_class_embeddings: Tensor (7, D)
    返回:
        Tensor: (N, T, D) 对应每个点、每帧的状态嵌入向量
    """
    N = len(filled_status_texts)
    T = len(filled_status_texts[0])
    D = status_class_embeddings.shape[1]

    embeddings = torch.zeros(N, T, D, device=device)

    for i in range(N):
        for t in range(T):
            raw = filled_status_texts[i][t]

            # 解包 list 类型值
            if isinstance(raw, list):
                status_str = raw[0] if len(raw) > 0 else ""
            else:
                status_str = raw

            # 保证是 str 类型后进行查找
            if not isinstance(status_str, str):
                status_str = str(status_str)

            idx = name_to_idx.get(status_str, -1)
            if idx >= 0:
                embeddings[i, t] = status_class_embeddings[idx]
            else:
                embeddings[i, t] = torch.zeros(D, device=device)

    return embeddings

def safe_collate(batch):
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger("StatusTrain")

from collections import defaultdict

def compute_status_accuracy(pred_all, gt_all, num_classes):
    """
    计算总准确率以及每类状态标签的准确率。
    输入:
        pred_all: Tensor [N_total]
        gt_all: Tensor [N_total]
    返回:
        total_acc: float
        per_class_acc: dict[class_index] = accuracy
        per_class_count: dict[class_index] = total_count
    """
    assert pred_all.shape == gt_all.shape
    correct = (pred_all == gt_all)

    total_acc = correct.float().mean().item()

    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    for i in range(len(gt_all)):
        cls = gt_all[i].item()
        per_class_total[cls] += 1
        if pred_all[i] == gt_all[i]:
            per_class_correct[cls] += 1

    per_class_acc = {
        cls: per_class_correct[cls] / per_class_total[cls]
        for cls in per_class_total
    }

    return total_acc, per_class_acc, per_class_total



def inference():
    logger = setup_logger()

    # ==== 配置 ====
    dataset_root = Path("/home/lenovo/acm_mm_dataset/tissue_text_guidance_dataset/test")
    checkpoint_path = "/home/lenovo/acm_mm_dataset/track_on/checkpoints/track_on_checkpoint.pt"

    # ==== 参数类 ====
    class Args:
        def __init__(self):
            self.input_size = [384, 512]
            self.N = 384
            self.T = 18
            self.stride = 4
            self.transformer_embedding_dim = 256
            self.cnn_corr = False
            self.linear_visibility = False
            self.val_memory_size = 96
            self.val_vis_delta = 0.9
            self.checkpoint_path = checkpoint_path
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
            self.lr = 1e-5
            self.wd = 1e-4
            self.bs = 1
            self.gradient_acc_steps = 1
            self.validation = False
            self.checkpoint_path = "/home/lenovo/acm_mm_dataset/track_on/checkpoints/track_on_checkpoint.pt"
            self.seed = 1234
            self.loss_after_query = True
            self.folder_index = -1
            self.cuda_device = 0

    args = Args()
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    # ==== 加载冻结 TrackOn ====
    logger.info("Loading frozen TrackOnFF")
    trackon = TrackOnFF(args).to(device)
    restart_from_checkpoint_not_dist(args, run_variables={}, model=trackon)
    trackon.eval()
    trackon.set_memory_size(args.memory_size, args.memory_size)
    for p in trackon.parameters():
        p.requires_grad = False

    # ==== 初始化 StatusAwareTrackerHead ====
    logger.info("Creating StatusAwareTrackerHead")
    train_model = Text_guidance_track(args).to(device)

    tg_ckpt_path = "/home/lenovo/acm_mm_dataset/track_on_text/save_text_checkpoints/text_guidance_checkpoint_epoch_10.pt"
    ckpt = torch.load(tg_ckpt_path, map_location=device)
    train_model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Loaded Text_guidance_track checkpoint from: {tg_ckpt_path}")

    # === 设置为推理模式 ===
    train_model.eval()
    
    
    # ==== 加载 CLIP 模型 ====
    logger.info("Loading CLIP model")
    local_path = "/home/lenovo/acm_mm_dataset/track_on_text/clip_pretrained/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

    clip_tokenizer = CLIPTokenizer.from_pretrained(local_path)
    clip_model = CLIPTextModel.from_pretrained(local_path).to(device).eval()
    status_class_embeddings = encode_status_classes(clip_tokenizer, clip_model, device)


    # ==== 数据加载 ===
    logger.info("Loading dataset")
    loader = get_dataloader(dataset_root, batch_size=1, folder_index=args.folder_index, collate_fn=safe_collate)

    logger.info("Starting inference")
    from computer_metrics import TrackingEvaluator, save_sequence_results  # 确保你有这些工具

    evaluator = TrackingEvaluator()
    num = 0
    with torch.no_grad():
        status_pred_all = []
        status_gt_all = []

        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None or batch['video'] is None:
                continue  # 跳过无效样本

            # === 基本信息 ===
            video = batch['video'].squeeze(0).to(device)         # (T, C, H, W)
            gt_points = batch['points'].squeeze(0).to(device)    # (N, T, 2)
            visibility = batch['visibility'].squeeze(0)          # (N, T)
            status_texts = batch['status_texts']
            scale = batch['scale']
            seq_info = batch['seq_info']
            T = video.shape[0]

            # === TrackOn 初始化 ===
            queries = gt_points[:, 0, :]  # 初始点
            trackon.init_queries_and_memory(queries, video[0].unsqueeze(0))

            # === 计算 status embedding ===
            status_embed = map_status_to_embeddings(status_texts, status_class_embeddings, status_name_to_idx, device=device)  # (N, T, D)

            results = []  # 保存每帧推理结果
            results.append((queries.cpu().numpy(), np.zeros(queries.shape[0], dtype=bool)))  # 初始点，视为“预测”

            for t in tqdm(range(1, T), desc="Processing frames", leave=False):
                frame = video[t].unsqueeze(0)  # [1, C, H, W]
                frame_out = trackon.ff_forward(frame)

                q_t = frame_out["q_t"]
                q_t_corr = frame_out["q_t_corr"]
                h_t = frame_out["h_t"]
                p_patch_t = frame_out["p_head_patch_t"]
                coord_pred = frame_out["coord_pred"]  # (1, N, 2)

                output = train_model(
                    q_t=q_t,
                    h_t=h_t,
                    q_t_corr=q_t_corr,
                    p_head_t=p_patch_t,
                    coord_pred=coord_pred,
                    is_training=True,
                    status_class_embeddings=status_class_embeddings
                )
                # === 收集 status prediction ===
                # === 计算状态分类准确率（总 + 每类）===

                # === 收集 status prediction ===
                logits = output["status_logits"][0]                       # (N, num_classes)
                pred_ids = torch.argmax(logits, dim=-1)                   # (N,)

                # 构造 GT 标签（按时间 t 获取）
                gt_idx_all = torch.full((pred_ids.shape[0],), fill_value=-1, dtype=torch.long)
                for i in range(pred_ids.shape[0]):
                    raw = status_texts[i][t]
                    if isinstance(raw, list):
                        status_str = raw[0] if len(raw) > 0 else ""
                    else:
                        status_str = raw
                    if not isinstance(status_str, str):
                        status_str = str(status_str)
                    gt_idx_all[i] = status_name_to_idx.get(status_str, -1)

                valid_mask = gt_idx_all >= 0
                status_pred_all.append(pred_ids[valid_mask])
                status_gt_all.append(gt_idx_all[valid_mask])




                # === 收集预测结果 ===
                coords = output["correct_pred"][0]  # [N, 2]
                results.append((coords.cpu().numpy(), np.zeros(coords.shape[0], dtype=bool)))  # 没有显式 occlusion 信息

            # === 坐标还原 ===
            scale_w, scale_h = scale
            for i in range(len(results)):
                results[i][0][:, 0] /= scale_w
                results[i][0][:, 1] /= scale_h
                
            scale_w = torch.tensor(scale[0], device=gt_points.device)
            scale_h = torch.tensor(scale[1], device=gt_points.device)


            gt_points[..., 0] /= scale_w
            gt_points[..., 1] /= scale_h

            # === 误差评估 ===
            sequence_result = evaluator.compute_sequence_error(
                results, gt_points.cpu(), visibility.cpu(), seq_info
            )

            # === 保存结果 ===
            def unpack_seq_info_field(x):
                return str(x[0]) if isinstance(x, list) else str(x)

            case = unpack_seq_info_field(seq_info["case"])
            side = unpack_seq_info_field(seq_info["side"])
            seq = unpack_seq_info_field(seq_info["seq"])


            save_path = Path("/home/lenovo/acm_mm_dataset/text_guidance_result") / case / side / seq
            save_sequence_results(results, sequence_result, save_path, model_name="track_on_text_guidance_longclip_train")
            
            # num = num + 1
            # if num == 10:
            #     break
        # === 推理结束后，统计总准确率和每类准确率 ===
        if len(status_pred_all) > 0:
            device = status_pred_all[0].device
            status_pred_all = [x.to(device) for x in status_pred_all]
            status_gt_all = [x.to(device) for x in status_gt_all]

            status_pred_all = torch.cat(status_pred_all, dim=0)
            status_gt_all = torch.cat(status_gt_all, dim=0)

            total_acc, per_class_acc, per_class_total = compute_status_accuracy(
                status_pred_all, status_gt_all, num_classes=len(STATUS_CLASS_NAMES)
            )

            logger.info(f"===> [Status Accuracy] Overall: {total_acc:.4f}")
            for idx, name in enumerate(STATUS_CLASS_NAMES):
                if idx in per_class_acc:
                    acc = per_class_acc[idx]
                    count = per_class_total[idx]
                    logger.info(f"    - {name:20s}: {acc:.4f} over {count:4d} samples")
                else:
                    logger.info(f"    - {name:20s}: No valid samples")
        else:
            logger.warning("No valid status labels found for accuracy computation.")





if __name__ == "__main__":
    inference()


