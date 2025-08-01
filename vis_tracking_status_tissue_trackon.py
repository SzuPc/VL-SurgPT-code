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
from collections import defaultdict
import json
from transformers import CLIPTokenizer, CLIPTextModel

from model.loss_tg import point_loss, status_cls_loss, smoothness_self_supervision
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

def inference():
    logger = setup_logger()

    # ==== 配置 ====
    dataset_root = Path("/home/lenovo/acm_mm_dataset/tissue_text_guidance_dataset/test")
    checkpoint_path = "/home/lenovo/acm_mm_dataset/track_on/checkpoints/track_on_checkpoint.pt"

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
            self.val_vis_delta = 0.5
            self.checkpoint_path = checkpoint_path
            self.num_layers = 3
            self.num_layers_offset_head = 3
            self.num_layers_rerank = 3
            self.num_layers_rerank_fusion = 1
            self.top_k_regions = 16
            self.num_layers_spatial_writer = 3
            self.num_layers_spatial_self = 1
            self.num_layers_spatial_cross = 1
            self.memory_size = 12
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
            self.checkpoint_path = checkpoint_path
            self.seed = 1234
            self.loss_after_query = True
            self.folder_index = -1
            self.cuda_device = 1

    args = Args()
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    logger.info("Loading frozen TrackOnFF")
    trackon = TrackOnFF(args).to(device)
    restart_from_checkpoint_not_dist(args, run_variables={}, model=trackon)
    trackon.eval()
    trackon.set_memory_size(args.memory_size, args.memory_size)
    for p in trackon.parameters():
        p.requires_grad = False

    logger.info("Creating StatusAwareTrackerHead")
    train_model = Text_guidance_track(args, num_classes=7).to(device)

    tg_ckpt_path = "/home/lenovo/acm_mm_dataset/track_on_text/save_text_checkpoints/text_guidance_shortclip_checkpoint_epoch_10.pt"
    ckpt = torch.load(tg_ckpt_path, map_location=device)
    train_model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Loaded Text_guidance_track checkpoint from: {tg_ckpt_path}")
    train_model.eval()
    
    logger.info("Loading CLIP model")
    local_path = "/home/lenovo/acm_mm_dataset/track_on_text/clip_pretrained/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    clip_tokenizer = CLIPTokenizer.from_pretrained(local_path)
    clip_model = CLIPTextModel.from_pretrained(local_path).to(device).eval()
    status_class_embeddings = encode_status_classes(clip_tokenizer, clip_model, device)

    logger.info("Loading dataset")
    loader = get_dataloader(dataset_root, batch_size=1, folder_index=args.folder_index, collate_fn=safe_collate)

    logger.info("Starting inference")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None or batch['video'] is None:
                continue

            video = batch['video'].squeeze(0).to(device)         # (T, C, H, W)
            gt_points = batch['points'].squeeze(0).to(device)    # (N, T, 2)
            status_texts = batch['status_texts']
            scale = batch['scale']
            seq_info = batch['seq_info']
            T = video.shape[0]

            queries = gt_points[:, 0, :]
            trackon.init_queries_and_memory(queries, video[0].unsqueeze(0))

            status_embed = map_status_to_embeddings(status_texts, status_class_embeddings, status_name_to_idx, device=device)

            all_frame_outputs = []  # 只保存预测坐标和状态

            for t in tqdm(range(1, T), desc="Processing frames", leave=False):
                frame = video[t].unsqueeze(0)
                frame_out = trackon.ff_forward(frame)

                q_t = frame_out["q_t"]
                q_t_corr = frame_out["q_t_corr"]
                h_t = frame_out["h_t"]
                p_patch_t = frame_out["p_head_patch_t"]
                coord_pred = frame_out["coord_pred"]

                # output = train_model(
                #     q_t=q_t,
                #     h_t=h_t,
                #     q_t_corr=q_t_corr,
                #     p_head_t=p_patch_t,
                #     coord_pred=coord_pred,
                #     is_training=True,
                #     status_class_embeddings=status_class_embeddings
                # )

                coords = frame_out["coord_pred"].cpu().numpy()  # (N, 2)
                # status_ids = torch.argmax(output["status_logits"][0], dim=-1).cpu().numpy()
                # status_names = [STATUS_CLASS_NAMES[i] for i in status_ids]
                # visibility = [s in ("Clear view", "Self occlusion") for s in status_names]

                # 恢复坐标比例
                coords[:, 0] /= scale[0]
                coords[:, 1] /= scale[1]

                all_frame_outputs.append({
                    "frame_index": t,
                    "coordinates": coords.tolist(),
                    # "status_preds": status_names,
                    # "visibility_preds": visibility
                })

            def unpack_seq_info_field(x):
                return str(x[0]) if isinstance(x, list) else str(x)

            case = unpack_seq_info_field(seq_info["case"])
            side = unpack_seq_info_field(seq_info["side"])
            seq = unpack_seq_info_field(seq_info["seq"])

            save_path = Path("/home/lenovo/acm_mm_dataset/vis_track_status/tissue") / case / side / seq
            save_path.mkdir(parents=True, exist_ok=True)

            with open(save_path / "trackon_point_and_status.json", "w") as f:
                json.dump(all_frame_outputs, f, indent=2)

            logger.info(f"Saved results to {save_path / 'trackon_point_and_status.json'}")



if __name__ == "__main__":
    inference()


