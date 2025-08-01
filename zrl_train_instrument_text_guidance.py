import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging

from model.track_on_ff import TrackOnFF
from model.track_on_text import Text_guidance_track
from load_dataset import get_dataloader
from utils.train_utils import restart_from_checkpoint_not_dist

from transformers import CLIPTokenizer, CLIPTextModel

from model.loss_tg import point_loss, status_cls_loss, kalman_trajectory_supervision, smoothness_self_supervision
STATUS_CLASS_NAMES = [
    "Eternal occlusion", "Self occlusion", "Clear View", "Off camera"
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

def safe_collate(batch):
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)



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



def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger("StatusTrain")

def train():
    logger = setup_logger()

    # ==== 配置 ====
    dataset_root = Path("/home/lenovo/acm_mm_dataset/instrument_text_guidance_dataset/train")
    checkpoint_path = "/home/lenovo/acm_mm_dataset/track_on/checkpoints/track_on_checkpoint.pt"
    num_epochs = 20
    lr = 1e-5

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
            self.cuda_device = 1

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
    train_model = Text_guidance_track(args, num_classes=4).to(device)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=args.lr)
    
    
    # ==== 加载 CLIP 模型 ====
    logger.info("Loading CLIP model")
    local_path = "/home/lenovo/acm_mm_dataset/track_on_text/clip_pretrained/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

    clip_tokenizer = CLIPTokenizer.from_pretrained(local_path)
    clip_model = CLIPTextModel.from_pretrained(local_path).to(device).eval()
    status_class_embeddings = encode_status_classes(clip_tokenizer, clip_model, device)


    # ==== 数据加载 ===
    logger.info("Loading dataset")
    loader = get_dataloader(dataset_root, batch_size=1, folder_index=args.folder_index, max_frames=181, collate_fn=safe_collate)

    logger.info("Starting training")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        for batch in tqdm(loader, desc="Training"):
            if batch is None or batch['video'] is None:
                continue  # 跳过无效样本
            all_coord_pred = []
            all_status_pred = []
            video = batch['video'].squeeze(0).to(device)         # (T, C, H, W)
            gt_points = batch['points'].squeeze(0).to(device)  
            # (N, T, 2)
            gt_status = batch['status_texts']    # (N,)
            gt_location = batch['location_texts']  # (N, 2)


            status_embed = map_status_to_embeddings(gt_status, status_class_embeddings, status_name_to_idx, device=device)  # (N, T, D)

            scale = batch['scale']
            T = video.shape[0]
            # N = gt_status.shape[0]
            # filled_location = fill_missing_labels(gt_location)  # 同理

            # === 初始化 TrackOn 查询与记忆 ===
            queries = gt_points[:, 0, :].to(device)  # 初始点
            with torch.no_grad():
                trackon.init_queries_and_memory(queries, video[0].unsqueeze(0).to(device))

            for t in tqdm(range(T), desc=f"Processing frames (Epoch {epoch + 1})"):
                frame = video[t].unsqueeze(0)  # [1, C, H, W]

                with torch.no_grad():
                    frame_out = trackon.ff_forward(frame)
                    
                status_input_t = status_embed[:, t, :]  # (N, D)

                q_t = frame_out["q_t"]
                q_t_corr = frame_out["q_t_corr"]    
                h_t = frame_out["h_t"]
                p_patch_t = frame_out["p_head_patch_t"]
                coord_pred = frame_out["coord_pred"]  # (1, N, 2)

                # # === 前向推理 ===
                output = train_model(
                    q_t=q_t,
                    h_t=h_t,
                    q_t_corr=q_t_corr,
                    p_head_t=p_patch_t,
                    gt_status_embed=status_input_t.unsqueeze(0),
                    is_training=True,
                    coord_pred=coord_pred.unsqueeze(0), # (1, N, 2)
                )
                all_coord_pred.append(output["correct_pred"][0]) # remove T
                all_status_pred.append(output["status_logits"][0])


            loss_point = point_loss(all_coord_pred, gt_points)
            loss_statue = status_cls_loss(all_status_pred, gt_status, device=device)
            loss_smooth = smoothness_self_supervision(all_coord_pred)
            
            loss = loss_point + loss_statue + loss_smooth


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"[Epoch {epoch+1}] Point_loss: {loss_point.item():.4f} Statue_loss: {loss_statue.item():.4f} Smooth_loss: {loss_smooth.item():.4f} Loss: {loss.item():.4f}")

        # === 每个 epoch 结束后保存模型 ===
        ckpt_dir = Path("/home/lenovo/acm_mm_dataset/track_on_text/checkpoint_instrument")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"text_guidance_checkpoint_epoch_{epoch + 1}.pt"

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': train_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)

        logger.info(f"Checkpoint saved at: {ckpt_path}")



if __name__ == "__main__":
    train()


