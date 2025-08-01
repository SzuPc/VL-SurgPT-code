import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from filterpy.kalman import KalmanFilter

import torch.nn.functional as F

def point_loss(pred_list, gt_points, delta=1.0):
    """
    使用 Huber 损失计算点误差（适用于稀疏 GT）。
    
    pred_list: list of (N, 2) tensors, length = T
    gt_points: (N, T, 2)
    delta: Huber 损失的平滑区间
    """
    T = len(pred_list)
    N = gt_points.shape[0]
    device = gt_points.device

    gt_points = gt_points.transpose(0, 1)  # (T, N, 2)

    total_loss = 0.0
    valid_count = 0

    for t in range(1, T):
        gt = gt_points[t]      # (N, 2)
        pred = pred_list[t]    # (N, 2)

        valid_mask = (gt != -1).all(dim=-1)  # (N,)
        if valid_mask.any():
            diff = pred[valid_mask] - gt[valid_mask]  # (M, 2)
            error = F.huber_loss(diff, torch.zeros_like(diff), delta=delta, reduction='sum')
            total_loss += error
            valid_count += valid_mask.sum()

    if valid_count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / valid_count


def status_cls_loss(all_status_pred, gt_status, device):
    import torch.nn.functional as F

    STATUS_CLASS_NAMES = [
    "Eternal occlusion", "Self occlusion", "Clear View", "Off camera"
    ]
    status_name_to_idx = {name: i for i, name in enumerate(STATUS_CLASS_NAMES)}

    N = len(gt_status)
    T = len(gt_status[0])
    num_classes = all_status_pred[0].shape[1]

    # Stack predictions into (N, T, num_classes)
    all_status_pred_tensor = torch.stack(all_status_pred, dim=1)  # (N, T, C)

    # Prepare label tensor (N, T)
    gt_status_labels = torch.full((N, T), -1, dtype=torch.long, device=device)
    for i in range(N):
        for t in range(T):
            label_raw = gt_status[i][t]
            label_str = label_raw[0] if isinstance(label_raw, list) else label_raw
            label_idx = status_name_to_idx.get(label_str, -1)
            gt_status_labels[i, t] = label_idx

    # Flatten for loss computation
    pred_flat = all_status_pred_tensor.view(N * T, num_classes)
    labels_flat = gt_status_labels.view(N * T)
    valid_mask = labels_flat != -1

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss = F.cross_entropy(
        pred_flat[valid_mask],           # (M, C)
        labels_flat[valid_mask],         # (M,)
        reduction="mean"
    )

    return loss




def smoothness_self_supervision(all_coord_pred, reduction='mean'):
    pred_traj = torch.stack(all_coord_pred, dim=1)  # (N, T, 2)
    N, T, _ = pred_traj.shape

    diffs = pred_traj[:, 1:] - pred_traj[:, :-1]           # (N, T-1, 2)
    acc = diffs[:, 1:] - diffs[:, :-1]                     # (N, T-2, 2)

    # 使用 Huber Loss 更稳健
    loss = F.smooth_l1_loss(acc, torch.zeros_like(acc), reduction=reduction)
    return loss
