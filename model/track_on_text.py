import torch
import torch.nn as nn
import torch.nn.functional as F

from model.status_head import Status_ClassifierHead
from model.text_guidance_head import StatusGuidedOffsetHead

class Text_guidance_track(nn.Module):
    def __init__(self, args, num_classes=7, text_embed_dim=512, nhead=8):
        super().__init__()

        self.num_classes = num_classes
        self.classifier_head = Status_ClassifierHead(args, num_classes=num_classes, nhead=nhead)
        self.offset_head = StatusGuidedOffsetHead(args, nhead=nhead, text_embed_dim=text_embed_dim)

        self.status_embed_mapper = nn.Sequential(
            nn.Linear(num_classes, text_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(text_embed_dim)
        )

    def forward(self, q_t, h_t, q_t_corr ,p_head_t, gt_status_embed=None, is_training=True, coord_pred=None, status_class_embeddings=None):
        """
        q_t: (B, N, C)
        h_t: (B, P, C)
        p_head_t: (B, N, 2)
        gt_status_label: (B, N) or None
        is_training: bool
        """
        
        # Step 1: always predict status logits
        status_logits = self.classifier_head(q_t, h_t, p_head_t)         # (B, N, num_classes)

        if is_training and gt_status_embed is not None:
            # Step 2a: use GT label -> one-hot -> embedding
            status_embed = gt_status_embed  # 已经是 embedding，形状 (B, N, 512)
        else:
            # Step 2b: use predicted logits
            pred_ids = torch.argmax(status_logits, dim=-1)             # (B, N)
            B, N = pred_ids.shape
            D = status_class_embeddings.shape[1]
            status_embed = torch.zeros(B, N, D, device=pred_ids.device)

            for b in range(B):
                for n in range(N):
                    status_embed[b, n] = status_class_embeddings[pred_ids[b, n]]                  # (B, N, text_embed_dim)

        # Step 3: Offset prediction
        offsets = self.offset_head(q_t_corr, h_t, p_head_t, status_embedding=status_embed)    # (B, L, N, 2)
        correct_pred = offsets[:, -1]  + coord_pred
        
        return {
            "status_logits": status_logits,
            "correct_pred": correct_pred
        }
