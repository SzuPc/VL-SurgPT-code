import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from model.modules import DMSMHA_Block, get_deformable_inputs

class Status_ClassifierHead(nn.Module):
    def __init__(self, args, num_classes=7, nhead=8):
        super().__init__()

        self.size = args.input_size
        self.stride = args.stride


        self.embedding_dim = args.transformer_embedding_dim
        self.num_level = 4

        self.status_decoder = DMSMHA_Block(self.embedding_dim, nhead, self.num_level)
        self.status_proj = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, num_classes)
        )

    def forward(self, q_t, h_t, p_head_t):
        """
        q_t:      (B, N, C)       - query features
        h_t:      (B, P, C)       - token features
        p_head_t: (B, N, 2)       - current point prediction in image coords
        return: logits: (B, N, num_classes)
        """

        B, N, C = q_t.shape
        device = q_t.device

        # Normalize coordinates to [0, 1]
        p_norm = p_head_t / torch.tensor([self.size[1], self.size[0]], device=device)
        p_norm = torch.clamp(p_norm, 0, 1)

        # Prepare deformable attention inputs
        f_scales, reference_points, spatial_shapes, start_levels = get_deformable_inputs(
            h_t, p_norm, self.size[0] // self.stride, self.size[1] // self.stride
        )

        # Local context
        q_local = self.status_decoder(
            q=q_t, k=f_scales, v=f_scales,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            start_levels=start_levels
        )

        # Predict class logits
        q_fused = torch.cat([q_t, q_local], dim=-1)  # (B, N, 2C)
        logits = self.status_proj(q_fused)           # (B, N, num_classes)

        return logits
