import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import DMSMHA_Block, get_deformable_inputs

class StatusGuidedOffsetHead(nn.Module):
    def __init__(self, args, nhead=8, text_embed_dim=512):
        super().__init__()

        self.nhead = nhead
        self.attn_layer_num = args.num_layers_offset_head
        self.size = args.input_size
        self.stride = args.stride

        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride
        self.P = self.H_prime * self.W_prime

        self.embedding_dim = args.transformer_embedding_dim
        self.status_cross_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=nhead, batch_first=True)

        # Project text embeddings to same dim as q_t
        self.status_mapper = nn.Linear(text_embed_dim, self.embedding_dim)
        self.fusion_mapper = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim)
        )

        self.offset_layer = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, 2),
            nn.Tanh()
        )

        self.local_transformer = nn.ModuleList([
            DMSMHA_Block(self.embedding_dim, nhead, num_levels=4)
            for _ in range(self.attn_layer_num)
        ])

    def forward(self, q_t, f_t, target_coordinates, status_embedding=None):
        """
        q_t: (B, N, C)
        f_t: (B, P, C)
        target_coordinates: (B, N, 2)
        status_embedding: (B, N, text_embed_dim) or None (in inference)
        """
        B, N, C = q_t.shape
        device = q_t.device

        # === Scale coordinates ===
        target_coordinates = target_coordinates / torch.tensor(
            [self.size[1], self.size[0]], device=device
        )
        target_coordinates = torch.clamp(target_coordinates, 0, 1)

        # === Get deformable attention inputs ===
        f_scales, reference_points, spatial_shapes, start_levels = get_deformable_inputs(
            f_t, target_coordinates, self.H_prime, self.W_prime
        )

        # === Fuse with text embedding if provided ===
        if status_embedding is not None:
            # Project CLIP embedding
            status_feat = self.status_mapper(status_embedding)  # (B, N, C)
            # Use cross-attention: q_t attends to status_feat
            q_t, _ = self.status_cross_attn(query=q_t, key=status_feat, value=status_feat)

        # === Offset regression ===
        o_t = torch.zeros(B, self.attn_layer_num, N, 2, device=device)
        for i in range(self.attn_layer_num):
            q_t = self.local_transformer[i](
                q=q_t,
                k=f_scales,
                v=f_scales,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                start_levels=start_levels
            )
            o_t[:, i] = self.offset_layer(q_t) * self.stride  # in pixel space

        return o_t
