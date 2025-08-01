# lora_model/lora.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / max(1, r)

        # 获取原始 Linear 参数（frozen）
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # 原始权重和 bias（冻结，仅做前向加权）
        self.register_buffer("weight", linear_layer.weight.data.clone())
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.clone())
        else:
            self.bias = None

        # LoRA 权重：A @ B（初始化在同一个 device 上）
        device = self.weight.device
        self.lora_A = nn.Parameter(torch.zeros((r, self.in_features), device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(x, torch.matmul(self.lora_B, self.lora_A))
        return base + self.scale * lora


def inject_lora_into_offset_head(model, r=4, alpha=1.0):
    injected = 0
    for name, module in model.named_modules():
        if 'offset_head' in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    full_name = f"{name}.{child_name}"
                    parent_module = dict(model.named_modules())[name]
                    setattr(parent_module, child_name, LoRALinear(child_module, r=r, alpha=alpha))
                    injected += 1

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 仅解冻 LoRA 参数
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True

    print(f"[LoRA] Injected into {injected} Linear layers inside offset_head.")
