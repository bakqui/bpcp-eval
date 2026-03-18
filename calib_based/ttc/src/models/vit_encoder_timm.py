import torch.nn as nn
from timm.models.vision_transformer import Block


class ViTEncoder1D(nn.Module):
    def __init__(self, embed_dim=512, depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()
        self.pos_drop = nn.Dropout(p=drop)
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x
