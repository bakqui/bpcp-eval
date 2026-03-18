import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_encoding_1d import FixedPositionalEncoding1D
from .resnet_patch_embed import ResNetPatchEmbed
from .vit_encoder_timm import ViTEncoder1D


class TTCModel(nn.Module):
    """ResNet patch embed + fixed PE + ViT + dual heads."""
    def __init__(
        self,
        in_ch=2,
        embed_dim=512,
        patch_size=30,
        patch_stride=15,
        depth=4,
        num_heads=8,
    ):
        super().__init__()
        self.patch = ResNetPatchEmbed(in_ch, embed_dim, patch_size, patch_stride)
        self.pos = FixedPositionalEncoding1D(embed_dim, max_len=4096)
        self.enc = ViTEncoder1D(embed_dim=embed_dim, depth=depth, num_heads=num_heads)

        self.ssl_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, in_ch * patch_size)
        )
        self.sl_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        tok = self.patch(x)
        tok = self.pos(tok)
        enc = self.enc(tok)
        recon = self.ssl_head(enc)
        pooled = self.pool(enc.transpose(1, 2)).squeeze(-1)
        bp = self.sl_head(pooled)
        return recon, bp

    @staticmethod
    def unfold_1d(x, kernel_size, stride):
        x2d = x.unsqueeze(2)
        patches = F.unfold(x2d, kernel_size=(1, kernel_size), stride=(1, stride))
        B, CK, N = patches.shape
        return patches.transpose(1, 2).contiguous().view(B, N, CK)

    @staticmethod
    def fold_1d(tokens, out_T, C, kernel_size, stride):
        B, N, CK = tokens.shape
        assert CK == C * kernel_size
        x2d = tokens.view(B, N, C, kernel_size).permute(0, 2, 3, 1).contiguous()
        x2d = x2d.view(B, C * kernel_size, N)
        out = F.fold(x2d, output_size=(1, out_T), kernel_size=(1, kernel_size), stride=(1, stride))
        ones = torch.ones((B, C, out_T), device=out.device)
        norm = F.fold(
            F.unfold(ones.unsqueeze(2),
            kernel_size=(1, kernel_size),
            stride=(1, stride)),
            output_size=(1, out_T),
            kernel_size=(1, kernel_size),
            stride=(1, stride),
        )
        return (out / norm.clamp_min(1e-8)).squeeze(2)
