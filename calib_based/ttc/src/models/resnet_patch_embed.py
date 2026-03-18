import torch.nn as nn


class ResNetBlock1D(nn.Module):
    """Single 1D ResNet block"""
    def __init__(self, in_ch=2, mid_ch=256, expand=2, kernel_size=5):
        super().__init__()
        out_ch = mid_ch * expand
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, mid_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(mid_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(mid_ch, out_ch, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.short = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return self.relu(out)


class ResNetPatchEmbed(nn.Module):
    """ResNet block + Conv1d patch projection"""
    def __init__(self, in_ch=2, embed_dim=512, patch_size=30, patch_stride=15):
        super().__init__()
        self.res = ResNetBlock1D(in_ch=in_ch, mid_ch=256, expand=2, kernel_size=5)
        self.proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat = self.res(x)
        tok = self.proj(feat)
        tok = tok.transpose(1, 2)
        return self.norm(tok)
