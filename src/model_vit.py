# src/model_vit.py
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Basic Patch MLP ViT used by both baseline & pruning models ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=48, patch=4, in_ch=3, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)          # [B,512,12,12]
        x = x.flatten(2).transpose(1, 2)   # [B,144,512]
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.last_attn = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)

        self.last_attn = attn.detach()

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim=512, heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img=48, patch=4, dim=512, depth=8, heads=8, mlp_ratio=4, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbed(img, patch, 3, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, 1 + 144, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        patches = self.patch_embed(x)
        cls_tok = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tok, patches], dim=1)
        x = x + self.pos[:, :x.size(1)]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.fc(x[:, 0])
