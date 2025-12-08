# src/pruning.py
import torch
import torch.nn as nn

class SimplePruner:
    def __init__(self, r_max=0.6, alpha=2):
        self.r_max = r_max
        self.alpha = alpha

    def compute_keep(self, score, layer, L):
        drop = self.r_max * ((layer + 1) / L) ** self.alpha
        N = score.size(0)

        keep = max(1, int(N * (1 - drop)))
        keep = min(keep, N)

        _, idx = torch.topk(score, keep)
        return idx.sort().values


class PrunedViT(nn.Module):
    def __init__(self, model, r_max=0.6, alpha=2):
        super().__init__()
        self.m = model
        self.pruner = SimplePruner(r_max, alpha)
        self.L = len(model.blocks)

    def forward(self, x):
        B = x.size(0)
        patches = self.m.patch_embed(x)
        cls = self.m.cls.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = x + self.m.pos[:, :x.size(1)]

        for l, blk in enumerate(self.m.blocks):
            attn = blk.attn.last_attn   # shape: [B,H,N,N]

            if attn is None:
                x = blk(x)
                continue

            attn_mean = attn.mean(1)[:, 0, 1:]   # CLS → patch tokens
            score = attn_mean.mean(0)            # [num_patches]

            keep_idx = self.pruner.compute_keep(score, l, self.L)
            keep_idx = keep_idx + 1              # shift for CLS

            final_idx = torch.cat([torch.tensor([0], device=x.device), keep_idx])
            x = x[:, final_idx]

            x = blk(x)

        x = self.m.norm(x)
        return self.m.fc(x[:, 0])
