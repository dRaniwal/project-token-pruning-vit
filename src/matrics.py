# src/metrics.py
#Token count / speed benchmarking functions
import torch
import time
from tqdm import tqdm

@torch.no_grad()
def count_tokens(model, loader):
    token_list = []

    for x, _ in tqdm(loader, desc="Token Count", leave=False):
        x = x.to(next(model.parameters()).device)
        model(x)

        if hasattr(model, "m"):
            N = x.size(1)
            token_list.append(N)

    return sum(token_list) / len(token_list)


@torch.no_grad()
def measure_throughput(model, loader, iters=30):
    device = next(model.parameters()).device
    imgs = 0
    t0 = time.time()

    for i, (x, _) in enumerate(loader):
        if i >= iters:
            break
        x = x.to(device)
        model(x)
        imgs += x.size(0)

    t1 = time.time()
    return imgs / (t1 - t0)
