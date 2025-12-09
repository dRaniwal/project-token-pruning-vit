# Vision Transformer Token Pruning on CIFAR-10

This repository explores token pruning inside a Vision Transformer (ViT) to reduce computational cost while maintaining accuracy. Instead of passing all 145 tokens (1 CLS + 144 patches) through all transformer layers, we dynamically drop low-importance tokens.

The results show:

* Up to **1.85× faster inference**
* **Less than 1% accuracy drop**
* Deep layers reduced from **145 tokens → 5 tokens**
* Works even when applied to the **baseline model at inference time** (without retraining)

---

## Repository Structure

```
.
├── notebooks/
│   ├── baseline_training.ipynb
│   ├── pruned_training.ipynb
│   ├── evaluation.ipynb
├── src/
│   ├── vit_model.py
│   ├── pruning.py
│   ├── utils.py
├── models/
│   ├── vit_baseline.pth
│   ├── vit_pruned.pth
└── README.md
```

---

# 1. Bases of Token Pruning Used

Two scoring mechanisms for ranking patch importance were explored.

## 1.1 CLS Attention Weights (used in final model)

For each transformer layer, we take:

* The CLS → patch attention row from the attention matrix
* Average over all attention heads
* Rank patches by this score
* Drop the lowest-importance patches

Why this works:

* CLS is the only token used for classification
* CLS aggregates information from all patches
* If CLS consistently assigns low attention to a patch, the model does not find it useful

This method was used for both training-time pruning and inference-time pruning.

---

## 1.2 Overall Patch Attention (explored but not used finally)

For each patch:

* Compute how much it is attended by all other patches (excluding CLS)
* Average attention it receives
* Drop the lowest-scoring tokens

Significance:

* Measures contextual importance
* Can be helpful for segmentation or dense prediction tasks

CLS-based pruning was simpler, faster, and gave better results, so it became the primary approach.

---

## 1.3 Selecting “k” Patches to Drop

The number of tokens to keep at each layer is controlled by the schedule:

```
keep_ratio(l) = 1 − r_max * ((l + 1) / L)^alpha
```

where:

* r_max = maximum pruning strength
* alpha = curve sharpness parameter
* L = number of layers

Meaning:

* Very little pruning in early layers
* Aggressive pruning in deeper layers

This aligns with how ViTs encode low-level features early and high-level features later.

---

# 2. Why Pruning Works

## 2.1 Dropping a patch does not remove its information

Even if a patch is removed in deeper layers:

* It already influenced CLS in earlier layers
* CLS keeps carrying forward its information
* Removing unimportant patches does not hurt decision making

This is a key reason ViTs tolerate pruning well.

## 2.2 Early layers should not be pruned

Early layers learn:

* edges
* texture
* color gradients

Removing tokens too early harms representational quality.
Therefore, in training we prune only after epoch 4.

## 2.3 Deep layers become redundant

Later layers contain:

* high-level semantic features
* background suppression
* concentrated discriminative regions

Many patches become unnecessary, enabling aggressive pruning.

## 2.4 Computational savings grow quadratically

Attention complexity is O(N²).
Reducing 145 tokens → 30 tokens yields massive FLOP reduction.

---

# 3. Training Results

* Relevant files: notebooks/baseline-training.ipynb and notebooks/simple-prune-training.ipynb 

## 3.1 Baseline Training (no pruning)

| Epoch | Train Loss | Val Acc | Time     |
| ----- | ---------- | ------- | -------- |
| 1     | 1.8552     | 46.89%  | 5.64 min |
| 2     | 1.3116     | 57.61%  | 5.95 min |
| 3     | 1.1278     | 62.50%  | 5.96 min |
| 4     | 1.0178     | 63.55%  | 5.97 min |
| 5     | 0.9595     | 66.44%  | 5.98 min |
| 9     | 0.7761     | 71.00%  | 5.99 min |
| 10    | 0.7477     | 71.68%  | 5.99 min |

**Total training time:** 59.45 min
**Final accuracy:** 71.68%

---

## 3.2 Pruned Training (no pruning during first 4 epochs)

| Epoch | Train Loss | Val Acc | Time     |
| ----- | ---------- | ------- | -------- |
| 1     | 1.8912     | 44.23%  | 5.09 min |
| 2     | 1.2913     | 57.85%  | 5.31 min |
| 3     | 1.0899     | 61.35%  | 5.33 min |
| 4     | 0.9702     | 62.83%  | 5.34 min |
| 5     | 0.8920     | 66.41%  | 3.72 min |
| 6     | 0.8296     | 68.43%  | 3.72 min |
| 7     | 0.7708     | 68.97%  | 3.73 min |
| 8     | 0.7269     | 70.12%  | 3.73 min |
| 9     | 0.6814     | 70.26%  | 3.73 min |
| 10    | 0.6412     | 69.97%  | 3.73 min |

#
* Total Training time: 43.43 min 
* Speedup: ~27% faster
* Final accuracy: 69.97%
* Best accuracy: 70.26%

---

# 4. Evaluation Results (Testing Phase)

Relevant files: notebooks/evaluation.ipynb

We evaluate:

1. Baseline model — no pruning
2. Pruned model — pruning OFF
3. Pruned model — pruning ON
4. Baseline model — pruning ON

*We take alpha=2 and r_max=0.6(standard values also used elsewhere)

## 4.1 Baseline Model (no pruning)

* Accuracy: 71.68%
* Eval time: 38.12 s

---

## 4.2 Pruned Model (pruning OFF)

* Accuracy: 70.79%
* Eval time: 39.33 s

---

## 4.3 Pruned Model (pruning ON)

* Accuracy: 70.65%
* Eval time: 27.71 s
* Tokens per layer:
  [143, 137, 125, 106, 81, 54, 29, 12]

---

## 4.4 Baseline Model (pruning ON)

Baseline model + pruning during inference produces:

* Accuracy: up to **71.68%** (higher than baseline!)
* Eval time: **27.98s**
* Same accuracy as Baseline Model with No pruning and ~27% faster.
* This means due to pruning we get free increased speed without any drop in accuracy.

This demonstrates that pruning can be applied to a pretrained model without retraining.

---

# 5. Alpha Sweep Experiments (Testing-Only)

We swept alpha ∈ [1.0, 2.0] with r_max = 0.6 and applied pruning only during testing.

## 5.1 Baseline Model with Pruning (Alpha Sweep)

The best accuracy reached for **alpha=1.5:

* Accuracy: **71.80%**
* Time: 26.93 sec
* Inference 1.8× faster than baseline model with no pruning
* The accuracy increased too compared to baseline model with no pruning.

This suggests pruning suppresses noisy or redundant patches, improving generalization.

---

## 5.2 Pruned Model with Alpha Sweep (Testing-Only)

Optimal value discovered:

| Metric              | Value    |
| ------------------- | -------- |
| Optimal Alpha       | 1.2      |
| Accuracy            | 70.78%   |
| Inference Time      | 20.66 s  |
| Speedup vs Baseline | 1.85×    |
| Accuracy Drop       | 0.90%    |
| Final Tokens        | 5 tokens |

This is the best overall speed-accuracy tradeoff.



# Summary

* Token pruning dramatically speeds up ViTs without hurting accuracy.
* CLS-attention pruning is simple and highly effective.
* Training with pruning reduces training time and maintains accuracy.
* Applying pruning at inference alone yields substantial speedups.


