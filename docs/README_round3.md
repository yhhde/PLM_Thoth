# Round 3 Experiment Changes Summary

This document summarizes the key changes introduced in Round 3 experiments compared to previous rounds, focusing on training script enhancements and experiment configuration updates.

## 1. Training Script Enhancements

The new training script `5_train_model_cosine.py` introduces several improvements over the original `5_train_model.py`.

### 1.1 Cosine Learning Rate Scheduler

```python
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
```

**Significance**: Cosine annealing gradually reduces the learning rate following a cosine curve, allowing for better convergence towards the end of training. This approach is widely adopted in modern LLM pretraining (GPT-3, LLaMA) as it prevents sudden learning rate changes that can destabilize training.

### 1.2 Training Accuracy Calculation

```python
@torch.no_grad()
def compute_accuracy(logits, labels, ignore_index=-100):
    predictions = torch.argmax(logits, dim=-1)
    mask = labels != ignore_index
    correct = (predictions == labels) & mask
    return correct.sum().item() / mask.sum().item()
```

**Significance**: Token-level accuracy provides an intuitive metric for monitoring training progress. Unlike loss (which is cross-entropy), accuracy directly measures how often the model predicts the correct next token.

### 1.3 Cumulative Metrics Tracking

```python
total_loss += loss.item()
total_accuracy += batch_acc
avg_loss = total_loss / batch_count
avg_acc = total_accuracy / batch_count
```

**Significance**: Cumulative averages smooth out batch-to-batch variance, providing a clearer view of the overall training trend. This is essential for generating meaningful training curves.

### 1.4 High-Frequency Logging

Logs metrics every 10 batches (configurable via `log_interval`):

| Metric | Description |
|--------|-------------|
| `batch_loss` | Loss of current batch |
| `avg_loss` | Cumulative average loss |
| `batch_accuracy` | Accuracy of current batch |
| `avg_accuracy` | Cumulative average accuracy |
| `lr` | Current learning rate |

**Significance**: Fine-grained logging enables:
- Early detection of training instabilities
- Visualization of learning rate schedule effects
- Precise comparison of hyperparameter configurations

### 1.5 Progress Checkpoint Validation

Runs validation at 10%, 20%, ..., 100% training progress:

```python
if step == save_intervals[save_intervals_idx]:
    progress_val_loss = eval_one_epoch(model, val_loader, ...)
```

**Significance**: 
- Monitor generalization throughout training
- Early detection of overfitting
- Fair comparison of model performance at different training stages

---

## 2. Experiment Configuration Changes

### Parameter Comparison: Round 2 vs Round 3

| Parameter | Round 2 | Round 3 | Rationale |
|-----------|---------|---------|-----------|
| `lr_scheduler_type` | linear | **cosine** | Better convergence |
| `adam_betas` | [0.9, 0.999] | **[0.9, 0.95]** | LLM-optimized (GPT-3 style) |
| `mixed_precision` | false | **true** | 2x faster, lower memory |
| `learning_rate` | 1e-4 ~ 1e-3 | **1e-4 ~ 6e-4** | Refined range |
| `warmup_ratio` | 0%, 1%, 10% | **1%, 3%, 5%** | Focused on practical range |

---

## 3. Round 3 Experiments Overview

### Naming Convention

```
r3v1_lr3e4_cos_w03_fp16_b95
│ │  │     │   │   │    └── adam_betas[1] = 0.95
│ │  │     │   │   └─────── mixed_precision = true (fp16)
│ │  │     │   └─────────── warmup_ratio = 0.03 (3%)
│ │  │     └─────────────── lr_scheduler_type = cosine
│ │  └───────────────────── learning_rate = 3e-4
│ └──────────────────────── version 1
└────────────────────────── round 3
```

### Experiment Matrix

| Experiment | LR | Scheduler | Warmup | FP16 | Betas | Purpose |
|------------|-----|-----------|--------|------|-------|---------|
| `r3v1_lr3e4_cos_w03_fp16_b95` | 3e-4 | cosine | 3% | ✅ | 0.95 | **Baseline** |
| `r3v2_lr1e4_cos_w03_fp16_b95` | 1e-4 | cosine | 3% | ✅ | 0.95 | Low LR |
| `r3v3_lr6e4_cos_w03_fp16_b95` | 6e-4 | cosine | 3% | ✅ | 0.95 | High LR |
| `r3v4_lr3e4_cos_w01_fp16_b95` | 3e-4 | cosine | 1% | ✅ | 0.95 | Less warmup |
| `r3v5_lr3e4_cos_w05_fp16_b95` | 3e-4 | cosine | 5% | ✅ | 0.95 | More warmup |
| `r3v6_lr3e4_lin_w03_fp16_b95` | 3e-4 | linear | 3% | ✅ | 0.95 | Linear control |
| `r3v7_lr3e4_cos_w03_fp32_b95` | 3e-4 | cosine | 3% | ❌ | 0.95 | No FP16 |
| `r3v8_lr3e4_cos_w03_fp16_b999` | 3e-4 | cosine | 3% | ✅ | 0.999 | Original betas |
| `r3v9_lr6e4_cos_w05_fp16_b95` | 6e-4 | cosine | 5% | ✅ | 0.95 | Aggressive |

### Ablation Study Design

| Comparison | Variable | Control | Purpose |
|------------|----------|---------|---------|
| r3v1 vs r3v6 | Scheduler | r3v1 | Cosine vs Linear |
| r3v1 vs r3v7 | Mixed Precision | r3v1 | FP16 impact |
| r3v1 vs r3v8 | Adam β₂ | r3v1 | 0.95 vs 0.999 |
| r3v2 vs r3v1 vs r3v3 | Learning Rate | r3v1 | Optimal LR |
| r3v4 vs r3v1 vs r3v5 | Warmup Ratio | r3v1 | Optimal warmup |

---


## 4. WandB Metrics

### Metrics Types Overview

| Type | Trigger Condition | Frequency | Purpose |
|------|-------------------|-----------|---------|
| **High-Frequency** | `step % log_interval == 0` | Every 10 batches | Detailed training curves |
| **Progress** | `step == 10%, 20%, ...` | Fixed 10 times | Milestone comparisons |
| **Checkpoint** | `step % checkpoint_steps == 0` | Every N steps | Validation + model saving |

### Detailed Explanation

#### High-Frequency Metrics
- **Trigger**: Every 10 batches (configurable via `log_interval`)
- **Purpose**: Plot smooth training curves
- **Feature**: Training metrics only, **no validation**
- **Metrics**: `batch_loss`, `avg_loss`, `batch_accuracy`, `avg_accuracy`, `lr`

#### Progress Metrics
- **Trigger**: Training progress reaches 10%, 20%, 30%, ..., 100%
- **Purpose**: Cross-experiment comparison at same progress points
- **Feature**: **Runs validation** + **saves checkpoint**
- **Metrics**: `progress_train_loss`, `progress_avg_loss`, `progress_val_loss`, `progress_avg_acc`, `progress_lr`

#### Checkpoint Metrics
- **Trigger**: Every `checkpoint_steps` (e.g., every 1000 steps)
- **Purpose**: Monitor overfitting, early stopping, save best model
- **Feature**: **Runs validation** + **updates best model** + **early stopping check**
- **Metrics**: `train_loss`, `val_loss`, `step`, `epoch`

### Timeline Example

Assuming `total_batches = 1000`, `checkpoint_steps = 100`, `log_interval = 10`:

```
Step 10:   High-Frequency ✅
Step 20:   High-Frequency ✅
...
Step 100:  High-Frequency ✅ + Progress (10%) ✅ + Checkpoint ✅
Step 110:  High-Frequency ✅
...
Step 200:  High-Frequency ✅ + Progress (20%) ✅ + Checkpoint ✅
...
Step 1000: High-Frequency ✅ + Progress (100%) ✅ + Checkpoint ✅
```

### Summary

- **High-Frequency** = View detailed training curves
- **Progress** = View validation performance at milestones
- **Checkpoint** = Periodic validation + decide whether to save/early-stop

