# CLIP + CoMA: Continual Learning via Continual Model Averaging

A rigorous empirical investigation and research implementation of **task-incremental continual learning** combining the [CLIP](https://openai.com/research/clip) vision-language foundation model with **CoMA** (Continual Model Averaging), evaluated on a 10-task split of CIFAR-100.

---

## 📑 Research Documentation Links

- **Full Academic Research Paper**: [`FINAL_PROJECT_REPORT.md`](file:///c:/Users/Mukhil%20Charles/OneDrive/Desktop/clip_coma_integration/FINAL_PROJECT_REPORT.md) — 14-section formal academic paper with complete background, methodology, experimental results, stability-plasticity analysis, limitations, and literature references.
- **Experimental Results & Audit Report**: [`FINAL_RESULTS_REPORT.md`](file:///c:/Users/Mukhil%20Charles/OneDrive/Desktop/clip_coma_integration/FINAL_RESULTS_REPORT.md) — Exhaustive verification audit across all 8 JSON outputs, 56 PyTorch checkpoints, and execution runtime logs.

---

## 🎯 Executive Summary & Key Results

Sequential fine-tuning of neural networks typically induces severe **catastrophic forgetting**. This project evaluates whether exponential moving average weight consolidation (**CoMA**) applied solely to CLIP's visual encoder can mitigate catastrophic forgetting without requiring an exemplar memory buffer.

### Primary Benchmark Results (CIFAR-100, 10 Tasks)

| Method / Stage | α | Average Accuracy (%) | Average Forgetting (%) | Plasticity (%) | Execution Time |
|---|:---:|:---:|:---:|:---:|:---:|
| **Stage 24: Zero-Shot CLIP Baseline** | — | **88.57%** | **0.00%** | 88.57% | 102.4 s |
| **Stage 25: Sequential CLIP Baseline** | 1.00 | **86.91%** | **5.67%** | 92.01% | 1,623.4 s |
| **Stage 26: CLIP + CoMA Core** | 0.70 | **92.47%** | **3.57%** | 95.54% | 1,728.8 s |
| **Stage 28: CLIP + CoMA** | 0.25 | **94.38%** | **1.33%** | 95.53% | 1,910.2 s |
| **Stage 28: CLIP + CoMA** | 0.50 | **93.78%** | **2.72%** | **96.22%** | 1,962.2 s |
| **Stage 28: CLIP + CoMA** | 0.90 | **90.59%** | **4.40%** | 94.39% | 1,848.8 s |

### Key Empirical Findings
1. **Sequential Fine-Tuning Degrades Out-of-the-Box Zero-Shot Ability**: Unconstrained sequential fine-tuning ($\alpha=1.00$) drops final Average Accuracy to $86.91\%$, below the Zero-Shot floor ($88.57\%$), due to $5.67\%$ catastrophic forgetting.
2. **CoMA Reverses Degradation**: CoMA ($\alpha=0.70$) achieves $92.47\%$ Average Accuracy ($+5.56\%$ over Sequential) and reduces forgetting to $3.57\%$ (a $37.06\%$ relative reduction).
3. **$\alpha$ Governs the Stability-Plasticity Trade-Off**:
   - Lower $\alpha$ values prioritize parameter anchor stability: $\alpha=0.25$ attains the highest Average Accuracy ($94.38\%$) and lowest forgetting ($1.33\%$).
   - Intermediate $\alpha$ yields peak single-task acquisition: $\alpha=0.50$ attains the highest Plasticity ($96.22\%$).
   - Forgetting scales strictly monotonically with $\alpha$: $\alpha=0.25$ ($1.33\%$) $\to 0.50$ ($2.72\%$) $\to 0.70$ ($3.57\%$) $\to 0.90$ ($4.40\%$) $\to 1.00$ ($5.67\%$).

---

## 🧠 Continual Model Averaging (CoMA)

CoMA consolidates knowledge by maintaining a running exponential moving average (EMA) of model parameters across the sequential task stream:

$$\bar{\theta}_t = \alpha \cdot \theta_t + (1 - \alpha) \cdot \bar{\theta}_{t-1}$$

where:
- $\theta_t$: Trained parameters of the visual encoder upon completing task $t$.
- $\bar{\theta}_{t-1}$: Consolidated historical parameters retained up to task $t-1$ ($\bar{\theta}_0 = \theta_0$).
- $\alpha \in (0, 1]$: Plasticity interpolation factor.
- When $\alpha = 1.0$, CoMA reduces identically to standard sequential fine-tuning ($\bar{\theta}_t = \theta_t$).

**Targeted Visual Averaging**: Averaging is strictly restricted to parameters matching `model.visual.*`. The text encoder remains frozen, logit temperature scale is unperturbed, and non-parameter state buffers are preserved.

---

## 🏗️ Repository Architecture

```
clip_coma_integration/
├── clip_models.py           # CLIPClassifier wrapper (OpenCLIP ViT-B-32-quickgelu)
├── clip_tasks.py            # CIFAR-100 10-task disjoint split generator
├── clip_coma.py             # CoMA EMA engine, training loop & evaluation harness
├── metrics.py               # Continual learning metric computation
├── run_experiments.py       # Orchestration CLI for Stages 24–28 with checkpointing
├── run_clip_coma.py         # Lightweight standalone execution entry point
├── smoke_test.py            # Comprehensive cross-platform smoke & runtime test
├── requirements.txt         # Pinned runtime dependencies
├── README.md                # Project overview and reproduction guide
├── FINAL_PROJECT_REPORT.md  # Comprehensive academic research paper
├── FINAL_RESULTS_REPORT.md  # Formal verification audit report
├── results/                 # Verified experimental artifacts (8 JSONs, 56 checkpoints)
└── data/                    # CIFAR-100 dataset storage
```

---

## ⚙️ Experimental Configuration

| Parameter | Configuration Value | Description |
|---|---|---|
| **Backbone Architecture** | `ViT-B-32-quickgelu` | OpenCLIP Vision Transformer (pretrained: `openai`) |
| **Dataset** | CIFAR-100 | 50,000 train / 10,000 test images across 100 classes |
| **Task Partitioning** | 10 Disjoint Tasks | 10 classes per task (Task 0: 0–9, Task 1: 10–19, ..., Task 9: 90–99) |
| **Label Remapping** | Local Indexing ($0 \dots 9$) | Task-incremental evaluation; isolated task heads via text prompts |
| **Prompt Template** | `"a photo of a {class}."` | Zero-shot linear classifier built from text encoder embeddings |
| **Trainable Parameters** | Visual Tower Only | Text encoder frozen; visual weights fine-tuned with AdamW |
| **Optimizer** | AdamW | Learning rate $= 1 \times 10^{-5}$, weight decay $= 0.01$ |
| **Batch Size** | 8 | Minibatch size during fine-tuning |
| **Epochs per Task** | 1 | Controlled fine-tuning pass per task |
| **Random Seed** | 42 | Deterministic execution and stream partitioning |
| **Device Configuration** | CUDA (RTX 4060 Laptop) | Dynamic fallback: CUDA $\to$ MPS $\to$ CPU |

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python $\ge 3.10$ (Tested on Python 3.14.2)
- NVIDIA GPU with CUDA drivers (optional, automatic CPU fallback available)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/clip_coma_integration.git
cd clip_coma_integration

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Dependency Installation

**For NVIDIA CUDA (Windows/Linux):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

**For CPU-only:**
```bash
pip install -r requirements.txt
```

**For Apple Silicon (macOS MPS):**
```bash
pip install -r requirements.txt
```

### 4. Verification Smoke Test
Run the diagnostic smoke test to verify CLIP loading, CIFAR-100 pipeline, and CUDA/CPU forward/backward pass:
```bash
python smoke_test.py
```

---

## 🔬 Executing Experiments

### Run Complete Pipeline (Stages 24–28)
```bash
python run_experiments.py --stage all
```

### Run Individual Stages
```bash
# Stage 24: Zero-Shot Baseline
python run_experiments.py --stage 24

# Stage 25: Sequential Baseline (alpha = 1.0)
python run_experiments.py --stage 25

# Stage 26: CLIP + CoMA Core (alpha = 0.7)
python run_experiments.py --stage 26

# Stage 27: Generate Core Comparison Table
python run_experiments.py --stage 27

# Stage 28: Alpha Sensitivity Sweep (0.25, 0.50, 0.70, 0.90, 1.00)
python run_experiments.py --stage 28
```

### Standalone Demonstration
For an interactive, standalone single run without stage orchestration:
```bash
python run_clip_coma.py
```

---

## 📂 Results Directory Structure

The repository contains all experimental outputs under `results/`:

```
results/
├── zero_shot_clip/
│   ├── results.json                  # Accuracy vector & Zero-Shot baseline metrics
│   └── final_model.pt                # Unmodified pretrained CLIP state
├── sequential_clip_alpha_1.0/
│   ├── results.json                  # Full 10x10 matrix & sequential baseline metrics
│   ├── final_model.pt                # Final model after Task 9
│   └── checkpoint_task_0..9.pt       # Checkpoints after each task (10 files)
├── clip_coma_alpha_0.7/
│   ├── results.json                  # Full 10x10 matrix & CoMA (0.7) metrics
│   ├── final_model.pt                # Final consolidated model after Task 9
│   └── checkpoint_task_0..9.pt       # Checkpoints after each task (10 files)
├── clip_coma_alpha_0.25/
│   ├── results.json                  # Full 10x10 matrix & CoMA (0.25) metrics
│   ├── final_model.pt
│   └── checkpoint_task_0..9.pt       # Checkpoints after each task (10 files)
├── clip_coma_alpha_0.50/
│   ├── results.json                  # Full 10x10 matrix & CoMA (0.50) metrics
│   ├── final_model.pt
│   └── checkpoint_task_0..9.pt       # Checkpoints after each task (10 files)
├── clip_coma_alpha_0.90/
│   ├── results.json                  # Full 10x10 matrix & CoMA (0.90) metrics
│   ├── final_model.pt
│   └── checkpoint_task_0..9.pt       # Checkpoints after each task (10 files)
├── core_comparison.json              # Stage 27 consolidated comparison matrix
└── alpha_sensitivity_sweep.json      # Stage 28 multi-alpha consolidated analysis
```

**Artifact Totals**:
- **8 Result JSONs**: 6 per-method result files + 2 consolidated multi-run comparison summaries.
- **56 Model Checkpoints**: 6 final model files (`final_model.pt`) + 50 intermediate per-task checkpoints (`checkpoint_task_*.pt`).

---

## 🛡️ Checkpoint & Resumption Mechanics

The pipeline incorporates automated state-recovery guards:
1. **Stage-Level Guard**: If `results/<method>/results.json` exists, the stage loads the cached outputs directly without re-executing.
2. **Task-Level Guard**: If execution terminates unexpectedly mid-stream, `run_clip_coma()` inspects existing `checkpoint_task_*.pt` files and automatically resumes from task $k+1$.
3. **Device Independence**: All checkpoints use safe cross-platform loading (`map_location=device`), allowing seamless transitions between CUDA, MPS, and CPU.

---

## ⚠️ Scope and Academic Limitations

To ensure honest, reproducible scientific claims:
1. **Single Random Seed**: All reported numbers stem from seed 42. Multi-seed confidence intervals are not computed.
2. **Fixed Task Ordering**: Experiments utilize a single sequential class order (0–9 $\to \dots \to$ 90–99).
3. **Single Training Epoch**: Each task is trained for 1 epoch.
4. **Single Vision-Language Backbone**: Evaluated solely on `ViT-B-32-quickgelu` with `openai` weights.
5. **No Universal Superiority Claimed**: While CoMA demonstrates clear retention benefits on this benchmark, claims cannot be generalized to all continual learning benchmarks without further empirical evaluation.

---

## 📚 References

- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021.
- McCloskey, M., & Cohen, N. J. (1989). *Catastrophic interference in connectionist networks: The sequential learning problem*. Psychology of Learning and Motivation.
- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS, 114(13), 3521-3526.
- Polyak, B. T., & Juditsky, A. B. (1992). *Acceleration of stochastic approximation by averaging*. SIAM Journal on Control and Optimization.
- Izmailov, P., et al. (2018). *Averaging Weights Leads to Wider Optima and Better Generalization*. UAI 2018.
- Wortsman, M., et al. (2022). *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*. ICML 2022.
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.
