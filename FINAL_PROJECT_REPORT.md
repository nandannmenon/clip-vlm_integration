# Empirical Evaluation of Continual Model Averaging (CoMA) with CLIP for Task-Incremental Visual Classification

**Document:** Comprehensive Academic Research Report  
**Project:** `clip_coma_integration`  
**Author / Repository Context:** Continual Learning Research Project  
**Date of Audit & Generation:** November 2026  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Windows 11 AMD64  
**Software Environment:** Python 3.14.2, PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126, OpenCLIP 4.2.2  

---

## 1. Abstract

Task-incremental learning in deep neural networks typically exhibits catastrophic forgetting, where sequential adaptation to novel task distributions degrades performance on previously acquired capabilities. Foundation vision-language models like Contrastive Language-Image Pre-training (CLIP) possess rich pre-trained visual representations, yet unconstrained fine-tuning still causes parameter drift and forgetting. This report presents an empirical investigation of Continual Model Averaging (CoMA) combined with a frozen-text OpenCLIP ViT-B/32 backbone evaluated across a 10-task sequential stream of CIFAR-100 (10 classes per task, disjoint class splits). 

The experimental suite encompasses six evaluations: an un-tuned Zero-Shot baseline, an ordinary sequential fine-tuning baseline ($\alpha=1.00$), and four CoMA parameter consolidation runs ($\alpha \in \{0.25, 0.50, 0.70, 0.90\}$). Empirical results show that naive sequential fine-tuning suffers $5.67\%$ average forgetting, causing its final average accuracy ($86.91\%$) to drop below the untrained Zero-Shot CLIP baseline ($88.57\%$). In contrast, applying visual tower parameter averaging at $\alpha=0.70$ improves final average accuracy to $92.47\%$ ($+5.56\%$ over sequential, $+3.90\%$ over zero-shot) and curbs forgetting to $3.57\%$. Parameter sensitivity exploration reveals that anchoring weights more heavily to the historical trajectory ($\alpha=0.25$) yields the highest retention, attaining $94.38\%$ average accuracy and reducing forgetting to $1.33\%$. The study documents full task-wise accuracy matrices, execution metrics, and explicitly delineates empirical observations from broader theoretical hypotheses under single-seed and single-epoch constraints.

---

## 2. Introduction

Continual learning seeks algorithms capable of incrementally acquiring new knowledge across an ongoing stream of tasks without catastrophic loss of historical competencies. In biological cognitive systems, memory consolidation mechanisms protect established neural representations while allowing synaptic modification for novel stimuli. Conversely, gradient-based artificial neural networks trained sequentially through stochastic gradient descent alter connection weights globally, overwriting representations necessary for past tasks.

Recent advances in large-scale multi-modal foundation models, such as CLIP, provide a compelling substrate for continual learning. Because CLIP is pre-trained contrastively on hundreds of millions of image-text pairs, its zero-shot capabilities provide strong visual-semantic priors. However, adapting CLIP to specialized downstream task sequences presents a dual challenge: task-specific fine-tuning is necessary to maximize discriminability on incoming classes, but parameter updates risk destroying the broad zero-shot alignment established during pre-training.

This project implements and evaluates the **CLIP + CoMA** framework. The visual encoder of a pre-trained OpenCLIP model is sequentially fine-tuned across 10 disjoint subsets of CIFAR-100, while the text encoder remains completely frozen. Following each task episode, an exponential parameter averaging rule blends the newly fine-tuned visual weights with the running historical weight vector. This document details the research objectives, methodology, verified empirical results, stability-plasticity dynamics, and scope constraints of the implementation.

---

## 3. Research Problem and Motivation

### 3.1 The Catastrophic Forgetting Dilemma in Vision-Language Models
In a conventional task-incremental vision benchmark, a model encounters tasks $\mathcal{T}_0, \mathcal{T}_1, \dots, \mathcal{T}_{T-1}$ sequentially. When standard backpropagation updates model weights on $\mathcal{T}_t$, gradients are computed solely with respect to the loss on task $t$:
$$\theta_t = \theta_{t-1} - \eta \nabla_\theta \mathcal{L}_t(\theta_{t-1})$$
In the absence of data rehearsal (replaying data from $\mathcal{T}_0 \dots \mathcal{T}_{t-1}$) or explicit parameter regularization, directions in weight space that preserve past task performance are overwritten. 

### 3.2 The Specific Vulnerability of Pre-trained Encoders
While one might expect that a massively pre-trained model like CLIP would resist forgetting due to its broad representations, empirical tests show that fine-tuning even for a single epoch on localized class subsets causes representations to collapse toward the immediate objective. The model becomes specialized to the current task at the expense of previously observed visual categories, often degrading below its initial zero-shot state. Developing lightweight, buffer-free parameter consolidation algorithms that prevent this representational collapse without storing historical image samples motivates this work.

---

## 4. Research Objectives

The research objectives established for this investigation are:
1. **Baseline Characterization:** Quantify the zero-shot classification performance of OpenCLIP ViT-B/32 on a 10-task partition of CIFAR-100 without any task-specific fine-tuning (Stage 24).
2. **Sequential Degradation Assessment:** Measure the degree of catastrophic forgetting and final accuracy degradation under unconstrained sequential fine-tuning ($\alpha = 1.00$) where no parameter consolidation occurs (Stage 25).
3. **Core CoMA Validation:** Implement and validate the continual model averaging mechanism with $\alpha = 0.70$ on the visual tower, evaluating whether parameter averaging mitigates forgetting relative to sequential fine-tuning (Stage 26).
4. **Comparative Analysis:** Conduct a controlled stage-by-stage comparison across Zero-Shot, Sequential, and CoMA formulations using standardized continual learning metrics (Stage 27).
5. **Sensitivity Analysis:** Systematically sweep the averaging parameter $\alpha \in \{0.25, 0.50, 0.70, 0.90, 1.00\}$ to characterize the stability-plasticity trade-off curve across sequential task boundaries (Stage 28).
6. **Execution Integrity & Reproducibility:** Ensure modularity, automatic checkpointing, cross-platform execution (CUDA/CPU), and complete verification of mathematical outputs against saved JSON artifacts.

---

## 5. Background and Related Concepts

### 5.1 Continual Learning Paradigms
Continual learning formulations are generally categorized into three regimes:
- **Task-Incremental Learning (Task-IL):** The task identity is provided at inference time, allowing the model to restrict its decision boundary to the classes relevant to that specific task.
- **Class-Incremental Learning (Class-IL):** The model must classify an input across all seen classes without knowing which task introduced the class.
- **Domain-Incremental Learning (Domain-IL):** The class set remains constant, but the input distribution shifts across task phases.

The current implementation operates in the **Task-Incremental** setting: evaluation on task $i$ measures accuracy against the prompt embeddings representing only the classes belonging to task $i$.

### 5.2 Catastrophic Forgetting
Catastrophic forgetting describes the abrupt drop in performance on previously mastered tasks upon training on new data. Formal metrics quantify this phenomenon by tracking the maximum accuracy achieved on a task immediately after learning versus its retention at the conclusion of the sequential training pipeline.

### 5.3 Contrastive Language-Image Pre-training (CLIP)
CLIP consists of two parallel encoders: a vision transformer $f_v(\cdot; \theta_v)$ and a text transformer $f_t(\cdot; \theta_t)$. During pre-training, image and text representations are projected into a shared embedding space of dimension $d$ and normalized to unit length:
$$z_v = \frac{f_v(x; \theta_v)}{\|f_v(x; \theta_v)\|_2}, \quad z_t = \frac{f_t(c; \theta_t)}{\|f_t(c; \theta_t)\|_2}$$
Zero-shot prediction computes normalized cosine similarities scaled by a temperature parameter $\tau$:
$$s(x, c) = \tau \cdot \langle z_v, z_t \rangle$$
In prompt-based classification, text tokens are synthesized from class labels using natural language prompts (e.g., `"a photo of a {class}."`).

### 5.4 Parameter Averaging and Weight Consolidation
Weight-space averaging has a rich history in stochastic optimization, tracing back to Polyak-Juditsky averaging, Stochastic Weight Averaging (SWA), and recent studies on "model soups" which demonstrate that fine-tuned checkpoints from a shared pre-trained initialization lie within the same low-loss basin. Continual Model Averaging (CoMA) applies recursive convex combinations in weight space after distinct learning episodes to retain historical parameter trajectories.

---

## 6. Proposed Methodology

### 6.1 Architecture and Integration
The architecture integrates a visual encoder and a frozen text encoder as implemented in `clip_models.py` (`CLIPClassifier`):
- **Model Backbone:** `ViT-B-32-quickgelu` with pre-trained weights from OpenAI via the OpenCLIP library.
- **Text Encoder Freezing:** Prior to training, all parameters of the text encoder, positional embeddings, and projection heads are permanently frozen (`requires_grad = False`).
- **Visual Encoder Unfreezing:** All parameters residing within `model.visual` are set to `requires_grad = True` ($87,849,216$ trainable parameters).
- **Prompt Representation:** Text embeddings are computed once per task using class names formatted via `"a photo of a {}."` with underscores replaced by spaces.

### 6.2 Dataset and Stream Configuration
The dataset configuration, implemented in `clip_tasks.py`, partitions CIFAR-100 into a controlled sequential stream:
- **Total Classes:** 100 classes partitioned into 10 disjoint tasks ($T=10$).
- **Classes per Task:** Exactly 10 classes per task.
- **Task Order:** Deterministic partition based on global class indices:
  - Task 0: classes $0..9$ (`apple` to `bottle`)
  - Task 1: classes $10..19$ (`bowl` to `cattle`)
  - Task 2: classes $20..29$ (`chair` to `dinosaur`)
  - Task 3: classes $30..39$ (`dolphin` to `keyboard`)
  - Task 4: classes $40..49$ (`lamp` to `mountain`)
  - Task 5: classes $50..59$ (`mouse` to `pine_tree`)
  - Task 6: classes $60..69$ (`plain` to `rocket`)
  - Task 7: classes $70..79$ (`rose` to `spider`)
  - Task 8: classes $80..89$ (`squirrel` to `tractor`)
  - Task 9: classes $90..99$ (`train` to `worm`)
- **Data Partitions:** 5,000 training images and 1,000 testing images per task.

### 6.3 Local Label Remapping
Because tasks are evaluated in a task-incremental regime, global CIFAR-100 labels $y \in \{0..99\}$ are mapped via `LocalLabelSubset` to local targets $\tilde{y} \in \{0..9\}$:
$$\tilde{y} = \sigma_t(y) = y - 10 \cdot t, \quad \text{for } y \in [10t, 10t + 9]$$
Cross-entropy loss is computed against the 10 task-specific text features.

### 6.4 Training Procedure
For task $t$, optimization proceeds across batches $(X, \tilde{y}) \sim \mathcal{D}_t^{\text{train}}$:
1. Normalized visual features $Z_v = \text{L2Norm}(f_v(X))$ are extracted.
2. Logits are computed via scaled dot product: $L = \exp(\tau) \cdot Z_v \cdot Z_t^{\top}$, where $Z_t \in \mathbb{R}^{10 \times 512}$ represents the normalized text embeddings of task $t$.
3. Loss is evaluated as $\mathcal{L} = \text{CrossEntropy}(L, \tilde{y})$.
4. Gradients are propagated through `model.visual` using AdamW.

### 6.5 CoMA Parameter Averaging Rule
Following the completion of optimization on task $t$, parameter weights are updated via `average_visual_parameters` (`clip_coma.py`):
$$\bar{\theta}_t^{(k)} = \alpha \cdot \theta_t^{(k)} + (1 - \alpha) \cdot \bar{\theta}_{t-1}^{(k)}, \quad \forall k \in \mathcal{K}_{\text{visual}}$$
where:
- $\theta_t^{(k)}$ denotes the parameter tensor for named visual parameter $k$ after fine-tuning on task $t$.
- $\bar{\theta}_{t-1}^{(k)}$ denotes the accumulated parameter average up to task $t-1$. For $t=0$, $\bar{\theta}_{-1}$ is initialized to the pre-trained weights.
- $\alpha \in (0, 1]$ is the scalar blending coefficient.
- $\mathcal{K}_{\text{visual}}$ strictly indexes parameters starting with the `"visual."` prefix. Text transformer parameters, logit scale buffers, and non-parameter tensors remain unaffected.
- The model's active visual weights are then replaced by $\bar{\theta}_t$.

### 6.6 Evaluation Metrics
Three standard continual learning metrics are computed via `metrics.py` from the task accuracy matrix $A \in \mathbb{R}^{T \times T}$, where $A_{i, j}$ represents the classification accuracy on task $i$ measured after training on task $j$:

1. **Average Accuracy ($A_{\text{avg}}$):** Mean performance across all tasks at the conclusion of the sequential stream:
   $$A_{\text{avg}} = \frac{1}{T} \sum_{i=0}^{T-1} A_{i, T-1}$$
2. **Average Forgetting ($F_{\text{avg}}$):** Mean difference between peak historical accuracy and final accuracy across all tasks prior to the final task:
   $$F_{\text{avg}} = \frac{1}{T-1} \sum_{i=0}^{T-2} \max_{j \in \{i, \dots, T-2\}} \left( A_{i, j} - A_{i, T-1} \right)$$
3. **Plasticity ($P_{\text{avg}}$):** Mean accuracy on tasks immediately after they are trained:
   $$P_{\text{avg}} = \frac{1}{T} \sum_{i=0}^{T-1} A_{i, i}$$

---

## 7. Experimental Setup

### 7.1 Hardware Platform
- **Processor / Host:** AMD64 architecture, Windows 11 host operating system.
- **Graphics Processing Unit:** NVIDIA GeForce RTX 4060 Laptop GPU.
- **Dedicated Video Memory (VRAM):** 8,188 MiB (8 GB GDDR6).
- **GPU Driver:** NVIDIA Display Driver version 581.86.

### 7.2 Software Environment
- **Runtime:** Python 3.14.2 (tags/v3.14.2:df79316, 64-bit).
- **Deep Learning Framework:** PyTorch version 2.14.1+cu126.
- **Computer Vision Library:** torchvision version 0.29.1+cu126.
- **Vision-Language Framework:** OpenCLIP version 4.2.2.
- **CUDA Toolkit Runtime:** CUDA 12.6.

### 7.3 Experimental Hyperparameters
The configuration parameters remained invariant across all evaluated models:
- **Random Seed:** 42 (controlling Python `random`, NumPy, and PyTorch RNGs).
- **Batch Size:** 8 images per mini-batch.
- **Epochs per Task:** 1 epoch.
- **Optimizer:** AdamW.
- **Learning Rate:** $1 \times 10^{-5}$.
- **Weight Decay:** $0.1$.
- **DataLoader Workers:** 0 (deterministic single-process loading).
- **Memory Pinning:** `pin_memory = False` (ensuring cross-platform host stability).

---

## 8. Experimental Results

### 8.1 Zero-Shot CLIP Baseline (Stage 24)
Stage 24 evaluates the pre-trained OpenCLIP model without fine-tuning. Because no training steps take place, evaluation occurs at a single point, producing a $10 \times 1$ evaluation vector:
- **Average Accuracy:** $88.57\%$ ($0.8857$)
- **Average Forgetting:** $0.00\%$ ($0.0000$) by definition.
- **Plasticity:** $88.57\%$ ($0.8857$).

#### Task-wise Zero-Shot Accuracies
- Task 0: $91.50\%$
- Task 1: $92.70\%$
- Task 2: $86.90\%$
- Task 3: $90.80\%$
- Task 4: $83.40\%$
- Task 5: $87.30\%$
- Task 6: $81.80\%$
- Task 7: $88.30\%$
- Task 8: $91.30\%$
- Task 9: $91.70\%$

### 8.2 Sequential CLIP Baseline (Stage 25, $\alpha=1.00$)
Stage 25 executes standard sequential fine-tuning where $\alpha=1.00$ disables parameter averaging.
- **Average Accuracy:** $86.91\%$ ($0.8691$)
- **Average Forgetting:** $5.67\%$ ($0.0567$)
- **Plasticity:** $92.01\%$ ($0.9201$)
- **Active Execution Time:** 1,623.4 seconds (~27.1 minutes).

### 8.3 CLIP + CoMA Core Experiment (Stage 26, $\alpha=0.70$)
Stage 26 evaluates the core proposed configuration blending weights with $\alpha=0.70$.
- **Average Accuracy:** $92.47\%$ ($0.9247$)
- **Average Forgetting:** $3.57\%$ ($0.0357$)
- **Plasticity:** $95.54\%$ ($0.9554$)
- **Active Execution Time:** 1,728.8 seconds (~28.8 minutes).

### 8.4 CoMA Alpha Sensitivity Sweep (Stage 28)
Stage 28 evaluates the effect of parameter $\alpha$ across five values: $\{0.25, 0.50, 0.70, 0.90, 1.00\}$. Results for $\alpha=0.70$ and $\alpha=1.00$ are directly linked to Stages 26 and 25 respectively.

| Alpha ($\alpha$) | Average Accuracy | Average Forgetting | Plasticity | Active Execution Time | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.25** | **94.38%** (0.9438) | **1.33%** (0.0133) | 95.53% (0.9553) | 1,910.2 s (~31.8 min) | Completed & Verified |
| **0.50** | 93.78% (0.9378) | 2.72% (0.0272) | **96.22%** (0.9622) | 1,962.2 s (~32.7 min) | Completed & Verified |
| **0.70** | 92.47% (0.9247) | 3.57% (0.0357) | 95.54% (0.9554) | 1,728.8 s (~28.8 min) | Completed & Verified |
| **0.90** | 90.59% (0.9059) | 4.40% (0.0440) | 94.39% (0.9439) | 1,848.8 s (~30.8 min) | Completed & Verified |
| **1.00** | 86.91% (0.8691) | 5.67% (0.0567) | 92.01% (0.9201) | 1,623.4 s (~27.1 min) | Completed & Verified |

### 8.5 Task-Wise Accuracy Matrices

The following complete matrices record evaluated accuracies across all 10 stages. Unlearned stages prior to a task's training episode ($j < i$) are marked with `-` (stored as $0.0000$ in JSON files).

#### 8.5.1 Sequential Fine-Tuning ($\alpha = 1.00$) Matrix ($10 \times 10$)
```
Task \ Stage   T0      T1      T2      T3      T4      T5      T6      T7      T8      T9
Task 0       0.9320  0.9100  0.9000  0.8640  0.8570  0.8890  0.8330  0.7910  0.8530  0.8150
Task 1         -     0.9340  0.9010  0.8510  0.8170  0.9030  0.8700  0.8340  0.8470  0.8400
Task 2         -       -     0.9200  0.8960  0.8880  0.8700  0.8450  0.8200  0.8680  0.8470
Task 3         -       -       -     0.9360  0.8890  0.8610  0.8570  0.8100  0.8680  0.8650
Task 4         -       -       -       -     0.9110  0.8930  0.8920  0.8620  0.9030  0.8860
Task 5         -       -       -       -       -     0.9000  0.8970  0.8690  0.8680  0.8220
Task 6         -       -       -       -       -       -     0.8840  0.8770  0.8810  0.8710
Task 7         -       -       -       -       -       -       -     0.8980  0.8790  0.8670
Task 8         -       -       -       -       -       -       -       -     0.9360  0.9280
Task 9         -       -       -       -       -       -       -       -       -     0.9500
```

#### 8.5.2 CoMA Sweep ($\alpha = 0.25$) Matrix ($10 \times 10$)
```
Task \ Stage   T0      T1      T2      T3      T4      T5      T6      T7      T8      T9
Task 0       0.9680  0.9700  0.9670  0.9680  0.9650  0.9530  0.9550  0.9590  0.9520  0.9490
Task 1         -     0.9730  0.9720  0.9690  0.9680  0.9560  0.9550  0.9470  0.9470  0.9500
Task 2         -       -     0.9490  0.9510  0.9500  0.9450  0.9460  0.9380  0.9400  0.9450
Task 3         -       -       -     0.9560  0.9590  0.9530  0.9460  0.9430  0.9390  0.9450
Task 4         -       -       -       -     0.9550  0.9440  0.9490  0.9360  0.9380  0.9310
Task 5         -       -       -       -       -     0.9340  0.9260  0.9230  0.9200  0.9260
Task 6         -       -       -       -       -       -     0.9160  0.9020  0.9030  0.9080
Task 7         -       -       -       -       -       -       -     0.9450  0.9370  0.9340
Task 8         -       -       -       -       -       -       -       -     0.9780  0.9710
Task 9         -       -       -       -       -       -       -       -       -     0.9790
```

#### 8.5.3 CoMA Sweep ($\alpha = 0.50$) Matrix ($10 \times 10$)
```
Task \ Stage   T0      T1      T2      T3      T4      T5      T6      T7      T8      T9
Task 0       0.9690  0.9670  0.9590  0.9560  0.9450  0.9460  0.9380  0.9480  0.9350  0.9270
Task 1         -     0.9720  0.9570  0.9420  0.9480  0.9510  0.9520  0.9420  0.9370  0.9230
Task 2         -       -     0.9600  0.9520  0.9530  0.9410  0.9420  0.9420  0.9310  0.9280
Task 3         -       -       -     0.9650  0.9460  0.9430  0.9380  0.9400  0.9410  0.9360
Task 4         -       -       -       -     0.9670  0.9630  0.9680  0.9610  0.9490  0.9410
Task 5         -       -       -       -       -     0.9490  0.9490  0.9400  0.9270  0.9210
Task 6         -       -       -       -       -       -     0.9340  0.9220  0.9070  0.9220
Task 7         -       -       -       -       -       -       -     0.9470  0.9390  0.9320
Task 8         -       -       -       -       -       -       -       -     0.9770  0.9660
Task 9         -       -       -       -       -       -       -       -       -     0.9820
```

#### 8.5.4 CLIP + CoMA Core ($\alpha = 0.70$) Matrix ($10 \times 10$)
```
Task \ Stage   T0      T1      T2      T3      T4      T5      T6      T7      T8      T9
Task 0       0.9580  0.9540  0.9470  0.9480  0.9150  0.9180  0.9130  0.9100  0.9010  0.9010
Task 1         -     0.9760  0.9630  0.9360  0.9180  0.9150  0.9020  0.9020  0.9250  0.9180
Task 2         -       -     0.9470  0.9460  0.9370  0.9220  0.9200  0.9220  0.9290  0.9080
Task 3         -       -       -     0.9610  0.9550  0.9300  0.9260  0.9110  0.9440  0.9080
Task 4         -       -       -       -     0.9690  0.9450  0.9390  0.9440  0.9540  0.9490
Task 5         -       -       -       -       -     0.9300  0.9390  0.9160  0.9180  0.9060
Task 6         -       -       -       -       -       -     0.9320  0.9040  0.9210  0.9060
Task 7         -       -       -       -       -       -       -     0.9410  0.9460  0.9180
Task 8         -       -       -       -       -       -       -       -     0.9750  0.9680
Task 9         -       -       -       -       -       -       -       -       -     0.9650
```

#### 8.5.5 CoMA Sweep ($\alpha = 0.90$) Matrix ($10 \times 10$)
```
Task \ Stage   T0      T1      T2      T3      T4      T5      T6      T7      T8      T9
Task 0       0.9410  0.9420  0.9060  0.9280  0.8970  0.8980  0.8890  0.8390  0.8820  0.8680
Task 1         -     0.9450  0.9320  0.9060  0.8740  0.9190  0.8730  0.8490  0.9030  0.8760
Task 2         -       -     0.9340  0.9220  0.9090  0.8890  0.8750  0.8700  0.9040  0.9070
Task 3         -       -       -     0.9490  0.9140  0.9100  0.8750  0.8790  0.9040  0.8910
Task 4         -       -       -       -     0.9640  0.9210  0.9160  0.9180  0.9220  0.9030
Task 5         -       -       -       -       -     0.9070  0.9220  0.8940  0.9000  0.9000
Task 6         -       -       -       -       -       -     0.9240  0.9060  0.9110  0.9100
Task 7         -       -       -       -       -       -       -     0.9430  0.9320  0.8990
Task 8         -       -       -       -       -       -       -       -     0.9720  0.9450
Task 9         -       -       -       -       -       -       -       -       -     0.9600
```

---

## 9. Comparative Analysis

### 9.1 Core Triad Comparison (Stage 27)

| Paradigm | Blending Factor $\alpha$ | Average Accuracy | Average Forgetting | Plasticity | Accuracy $\Delta$ vs Sequential | Forgetting $\Delta$ vs Sequential |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Zero-Shot CLIP** | — | 88.57% | 0.00% | 88.57% | +1.66% | -5.67% |
| **Sequential Baseline** | 1.00 | 86.91% | 5.67% | 92.01% | 0.00% | 0.00% |
| **CLIP + CoMA Core** | 0.70 | **92.47%** | **3.57%** | **95.54%** | **+5.56%** | **-2.10%** |

#### Key Analytical Insights
1. **Representational Degradation in Sequential Tuning:**
   Sequential fine-tuning achieves $92.01\%$ plasticity, reflecting rapid acquisition of task-specific features during the active epoch. However, as subsequent tasks are encountered, performance on Task 0 drops from $93.20\%$ down to $81.50\%$ (a $11.70\%$ absolute degradation). As a consequence, the final average accuracy ($86.91\%$) is $1.66\%$ lower than if the network had received no training at all ($88.57\%$).
2. **Efficacy of CoMA Parameter Consolidation:**
   Under $\alpha=0.70$, CoMA achieves $92.47\%$ average accuracy, representing a $+5.56\%$ absolute increase over sequential fine-tuning and a $+3.90\%$ absolute gain over the zero-shot baseline. Average forgetting decreases from $5.67\%$ to $3.57\%$, representing a $37.06\%$ relative reduction in forgetting.

### 9.2 The Stability-Plasticity Spectrum Across $\alpha$

Examining the full sweep across $\alpha \in \{0.25, 0.50, 0.70, 0.90, 1.00\}$ reveals clear empirical trends:
- **Forgetting scales monotonically with $\alpha$:** $1.33\% \to 2.72\% \to 3.57\% \to 4.40\% \to 5.67\%$. Higher values of $\alpha$ assign larger weight to the most recent gradient steps, increasing drift away from earlier tasks.
- **Average Accuracy decreases monotonically with $\alpha$:** $94.38\% \to 93.78\% \to 92.47\% \to 90.59\% \to 86.91\%$. In this setting, mitigating catastrophic forgetting dominates final accuracy outcomes.
- **Plasticity peaks at intermediate $\alpha$:** Plasticity reaches its maximum at $\alpha=0.50$ ($96.22\%$), outperforming both $\alpha=1.00$ ($92.01\%$) and $\alpha=0.25$ ($95.53\%$).

---

## 10. Discussion

### 10.1 Mechanism of Action: Why Parameter Averaging Preserves Representations
In deep parameter spaces initialized from pre-trained foundation models, fine-tuning on disjoint class distributions shifts weights along task-specific manifolds. When $\alpha=1.00$, the trajectory wanders away from the shared zero-shot initialization, breaking cross-modal alignments with earlier task prompts.

When CoMA computes $\bar{\theta}_t = \alpha \theta_t + (1-\alpha) \bar{\theta}_{t-1}$, it acts as an implicit regularizer in parameter space. The weight vector is anchored near the center of the shared basin of attraction. Because the text embeddings remain fixed, preserving visual feature alignment with the shared space prevents catastrophic degradation of past class separations.

### 10.2 Distinguishing Observations from Hypotheses
To maintain academic rigor, we delineate direct experimental observations from theoretical hypotheses:

- **Confirmed Observations:**
  - On this specific CIFAR-100 stream, sequential fine-tuning without averaging degraded final average accuracy below the un-tuned zero-shot baseline.
  - Across the five tested values, decreasing $\alpha$ from $1.00$ to $0.25$ monotonically decreased average forgetting and monotonically increased final average accuracy.
  - The maximum plasticity was measured at $\alpha = 0.50$ ($96.22\%$).
- **Theoretical Hypotheses (Requiring Further Independent Verification):**
  - *Hypothesis 1 (Geometric Basin Hypothesis):* Weight averaging succeeds because OpenCLIP's pre-training establishes a wide, convex loss basin where linear interpolations between task optima retain low loss.
  - *Hypothesis 2 (Asymptotic Saturation):* For $\alpha \ll 0.25$ (e.g., $\alpha = 0.05$), plasticity will eventually collapse because the updates from new tasks would be attenuated too heavily to adapt to novel visual distributions.

---

## 11. Limitations

The empirical findings documented in this report must be interpreted within the specific boundaries of the experimental protocol:

1. **Single Random Seed:** All experiments were run with random seed 42. While standard seeds ensure internal consistency across comparative conditions, no cross-seed variance or statistical significance tests (e.g., paired t-tests or confidence intervals) can be claimed.
2. **Single Fixed Class Ordering:** Classes were divided in natural sequential order ($0..9, 10..19, \dots$). Performance invariance across randomized or adversarial class permutations has not been verified.
3. **Single Epoch per Task:** In keeping with fast-evaluation experimental protocols, each task was trained for exactly one epoch ($E=1$). Extended multi-epoch convergence regimes (e.g., 5–10 epochs per task with learning rate schedules) may alter weight drift dynamics.
4. **Single Model Backbone:** Experiments were conducted exclusively using OpenCLIP ViT-B/32 (`quickgelu`, OpenAI pre-trained weights). Generalizability to larger transformer architectures (e.g., ViT-L/14) or alternative pre-training distributions (e.g., LAION-2B) remains unverified.
5. **Absence of Rehearsal Baselines:** Comparison was made strictly against zero-shot and sequential fine-tuning baselines; comparisons against replay-based methods (e.g., Experience Replay, GEM) or regularization-based methods (e.g., EWC, LwF) were not evaluated in this codebase.
6. **No Universal Superiority Claim:** CoMA is not claimed to be universally superior across all continual learning benchmarks or arbitrary domain shifts.

---

## 12. Conclusion

This research implemented and evaluated Continual Model Averaging (CoMA) combined with OpenCLIP ViT-B/32 for task-incremental visual classification on CIFAR-100. The empirical evaluation verified that:
1. Ordinary sequential fine-tuning causes noticeable representational collapse ($5.67\%$ forgetting), resulting in an average accuracy ($86.91\%$) that underperforms the baseline Zero-Shot model ($88.57\%$).
2. Integrating exponential model averaging on the visual tower parameters successfully mitigates this degradation. At $\alpha=0.70$, CoMA achieves $92.47\%$ average accuracy and reduces forgetting to $3.57\%$.
3. Lower blending coefficients exhibit superior retention; $\alpha=0.25$ attained the highest average accuracy ($94.38\%$) and lowest forgetting ($1.33\%$).

The results indicate that parameter-space weight consolidation is an effective, buffer-free strategy for adapting multi-modal foundation models to sequential task streams without requiring rehearsal buffers or architectural expansion.

---

## 13. Future Work

Recommended directions for extending this work include:
1. **Multi-Seed and Multi-Permutation Validation:** Execute 5-seed runs across randomized task class permutations to calculate mean and standard deviation confidence intervals.
2. **Multi-Epoch and Learning Rate Scheduling:** Evaluate 5–10 epochs per task combined with cosine decay to assess stability under full convergence.
3. **Cross-Architecture Scaling:** Extend evaluations to ViT-B/16, ViT-L/14, and ConvNeXt visual backbones to examine the relationship between parameter scale and averaging stability.
4. **Cross-Dataset Generalization:** Evaluate CoMA across diverse continual learning benchmarks including Split ImageNet-R, CUB-200, and DomainNet.
5. **Dynamic / Adaptive Alpha Scheduling:** Investigate state-dependent $\alpha_t$ schedules that dynamically modulate blending coefficients based on gradient norm or fisher information metrics.

---

## 14. References

1. **CLIP:**  
   Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* In Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139:8748-8763.
2. **Catastrophic Forgetting & Continual Learning Foundations:**  
   - McCloskey, M., & Cohen, N. J. (1989). *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 24, 109-165.  
   - Ratcliff, R. (1990). *Connectionist Models of Recognition Memory: Constraints Imposed by Learning and Forgetting Functions.* Psychological Review, 97(2), 285-308.  
   - Kirkpatrick, J., et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks.* Proceedings of the National Academy of Sciences (PNAS), 114(13), 3521-3526.
3. **Model Averaging & Stochastic Weight Averaging:**  
   - Polyak, B. T., & Juditsky, A. B. (1992). *Acceleration of Stochastic Approximation by Averaging.* SIAM Journal on Control and Optimization, 30(4), 838-855.  
   - Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). *Averaging Weights Leads to Wider Optima and Better Generalization.* In Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence (UAI 2018), 876-885.  
   - Wortsman, M., et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time.* In Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162:23965-23998.
4. **CIFAR-100 Benchmark:**  
   Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* Technical Report, University of Toronto.
5. **Project Origin & Implementation Note:**  
   The CoMA algorithm in this repository (`clip_coma_integration`) specifically denotes exponential moving average weight consolidation applied to the visual transformer weights of CLIP. The empirical data, metrics, and matrices presented in this report were verified directly from the project's generated artifacts (`results/*.json`). Any external literature cited above provides contextual theoretical grounding and does not alter the standalone empirical record of this repository.
