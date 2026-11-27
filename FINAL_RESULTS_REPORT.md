# CLIP + CoMA Continual Learning: Final Experimental Results Report

**Project:** `clip_coma_integration`  
**Execution Environment:** Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), CUDA 12.6, PyTorch 2.14.1+cu126  
**Evaluation Dataset:** CIFAR-100 (10 Tasks × 10 Classes, Class-Disjoint Splits)  
**Execution Date:** 2026-11-26  
**Status:** All Stages (24, 25, 26, 27, 28) Completed and Rigorously Audited  

---

## Executive Summary

This report documents the empirical findings from evaluating **CLIP + CoMA (Cosine Model Averaging)** against standard continual learning baselines on a 10-task sequential stream of CIFAR-100.

### Key Findings
1. **Catastrophic Forgetting in Standard Sequential Fine-Tuning:**
   Ordinary sequential fine-tuning ($\alpha = 1.00$) achieves high immediate learning capacity ($\text{Plasticity} = 92.01\%$), but suffers significant forgetting ($\text{Average Forgetting} = 5.67\%$). Crucially, its final average accuracy of **$86.91\%$** falls below the **$88.57\%$** of the un-tuned **Zero-Shot CLIP baseline**, demonstrating that naive sequential fine-tuning damages CLIP's pre-trained zero-shot generalisation.
2. **CoMA Prevents Forgetting and Outperforms Baselines:**
   The core CoMA algorithm ($\alpha = 0.70$) yields **$92.47\%$** Average Accuracy ($+5.56\%$ absolute over Sequential Fine-Tuning, $+3.90\%$ absolute over Zero-Shot CLIP) and reduces Average Forgetting to **$3.57\%$** (a $37.06\%$ relative reduction in forgetting compared to sequential fine-tuning).
3. **Alpha Sensitivity: Lower $\alpha$ Maximises Stability:**
   Sweeping $\alpha \in \{0.25, 0.50, 0.70, 0.90, 1.00\}$ reveals that smaller $\alpha$ values provide the strongest retention of past task features without sacrificing plasticity. The optimal configuration is **$\alpha = 0.25$**, achieving **$94.38\%$ Average Accuracy** and reducing Average Forgetting to **$1.33\%$** (a $76.47\%$ relative reduction in forgetting compared to $\alpha=1.00$).

---

## 1. Experimental Methodology & Configuration

### Architecture & Backbone
- **Vision-Language Model:** OpenCLIP ViT-B/32 (`ViT-B-32-quickgelu`, pre-trained on OpenAI weights).
- **Text Tower:** Frozen across all stages. Class names are mapped using the prompt template `"a photo of a {}."` into normalized text embeddings.
- **Visual Tower:** Trainable visual encoder parameters ($87,849,216$ parameters).
- **Classification Mechanism:** Cosine similarity scaled by `logit_scale.exp()` between normalized image features and task text features. No task-specific linear classification heads are added.

### CoMA Averaging Rule
After completing task $t$, the visual tower weights $\theta_t$ are blended into the cumulative parameter average $\bar{\theta}_t$:
$$\bar{\theta}_t = \alpha \cdot \theta_t + (1 - \alpha) \cdot \bar{\theta}_{t-1}$$
Only named parameters starting with `visual.` participate in averaging. Non-parameter buffers, text tower weights, and logit scales are retained from the current model.

### Hyperparameters (Constant Across All Stages)
- **Dataset:** CIFAR-100 ($50,000$ train, $10,000$ test images).
- **Stream Structure:** $10$ sequential tasks, $10$ non-overlapping classes per task.
- **Labels:** Remapped to local task indices $0..9$ per task.
- **Batch Size:** $8$
- **Epochs per Task:** $1$
- **Optimizer:** AdamW ($\text{learning rate} = 1\times 10^{-5}$, $\text{weight decay} = 0.1$).
- **Seed:** $42$
- **Hardware Device:** `cuda` (NVIDIA GeForce RTX 4060 Laptop GPU).

---

## 2. Experimental Verification & Checkpoint Audit

All **56 checkpoint files (`.pt`)** and **8 output JSON files** across all directories have been audited and verified intact. No NaN, infinite, or missing values exist in any file.

### File Inventory & Audit Summary

| Stage / Experiment Group | Directory / File Path | Checkpoint Count & Files | JSON File Status | Verification Status |
|:---|:---|:---:|:---:|:---:|
| **Stage 24: Zero-Shot Baseline** | `results/zero_shot_clip/` | 1 (`final_model.pt`) | `results.json` | Verified (Complete, No NaNs) |
| **Stage 25: Sequential Baseline ($\alpha=1.0$)** | `results/sequential_clip_alpha_1.0/` | 11 (`checkpoint_task_0.pt`–`9.pt`, `final_model.pt`) | `results.json` | Verified (Complete, No NaNs) |
| **Stage 26: CLIP + CoMA Core ($\alpha=0.7$)** | `results/clip_coma_alpha_0.7/` | 11 (`checkpoint_task_0.pt`–`9.pt`, `final_model.pt`) | `results.json` | Verified (Complete, No NaNs) |
| **Stage 27: Core Comparison Summary** | `results/core_comparison.json` | — | `core_comparison.json` | Verified (Complete, No NaNs) |
| **Stage 28: CoMA Sweep ($\alpha=0.25$)** | `results/clip_coma_alpha_0.25/` | 11 (`checkpoint_task_0.pt`–`9.pt`, `final_model.pt`) | `results.json` | Verified (Complete, No NaNs) |
| **Stage 28: CoMA Sweep ($\alpha=0.50$)** | `results/clip_coma_alpha_0.50/` | 11 (`checkpoint_task_0.pt`–`9.pt`, `final_model.pt`) | `results.json` | Verified (Complete, No NaNs) |
| **Stage 28: CoMA Sweep ($\alpha=0.90$)** | `results/clip_coma_alpha_0.90/` | 11 (`checkpoint_task_0.pt`–`9.pt`, `final_model.pt`) | `results.json` | Verified (Complete, No NaNs) |
| **Stage 28: Sweep Summary** | `results/alpha_sensitivity_sweep.json` | — | `alpha_sensitivity_sweep.json` | Verified (Complete, No NaNs) |
| **Total Counts** | **8 directories / root files** | **56 `.pt` files** | **8 `.json` files** | **All Verified** |

---

## 3. Stage 27: Core Method Comparison

Comparison of the three primary paradigms after learning all 10 tasks sequentially:

| Metric | Zero-Shot CLIP Baseline (Stage 24) | Sequential Fine-Tuning $\alpha=1.0$ (Stage 25) | CLIP + CoMA Core $\alpha=0.7$ (Stage 26) | CoMA vs Sequential $\Delta$ | CoMA vs Zero-Shot $\Delta$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Average Accuracy ($A_{\text{avg}}$)** | $88.57\%$ ($0.8857$) | $86.91\%$ ($0.8691$) | **$92.47\%$ ($0.9247$)** | **$+5.56\%$** ($+0.0556$) | **$+3.90\%$** ($+0.0390$) |
| **Average Forgetting ($F_{\text{avg}}$)** | $0.00\%$ ($0.0000$) | $5.67\%$ ($0.0567$) | **$3.57\%$ ($0.0357$)** | **$-2.10\%$** ($-0.0210$) | $+3.57\%$ ($+0.0357$) |
| **Plasticity ($P_{\text{avg}}$)** | $88.57\%$ ($0.8857$) | $92.01\%$ ($0.9201$) | **$95.54\%$ ($0.9554$)** | **$+3.53\%$** ($+0.0353$) | $+6.97\%$ ($+0.0697$) |

### Final Task-by-Task Accuracies
Accuracies evaluated on all 10 tasks after the completion of task 10:

| Task ID | Task Classes | Zero-Shot Acc | Sequential Acc ($\alpha=1.0$) | CoMA Acc ($\alpha=0.7$) | CoMA Delta vs. Sequential |
|:---:|:---|:---:|:---:|:---:|:---:|
| **Task 0** | apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle | $91.50\%$ | $81.50\%$ | **$90.10\%$** | $+8.60\%$ |
| **Task 1** | bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle | $92.70\%$ | $84.00\%$ | **$91.80\%$** | $+7.80\%$ |
| **Task 2** | chair, chimpanzee, clock, cloud, cockroach, couch, crab, crocodile, cup, dinosaur | $86.90\%$ | $84.70\%$ | **$90.80\%$** | $+6.10\%$ |
| **Task 3** | dolphin, elephant, flatfish, forest, fox, girl, hamster, house, kangaroo, keyboard | $90.80\%$ | $86.50\%$ | **$90.80\%$** | $+4.30\%$ |
| **Task 4** | lamp, lawn_mower, leopard, lion, lizard, lobster, man, maple_tree, motorcycle, mountain | $83.40\%$ | $88.60\%$ | **$94.90\%$** | $+6.30\%$ |
| **Task 5** | mouse, mushroom, oak_tree, orange, orchid, otter, palm_tree, pear, pickup_truck, pine_tree | $87.30\%$ | $82.20\%$ | **$90.60\%$** | $+8.40\%$ |
| **Task 6** | plain, plate, poppy, porcupine, possum, rabbit, raccoon, ray, road, rocket | $81.80\%$ | $87.10\%$ | **$90.60\%$** | $+3.50\%$ |
| **Task 7** | rose, sea, seal, shark, shrew, skunk, skyscraper, snail, snake, spider | $88.30\%$ | $86.70\%$ | **$91.80\%$** | $+5.10\%$ |
| **Task 8** | squirrel, streetcar, sunflower, sweet_pepper, table, tank, telephone, television, tiger, tractor | $91.30\%$ | $92.80\%$ | **$96.80\%$** | $+4.00\%$ |
| **Task 9** | train, trout, tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm | $91.70\%$ | $95.00\%$ | **$96.50\%$** | $+1.50\%$ |
| **Mean** | — | $88.57\%$ | $86.91\%$ | **$92.47\%$** | **$+5.56\%$** |

---

## 4. Stage 28: CoMA Alpha Sensitivity Sweep

The parameter $\alpha$ controls the plasticity-stability balance: $\alpha = 1.0$ is unconstrained sequential learning (maximum plasticity, zero parameter averaging), while smaller $\alpha$ anchors the visual tower weights more firmly to the historical parameter average.

| Alpha ($\alpha$) | Average Accuracy | Average Forgetting | Plasticity | Source Experiment Directory | Matrix Dimensions |
|:---:|:---:|:---:|:---:|:---|:---:|
| **$0.25$** | **$94.38\%$ ($0.9438$)** | **$1.33\%$ ($0.0133$)** | $95.53\%$ ($0.9553$) | `results/clip_coma_alpha_0.25/` | $10 \times 10$ |
| **$0.50$** | $93.78\%$ ($0.9378$) | $2.72\%$ ($0.0272$) | **$96.22\%$ ($0.9622$)** | `results/clip_coma_alpha_0.50/` | $10 \times 10$ |
| **$0.70$** | $92.47\%$ ($0.9247$) | $3.57\%$ ($0.0357$) | $95.54\%$ ($0.9554$) | `results/clip_coma_alpha_0.7/` | $10 \times 10$ |
| **$0.90$** | $90.59\%$ ($0.9059$) | $4.40\%$ ($0.0440$) | $94.39\%$ ($0.9439$) | `results/clip_coma_alpha_0.90/` | $10 \times 10$ |
| **$1.00$** | $86.91\%$ ($0.8691$) | $5.67\%$ ($0.0567$) | $92.01\%$ ($0.9201$) | `results/sequential_clip_alpha_1.0/` | $10 \times 10$ |

### Observations
1. **Monotonic Retention Improvement:** As $\alpha$ decreases from $1.00$ to $0.25$, Average Forgetting decreases monotonically: $5.67\% \to 4.40\% \to 3.57\% \to 2.72\% \to 1.33\%$.
2. **Monotonic Accuracy Gain:** Average Accuracy increases monotonically from $86.91\%$ at $\alpha=1.00$ up to $94.38\%$ at $\alpha=0.25$.
3. **Plasticity Retention:** Even at $\alpha = 0.25$, initial task learning accuracy remains high ($95.53\%$), demonstrating that parameter averaging preserves the visual encoder's plasticity for new task learning.

---

## 5. Task-Wise Accuracy Matrices

### Matrix Structural Distinctions
- **Zero-Shot CLIP Baseline (Stage 24):** Contains a **$10 \times 1$ evaluation vector** (one accuracy value per task). Because Zero-Shot CLIP is evaluated without sequential training, it does not generate an evolving $10 \times 10$ stage matrix.
- **Sequential Fine-Tuning & CoMA Sweep Runs:** Contain **full $10 \times 10$ sequential progression matrices**, where row $i$ represents task $i$, and column $j$ represents evaluation after sequentially training on task $j$. Entries before a task is learned ($j < i$) are unlearned states (stored as $0.0000$ in JSON, displayed as `-`).

---

### 5.1 Zero-Shot CLIP Baseline (Stage 24) — $10 \times 1$ Evaluation Vector

| Evaluated Task | Classes Evaluated | Zero-Shot Accuracy |
|:---:|:---|:---:|
| **Task 0** | apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle | $91.50\%$ ($0.9150$) |
| **Task 1** | bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle | $92.70\%$ ($0.9270$) |
| **Task 2** | chair, chimpanzee, clock, cloud, cockroach, couch, crab, crocodile, cup, dinosaur | $86.90\%$ ($0.8690$) |
| **Task 3** | dolphin, elephant, flatfish, forest, fox, girl, hamster, house, kangaroo, keyboard | $90.80\%$ ($0.9080$) |
| **Task 4** | lamp, lawn_mower, leopard, lion, lizard, lobster, man, maple_tree, motorcycle, mountain | $83.40\%$ ($0.8340$) |
| **Task 5** | mouse, mushroom, oak_tree, orange, orchid, otter, palm_tree, pear, pickup_truck, pine_tree | $87.30\%$ ($0.8730$) |
| **Task 6** | plain, plate, poppy, porcupine, possum, rabbit, raccoon, ray, road, rocket | $81.80\%$ ($0.8180$) |
| **Task 7** | rose, sea, seal, shark, shrew, skunk, skyscraper, snail, snake, spider | $88.30\%$ ($0.8830$) |
| **Task 8** | squirrel, streetcar, sunflower, sweet_pepper, table, tank, telephone, television, tiger, tractor | $91.30\%$ ($0.9130$) |
| **Task 9** | train, trout, tulip, turtle, wardrobe, whale, willow_tree, wolf, woman, worm | $91.70\%$ ($0.9170$) |
| **Mean** | — | **$88.57\%$ ($0.8857$)** |

---

### 5.2 Sequential Fine-Tuning ($\alpha = 1.00$) (Stage 25) — $10 \times 10$ Matrix

| Evaluated Task | After T0 | After T1 | After T2 | After T3 | After T4 | After T5 | After T6 | After T7 | After T8 | After T9 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Task 0** | 0.9320 | 0.9100 | 0.9000 | 0.8640 | 0.8570 | 0.8890 | 0.8330 | 0.7910 | 0.8530 | 0.8150 |
| **Task 1** | - | 0.9340 | 0.9010 | 0.8510 | 0.8170 | 0.9030 | 0.8700 | 0.8340 | 0.8470 | 0.8400 |
| **Task 2** | - | - | 0.9200 | 0.8960 | 0.8880 | 0.8700 | 0.8450 | 0.8200 | 0.8680 | 0.8470 |
| **Task 3** | - | - | - | 0.9360 | 0.8890 | 0.8610 | 0.8570 | 0.8100 | 0.8680 | 0.8650 |
| **Task 4** | - | - | - | - | 0.9110 | 0.8930 | 0.8920 | 0.8620 | 0.9030 | 0.8860 |
| **Task 5** | - | - | - | - | - | 0.9000 | 0.8970 | 0.8690 | 0.8680 | 0.8220 |
| **Task 6** | - | - | - | - | - | - | 0.8840 | 0.8770 | 0.8810 | 0.8710 |
| **Task 7** | - | - | - | - | - | - | - | 0.8980 | 0.8790 | 0.8670 |
| **Task 8** | - | - | - | - | - | - | - | - | 0.9360 | 0.9280 |
| **Task 9** | - | - | - | - | - | - | - | - | - | 0.9500 |

---

### 5.3 CoMA Sweep ($\alpha = 0.25$) (Stage 28) — $10 \times 10$ Matrix

| Evaluated Task | After T0 | After T1 | After T2 | After T3 | After T4 | After T5 | After T6 | After T7 | After T8 | After T9 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Task 0** | 0.9680 | 0.9700 | 0.9670 | 0.9680 | 0.9650 | 0.9530 | 0.9550 | 0.9590 | 0.9520 | 0.9490 |
| **Task 1** | - | 0.9730 | 0.9720 | 0.9690 | 0.9680 | 0.9560 | 0.9550 | 0.9470 | 0.9470 | 0.9500 |
| **Task 2** | - | - | 0.9490 | 0.9510 | 0.9500 | 0.9450 | 0.9460 | 0.9380 | 0.9400 | 0.9450 |
| **Task 3** | - | - | - | 0.9560 | 0.9590 | 0.9530 | 0.9460 | 0.9430 | 0.9390 | 0.9450 |
| **Task 4** | - | - | - | - | 0.9550 | 0.9440 | 0.9490 | 0.9360 | 0.9380 | 0.9310 |
| **Task 5** | - | - | - | - | - | 0.9340 | 0.9260 | 0.9230 | 0.9200 | 0.9260 |
| **Task 6** | - | - | - | - | - | - | 0.9160 | 0.9020 | 0.9030 | 0.9080 |
| **Task 7** | - | - | - | - | - | - | - | 0.9450 | 0.9370 | 0.9340 |
| **Task 8** | - | - | - | - | - | - | - | - | 0.9780 | 0.9710 |
| **Task 9** | - | - | - | - | - | - | - | - | - | 0.9790 |

---

### 5.4 CoMA Sweep ($\alpha = 0.50$) (Stage 28) — $10 \times 10$ Matrix

| Evaluated Task | After T0 | After T1 | After T2 | After T3 | After T4 | After T5 | After T6 | After T7 | After T8 | After T9 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Task 0** | 0.9690 | 0.9670 | 0.9590 | 0.9560 | 0.9450 | 0.9460 | 0.9380 | 0.9480 | 0.9350 | 0.9270 |
| **Task 1** | - | 0.9720 | 0.9570 | 0.9420 | 0.9480 | 0.9510 | 0.9520 | 0.9420 | 0.9370 | 0.9230 |
| **Task 2** | - | - | 0.9600 | 0.9520 | 0.9530 | 0.9410 | 0.9420 | 0.9420 | 0.9310 | 0.9280 |
| **Task 3** | - | - | - | 0.9650 | 0.9460 | 0.9430 | 0.9380 | 0.9400 | 0.9410 | 0.9360 |
| **Task 4** | - | - | - | - | 0.9670 | 0.9630 | 0.9680 | 0.9610 | 0.9490 | 0.9410 |
| **Task 5** | - | - | - | - | - | 0.9490 | 0.9490 | 0.9400 | 0.9270 | 0.9210 |
| **Task 6** | - | - | - | - | - | - | 0.9340 | 0.9220 | 0.9070 | 0.9220 |
| **Task 7** | - | - | - | - | - | - | - | 0.9470 | 0.9390 | 0.9320 |
| **Task 8** | - | - | - | - | - | - | - | - | 0.9770 | 0.9660 |
| **Task 9** | - | - | - | - | - | - | - | - | - | 0.9820 |

---

### 5.5 CLIP + CoMA Core ($\alpha = 0.70$) (Stage 26) — $10 \times 10$ Matrix

| Evaluated Task | After T0 | After T1 | After T2 | After T3 | After T4 | After T5 | After T6 | After T7 | After T8 | After T9 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Task 0** | 0.9580 | 0.9540 | 0.9470 | 0.9480 | 0.9150 | 0.9180 | 0.9130 | 0.9100 | 0.9010 | 0.9010 |
| **Task 1** | - | 0.9760 | 0.9630 | 0.9360 | 0.9180 | 0.9150 | 0.9020 | 0.9020 | 0.9250 | 0.9180 |
| **Task 2** | - | - | 0.9470 | 0.9460 | 0.9370 | 0.9220 | 0.9200 | 0.9220 | 0.9290 | 0.9080 |
| **Task 3** | - | - | - | 0.9610 | 0.9550 | 0.9300 | 0.9260 | 0.9110 | 0.9440 | 0.9080 |
| **Task 4** | - | - | - | - | 0.9690 | 0.9450 | 0.9390 | 0.9440 | 0.9540 | 0.9490 |
| **Task 5** | - | - | - | - | - | 0.9300 | 0.9390 | 0.9160 | 0.9180 | 0.9060 |
| **Task 6** | - | - | - | - | - | - | 0.9320 | 0.9040 | 0.9210 | 0.9060 |
| **Task 7** | - | - | - | - | - | - | - | 0.9410 | 0.9460 | 0.9180 |
| **Task 8** | - | - | - | - | - | - | - | - | 0.9750 | 0.9680 |
| **Task 9** | - | - | - | - | - | - | - | - | - | 0.9650 |

---

### 5.6 CoMA Sweep ($\alpha = 0.90$) (Stage 28) — $10 \times 10$ Matrix

| Evaluated Task | After T0 | After T1 | After T2 | After T3 | After T4 | After T5 | After T6 | After T7 | After T8 | After T9 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Task 0** | 0.9410 | 0.9420 | 0.9060 | 0.9280 | 0.8970 | 0.8980 | 0.8890 | 0.8390 | 0.8820 | 0.8680 |
| **Task 1** | - | 0.9450 | 0.9320 | 0.9060 | 0.8740 | 0.9190 | 0.8730 | 0.8490 | 0.9030 | 0.8760 |
| **Task 2** | - | - | 0.9340 | 0.9220 | 0.9090 | 0.8890 | 0.8750 | 0.8700 | 0.9040 | 0.9070 |
| **Task 3** | - | - | - | 0.9490 | 0.9140 | 0.9100 | 0.8750 | 0.8790 | 0.9040 | 0.8910 |
| **Task 4** | - | - | - | - | 0.9640 | 0.9210 | 0.9160 | 0.9180 | 0.9220 | 0.9030 |
| **Task 5** | - | - | - | - | - | 0.9070 | 0.9220 | 0.8940 | 0.9000 | 0.9000 |
| **Task 6** | - | - | - | - | - | - | 0.9240 | 0.9060 | 0.9110 | 0.9100 |
| **Task 7** | - | - | - | - | - | - | - | 0.9430 | 0.9320 | 0.8990 |
| **Task 8** | - | - | - | - | - | - | - | - | 0.9720 | 0.9450 |
| **Task 9** | - | - | - | - | - | - | - | - | - | 0.9600 |

---

## 6. Execution Runtime & Hardware Profile

- **GPU Acceleration:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- **Host Operating System:** Windows 11 AMD64
- **Initial Dataset Download:** CIFAR-100 archive download ($169\,\text{MB}$) took $1,611.0\,\text{s}$ ($26\,\text{min}\,51\,\text{s}$).
- **Active Compute Execution Breakdown:**

| Phase | Stage / Description | Active Execution Time | Hardware Throughput / Task |
|:---|:---|:---:|:---:|
| **Stage 24** | Zero-Shot Evaluation (10 tasks) | $102.4\,\text{s}$ ($1.71\,\text{min}$) | ~$98$ test batches/s |
| **Stage 25** | Sequential CLIP Baseline ($\alpha=1.00$) | $1,623.4\,\text{s}$ ($27.06\,\text{min}$) | ~$162.3\,\text{s}$ per task |
| **Stage 26** | CLIP + CoMA Core ($\alpha=0.70$) | $1,728.8\,\text{s}$ ($28.81\,\text{min}$) | ~$172.9\,\text{s}$ per task |
| **Stage 27** | Core Method Comparison Table | $<1.0\,\text{s}$ | Instantaneous JSON calculation |
| **Stage 28** | CoMA $\alpha=0.25$ Sweep Run | $1,910.2\,\text{s}$ ($31.84\,\text{min}$) | ~$191.0\,\text{s}$ per task |
| **Stage 28** | CoMA $\alpha=0.50$ Sweep Run | $1,962.2\,\text{s}$ ($32.70\,\text{min}$) | ~$196.2\,\text{s}$ per task |
| **Stage 28** | CoMA $\alpha=0.70$ Sweep Reuse | $0.0\,\text{s}$ | Reused Stage 26 result directly |
| **Stage 28** | CoMA $\alpha=0.90$ Sweep Run | $1,848.8\,\text{s}$ ($30.81\,\text{min}$) | ~$184.9\,\text{s}$ per task |
| **Stage 28** | CoMA $\alpha=1.00$ Sweep Reuse | $0.0\,\text{s}$ | Reused Stage 25 result directly |
| **Total Active Training Time** | **5 Full Training Streams + 1 Zero-Shot** | **$9,175.8\,\text{s}$ ($152.93\,\text{min}$ / $2.55\,\text{hours}$)** | **50 task training episodes** |
| **Total Wall-Clock Time** | **Active Compute + Dataset Download** | **$10,786.8\,\text{s}$ ($179.78\,\text{min}$ / $3.00\,\text{hours}$)** | — |

---

## 7. Experimental Limitations

To maintain empirical rigor, the following specific constraints of the current experimental setup are documented:
1. **Epoch Count:** All training stages used **1 epoch per task** ($E=1$, rapid evaluation protocol). Standard benchmark papers frequently evaluate across 5–10 epochs per task with learning rate scheduling.
2. **Single Random Seed:** All experiments were executed using **seed 42**. While consistent across all comparative conditions, multi-seed averaging (e.g., seeds 42, 123, 456) with standard deviations is required for formal publication.
3. **Fixed Class Ordering:** Tasks followed the default CIFAR-100 class index partition ($0..9, 10..19, \dots$). Testing randomized task permutations is recommended to verify invariance to task sequence difficulty.
4. **Single Model Architecture:** Evaluation was performed exclusively on OpenCLIP `ViT-B-32-quickgelu` with OpenAI pre-trained weights. Scaling behaviors across larger backbones (e.g., ViT-B/16, ViT-L/14) remain unverified.
