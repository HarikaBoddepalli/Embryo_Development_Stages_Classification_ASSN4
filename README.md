# Embryo Development Stage Classification

**Assignment 4 — Deep Learning for Medical Image Processing**

Automated 16-class morphokinetic stage classification of human embryo time-lapse images using custom loss functions and pretrained CNN architectures.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Custom Loss Function](#custom-loss-function)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

---

## Overview

In IVF (In Vitro Fertilization), embryologists assess embryo quality by observing its development through a time-lapse imaging incubator. This project automates that process by classifying each frame of an embryo video into one of **16 ordered developmental stages** — from polar body appearance all the way to blastocyst hatching.

The key focus of this assignment is:
1. Designing a **custom ordinal-aware loss function** that respects the ordered nature of the 16 stages
2. Comparing **5 pretrained CNN architectures** under identical training conditions
3. Training on the first publicly available embryo time-lapse dataset (~297K labeled images)

---

## Dataset

**Source:** [Kaggle — Embryo Dataset](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset) (originally from Zenodo by Gomez et al., 2022)

| Property | Value |
|---|---|
| Total embryo videos | 704 |
| Total labeled images | 297,428 |
| Image format | 500×500 grayscale JPEG |
| Focal planes available | 7 (F0, F±15, F±30, F±45) |
| Focal plane used | F0 (central plane) |
| Annotation format | CSV (frame index ranges per phase) |

### 16 Development Stages

| Label | Phase | Description |
|---|---|---|
| 0 | pPB2 | Second polar body detached |
| 1 | pPNa | Pro-nuclei appearance |
| 2 | pPNf | Pro-nuclei disappearance |
| 3 | p2 | 2-cell stage |
| 4 | p3 | 3-cell stage |
| 5 | p4 | 4-cell stage |
| 6 | p5 | 5-cell stage |
| 7 | p6 | 6-cell stage |
| 8 | p7 | 7-cell stage |
| 9 | p8 | 8-cell stage |
| 10 | p9+ | 9+ cells |
| 11 | pM | End of compaction |
| 12 | pSB | Start of blastulation |
| 13 | pB | Full blastocyst |
| 14 | pEB | Expanded blastocyst |
| 15 | pHB | Hatched blastocyst |

### Dataset Split

Split is done at the **embryo level** (not image level) to prevent data leakage — the same embryo never appears in both train and test sets.

| Split | Embryos | Images |
|---|---|---|
| Train | 492 (70%) | 210,408 |
| Validation | 105 (15%) | 42,611 |
| Test | 107 (15%) | 44,409 |

---

## Custom Loss Function

The total loss combines three components:

```
L_total = L_CCE + λ1 · L_DL + λ2 · L_PL
```

where **λ1 = 0.5** and **λ2 = 0.3**.

### Component 1 — Categorical Cross-Entropy (L_CCE)

Standard classification loss. Provides well-calibrated probability supervision.

```
L_CCE = -Σ_c [ y_c · log(p_c + ε) ]
```

### Component 2 — Distance Loss (L_DL)

Penalises predictions whose probability mass is spread far from the true class in ordinal terms. Defined as the **squared expected ordinal distance**:

```
L_DL = ( Σ_c [ p_c · |c - y| ] )²
```

**Proved properties:**
- **Monotonically increasing** — D² strictly increases as prediction mass moves away from true class
- **Differentiable** — D is linear in softmax outputs; D² is smooth everywhere. ∂L_DL/∂p_c = 2D·|c−y|
- **Faster convergence** — large gradient signal early in training when predictions are distant; acts as implicit curriculum

### Component 3 — Piecewise Penalty Loss (L_PL)

Introduces an asymmetric cost matrix that distinguishes minor misclassifications (adjacent stages) from major ones (distant stages):

```
L_PL = Σ_c [ p_c · W(y, c) ]

       ⎧  0                  if c = y
W(y,c)=⎨  α · |c−y|          if |c−y| ≤ δ   (linear zone)
       ⎩  β · |c−y|²         if |c−y| > δ   (quadratic zone)
```

Parameters: **α = 1.0**, **β = 2.0**, **δ = 3**

**Proved properties:**
- **Monotonically increasing** — W is non-decreasing in |c−y| since α, β > 0
- **Piecewise differentiable** — both linear and quadratic zones have well-defined gradients; boundary at δ=3 is a single non-differentiable point (never reached for integer distances)
- **Faster convergence** — quadratic penalty (β·d²) for distant errors creates very large gradients early in training, rapidly correcting large ordinal errors

---

## Models

All models use **ImageNet pretrained weights** with a custom classification head replacing the original top layers. Input images (grayscale) are decoded as 3-channel RGB for compatibility with pretrained weights.

### 1. MobileNetV1
- **Key idea:** Depthwise separable convolutions — ~8-9× fewer operations than standard convs
- **Head:** GAP(1024) → Dense(256) → BN → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(16)
- **Fine-tuning:** First 80% of layers frozen

### 2. MobileNetV2
- **Key idea:** Inverted residuals with linear bottlenecks — expands then compresses, no ReLU in bottleneck
- **Head:** GAP(1280) → Dense(256) → BN → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(16)
- **Fine-tuning:** First 80% of layers frozen

### 3. InceptionV3
- **Key idea:** Parallel multi-scale feature extraction (1×1, 3×3, 5×5 convolutions in each block)
- **Head:** GAP(2048) → Dense(512) → BN → Dropout(0.5) → Dense(256) → Dropout(0.4) → Dense(16)
- **Fine-tuning:** Last 50 layers trainable

### 4. VGG16
- **Key idea:** Deep sequential architecture with only 3×3 convolutions; depth over filter complexity
- **Head:** GAP(512) → Dense(512) → BN → Dropout(0.5) → Dense(256) → Dropout(0.4) → Dense(16)
- **Fine-tuning:** Block5 (last conv block) trainable

### 5. VGG19 + Residual Connections -- Best
- **Key idea:** VGG19 depth + skip connections in dense head to combat vanishing gradients
- **Residual Block 1:** Dense(512) + BN + ReLU + Dropout(0.3) + **Identity shortcut** (512→512)
- **Residual Block 2:** Dense(256) + BN + ReLU + Dropout(0.3) + **Projection shortcut** (512→256, no bias)
- **Head:** GAP(512) → ResBlock1 → ResBlock2 → Dropout(0.4) → Dense(16)
- **Fine-tuning:** Block5 trainable

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 64 |
| Max Epochs | 10 |
| EarlyStopping | patience=3, monitor=val_loss |
| ReduceLROnPlateau | factor=0.2, patience=2 |
| Loss | Custom (L_CCE + 0.5·L_DL + 0.3·L_PL) |

---

## Results

| Model | Train Acc | Val Acc | Test Acc | F1 (Weighted) | Epochs |
|---|---|---|---|---|---|
| MobileNetV1 | 84.76% | 62.18% | 61.08% | 0.5992 | 9 |
| MobileNetV2 | 86.28% | 61.37% | 57.92% | 0.5769 | 10 |
| InceptionV3 | 80.24% | 54.43% | 57.10% | 0.5527 | 10 |
| VGG16 | 83.32% | 60.80% | 59.58% | 0.5827 | 10 |
| **VGG19+Residual** | **81.63%** | **61.86%** | **63.12%** | **0.6202** | **9** |

**VGG19+Residual** achieves the best test accuracy and F1 despite not having the highest training accuracy — indicating better generalization due to the residual skip connections acting as an implicit regularizer.

### Key Observations

- The ~20% train-val accuracy gap across all models reflects the limited embryo-level diversity (492 training embryos), not dataset size
- InceptionV3 underperforms because multi-scale inception modules are designed for natural images; embryo classification is a texture/morphology task at a consistent scale
- MobileNetV2 overfits more than V1 — its higher expressiveness requires more training data diversity to generalize
- The custom loss enables all 5 models to converge within 10 epochs to well above random chance (6.25% for 16 classes)

---

## Project Structure

```
Assignment4_Embryo_Stages_Classification_ASSN4.ipynb   ← Main notebook
README.md                                               ← This file
Assignment4_Report.docx                                 ← Full assignment report
outputs/
  class_distribution.png     ← Class distribution across splits
  penalty_matrix.png         ← Custom loss penalty weight matrix W(y,c)
  sample_images.png          ← One sample image per developmental stage
  training_curves.png        ← Loss and accuracy curves for all 5 models
  final_comparison_complete.csv  ← Model comparison results table
  MobileNetV1_best.keras     ← Saved best model weights
  MobileNetV2_best.keras
  InceptionV1_best.keras
  VGG16_best.keras
  VGG19_Residual_best.keras
```

---

## How to Run

### 1. Open in Google Colab

Upload `Assignment4_Embryo_Stages_Classification_ASSN4.ipynb` to [Google Colab](https://colab.research.google.com) and set runtime to **T4 GPU**.

### 2. Get Kaggle API Key

Go to [kaggle.com](https://www.kaggle.com) → Profile → Settings → API → **Create New Token**. This downloads `kaggle.json`.

### 3. Run the Notebook

Run cells in order. When prompted in Step 3, upload your `kaggle.json`. The dataset (~12 GB) will download and extract automatically.


### Key Cells

| Cell | What it does |
|---|---|
| Step 1 | Install packages, set all hyperparameters |
| Step 2 | Mount Google Drive for saving outputs |
| Step 3 | Download full dataset via Kaggle API |
| Step 4 | Auto-detect folder paths, parse all 704 annotation CSVs |
| Step 5 | Parse annotations, build labeled DataFrame |
| Step 6 | Train/val/test split by embryo ID |
| Step 7 | Corrupt image scan and removal |
| Step 8 | tf.data pipeline with augmentation |
| Step 9 | Custom loss function definition |
| Step 10 | Build all 5 models, print model.summary() |
| Step 11 | Train all models with callbacks |
| Step 12 | Evaluate, plot curves, confusion matrices |

---

## Requirements

```
tensorflow>=2.19.0
numpy
pandas
opencv-python-headless
matplotlib
seaborn
scikit-learn
kaggle
tqdm
```

All installed automatically by the first notebook cell.

---

