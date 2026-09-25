# Reliability-Aware Multimodal Failure Detection

This repository contains the code for **reliability-aware multimodal failure detection** using video, optical flow, and audio modalities.

The project builds on the **ACR (Adaptive Confidence Regularization)** framework for multimodal failure detection and extends it with **per-modality reliability estimation**. The reliability module learns how trustworthy each modality is for a given sample and uses these estimates to reweight the multimodal representation before classification.

The repository also contains the original multimodal/OOD evaluation pipelines for HMDB, EPIC, and related datasets.

---

## Overview

Multimodal models typically combine information from several modalities such as:

* 🎥 RGB video
* 🌊 Optical flow
* 🔊 Audio

A conventional fusion model treats these modalities similarly during fusion. However, the reliability of a modality can vary significantly from sample to sample. For example:

* Audio may be noisy or unavailable.
* Optical flow may be unreliable because of poor motion information.
* Video may provide the strongest evidence for a particular action.

This repository introduces a **reliability-aware fusion mechanism** that estimates a reliability score for each modality and uses it to adapt the fusion process.

### Core idea

For each modality \(k\), the model predicts a reliability value:

$$
r_k = \sigma(\mathrm{MLP}(E_k)), \qquad r_k \in [0,1]
$$

where \(E_k\) is the modality embedding.

The embeddings are then reweighted:

$$
E'_k = r_k E_k
$$

and concatenated before the final fusion classifier:

$$
E_{\mathrm{fusion}}
=
[E'_v;E'_a;E'_f].
$$

The reliability heads are supervised using the correctness of the corresponding unimodal predictions, without requiring additional modality-reliability labels.

---

# Repository Structure

```text
mmd-main/
│
├── HMDB-rgb-flow/
│   ├── acr_modules.py
│   │
│   ├── train_video_flow.py
│   ├── test_video_flow.py
│   │
│   ├── train_video_flow_CE.py
│   ├── test_video_flow_CE.py
│   │
│   ├── train_video_flow_acr.py
│   ├── test_video_flow_acr.py
│   │
│   ├── train_video_flow_hac_acr.py
│   ├── test_video_flow_hac_acr.py
│   │
│   ├── train_video_flow_hac_reliability.py
│   ├── test_video_flow_hac_reliability.py
│   ├── test_video_flow_hac_acr_missing_modality.py
│   ├── test_video_flow_hac_reliability.py
│   │
│   ├── eval_hac_calibration.py
│   ├── eval_phase1_degradation.py
│   ├── eval_phase2a_outlier_head.py
│   ├── eval_phase3_missing_modality.py
│   ├── plot_reliability_diagram.py
│   │
│   ├── dataloader_video_flow.py
│   ├── dataloader_video_flow_audio.py
│   ├── dataloader_video_flow_hac.py
│   ├── dataloader_video_flow_hac_audio.py
│   │
│   ├── configs/
│   ├── pretrained_models/
│   ├── models/
│   └── splits/
│
├── EPIC-rgb-flow/
│   ├── train_video_flow.py
│   ├── train_video_flow_audio_epic.py
│   ├── train_video_flow_epic.py
│   ├── test_video_flow.py
│   ├── test_video_flow_audio_epic.py
│   ├── test_video_flow_epic.py
│   └── ...
│
├── VGGSound/
│   ├── model.py
│   ├── datasets/
│   ├── models/
│   ├── preprocess_audio.py
│   ├── train/evaluation utilities
│   └── data/
│
├── utils/
│   ├── video2flow.py
│   ├── flow_img2mp4.py
│   └── generate_audio_files.py
│
├── metrics.py
├── eval_video_flow_far_ood.py
├── eval_video_flow_near_ood.py
│
├── environment.yml
├── environment_exact.yml
└── requirement.txt
```

---

# Method

The reliability-aware model consists of three modality-specific encoders:

```text
             RGB Video
                 │
           Video Encoder
                 │
              E_video
                 │
          Reliability Head
                 │
              r_video
                 │
              r·E_video
                 │
                 │
Audio ──> Audio Encoder ──> E_audio ──> Reliability ──> r·E_audio
                 │
                 │
Flow ──> Flow Encoder ──> E_flow ──> Reliability ──> r·E_flow
                 │
                 └───────────┬───────────────┘
                             │
                       Concatenation
                             │
                       Fusion Classifier
                             │
                       Failure Detection
```

## Reliability Learning

The target for each reliability head is derived from the unimodal classifier:

$$
t_k =
\mathbf{1}
[
\arg\max(\hat{y}_k)=y
].
$$

The reliability loss is:

$$
\mathcal{L}_{rel}
=
\frac{1}{M}
\sum_k
\mathrm{BCE}(r_k,t_k).
$$

The complete training objective combines classification, outlier, ACR, and reliability losses:

$$
\mathcal{L}
=
\mathcal{L}_{cls}
+
\mathcal{L}_{outlier}
+
\lambda_{acl}\mathcal{L}_{acl}
+
\lambda_{rel}\mathcal{L}_{rel}.
$$

The current implementation uses:

```text
lambda_acl = 2.0
lambda_rel = 1.0
```

---

# Modalities

The main HAC/HMDB multimodal experiments use:

| Modality     | Backbone                            | Feature dimension |
| ------------ | ----------------------------------- | ----------------: |
| Video        | SlowFast-R101                       |              2304 |
| Optical Flow | SlowOnly-R50                        |              2048 |
| Audio        | VGGSound / ResNet-based audio model |               512 |

The fused representation therefore has:

```text
2304 + 2048 + 512 = 4864 dimensions
```

before being passed to the fusion classifier.

---

# Datasets

The repository contains pipelines for multiple datasets, including:

* **HMDB51**
* **HAC**
* **EPIC**
* **VGGSound**
* **Kinetics**
* **UCF**

The reliability-aware experiments primarily use the **HAC** dataset, while HMDB is also used for multimodal and reliability experiments.

## HAC

The HAC experiments use:

* 3,381 video clips
* 7 action classes
* Video
* Optical flow
* Audio

The experiments can use a validation split carved from the training data.

Default validation fraction:

```text
0.15
```

---

# Installation

The original environment is based on Python 3.8 and older versions of PyTorch/MMCV/MMACTION2.

Create the environment using:

```bash
conda env create -f environment.yml
conda activate acr
```

Alternatively:

```bash
conda create -n acr python=3.8
conda activate acr
pip install -r requirement.txt
```

The pinned environment includes approximately:

```text
Python       3.8
PyTorch      1.11.0 + CUDA 11.3
TorchVision  0.12.0 + CUDA 11.3
MMCV         1.2.7
MMAction2    0.13.0
NumPy        1.23.5
Pandas       1.4.2
SciPy        1.10.1
SoundFile    0.11.0
```

> **Note:** These are legacy dependencies. Newer CUDA/PyTorch systems may require adapting the environment or using the provided exact environment configuration.

---

# Data Preparation

## HMDB51

The expected HMDB directory structure is:

```text
~/data/hmdb51/
├── video/
│   ├── class_1/
│   │   ├── video1.avi
│   │   └── ...
│   ├── class_2/
│   └── ...
│
└── flow/
    ├── video1_flow_x.mp4
    ├── video1_flow_y.mp4
    └── ...
```

Use the original HMDB51 filenames and splits.

The repository's HMDB pipeline expects the original, unsanitized filenames.

---

# Pretrained Models

The main HMDB/HAC pipeline uses pretrained video, optical-flow, and audio models.

Place the required checkpoints in:

```text
HMDB-rgb-flow/pretrained_models/
```

Important checkpoints include:

```text
slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth

slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth

vggsound_avgpool.pth.tar
```

These pretrained weights are required by the corresponding training/evaluation scripts.

---

# Training

Move into the HMDB directory:

```bash
cd HMDB-rgb-flow
```

## ACR baseline

The ACR baseline can be trained using:

```bash
python train_video_flow_hac_acr.py \
    --datapath ~/data/hac/ \
    --lr 1e-4 \
    --bsz 16 \
    --nepochs 50 \
    --num_workers 2 \
    --lambda_acl 2.0 \
    --save_best \
    --appen acr_
```

---

# Reliability-Aware Training

The reliability-aware model is trained using:

```bash
python train_video_flow_hac_reliability.py
```

A typical configuration is:

```bash
python train_video_flow_hac_reliability.py \
    --datapath ~/data/hac/ \
    --lr 1e-4 \
    --bsz 16 \
    --nepochs 50 \
    --num_workers 2 \
    --lambda_acl 2.0 \
    --lambda_rel 1.0 \
    --save_best \
    --appen reliability_
```

The exact arguments available can be checked with:

```bash
python train_video_flow_hac_reliability.py --help
```

---

# Evaluation

After training, evaluate a reliability-aware checkpoint with:

```bash
python test_video_flow_hac_reliability.py \
    --datapath ~/data/hac/ \
    --resumef models/<CHECKPOINT>.pt
```

For a complete list of options:

```bash
python test_video_flow_hac_reliability.py --help
```

---

# Missing-Modality Evaluation

One of the main experiments evaluates how the model behaves when a modality is unavailable at inference time.

The reliability-aware model supports:

```text
All modalities
Audio missing
Flow missing
Video missing
```

For example:

```bash
python test_video_flow_hac_reliability.py \
    --datapath ~/data/hac/ \
    --resumef models/<CHECKPOINT>.pt \
    --drop audio
```

Other options:

```bash
--drop flow
```

or:

```bash
--drop video
```

When a modality is missing, its reliability is set to zero and the surviving modalities are renormalized.

Conceptually:

$$
r_k = 0
$$

for the missing modality, followed by normalization of the remaining reliability weights.

No additional retraining is required for the missing-modality evaluation.

---

# Failure Detection Metrics

The repository evaluates multimodal failure detection using metrics including:

### AURC

Area Under the Risk-Coverage Curve.

Lower values indicate better selective prediction behavior.

### AUROC

Area Under the Receiver Operating Characteristic curve.

Higher values indicate better separation between reliable and failed predictions.

### FPR95

False Positive Rate at 95% True Positive Rate.

Lower values indicate better failure/OOD separation.

### Accuracy

Classification accuracy on the evaluated samples.

### ECE

Expected Calibration Error.

Used to evaluate whether confidence estimates are calibrated.

### NLL

Negative Log-Likelihood.

Used as an additional calibration metric.

---

# Reliability Evaluation

The repository contains scripts for analyzing the learned reliability estimates.

Important scripts include:

```text
eval_hac_calibration.py
eval_phase1_degradation.py
eval_phase2a_outlier_head.py
eval_phase3_missing_modality.py
plot_reliability_diagram.py
```

These can be used to investigate:

* Confidence degradation
* Calibration
* Missing-modality robustness
* Outlier detection
* Reliability behavior
* Reliability diagrams

---

# OOD Evaluation

The repository also includes near-OOD and far-OOD evaluation pipelines.

## Far-OOD

```bash
python eval_video_flow_far_ood.py
```

Supported post-processing methods include:

```text
MSP
EBO
MaxLogit
Mahalanobis
ASH
ReAct
kNN
GEN
ViM
```

Arguments can be inspected using:

```bash
python eval_video_flow_far_ood.py --help
```

## Near-OOD

```bash
python eval_video_flow_near_ood.py
```

The same family of post-processing methods is supported.

---

# Baselines

The repository contains implementations/evaluation pipelines for several approaches.

These include:

* Standard multimodal classification
* Cross-Entropy baseline
* ACR
* Reliability-aware fusion
* MSP-based confidence estimation
* Several OOD post-processing methods

The ACR implementation serves as the main baseline for the reliability-aware experiments.

---

# Reproducibility

For reproducible experiments, explicitly set the random seed:

```bash
--seed 0
```

The reliability evaluation scripts also use deterministic CUDA settings where applicable.

For multiple-seed experiments, run the same configuration with:

```text
--seed 0
--seed 1
--seed 2
```

and report mean and standard deviation across runs.

---

# Recommended Experiment Workflow

A typical experiment can be organized as follows:

### 1. Prepare datasets

```text
Dataset
   ↓
RGB videos
   ↓
Optical flow
   ↓
Audio
```

### 2. Prepare pretrained backbones

```text
SlowFast-R101
SlowOnly-R50
VGGSound
```

### 3. Train ACR baseline

```bash
python train_video_flow_hac_acr.py ...
```

### 4. Train reliability-aware model

```bash
python train_video_flow_hac_reliability.py ...
```

### 5. Evaluate full-modality performance

```bash
python test_video_flow_hac_reliability.py ...
```

### 6. Evaluate missing modalities

```bash
--drop audio
--drop flow
--drop video
```

### 7. Evaluate calibration

```bash
python eval_hac_calibration.py ...
```

### 8. Generate reliability plots

```bash
python plot_reliability_diagram.py ...
```

---

# Key Research Questions

The codebase is designed to investigate the following questions:

### Q1. Does modality reliability improve failure detection?

Compare:

```text
ACR
vs.
Reliability-aware ACR
```

using:

```text
AURC
AUROC
FPR95
ACC
```

### Q2. Does reliability-aware fusion degrade gracefully?

Remove individual modalities at inference:

```text
Audio
Flow
Video
```

and measure the resulting change in failure-detection performance.

### Q3. Are learned reliability values meaningful?

Compare learned reliability scores against independent measurements of modality importance or masking sensitivity.

### Q4. Does reliability improve calibration?

Evaluate:

```text
ECE
NLL
```

under both full-modality and missing-modality conditions.

---

# Important Implementation Details

The reliability heads are modality-specific:

```text
Video → ReliabilityHead
Audio → ReliabilityHead
Flow  → ReliabilityHead
```

Each head produces a scalar value in:

```text
[0, 1]
```

The scalar is applied to every dimension of the corresponding modality embedding.

For example:

```python
v_weighted = v_embedding * r_video
```

The three weighted embeddings are then concatenated:

```python
fusion = torch.cat(
    (v_weighted, a_weighted, f_weighted),
    dim=1
)
```

This preserves the existing fusion classifier while introducing modality-level reliability weighting.

---

# Project Status

The repository contains both the original multimodal/OOD pipeline and the experimental reliability-aware extension.

The reliability-aware experiments should be interpreted according to the exact dataset split, random seed, checkpoint, and evaluation protocol used.

In particular, results from different protocols should **not** be directly compared unless their:

* Dataset
* Split
* Modalities
* Backbone
* Training procedure
* Evaluation procedure

are matched.


---

# Acknowledgements

This project builds upon publicly available multimodal video understanding, OOD detection, and failure-detection research code, including components from the MMAction2 ecosystem and the ACR-based failure-detection pipeline.

Please refer to the respective upstream projects and licenses before redistributing modified components.

---

