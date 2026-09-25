# When Do Reliability Weights Help? A Multi-Seed Study of Calibrated Multimodal Prediction Under Modality Loss

Code for the AAAI 2026 student abstract by Nisarg Trivedi (Dhirubhai Ambani University).

We add a small **per-modality reliability head** to Adaptive Confidence Regularization (ACR) (Liu et al., CVPR 2026). Each head is trained on whether its modality alone classifies the sample correctly, and the resulting weights rescale the embeddings before fusion. We compare the method against plain ACR in **matched ON/OFF pairs over three seeds** on the Human–Animal–Cartoon (HAC) benchmark.

**Findings (mean over 3 seeds):**

- **Failure detection:** no robust gain. All-modality ΔAURC is +0.6 ± 3.0. Two identical ACR runs differ by 6.6 AURC, and no modality-loss condition clears that noise floor.
- **Calibration under severe modality loss:** when the dominant video modality is dropped, ECE falls from **0.143 ± 0.011 (ACR)** to **0.047 ± 0.005 (ours)**, and no seed reverses the gap. NLL is about the same (~0.77). Under mild loss (audio or flow dropped), plain ACR is slightly better calibrated.
- **Interpretability:** on every seed the learned weights rank video > audio ≈ flow. They also track the AURC increase when each modality is removed (Pearson r = 0.94, Spearman ρ = 0.75, over 9 points).

Dataset: [Human-Animal-Cartoon on Hugging Face](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon)

---

## Method

For each modality k ∈ {video, flow, audio} with embedding E_k:

```
r_k   = σ(MLP(E_k))                 # one hidden layer, width 128, soft-initialised so r_k ≈ 0.95
Ẽ_k   = r_k · E_k                   # reliability-weighted embedding
fused = Linear([Ẽ_v ; Ẽ_f ; Ẽ_a])   # C+1 logits (C = 7; the ACR outlier column is ignored at inference)
```

- **Free supervision.** The target for r_k is `1[argmax(unimodal_logits_k) == y]`, so no extra labels are needed. The loss is `L_rel = mean_k BCE(r_k, t_k)`, with the target detached.
- **Objective.** `L = L_cls + L_out + 2·L_acl + λ_rel·L_rel`, with λ_rel = 1. ACL and multimodal feature swapping (MFS) are the same as in ACR and act on the reweighted embeddings.
- **Missing modality.** The absent modality is removed and the remaining weights are renormalised. The plain-ACR baseline has no weights to renormalise, so its absent embedding is set to zero.
- **Only difference between ON and OFF:** whether the reliability head is active (`--reliability`). The script, seed and hyperparameters are the same.

| Modality | Backbone (pretraining)           | Dim  |
| -------- | -------------------------------- | ---: |
| Video    | SlowFast-R101 (Kinetics-400)     | 2304 |
| Flow     | SlowOnly-R50 (Kinetics-400 flow) | 2048 |
| Audio    | VGGSound ResNet-18               |  512 |

---

## Repository layout

Everything used in the paper is under `HMDB-rgb-flow/`. The folder name comes from the original codebase; the experiments use **HAC only**.

```text
HMDB-rgb-flow/
├── train_video_flow_hac_reliability.py         # main trainer: ON (--reliability) and OFF (plain ACR)
├── test_video_flow_hac_reliability.py          # FD metrics + mean r_k for ON checkpoints, supports --drop
├── test_video_flow_hac_acr_missing_modality.py # FD metrics for OFF checkpoints, supports --drop
├── eval_hac_calibration.py                     # ECE / NLL (+ FD) for ON or OFF checkpoints, supports --drop
├── plot_reliability_diagram.py                 # reliability diagrams from eval_hac_calibration.py dumps
├── train_video_flow_hac_acr.py                 # standalone 3-modality ACR reproduction (sanity check)
├── test_video_flow_hac_acr.py                  # test script for the standalone ACR reproduction
├── acr_modules.py                              # ACL, MFS, MSP confidence, AURC/AUROC/FPR95
├── dataloader_video_flow_hac_audio.py          # HAC loader (video + flow + audio), seeded val split
├── dataloader_video_flow_hac.py                # HAC loader (video + flow only)
├── splits/HAC_{train,test}_only_{human,animal,cartoon}.csv
├── configs/  mmaction/  VGGSound/              # backbone configs and model code
├── pretrained_models/                          # backbone weights go here (not tracked)
├── calib_dump_video_{ON,OFF}.npz               # per-sample dumps for the drop-video reliability diagram
└── reliability_drop_video.png                  # the resulting diagram
```

The following files are **not needed** to reproduce the paper:

- `eval_phase*.py`: early HMDB exploration scripts.
- The HMDB, Kinetics and UCF split files in `splits/`.
- `PAPER_SKELETON.md` and `READ_ME_ACR_FIRST.md`: working notes.

---

## Installation

The environment is Python 3.8 with older versions of PyTorch, MMCV and MMAction2.

```bash
conda env create -f environment.yml      # or environment_exact.yml for the fully pinned build
conda activate acr
```

Core versions: PyTorch 1.11.0 + CUDA 11.3, TorchVision 0.12.0, MMCV-full 1.2.7, MMAction2 0.13.0, NumPy 1.23.5, SciPy 1.10.1, SoundFile 0.11.0.

---

## Data

Download HAC from the [Hugging Face dataset page](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon) and arrange it as follows (`--datapath` points to the folder that **contains** `HAC/`):

```text
<datapath>/HAC/
├── human/   ├── videos/<name>.mp4
│            ├── flow/<name>_flow_x.mp4, <name>_flow_y.mp4
│            └── audio/<name>.wav
├── animal/  (same layout)
└── cartoon/ (same layout)
```

The dataset has 3,381 clips and 7 action classes. The three domains are pooled into one closed-set problem, and the test set has 670 clips. A 15% validation split is taken from the training clips using the run seed (`--seed`, `--val_frac 0.15`).

The helpers in `utils/` (`video2flow.py`, `flow_img2mp4.py`, `generate_audio_files.py`) can regenerate flow and audio from raw video if needed.

## Pretrained backbones

Place these in `HMDB-rgb-flow/pretrained_models/`:

| File | Source |
| ---- | ------ |
| `slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth` | [MMAction2 model zoo](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) |
| `slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth` | [MMAction2 model zoo](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) |
| `vggsound_avgpool.pth.tar` | [VGGSound "model H"](https://www.dropbox.com/s/jhyy73z5l0mjq23/vggsound_avgpool.pth.tar?dl=0) |

---

## Reproducing the paper

Run all commands from `HMDB-rgb-flow/`. The paper uses seeds **0, 1, 2**. Checkpoint names do **not** include the seed, so give each run its own `--appen` suffix, otherwise later seeds overwrite earlier ones.

### 1. Train matched ON/OFF pairs

```bash
cd HMDB-rgb-flow
for SEED in 0 1 2; do
  # ON: reliability-weighted fusion
  python train_video_flow_hac_reliability.py --datapath <datapath> --seed $SEED \
      --lr 1e-4 --bsz 16 --nepochs 50 --lambda_acl 2.0 --n_min 32 --n_max 256 \
      --reliability --lambda_rel 1.0 \
      --select aurc --patience 12 --save_best --appen hac_rel_s${SEED}_

  # OFF: plain ACR, same script / seed / hyperparameters
  python train_video_flow_hac_reliability.py --datapath <datapath> --seed $SEED \
      --lr 1e-4 --bsz 16 --nepochs 50 --lambda_acl 2.0 --n_min 32 --n_max 256 \
      --select aurc --patience 12 --save_best --appen hac_off_s${SEED}_
done
```

The best checkpoint is chosen by validation AURC and saved to `models/log_video_flow_audio_HAC_..._best.pt`.

### 2. Failure detection, all modalities and each modality dropped (Table 1)

```bash
# ON checkpoint (also prints mean r_video / r_audio / r_flow, used in the supplement's Table 2)
python test_video_flow_hac_reliability.py --datapath <datapath> --seed $SEED --resumef <on_best.pt>
python test_video_flow_hac_reliability.py --datapath <datapath> --seed $SEED --resumef <on_best.pt> --drop video   # also: flow, audio

# OFF checkpoint
python test_video_flow_hac_acr_missing_modality.py --datapath <datapath> --seed $SEED --resumef <off_best.pt>
python test_video_flow_hac_acr_missing_modality.py --datapath <datapath> --seed $SEED --resumef <off_best.pt> --drop video   # also: flow, audio
```

Each run prints AURC (×1000, lower is better), AUROC, FPR95 and accuracy. Confidence is the maximum softmax probability over the 7 real classes.

**Modality importance** (supplement, Table 2 and Fig. 2) is the AURC with modality k dropped minus the all-modality AURC, both from the same ON checkpoint.

### 3. Calibration (Table 2, Fig. 1)

```bash
python eval_hac_calibration.py --datapath <datapath> --seed $SEED --resumef <ckpt_best.pt>                # all modalities
python eval_hac_calibration.py --datapath <datapath> --seed $SEED --resumef <ckpt_best.pt> --drop video   # also: flow, audio
```

The script detects whether a checkpoint is ON or OFF and applies the matching fusion and masking. It reports ECE (15 equal-width bins) and NLL, and writes `calib_dump_<drop>_<ON|OFF>.npz` for the reliability diagrams.

### 4. Reliability diagram

```bash
python plot_reliability_diagram.py --on calib_dump_video_ON.npz --off calib_dump_video_OFF.npz \
    --title "Drop video (severe)" --out reliability_drop_video.png
```

### Aggregation

The paper reports the mean ± std over the three seeds of the per-seed numbers printed above. Every value was computed by reloading the saved checkpoint and evaluating on the test set, never read from a training log. ON/OFF pairs were checked by the classifier-head dimension stored in each checkpoint, not by filename.

### Optional: ACR reproduction check

`train_video_flow_hac_acr.py` / `test_video_flow_hac_acr.py` are a standalone 3-modality ACR implementation. They were used to confirm the pipeline before adding the reliability head. The OFF runs above produce the paper's baseline numbers.

---

## Metrics

| Metric | Meaning | Better |
| ------ | ------- | ------ |
| AURC (×1000) | Area under the risk–coverage curve | lower |
| AUROC (%) | Separation of correct vs. incorrect predictions | higher |
| FPR95 (%) | FPR on errors at 95% TPR on correct predictions | lower |
| ECE | Expected calibration error, 15 bins | lower |
| NLL | Negative log-likelihood of the true class | lower |

---

## Acknowledgements

This code builds on the MultiOOD / ACR codebase, [MMAction2](https://github.com/open-mmlab/mmaction2), and [VGGSound](https://github.com/hche11/VGGSound). The HAC dataset is from SimMMDG (Dong et al., NeurIPS 2023).

## References

- Liu, M. et al. 2026. *Adaptive Confidence Regularization for Multimodal Failure Detection.* CVPR. arXiv:2603.02200.
- Dong, H. et al. 2023. *SimMMDG: A Simple and Effective Framework for Multimodal Domain Generalization.* NeurIPS.
- Geifman, Y. and El-Yaniv, R. 2017. *Selective Classification for Deep Neural Networks.* NeurIPS.
- Guo, C. et al. 2017. *On Calibration of Modern Neural Networks.* ICML.
- Hendrycks, D. and Gimpel, K. 2017. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR.
- Ma, H. et al. 2023. *Calibrating Multimodal Learning.* ICML.
- Feichtenhofer, C. et al. 2019. *SlowFast Networks for Video Recognition.* ICCV.
