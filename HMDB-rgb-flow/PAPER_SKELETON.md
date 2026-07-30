# Paper Skeleton — Reliability-Aware Multimodal Failure Detection
Target: submission-ready draft by ~July 15, buffer to July 18.
Write NOW where marked [WRITE NOW] — does not wait on any experiment.
Everything else is [WAIT: X] — gated on a specific pending result.

================================================================================
## Title (draft, iterate later)
"Reliability-Aware Fusion for Multimodal Failure Detection"
(avoid over-claiming "beats ACR" in the title — frame as extension/axis, matches
 what you can actually defend: gains on FD + missing-modality, honest tie on
 primary-modality loss.)

================================================================================
## Abstract  [WRITE LAST — 150-200 words, do after everything else is drafted]
Template shape: (1) multimodal FD needs per-modality trust, ACR doesn't model it
(2) we add per-modality reliability, learned for free from unimodal correctness
(3) two results: FD gain (X AURC / X AUROC, seeded), graceful degradation under
    secondary-modality loss where the gain GROWS, honest tie under primary loss
(4) mechanism is interpretable: learned weights match independent modality-
    importance evidence.

================================================================================
## 1. Introduction  [WRITE NOW — you know this cold]
- Multimodal FD matters (safety-critical systems, ACR's own motivation — can
  paraphrase, don't quote).
- ACR (Liu et al., CVPR 2026) is the SOTA: ACL + MFS. State what it does in
  2-3 sentences (you have this memorized).
- THE GAP (this is your whole paper in one paragraph): ACR regularizes
  confidence but treats every modality as equally trustworthy. It has no
  mechanism to represent "trust this modality more/less" — not per-sample,
  not under corruption, not when a modality is missing. We make that
  reliability explicit.
- Contributions (bullet list, map 1:1 to results you have):
  1. Reliability-aware fusion: per-modality reliability estimated for free
     from unimodal correctness, reweights modalities before fusion.
  2. [RESULT, pending seeds] FD gain over matched ACR baseline on HAC.
  3. [RESULT, done] Graceful degradation under secondary-modality loss,
     where the gain GROWS relative to the full-modality setting — with an
     honest, mechanistically-explained limit under primary-modality loss.
  4. [RESULT, pending calibration script run] Calibration analysis ACR does
     not report.
  5. [OPTIONAL, pending interpretability-validation script] Learned
     reliability correlates with measured masking sensitivity — a validated,
     not decorative, interpretability signal.
- DO NOT claim "we beat ACR" broadly — claim specific axes. Do not claim
  reproduction of their Table 3 (protocol mismatch, documented in memory).

================================================================================
## 2. Related Work  [WRITE NOW — you know this cold]
- Failure detection: MSP, ConfidNet, OpenMix, CRL — one paragraph, paraphrase
  paper's own related-work framing, do not copy structure/wording.
- Multimodal OOD/FD: MultiOOD (Dong et al. 2024), A2D. One paragraph.
- ACR itself: describe faithfully (ACL Eq.3, MFS Eq.4, C+1 outlier class,
  inference discards outlier column). This is the method you extend.
- POSITIONING PARAGRAPH (important, write carefully): official ACR code was
  released (CVPR 2026) during this project. State plainly: we build on/compare
  against the released implementation; our released-code HAC baseline uses
  the same 2-stream training path extended to 3 modalities per the paper's own
  Eq.7/Algorithm 2 (which the release does not ship in a runnable 3-modality
  form); reliability-aware fusion, missing-modality handling, and calibration
  analysis are absent from the release.
- Reliability/trust-weighted fusion in OTHER contexts (sensor fusion,
  robotics, multimodal fusion broadly) — a citation-search TODO, likely exists
  under different names (attention-based fusion, gating networks, modality
  dropout literature). [TODO: literature search before final draft — do not
  claim reliability-weighting is unprecedented in ML broadly, only that it's
  absent from the FD/ACR context specifically.]

================================================================================
## 3. Method  [WRITE NOW — fully designed, fully implemented]
### 3.1 Background: ACR
Eq: L_acl = (1/M) sum_k max(0, conf_k - conf)   [cite paper Eq.3/7]
MFS soft label: y_swapped = (1-lam) y_true + lam y_outlier   [cite Eq.4]

### 3.2 Per-modality reliability
- Motivation: ACR's confidence-degradation observation implicitly treats
  degradation as always undesirable; but degradation can also correctly
  signal that a modality's information is being appropriately down-weighted.
  ACR has no way to express "modality k is less reliable HERE" -- only a
  global penalty against fused confidence dropping below unimodal confidence.
- Definition: r_k = sigma(MLP(E_k)) in [0,1], one small head per modality.
- Supervision: target = 1[argmax(unimodal head_k) == y], DETACHED (free --
  no extra labels, no extra forward pass beyond what ACR already computes).
- Loss: L_rel = (1/M) sum_k BCE(r_k, target_k).
- Soft-init: final-layer bias initialized so r_k ~ 1 at step 0 (state WHY:
  avoids cold-start corruption of the fused representation; reliability is
  active from epoch 0, ablation could show hard-init is unstable -- OPTIONAL
  ablation if time permits, else just state the design choice and rationale).

### 3.3 Reliability-weighted fusion
E_k' = r_k * E_k  (elementwise scale by scalar r_k per sample per modality)
Fused input = concat(E_v', E_a', E_f')  -- same fusion head as ACR, unchanged.
ACL and MFS operate on the REWEIGHTED embeddings (state this explicitly --
it's why the method composes with ACR's existing losses rather than replacing
them).

### 3.4 Missing-modality inference
At test time, if modality k is unavailable: r_k := 0, then renormalize
surviving r_j := r_j * (M / sum of surviving r), preserving fused-input scale.
No retraining required -- the SAME trained model handles missing modalities.
Contrast explicitly with plain ACR: no mechanism to represent "modality k is
absent" other than feeding zeros, which the fusion head was never trained to
expect.

### 3.5 Total loss
L = L_cls + L_outlier + lambda_acl * L_acl + lambda_rel * L_rel
[state hyperparameters: lambda_acl=2.0 (from ACR paper), lambda_rel=1.0
(your choice -- mention if you did/do a sweep, else state as a design choice
with a note that tuning is future work)]

================================================================================
## 4. Experimental Setup  [WRITE NOW, fill numbers when ready]
- Dataset: HAC (Dong et al.), 3,381 clips, 7 actions, human/animal/cartoon,
  video+flow+audio. [STATE PROTOCOL EXPLICITLY: pooled across domains,
  train/val/test with val carved from train (val_frac=0.15, stratified,
  seeded) -- NOTE this differs from the official release's single-domain
  (cartoon) evaluation; state why (pooled tests general multimodal reliability
  rather than one visual domain) and flag as a limitation/difference, not a
  reproduction.]
- Backbones: SlowFast-R101 (video), SlowOnly-R50 (flow), ResNet-18/VGGSound
  (audio) -- all matching the paper's stated architecture (Sec 3.1 of ACR).
- Training: Adam, lr 1e-4, bsz 16, up to 50 epochs, early stop patience 12,
  model selected on val AURC. [N] seeds: 0 [, 1, 2 -- fill in].
- Baselines: ACR (matched -- same script, --reliability off; NOT the paper's
  own reported numbers, which use a different protocol -- state this clearly
  to preempt a reviewer asking "why don't your numbers match Table 3").
- Metrics: AURC (x1000), AUROC, FPR95, ACC (standard FD, cite OpenMix/ACR
  convention) + ECE, NLL (calibration, novel to this comparison).

================================================================================
## 5. Results
### 5.1 Failure detection  [WAIT: seeds 1,2 finishing]
Table: ON vs OFF, mean +/- std across seeds. Currently seed-0-only:
  AURC 11.71 vs 15.90 (Delta -4.19) / AUROC 93.59 vs 90.43 (+3.16) /
  FPR95 39.06 vs 56.06 (-16.99) / ACC 90.45 vs 90.15 (+0.30)
[DO NOT WRITE THE FINAL TABLE UNTIL SEEDS LAND -- draft the table structure now,
 fill numbers later.]

### 5.2 Missing-modality robustness  [DONE — seed 0]
Table (have this exact data already, see memory):
  All-modalities: Rel 11.71/93.59/39.06/90.45 vs ACR 15.90/90.43/56.06/90.15
  Drop audio:     Rel 15.27/92.92/43.24/88.96 vs ACR 21.18/88.16/56.52/89.70
  Drop flow:      Rel 14.99/91.87/56.52/89.70 vs ACR 21.54/88.74/61.33/88.81
  Drop video:     Rel 78.55/82.65/73.29/75.97 vs ACR 76.39/82.71/70.13/77.01
Narrative (already validated, write this now):
  "The reliability gain over ACR GROWS under secondary-modality loss (+5.9,
  +6.5 AURC vs +4.2 at full modality) -- the method's advantage is largest
  exactly where it is designed to help. Under primary-modality (video) loss,
  the two methods are statistically indistinguishable (gap within the
  project's documented single-seed noise floor, Sec X) -- no reweighting
  scheme can recover information absent from the surviving modalities. This
  is consistent with the model's own learned reliability (r_video ~ 0.95,
  correctly identifying video as load-bearing) and with an independent
  finding on HMDB (video-only ACC >> flow-only ACC; different dataset,
  architecture, and modality set)."
CAVEAT to state: single seed on this table; illustrative given time budget;
flag as future work to extend with variance.

### 5.3 Calibration  [WAIT: run eval_hac_calibration.py, 8 calls]
Table: ECE/NLL for Rel vs ACR, all 4 conditions (all-modalities + 3 drops).
Hypothesis from HMDB pilot: Rel's confidence should collapse more honestly
under modality loss than ACR's -- confirm or report honestly if it doesn't.

### 5.4 Interpretability validation  [WAIT: build+run correlation script]
Correlate per-sample r_k against a measured masking-sensitivity signal (does
low r_flow predict that dropping flow flips this sample's prediction?).
Report a correlation coefficient, not just "the weights look sensible."
[NOT YET BUILT -- lowest priority given time; can be cut if time runs out,
 downgrade to a qualitative note (learned weights match expected ordering)
 if the quantitative version doesn't get built in time.]

================================================================================
## 6. Limitations  [WRITE NOW, mostly known]
- Single dataset (HAC). State plainly. Do not oversell generality.
- Missing-modality table is single-seed (if it stays that way).
- Video-loss tie: reweighting cannot manufacture missing information --
  state as a principled limitation, not a weakness of the method.
- Protocol differs from official release (pooled vs single-domain) --
  state why, and that it means headline numbers are not directly comparable
  to the paper's Table 3.
- lambda_rel not swept; single value used.

================================================================================
## 7. Conclusion  [WRITE LAST, 1 paragraph]

================================================================================
## WRITING ORDER (given July 18, prioritized)
1. Method (Sec 3) -- NOW, no dependencies.
2. Related Work (Sec 2) -- NOW, no dependencies.
3. Intro (Sec 1) -- NOW, but revisit once Sec 5 numbers are final (the
   contributions list must match what actually got proven).
4. Experimental Setup (Sec 4) -- NOW, skeleton; fill numeric details as they
   firm up.
5. Sec 5.2 (missing-modality) -- WRITE NOW, you have the final numbers.
6. Sec 5.3/5.4 -- write once those scripts run (this week).
7. Sec 5.1 (headline FD) -- write once seeds land (do NOT write final numbers
   before then -- this is the one table that can still change).
8. Limitations, Conclusion, Abstract -- last, after everything else is true.
