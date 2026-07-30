"""
acr_modules.py  —  self-contained ACR components for the MultiOOD codebase.

Drop this file into  MultiOOD-main/HMDB-rgb-flow/  (next to train_video_flow.py).
It has no dependencies beyond torch / numpy / sklearn, so there is nothing to
wire up. `train_video_flow_acr.py` and `test_video_flow_acr.py` import from it.

Implements, directly from the ACR paper:
  * Adaptive Confidence Loss            ACL   (Eq. 3 / Eq. 7)
  * Multimodal Feature Swapping         MFS   (Algorithm 1 + Eq. 4)
  * failure-detection metrics           AURC, AUROC, FPR95, ACC (Section 3.1)

Conventions match the MultiOOD model:
  * fused head (mlp_cls) outputs C+1 logits; column C is the MFS "outlier" class.
  * per-modality heads (v_predict, f_predict) output C logits.
  * all confidences are MSP over the REAL C classes (Section 2.6 inference).
"""
import torch
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Confidence + Adaptive Confidence Loss
# --------------------------------------------------------------------------- #
def msp_confidence(logits, num_classes):
    """Max softmax prob over the first `num_classes` columns. logits: [B, >=C]."""
    return F.softmax(logits[:, :num_classes], dim=1).max(dim=1).values


def adaptive_confidence_loss(fused_logits, unimodal_logits, num_classes):
    """ACL: penalise confidence degradation (Eq. 3 / Eq. 7).

        L_acl = (1/M) * sum_k max(0, conf_k - conf)

    fused_logits:    [B, C+1]
    unimodal_logits: list of M tensors, each [B, C]
    """
    conf = msp_confidence(fused_logits, num_classes)              # [B]
    terms = [torch.clamp(msp_confidence(lk, num_classes) - conf, min=0.0)
             for lk in unimodal_logits]                           # each [B]
    return torch.stack(terms, dim=0).mean(dim=0).mean()


def soft_cross_entropy(logits, soft_targets):
    """CE against a soft label distribution. Both [B, C+1]."""
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


# --------------------------------------------------------------------------- #
#  Multimodal Feature Swapping (Algorithm 1, two modalities)
# --------------------------------------------------------------------------- #
def _one_hot(targets, num_cols):
    oh = torch.zeros(targets.size(0), num_cols, device=targets.device)
    oh.scatter_(1, targets.view(-1, 1), 1.0)
    return oh


def mfs_two_modality(E1, E2, targets, num_classes, n_min, n_max):
    """Swap one contiguous block of dims between modalities, build a soft label.

    E1: [B, D1] (e.g. video v_emd, D1=2304)
    E2: [B, D2] (e.g. flow  f_emd, D2=2048)
    Returns:
      Eo        : [B, D1+D2]  concatenated outlier feature [E1~, E2~]
      y_swapped : [B, C+1]    soft label  (1-lam)*y_true + lam*one_hot(C)
      lam       : float
    """
    D1 = E1.shape[1]
    D2 = E2.shape[1]
    upper = min(D1, D2, n_max)
    n_swap = int(torch.randint(n_min, upper + 1, (1,)).item())
    lam = n_swap / float(n_max)

    s1 = int(torch.randint(0, D1 - n_swap + 1, (1,)).item())
    s2 = int(torch.randint(0, D2 - n_swap + 1, (1,)).item())

    E1t = E1.clone()
    E2t = E2.clone()
    E1t[:, s1:s1 + n_swap] = E2[:, s2:s2 + n_swap]   # read from ORIGINALS
    E2t[:, s2:s2 + n_swap] = E1[:, s1:s1 + n_swap]
    Eo = torch.cat([E1t, E2t], dim=1)

    y_true = _one_hot(targets, num_classes + 1)
    y_out = torch.zeros_like(y_true)
    y_out[:, num_classes] = 1.0                       # outlier class == index C
    y_swapped = (1.0 - lam) * y_true + lam * y_out
    return Eo, y_swapped, lam


# --------------------------------------------------------------------------- #
#  Failure-detection metrics  (Section 3.1)
# --------------------------------------------------------------------------- #
def _aurc(conf, correct):
    """Area under risk-coverage curve, x1000 (lower better)."""
    n = len(conf)
    order = np.argsort(-conf)                          # most confident first
    errs = (1 - correct.astype(np.float64))[order]
    risks = np.cumsum(errs) / np.arange(1, n + 1)
    return float(risks.mean()) * 1000.0


def _auroc(conf, correct):
    """ROC-AUC, correct=positive, score=confidence, x100 (higher better)."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(correct, conf)) * 100.0
    except Exception:
        pos, neg = conf[correct == 1], conf[correct == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        allv = np.concatenate([pos, neg])
        order = np.argsort(allv)
        ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
        u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
        return float(u / (len(pos) * len(neg))) * 100.0


def _fpr95(conf, correct, tpr_target=0.95):
    """FPR (among incorrect) at 95% TPR (among correct), x100 (lower better)."""
    pos, neg = conf[correct == 1], conf[correct == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 1.0 - tpr_target)
    return float((neg >= thr).mean()) * 100.0


def compute_fd_metrics(conf, pred, label):
    """conf/pred/label: 1-D numpy arrays. Returns dict of the 4 FD metrics."""
    conf = np.asarray(conf, dtype=np.float64).ravel()
    pred = np.asarray(pred).ravel()
    label = np.asarray(label).ravel()
    correct = (pred == label).astype(np.int64)
    return {
        "AURC": _aurc(conf, correct),
        "AUROC": _auroc(conf, correct),
        "FPR95": _fpr95(conf, correct),
        "ACC": float(correct.mean()) * 100.0,
    }


def print_fd_row(name, m):
    print("%-16s AURC %7.2f   AUROC %6.2f   FPR95 %6.2f   ACC %6.2f"
          % (name, m["AURC"], m["AUROC"], m["FPR95"], m["ACC"]))
