"""
eval_phase2a_outlier_head.py  —  Phase 2a: use the trained outlier class as a
confidence signal at INFERENCE time.

The ACR fused head outputs C+1 logits. MFS trains the (C+1)-th class to absorb
synthetic cross-modal-inconsistency outliers (soft label  (1-lambda)*y_true +
lambda*y_outlier).  At inference the paper (Section 2.6) takes argmax over the
first C columns and scores with MSP over those C columns — discarding p(C+1)
entirely. Phase 2a asks: does p(C+1) carry usable test-time information?

We keep PREDICTIONS unchanged (so ACC is identical to the paper protocol) and
only vary the SCORE kappa used to rank/reject:

  (1) MSP_C        kappa = max_{y<=C} softmax(logits[:C])[y]      (paper baseline)
  (2) MSP_{C+1}    kappa = max_{y<=C} softmax(logits[:C+1])[y]    (full denom)
  (3) 1 - p_out    kappa = 1 - softmax(logits[:C+1])[C+1]
  (4) MSP * (1-p_out)
  (5) MSP - alpha * p_out          alpha tuned on VAL only
  (6) learned (logreg on MSP, p_out, log p_out, MSP*log p_out — VAL-fit)

Protocol (no leakage): scalar alpha (and the logreg) are tuned on VAL, then
applied unchanged to TEST. AURC / AUROC / FPR95 on both splits.

This script requires an ACR checkpoint (fused head out_dim = C+1 = 44 on HMDB).
A CE checkpoint (out_dim = C) has no outlier slot and is skipped with a clear
error message.

Run (first time — does one forward pass and caches per-sample arrays):
  python eval_phase2a_outlier_head.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --num_workers 2 \
      --resumef models/log_video_flow_HMDB_ACR_lr_0.0001_bsz_16_50_lacl_2.0_nmin_32_nmax_256acr16__best.pt

Rerun analysis only (re-tune alpha, try new scores) with no GPU:
  python eval_phase2a_outlier_head.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --resumef <same path> --use_cache
"""
from mmaction.apis import init_recognizer
import torch, argparse, os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dataloader_video_flow import EPICDOMAIN
from acr_modules import _aurc, _auroc, _fpr95


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super().__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
    def forward(self, v, f): return self.enc_net(torch.cat((v, f), dim=1))


def softmax_np(x, axis=-1):
    """Numerically stable softmax on numpy arrays."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def collect(model, model_flow, mlp_cls, loader, num_class):
    """One forward pass; returns fused logits [N, C+1] and labels [N]."""
    fused_all, y_all = [], []
    with torch.no_grad():
        for clip, flow, y in tqdm(loader, desc='forward'):
            clip = clip['imgs'].cuda().squeeze(1)
            flow = flow['imgs'].cuda().squeeze(1)
            xs, xf = model.module.backbone.get_feature(clip)
            vfeat = model.module.backbone.get_predict((xs.detach(), xf.detach()))
            _, v_emd = model.module.cls_head(vfeat)
            ffeat = model_flow.module.backbone.get_feature(flow)
            ffeat = model_flow.module.backbone.get_predict(ffeat)
            _, f_emd = model_flow.module.cls_head(ffeat)
            fused = mlp_cls(v_emd, f_emd)  # [B, C+1]
            fused_all.append(fused.cpu().numpy())
            y_all.append(y.numpy())
    return np.concatenate(fused_all), np.concatenate(y_all)


def metrics(score, correct):
    return {'AURC': _aurc(score, correct), 'AUROC': _auroc(score, correct),
            'FPR95': _fpr95(score, correct)}


def row(name, m, acc):
    print("%-26s AURC %7.2f   AUROC %6.2f   FPR95 %6.2f   ACC %6.2f"
          % (name, m['AURC'], m['AUROC'], m['FPR95'], acc))


# ---- scoring functions (operate on cached numpy arrays) ----
def score_msp_c(logits_c1, num_class):
    """Paper baseline: softmax over the C real classes only."""
    return softmax_np(logits_c1[:, :num_class]).max(axis=1)

def score_msp_full(logits_c1, num_class):
    """Softmax over all C+1 classes, max over the real C."""
    p = softmax_np(logits_c1)
    return p[:, :num_class].max(axis=1)

def score_one_minus_pout(logits_c1, num_class):
    p = softmax_np(logits_c1)
    return 1.0 - p[:, num_class]   # p_out = the (C+1)-th probability

def score_msp_times_complement(logits_c1, num_class):
    p = softmax_np(logits_c1)
    msp = softmax_np(logits_c1[:, :num_class]).max(axis=1)
    return msp * (1.0 - p[:, num_class])

def score_msp_minus_alpha_pout(logits_c1, num_class, alpha):
    p = softmax_np(logits_c1)
    msp = softmax_np(logits_c1[:, :num_class]).max(axis=1)
    return msp - alpha * p[:, num_class]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datapath', type=str, required=True)
    ap.add_argument('--dataset', type=str, default='HMDB')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--bsz', type=int, default=16)
    ap.add_argument('--resumef', type=str, required=True)
    ap.add_argument('--use_cache', action='store_true',
                    help="skip forward pass; use existing cache file")
    args = ap.parse_args()
    if not args.datapath.endswith('/'):
        args.datapath += '/'

    # cache name: keyed on checkpoint basename so multiple ACR runs don't collide
    ckpt_tag = os.path.splitext(os.path.basename(args.resumef))[0]
    cache_path = f"phase2a_cache_{args.dataset}_{ckpt_tag}.npz"

    num_class = {'HMDB': 43, 'Kinetics': 229}[args.dataset]

    if args.use_cache and os.path.exists(cache_path):
        print(f"loading cache <- {cache_path}")
        d = np.load(cache_path)
        val_logits, val_y = d['val_logits'], d['val_y']
        test_logits, test_y = d['test_logits'], d['test_y']
    else:
        # ---- build model ----
        device = torch.device('cuda:0')
        v_dim, f_dim = 2304, 2048
        config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
        checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'
        config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
        checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
        cfg = model.cfg
        model = torch.nn.DataParallel(model)

        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        # ---- load checkpoint & detect head dim (ACR vs CE) ----
        ck = torch.load(args.resumef, map_location='cuda:0')
        head_w = ck['mlp_cls_state_dict']['enc_net.weight']
        head_out = head_w.shape[0]
        if head_out != num_class + 1:
            raise SystemExit(
                f"\nThis checkpoint has fused head out_dim = {head_out} "
                f"(expected {num_class + 1}).\n"
                f"Phase 2a requires an ACR checkpoint with the MFS outlier slot. "
                f"CE checkpoints (out_dim = C = {num_class}) cannot participate "
                f"in this analysis — they have no outlier head.\n")
        mlp_cls = Encoder(input_dim=v_dim + f_dim, out_dim=num_class + 1).cuda()
        model.load_state_dict(ck['model_state_dict'])
        model_flow.load_state_dict(ck['model_flow_state_dict'])
        mlp_cls.load_state_dict(ck['mlp_cls_state_dict'])
        model.eval(); model_flow.eval(); mlp_cls.eval()
        print(f"checkpoint loaded: head out_dim {head_out} (= {num_class}+1, ACR ✓)")

        # ---- dataloaders ----
        val_ds = EPICDOMAIN(split='val', cfg=cfg, cfg_flow=cfg_flow,
                            datapath=args.datapath, dataset=args.dataset, near_ood=False)
        test_ds = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow,
                             datapath=args.datapath, dataset=args.dataset, near_ood=False)
        mk = lambda ds: torch.utils.data.DataLoader(
            ds, batch_size=args.bsz, num_workers=args.num_workers,
            shuffle=False, pin_memory=True, drop_last=False)
        val_logits, val_y = collect(model, model_flow, mlp_cls, mk(val_ds), num_class)
        test_logits, test_y = collect(model, model_flow, mlp_cls, mk(test_ds), num_class)
        np.savez(cache_path,
                 val_logits=val_logits, val_y=val_y,
                 test_logits=test_logits, test_y=test_y)
        print(f"cached -> {cache_path}")

    # ---- predictions (unchanged across all score variants) ----
    val_pred = val_logits[:, :num_class].argmax(axis=1)
    test_pred = test_logits[:, :num_class].argmax(axis=1)
    val_corr = (val_pred == val_y).astype(np.int64)
    test_corr = (test_pred == test_y).astype(np.int64)
    val_acc, test_acc = 100 * val_corr.mean(), 100 * test_corr.mean()

    # ---- alpha sweep for MSP - alpha*p_out on VAL ----
    alphas = np.arange(0.0, 5.01, 0.1)
    val_aurc_by_alpha = [
        _aurc(score_msp_minus_alpha_pout(val_logits, num_class, a), val_corr)
        for a in alphas
    ]
    best_alpha = float(alphas[int(np.argmin(val_aurc_by_alpha))])

    # ---- logreg on richer features (val-fit, no leakage) ----
    from sklearn.linear_model import LogisticRegression
    def feats(logits_c1):
        p = softmax_np(logits_c1)
        msp = softmax_np(logits_c1[:, :num_class]).max(axis=1)
        p_out = p[:, num_class]
        return np.stack([msp, p_out, np.log(np.clip(p_out, 1e-8, 1.0)),
                         msp * np.log(np.clip(1 - p_out, 1e-8, 1.0))], axis=1)
    lr = LogisticRegression(max_iter=2000).fit(feats(val_logits), val_corr)
    val_lr  = lr.predict_proba(feats(val_logits))[:, 1]
    test_lr = lr.predict_proba(feats(test_logits))[:, 1]

    print("\n================ PHASE 2a: outlier-head scoring (predictions unchanged) ================")
    print(f"tuned on VAL:  alpha (MSP - alpha*p_out) = {best_alpha:.2f}")
    print("\n--- VAL ---")
    row("MSP_C (paper baseline)", metrics(score_msp_c(val_logits, num_class), val_corr), val_acc)
    row("MSP_{C+1}",              metrics(score_msp_full(val_logits, num_class), val_corr), val_acc)
    row("1 - p_out",              metrics(score_one_minus_pout(val_logits, num_class), val_corr), val_acc)
    row("MSP * (1 - p_out)",      metrics(score_msp_times_complement(val_logits, num_class), val_corr), val_acc)
    row(f"MSP - {best_alpha:.2f}*p_out",
        metrics(score_msp_minus_alpha_pout(val_logits, num_class, best_alpha), val_corr), val_acc)
    row("learned (logreg)",       metrics(val_lr, val_corr), val_acc)

    print("\n--- TEST (the number that matters) ---")
    row("MSP_C (paper baseline)", metrics(score_msp_c(test_logits, num_class), test_corr), test_acc)
    row("MSP_{C+1}",              metrics(score_msp_full(test_logits, num_class), test_corr), test_acc)
    row("1 - p_out",              metrics(score_one_minus_pout(test_logits, num_class), test_corr), test_acc)
    row("MSP * (1 - p_out)",      metrics(score_msp_times_complement(test_logits, num_class), test_corr), test_acc)
    row(f"MSP - {best_alpha:.2f}*p_out",
        metrics(score_msp_minus_alpha_pout(test_logits, num_class, best_alpha), test_corr), test_acc)
    row("learned (logreg)",       metrics(test_lr, test_corr), test_acc)

    # ---- diagnostic: how often is p_out non-trivial? ----
    p_out_test = softmax_np(test_logits)[:, num_class]
    print(f"\nDiagnostic on TEST  p_out:  mean {p_out_test.mean():.4f}   "
          f"median {np.median(p_out_test):.4f}   "
          f"95th %ile {np.quantile(p_out_test, 0.95):.4f}   "
          f"max {p_out_test.max():.4f}")
    print("If p_out is ~0 everywhere, the outlier head is dormant on real test data — "
          "expected when MFS outliers were easy to separate during training.")
    print("\nLower AURC/FPR95 = better, higher AUROC = better. ACC is identical across rows.")


if __name__ == '__main__':
    main()
