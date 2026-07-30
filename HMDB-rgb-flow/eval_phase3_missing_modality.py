"""
eval_phase3_missing_modality.py  —  Phase 3 Step 1: missing-modality evaluation
(EVAL ONLY, no training).

Question this script answers — three things at once:
  (a) Does the fused model degrade when one modality is removed at test time,
      and does an ACR model stay OVERCONFIDENT when blind (high MSP on samples
      it now gets wrong) more than a CE model? -> decision gate G1.
  (b) Is HMDB even the right venue for the missing-modality story, or do we need
      HAC? -> read the size of the degradation vs the noise floor.
  (c) Is there signal to learn per-modality reliability r_k at all? -> the
      diagnostics print modality-disagreement breakdown + RECOVERABILITY of
      fused errors + oracle-of-2 accuracy ceiling. -> decision gate G2.

Method (no leakage, no retraining):
  - One forward pass extracts per-modality embeddings  v_emd [N,2304],
    f_emd [N,2048], unimodal logits, and labels. Cached.
  - The fused head is a single Linear (enc_net) on cat(v_emd, f_emd). We cache
    its weight/bias too, so ALL masking is recomputed in numpy:
        fused = X @ W.T + b ,  X = concat(v_part, f_part)
  - "Remove modality k" = replace that modality's embedding block with an
    imputation value, then recompute fused. Two imputations:
        zeros           (hard absence)
        val-mean        (mean embedding over the VAL split — zero test leakage;
                         a cheap proxy for train-mean that needs no train forward)
  - Predictions/scoring identical protocol to the rest of the project:
    pred = argmax over first C columns, score = MSP over first C columns.

Head dim auto-detected: ACR = C+1 (44 on HMDB), CE = C (43). Both run fine; the
outlier column (if present) is never used here.

Run (first time — one forward pass, caches per-sample arrays):
  python eval_phase3_missing_modality.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --num_workers 2 --resumef models/<checkpoint>__best.pt

Rerun analysis only (no GPU):
  python eval_phase3_missing_modality.py --dataset HMDB --datapath ~/data/hmdb51/ \
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
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def collect(model, model_flow, loader):
    """One forward pass. Returns per-modality embeddings, unimodal logits, labels."""
    v_emd_all, f_emd_all, v_log_all, f_log_all, y_all = [], [], [], [], []
    with torch.no_grad():
        for clip, flow, y in tqdm(loader, desc='forward'):
            clip = clip['imgs'].cuda().squeeze(1)
            flow = flow['imgs'].cuda().squeeze(1)
            xs, xf = model.module.backbone.get_feature(clip)
            vfeat = model.module.backbone.get_predict((xs.detach(), xf.detach()))
            v_predict, v_emd = model.module.cls_head(vfeat)
            ffeat = model_flow.module.backbone.get_feature(flow)
            ffeat = model_flow.module.backbone.get_predict(ffeat)
            f_predict, f_emd = model_flow.module.cls_head(ffeat)
            v_emd_all.append(v_emd.cpu().numpy())
            f_emd_all.append(f_emd.cpu().numpy())
            v_log_all.append(v_predict.cpu().numpy())
            f_log_all.append(f_predict.cpu().numpy())
            y_all.append(y.numpy())
    return (np.concatenate(v_emd_all), np.concatenate(f_emd_all),
            np.concatenate(v_log_all), np.concatenate(f_log_all),
            np.concatenate(y_all))


def fuse_np(v_part, f_part, W, b):
    """Recompute the fused Linear head in numpy. v_part/f_part already imputed."""
    X = np.concatenate([v_part, f_part], axis=1)      # [N, 2304+2048]
    return X @ W.T + b                                # [N, C(+1)]


def fd(logits, y, num_class):
    """predictions over first C cols; MSP score over first C cols; FD metrics."""
    pred = logits[:, :num_class].argmax(axis=1)
    corr = (pred == y).astype(np.int64)
    msp = softmax_np(logits[:, :num_class]).max(axis=1)
    return {
        'pred': pred, 'corr': corr, 'msp': msp,
        'ACC': 100.0 * corr.mean(),
        'AURC': _aurc(msp, corr), 'AUROC': _auroc(msp, corr), 'FPR95': _fpr95(msp, corr),
    }


def row(name, m):
    print("%-24s ACC %6.2f   AURC %7.2f   AUROC %6.2f   FPR95 %6.2f"
          % (name, m['ACC'], m['AURC'], m['AUROC'], m['FPR95']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datapath', type=str, required=True)
    ap.add_argument('--dataset', type=str, default='HMDB')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--bsz', type=int, default=16)
    ap.add_argument('--resumef', type=str, required=True)
    ap.add_argument('--use_cache', action='store_true')
    args = ap.parse_args()
    if not args.datapath.endswith('/'):
        args.datapath += '/'

    ckpt_tag = os.path.splitext(os.path.basename(args.resumef))[0]
    cache_path = f"phase3_cache_{args.dataset}_{ckpt_tag}.npz"
    num_class = {'HMDB': 43, 'Kinetics': 229}[args.dataset]
    v_dim, f_dim = 2304, 2048

    if args.use_cache and os.path.exists(cache_path):
        print(f"loading cache <- {cache_path}")
        d = np.load(cache_path)
        val = {k: d[f'val_{k}'] for k in ['v_emd', 'f_emd', 'v_log', 'f_log', 'y']}
        test = {k: d[f'test_{k}'] for k in ['v_emd', 'f_emd', 'v_log', 'f_log', 'y']}
        head_W, head_b = d['head_W'], d['head_b']
        head_out = head_W.shape[0]
    else:
        device = torch.device('cuda:0')
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

        ck = torch.load(args.resumef, map_location='cuda:0')
        head_W = ck['mlp_cls_state_dict']['enc_net.weight'].cpu().numpy()
        head_b = ck['mlp_cls_state_dict']['enc_net.bias'].cpu().numpy()
        head_out = head_W.shape[0]
        kind = 'ACR (C+1)' if head_out == num_class + 1 else (
               'CE (C)'    if head_out == num_class     else f'?? ({head_out})')
        print(f"checkpoint head out_dim {head_out} -> {kind}")
        model.load_state_dict(ck['model_state_dict'])
        model_flow.load_state_dict(ck['model_flow_state_dict'])
        model.eval(); model_flow.eval()

        mk = lambda split: torch.utils.data.DataLoader(
            EPICDOMAIN(split=split, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath,
                       dataset=args.dataset, near_ood=False),
            batch_size=args.bsz, num_workers=args.num_workers,
            shuffle=False, pin_memory=True, drop_last=False)
        ve, fe, vl, fl, y = collect(model, model_flow, mk('val'))
        val = {'v_emd': ve, 'f_emd': fe, 'v_log': vl, 'f_log': fl, 'y': y}
        ve, fe, vl, fl, y = collect(model, model_flow, mk('test'))
        test = {'v_emd': ve, 'f_emd': fe, 'v_log': vl, 'f_log': fl, 'y': y}
        np.savez(cache_path,
                 head_W=head_W, head_b=head_b,
                 **{f'val_{k}': v for k, v in val.items()},
                 **{f'test_{k}': v for k, v in test.items()})
        print(f"cached -> {cache_path}")

    # imputation values (computed on VAL only -> zero test leakage)
    v_mean = val['v_emd'].mean(axis=0, keepdims=True)
    f_mean = val['f_emd'].mean(axis=0, keepdims=True)
    v_zero = np.zeros((1, v_dim), dtype=val['v_emd'].dtype)
    f_zero = np.zeros((1, f_dim), dtype=val['f_emd'].dtype)

    def conditions(split):
        ve, fe, y = split['v_emd'], split['f_emd'], split['y']
        N = len(y)
        tile = lambda m: np.repeat(m, N, axis=0)
        return {
            'both modalities':       fd(fuse_np(ve, fe, head_W, head_b), y, num_class),
            'video only (flow=0)':   fd(fuse_np(ve, tile(f_zero), head_W, head_b), y, num_class),
            'video only (flow=mean)':fd(fuse_np(ve, tile(f_mean), head_W, head_b), y, num_class),
            'flow only (video=0)':   fd(fuse_np(tile(v_zero), fe, head_W, head_b), y, num_class),
            'flow only (video=mean)':fd(fuse_np(tile(v_mean), fe, head_W, head_b), y, num_class),
        }

    print(f"\n================ PHASE 3 Step 1: missing modality  [{ckpt_tag}] ================")
    print(f"head out_dim {head_out}  (num_class={num_class})")

    for split_name, split in [('VAL', val), ('TEST (the numbers that matter)', test)]:
        print(f"\n--- {split_name} ---")
        conds = conditions(split)
        for name, m in conds.items():
            row(name, m)
        full = conds['both modalities']
        # G1: overconfidence when blind — mean MSP on samples now WRONG under masking
        print("  overconfidence-when-blind (mean MSP on samples that are WRONG under masking):")
        for name in ['video only (flow=0)', 'video only (flow=mean)',
                     'flow only (video=0)', 'flow only (video=mean)']:
            m = conds[name]
            wrong = (m['corr'] == 0)
            oc = m['msp'][wrong].mean() if wrong.any() else float('nan')
            dacc = m['ACC'] - full['ACC']
            print("    %-24s  ΔACC %+6.2f   mean MSP|wrong %5.3f" % (name, dacc, oc))

    # ---- G2: is there reliability signal? (TEST split) ----
    y = test['y']
    v_pred = test['v_log'][:, :num_class].argmax(axis=1)
    f_pred = test['f_log'][:, :num_class].argmax(axis=1)
    fused_full = fd(fuse_np(test['v_emd'], test['f_emd'], head_W, head_b), y, num_class)
    fused_pred = fused_full['pred']

    vc = (v_pred == y); fc = (f_pred == y); uc = (fused_pred == y)
    both    = (vc & fc).mean()
    only_v  = (vc & ~fc).mean()
    only_f  = (~vc & fc).mean()
    neither = (~vc & ~fc).mean()
    oracle2 = (vc | fc).mean()                      # ceiling: always pick the right modality
    err = ~uc
    recoverable = (err & (vc | fc)).sum() / max(err.sum(), 1)   # fused errors where a modality knew it

    print("\n--- DIAGNOSTICS on TEST (reliability premise, gate G2) ---")
    print("unimodal accuracy:   video %.2f%%   flow %.2f%%   fused %.2f%%"
          % (100*vc.mean(), 100*fc.mean(), 100*uc.mean()))
    print("agreement breakdown: both %.3f   only-video %.3f   only-flow %.3f   neither %.3f"
          % (both, only_v, only_f, neither))
    print("oracle-of-2 accuracy (always trust the right modality): %.2f%%"
          % (100*oracle2))
    print("  -> headroom over fused: %+.2f pts" % (100*(oracle2 - uc.mean())))
    print("RECOVERABILITY of fused errors (>=1 modality alone was correct): %.3f"
          % recoverable)
    print("  Interpretation: recoverability >~0.4 => reliability gating has real headroom")
    print("  on HMDB (premise holds). ~0 => fused errors are mostly 'both modalities")
    print("  wrong' => reliability can't fix accuracy here; lean on missing-modality +")
    print("  calibration, validated on HAC.")
    print("\nSanity: 'both modalities' TEST AURC above should match this checkpoint's")
    print("Phase 2a MSP_C TEST AURC (Run B 29.85 / Run A 26.53 / plainCE43 ~28.96).")
    print("Lower AURC/FPR95 = better, higher AUROC/ACC = better.")


if __name__ == '__main__':
    main()
