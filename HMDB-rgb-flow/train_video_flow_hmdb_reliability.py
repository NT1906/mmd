"""
train_video_flow_hmdb_reliability.py  —  reliability-aware 2-modality (video+
flow) failure detection on HMDB/EPIC/Kinetics-family datasets, ported from the
validated 3-modality HAC reliability trainer.

Same idea as the HAC version, dropped to 2 modalities:
  Per modality k in {video, flow}, a small head estimates reliability r_k in
  [0,1] from that modality's embedding, supervised FOR FREE by whether
  modality k ALONE was correct on the sample. r_k reweights each modality's
  embedding before fusion (design A, "reliability-weighted fusion"). Missing
  modality at test = zero it and renormalize the survivor's weight (with only
  one survivor, renorm just means "use it at full/undiminished weight").

Discipline (same as HAC): --reliability OFF reproduces plain ACR exactly
(uses YOUR validated train_video_flow_acr.py math verbatim -- Encoder,
adaptive_confidence_loss, mfs_two_modality, all unchanged), so any gain with
the flag ON is attributable to the method, not a different pipeline.

Soft-init: reliability gates initialize near 1.0 (trusted), so epoch 0 ≈
plain ACR; the head LEARNS to pull weights down. No warmup epochs needed.

Example (reliability method):
  python train_video_flow_hmdb_reliability.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --lr 1e-4 --bsz 16 --nepochs 50 --num_workers 8 \
      --lambda_acl 2.0 --n_min 32 --n_max 256 \
      --reliability --lambda_rel 1.0 \
      --select aurc --patience 12 --save_best --appen hmdb_rel_

Example (matched baseline, same script, flag off):
  python train_video_flow_hmdb_reliability.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --lr 1e-4 --bsz 16 --nepochs 50 --num_workers 8 \
      --lambda_acl 2.0 --n_min 32 --n_max 256 \
      --select aurc --patience 12 --save_best --appen hmdb_off_
"""
from mmaction.apis import init_recognizer
import torch
import argparse
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

from dataloader_video_flow import EPICDOMAIN
from acr_modules import (
    adaptive_confidence_loss, mfs_two_modality, soft_cross_entropy,
    msp_confidence, compute_fd_metrics, print_fd_row,
)


# --------------------------------------------------------------------------- #
#  Fusion head — identical to train_video_flow_acr.py's Encoder.
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, input_dim=4352, out_dim=44, p_drop=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.enc_net = nn.Linear(input_dim, out_dim)
    def logits_from_concat(self, x):
        return self.enc_net(self.drop(x))
    def forward(self, vfeat, ffeat):
        return self.logits_from_concat(torch.cat((vfeat, ffeat), dim=1))


# --------------------------------------------------------------------------- #
#  Per-modality reliability head (NEW). Same soft-init convention as HAC.
# --------------------------------------------------------------------------- #
class ReliabilityHead(nn.Module):
    def __init__(self, in_dim, hidden=128, soft_init=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        if soft_init:
            nn.init.constant_(self.net[-1].bias, 3.0)   # sigmoid(3) ~ 0.95
            nn.init.normal_(self.net[-1].weight, std=1e-3)
    def forward(self, emb):
        return torch.sigmoid(self.net(emb)).squeeze(1)


def weighted_embeds(v_emd, f_emd, rv, rf, mask=None):
    """mask: optional (mv, mf) 0/1 tensors to simulate a missing modality.
    With 2 modalities, renorm to sum-to-2 (the 2-modality analogue of HAC's
    sum-to-3): if one survives, it's scaled to weight ~1 (full trust)."""
    if mask is not None:
        mv, mf = mask
        rv = rv * mv; rf = rf * mf
        denom = (rv + rf).clamp(min=1e-6)
        scale = 2.0 / denom
        rv, rf = rv * scale, rf * scale
    return v_emd * rv.unsqueeze(1), f_emd * rf.unsqueeze(1)


def reliability_targets(v_predict, f_predict, labels, num_class):
    with torch.no_grad():
        tv = (v_predict[:, :num_class].argmax(1) == labels).float()
        tf = (f_predict[:, :num_class].argmax(1) == labels).float()
    return tv, tf


def is_better(new, best, metric):
    if best is None: return True
    return (new <= best) if metric == 'aurc' else (new >= best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='HMDB')
    p.add_argument('--datapath', type=str, required=True)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--bsz', type=int, default=16)
    p.add_argument('--nepochs', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lambda_acl', type=float, default=2.0)
    p.add_argument('--n_min', type=int, default=32)
    p.add_argument('--n_max', type=int, default=256)
    p.add_argument('--no_acl', action='store_true')
    p.add_argument('--no_mfs', action='store_true')
    p.add_argument('--reliability', action='store_true',
                   help="turn ON reliability-weighted fusion (off = plain ACR)")
    p.add_argument('--lambda_rel', type=float, default=1.0)
    p.add_argument('--rel_hidden', type=int, default=128)
    p.add_argument('--hard_gate', action='store_true')
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--p_drop', type=float, default=0.0)
    p.add_argument('--select', type=str, default='aurc', choices=['acc', 'aurc', 'auroc'])
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--save_best', action='store_true')
    p.add_argument('--outdir', type=str, default='models/')
    p.add_argument('--appen', type=str, default='hmdb_rel_')
    args = p.parse_args()
    if not args.datapath.endswith('/'): args.datapath += '/'

    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    v_dim, f_dim = 2304, 2048
    num_class = {'HMDB': 43, 'Kinetics': 229}.get(args.dataset)
    if num_class is None:
        raise ValueError("set num_class for dataset " + args.dataset)
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

    mlp_cls = Encoder(input_dim=v_dim + f_dim, out_dim=num_class + 1, p_drop=args.p_drop).cuda()

    soft = not args.hard_gate
    rel_v = ReliabilityHead(v_dim, args.rel_hidden, soft).cuda()
    rel_f = ReliabilityHead(f_dim, args.rel_hidden, soft).cuda()

    os.makedirs('checkpoints/', exist_ok=True); os.makedirs(args.outdir, exist_ok=True)
    mode = "REL" if args.reliability else "ACR"
    log_name = ("log_video_flow_%s_%s_lr_%s_bsz_%s_%s_lacl_%s_lrel_%s_nmin_%s_nmax_%s%s"
                % (args.dataset, mode, args.lr, args.bsz, args.nepochs, args.lambda_acl,
                   args.lambda_rel if args.reliability else 0, args.n_min, args.n_max, args.appen))
    print(f"MODE: {mode}  (reliability {'ON' if args.reliability else 'OFF'}"
          + (f", soft_init={soft}" if args.reliability else "") + f")  dataset={args.dataset}")

    train_ds = EPICDOMAIN(split='train', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=False)
    val_ds = EPICDOMAIN(split='val', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=False)
    test_ds = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=False)
    mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=args.bsz,
         num_workers=args.num_workers, shuffle=sh, pin_memory=True, drop_last=False)
    dataloaders = {'train': mk(train_ds, True), 'val': mk(val_ds, False), 'test': mk(test_ds, False)}
    print("split sizes:", {k: len(v.dataset) for k, v in dataloaders.items()})

    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters()) + list(model_flow.parameters()) + list(mlp_cls.parameters())
    if args.reliability:
        params += list(rel_v.parameters()) + list(rel_f.parameters())
    optim = torch.optim.Adam(params, lr=args.lr)

    BestScore, BestEpoch = None, 0
    log_path = args.outdir + log_name + '_log.csv'
    with open(log_path, 'w') as f:
        f.write("epoch,split,loss,acc,aurc,auroc,fpr95,rel_loss,rv,rf\n")

    def save_ck(path, epoch_i):
        d = {'epoch': epoch_i, 'BestEpoch': BestEpoch, 'BestScore': BestScore,
             'select': args.select, 'reliability': args.reliability,
             'model_state_dict': model.state_dict(),
             'model_flow_state_dict': model_flow.state_dict(),
             'mlp_cls_state_dict': mlp_cls.state_dict()}
        if args.reliability:
            d['rel_v_state_dict'] = rel_v.state_dict()
            d['rel_f_state_dict'] = rel_f.state_dict()
        torch.save(d, path)

    def extract(clip, flow):
        clip = clip['imgs'].cuda().squeeze(1)
        flow = flow['imgs'].cuda().squeeze(1)
        with torch.no_grad():
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            v_feat = model.module.backbone.get_predict((x_slow.detach(), x_fast.detach()))
            f_feat0 = model_flow.module.backbone.get_feature(flow)
        v_predict, v_emd = model.module.cls_head(v_feat)
        f_feat = model_flow.module.backbone.get_predict(f_feat0.detach())
        f_predict, f_emd = model_flow.module.cls_head(f_feat)
        return v_predict, v_emd, f_predict, f_emd

    def fuse(v_emd, f_emd):
        if not args.reliability:
            return mlp_cls(v_emd, f_emd), None
        rv, rf = rel_v(v_emd), rel_f(f_emd)
        vw, fw = weighted_embeds(v_emd, f_emd, rv, rf)
        return mlp_cls(vw, fw), (rv, rf)

    epochs_no_improve = 0
    for epoch_i in range(args.nepochs):
        for split in ['train', 'val', 'test']:
            tm = (split == 'train')
            model.train(tm); model_flow.train(tm); mlp_cls.train(tm)
            rel_v.train(tm); rel_f.train(tm)

            confs, preds, labels_all = [], [], []
            run_loss, run_rel, rsum = 0.0, 0.0, [0.0, 0.0]
            count, steps = 0, len(dataloaders[split])
            torch.set_grad_enabled(tm)
            for i, (clip, flow, labels) in enumerate(dataloaders[split]):
                labels = labels.cuda()
                v_pr, v_e, f_pr, f_e = extract(clip, flow)

                if tm:
                    fused, rels = fuse(v_e, f_e)
                    l_cls = (criterion(fused, labels) + criterion(v_pr, labels)
                             + criterion(f_pr, labels)) / 3.0
                    l_acl = torch.zeros((), device=fused.device) if args.no_acl else \
                            adaptive_confidence_loss(fused, [v_pr, f_pr], num_class)
                    if args.no_mfs:
                        l_out = torch.zeros((), device=fused.device)
                    else:
                        if args.reliability:
                            rv, rf = rels
                            mv, mf = weighted_embeds(v_e, f_e, rv, rf)
                        else:
                            mv, mf = v_e, f_e
                        Eo, y_sw, _ = mfs_two_modality(mv, mf, labels, num_class, args.n_min, args.n_max)
                        l_out = soft_cross_entropy(mlp_cls.logits_from_concat(Eo), y_sw)
                    if args.reliability:
                        tv, tf = reliability_targets(v_pr, f_pr, labels, num_class)
                        rv, rf = rels
                        l_rel = (F.binary_cross_entropy(rv, tv) + F.binary_cross_entropy(rf, tf)) / 2.0
                    else:
                        l_rel = torch.zeros((), device=fused.device)

                    loss = l_cls + l_out + args.lambda_acl * l_acl + args.lambda_rel * l_rel
                    (loss / args.accum_steps).backward()
                    if (i + 1) % args.accum_steps == 0 or (i + 1 == steps):
                        optim.step(); optim.zero_grad()
                    run_loss += loss.item(); run_rel += float(l_rel)
                    if rels is not None:
                        rsum[0] += float(rels[0].mean()); rsum[1] += float(rels[1].mean())
                else:
                    fused, rels = fuse(v_e, f_e)
                    run_loss += criterion(fused, labels).item()
                    if rels is not None:
                        rsum[0] += float(rels[0].mean()); rsum[1] += float(rels[1].mean())

                confs.append(msp_confidence(fused, num_class).detach().cpu().numpy())
                preds.append(fused[:, :num_class].argmax(1).detach().cpu().numpy())
                labels_all.append(labels.detach().cpu().numpy())
                count += 1
            torch.set_grad_enabled(True)

            m = compute_fd_metrics(np.concatenate(confs), np.concatenate(preds), np.concatenate(labels_all))
            rv_m, rf_m = (rsum[0]/max(count,1), rsum[1]/max(count,1))
            with open(log_path, 'a') as f:
                f.write("%d,%s,%f,%f,%f,%f,%f,%f,%f,%f\n" % (epoch_i, split, run_loss/max(count,1),
                        m['ACC'], m['AURC'], m['AUROC'], m['FPR95'], run_rel/max(count,1), rv_m, rf_m))
            extra = f"  rel {run_rel/max(count,1):.3f}  r[v/f] {rv_m:.2f}/{rf_m:.2f}" if args.reliability else ""
            print("ep %d [%-5s] loss %.4f  ACC %.2f  AURC %.2f  AUROC %.2f  FPR95 %.2f%s"
                  % (epoch_i, split, run_loss/max(count,1), m['ACC'], m['AURC'], m['AUROC'], m['FPR95'], extra))

            if split == 'val':
                score = {'acc': m['ACC'], 'aurc': m['AURC'], 'auroc': m['AUROC']}[args.select]
                if is_better(score, BestScore, args.select):
                    BestScore, BestEpoch = score, epoch_i; epochs_no_improve = 0
                    if args.save_best: save_ck(args.outdir + log_name + '_best.pt', epoch_i)
                else:
                    epochs_no_improve += 1

        save_ck(args.outdir + log_name + '_last.pt', epoch_i)
        if args.patience and epochs_no_improve >= args.patience:
            print("early stop: no %s improvement in %d epochs (best ep %d)"
                  % (args.select, args.patience, BestEpoch)); break

    print("done. best %s = %.4f at epoch %d" % (args.select, BestScore, BestEpoch))


if __name__ == '__main__':
    main()
