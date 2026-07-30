"""
train_video_flow_hac_reliability.py  —  Phase 4: RELIABILITY-AWARE 3-modality
failure detection on HAC, built on the VALIDATED ACR trainer.

Idea (the contribution):
  Per modality k, a small head estimates reliability r_k in [0,1] from that
  modality's embedding. r_k is supervised FOR FREE by whether modality k ALONE
  was correct on the sample (target = (argmax(unimodal_logits)==label)).
  r_k then REWEIGHTS each modality's embedding before fusion (design A,
  "reliability-weighted fusion"). Missing modality at test = zero it and
  renormalize the surviving weights.

Discipline:
  Everything new is behind  --reliability.  With the flag OFF this script
  reproduces plain 3-modality ACR exactly (the validated 10.99-AURC baseline),
  so any gain is attributable to the method and not the pipeline.

Soft-init (honors "reliability on from epoch 0" WITHOUT cold-start blowup):
  the per-modality gate is initialized near 1.0 (all modalities trusted), so at
  step 0 weighted fusion ≈ plain ACR; the head LEARNS to pull weights down as
  its correctness predictions sharpen. No warmup epochs. Toggle with
  --hard_gate to disable the soft init (raw r_k from step 0).

Losses:
  L = L_cls + L_outlier + lambda_acl * L_acl + lambda_rel * L_rel
  L_rel = mean_k BCE(r_k, (unimodal_k correct))      [target detached]
  ACL and MFS operate on the REWEIGHTED embeddings (the whole point).

Example (reliability method):
  python train_video_flow_hac_reliability.py --datapath ~/data/ \
      --lr 1e-4 --bsz 16 --nepochs 50 --num_workers 8 \
      --lambda_acl 2.0 --n_min 32 --n_max 256 \
      --reliability --lambda_rel 1.0 \
      --select aurc --patience 12 --save_best --appen hac_rel_

Example (re-confirm baseline with this same script — flag OFF):
  python train_video_flow_hac_reliability.py --datapath ~/data/ \
      --lr 1e-4 --bsz 16 --nepochs 50 --num_workers 8 \
      --lambda_acl 2.0 --n_min 32 --n_max 256 \
      --select aurc --patience 12 --save_best --appen hac_acr_repro_
"""
from mmaction.apis import init_recognizer
import torch
import argparse
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_flow_hac_audio import HACAUDIODOMAIN
from acr_modules import (
    adaptive_confidence_loss, soft_cross_entropy,
    msp_confidence, compute_fd_metrics, print_fd_row, _one_hot,
)


# --------------------------------------------------------------------------- #
#  Fusion head (unchanged from validated trainer): 4864 -> C+1
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, input_dim=4864, out_dim=8, p_drop=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.enc_net = nn.Linear(input_dim, out_dim)
    def logits_from_concat(self, x):
        return self.enc_net(self.drop(x))
    def forward(self, vfeat, afeat, ffeat):
        return self.logits_from_concat(torch.cat((vfeat, afeat, ffeat), dim=1))


# --------------------------------------------------------------------------- #
#  Per-modality reliability head (NEW).  embedding -> scalar in [0,1].
#  soft_init: final bias set high so sigmoid starts ~1 (modality trusted).
# --------------------------------------------------------------------------- #
class ReliabilityHead(nn.Module):
    def __init__(self, in_dim, hidden=128, soft_init=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        if soft_init:
            # bias the output logit high -> sigmoid ~0.95 at init -> weight ~1
            nn.init.constant_(self.net[-1].bias, 3.0)
            nn.init.normal_(self.net[-1].weight, std=1e-3)
    def forward(self, emb):
        return torch.sigmoid(self.net(emb)).squeeze(1)        # [B] in (0,1)


# --------------------------------------------------------------------------- #
#  MFS — three modalities, cyclic (paper Algorithm 2). Same as ACR trainer.
# --------------------------------------------------------------------------- #
def mfs_three_modality_cyclic(E1, E2, E3, targets, num_classes, n_min, n_max):
    D1, D2, D3 = E1.shape[1], E2.shape[1], E3.shape[1]
    upper = min(D1, D2, D3, n_max)
    n_min_eff = min(n_min, upper)
    n_swap = int(torch.randint(n_min_eff, upper + 1, (1,)).item())
    lam = n_swap / float(n_max)
    s1 = int(torch.randint(0, D1 - n_swap + 1, (1,)).item())
    s2 = int(torch.randint(0, D2 - n_swap + 1, (1,)).item())
    s3 = int(torch.randint(0, D3 - n_swap + 1, (1,)).item())
    E1t, E2t, E3t = E1.clone(), E2.clone(), E3.clone()
    E1t[:, s1:s1 + n_swap] = E3[:, s3:s3 + n_swap]
    E2t[:, s2:s2 + n_swap] = E1[:, s1:s1 + n_swap]
    E3t[:, s3:s3 + n_swap] = E2[:, s2:s2 + n_swap]
    Eo = torch.cat([E1t, E2t, E3t], dim=1)
    y_true = _one_hot(targets, num_classes + 1)
    y_out = torch.zeros_like(y_true); y_out[:, num_classes] = 1.0
    y_swapped = (1.0 - lam) * y_true + lam * y_out
    return Eo, y_swapped, lam


def extract(model, model_flow, audio_model, audio_cls_model, clip, flow, spec):
    clip = clip['imgs'].cuda().squeeze(1)
    flow = flow['imgs'].cuda().squeeze(1)
    spec = spec.unsqueeze(1).type(torch.FloatTensor).cuda()
    with torch.no_grad():
        _, audio_feat, _ = audio_model(spec)
        f_feat0 = model_flow.module.backbone.get_feature(flow)
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        v_feat = (x_slow.detach(), x_fast.detach())
    v_feat = model.module.backbone.get_predict(v_feat)
    v_predict, v_emd = model.module.cls_head(v_feat)
    f_feat = model_flow.module.backbone.get_predict(f_feat0.detach())
    f_predict, f_emd = model_flow.module.cls_head(f_feat)
    a_predict, a_emd = audio_cls_model(audio_feat.detach())
    return v_predict, v_emd, f_predict, f_emd, a_predict, a_emd


# --------------------------------------------------------------------------- #
#  Reliability-weighted fusion.
#  weights r = [rv, ra, rf] per sample; each modality embedding scaled by its r.
#  mask: optional per-modality 0/1 to simulate a MISSING modality at inference
#        (zeroes that modality's weight, renormalizes the rest to keep scale).
# --------------------------------------------------------------------------- #
def weighted_embeds(v_emd, a_emd, f_emd, rv, ra, rf, mask=None):
    if mask is not None:
        mv, ma, mf = mask
        rv = rv * mv; ra = ra * ma; rf = rf * mf
        denom = (rv + ra + rf).clamp(min=1e-6)
        # renormalize so the surviving weights sum to the same total they would
        # have summed to with all present (keeps fused-input scale stable)
        scale = 3.0 / denom
        rv, ra, rf = rv * scale, ra * scale, rf * scale
    return (v_emd * rv.unsqueeze(1), a_emd * ra.unsqueeze(1), f_emd * rf.unsqueeze(1))


def reliability_targets(v_predict, f_predict, a_predict, labels, num_class):
    """Binary 'was this modality alone correct', detached (no backbone backprop)."""
    with torch.no_grad():
        tv = (v_predict[:, :num_class].argmax(1) == labels).float()
        tf = (f_predict[:, :num_class].argmax(1) == labels).float()
        ta = (a_predict[:, :num_class].argmax(1) == labels).float()
    return tv, ta, tf


def is_better(new, best, metric):
    if best is None: return True
    return (new <= best) if metric == 'aurc' else (new >= best)


def main():
    p = argparse.ArgumentParser()
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
    # ---- reliability method ----
    p.add_argument('--reliability', action='store_true',
                   help="turn ON reliability-weighted fusion (off = plain ACR)")
    p.add_argument('--lambda_rel', type=float, default=1.0)
    p.add_argument('--rel_hidden', type=int, default=128)
    p.add_argument('--hard_gate', action='store_true',
                   help="disable soft-init (raw r_k from step 0)")
    # ----
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--p_drop', type=float, default=0.0)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--select', type=str, default='aurc', choices=['acc', 'aurc', 'auroc'])
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--save_best', action='store_true')
    p.add_argument('--outdir', type=str, default='models/')
    p.add_argument('--appen', type=str, default='hac_rel_')
    p.add_argument('--audio_pretrain', type=str, default='pretrained_models/vggsound_avgpool.pth.tar')
    args = p.parse_args()
    if not args.datapath.endswith('/'): args.datapath += '/'

    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    num_class = 7
    v_dim, f_dim, a_dim = 2304, 2048, 512
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

    if not os.path.exists(args.audio_pretrain):
        raise SystemExit(f"\nAudio weight not found at {args.audio_pretrain}. "
                         f"cp VGGSound/models/vggsound_avgpool.pth.tar pretrained_models/\n")
    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    ck_audio = torch.load(args.audio_pretrain)
    audio_model.load_state_dict(ck_audio['model_state_dict'])
    audio_model = audio_model.cuda(); audio_model.eval()
    audio_cls_model = AudioAttGenModule()
    audio_cls_model.load_state_dict(ck_audio['model_state_dict'], strict=False)
    audio_cls_model.fc = nn.Linear(a_dim, num_class)
    audio_cls_model = audio_cls_model.cuda()

    mlp_cls = Encoder(input_dim=v_dim + f_dim + a_dim, out_dim=num_class + 1, p_drop=args.p_drop).cuda()

    # reliability heads (only used when --reliability)
    soft = not args.hard_gate
    rel_v = ReliabilityHead(v_dim, args.rel_hidden, soft).cuda()
    rel_a = ReliabilityHead(a_dim, args.rel_hidden, soft).cuda()
    rel_f = ReliabilityHead(f_dim, args.rel_hidden, soft).cuda()

    os.makedirs('checkpoints/', exist_ok=True); os.makedirs(args.outdir, exist_ok=True)
    mode = "REL" if args.reliability else "ACR"
    log_name = ("log_video_flow_audio_HAC_%s_lr_%s_bsz_%s_%s_lacl_%s_lrel_%s_nmin_%s_nmax_%s%s"
                % (mode, args.lr, args.bsz, args.nepochs, args.lambda_acl,
                   args.lambda_rel if args.reliability else 0, args.n_min, args.n_max, args.appen))
    print(f"MODE: {mode}  (reliability {'ON' if args.reliability else 'OFF'}"
          + (f", soft_init={soft}" if args.reliability else "") + ")")

    mk_ds = lambda sp: HACAUDIODOMAIN(split=sp, cfg=cfg, cfg_flow=cfg_flow,
                                      datapath=args.datapath, val_frac=args.val_frac, seed=args.seed)
    mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=args.bsz,
                         num_workers=args.num_workers, shuffle=sh, pin_memory=True, drop_last=False)
    dataloaders = {'train': mk(mk_ds('train'), True), 'val': mk(mk_ds('val'), False),
                   'test': mk(mk_ds('test'), False)}
    print("split sizes:", {k: len(v.dataset) for k, v in dataloaders.items()})

    criterion = nn.CrossEntropyLoss()
    params = (list(model.parameters()) + list(model_flow.parameters())
              + list(audio_cls_model.parameters()) + list(mlp_cls.parameters()))
    if args.reliability:
        params += list(rel_v.parameters()) + list(rel_a.parameters()) + list(rel_f.parameters())
    optim = torch.optim.Adam(params, lr=args.lr)

    BestScore, BestEpoch = None, 0
    log_path = args.outdir + log_name + '_log.csv'
    with open(log_path, 'w') as f:
        f.write("epoch,split,loss,acc,aurc,auroc,fpr95,rel_loss,rv,ra,rf\n")

    def save_ck(path, epoch_i):
        d = {'epoch': epoch_i, 'BestEpoch': BestEpoch, 'BestScore': BestScore,
             'select': args.select, 'reliability': args.reliability,
             'model_state_dict': model.state_dict(),
             'model_flow_state_dict': model_flow.state_dict(),
             'audio_cls_model_state_dict': audio_cls_model.state_dict(),
             'mlp_cls_state_dict': mlp_cls.state_dict()}
        if args.reliability:
            d['rel_v_state_dict'] = rel_v.state_dict()
            d['rel_a_state_dict'] = rel_a.state_dict()
            d['rel_f_state_dict'] = rel_f.state_dict()
        torch.save(d, path)

    def fuse(v_emd, a_emd, f_emd, training):
        """Returns fused logits + (rv,ra,rf) (or None). Applies weighting if --reliability."""
        if not args.reliability:
            return mlp_cls(v_emd, a_emd, f_emd), None
        rv, ra, rf = rel_v(v_emd), rel_a(a_emd), rel_f(f_emd)
        vw, aw, fw = weighted_embeds(v_emd, a_emd, f_emd, rv, ra, rf)
        return mlp_cls(vw, aw, fw), (rv, ra, rf)

    epochs_no_improve = 0
    for epoch_i in range(args.nepochs):
        for split in ['train', 'val', 'test']:
            tm = (split == 'train')
            model.train(tm); model_flow.train(tm); mlp_cls.train(tm); audio_cls_model.train(tm)
            rel_v.train(tm); rel_a.train(tm); rel_f.train(tm); audio_model.eval()

            confs, preds, labels_all = [], [], []
            run_loss, run_rel, rsum = 0.0, 0.0, [0.0, 0.0, 0.0]
            count, steps = 0, len(dataloaders[split])
            torch.set_grad_enabled(tm)
            for i, (clip, flow, spec, labels) in enumerate(dataloaders[split]):
                labels = labels.cuda()
                v_pr, v_e, f_pr, f_e, a_pr, a_e = extract(
                    model, model_flow, audio_model, audio_cls_model, clip, flow, spec)

                if tm:
                    fused, rels = fuse(v_e, a_e, f_e, True)
                    l_cls = (criterion(fused, labels) + criterion(v_pr, labels)
                             + criterion(f_pr, labels) + criterion(a_pr, labels)) / 4.0
                    l_acl = torch.zeros((), device=fused.device) if args.no_acl else \
                            adaptive_confidence_loss(fused, [v_pr, f_pr, a_pr], num_class)
                    if args.no_mfs:
                        l_out = torch.zeros((), device=fused.device)
                    else:
                        # MFS on the (reweighted, if reliability) embeddings
                        if args.reliability:
                            rv, ra, rf = rels
                            mv, ma, mf = weighted_embeds(v_e, a_e, f_e, rv, ra, rf)
                        else:
                            mv, ma, mf = v_e, a_e, f_e
                        Eo, y_sw, _ = mfs_three_modality_cyclic(mv, ma, mf, labels, num_class, args.n_min, args.n_max)
                        l_out = soft_cross_entropy(mlp_cls.logits_from_concat(Eo), y_sw)
                    # reliability loss
                    if args.reliability:
                        tv, ta, tf = reliability_targets(v_pr, f_pr, a_pr, labels, num_class)
                        rv, ra, rf = rels
                        l_rel = (F.binary_cross_entropy(rv, tv)
                                 + F.binary_cross_entropy(ra, ta)
                                 + F.binary_cross_entropy(rf, tf)) / 3.0
                    else:
                        l_rel = torch.zeros((), device=fused.device)

                    loss = l_cls + l_out + args.lambda_acl * l_acl + args.lambda_rel * l_rel
                    (loss / args.accum_steps).backward()
                    if (i + 1) % args.accum_steps == 0 or (i + 1 == steps):
                        optim.step(); optim.zero_grad()
                    run_loss += loss.item(); run_rel += float(l_rel)
                    if rels is not None:
                        rsum[0]+=float(rels[0].mean()); rsum[1]+=float(rels[1].mean()); rsum[2]+=float(rels[2].mean())
                else:
                    fused, rels = fuse(v_e, a_e, f_e, False)
                    run_loss += criterion(fused, labels).item()
                    if rels is not None:
                        rsum[0]+=float(rels[0].mean()); rsum[1]+=float(rels[1].mean()); rsum[2]+=float(rels[2].mean())

                confs.append(msp_confidence(fused, num_class).detach().cpu().numpy())
                preds.append(fused[:, :num_class].argmax(1).detach().cpu().numpy())
                labels_all.append(labels.detach().cpu().numpy())
                count += 1
            torch.set_grad_enabled(True)

            m = compute_fd_metrics(np.concatenate(confs), np.concatenate(preds), np.concatenate(labels_all))
            rv_m, ra_m, rf_m = (rsum[0]/max(count,1), rsum[1]/max(count,1), rsum[2]/max(count,1))
            with open(log_path, 'a') as f:
                f.write("%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (epoch_i, split, run_loss/max(count,1),
                        m['ACC'], m['AURC'], m['AUROC'], m['FPR95'], run_rel/max(count,1), rv_m, ra_m, rf_m))
            extra = (f"  rel {run_rel/max(count,1):.3f}  r[v/a/f] {rv_m:.2f}/{ra_m:.2f}/{rf_m:.2f}"
                     if args.reliability else "")
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
