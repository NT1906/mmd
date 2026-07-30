"""
train_video_flow_acr.py  —  ACR training for HMDB51 (video + optical flow).
v2: anti-overfitting + FD-aware checkpoint selection.

Drop into  MultiOOD-main/HMDB-rgb-flow/  (replaces the previous version; all old
flags still work, all new behaviour is opt-in via flags).

PAPER-FAITHFUL RUN (recommended to close the gap to the paper's Table 1):

  python train_video_flow_acr.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --lr 1e-4 --bsz 16 --nepochs 50 --num_workers 2 \
      --lambda_acl 2.0 --n_min 32 --n_max 256 \
      --select aurc --patience 12 --save_best --appen acr16_

  (paper uses bsz 16; if 16 OOMs use  --bsz 8 --accum_steps 2  for the same
   effective batch.)

WHAT'S NEW vs v1
  --select {acc,aurc,auroc}   metric used to pick the _best.pt checkpoint.
                              'aurc' computes VAL failure-detection metrics every
                              epoch and keeps the checkpoint with the lowest val
                              AURC — usually a different (and better-for-FD)
                              epoch than best val accuracy. Default 'acc'
                              reproduces the old behaviour exactly.
  --patience N                early stopping: stop if the --select metric has
                              not improved for N consecutive epochs (0 = off).
  --accum_steps K             gradient accumulation: effective batch = bsz * K.
  --scheduler {none,cosine}   cosine LR decay to --lr_min over nepochs
                              (constant LR = paper default = 'none').
  --weight_decay W            Adam weight decay (default 1e-4, unchanged).
  --dropout P                 dropout on the concatenated [v_emd, f_emd] before
                              the fusion head (default 0 = off = paper).
  --label_smooth S            label smoothing on the CE losses (default 0 = off).
                              CAUTION: changes the MSP confidence distribution,
                              i.e. the very thing the FD metrics score. Use only
                              for ablation, not for paper comparison rows.
  CSV log now also records VAL/TEST AURC, AUROC, FPR95 every epoch so you can
  see *failure-detection* overfitting, not just accuracy.

Method (unchanged, Eq. 6):
  L = L_cls + L_outlier + lambda_acl * L_acl
"""
from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import math
import numpy as np
import torch.nn as nn
import random
from dataloader_video_flow import EPICDOMAIN

from acr_modules import (
    adaptive_confidence_loss, mfs_two_modality, soft_cross_entropy,
    msp_confidence, compute_fd_metrics,
)


class Encoder(nn.Module):
    """Fusion head h(.). out_dim = num_class + 1 (the +1 is the MFS outlier class).
    Optional dropout on the concatenated input (off by default = paper)."""
    def __init__(self, input_dim=2816, out_dim=8, p_drop=0.0):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.enc_net = nn.Linear(input_dim, out_dim)

    def logits_from_concat(self, x):
        return self.enc_net(self.drop(x))

    def forward(self, vfeat, afeat):
        return self.logits_from_concat(torch.cat((vfeat, afeat), dim=1))


def train_one_step(model, clip, labels, flow, model_flow, step_in_epoch, steps_in_epoch):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    # ---- feature extraction (identical to the repo) ----
    with torch.no_grad():
        f_feat0 = model_flow.module.backbone.get_feature(flow)
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        v_feat = (x_slow.detach(), x_fast.detach())

    v_feat = model.module.backbone.get_predict(v_feat)
    v_predict, v_emd = model.module.cls_head(v_feat)               # v_predict:[B,C]  v_emd:[B,2304]

    f_feat = model_flow.module.backbone.get_predict(f_feat0.detach())
    f_predict, f_emd = model_flow.module.cls_head(f_feat)          # f_predict:[B,C]  f_emd:[B,2048]

    fused = mlp_cls(v_emd, f_emd)                                  # [B, C+1] or [B, C]

    # ---- L_cls : fused + per-modality CE, averaged (use_single_pred) ----
    l_cls = (criterion(fused, labels)
             + criterion(v_predict, labels)
             + criterion(f_predict, labels)) / 3.0

    # ---- L_acl : penalise confidence degradation (skipped if --no_acl) ----
    if args.no_acl:
        l_acl = torch.zeros((), device=fused.device)
    else:
        l_acl = adaptive_confidence_loss(fused, [v_predict, f_predict], num_class)

    # ---- L_outlier : MFS synthetic failures with soft labels (skipped if --no_mfs) ----
    if args.no_mfs:
        l_out = torch.zeros((), device=fused.device)
    else:
        Eo, y_swapped, _ = mfs_two_modality(v_emd, f_emd, labels, num_class,
                                            args.n_min, args.n_max)
        outlier_logits = mlp_cls.logits_from_concat(Eo)           # fused head on swapped feats
        l_out = soft_cross_entropy(outlier_logits, y_swapped)

    loss = l_cls + l_out + args.lambda_acl * l_acl

    # ---- gradient accumulation: effective batch = bsz * accum_steps ----
    (loss / args.accum_steps).backward()
    is_last = (step_in_epoch + 1 == steps_in_epoch)
    if (step_in_epoch + 1) % args.accum_steps == 0 or is_last:
        optim.step()
        optim.zero_grad()
    return fused, loss


def validate_one_step(model, clip, labels, flow, model_flow):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)
    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        v_feat = (x_slow.detach(), x_fast.detach())
        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)

        f_feat = model_flow.module.backbone.get_feature(flow)
        f_feat = model_flow.module.backbone.get_predict(f_feat)
        f_predict, f_emd = model_flow.module.cls_head(f_feat)

        fused = mlp_cls(v_emd, f_emd)
    loss = criterion(fused, labels)
    return fused, loss


def is_better(new, best, metric):
    """Higher is better for acc/auroc; lower is better for aurc."""
    if best is None:
        return True
    return (new <= best) if metric == 'aurc' else (new >= best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/path/to/video_datasets/')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bsz', type=int, default=16)   # paper batch size
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument("--opt", type=str, default='adam')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appen", type=str, default='acr_')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default='HMDB')   # HMDB / Kinetics
    parser.add_argument("--resume", type=str, default='', help='path to _last.pt to resume from')
    parser.add_argument("--outdir", type=str, default='models/', help='where checkpoints are saved')
    # ACR hyperparameters (paper Section 3.1)
    parser.add_argument('--lambda_acl', type=float, default=2.0)
    parser.add_argument('--n_min', type=int, default=32)
    parser.add_argument('--n_max', type=int, default=256)
    # ablations (Table 2)
    parser.add_argument('--no_acl', action='store_true', help='disable Adaptive Confidence Loss')
    parser.add_argument('--no_mfs', action='store_true', help='disable Multimodal Feature Swapping')
    # ---- NEW: overfitting control / FD-aware selection ----
    parser.add_argument('--select', type=str, default='acc', choices=['acc', 'aurc', 'auroc'],
                        help="metric on VAL used to pick _best.pt ('aurc' recommended for FD)")
    parser.add_argument('--patience', type=int, default=0,
                        help='early stop after N epochs without --select improvement (0 = off)')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='gradient accumulation steps; effective batch = bsz * accum_steps')
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'cosine'])
    parser.add_argument('--lr_min', type=float, default=1e-6, help='cosine floor LR')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout on concat features before fusion head (0 = paper)')
    parser.add_argument('--label_smooth', type=float, default=0.0,
                        help='CE label smoothing. WARNING: alters MSP confidences; '
                             'do not use for paper-comparison rows')
    args = parser.parse_args()
    if not args.datapath.endswith('/'):
        args.datapath += '/'   # normalize datapath (avoids hmdb51video/ path bug)

    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    device = torch.device('cuda:0')
    v_dim, f_dim = 2304, 2048

    if args.dataset == 'HMDB':
        num_class = 43
    elif args.dataset == 'Kinetics':
        num_class = 229
    else:
        raise ValueError("set num_class for dataset " + args.dataset)

    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()    # unimodal head: C classes
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    # =====================================================================
    # FIXED: fused head dimension depends on whether ACR/MFS are active.
    # Plain CE (no_acl + no_mfs)  -> out_dim = num_class (43)
    # ACR or MFS enabled          -> out_dim = num_class + 1 (44)
    # =====================================================================
    out_dim = num_class if (args.no_acl and args.no_mfs) else num_class + 1
    mlp_cls = Encoder(input_dim=v_dim + f_dim, out_dim=out_dim,
                      p_drop=args.dropout).cuda()
    # =====================================================================

    base_path = "checkpoints/"; os.makedirs(base_path, exist_ok=True)
    base_path_model = args.outdir.rstrip('/') + '/'; os.makedirs(base_path_model, exist_ok=True)
    log_name = "log_video_flow_%s_ACR_lr_%s_bsz_%s_%s_lacl_%s_nmin_%s_nmax_%s%s" % (
        args.dataset, args.lr, args.bsz, args.nepochs,
        args.lambda_acl, args.n_min, args.n_max, args.appen)
    log_path = base_path + log_name + '.csv'
    print(log_path)
    if args.label_smooth > 0:
        print("WARNING: label smoothing %.3f will change MSP confidence calibration "
              "(FD metrics not comparable to the paper)." % args.label_smooth)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).cuda()
    batch_size = args.bsz

    # SAME trainable parameter set as the repo (frozen lower backbone layers)
    params = (list(model.module.backbone.fast_path.layer4.parameters())
              + list(model.module.backbone.slow_path.layer4.parameters())
              + list(model.module.cls_head.parameters())
              + list(model_flow.module.backbone.layer4.parameters())
              + list(model_flow.module.cls_head.parameters())
              + list(mlp_cls.parameters()))
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.nepochs,
                                                            eta_min=args.lr_min)
                 if args.scheduler == 'cosine' else None)

    BestLoss, BestEpoch, BestAcc, BestTestAcc = float("inf"), 0, 0, 0
    BestScore = None              # value of the --select metric at the best epoch
    epochs_no_improve = 0
    starting_epoch = 0

    # ---- resume ----
    if args.resume and os.path.exists(args.resume):
        print("Resuming from", args.resume)
        ck = torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(ck['model_state_dict'])
        model_flow.load_state_dict(ck['model_flow_state_dict'])
        mlp_cls.load_state_dict(ck['mlp_cls_state_dict'])
        optim.load_state_dict(ck['optimizer'])
        if scheduler is not None and 'scheduler' in ck and ck['scheduler'] is not None:
            scheduler.load_state_dict(ck['scheduler'])
        starting_epoch = ck['epoch'] + 1
        BestLoss, BestEpoch = ck.get('BestLoss', BestLoss), ck.get('BestEpoch', 0)
        BestAcc, BestTestAcc = ck.get('BestAcc', 0), ck.get('BestTestAcc', 0)
        BestScore = ck.get('BestScore', BestAcc if args.select == 'acc' else None)
        epochs_no_improve = ck.get('epochs_no_improve', 0)
        print("Resumed at epoch %d (BestValAcc %.4f)" % (starting_epoch, BestAcc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EPICDOMAIN(split='train', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=False)
    val_dataset = EPICDOMAIN(split='val', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=False)
    test_dataset = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, dataset=args.dataset, near_ood=False)
    mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=args.num_workers,
                                                    shuffle=sh, pin_memory=(device.type == "cuda"), drop_last=sh)
    dataloaders = {'train': mk(train_dataset, True), 'val': mk(val_dataset, False), 'test': mk(test_dataset, False)}
    splits = ['train', 'val'] if args.dataset == 'Kinetics' else ['train', 'val', 'test']

    def save_ck(path, epoch_i):
        torch.save({'epoch': epoch_i, 'BestLoss': BestLoss, 'BestEpoch': BestEpoch,
                    'BestAcc': BestAcc, 'BestTestAcc': BestTestAcc,
                    'BestScore': BestScore, 'epochs_no_improve': epochs_no_improve,
                    'select': args.select,
                    'model_state_dict': model.state_dict(),
                    'model_flow_state_dict': model_flow.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'mlp_cls_state_dict': mlp_cls.state_dict()}, path)

    stop_early = False
    with open(log_path, "a") as f:
        # CSV columns: epoch,split,loss,acc,aurc,auroc,fpr95
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d  (lr %.2e)" % (epoch_i, optim.param_groups[0]['lr']))
            val_improved = False
            for split in splits:
                acc = count = 0; total_loss = 0.0
                conf_all, pred_all, lab_all = [], [], []
                print(split)
                model.train(split == 'train'); model_flow.train(split == 'train'); mlp_cls.train(split == 'train')
                if split == 'train':
                    optim.zero_grad()
                steps_in_epoch = len(dataloaders[split])
                with tqdm.tqdm(total=steps_in_epoch) as pbar:
                    for i, (clip, flow, labels) in enumerate(dataloaders[split]):
                        if split == 'train':
                            predict1, loss = train_one_step(model, clip, labels, flow, model_flow, i, steps_in_epoch)
                        else:
                            predict1, loss = validate_one_step(model, clip, labels, flow, model_flow)
                        total_loss += loss.item() * batch_size
                        # accuracy uses the REAL C classes only (drop outlier column if present)
                        logits_c = predict1.detach().cpu()[:, :num_class]
                        _, predict = torch.max(logits_c, dim=1)
                        acc += int((predict == labels).sum().item())
                        count += predict1.size(0)
                        if split != 'train':
                            conf_all.append(msp_confidence(predict1.detach().cpu(), num_class).numpy())
                            pred_all.append(predict.numpy())
                            lab_all.append(labels.numpy())
                        pbar.set_postfix_str("loss %.4f acc %.4f" % (total_loss / count, acc / count))
                        pbar.update()

                    fd = None
                    if split != 'train':
                        fd = compute_fd_metrics(np.concatenate(conf_all),
                                                np.concatenate(pred_all),
                                                np.concatenate(lab_all))
                        print("  %s FD: AURC %.2f  AUROC %.2f  FPR95 %.2f" %
                              (split, fd["AURC"], fd["AUROC"], fd["FPR95"]))

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        score = {'acc': currentvalAcc,
                                 'aurc': fd["AURC"],
                                 'auroc': fd["AUROC"]}[args.select]
                        if is_better(score, BestScore, args.select):
                            BestScore = score
                            BestLoss = total_loss / float(count); BestEpoch = epoch_i
                            BestAcc = currentvalAcc
                            val_improved = True
                            epochs_no_improve = 0
                            if args.save_best:
                                save_ck(base_path_model + log_name + '_best.pt', epoch_i)
                        else:
                            epochs_no_improve += 1
                    if split == 'test' and val_improved:
                        BestTestAcc = acc / float(count)

                    if fd is None:
                        f.write("%d,%s,%f,%f,,,\n" % (epoch_i, split, total_loss / count, acc / count))
                    else:
                        f.write("%d,%s,%f,%f,%f,%f,%f\n" % (epoch_i, split, total_loss / count,
                                                            acc / count, fd["AURC"], fd["AUROC"], fd["FPR95"]))
                    f.flush()
                    print("epoch %d %s acc %.4f | Best %s %.4f (epoch %d)" %
                          (epoch_i, split, acc / count, args.select,
                           BestScore if BestScore is not None else float('nan'), BestEpoch))

            if scheduler is not None:
                scheduler.step()

            # ---- save _last.pt every epoch (resume point) ----
            save_ck(base_path_model + log_name + '_last.pt', epoch_i)

            if args.patience > 0 and epochs_no_improve >= args.patience:
                print("Early stopping: no val %s improvement for %d epochs (best epoch %d)."
                      % (args.select, args.patience, BestEpoch))
                stop_early = True
                break

        f.write("BestEpoch,%d,BestVal_%s,%f\n" % (BestEpoch, args.select,
                                                  BestScore if BestScore is not None else float('nan')))
        f.flush()
    print("Done%s. Best val %s %.4f at epoch %d. Best checkpoint: %s_best.pt"
          % (" (early-stopped)" if stop_early else "", args.select,
             BestScore if BestScore is not None else float('nan'),
             BestEpoch, base_path_model + log_name))
    print("Now run test_video_flow_acr.py with --resumef %s_best.pt to get FD metrics."
          % (base_path_model + log_name))