"""
train_video_flow_hac_acr.py  —  ACR on HAC with THREE modalities (video+flow+audio).

Reproduction target (paper Table 3, video+flow+audio, ACR):
    AURC 15.09   AUROC 92.26   FPR95 34.78   ACC 89.45

This file MERGES two known-good sources, changing as little as possible:
  * audio backbone construction + extraction  — verbatim from train_video_flow_audio.py
  * ACL, MFS soft-CE, FD metrics, FD-aware     — from acr_modules.py / train_video_flow_acr.py
    checkpoint selection, save/resume
New code (faithful to the paper):
  * ACL over M=3 modalities          — paper Eq. 7 (acr_modules.adaptive_confidence_loss
                                        already averages over a list; we pass 3)
  * cyclic MFS over 3 modalities     — paper Algorithm 2 (implemented below as
                                        mfs_three_modality_cyclic)

PROTOCOL: closed-set pooled HAC (matches the ACR FD table), val carved from train.

First run plain ACR (defaults) and match Table 3 BEFORE adding any new method.

Example:
  # copy the audio weight to the path the loader expects, once:
  #   cp VGGSound/models/vggsound_avgpool.pth.tar pretrained_models/
  python train_video_flow_hac_acr.py --datapath ~/data/HAC/ \
      --lr 1e-4 --bsz 16 --nepochs 50 --num_workers 2 \
      --lambda_acl 2.0 --n_min 32 --n_max 256 \
      --select aurc --patience 12 --save_best --appen hac_acr_
"""
from mmaction.apis import init_recognizer
import torch
import argparse
import os
import numpy as np
import torch.nn as nn
import random

from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_flow_hac_audio import HACAUDIODOMAIN
from acr_modules import (
    adaptive_confidence_loss, soft_cross_entropy,
    msp_confidence, compute_fd_metrics, print_fd_row,
    _one_hot,
)


# --------------------------------------------------------------------------- #
#  Fusion head (3-modality): input = v_dim + f_dim + a_dim, out = C+1
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, input_dim=4864, out_dim=8, p_drop=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.enc_net = nn.Linear(input_dim, out_dim)

    def logits_from_concat(self, x):
        return self.enc_net(self.drop(x))

    def forward(self, vfeat, afeat, ffeat):
        # NOTE order: video, audio, flow  — must match extraction + MFS concat below
        return self.logits_from_concat(torch.cat((vfeat, afeat, ffeat), dim=1))


# --------------------------------------------------------------------------- #
#  Multimodal Feature Swapping — THREE modalities, cyclic (paper Algorithm 2)
#     E1 receives from E3,  E2 receives from E1,  E3 receives from E2
#     single n_swap shared across modalities; lam = n_swap / n_max
# --------------------------------------------------------------------------- #
def mfs_three_modality_cyclic(E1, E2, E3, targets, num_classes, n_min, n_max):
    """Algorithm 2. E1=video[B,D1], E2=audio[B,D2], E3=flow[B,D3].
    Returns concatenated outlier feature Eo=[E1~,E2~,E3~] in (video,audio,flow)
    order to match Encoder.forward, plus the soft label."""
    D1, D2, D3 = E1.shape[1], E2.shape[1], E3.shape[1]
    upper = min(D1, D2, D3, n_max)
    n_min_eff = min(n_min, upper)
    n_swap = int(torch.randint(n_min_eff, upper + 1, (1,)).item())
    lam = n_swap / float(n_max)

    s1 = int(torch.randint(0, D1 - n_swap + 1, (1,)).item())
    s2 = int(torch.randint(0, D2 - n_swap + 1, (1,)).item())
    s3 = int(torch.randint(0, D3 - n_swap + 1, (1,)).item())

    E1t, E2t, E3t = E1.clone(), E2.clone(), E3.clone()
    # read from ORIGINALS (cyclic): 1<-3, 2<-1, 3<-2
    E1t[:, s1:s1 + n_swap] = E3[:, s3:s3 + n_swap]
    E2t[:, s2:s2 + n_swap] = E1[:, s1:s1 + n_swap]
    E3t[:, s3:s3 + n_swap] = E2[:, s2:s2 + n_swap]
    Eo = torch.cat([E1t, E2t, E3t], dim=1)          # (video, audio, flow)

    y_true = _one_hot(targets, num_classes + 1)
    y_out = torch.zeros_like(y_true)
    y_out[:, num_classes] = 1.0
    y_swapped = (1.0 - lam) * y_true + lam * y_out
    return Eo, y_swapped, lam


# --------------------------------------------------------------------------- #
#  one training / validation forward
# --------------------------------------------------------------------------- #
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
    v_predict, v_emd = model.module.cls_head(v_feat)                 # [B,C], [B,2304]
    f_feat = model_flow.module.backbone.get_predict(f_feat0.detach())
    f_predict, f_emd = model_flow.module.cls_head(f_feat)            # [B,C], [B,2048]
    a_predict, a_emd = audio_cls_model(audio_feat.detach())          # [B,C], [B,512]
    return v_predict, v_emd, f_predict, f_emd, a_predict, a_emd


def train_one_step(model, model_flow, audio_model, audio_cls_model, mlp_cls,
                   clip, flow, spec, labels, args, num_class, criterion,
                   optim, step_in_epoch, steps_in_epoch):
    labels = labels.cuda()
    v_predict, v_emd, f_predict, f_emd, a_predict, a_emd = extract(
        model, model_flow, audio_model, audio_cls_model, clip, flow, spec)

    fused = mlp_cls(v_emd, a_emd, f_emd)                             # [B, C+1]

    # L_cls: fused + 3 unimodal CE, averaged
    l_cls = (criterion(fused, labels) + criterion(v_predict, labels)
             + criterion(f_predict, labels) + criterion(a_predict, labels)) / 4.0

    # L_acl: Eq. 7, M=3
    if args.no_acl:
        l_acl = torch.zeros((), device=fused.device)
    else:
        l_acl = adaptive_confidence_loss(fused, [v_predict, f_predict, a_predict], num_class)

    # L_outlier: cyclic MFS (Algorithm 2)
    if args.no_mfs:
        l_out = torch.zeros((), device=fused.device)
    else:
        Eo, y_swapped, _ = mfs_three_modality_cyclic(
            v_emd, a_emd, f_emd, labels, num_class, args.n_min, args.n_max)
        outlier_logits = mlp_cls.logits_from_concat(Eo)
        l_out = soft_cross_entropy(outlier_logits, y_swapped)

    loss = l_cls + l_out + args.lambda_acl * l_acl
    (loss / args.accum_steps).backward()
    is_last = (step_in_epoch + 1 == steps_in_epoch)
    if (step_in_epoch + 1) % args.accum_steps == 0 or is_last:
        optim.step(); optim.zero_grad()
    return fused, loss


def is_better(new, best, metric):
    if best is None:
        return True
    return (new <= best) if metric == 'aurc' else (new >= best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datapath', type=str, required=True)
    p.add_argument('--dataset', type=str, default='HAC')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--bsz', type=int, default=16)
    p.add_argument('--nepochs', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lambda_acl', type=float, default=2.0)
    p.add_argument('--n_min', type=int, default=32)
    p.add_argument('--n_max', type=int, default=256)
    p.add_argument('--no_acl', action='store_true')
    p.add_argument('--no_mfs', action='store_true')
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--p_drop', type=float, default=0.0)
    p.add_argument('--val_frac', type=float, default=0.15)
    p.add_argument('--select', type=str, default='aurc', choices=['acc', 'aurc', 'auroc'])
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--save_best', action='store_true')
    p.add_argument('--outdir', type=str, default='models/')
    p.add_argument('--appen', type=str, default='hac_acr_')
    p.add_argument('--audio_pretrain', type=str,
                   default='pretrained_models/vggsound_avgpool.pth.tar')
    args = p.parse_args()
    if not args.datapath.endswith('/'):
        args.datapath += '/'

    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    num_class = 7                       # HAC: 7 action classes
    v_dim, f_dim, a_dim = 2304, 2048, 512
    device = torch.device('cuda:0')

    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # ---- video + flow backbones ----
    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)
    model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    # ---- audio backbone (verbatim recipe from train_video_flow_audio.py) ----
    if not os.path.exists(args.audio_pretrain):
        raise SystemExit(
            f"\nAudio weight not found at {args.audio_pretrain}.\n"
            f"Copy it there:  cp VGGSound/models/vggsound_avgpool.pth.tar pretrained_models/\n"
            f"or pass --audio_pretrain VGGSound/models/vggsound_avgpool.pth.tar\n")
    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    ckpt_audio = torch.load(args.audio_pretrain)
    audio_model.load_state_dict(ckpt_audio['model_state_dict'])
    audio_model = audio_model.cuda(); audio_model.eval()
    audio_cls_model = AudioAttGenModule()
    audio_cls_model.load_state_dict(ckpt_audio['model_state_dict'], strict=False)
    audio_cls_model.fc = nn.Linear(a_dim, num_class)
    audio_cls_model = audio_cls_model.cuda()

    # ---- fused head: C+1 (the +1 is the MFS outlier class) ----
    mlp_cls = Encoder(input_dim=v_dim + f_dim + a_dim, out_dim=num_class + 1,
                      p_drop=args.p_drop).cuda()

    os.makedirs('checkpoints/', exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    log_name = ("log_video_flow_audio_HAC_ACR_lr_%s_bsz_%s_%s_lacl_%s_nmin_%s_nmax_%s%s"
                % (args.lr, args.bsz, args.nepochs, args.lambda_acl,
                   args.n_min, args.n_max, args.appen))

    # ---- data (closed-set pooled; val carved from train) ----
    mk_ds = lambda sp: HACAUDIODOMAIN(split=sp, cfg=cfg, cfg_flow=cfg_flow,
                                      datapath=args.datapath, val_frac=args.val_frac,
                                      seed=args.seed)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=args.bsz, num_workers=args.num_workers,
        shuffle=sh, pin_memory=True, drop_last=False)
    dataloaders = {'train': mk(mk_ds('train'), True),
                   'val':   mk(mk_ds('val'),   False),
                   'test':  mk(mk_ds('test'),  False)}
    print("split sizes:", {k: len(v.dataset) for k, v in dataloaders.items()})

    criterion = nn.CrossEntropyLoss()
    params = (list(model.parameters()) + list(model_flow.parameters())
              + list(audio_cls_model.parameters()) + list(mlp_cls.parameters()))
    optim = torch.optim.Adam(params, lr=args.lr)

    BestScore, BestEpoch = None, 0
    log_path = args.outdir + log_name + '_log.csv'
    with open(log_path, 'w') as f:
        f.write("epoch,split,loss,acc,aurc,auroc,fpr95\n")

    def save_ck(path, epoch_i):
        torch.save({'epoch': epoch_i, 'BestEpoch': BestEpoch, 'BestScore': BestScore,
                    'select': args.select,
                    'model_state_dict': model.state_dict(),
                    'model_flow_state_dict': model_flow.state_dict(),
                    'audio_cls_model_state_dict': audio_cls_model.state_dict(),
                    'mlp_cls_state_dict': mlp_cls.state_dict()}, path)

    epochs_no_improve = 0
    for epoch_i in range(args.nepochs):
        for split in ['train', 'val', 'test']:
            train_mode = (split == 'train')
            model.train(train_mode); model_flow.train(train_mode); mlp_cls.train(train_mode)
            audio_cls_model.train(train_mode)
            audio_model.eval()      # frozen feature extractor

            confs, preds, labels_all = [], [], []
            run_loss, count = 0.0, 0
            steps_in_epoch = len(dataloaders[split])
            torch.set_grad_enabled(train_mode)
            for i, (clip, flow, spec, labels) in enumerate(dataloaders[split]):
                if train_mode:
                    fused, loss = train_one_step(
                        model, model_flow, audio_model, audio_cls_model, mlp_cls,
                        clip, flow, spec, labels, args, num_class, criterion,
                        optim, i, steps_in_epoch)
                    run_loss += loss.item()
                else:
                    labels_c = labels.cuda()
                    v_pr, v_e, f_pr, f_e, a_pr, a_e = extract(
                        model, model_flow, audio_model, audio_cls_model, clip, flow, spec)
                    fused = mlp_cls(v_e, a_e, f_e)
                    run_loss += criterion(fused, labels_c).item()
                conf = msp_confidence(fused, num_class).detach().cpu().numpy()
                pred = fused[:, :num_class].argmax(1).detach().cpu().numpy()
                confs.append(conf); preds.append(pred); labels_all.append(labels.numpy())
                count += 1
            torch.set_grad_enabled(True)

            confs = np.concatenate(confs); preds = np.concatenate(preds)
            labels_all = np.concatenate(labels_all)
            fd = compute_fd_metrics(confs, preds, labels_all)
            with open(log_path, 'a') as f:
                f.write("%d,%s,%f,%f,%f,%f,%f\n" % (epoch_i, split, run_loss / max(count, 1),
                        fd['ACC'], fd['AURC'], fd['AUROC'], fd['FPR95']))
            print("ep %d [%-5s] loss %.4f  ACC %.2f  AURC %.2f  AUROC %.2f  FPR95 %.2f"
                  % (epoch_i, split, run_loss / max(count, 1),
                     fd['ACC'], fd['AURC'], fd['AUROC'], fd['FPR95']))

            if split == 'val':
                score = {'acc': fd['ACC'], 'aurc': fd['AURC'], 'auroc': fd['AUROC']}[args.select]
                if is_better(score, BestScore, args.select):
                    BestScore, BestEpoch = score, epoch_i
                    epochs_no_improve = 0
                    if args.save_best:
                        save_ck(args.outdir + log_name + '_best.pt', epoch_i)
                else:
                    epochs_no_improve += 1

        save_ck(args.outdir + log_name + '_last.pt', epoch_i)
        if args.patience and epochs_no_improve >= args.patience:
            print("early stop: no %s improvement in %d epochs (best ep %d)"
                  % (args.select, args.patience, BestEpoch))
            break

    print("done. best %s = %.4f at epoch %d" % (args.select, BestScore, BestEpoch))


if __name__ == '__main__':
    main()
