"""
test_video_flow_CE.py  —  failure-detection evaluation for a CE checkpoint.

Companion to test_video_flow_acr.py. A plain cross-entropy baseline has a fused
head of width C (num_class), NOT C+1 — there is no MFS outlier slot. The ACR
test script hardcodes out_dim = num_class + 1, so loading a CE checkpoint into
it fails with a size mismatch (43 vs 44 on HMDB). This script sizes the fused
head from the checkpoint itself, so it loads CE (dim C) cleanly and rejects an
ACR checkpoint (dim C+1) with a clear message — the symmetric counterpart to
eval_phase2a_outlier_head.py rejecting CE.

Drop into  MultiOOD-main/HMDB-rgb-flow/  and run:

  python test_video_flow_CE.py --dataset HMDB --datapath /path/to/HMDB51/ \
      --resumef models/<plainCE_log_name>_best.pt

Loads the best CE checkpoint, runs the val and test splits, scores each sample
with MSP over the C classes, and prints the Table-1 row:

      AURC (x1000, down)   AUROC (%, up)   FPR95 (%, down)   ACC (%, up)

Feature/prediction extraction mirrors test_video_flow_acr.py exactly; the only
difference is the head width (C, not C+1).
"""
from mmaction.apis import init_recognizer
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import random
from dataloader_video_flow import EPICDOMAIN

from acr_modules import msp_confidence, compute_fd_metrics, print_fd_row


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)

    def forward(self, vfeat, afeat):
        return self.enc_net(torch.cat((vfeat, afeat), dim=1))


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

        fused = mlp_cls(v_emd, f_emd)                      # [B, C]
    return fused, v_predict, f_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/path/to/video_datasets/')
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument("--resumef", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default='HMDB')
    args = parser.parse_args()
    if not args.datapath.endswith('/'):
        args.datapath += '/'   # normalize datapath (avoids hmdb51video/ path bug)

    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    device = torch.device('cuda:0')
    v_dim, f_dim = 2304, 2048
    num_class = 43 if args.dataset == 'HMDB' else (229 if args.dataset == 'Kinetics' else None)
    assert num_class is not None, "set num_class for " + args.dataset

    model = init_recognizer(config_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    model_flow = init_recognizer(config_file_flow, device=device, use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    print("Resuming from", args.resumef)
    ckpt = torch.load(args.resumef, map_location=device)

    # ---- size the fused head from the checkpoint (CE = C, ACR = C+1) ----
    head_out = ckpt['mlp_cls_state_dict']['enc_net.weight'].shape[0]
    if head_out == num_class + 1:
        raise SystemExit(
            f"\nThis checkpoint has fused head out_dim = {head_out} "
            f"(= {num_class}+1), i.e. an ACR checkpoint with the MFS outlier slot.\n"
            f"Use test_video_flow_acr.py for ACR checkpoints. This script is for "
            f"CE baselines (out_dim = C = {num_class}).\n")
    if head_out != num_class:
        raise SystemExit(
            f"\nUnexpected fused head out_dim = {head_out} "
            f"(expected C = {num_class} for a CE checkpoint).\n")
    mlp_cls = Encoder(input_dim=v_dim + f_dim, out_dim=head_out).cuda()
    print(f"checkpoint loaded: head out_dim {head_out} (= C = {num_class}, CE \u2713)")

    model.load_state_dict(ckpt['model_state_dict'])
    model_flow.load_state_dict(ckpt['model_flow_state_dict'])
    mlp_cls.load_state_dict(ckpt['mlp_cls_state_dict'])
    model.eval(); model_flow.eval(); mlp_cls.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = ['val', 'test']
    loaders = {}
    for s in splits:
        ds = EPICDOMAIN(split=s, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath,
                        dataset=args.dataset, near_ood=False)
        loaders[s] = torch.utils.data.DataLoader(ds, batch_size=args.bsz, num_workers=args.num_workers,
                                                 shuffle=False, pin_memory=(device.type == "cuda"), drop_last=False)

    print("\n=== CE failure-detection results (MSP over %d classes) ===" % num_class)
    for s in splits:
        confs, preds, labels = [], [], []
        for clip, flow, y in tqdm(loaders[s], desc=s):
            fused, _, _ = validate_one_step(model, clip, y, flow, model_flow)
            confs.append(msp_confidence(fused, num_class).cpu())
            preds.append(fused[:, :num_class].argmax(1).cpu())
            labels.append(y)
        m = compute_fd_metrics(torch.cat(confs).numpy(),
                               torch.cat(preds).numpy(),
                               torch.cat(labels).numpy())
        print_fd_row("CE/%s" % s, m)

    print("\nReference — Paper Table 1 (HMDB, ACR): AURC 19.97  AUROC 92.02  FPR95 41.96  ACC 87.23")
    print("Reference — Paper Table 1 (HMDB, MSP): AURC 29.56  AUROC 88.28  FPR95 52.07  ACC 86.20")
    print("Reference — project CE (plainCE16_):   AURC 27.47  AUROC 86.54  FPR95 61.17  ACC 88.08")