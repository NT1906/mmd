"""
test_video_flow_acr.py  —  failure-detection evaluation for an ACR checkpoint.

Drop into  MultiOOD-main/HMDB-rgb-flow/  and run:

  python test_video_flow_acr.py --dataset HMDB --datapath /path/to/HMDB51/ \
      --resumef models/<log_name>_best.pt

Loads the best ACR checkpoint, runs the test (and val) split, scores each sample
with MSP over the REAL C classes (Section 2.6), and prints the Table-1 row:

      AURC (x1000, down)   AUROC (%, up)   FPR95 (%, down)   ACC (%, up)

Mirrors the repo's test_video_flow.py feature/prediction extraction exactly;
only the fused head is C+1 wide and we evaluate FD metrics instead of saving
OOD score files.
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

        fused = mlp_cls(v_emd, f_emd)                      # [B, C+1]
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

    mlp_cls = Encoder(input_dim=v_dim + f_dim, out_dim=num_class + 1).cuda()

    print("Resuming from", args.resumef)
    ckpt = torch.load(args.resumef, map_location=device)
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

    print("\n=== ACR failure-detection results (MSP over %d real classes) ===" % num_class)
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
        print_fd_row("ACR/%s" % s, m)

    print("\nPaper Table 1 (HMDB, ACR): AURC 19.97  AUROC 92.02  FPR95 41.96  ACC 87.23")
    print("Paper Table 1 (HMDB, MSP): AURC 29.56  AUROC 88.28  FPR95 52.07  ACC 86.20")
