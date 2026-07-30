"""
test_video_flow_hmdb_reliability.py  —  standalone test for a RELIABILITY-AWARE
2-modality (video+flow) HMDB/EPIC/Kinetics-family checkpoint. Loads the
per-modality reliability heads (rel_v/rel_f) and APPLIES the same
reliability-weighted fusion used at training time, so the printed number
matches the method's true inference path (mirrors test_video_flow_hac_reliability.py).

Also supports missing-modality eval via --drop {video,flow}: zero that
modality's embedding and renormalize the surviving weight (with 2 modalities,
the survivor gets scaled to weight ~1 = full trust).

Run (normal, both modalities):
  python test_video_flow_hmdb_reliability.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --resumef models/<...hmdb_rel__best.pt>

Run (drop one modality at test time):
  python test_video_flow_hmdb_reliability.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --resumef models/<...hmdb_rel__best.pt> --drop flow
"""
from mmaction.apis import init_recognizer
import torch, argparse, numpy as np, torch.nn as nn, random
from tqdm import tqdm

from dataloader_video_flow import EPICDOMAIN
from acr_modules import msp_confidence, compute_fd_metrics, print_fd_row


class Encoder(nn.Module):
    def __init__(self, input_dim=4352, out_dim=44, p_drop=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.enc_net = nn.Linear(input_dim, out_dim)
    def logits_from_concat(self, x):
        return self.enc_net(self.drop(x))
    def forward(self, vfeat, ffeat):
        return self.logits_from_concat(torch.cat((vfeat, ffeat), dim=1))


class ReliabilityHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
    def forward(self, emb):
        return torch.sigmoid(self.net(emb)).squeeze(1)


def weighted_embeds(v, f, rv, rf, mask=None):
    if mask is not None:
        mv, mf = mask
        rv = rv * mv; rf = rf * mf
        denom = (rv + rf).clamp(min=1e-6)
        scale = 2.0 / denom
        rv, rf = rv * scale, rf * scale
    return v * rv.unsqueeze(1), f * rf.unsqueeze(1)


def extract(model, model_flow, clip, flow):
    clip = clip['imgs'].cuda().squeeze(1)
    flow = flow['imgs'].cuda().squeeze(1)
    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        v_feat = model.module.backbone.get_predict((x_slow.detach(), x_fast.detach()))
        v_predict, v_emd = model.module.cls_head(v_feat)
        f_feat0 = model_flow.module.backbone.get_feature(flow)
        f_feat = model_flow.module.backbone.get_predict(f_feat0.detach())
        f_predict, f_emd = model_flow.module.cls_head(f_feat)
    return v_emd, f_emd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='HMDB')
    ap.add_argument('--datapath', type=str, required=True)
    ap.add_argument('--resumef', type=str, required=True)
    ap.add_argument('--bsz', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rel_hidden', type=int, default=128)
    ap.add_argument('--drop', type=str, default=None, choices=[None, 'video', 'flow'])
    args = ap.parse_args()
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

    mlp_cls = Encoder(input_dim=v_dim + f_dim, out_dim=num_class + 1).cuda()
    rel_v = ReliabilityHead(v_dim, args.rel_hidden).cuda()
    rel_f = ReliabilityHead(f_dim, args.rel_hidden).cuda()

    print("Resuming from", args.resumef)
    ck = torch.load(args.resumef, map_location=device)
    if 'rel_v_state_dict' not in ck:
        raise SystemExit("\nThis checkpoint has no reliability heads — it is a plain ACR "
                         "checkpoint. Use test_video_flow_acr.py instead.\n")
    model.load_state_dict(ck['model_state_dict'])
    model_flow.load_state_dict(ck['model_flow_state_dict'])
    mlp_cls.load_state_dict(ck['mlp_cls_state_dict'])
    rel_v.load_state_dict(ck['rel_v_state_dict'])
    rel_f.load_state_dict(ck['rel_f_state_dict'])
    for m in (model, model_flow, mlp_cls, rel_v, rel_f): m.eval()
    print(f"reliability checkpoint loaded (best epoch {ck.get('BestEpoch','?')})")

    mask = None
    if args.drop is not None:
        mask = {'video': [0., 1.], 'flow': [1., 0.]}[args.drop]
        print(f"DROPPING modality: {args.drop}  (mask v,f = {mask}, renormalized)")

    test_ds = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath,
                         dataset=args.dataset, near_ood=False)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.bsz,
             num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    print(f"test clips: {len(test_ds)}")

    confs, preds, labels = [], [], []
    rsum = [0., 0.]; nb = 0
    with torch.no_grad():
        for clip, flow, y in tqdm(loader, desc='test'):
            v_e, f_e = extract(model, model_flow, clip, flow)
            rv, rf = rel_v(v_e), rel_f(f_e)
            bmask = None
            if mask is not None:
                B = v_e.shape[0]
                bmask = (torch.full((B,), mask[0], device=v_e.device),
                         torch.full((B,), mask[1], device=v_e.device))
            vw, fw = weighted_embeds(v_e, f_e, rv, rf, mask=bmask)
            fused = mlp_cls(vw, fw)
            confs.append(msp_confidence(fused, num_class).detach().cpu().numpy())
            preds.append(fused[:, :num_class].argmax(1).detach().cpu().numpy())
            labels.append(y.numpy())
            rsum[0] += float(rv.mean()); rsum[1] += float(rf.mean()); nb += 1

    m = compute_fd_metrics(np.concatenate(confs), np.concatenate(preds), np.concatenate(labels))
    tag = "ALL-MODALITIES" if args.drop is None else f"DROP-{args.drop.upper()}"
    print(f"\n=== HMDB reliability-aware test  [{tag}] ===")
    print_fd_row("REL/test", m)
    print("mean reliability  r[v/f] = %.3f / %.3f" % (rsum[0]/nb, rsum[1]/nb))
    print("\nLower AURC/FPR95 = better, higher AUROC/ACC = better. Single seed.")


if __name__ == '__main__':
    main()
