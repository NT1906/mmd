"""
test_video_flow_hac_acr.py  —  standalone FD evaluation for a 3-modality
HAC ACR checkpoint (video + flow + audio).

Loads a saved *_best.pt, runs the HAC test split ONCE in clean eval mode,
scores each sample with MSP over the 7 real classes (paper Section 2.6 inference),
and prints the Table-1 row:

      AURC (x1000, down)   AUROC (%, up)   FPR95 (%, down)   ACC (%, up)

Use this to confirm the saved checkpoint reproduces the training-log number
(and that it saved/loads correctly) before building anything on top of it.

Run:
  python test_video_flow_hac_acr.py --datapath ~/data/ \
      --resumef models/log_video_flow_audio_HAC_ACR_lr_0.0001_bsz_16_50_lacl_2.0_nmin_32_nmax_256hac_acr__best.pt
"""
from mmaction.apis import init_recognizer
import torch
import argparse
import numpy as np
import torch.nn as nn
import random
from tqdm import tqdm

from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_flow_hac_audio import HACAUDIODOMAIN
from acr_modules import msp_confidence, compute_fd_metrics, print_fd_row


class Encoder(nn.Module):
    """Must match the trainer's fused head exactly (3-modality, C+1)."""
    def __init__(self, input_dim=4864, out_dim=8, p_drop=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.enc_net = nn.Linear(input_dim, out_dim)

    def logits_from_concat(self, x):
        return self.enc_net(self.drop(x))

    def forward(self, vfeat, afeat, ffeat):
        return self.logits_from_concat(torch.cat((vfeat, afeat, ffeat), dim=1))


def extract(model, model_flow, audio_model, audio_cls_model, clip, flow, spec):
    """Identical extraction path to the trainer (clean, no grad)."""
    clip = clip['imgs'].cuda().squeeze(1)
    flow = flow['imgs'].cuda().squeeze(1)
    spec = spec.unsqueeze(1).type(torch.FloatTensor).cuda()
    with torch.no_grad():
        _, audio_feat, _ = audio_model(spec)
        f_feat0 = model_flow.module.backbone.get_feature(flow)
        x_slow, x_fast = model.module.backbone.get_feature(clip)
        v_feat = model.module.backbone.get_predict((x_slow.detach(), x_fast.detach()))
        v_predict, v_emd = model.module.cls_head(v_feat)
        f_feat = model_flow.module.backbone.get_predict(f_feat0.detach())
        f_predict, f_emd = model_flow.module.cls_head(f_feat)
        a_predict, a_emd = audio_cls_model(audio_feat.detach())
    return v_emd, f_emd, a_emd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datapath', type=str, required=True)
    ap.add_argument('--resumef', type=str, required=True)
    ap.add_argument('--bsz', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--audio_pretrain', type=str,
                    default='pretrained_models/vggsound_avgpool.pth.tar')
    args = ap.parse_args()
    if not args.datapath.endswith('/'):
        args.datapath += '/'

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

    # audio backbone (frozen extractor)
    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    ck_audio = torch.load(args.audio_pretrain)
    audio_model.load_state_dict(ck_audio['model_state_dict'])
    audio_model = audio_model.cuda(); audio_model.eval()
    audio_cls_model = AudioAttGenModule()
    audio_cls_model.fc = nn.Linear(a_dim, num_class)
    audio_cls_model = audio_cls_model.cuda()

    mlp_cls = Encoder(input_dim=v_dim + f_dim + a_dim, out_dim=num_class + 1).cuda()

    # ---- load the trained checkpoint & detect head dim ----
    print("Resuming from", args.resumef)
    ck = torch.load(args.resumef, map_location=device)
    head_out = ck['mlp_cls_state_dict']['enc_net.weight'].shape[0]
    if head_out != num_class + 1:
        raise SystemExit(
            f"\nFused head out_dim = {head_out}, expected {num_class+1} for a 3-modality "
            f"HAC ACR checkpoint. Wrong checkpoint?\n")
    model.load_state_dict(ck['model_state_dict'])
    model_flow.load_state_dict(ck['model_flow_state_dict'])
    audio_cls_model.load_state_dict(ck['audio_cls_model_state_dict'])
    mlp_cls.load_state_dict(ck['mlp_cls_state_dict'])
    model.eval(); model_flow.eval(); audio_cls_model.eval(); mlp_cls.eval()
    saved_ep = ck.get('BestEpoch', ck.get('epoch', '?'))
    print(f"checkpoint loaded (head {head_out} = {num_class}+1, best epoch {saved_ep})")

    # ---- HAC test split (same pooled/val_frac/seed convention as training) ----
    test_ds = HACAUDIODOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow,
                             datapath=args.datapath, val_frac=args.val_frac, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.bsz, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=False)
    print(f"test clips: {len(test_ds)}")

    confs, preds, labels = [], [], []
    with torch.no_grad():
        for clip, flow, spec, y in tqdm(loader, desc='test'):
            v_emd, f_emd, a_emd = extract(model, model_flow, audio_model, audio_cls_model, clip, flow, spec)
            fused = mlp_cls(v_emd, a_emd, f_emd)                      # [B, C+1]
            confs.append(msp_confidence(fused, num_class).detach().cpu().numpy())
            preds.append(fused[:, :num_class].argmax(1).detach().cpu().numpy())
            labels.append(y.numpy())

    m = compute_fd_metrics(np.concatenate(confs), np.concatenate(preds), np.concatenate(labels))

    print("\n=== HAC 3-modality (video+flow+audio) ACR — standalone test ===")
    print_fd_row("ACR/test", m)
    print("\nPaper Table 3 (HAC v+f+audio, ACR): AURC 15.09  AUROC 92.26  FPR95 34.78  ACC 89.45")
    print("Lower AURC/FPR95 = better, higher AUROC/ACC = better.")
    print("(Single seed. FPR95 is the noisiest metric on this test size — read with seed variance.)")


if __name__ == '__main__':
    main()