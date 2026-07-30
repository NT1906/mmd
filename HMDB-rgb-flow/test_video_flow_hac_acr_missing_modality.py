"""
test_video_flow_hac_acr_missing_modality.py  —  missing-modality eval for a
PLAIN ACR (reliability-OFF) 3-modality HAC checkpoint. The counterpart to
test_video_flow_hac_reliability.py --drop, so the two can be compared directly.

Plain ACR has no reliability mechanism to renormalize around a missing
modality — it just concatenates whatever embeddings it has. So "drop" here
means ZERO that modality's embedding before concatenation into the fused head
(the same convention as the HMDB eval_phase3_missing_modality.py masking:
zeros = hard absence). No renormalization, because ACR has nothing to
renormalize — this IS the fair comparison: reliability's renorm mechanism is
exactly what ACR lacks, so its absence here is the point, not an omission.

Run all four to get the full comparison row:
  python test_video_flow_hac_acr_missing_modality.py --datapath ~/data/ --resumef <acr_off_best.pt>
  python test_video_flow_hac_acr_missing_modality.py --datapath ~/data/ --resumef <...> --drop video
  python test_video_flow_hac_acr_missing_modality.py --datapath ~/data/ --resumef <...> --drop flow
  python test_video_flow_hac_acr_missing_modality.py --datapath ~/data/ --resumef <...> --drop audio
"""
from mmaction.apis import init_recognizer
import torch, argparse, numpy as np, torch.nn as nn, random
from tqdm import tqdm

from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_flow_hac_audio import HACAUDIODOMAIN
from acr_modules import msp_confidence, compute_fd_metrics, print_fd_row


class Encoder(nn.Module):
    def __init__(self, input_dim=4864, out_dim=8, p_drop=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.enc_net = nn.Linear(input_dim, out_dim)
    def logits_from_concat(self, x):
        return self.enc_net(self.drop(x))
    def forward(self, vfeat, afeat, ffeat):
        return self.logits_from_concat(torch.cat((vfeat, afeat, ffeat), dim=1))


def extract(model, model_flow, audio_model, audio_cls_model, clip, flow, spec):
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
    ap.add_argument('--drop', type=str, default=None, choices=[None, 'video', 'flow', 'audio'])
    ap.add_argument('--audio_pretrain', type=str, default='pretrained_models/vggsound_avgpool.pth.tar')
    args = ap.parse_args()
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
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda(); cfg = model.cfg
    model = torch.nn.DataParallel(model)
    model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda(); cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    audio_args = get_arguments()
    audio_model = AVENet(audio_args)
    ck_audio = torch.load(args.audio_pretrain)
    audio_model.load_state_dict(ck_audio['model_state_dict'])
    audio_model = audio_model.cuda(); audio_model.eval()
    audio_cls_model = AudioAttGenModule()
    audio_cls_model.fc = nn.Linear(a_dim, num_class)
    audio_cls_model = audio_cls_model.cuda()

    print("Resuming from", args.resumef)
    ck = torch.load(args.resumef, map_location=device)
    if 'rel_v_state_dict' in ck:
        print("WARNING: this checkpoint HAS reliability heads — you probably want "
              "test_video_flow_hac_reliability.py --drop instead. Continuing anyway "
              "(reliability heads will simply be ignored).")
    head_out = ck['mlp_cls_state_dict']['enc_net.weight'].shape[0]
    if head_out != num_class + 1:
        raise SystemExit(f"\nFused head out_dim={head_out}, expected {num_class+1}. Wrong checkpoint?\n")
    mlp_cls = Encoder(input_dim=v_dim + f_dim + a_dim, out_dim=num_class + 1).cuda()

    model.load_state_dict(ck['model_state_dict'])
    model_flow.load_state_dict(ck['model_flow_state_dict'])
    audio_cls_model.load_state_dict(ck['audio_cls_model_state_dict'])
    mlp_cls.load_state_dict(ck['mlp_cls_state_dict'])
    for m in (model, model_flow, audio_cls_model, mlp_cls): m.eval()
    print(f"plain ACR checkpoint loaded (best epoch {ck.get('BestEpoch','?')})")

    test_ds = HACAUDIODOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow,
                             datapath=args.datapath, val_frac=args.val_frac, seed=args.seed)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.bsz,
             num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    print(f"test clips: {len(test_ds)}")
    if args.drop:
        print(f"DROPPING (zeroing) modality: {args.drop} — plain ACR has no renorm mechanism")

    confs, preds, labels = [], [], []
    with torch.no_grad():
        for clip, flow, spec, y in tqdm(loader, desc='test'):
            v_e, f_e, a_e = extract(model, model_flow, audio_model, audio_cls_model, clip, flow, spec)
            if args.drop == 'video': v_e = torch.zeros_like(v_e)
            elif args.drop == 'flow': f_e = torch.zeros_like(f_e)
            elif args.drop == 'audio': a_e = torch.zeros_like(a_e)
            fused = mlp_cls(v_e, a_e, f_e)
            confs.append(msp_confidence(fused, num_class).detach().cpu().numpy())
            preds.append(fused[:, :num_class].argmax(1).detach().cpu().numpy())
            labels.append(y.numpy())

    m = compute_fd_metrics(np.concatenate(confs), np.concatenate(preds), np.concatenate(labels))
    tag = "ALL-MODALITIES" if args.drop is None else f"DROP-{args.drop.upper()}"
    print(f"\n=== HAC plain-ACR missing-modality test  [{tag}] ===")
    print_fd_row("ACR/test", m)
    print("\nLower AURC/FPR95 = better, higher AUROC/ACC = better. Single seed.")


if __name__ == '__main__':
    main()
