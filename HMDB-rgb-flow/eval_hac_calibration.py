"""
eval_hac_calibration.py  —  ECE + NLL for a HAC checkpoint (reliability OR
plain ACR), on the SAME conditions already used for the FD/missing-modality
tables (all-modalities, or --drop {video,flow,audio}).

Why this matters for the paper: the ACR paper never reports calibration.
Your own HMDB Phase-3 probe hinted that ACR's confidence collapses more
honestly than CE under modality loss -- this script gets the real numbers
for the HAC reliability-vs-ACR comparison you already have FD numbers for,
so calibration becomes a THIRD verified axis alongside FD and missing-modality.

Metrics:
  ECE  (Expected Calibration Error, 15 equal-width bins, standard definition):
       mean over bins of |accuracy_in_bin - confidence_in_bin|, weighted by
       bin size. Lower = better calibrated. Reported as a fraction (0-1) and
       as a percentage.
  NLL  (Negative Log-Likelihood): mean of -log(p_true_class). Lower = better;
       unbounded, comparative only.

Auto-detects checkpoint type (reliability heads present or not) and applies
the correct fusion path (weighted vs plain) so the SAME script produces a
calibration number for both your reliability and plain-ACR checkpoints, and
respects the same --drop convention as the two existing missing-modality
scripts (zero embedding for plain ACR; zero-and-renormalize weights for
reliability).

Run (one call per checkpoint per condition -- 8 calls total for the full
reliability-vs-ACR x {all,drop-video,drop-flow,drop-audio} table):
  python eval_hac_calibration.py --datapath ~/data/ --resumef <rel_best.pt>
  python eval_hac_calibration.py --datapath ~/data/ --resumef <rel_best.pt> --drop video
  python eval_hac_calibration.py --datapath ~/data/ --resumef <acr_off_best.pt>
  python eval_hac_calibration.py --datapath ~/data/ --resumef <acr_off_best.pt> --drop video
  ... etc.
"""
from mmaction.apis import init_recognizer
import torch, argparse, numpy as np, torch.nn as nn, random
from tqdm import tqdm

from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_flow_hac_audio import HACAUDIODOMAIN
from acr_modules import compute_fd_metrics, msp_confidence


class Encoder(nn.Module):
    def __init__(self, input_dim=4864, out_dim=8, p_drop=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.enc_net = nn.Linear(input_dim, out_dim)
    def logits_from_concat(self, x):
        return self.enc_net(self.drop(x))
    def forward(self, vfeat, afeat, ffeat):
        return self.logits_from_concat(torch.cat((vfeat, afeat, ffeat), dim=1))


class ReliabilityHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
    def forward(self, emb):
        return torch.sigmoid(self.net(emb)).squeeze(1)


def weighted_embeds(v, a, f, rv, ra, rf, mask=None):
    if mask is not None:
        mv, ma, mf = mask
        rv = rv * mv; ra = ra * ma; rf = rf * mf
        denom = (rv + ra + rf).clamp(min=1e-6)
        scale = 3.0 / denom
        rv, ra, rf = rv * scale, ra * scale, rf * scale
    return v * rv.unsqueeze(1), a * ra.unsqueeze(1), f * rf.unsqueeze(1)


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


def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def ece_score(probs_true_class, correct, n_bins=15):
    """Standard ECE: bin by confidence (= probs_true_class here is actually the
    MAX predicted prob, i.e. the model's stated confidence in its own top
    prediction), compare bin-mean-confidence to bin-accuracy."""
    conf = probs_true_class  # here: max softmax prob (the model's confidence)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def nll_score(probs, labels, num_class):
    """Mean negative log-likelihood of the TRUE class (over the C real classes)."""
    p_true = probs[np.arange(len(labels)), labels]
    p_true = np.clip(p_true, 1e-12, 1.0)
    return float(-np.log(p_true).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datapath', type=str, required=True)
    ap.add_argument('--resumef', type=str, required=True)
    ap.add_argument('--bsz', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--rel_hidden', type=int, default=128)
    ap.add_argument('--drop', type=str, default=None, choices=[None, 'video', 'flow', 'audio'])
    ap.add_argument('--n_bins', type=int, default=15)
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

    mlp_cls = Encoder(input_dim=v_dim + f_dim + a_dim, out_dim=num_class + 1).cuda()

    print("Resuming from", args.resumef)
    ck = torch.load(args.resumef, map_location=device)
    is_reliability = 'rel_v_state_dict' in ck
    model.load_state_dict(ck['model_state_dict'])
    model_flow.load_state_dict(ck['model_flow_state_dict'])
    audio_cls_model.load_state_dict(ck['audio_cls_model_state_dict'])
    mlp_cls.load_state_dict(ck['mlp_cls_state_dict'])

    rel_v = rel_a = rel_f = None
    if is_reliability:
        rel_v = ReliabilityHead(v_dim, args.rel_hidden).cuda()
        rel_a = ReliabilityHead(a_dim, args.rel_hidden).cuda()
        rel_f = ReliabilityHead(f_dim, args.rel_hidden).cuda()
        rel_v.load_state_dict(ck['rel_v_state_dict'])
        rel_a.load_state_dict(ck['rel_a_state_dict'])
        rel_f.load_state_dict(ck['rel_f_state_dict'])
        for m in (rel_v, rel_a, rel_f): m.eval()
    for m in (model, model_flow, audio_cls_model, mlp_cls): m.eval()
    kind = "RELIABILITY (weighted fusion)" if is_reliability else "PLAIN ACR (no reweighting)"
    print(f"checkpoint type: {kind}  (best epoch {ck.get('BestEpoch','?')})")

    mask = None
    if args.drop is not None:
        mask = {'video': [0.,1.,1.], 'audio': [1.,0.,1.], 'flow': [1.,1.,0.]}[args.drop]
        print(f"DROPPING modality: {args.drop}"
              + (" (renormalized)" if is_reliability else " (zeroed, no renorm — plain ACR)"))

    test_ds = HACAUDIODOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow,
                             datapath=args.datapath, val_frac=args.val_frac, seed=args.seed)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.bsz,
             num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    print(f"test clips: {len(test_ds)}")

    all_logits, all_labels = [], []
    with torch.no_grad():
        for clip, flow, spec, y in tqdm(loader, desc='eval'):
            v_e, f_e, a_e = extract(model, model_flow, audio_model, audio_cls_model, clip, flow, spec)
            if is_reliability:
                rv, ra, rf = rel_v(v_e), rel_a(a_e), rel_f(f_e)
                bmask = None
                if mask is not None:
                    B = v_e.shape[0]
                    bmask = tuple(torch.full((B,), mv, device=v_e.device) for mv in mask)
                v_e, a_e, f_e = weighted_embeds(v_e, a_e, f_e, rv, ra, rf, mask=bmask)
            else:
                if args.drop == 'video': v_e = torch.zeros_like(v_e)
                elif args.drop == 'flow': f_e = torch.zeros_like(f_e)
                elif args.drop == 'audio': a_e = torch.zeros_like(a_e)
            fused = mlp_cls(v_e, a_e, f_e)
            all_logits.append(fused[:, :num_class].detach().cpu().numpy())
            all_labels.append(y.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.int64)
    conf = probs.max(axis=1)  # model's stated confidence in its top prediction

    np.savez(f"calib_dump_{args.drop or 'all'}_{'ON' if is_reliability else 'OFF'}.npz",
         conf=conf, correct=correct, labels=labels)
    ece = ece_score(conf, correct, n_bins=args.n_bins)
    nll = nll_score(probs, labels, num_class)
    fd = compute_fd_metrics(conf, pred, labels)

    tag = "ALL-MODALITIES" if args.drop is None else f"DROP-{args.drop.upper()}"
    
    print(f"\n=== HAC calibration  [{kind}]  [{tag}] ===")
    print("ECE  %6.4f  (%.2f%%)" % (ece, ece * 100))
    print("NLL  %6.4f" % nll)
    print("ACC  %6.2f   AURC %7.2f   AUROC %6.2f   FPR95 %6.2f"
          % (fd['ACC'], fd['AURC'], fd['AUROC'], fd['FPR95']))
    print("\nLower ECE/NLL = better calibrated. Single seed.")


if __name__ == '__main__':
    main()
