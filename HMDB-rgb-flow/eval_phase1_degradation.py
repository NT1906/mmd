"""
eval_phase1_degradation.py  —  Phase 1: use the confidence-degradation signal at TEST time.

The paper trains with degradation-awareness (ACL) but at inference scores with plain
MSP on the fused head (Section 2.6: kappa = max_y p_hat). This script keeps the model's
PREDICTIONS unchanged (so accuracy is identical) and only replaces the confidence SCORE
used to rank/reject, comparing:

  (1) MSP            kappa = conf                         (paper baseline)
  (2) deg-additive   kappa = conf - a * sum_k max(0, conf_k - conf)
  (3) deg-maxgap     kappa = conf - a * max(0, max_k conf_k - conf)
  (4) learned        logistic regression on per-sample confidence features

where conf = fused MSP, conf_k = unimodal MSP (video / flow), over the real C classes.
Higher kappa => more likely correct.

Protocol (no leakage): the scalar `a` (and the logistic model) are tuned on VAL, then
applied unchanged to TEST. We report AURC / AUROC / FPR95 on both splits.

Run:
  python eval_phase1_degradation.py --dataset HMDB --datapath ~/data/hmdb51/ \
      --resumef models/<best>.pt --num_workers 2
The first run does one forward pass over val+test and caches per-sample arrays to
phase1_cache_<dataset>.npz; rerun with --use_cache to re-analyse instantly (no GPU).
"""
from mmaction.apis import init_recognizer
import torch, argparse, os, random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dataloader_video_flow import EPICDOMAIN
from acr_modules import msp_confidence, _aurc, _auroc, _fpr95


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super().__init__(); self.enc_net = nn.Linear(input_dim, out_dim)
    def forward(self, v, f): return self.enc_net(torch.cat((v, f), dim=1))


def collect(model, model_flow, mlp_cls, loader, num_class, device):
    """One forward pass; returns per-sample conf, conf_v, conf_f, correct."""
    conf, cv, cf, correct = [], [], [], []
    with torch.no_grad():
        for clip, flow, y in tqdm(loader, desc='forward'):
            clip = clip['imgs'].cuda().squeeze(1); flow = flow['imgs'].cuda().squeeze(1)
            xs, xf = model.module.backbone.get_feature(clip)
            vfeat = model.module.backbone.get_predict((xs.detach(), xf.detach()))
            v_predict, v_emd = model.module.cls_head(vfeat)
            ffeat = model_flow.module.backbone.get_feature(flow)
            ffeat = model_flow.module.backbone.get_predict(ffeat)
            f_predict, f_emd = model_flow.module.cls_head(ffeat)
            fused = mlp_cls(v_emd, f_emd)
            conf.append(msp_confidence(fused, num_class).cpu().numpy())
            cv.append(msp_confidence(v_predict, num_class).cpu().numpy())
            cf.append(msp_confidence(f_predict, num_class).cpu().numpy())
            pred = fused[:, :num_class].argmax(1).cpu().numpy()
            correct.append((pred == y.numpy()).astype(np.int64))
    return (np.concatenate(conf), np.concatenate(cv),
            np.concatenate(cf), np.concatenate(correct))


# ---- scoring functions (operate on cached numpy arrays) ----
def gap_sum(conf, cv, cf):
    return np.maximum(0, cv - conf) + np.maximum(0, cf - conf)

def gap_max(conf, cv, cf):
    return np.maximum(0, np.maximum(cv, cf) - conf)

def metrics(score, correct):
    return {'AURC': _aurc(score, correct), 'AUROC': _auroc(score, correct),
            'FPR95': _fpr95(score, correct)}

def row(name, m, acc):
    print("%-22s AURC %7.2f   AUROC %6.2f   FPR95 %6.2f   ACC %6.2f"
          % (name, m['AURC'], m['AUROC'], m['FPR95'], acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datapath', required=True)
    ap.add_argument('--resumef', default='')
    ap.add_argument('--dataset', default='HMDB')
    ap.add_argument('--bsz', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--use_cache', action='store_true')
    args = ap.parse_args()
    if not args.datapath.endswith('/'): args.datapath += '/'
    cache = f'phase1_cache_{args.dataset}.npz'

    if args.use_cache and os.path.exists(cache):
        d = np.load(cache)
        vconf, vcv, vcf, vcorr = d['vconf'], d['vcv'], d['vcf'], d['vcorr']
        tconf, tcv, tcf, tcorr = d['tconf'], d['tcv'], d['tcf'], d['tcorr']
        print('loaded cache', cache)
    else:
        assert args.resumef, "need --resumef on first run"
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        cfgf = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
        cfgo = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
        device = torch.device('cuda:0'); v_dim, f_dim = 2304, 2048
        num_class = 43 if args.dataset == 'HMDB' else (229 if args.dataset == 'Kinetics' else None)

        model = init_recognizer(cfgf, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda(); cfg = model.cfg
        model = torch.nn.DataParallel(model)
        model_flow = init_recognizer(cfgo, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda(); cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)
        mlp_cls = Encoder(v_dim + f_dim, num_class + 1).cuda()

        ck = torch.load(args.resumef, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        model_flow.load_state_dict(ck['model_flow_state_dict'])
        mlp_cls.load_state_dict(ck['mlp_cls_state_dict'])
        model.eval(); model_flow.eval(); mlp_cls.eval()

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mk = lambda s: torch.utils.data.DataLoader(
            EPICDOMAIN(split=s, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath,
                       dataset=args.dataset, near_ood=False),
            batch_size=args.bsz, num_workers=args.num_workers, shuffle=False,
            pin_memory=(dev.type == 'cuda'), drop_last=False)
        vconf, vcv, vcf, vcorr = collect(model, model_flow, mlp_cls, mk('val'), num_class, dev)
        tconf, tcv, tcf, tcorr = collect(model, model_flow, mlp_cls, mk('test'), num_class, dev)
        np.savez(cache, vconf=vconf, vcv=vcv, vcf=vcf, vcorr=vcorr,
                 tconf=tconf, tcv=tcv, tcf=tcf, tcorr=tcorr)
        print('cached ->', cache)

    vacc, tacc = 100*vcorr.mean(), 100*tcorr.mean()

    # ---- tune alpha on VAL for the two degradation scores (by AUROC) ----
    alphas = np.round(np.arange(0.0, 3.01, 0.1), 2)
    vg_s, tg_s = gap_sum(vconf, vcv, vcf), gap_sum(tconf, tcv, tcf)
    vg_m, tg_m = gap_max(vconf, vcv, vcf), gap_max(tconf, tcv, tcf)
    best_add = max(alphas, key=lambda a: _auroc(vconf - a*vg_s, vcorr))
    best_max = max(alphas, key=lambda a: _auroc(vconf - a*vg_m, vcorr))

    # ---- learned: logistic regression on [conf, cv, cf, gap_sum, gap_max] (fit on VAL) ----
    learned_ok = True
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xv = np.stack([vconf, vcv, vcf, vg_s, vg_m], 1)
        Xt = np.stack([tconf, tcv, tcf, tg_s, tg_m], 1)
        sc = StandardScaler().fit(Xv)
        lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xv), vcorr)
        v_learn = lr.predict_proba(sc.transform(Xv))[:, 1]
        t_learn = lr.predict_proba(sc.transform(Xt))[:, 1]
    except Exception as e:
        learned_ok = False; print('(sklearn unavailable, skipping learned variant:', e, ')')

    print("\n================ PHASE 1: test-time scoring (predictions unchanged) ================")
    print(f"tuned on VAL:  alpha_add={best_add}  alpha_maxgap={best_max}\n")
    print("--- VAL ---")
    row('MSP (baseline)', metrics(vconf, vcorr), vacc)
    row(f'deg-additive a={best_add}', metrics(vconf - best_add*vg_s, vcorr), vacc)
    row(f'deg-maxgap   a={best_max}', metrics(vconf - best_max*vg_m, vcorr), vacc)
    if learned_ok: row('learned (logreg)', metrics(v_learn, vcorr), vacc)

    print("\n--- TEST (the number that matters) ---")
    row('MSP (baseline)', metrics(tconf, tcorr), tacc)
    row(f'deg-additive a={best_add}', metrics(tconf - best_add*tg_s, tcorr), tacc)
    row(f'deg-maxgap   a={best_max}', metrics(tconf - best_max*tg_m, tcorr), tacc)
    if learned_ok: row('learned (logreg)', metrics(t_learn, tcorr), tacc)

    print("\nLower AURC/FPR95 = better, higher AUROC = better. ACC is identical across rows")
    print("(predictions are unchanged; only the confidence score differs).")


if __name__ == '__main__':
    main()
