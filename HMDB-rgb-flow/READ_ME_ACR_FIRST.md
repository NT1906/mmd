# ACR on HMDB51 — READ ME FIRST (fixed, ready-to-run)

This is the MultiOOD repo with the ACR training/eval added and the issues you hit
already fixed:
- ACR files added in `HMDB-rgb-flow/`: `acr_modules.py`, `train_video_flow_acr.py`,
  `test_video_flow_acr.py`.
- The vendored `mmaction` numpy bug fixed (deprecated `np.int`/`np.float`/`np.bool`/
  `np.object` → builtins; 29 occurrences). No numpy downgrade needed.
- `--datapath` now tolerates a missing trailing slash.
- Splits are the ORIGINAL ones (names with `#`, `!`, etc.) — they match the
  original HMDB51 data. Do NOT sanitise names; use original data + these splits.

## Data layout expected
```
~/data/hmdb51/
  video/<class>/*.avi
  flow/*_flow_x.mp4  *_flow_y.mp4
```
Use your ORIGINAL (unsanitised) HMDB51. Copy it INTO the Linux/WSL filesystem
(`~/data/...`), not `/mnt/c`, or decoding will be very slow.

## Pretrained weights
Put the two files in `HMDB-rgb-flow/pretrained_models/`:
- `slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth`
- `slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth`

## Run (from inside HMDB-rgb-flow)
```bash
cd HMDB-rgb-flow

# 0) quick checks
python -c "import torch, mmcv; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python - <<'PY'
import os; root=os.path.expanduser('~/data/hmdb51/'); tot=miss=0
for l in open('splits/HMDB_train.txt'):
    l=l.strip()
    if not l: continue
    p,_=l.rsplit(' ',1); tot+=1
    v=root+'video/'+p; s=p.index('/')+1; e=p.index('.'); pure=p[s:e]
    if not (os.path.exists(v) and os.path.exists(root+'flow/'+pure+'_flow_x.mp4')): miss+=1
print(f'{tot} entries, {miss} missing')
PY

# 1) sanity: 1 epoch
python train_video_flow_acr.py --dataset HMDB --datapath ~/data/hmdb51/ \
    --lr 1e-4 --bsz 8 --nepochs 1 --num_workers 0 \
    --lambda_acl 2.0 --n_min 32 --n_max 256 --save_best --appen acr_

# 2) full: 50 epochs (use tmux)
python train_video_flow_acr.py --dataset HMDB --datapath ~/data/hmdb51/ \
    --lr 1e-4 --bsz 8 --nepochs 50 --num_workers 2 \
    --lambda_acl 2.0 --n_min 32 --n_max 256 --save_best --appen acr_

# 3) evaluate (use the _best.pt name printed at the end of training)
python test_video_flow_acr.py --dataset HMDB --datapath ~/data/hmdb51/ \
    --num_workers 0 --resumef models/<printed_best_name>_best.pt
```
A4000 is 16 GB: `--bsz 8` (drop to 6/4 if OOM). Expect ~10 min/epoch.
Target (paper): AURC 19.97 / AUROC 92.02 / FPR95 41.96 / ACC 87.23; beat MSP
(29.56 / 88.28 / 52.07 / 86.20).

If the "missing" count above is >0, a few clips are absent from your data share;
tell me and I'll give a one-liner to drop just those split lines.
