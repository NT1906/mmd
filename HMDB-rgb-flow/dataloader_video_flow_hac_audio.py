"""
dataloader_video_flow_hac_audio.py  —  HAC, 3 modalities (video + flow + audio).

Built for the ACR failure-detection reproduction (paper Table 3, video+flow+audio).
Protocol: CLOSED-SET POOLED, matching the ACR paper's HAC FD table — all three
domains (human/animal/cartoon) pooled, NOT leave-one-domain-out (that is the
SimMMDG domain-generalization protocol, a different task).

Splits: the dataset ships only HAC_{train,test}_only_<domain>.csv (no val).
ACR selects the best checkpoint on a validation set, so we carve a deterministic
stratified val split out of the pooled TRAIN clips (seed-fixed, default 15%).
TEST is left untouched.

Path transformation (verified on disk; CSV stores '<name>.mp4'):
    video :  <datapath>/HAC/<domain>/videos/<name>.mp4
    flow  :  <datapath>/HAC/<domain>/flow/<stem>_flow_x.mp4  (+ _flow_y.mp4)
    audio :  <datapath>/HAC/<domain>/audio/<stem>.wav
where stem = <name> with the trailing '.mp4' removed.

Returns: (data, flow, spectrogram[float32], label)  — the 4-tuple the
3-modality trainer expects.

Usage (constructed by train_video_flow_hac_acr.py):
    HACAUDIODOMAIN(split='train'|'val'|'test', cfg=cfg, cfg_flow=cfg_flow,
                   datapath=args.datapath, val_frac=0.15, seed=0)
"""
from mmaction.datasets.pipelines import Compose
import torch.utils.data
import soundfile as sf
from scipy import signal
import numpy as np
import imageio.v3 as iio
import csv
import os
import random as _random


DOMAINS = ['human', 'animal', 'cartoon']


def _read_split_csv(path):
    """Read a HAC split CSV: rows of '<name>.mp4,<label>'. Returns list of (name,label)."""
    out = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            out.append((row[0], int(row[1])))
    return out


def get_spectrogram_piece(samples, start_time, end_time, duration, samplerate, training=False):
    """Identical to dataloader_video_flow_audio.py — log-mel spectrogram piece."""
    start1 = start_time / duration * len(samples)
    end1 = end_time / duration * len(samples)
    start1 = int(np.round(start1))
    end1 = int(np.round(end1))
    samples = samples[start1:end1]

    resamples = samples[:160000]
    if len(resamples) == 0:
        resamples = np.zeros((160000))
    while len(resamples) < 160000:
        resamples = np.tile(resamples, 10)[:160000]

    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)

    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    interval = 9
    if training is True:
        noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
        spectrogram = spectrogram + noise
        start1 = np.random.choice(256 - interval, (1,))[0]
        spectrogram[start1:(start1 + interval), :] = 0

    return spectrogram


class HACAUDIODOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', cfg=None, cfg_flow=None, datapath='',
                 val_frac=0.15, seed=0, splits_dir='splits'):
        assert split in ('train', 'val', 'test')
        self.base_path = datapath if datapath.endswith('/') else datapath + '/'
        self.split = split
        self.interval = 9

        # --- pipelines: train pipeline only for the actual training split ---
        if split == 'train':
            self.pipeline = Compose(cfg.data.train.pipeline)
            self.pipeline_flow = Compose(cfg_flow.data.train.pipeline)
            self.train = True
        else:
            self.pipeline = Compose(cfg.data.val.pipeline)
            self.pipeline_flow = Compose(cfg_flow.data.val.pipeline)
            self.train = False

        # --- build the pooled sample list across all 3 domains ---
        # Each entry: (domain, name, stem, label)
        train_pool, test_pool = [], []
        for domain in DOMAINS:
            for name, label in _read_split_csv(os.path.join(splits_dir, f'HAC_train_only_{domain}.csv')):
                stem = name[:-4] if name.endswith('.mp4') else name
                train_pool.append((domain, name, stem, label))
            for name, label in _read_split_csv(os.path.join(splits_dir, f'HAC_test_only_{domain}.csv')):
                stem = name[:-4] if name.endswith('.mp4') else name
                test_pool.append((domain, name, stem, label))

        if split == 'test':
            self.items = test_pool
        else:
            # deterministic STRATIFIED carve of val out of train (per-class, seeded)
            by_class = {}
            for it in train_pool:
                by_class.setdefault(it[3], []).append(it)
            rng = _random.Random(seed)
            train_items, val_items = [], []
            for label in sorted(by_class.keys()):
                grp = by_class[label][:]
                rng.shuffle(grp)
                n_val = max(1, int(round(len(grp) * val_frac)))
                val_items.extend(grp[:n_val])
                train_items.extend(grp[n_val:])
            self.items = train_items if split == 'train' else val_items

        self.cfg = cfg
        self.cfg_flow = cfg_flow

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        domain, name, stem, label = self.items[index]
        dom_path = f'{self.base_path}HAC/{domain}/'

        # ---- video ----
        video_file = f'{dom_path}videos/{name}'
        vid = iio.imread(video_file, plugin="pyav")
        frame_num = vid.shape[0]
        start_frame, end_frame = 0, frame_num - 1
        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data = dict(frame_dir='', total_frames=end_frame - start_frame, label=-1,
                    start_index=start_index, video=vid, frame_num=frame_num,
                    filename_tmpl=filename_tmpl, modality=modality)
        data, frame_inds = self.pipeline(data)

        # ---- flow (x/y pair) ----
        video_file_x = f'{dom_path}flow/{stem}_flow_x.mp4'
        video_file_y = f'{dom_path}flow/{stem}_flow_y.mp4'
        vid_x = iio.imread(video_file_x, plugin="pyav")
        vid_y = iio.imread(video_file_y, plugin="pyav")
        frame_num_f = vid_x.shape[0]
        start_frame_f, end_frame_f = 0, frame_num_f - 1
        filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
        modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
        start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame_f)
        flow = dict(frame_dir='', total_frames=end_frame_f - start_frame_f, label=-1,
                    start_index=start_index_flow, video=vid_x, video_y=vid_y,
                    frame_num=frame_num_f, filename_tmpl=filename_tmpl_flow,
                    modality=modality_flow)
        flow, frame_inds_flow = self.pipeline_flow(flow)

        # ---- audio ----
        audio_path = f'{dom_path}audio/{stem}.wav'
        start_time = frame_inds[0] / 24.0
        end_time = frame_inds[-1] / 24.0
        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate
        spectrogram = get_spectrogram_piece(samples, start_time, end_time, duration,
                                            samplerate, training=self.train)

        return data, flow, spectrogram.astype(np.float32), label
