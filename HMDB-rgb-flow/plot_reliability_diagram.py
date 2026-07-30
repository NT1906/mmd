#!/usr/bin/env python3
"""
plot_reliability_diagram.py

Turn the per-sample dumps written by eval_hac_calibration.py into publication
reliability diagrams (confidence vs. accuracy, 15 equal-width bins).

Prereq: patch eval_hac_calibration.py so that, right before it computes ECE, it
saves the per-sample arrays. Add this ONE line there (conf/correct/labels already
exist in that scope):

    np.savez(f"calib_dump_{args.drop or 'all'}_{'ON' if is_reliability else 'OFF'}.npz",
             conf=conf, correct=correct, labels=labels)

Then re-run the eval you care about (e.g. drop-video ON and OFF) so the two
.npz files appear, and run this script.

Usage:
    python plot_reliability_diagram.py \
        --on  calib_dump_video_ON.npz \
        --off calib_dump_video_OFF.npz \
        --title "Drop video (severe)" \
        --out reliability_drop_video.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#111827"; BODY = "#374151"; MUTE = "#9CA3AF"; LINE = "#E5E7EB"
ON_C = "#10B981"; OFF_C = "#9CA3AF"; BAD = "#DC2626"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": BODY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": BODY, "ytick.color": BODY, "axes.linewidth": 0.9,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def bin_stats(conf, correct, n_bins=15):
    """Return per-bin (mean_conf, mean_acc, weight) and the ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ws = [], [], []
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        c = conf[m].mean()
        a = correct[m].mean()
        w = m.sum() / n
        xs.append(c); ys.append(a); ws.append(w)
        ece += w * abs(a - c)
    return np.array(xs), np.array(ys), np.array(ws), ece


def load(path):
    d = np.load(path)
    conf = d["conf"].astype(float)
    correct = d["correct"].astype(float)
    return conf, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="npz dump for reliability (ON)")
    ap.add_argument("--off", required=True, help="npz dump for plain ACR (OFF)")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="reliability_diagram.png")
    ap.add_argument("--n_bins", type=int, default=15)
    args = ap.parse_args()

    on_conf, on_corr = load(args.on)
    off_conf, off_corr = load(args.off)
    xo, yo, wo, ece_on = bin_stats(on_conf, on_corr, args.n_bins)
    xf, yf, wf, ece_off = bin_stats(off_conf, off_corr, args.n_bins)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTE, lw=1.2, label="perfect calibration")
    # gap shading for OFF (the overconfident one)
    ax.fill_between(xf, yf, xf, color=BAD, alpha=0.10)
    ax.plot(xo, yo, marker="o", ms=6, lw=2, color=ON_C,
            label=f"Reliability (ON)  ECE={ece_on:.3f}")
    ax.plot(xf, yf, marker="s", ms=6, lw=2, color=OFF_C,
            label=f"Plain ACR (OFF)  ECE={ece_off:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ttl = "Reliability diagram" + (f" — {args.title}" if args.title else "")
    ax.set_title(ttl, fontsize=11.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.out)
    print(f"wrote {args.out}   (ON ECE={ece_on:.4f}, OFF ECE={ece_off:.4f})")


if __name__ == "__main__":
    main()
