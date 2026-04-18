"""Generate FVD-vs-time and method metrics comparison charts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

GRAY = "#6B7280"
LIGHT_GRAY = "#9CA3AF"
BLUE_1 = "#1E40AF"
BLUE_2 = "#2563EB"
BLUE_3 = "#60A5FA"
GREEN = "#16A34A"
RED = "#DC2626"

# ----------------------------------------------------------------
# Data: Best configs (default guidance)
# NOTE: LoRA total time is estimated (2x Delta Vector TTA time) until
#       we have measured wall-clock from the LoRA step-sweep runs.
# ----------------------------------------------------------------

methods = {
    "No-TTA": {
        "fvd": 641.1, "psnr": 18.612, "ssim": 0.682, "lpips": 0.320,
        "fid": 77.5,
        "train": 0.0, "gen": 80.4, "total": 80.4,
    },
    "LoRA\n(20 steps)": {
        "fvd": 641.5, "psnr": 18.569, "ssim": 0.6821, "lpips": 0.3201,
        "fid": 77.2,
        "train": 109.2, "gen": 80.0, "total": 189.2,
    },
    "Delta Vector\n(5 steps)": {
        "fvd": 571.2, "psnr": 18.603, "ssim": 0.6828, "lpips": 0.3197,
        "fid": 77.5,
        "train": 27.3, "gen": 80.0, "total": 107.3,
    },
    "Delta Vector\n(10 steps)": {
        "fvd": 568.7, "psnr": 18.590, "ssim": 0.6841, "lpips": 0.3164,
        "fid": 74.1,
        "train": 54.6, "gen": 79.9, "total": 134.4,
    },
}

keys = list(methods.keys())
n = len(keys)

# ----------------------------------------------------------------
# CHART 1: FVD versus Time Cost
# ----------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

fvd_colors = [LIGHT_GRAY, BLUE_3, BLUE_2, BLUE_1]
fvd_markers = ["o", "D", "s", "s"]
fvd_sizes = [180, 160, 180, 200]

annot_offsets = {
    "No-TTA":              (8, 18),
    "LoRA (20 steps)":     (-50, -25),
    "Delta Vector (5 steps)":  (-40, -30),
    "Delta Vector (10 steps)": (8, -25),
}

for i, k in enumerate(keys):
    m = methods[k]
    label_clean = k.replace("\n", " ")
    ax.scatter(m["total"], m["fvd"], s=fvd_sizes[i], c=fvd_colors[i],
               marker=fvd_markers[i], edgecolors="black", linewidths=1.5,
               zorder=5, label=label_clean)
    ox, oy = annot_offsets.get(label_clean, (8, 15))
    ax.annotate(
        f'{label_clean}\nFVD={m["fvd"]:.1f}, {m["total"]:.0f}s',
        xy=(m["total"], m["fvd"]),
        xytext=(m["total"] + ox, m["fvd"] + oy),
        fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=fvd_colors[i], lw=1.2),
    )

ax.set_xlabel("Total Time per Video (seconds)")
ax.set_ylabel("FVD (lower = better)")
ax.set_title("FVD vs Time Cost: Method Comparison")
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax.set_xlim(60, 210)
ax.set_ylim(520, 700)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fvd_vs_time.png"))
plt.close(fig)
print("  [1/2] fvd_vs_time.png")

# ----------------------------------------------------------------
# CHART 2: Metric Comparison (5 panels)
# ----------------------------------------------------------------

metrics_list = [
    ("PSNR", "psnr", True),
    ("SSIM", "ssim", True),
    ("LPIPS", "lpips", False),
    ("FVD", "fvd", False),
    ("FID", "fid", False),
]

y_ranges = {
    "psnr":  (18.0, 19.0),
    "ssim":  (0.65, 0.72),
    "lpips": (0.28, 0.36),
    "fvd":   (400, 750),
    "fid":   (60, 90),
}

bar_colors = [LIGHT_GRAY, BLUE_3, BLUE_2, BLUE_1]

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

for ax_i, (metric_name, metric_key, higher_better) in enumerate(metrics_list):
    ax = axes[ax_i]
    vals = [methods[k][metric_key] for k in keys]
    x = np.arange(n)
    w = 0.55

    bars = ax.bar(x, vals, w, color=bar_colors, edgecolor="white", linewidth=1.5)

    for xi, (bar, v) in enumerate(zip(bars, vals)):
        fmt = f"{v:.3f}" if metric_key in ("ssim", "lpips") else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt, ha="center", va="bottom", fontsize=10, fontweight="bold")

    short_labels = ["No-TTA", "LoRA\n20-step", "Delta Vector\n5-step", "Delta Vector\n10-step"]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_title(metric_name, fontsize=14, fontweight="bold")

    direction = "higher=better" if higher_better else "lower=better"
    ax.set_ylabel(f"{metric_name} ({direction})", fontsize=9)

    if metric_key in y_ranges:
        ax.set_ylim(y_ranges[metric_key])

fig.suptitle(
    "Metric Comparison: No-TTA vs LoRA vs Delta Vector (Panda-70M, N=100)",
    fontsize=16, fontweight="bold", y=1.03,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "method_metrics_no_g0.png"))
plt.close(fig)
print("  [2/2] method_metrics_no_g0.png")

print(f"\nNew charts saved to {OUT}/")
