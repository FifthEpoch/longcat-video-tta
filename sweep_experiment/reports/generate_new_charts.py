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
# Data: Best configs at G=4.0
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
        "train": 37.3, "gen": 81.3, "total": 118.5,
    },
    "AdaSteer\n(10 steps)": {
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

fig, ax = plt.subplots(figsize=(8, 6))

colors = [LIGHT_GRAY, BLUE_3, BLUE_1]
markers = ["o", "D", "s"]
sizes = [180, 160, 200]

for i, k in enumerate(keys):
    m = methods[k]
    label_clean = k.replace("\n", " ")
    ax.scatter(m["total"], m["fvd"], s=sizes[i], c=colors[i], marker=markers[i],
               edgecolors="black", linewidths=1.5, zorder=5, label=label_clean)
    offset_x = 4
    offset_y = 15 if i < 2 else -20
    ax.annotate(
        f'{label_clean}\nFVD={m["fvd"]:.1f}, {m["total"]:.0f}s',
        xy=(m["total"], m["fvd"]),
        xytext=(m["total"] + offset_x, m["fvd"] + offset_y),
        fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.2),
    )

ax.set_xlabel("Total Time per Video (seconds)")
ax.set_ylabel("FVD (lower = better)")
ax.set_title("FVD vs Time Cost: Method Comparison (G=4.0)")
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax.set_xlim(60, 160)
ax.set_ylim(520, 700)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fvd_vs_time.png"))
plt.close(fig)
print("  [1/2] fvd_vs_time.png")

# ----------------------------------------------------------------
# CHART 2: Metric Comparison (5 panels, no G=0)
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

bar_colors = [LIGHT_GRAY, BLUE_3, BLUE_1]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

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

    short_labels = ["No-TTA", "LoRA", "AdaSteer"]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_title(metric_name, fontsize=14, fontweight="bold")

    direction = "higher=better" if higher_better else "lower=better"
    ax.set_ylabel(f"{metric_name} ({direction})", fontsize=9)

    if metric_key in y_ranges:
        ax.set_ylim(y_ranges[metric_key])

fig.suptitle(
    "Metric Comparison: No-TTA vs LoRA vs AdaSteer (G=4.0, Panda-70M, N=100)",
    fontsize=16, fontweight="bold", y=1.03,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "method_metrics_no_g0.png"))
plt.close(fig)
print("  [2/2] method_metrics_no_g0.png")

print(f"\nNew charts saved to {OUT}/")
